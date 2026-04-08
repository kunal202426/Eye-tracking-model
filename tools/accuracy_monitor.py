"""tools/accuracy_monitor.py
--------------------------------------------------------------------------------
Diagnostic monitor for the eye-tracking inference pipeline.

Two usage modes
---------------
EMBEDDED (called from main.py):
    from tools.accuracy_monitor import AccuracyMonitor
    monitor = AccuracyMonitor(log_dir="logs")
    # inside the frame loop:
    monitor.record(features, preds, gaze_screen_xy, calib_point_xy)
    # runs its own background thread; writes a JSON report every 30 s

STANDALONE (live file watcher):
    python tools/accuracy_monitor.py --log_dir logs
    Tails the latest JSON report and prints a live dashboard every 5 s.

Log content (JSON, appended every 30 s)
-----------------------------------------
- emotion class distribution and confusion
- attention class distribution
- cognitive load distribution
- gaze error estimate (vs calibration targets if available)
- feature value ranges vs expected training ranges
- timestamp and session elapsed time
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("accuracy_monitor")

# ---------------------------------------------------------------------------
# Expected feature ranges derived from training datasets
# (VREED + cognitive_load_dataset.csv exploration)
# ---------------------------------------------------------------------------
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    # Emotion MLP inputs (VREED dataset stats)
    "num_blink":              (0.0,   25.0),   # blinks per 10-s window
    "mean_blink_duration_ms": (80.0,  450.0),  # ms
    "mean_fixation_duration_ms": (100.0, 700.0),  # ms
    "mean_saccade_amplitude": (5.0,   120.0),  # px

    # Cognitive-load XGBoost inputs (cognitive_load_dataset.csv)
    "pupil_dilation_proxy":   (8.0,   65.0),   # normalised iris/IOD * 100
    "blink_rate_per_min":     (8.0,   35.0),   # blinks/min
    "fixation_duration_ms":   (100.0, 700.0),  # ms
    "saccade_duration_ms":    (15.0,  250.0),  # ms

    # Raw EAR debug field
    "raw_ear_mean":           (0.15,  0.40),
}

# ---------------------------------------------------------------------------
# Label name maps (must match model_runner.py)
# ---------------------------------------------------------------------------
EMOTION_NAMES  = {0: "sad", 1: "calm", 2: "angry", 3: "happy"}
ATTENTION_NAMES = {0: "focused", 1: "distracted", 2: "off_task"}
COGLOAD_NAMES  = {0: "low", 1: "medium", 2: "high"}

# ---------------------------------------------------------------------------
# Gaze error tracking
# ---------------------------------------------------------------------------
CALIB_SAMPLE_RADIUS_PX = 50   # gaze samples collected within this radius of target


class AccuracyMonitor:
    """Accumulates per-frame diagnostic data; flushes a JSON report every N seconds.

    Parameters
    ----------
    log_dir : str or Path
        Directory where JSON log files are written.
    flush_interval_s : float
        How often to write a report (default 30 seconds).
    screen_wh : tuple
        Screen width and height in pixels (for gaze error normalisation).
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        flush_interval_s: float = 30.0,
        screen_wh: tuple[int, int] = (1920, 1080),
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._flush_interval = flush_interval_s
        self._sw, self._sh = screen_wh

        self._start_time = time.time()
        self._lock = threading.Lock()

        # ----- accumulators -----
        self._emotion_counts:  collections.Counter = collections.Counter()
        self._attention_counts: collections.Counter = collections.Counter()
        self._cogload_counts:  collections.Counter = collections.Counter()

        # confusion matrix: rows=true (stable), cols=raw
        # we use raw single-frame prediction vs stable voted prediction
        self._emotion_confusion:  np.ndarray = np.zeros((4, 4), dtype=np.int32)
        self._attn_confusion:     np.ndarray = np.zeros((3, 3), dtype=np.int32)

        # feature accumulator: list of (name, value)
        self._feature_log: list[dict[str, float]] = []

        # gaze error accumulator: list of px distances
        self._gaze_errors: list[float] = []

        # calib target for current window (set externally via set_calib_target)
        self._calib_target: Optional[tuple[float, float]] = None

        self._frame_count = 0
        self._last_flush  = time.time()

        # ---- background thread ----
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        log.info("AccuracyMonitor started; reports -> %s every %.0fs", self._log_dir, flush_interval_s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_calib_target(self, screen_xy: Optional[tuple[float, float]]) -> None:
        """Inform the monitor that a calibration target is currently shown at screen_xy.

        While a target is active, gaze samples are collected to estimate accuracy.
        Pass None to stop collecting.
        """
        with self._lock:
            self._calib_target = screen_xy

    def record(
        self,
        features,           # ExtractedFeatures dataclass (or None)
        preds,              # ModelPredictions dataclass (or None)
        gaze_screen_xy: Optional[tuple[float, float]] = None,
    ) -> None:
        """Record one frame's data.  Call from the main loop every frame.

        Parameters
        ----------
        features : ExtractedFeatures | None
        preds    : ModelPredictions | None
        gaze_screen_xy : (screen_x, screen_y) in pixels, or None
        """
        with self._lock:
            self._frame_count += 1

            if preds is not None:
                # raw single-frame predictions (index)
                raw_emo  = getattr(preds, "emotion_idx",  None)
                raw_attn = getattr(preds, "attention_idx", None)
                raw_cog  = getattr(preds, "cogload_idx",  None)

                # stable voted predictions
                stable_emo_name  = getattr(preds, "emotion_stable_name",  "")
                stable_attn_name = getattr(preds, "attention_stable_name", "")
                stable_cog_name  = getattr(preds, "cogload_stable_name",  "")

                # count stable distributions
                if stable_emo_name and stable_emo_name not in ("uncertain",):
                    self._emotion_counts[stable_emo_name] += 1
                if stable_attn_name and stable_attn_name not in ("uncertain",):
                    self._attention_counts[stable_attn_name] += 1
                if stable_cog_name and stable_cog_name not in ("uncertain",):
                    self._cogload_counts[stable_cog_name] += 1

                # confusion: raw vs stable
                if raw_emo is not None:
                    stable_emo_idx = _name_to_idx(stable_emo_name, EMOTION_NAMES)
                    if stable_emo_idx is not None and 0 <= raw_emo < 4:
                        self._emotion_confusion[stable_emo_idx, raw_emo] += 1

                if raw_attn is not None:
                    stable_attn_idx = _name_to_idx(stable_attn_name, ATTENTION_NAMES)
                    if stable_attn_idx is not None and 0 <= raw_attn < 3:
                        self._attn_confusion[stable_attn_idx, raw_attn] += 1

            if features is not None:
                snap: dict[str, float] = {
                    "num_blink":                 float(features.num_blink),
                    "mean_blink_duration_ms":    float(features.mean_blink_duration_ms),
                    "mean_fixation_duration_ms": float(features.mean_fixation_duration_ms),
                    "mean_saccade_amplitude":    float(features.mean_saccade_amplitude),
                    "pupil_dilation_proxy":      float(features.pupil_dilation_proxy),
                    "blink_rate_per_min":        float(features.blink_rate_per_min),
                    "fixation_duration_ms":      float(features.fixation_duration_ms),
                    "saccade_duration_ms":       float(features.saccade_duration_ms),
                    "raw_ear_mean":              float(features.raw_ear_mean),
                    "current_velocity_px":       float(features.current_velocity_px),
                }
                self._feature_log.append(snap)

            # gaze error vs calibration target
            if gaze_screen_xy is not None and self._calib_target is not None:
                tx, ty = self._calib_target
                gx, gy = gaze_screen_xy
                err = float(np.hypot(gx - tx, gy - ty))
                self._gaze_errors.append(err)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flush_loop(self) -> None:
        """Background thread: flush every flush_interval_s seconds."""
        while True:
            time.sleep(1.0)
            if time.time() - self._last_flush >= self._flush_interval:
                try:
                    self._flush()
                except Exception as exc:
                    log.warning("AccuracyMonitor flush error: %s", exc)

    def _flush(self) -> None:
        """Build a report dict from accumulators and write to a JSON log file."""
        with self._lock:
            now     = time.time()
            elapsed = now - self._start_time
            report  = _build_report(
                elapsed_s        = elapsed,
                frame_count      = self._frame_count,
                emotion_counts   = dict(self._emotion_counts),
                attention_counts = dict(self._attention_counts),
                cogload_counts   = dict(self._cogload_counts),
                emotion_confusion = self._emotion_confusion.tolist(),
                attn_confusion    = self._attn_confusion.tolist(),
                feature_log       = self._feature_log[-300:],  # last 300 frames max
                gaze_errors       = list(self._gaze_errors),
                screen_wh         = (self._sw, self._sh),
            )
            self._last_flush = now

        # write to file (appended per session, one JSON object per line)
        ts_str   = datetime.datetime.now().strftime("%Y%m%d")
        log_file = self._log_dir / f"accuracy_{ts_str}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")

        log.info(
            "Monitor report @ %.0fs | emo=%s attn=%s cog=%s gaze_err=%.1fpx",
            elapsed,
            _dist_str(report["emotion"]["distribution"]),
            _dist_str(report["attention"]["distribution"]),
            _dist_str(report["cogload"]["distribution"]),
            report["gaze"]["mean_error_px"],
        )

    def flush_now(self) -> None:
        """Force an immediate flush (useful on shutdown)."""
        self._flush()


# ---------------------------------------------------------------------------
# Report builder (pure function, easy to unit-test)
# ---------------------------------------------------------------------------

def _build_report(
    elapsed_s: float,
    frame_count: int,
    emotion_counts:   dict,
    attention_counts: dict,
    cogload_counts:   dict,
    emotion_confusion: list,
    attn_confusion:    list,
    feature_log:  list[dict[str, float]],
    gaze_errors:  list[float],
    screen_wh:    tuple[int, int],
) -> dict:
    """Assemble diagnostic report dictionary."""

    ts = datetime.datetime.now().isoformat()
    sw, sh = screen_wh

    # ---- class distributions ----
    emo_dist  = _normalise_counts(emotion_counts,  list(EMOTION_NAMES.values()))
    attn_dist = _normalise_counts(attention_counts, list(ATTENTION_NAMES.values()))
    cog_dist  = _normalise_counts(cogload_counts,  list(COGLOAD_NAMES.values()))

    # ---- stuck-class warnings ----
    emo_warn  = _stuck_warning(emo_dist,  threshold=0.85)
    attn_warn = _stuck_warning(attn_dist, threshold=0.85)
    cog_warn  = _stuck_warning(cog_dist,  threshold=0.85)

    # ---- gaze error ----
    gaze_errs = np.array(gaze_errors, dtype=np.float64)
    gaze_report = {
        "n_samples":       int(len(gaze_errs)),
        "mean_error_px":   float(np.mean(gaze_errs)) if len(gaze_errs) > 0 else -1.0,
        "median_error_px": float(np.median(gaze_errs)) if len(gaze_errs) > 0 else -1.0,
        "p90_error_px":    float(np.percentile(gaze_errs, 90)) if len(gaze_errs) > 0 else -1.0,
        "mean_error_deg_approx":
            float(np.degrees(np.arctan2(np.mean(gaze_errs), (sw + sh) / 2)))
            if len(gaze_errs) > 0 else -1.0,
    }

    # ---- feature ranges ----
    feat_arr: dict[str, list[float]] = collections.defaultdict(list)
    for snap in feature_log:
        for k, v in snap.items():
            feat_arr[k].append(v)

    feature_report: dict[str, dict] = {}
    for feat_name, values in feat_arr.items():
        arr = np.array(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        lo, hi = FEATURE_RANGES.get(feat_name, (None, None))
        n_out_of_range = 0
        if lo is not None:
            n_out_of_range = int(np.sum((arr < lo) | (arr > hi)))
        feature_report[feat_name] = {
            "min":          float(np.min(arr)),
            "max":          float(np.max(arr)),
            "mean":         float(np.mean(arr)),
            "p5":           float(np.percentile(arr, 5)),
            "p95":          float(np.percentile(arr, 95)),
            "expected_min": lo,
            "expected_max": hi,
            "n_out_of_range": n_out_of_range,
            "pct_out_of_range": float(n_out_of_range / len(arr) * 100) if len(arr) > 0 else 0.0,
        }

    return {
        "timestamp":    ts,
        "elapsed_s":    round(elapsed_s, 1),
        "frame_count":  frame_count,
        "emotion": {
            "distribution":     emo_dist,
            "confusion_stable_vs_raw": emotion_confusion,
            "warning":           emo_warn,
            "label_names":       list(EMOTION_NAMES.values()),
        },
        "attention": {
            "distribution":     attn_dist,
            "confusion_stable_vs_raw": attn_confusion,
            "warning":           attn_warn,
            "label_names":       list(ATTENTION_NAMES.values()),
        },
        "cogload": {
            "distribution":     cog_dist,
            "warning":           cog_warn,
            "label_names":       list(COGLOAD_NAMES.values()),
        },
        "gaze":    gaze_report,
        "features": feature_report,
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _normalise_counts(
    counts: dict[str, int],
    all_names: list[str],
) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {name: round(counts.get(name, 0) / total, 4) for name in all_names}


def _stuck_warning(dist: dict[str, float], threshold: float = 0.85) -> Optional[str]:
    """Return warning string if one class dominates above threshold."""
    for name, frac in dist.items():
        if frac >= threshold:
            return f"STUCK ON '{name}' ({frac*100:.0f}%)"
    return None


def _name_to_idx(name: str, name_map: dict[int, str]) -> Optional[int]:
    for idx, n in name_map.items():
        if n == name:
            return idx
    return None


def _dist_str(dist: dict) -> str:
    parts = [f"{k[0].upper()}:{v:.0%}" for k, v in dist.items()]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Standalone watcher: tail the latest .jsonl and print a live dashboard
# ---------------------------------------------------------------------------

def _tail_latest_report(log_dir: Path) -> None:
    """Find the newest .jsonl in log_dir and print its last record as a table."""
    files = sorted(log_dir.glob("accuracy_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        print("[monitor] No log files found in", log_dir)
        return
    latest = files[-1]
    lines = []
    try:
        with open(latest, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        print("[monitor] Cannot read", latest)
        return
    if not lines:
        print("[monitor] Log file is empty:", latest)
        return

    try:
        rec = json.loads(lines[-1])
    except json.JSONDecodeError:
        print("[monitor] Malformed JSON in", latest)
        return

    _print_dashboard(rec, str(latest))


def _print_dashboard(rec: dict, source: str) -> None:
    """Pretty-print a single report record as an ASCII dashboard."""
    os.system("cls" if sys.platform == "win32" else "clear")
    elapsed = rec.get("elapsed_s", 0)
    frames  = rec.get("frame_count", 0)
    ts      = rec.get("timestamp", "")
    print(f"== Eye Tracking Accuracy Monitor ==  [{source}]")
    print(f"   Session elapsed: {elapsed:.0f}s   Frames recorded: {frames}   {ts}")
    print()

    # --- distributions ---
    for section_key, title in [("emotion", "EMOTION"), ("attention", "ATTENTION"), ("cogload", "COGNITIVE LOAD")]:
        sec  = rec.get(section_key, {})
        dist = sec.get("distribution", {})
        warn = sec.get("warning", None)
        label = f"{title} distribution:"
        warn_str = f"  !! {warn}" if warn else ""
        print(f"  {label}{warn_str}")
        for name, frac in dist.items():
            bar = "#" * int(frac * 30)
            print(f"    {name:<12s} {frac:5.1%}  [{bar:<30s}]")
        print()

    # --- gaze error ---
    gaze = rec.get("gaze", {})
    n_s  = gaze.get("n_samples", 0)
    if n_s > 0:
        print(f"  GAZE ERROR (vs calibration targets, {n_s} samples):")
        print(f"    mean={gaze.get('mean_error_px',-1):.1f}px  "
              f"median={gaze.get('median_error_px',-1):.1f}px  "
              f"p90={gaze.get('p90_error_px',-1):.1f}px  "
              f"~{gaze.get('mean_error_deg_approx',-1):.2f}deg")
    else:
        print("  GAZE ERROR: no calibration target samples collected yet")
    print()

    # --- feature ranges ---
    feats = rec.get("features", {})
    if feats:
        print("  FEATURE RANGES  (expected range | actual | out-of-range %)")
        for fname, finfo in feats.items():
            lo   = finfo.get("expected_min")
            hi   = finfo.get("expected_max")
            mn   = finfo.get("mean", 0.0)
            pct  = finfo.get("pct_out_of_range", 0.0)
            flag = " <-- OUT OF RANGE" if pct > 20 else ""
            exp_str = f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "[?]"
            print(f"    {fname:<30s}  exp={exp_str:<18s}  mean={mn:8.2f}  oob={pct:4.1f}%{flag}")
    print()
    print("  (refreshes every 5 s.  Ctrl-C to quit)")


def _run_watcher(log_dir: Path, interval_s: float = 5.0) -> None:
    """Continuously tail the latest report file and redraw the dashboard."""
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Watching {log_dir} for accuracy reports...")
    while True:
        _tail_latest_report(log_dir)
        try:
            time.sleep(interval_s)
        except KeyboardInterrupt:
            print("\n[monitor] Stopped.")
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Eye-tracking accuracy monitor (standalone watcher mode)"
    )
    ap.add_argument(
        "--log_dir", type=str, default="logs",
        help="Directory containing accuracy_*.jsonl files written by main.py"
    )
    ap.add_argument(
        "--interval", type=float, default=5.0,
        help="Dashboard refresh interval in seconds (default 5)"
    )
    return ap.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    _run_watcher(Path(args.log_dir), interval_s=args.interval)
