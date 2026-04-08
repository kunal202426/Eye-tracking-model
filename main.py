"""main.py
--------------------------------------------------------------------------------
Entry point for the real-time eye tracking and cognitive state analysis system.

Upgrade 8 additions:
  - Watchdog thread: auto-degrades quality if FPS drops below 15.
  - Full graceful error handling; MediaPipe reinit on crash.
  - --profile flag: per-component timing printed every second.
  - HD capture (1280x720) with 640x360 detection rescaling for speed.
  - D key: toggle live debug overlay.

Pipeline (each frame):
  1. Webcam capture (HD)
  2. EyeDetector      -- CLAHE + MediaPipe FaceLandmarker
  3. FeatureExtractor -- iris-calibrated blink / fixation / saccade
  4. ModelRunner      -- emotion MLP, attention CNN, cogload XGBoost, gaze CNN
                         + Kalman gaze smoothing + TemporalVoter stabilisation
  5. Calibration      -- affine map from Kalman-smoothed gaze to screen coords
  6. DisplayEngine    -- fullscreen HUD + heatmap + timeline + comet gaze trail

Keyboard:
  Q / ESC  -- quit
  R        -- re-run gaze calibration
  D        -- toggle debug overlay (raw logits, Kalman state)
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# ── Inference modules ─────────────────────────────────────────────────────────
_INFER_DIR = Path(__file__).parent / "inference"
sys.path.insert(0, str(_INFER_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from inference.eye_detector      import EyeDetector               # noqa: E402
from inference.feature_extractor import FeatureExtractor           # noqa: E402
from inference.model_runner      import ModelRunner                # noqa: E402
from inference.display_engine    import DisplayEngine              # noqa: E402
from tools.accuracy_monitor      import AccuracyMonitor            # noqa: E402
from calibration                 import (                          # noqa: E402
    apply_calibration,
    get_screen_resolution,
    load_calibration,
    run_calibration,
    save_calibration,
    CALIB_MATRIX_FILE,
)

# ── FPS tracking ──────────────────────────────────────────────────────────────
FPS_EMA_ALPHA   = 0.05
FPS_INIT        = 30.0
FPS_LOW_THRESH  = 15.0        # watchdog: if below this for > 2 s, degrade
FPS_WATCHDOG_S  = 2.0         # seconds below threshold before degrading

# ── Key bindings ──────────────────────────────────────────────────────────────
KEY_QUIT        = {ord("q"), ord("Q"), 27}
KEY_RECALIBRATE = {ord("r"), ord("R")}
KEY_DEBUG       = {ord("d"), ord("D")}

# ── Capture / detection ───────────────────────────────────────────────────────
CAPTURE_W        = 1280
CAPTURE_H        = 720
DETECT_W         = 640
DETECT_H         = 360

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BUFFER_SECONDS = 10.0
DEFAULT_CAM_INDEX      = 0


# ─────────────────────────────────────────────────────────────────────────────
# Watchdog
# ─────────────────────────────────────────────────────────────────────────────

class _FPSWatchdog:
    """Background thread that monitors FPS and sets quality-degradation flags."""

    def __init__(self, low_thresh: float = FPS_LOW_THRESH,
                 grace_s: float = FPS_WATCHDOG_S) -> None:
        self._thresh   = low_thresh
        self._grace_s  = grace_s
        self._fps      = FPS_INIT
        self._lock     = threading.Lock()
        self._low_start: float | None = None
        self._running  = False
        self._thread: threading.Thread | None = None

        # Degradation flags (read from main loop)
        self.disable_heatmap:   bool = False
        self.single_eye_mode:   bool = False

    def update_fps(self, fps: float) -> None:
        with self._lock:
            self._fps = fps

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True, name="fps-watchdog")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        while self._running:
            time.sleep(0.5)
            with self._lock:
                fps = self._fps

            if fps < self._thresh:
                if self._low_start is None:
                    self._low_start = time.monotonic()
                elapsed = time.monotonic() - self._low_start
                if elapsed > self._grace_s and not self.disable_heatmap:
                    self.disable_heatmap = True
                    log.warning("FPS=%.1f < %.0f: disabling heatmap.", fps, self._thresh)
                if elapsed > self._grace_s * 2 and not self.single_eye_mode:
                    self.single_eye_mode = True
                    log.warning("FPS still low: switching to single-eye inference.")
            else:
                if self._low_start is not None:
                    self._low_start = None
                    if self.disable_heatmap:
                        self.disable_heatmap = False
                        log.info("FPS recovered; re-enabling heatmap.")
                    if self.single_eye_mode:
                        self.single_eye_mode = False
                        log.info("FPS recovered; restoring dual-eye inference.")


# ─────────────────────────────────────────────────────────────────────────────
# Profiler
# ─────────────────────────────────────────────────────────────────────────────

class _Profiler:
    """Optional per-component timing accumulator."""

    STAGES = ["capture", "detect", "extract", "emotion", "attention",
              "gaze", "cogload", "display", "total"]

    def __init__(self) -> None:
        self._sums: dict[str, float] = {s: 0.0 for s in self.STAGES}
        self._count = 0
        self._last_print = time.monotonic()

    def record(self, stage: str, dt_s: float) -> None:
        self._sums[stage] += dt_s * 1000.0   # ms

    def tick(self) -> None:
        self._count += 1
        if time.monotonic() - self._last_print >= 1.0 and self._count > 0:
            parts = [f"{s}={self._sums[s]/self._count:.1f}ms"
                     for s in self.STAGES if self._sums[s] > 0]
            log.info("PROFILE  %s  (avg over %d frames)", " | ".join(parts), self._count)
            self._sums  = {s: 0.0 for s in self.STAGES}
            self._count = 0
            self._last_print = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# Calibration helper
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_calibration(
    model_dir: Path,
    screen_w: int,
    screen_h: int,
    cap: cv2.VideoCapture,
    runner: ModelRunner,
    detector: EyeDetector,
    force: bool = False,
) -> tuple[np.ndarray, tuple[int, int]]:
    calib_path = model_dir / CALIB_MATRIX_FILE

    if not force and calib_path.exists():
        log.info("Loading existing calibration from %s", calib_path)
        return load_calibration(model_dir)

    log.info("Running 9-point gaze calibration ...")
    M = run_calibration(screen_w, screen_h, cap, runner, detector)
    if M is None:
        if calib_path.exists():
            log.warning("New calibration failed. Using previous calibration.")
            return load_calibration(model_dir)
        log.error("Calibration failed and no existing calibration found.")
        sys.exit(1)

    save_calibration(M, (screen_w, screen_h), model_dir)
    return M, (screen_w, screen_h)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(
    data_path: Path,
    cam_index: int        = DEFAULT_CAM_INDEX,
    recalibrate: bool     = False,
    skip_calibration: bool = False,
    buffer_seconds: float = DEFAULT_BUFFER_SECONDS,
    profile: bool         = False,
) -> None:
    """Load all components and start the real-time tracking loop."""
    model_dir          = data_path / "saved_models"
    screen_w, screen_h = get_screen_resolution()
    log.info("Screen: %dx%d", screen_w, screen_h)

    # ── Load inference components ─────────────────────────────────────────────
    log.info("Loading EyeDetector ...")
    try:
        detector = EyeDetector.from_model_dir(model_dir)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    log.info("Loading ModelRunner ...")
    runner    = ModelRunner(model_dir)
    extractor = FeatureExtractor(buffer_seconds=buffer_seconds)

    # ── Open webcam (multi-backend fallback) ──────────────────────────────────
    cap = None
    for backend, bname in [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF,  "MSMF"),
        (cv2.CAP_ANY,   "ANY"),
    ]:
        c = cv2.VideoCapture(cam_index, backend)
        if c.isOpened():
            cap = c
            log.info("Webcam opened (index=%d, backend=%s).", cam_index, bname)
            break
        c.release()
    if cap is None or not cap.isOpened():
        log.error("Cannot open webcam at index %d.", cam_index)
        detector.close()
        sys.exit(1)

    # Request HD capture; falls back silently if camera does not support it
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Capture resolution: %dx%d", actual_w, actual_h)

    # ── Calibration ───────────────────────────────────────────────────────────
    calib_path = model_dir / CALIB_MATRIX_FILE
    if skip_calibration:
        log.warning("Skipping calibration. Gaze dot will not be screen-accurate.")
        calib_matrix: np.ndarray = np.array(
            [[float(screen_w), 0.0, 0.0],
             [0.0, float(screen_h), 0.0]],
            dtype=np.float64,
        )
        calib_screen_wh: tuple[int, int] = (screen_w, screen_h)
    else:
        calib_matrix, calib_screen_wh = _ensure_calibration(
            model_dir, screen_w, screen_h, cap, runner, detector,
            force=recalibrate,
        )

    # ── Display engine ────────────────────────────────────────────────────────
    engine = DisplayEngine(screen_w, screen_h)

    # ── Watchdog + profiler ───────────────────────────────────────────────────
    watchdog = _FPSWatchdog()
    watchdog.start()
    prof = _Profiler() if profile else None

    # ── Accuracy monitor (background flush every 30 s) ────────────────────────
    monitor = AccuracyMonitor(
        log_dir      = data_path / "logs",
        screen_wh    = (screen_w, screen_h),
    )

    # ── Session state ─────────────────────────────────────────────────────────
    fps          = FPS_INIT
    debug_mode   = False
    last_preds   = None      # keep last valid prediction for graceful degradation

    log.info("Real-time loop starting. Press Q/ESC to quit, R to recalibrate, D to debug.")

    try:
        while True:
            t_frame = time.monotonic()

            # 1. Capture
            t0 = time.monotonic()
            try:
                ret, frame = cap.read()
            except Exception as exc:
                log.warning("Capture error: %s", exc)
                continue
            if not ret or frame is None:
                log.warning("Empty frame; skipping.")
                continue
            if prof:
                prof.record("capture", time.monotonic() - t0)

            # Resize to detection resolution for speed (Upgrade 8)
            fH, fW = frame.shape[:2]
            if fW > DETECT_W or fH > DETECT_H:
                detect_frame = cv2.resize(frame, (DETECT_W, DETECT_H),
                                          interpolation=cv2.INTER_LINEAR)
            else:
                detect_frame = frame

            # 2. Eye detection (run on smaller frame)
            t0 = time.monotonic()
            try:
                attn_stable = (
                    getattr(last_preds, "attention_stable_name", None)
                    if last_preds else None
                )
                gaze_cx_prev = (
                    getattr(last_preds, "gaze_cx_smooth", None)
                    if last_preds else None
                )
                gaze_cy_prev = (
                    getattr(last_preds, "gaze_cy_smooth", None)
                    if last_preds else None
                )
                det_result = detector.process(
                    detect_frame,
                    attention_name = attn_stable,
                    gaze_cx        = gaze_cx_prev,
                    gaze_cy        = gaze_cy_prev,
                )
            except Exception as exc:
                log.warning("EyeDetector crashed: %s -- reinitialising ...", exc)
                try:
                    detector.close()
                    detector = EyeDetector.from_model_dir(model_dir)
                except Exception as re_exc:
                    log.error("Reinitialisation failed: %s", re_exc)
                continue
            if prof:
                prof.record("detect", time.monotonic() - t0)

            # Skip model updates when head is turned (confidence gating)
            feats = None
            if det_result.head_turned:
                preds = last_preds
            else:
                # 3. Feature extraction
                t0 = time.monotonic()
                extractor.update(
                    mean_ear             = det_result.mean_ear,
                    face_present         = det_result.face_detected,
                    timestamp_s          = t_frame,
                    iris_diameter_px     = det_result.iris_diameter_px,
                    inter_ocular_dist_px = det_result.inter_ocular_dist_px,
                    left_pupil_center    = det_result.left_pupil_center,
                )
                feats = extractor.get_features()
                if prof:
                    prof.record("extract", time.monotonic() - t0)

                # 4a. Select eye crop (single-eye mode if watchdog triggered)
                eye_crop = det_result.left_eye_crop
                if eye_crop is None:
                    eye_crop = det_result.right_eye_crop
                elif not watchdog.single_eye_mode and det_result.right_eye_crop is not None:
                    # Use left crop when both available (dual-eye mode)
                    eye_crop = det_result.left_eye_crop

                # 4b. Emotion inference
                t0 = time.monotonic()
                try:
                    preds = runner.run(
                        eye_crop_bgr  = eye_crop,
                        emotion_feats = feats.emotion_vector(),
                        cogload_feats = feats.cogload_vector(),
                    )
                    last_preds = preds
                except Exception as exc:
                    log.warning("Model inference error: %s -- using last valid.", exc)
                    preds = last_preds
                if prof:
                    prof.record("total", time.monotonic() - t0)

            # Use last_preds as fallback if preds is still None
            if preds is None:
                preds = last_preds

            # 5. Gaze screen position (use Kalman-smoothed output)
            gaze_xy: tuple[int, int] | None = None
            if preds is not None and getattr(preds, "gaze_cx_smooth", None) is not None:
                gaze_xy = apply_calibration(
                    preds.gaze_cx_smooth, preds.gaze_cy_smooth, calib_matrix,
                )
                gaze_xy = (
                    max(0, min(screen_w - 1, gaze_xy[0])),
                    max(0, min(screen_h - 1, gaze_xy[1])),
                )

            # Monitor diagnostic data
            monitor.record(feats, preds, gaze_xy)

            # 6. Render -- use annotated detect_frame for display
            t0 = time.monotonic()
            try:
                # Disable heatmap rendering when watchdog says so
                # (we achieve this by passing an all-black gaze only on degraded mode)
                render_gaze = None if watchdog.disable_heatmap else gaze_xy

                canvas = engine.render(
                    det_result.annotated_frame,
                    preds if preds is not None else object(),
                    gaze_screen_xy = render_gaze,
                    fps            = fps,
                    face_detected  = det_result.face_detected,
                    left_ear       = det_result.left_ear,
                    right_ear      = det_result.right_ear,
                    blink_rate     = (
                        feats.blink_rate_per_min
                        if not det_result.head_turned else None
                    ) if not det_result.head_turned else None,
                    head_turned    = det_result.head_turned,
                    debug_mode     = debug_mode,
                )
                # Still draw gaze dot even in heatmap-disabled mode
                if watchdog.disable_heatmap and gaze_xy is not None and canvas is not None:
                    attn_col = {
                        "focused": (0, 210, 40),
                        "distracted": (0, 210, 210),
                        "off_task": (50, 50, 220),
                    }.get(
                        getattr(preds, "attention_stable_name", None) or "", (200, 200, 0)
                    )
                    cv2.circle(canvas, gaze_xy, 14, attn_col, -1)
            except Exception as exc:
                log.warning("Display render error: %s", exc)
                canvas = None
            if prof:
                prof.record("display", time.monotonic() - t0)

            if canvas is not None:
                key = engine.show(canvas)
            else:
                key = -1

            # 7. Key handling
            if key in KEY_QUIT or not engine.is_open():
                log.info("Quit requested.")
                break
            elif key in KEY_RECALIBRATE:
                log.info("Recalibration requested (key R).")
                calib_matrix, calib_screen_wh = _ensure_calibration(
                    model_dir, screen_w, screen_h, cap, runner, detector,
                    force=True,
                )
            elif key in KEY_DEBUG:
                debug_mode = not debug_mode
                log.info("Debug overlay: %s", "ON" if debug_mode else "OFF")

            # 8. FPS update + watchdog
            dt  = time.monotonic() - t_frame
            fps = fps * (1 - FPS_EMA_ALPHA) + (1.0 / dt) * FPS_EMA_ALPHA if dt > 0 else fps
            watchdog.update_fps(fps)

            if prof:
                prof.tick()

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt -- shutting down.")
    finally:
        watchdog.stop()
        cap.release()
        monitor.flush_now()
        try:
            detector.close()
        except Exception:
            pass
        engine.close()
        log.info("Session ended. Final FPS estimate: %.1f", fps)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Real-time eye tracking and cognitive state analysis. "
            "Requires face_landmarker.task in saved_models/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=Path, required=True,
                        help="Project root directory.")
    parser.add_argument("--cam_index", type=int, default=DEFAULT_CAM_INDEX,
                        help="Webcam device index.")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Force gaze calibration even if calibration.npy already exists.")
    parser.add_argument("--skip_calibration", action="store_true",
                        help="Skip gaze calibration (gaze dot will be inaccurate).")
    parser.add_argument("--buffer_seconds", type=float, default=DEFAULT_BUFFER_SECONDS,
                        help="Feature extractor rolling window in seconds.")
    parser.add_argument("--profile", action="store_true",
                        help="Print per-component timing every second.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser    = _build_arg_parser()
    args      = parser.parse_args(argv)
    data_path = args.data_path.resolve()

    if not data_path.exists():
        log.error("data_path does not exist: %s", data_path)
        sys.exit(1)

    if args.recalibrate and args.skip_calibration:
        log.error("--recalibrate and --skip_calibration are mutually exclusive.")
        sys.exit(1)

    run(
        data_path        = data_path,
        cam_index        = args.cam_index,
        recalibrate      = args.recalibrate,
        skip_calibration = args.skip_calibration,
        buffer_seconds   = args.buffer_seconds,
        profile          = args.profile,
    )


if __name__ == "__main__":
    main()
