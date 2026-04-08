"""inference/feature_extractor.py
--------------------------------------------------------------------------------
Rolling-window feature extraction from EAR and iris signals.

Upgrade 4 improvements:
  BLINK DETECTION:
    - Two-threshold state machine: EAR drops below 0.21 to open,
      must recover above 0.25 within 10 frames (real blink vs squint).
    - 90-frame EAR history (3 s) for accurate rate estimation.
  PUPIL DILATION:
    - Uses iris diameter from iris landmarks, normalised by inter-ocular
      distance to remove distance-from-camera effect.
    - Median filtered over last 15 frames to reduce noise.
  FIXATION/SACCADE:
    - Gaze velocity from frame-to-frame pupil centre displacement.
    - Velocity thresholds: <5 px/frame = fixation, >20 px/frame = saccade.
    - Fixation duration and saccade amplitude from consecutive run lengths.

Feature mapping:
  Emotion MLP  : [num_blink, mean_blink_duration_ms, mean_fixation_duration_ms,
                  mean_saccade_amplitude]
  CogLoad XGBoost: [pupil_dilation_proxy, blink_rate_per_min,
                    fixation_duration_ms, saccade_duration_ms]
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np

log = logging.getLogger("feature_extractor")

# ── Blink detection thresholds ────────────────────────────────────────────────
EAR_BLINK_ONSET     = 0.21   # EAR drops below this -> potential blink
EAR_BLINK_RECOVERY  = 0.25   # EAR must rise above this to confirm blink
EAR_MAX_BLINK_FRAMES = 10    # if eye stays closed > this, it is a squint not blink
EAR_CONSEC_MIN_FRAMES = 2    # must stay below onset for >= N consecutive frames

# ── Gaze velocity thresholds ──────────────────────────────────────────────────
FIXATION_VEL_THRESH  = 5.0   # px/frame: below = fixation
SACCADE_VEL_THRESH   = 20.0  # px/frame: above = saccade

# ── Pupil dilation ────────────────────────────────────────────────────────────
DILATION_MEDIAN_WINDOW = 15  # frames to median-filter iris diameter ratio

# ── Legacy EAR-based saccade threshold (kept for fallback) ───────────────────
EAR_SACCADE_DIFF_THRESH = 0.02

# ── Feature window ────────────────────────────────────────────────────────────
DEFAULT_BUFFER_SECONDS = 10.0
DEFAULT_FPS            = 30.0
EAR_HISTORY_FRAMES     = 90   # 3 seconds at 30 fps

# ── Fallback defaults ─────────────────────────────────────────────────────────
FALLBACK_EAR           = 0.30
FALLBACK_DURATION_MS   = 300.0
FALLBACK_BLINK_RATE    = 15.0
FALLBACK_DILATION      = 30.0  # arbitrary normalised unit


# ─────────────────────────────────────────────────────────────────────────────
# Internal data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class _FrameSample:
    timestamp_s:  float
    mean_ear:     float    # nan if no face
    face_present: bool
    pupil_x:      float    # screen-space pupil x, or nan
    pupil_y:      float    # screen-space pupil y, or nan
    iris_diam:    float    # iris diameter in px, or nan
    iod:          float    # inter-ocular distance in px, or nan


@dataclasses.dataclass(frozen=True)
class _BlinkEvent:
    start_s:     float
    end_s:       float
    duration_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# Output feature dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ExtractedFeatures:
    """Feature snapshot from the rolling buffer.

    Attributes:
        num_blink:                Total blinks in current window.
        mean_blink_duration_ms:   Mean blink duration (ms).
        mean_fixation_duration_ms: Mean fixation interval (ms).
        mean_saccade_amplitude:   Mean saccade displacement (px).
        pupil_dilation_proxy:     Normalised iris diameter ratio * 100.
        blink_rate_per_min:       Blinks per minute.
        fixation_duration_ms:     Same as mean_fixation_duration_ms.
        saccade_duration_ms:      Mean duration of saccade runs (ms).
        window_duration_s:        Span covered by the buffer.
        n_valid_frames:           Frames with face detected.
        raw_ear_mean:             Mean EAR over window (for debug).
        current_velocity_px:      Latest gaze velocity in px/frame.
    """
    num_blink:                float
    mean_blink_duration_ms:   float
    mean_fixation_duration_ms: float
    mean_saccade_amplitude:   float

    pupil_dilation_proxy:  float
    blink_rate_per_min:    float
    fixation_duration_ms:  float
    saccade_duration_ms:   float

    window_duration_s: float
    n_valid_frames:    int
    raw_ear_mean:      float
    current_velocity_px: float

    def emotion_vector(self) -> np.ndarray:
        """4-D feature vector for the emotion MLP."""
        return np.array([
            self.num_blink,
            self.mean_blink_duration_ms,
            self.mean_fixation_duration_ms,
            self.mean_saccade_amplitude,
        ], dtype=np.float32)

    def cogload_vector(self) -> np.ndarray:
        """4-D feature vector for the cognitive-load XGBoost model."""
        return np.array([
            self.pupil_dilation_proxy,
            self.blink_rate_per_min,
            self.fixation_duration_ms,
            self.saccade_duration_ms,
        ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Improved blink detector (Upgrade 4)
# ─────────────────────────────────────────────────────────────────────────────

class _BlinkDetector:
    """Two-threshold blink detector with squint rejection.

    A real blink: EAR drops below EAR_BLINK_ONSET for >= EAR_CONSEC_MIN_FRAMES
    AND recovers above EAR_BLINK_RECOVERY within EAR_MAX_BLINK_FRAMES.
    Prolonged closure (squint) is rejected if recovery takes longer.
    """

    def __init__(self) -> None:
        self._below_count:  int            = 0
        self._blink_start:  Optional[float] = None
        self._in_blink:     bool           = False

    def update(self, ear: float, ts: float) -> Optional[_BlinkEvent]:
        """Feed one EAR sample. Returns _BlinkEvent when a blink completes."""
        if ear < EAR_BLINK_ONSET:
            if not self._in_blink:
                if self._blink_start is None:
                    self._blink_start = ts
            self._below_count += 1
            if self._below_count >= EAR_CONSEC_MIN_FRAMES:
                self._in_blink = True
            # Squint rejection
            if self._below_count > EAR_MAX_BLINK_FRAMES:
                self._reset()
            return None
        else:
            # EAR above onset threshold
            if self._in_blink and ear >= EAR_BLINK_RECOVERY:
                # Confirmed blink
                event = _BlinkEvent(
                    start_s     = self._blink_start or ts,
                    end_s       = ts,
                    duration_ms = (ts - (self._blink_start or ts)) * 1000.0,
                )
                self._reset()
                return event
            elif not self._in_blink:
                self._reset()
            return None

    def _reset(self) -> None:
        self._below_count = 0
        self._blink_start = None
        self._in_blink    = False

    def reset(self) -> None:
        self._reset()


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """Converts a stream of detection results into tabular features.

    update() accepts both original and new (Upgrade 4) fields.
    get_features() returns an ExtractedFeatures snapshot.
    """

    def __init__(
        self,
        buffer_seconds: float = DEFAULT_BUFFER_SECONDS,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self._buffer_s = buffer_seconds
        self._fps      = fps
        frame_cap      = int(buffer_seconds * fps) + 10

        self._samples:  Deque[_FrameSample] = collections.deque(maxlen=frame_cap * 4)
        self._blinks:   Deque[_BlinkEvent]  = collections.deque(maxlen=500)
        self._blink_det = _BlinkDetector()

        # Pupil dilation median buffer
        self._dilation_buf: Deque[float] = collections.deque(maxlen=DILATION_MEDIAN_WINDOW)

        # Latest gaze velocity
        self._prev_pupil_xy: Optional[tuple[float, float]] = None
        self._cur_velocity:  float = 0.0

        # Fixation/saccade run tracking
        self._fixation_runs:  List[int] = []
        self._saccade_runs:   List[int] = []
        self._fix_run_len:    int = 0
        self._sac_run_len:    int = 0
        self._sac_max_disp:   float = 0.0   # max displacement in current saccade

    # ── Public interface ──────────────────────────────────────────────────────

    def update(
        self,
        mean_ear: Optional[float],
        face_present: bool,
        timestamp_s: float,
        # New Upgrade 4 fields (all optional for backward compatibility)
        iris_diameter_px:     Optional[float] = None,
        inter_ocular_dist_px: Optional[float] = None,
        left_pupil_center:    Optional[tuple[int, int]] = None,
    ) -> None:
        """Feed one frame's detection data into the rolling buffer."""
        ear = mean_ear if (face_present and mean_ear is not None) else float("nan")

        # Pupil position
        px_f = float(left_pupil_center[0]) if left_pupil_center else float("nan")
        py_f = float(left_pupil_center[1]) if left_pupil_center else float("nan")

        # Iris diameter (normalised)
        iod_f  = float(inter_ocular_dist_px) if inter_ocular_dist_px else float("nan")
        diam_f = float(iris_diameter_px)     if iris_diameter_px     else float("nan")

        sample = _FrameSample(
            timestamp_s  = timestamp_s,
            mean_ear     = ear,
            face_present = face_present,
            pupil_x      = px_f,
            pupil_y      = py_f,
            iris_diam    = diam_f,
            iod          = iod_f,
        )
        self._samples.append(sample)

        # Pupil dilation normalised ratio
        if (not np.isnan(diam_f) and not np.isnan(iod_f) and iod_f > 1.0):
            self._dilation_buf.append((diam_f / iod_f) * 100.0)

        # Gaze velocity and fixation/saccade classification
        if not np.isnan(px_f) and not np.isnan(py_f):
            if self._prev_pupil_xy is not None:
                dx = px_f - self._prev_pupil_xy[0]
                dy = py_f - self._prev_pupil_xy[1]
                vel = float(np.sqrt(dx * dx + dy * dy))
                self._cur_velocity = vel

                disp = vel  # displacement this frame

                if vel < FIXATION_VEL_THRESH:
                    # Fixation frame
                    if self._sac_run_len > 0:
                        self._saccade_runs.append(self._sac_run_len)
                    self._sac_run_len = 0
                    self._sac_max_disp = 0.0
                    self._fix_run_len += 1
                elif vel > SACCADE_VEL_THRESH:
                    # Saccade frame
                    if self._fix_run_len > 0:
                        self._fixation_runs.append(self._fix_run_len)
                    self._fix_run_len = 0
                    self._sac_run_len += 1
                    if disp > self._sac_max_disp:
                        self._sac_max_disp = disp

            self._prev_pupil_xy = (px_f, py_f)

        # Blink detection (only when face present and valid EAR)
        if face_present and not np.isnan(ear):
            event = self._blink_det.update(ear, timestamp_s)
            if event is not None:
                self._blinks.append(event)

        # Trim old data
        self._trim(timestamp_s)

    def get_features(self) -> ExtractedFeatures:
        """Compute and return the current feature snapshot."""
        samples = list(self._samples)
        blinks  = list(self._blinks)

        # Window duration
        window_s = (
            samples[-1].timestamp_s - samples[0].timestamp_s
            if len(samples) >= 2 else 0.0
        )

        # Valid EAR values
        valid_ears = [
            s.mean_ear for s in samples
            if s.face_present and not np.isnan(s.mean_ear)
        ]
        n_valid  = len(valid_ears)
        ear_mean = float(np.mean(valid_ears)) if valid_ears else FALLBACK_EAR

        # ── Blink stats ───────────────────────────────────────────────────────
        num_blink    = float(len(blinks))
        blink_durs   = [b.duration_ms for b in blinks]
        mean_blink_duration_ms = (
            float(np.mean(blink_durs)) if blink_durs else FALLBACK_DURATION_MS
        )

        if window_s > 0.5:
            blink_rate_per_min = num_blink / (window_s / 60.0)
        else:
            blink_rate_per_min = FALLBACK_BLINK_RATE

        # ── Fixation duration (inter-blink interval) ──────────────────────────
        if len(blinks) >= 2:
            intervals = [
                (blinks[i + 1].start_s - blinks[i].end_s) * 1000.0
                for i in range(len(blinks) - 1)
                if blinks[i + 1].start_s > blinks[i].end_s
            ]
            mean_fixation_duration_ms = (
                float(np.mean(intervals)) if intervals else FALLBACK_DURATION_MS
            )
        else:
            mean_fixation_duration_ms = (
                60_000.0 / blink_rate_per_min
                if blink_rate_per_min > 0 else FALLBACK_DURATION_MS
            )

        # ── Fixation duration from gaze velocity (Upgrade 4) ─────────────────
        if self._fixation_runs:
            frame_ms = 1000.0 / self._fps
            fix_dur_from_vel = float(np.mean(self._fixation_runs)) * frame_ms
            # Average with inter-blink approach
            fixation_duration_ms = (mean_fixation_duration_ms + fix_dur_from_vel) / 2.0
        else:
            fixation_duration_ms = mean_fixation_duration_ms

        # ── Saccade amplitude from gaze velocity (Upgrade 4) ──────────────────
        if self._saccade_runs:
            mean_saccade_amplitude = float(self._sac_max_disp) if self._sac_max_disp > 0 else 0.0
        else:
            # Fallback: EAR-based amplitude (legacy)
            ear_diffs = []
            prev = None
            for s in samples:
                e = s.mean_ear if not np.isnan(s.mean_ear) else FALLBACK_EAR
                if prev is not None:
                    d = abs(e - prev)
                    if d > EAR_SACCADE_DIFF_THRESH:
                        ear_diffs.append(d)
                prev = e
            mean_saccade_amplitude = float(np.mean(ear_diffs)) if ear_diffs else 0.0

        # ── Saccade duration from gaze velocity ───────────────────────────────
        if self._saccade_runs:
            frame_ms = 1000.0 / self._fps
            saccade_duration_ms = float(np.mean(self._saccade_runs[-20:])) * frame_ms
        else:
            saccade_duration_ms = FALLBACK_DURATION_MS

        # ── Pupil dilation (Upgrade 4) ─────────────────────────────────────────
        if self._dilation_buf:
            pupil_dilation_proxy = float(np.median(list(self._dilation_buf)))
        else:
            # Fallback: mean EAR as proxy
            pupil_dilation_proxy = ear_mean

        return ExtractedFeatures(
            num_blink                 = num_blink,
            mean_blink_duration_ms    = mean_blink_duration_ms,
            mean_fixation_duration_ms = mean_fixation_duration_ms,
            mean_saccade_amplitude    = mean_saccade_amplitude,
            pupil_dilation_proxy      = pupil_dilation_proxy,
            blink_rate_per_min        = blink_rate_per_min,
            fixation_duration_ms      = fixation_duration_ms,
            saccade_duration_ms       = saccade_duration_ms,
            window_duration_s         = window_s,
            n_valid_frames            = n_valid,
            raw_ear_mean              = ear_mean,
            current_velocity_px       = self._cur_velocity,
        )

    def reset(self) -> None:
        """Clear all buffers and detector state."""
        self._samples.clear()
        self._blinks.clear()
        self._dilation_buf.clear()
        self._blink_det.reset()
        self._prev_pupil_xy = None
        self._cur_velocity  = 0.0
        self._fixation_runs.clear()
        self._saccade_runs.clear()
        self._fix_run_len   = 0
        self._sac_run_len   = 0
        self._sac_max_disp  = 0.0
        log.debug("FeatureExtractor reset.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _trim(self, now_s: float) -> None:
        cutoff = now_s - self._buffer_s
        while self._samples and self._samples[0].timestamp_s < cutoff:
            self._samples.popleft()
        while self._blinks and self._blinks[0].end_s < cutoff:
            self._blinks.popleft()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Live EAR-based feature extraction readout. Runs eye_detector "
            "and prints feature values to the log every second."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=Path, required=True,
                        help="Project root directory.")
    parser.add_argument("--cam_index", type=int, default=0,
                        help="OpenCV VideoCapture device index.")
    parser.add_argument("--buffer_seconds", type=float,
                        default=DEFAULT_BUFFER_SECONDS,
                        help="Feature window length in seconds.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run live feature-extraction readout using webcam input."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    import time
    import cv2

    _here = Path(__file__).parent
    sys.path.insert(0, str(_here))
    from eye_detector import EyeDetector

    parser    = _build_arg_parser()
    args      = parser.parse_args(argv)
    data_path = args.data_path.resolve()

    if not data_path.exists():
        log.error("data_path does not exist: %s", data_path)
        sys.exit(1)

    model_dir = data_path / "saved_models"
    try:
        detector = EyeDetector.from_model_dir(model_dir)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    cap = None
    for backend, bname in [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF,  "MSMF"),
        (cv2.CAP_ANY,   "ANY"),
    ]:
        c = cv2.VideoCapture(args.cam_index, backend)
        if c.isOpened():
            cap = c
            break
        c.release()
    if cap is None or not cap.isOpened():
        log.error("Cannot open webcam at index %d.", args.cam_index)
        detector.close()
        sys.exit(1)

    extractor   = FeatureExtractor(buffer_seconds=args.buffer_seconds)
    last_print  = time.monotonic()
    PRINT_EVERY = 1.0

    log.info("Feature extraction running. Press Q to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            t   = time.monotonic()
            res = detector.process(frame)
            extractor.update(
                mean_ear             = res.mean_ear,
                face_present         = res.face_detected,
                timestamp_s          = t,
                iris_diameter_px     = res.iris_diameter_px,
                inter_ocular_dist_px = res.inter_ocular_dist_px,
                left_pupil_center    = res.left_pupil_center,
            )

            if t - last_print >= PRINT_EVERY:
                feats = extractor.get_features()
                log.info(
                    "EMOTION  num_blink=%.0f  blink_dur=%.0f ms  "
                    "fix_dur=%.0f ms  sacc_amp=%.2f px",
                    feats.num_blink, feats.mean_blink_duration_ms,
                    feats.mean_fixation_duration_ms, feats.mean_saccade_amplitude,
                )
                log.info(
                    "COGLOAD  dilation=%.1f  blink_rate=%.1f/min  "
                    "fix=%.0f ms  sacc=%.0f ms  vel=%.1f px/fr",
                    feats.pupil_dilation_proxy, feats.blink_rate_per_min,
                    feats.fixation_duration_ms, feats.saccade_duration_ms,
                    feats.current_velocity_px,
                )
                last_print = t

            vis = res.annotated_frame
            if res.face_detected:
                cv2.putText(
                    vis,
                    f"EAR={res.mean_ear:.3f}  blinks={int(extractor.get_features().num_blink)}"
                    f"  vel={extractor._cur_velocity:.1f}px/fr",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 0), 1, cv2.LINE_AA,
                )
            cv2.imshow("Feature Extractor", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
