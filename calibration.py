"""calibration.py
--------------------------------------------------------------------------------
Gaze calibration for the real-time eye tracking pipeline.

Upgrade 10:
  - 13-point option (--calibration_points 9|13) for better accuracy at edges.
  - Per-point quality scoring: warns when a point has high gaze variance.
    Low-quality points are re-collected automatically (up to 1 retry).
  - Polynomial (degree-2) regression as an alternative to affine transform
    (--calibration_model affine|polynomial).
  - Metadata file saved alongside the matrix (timestamp, model_type, quality).

Backward compatibility:
  apply_calibration(cx, cy, matrix) auto-detects matrix shape:
    (2, 3) -> affine
    (2, 6) -> polynomial degree-2
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import logging
import sys
import time
import ctypes
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("calibration")

# ── Artefact filenames ────────────────────────────────────────────────────────
CALIB_MATRIX_FILE = "calibration.npy"
CALIB_SCREEN_FILE = "calibration_screen.npy"
CALIB_META_FILE   = "calibration_meta.npz"

# ── Calibration point layouts ─────────────────────────────────────────────────
CALIB_POINTS_9: list[tuple[float, float]] = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]

CALIB_POINTS_13: list[tuple[float, float]] = CALIB_POINTS_9 + [
    (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7),
]

# ── Timing ────────────────────────────────────────────────────────────────────
FIXATION_DELAY_S    = 1.5
COLLECT_DURATION_S  = 1.0
MIN_VALID_SAMPLES   = 10

# ── Quality thresholds ────────────────────────────────────────────────────────
QUALITY_VAR_THRESH   = 0.002  # sum of variance in cx + cy; above = low quality
QUALITY_MAX_RETRIES  = 1      # re-collect a bad point at most once

# ── Visual constants ──────────────────────────────────────────────────────────
DOT_RADIUS_IDLE   = 20
DOT_RADIUS_ACTIVE = 30
DOT_RADIUS_DONE   = 12
DOT_RADIUS_BAD    = 12
COLOR_IDLE        = (120, 120, 120)
COLOR_ACTIVE      = (0, 220, 255)
COLOR_DONE        = (0, 220, 0)
COLOR_BAD         = (50,  50, 220)
COLOR_BG          = (10, 10, 10)
FONT              = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE        = 0.65
FONT_THICK        = 2
WINDOW_NAME       = "Gaze Calibration"


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _CalibPoint:
    screen_x_px:  float
    screen_y_px:  float
    gaze_cx_mean: float
    gaze_cy_mean: float
    n_samples:    int
    quality:      float   # sum of gaze cx + cy variance (lower is better)


# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────

def _poly2_features(cx: float, cy: float) -> np.ndarray:
    """Return degree-2 polynomial feature vector [1, cx, cy, cx^2, cx*cy, cy^2]."""
    return np.array([1.0, cx, cy, cx * cx, cx * cy, cy * cy], dtype=np.float64)


def fit_affine_transform(
    gaze_pts: np.ndarray,
    screen_pts: np.ndarray,
) -> np.ndarray:
    """Fit a least-squares affine transform from gaze -> screen.

    Args:
        gaze_pts:   (N, 2) float array of (cx_norm, cy_norm).
        screen_pts: (N, 2) float array of (x_px, y_px).

    Returns:
        Matrix M of shape (2, 3).
    """
    if len(gaze_pts) < 3:
        raise ValueError(f"Need >= 3 points, got {len(gaze_pts)}.")
    A = np.hstack([gaze_pts, np.ones((len(gaze_pts), 1))])
    mx, _, _, _ = np.linalg.lstsq(A, screen_pts[:, 0], rcond=None)
    my, _, _, _ = np.linalg.lstsq(A, screen_pts[:, 1], rcond=None)
    return np.stack([mx, my]).astype(np.float64)    # (2, 3)


def fit_polynomial_transform(
    gaze_pts: np.ndarray,
    screen_pts: np.ndarray,
) -> np.ndarray:
    """Fit a degree-2 polynomial transform from gaze -> screen.

    Args:
        gaze_pts:   (N, 2) float array.
        screen_pts: (N, 2) float array.

    Returns:
        Coefficient matrix of shape (2, 6).
    """
    if len(gaze_pts) < 6:
        raise ValueError(f"Polynomial degree-2 needs >= 6 points, got {len(gaze_pts)}.")
    A = np.stack([_poly2_features(cx, cy) for cx, cy in gaze_pts])  # (N, 6)
    mx, _, _, _ = np.linalg.lstsq(A, screen_pts[:, 0], rcond=None)
    my, _, _, _ = np.linalg.lstsq(A, screen_pts[:, 1], rcond=None)
    return np.stack([mx, my]).astype(np.float64)    # (2, 6)


def apply_calibration(
    cx_norm: float,
    cy_norm: float,
    matrix: np.ndarray,
) -> tuple[int, int]:
    """Map normalised gaze -> screen pixels.

    Automatically detects matrix type by shape:
      (2, 3) -> affine
      (2, 6) -> polynomial degree-2

    Args:
        cx_norm: Normalised pupil centroid x in [0, 1].
        cy_norm: Normalised pupil centroid y in [0, 1].
        matrix:  (2, 3) or (2, 6) transform matrix.

    Returns:
        Integer (screen_x_px, screen_y_px).
    """
    if matrix.shape == (2, 3):
        v = np.array([cx_norm, cy_norm, 1.0], dtype=np.float64)
    elif matrix.shape == (2, 6):
        v = _poly2_features(cx_norm, cy_norm)
    else:
        raise ValueError(f"Unexpected calibration matrix shape: {matrix.shape}")

    out = matrix @ v
    return int(round(float(out[0]))), int(round(float(out[1])))


# ─────────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_calibration(
    matrix: np.ndarray,
    screen_wh: tuple[int, int],
    save_dir: Path,
    quality_scores: Optional[list[float]] = None,
    model_type: str = "affine",
) -> tuple[Path, Path]:
    """Save calibration matrix, screen reference, and metadata."""
    save_dir.mkdir(parents=True, exist_ok=True)
    m_path = save_dir / CALIB_MATRIX_FILE
    s_path = save_dir / CALIB_SCREEN_FILE
    meta_p = save_dir / CALIB_META_FILE

    np.save(str(m_path), matrix.astype(np.float64))
    np.save(str(s_path), np.array(list(screen_wh), dtype=np.int32))

    # Metadata
    meta_qs = np.array(quality_scores or [], dtype=np.float64)
    ts_str  = datetime.datetime.now().isoformat()
    np.savez(str(meta_p),
             timestamp=np.array([ts_str]),
             model_type=np.array([model_type]),
             quality_scores=meta_qs)

    log.info("Calibration saved -> %s  (model=%s)", m_path, model_type)
    log.info("Reference screen  -> %dx%d  (%s)", screen_wh[0], screen_wh[1], s_path)
    return m_path, s_path


def load_calibration(save_dir: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load calibration artefacts from disk."""
    m_path = save_dir / CALIB_MATRIX_FILE
    s_path = save_dir / CALIB_SCREEN_FILE

    if not m_path.exists():
        raise FileNotFoundError(
            f"Calibration matrix not found: {m_path}\n"
            "Run: python calibration.py --data_path <project_root>"
        )
    if not s_path.exists():
        raise FileNotFoundError(f"Calibration screen file not found: {s_path}")

    matrix    = np.load(str(m_path)).astype(np.float64)
    screen_wh = tuple(int(x) for x in np.load(str(s_path)))

    # Log metadata if present
    meta_p = save_dir / CALIB_META_FILE
    if meta_p.exists():
        meta = np.load(str(meta_p), allow_pickle=True)
        model_type = str(meta["model_type"][0]) if "model_type" in meta else "?"
        ts         = str(meta["timestamp"][0])  if "timestamp"  in meta else "?"
        log.info("Calibration loaded: model=%s  time=%s  matrix=%s  screen=%dx%d",
                 model_type, ts, matrix.shape, screen_wh[0], screen_wh[1])
    else:
        log.info("Calibration loaded: matrix=%s  screen=%dx%d",
                 matrix.shape, screen_wh[0], screen_wh[1])
    return matrix, screen_wh


# ─────────────────────────────────────────────────────────────────────────────
# Screen resolution
# ─────────────────────────────────────────────────────────────────────────────

def get_screen_resolution() -> tuple[int, int]:
    """Return (width, height) of the primary monitor."""
    try:
        user32 = ctypes.windll.user32
        w = int(user32.GetSystemMetrics(0))
        h = int(user32.GetSystemMetrics(1))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    log.warning("Could not query screen resolution; defaulting to 1920x1080.")
    return 1920, 1080


# ─────────────────────────────────────────────────────────────────────────────
# Interactive calibration routine
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(
    screen_w: int,
    screen_h: int,
    cap: cv2.VideoCapture,
    gaze_runner,        # ModelRunner
    eye_detector,       # EyeDetector
    n_points: int = 9,
    calib_model: str = "affine",
) -> Optional[np.ndarray]:
    """Run the interactive gaze calibration routine.

    Upgrade 10 additions:
    - 13-point support.
    - Per-point quality scoring with one automatic retry for low-quality points.
    - Polynomial regression option.

    Args:
        screen_w:    Canvas width in pixels.
        screen_h:    Canvas height in pixels.
        cap:         Open cv2.VideoCapture.
        gaze_runner: ModelRunner instance.
        eye_detector: EyeDetector instance.
        n_points:    9 or 13.
        calib_model: "affine" or "polynomial".

    Returns:
        (2,3) or (2,6) numpy matrix, or None if calibration failed.
    """
    points_norm = CALIB_POINTS_13[:n_points] if n_points == 13 else CALIB_POINTS_9
    n_total     = len(points_norm)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    completed:  list[_CalibPoint] = []
    bad_points: set[int]          = set()

    def _collect_point(pt_idx: int, nx: float, ny: float, retry: bool = False) -> Optional[_CalibPoint]:
        """Collect gaze samples for one calibration target. Returns _CalibPoint or None."""
        target_x = int(nx * screen_w)
        target_y = int(ny * screen_h)
        label    = f"(retry)" if retry else ""

        # Phase 1: fixation delay
        phase_start = time.monotonic()
        while time.monotonic() - phase_start < FIXATION_DELAY_S:
            bg = _make_bg(screen_w, screen_h)
            _draw_completed_dots(bg, completed, bad_points, screen_w, screen_h)
            cv2.circle(bg, (target_x, target_y), DOT_RADIUS_IDLE, COLOR_IDLE, -1)
            _draw_progress(bg, pt_idx, n_total, screen_w, screen_h,
                           collecting=False, label=label)
            cv2.imshow(WINDOW_NAME, bg)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                return None

        # Phase 2: collection
        gaze_cx_list: list[float] = []
        gaze_cy_list: list[float] = []
        collect_start = time.monotonic()

        while time.monotonic() - collect_start < COLLECT_DURATION_S:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            det = eye_detector.process(frame)
            if det.face_detected and det.left_eye_crop is not None:
                preds = gaze_runner.run(
                    eye_crop_bgr  = det.left_eye_crop,
                    emotion_feats = None,
                    cogload_feats = None,
                )
                if preds.gaze_cx is not None:
                    gaze_cx_list.append(preds.gaze_cx)
                    gaze_cy_list.append(preds.gaze_cy)

            bg = _make_bg(screen_w, screen_h)
            _draw_completed_dots(bg, completed, bad_points, screen_w, screen_h)
            cv2.circle(bg, (target_x, target_y), DOT_RADIUS_ACTIVE, COLOR_ACTIVE, -1)
            remaining = COLLECT_DURATION_S - (time.monotonic() - collect_start)
            _draw_progress(bg, pt_idx, n_total, screen_w, screen_h,
                           collecting=True, remaining=remaining,
                           n_samples=len(gaze_cx_list), label=label)
            cv2.imshow(WINDOW_NAME, bg)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                return None

        n_ok = len(gaze_cx_list)
        if n_ok < MIN_VALID_SAMPLES:
            log.warning("Point %d/%d: only %d samples (need %d) -- skipping.",
                        pt_idx + 1, n_total, n_ok, MIN_VALID_SAMPLES)
            return None

        var_cx  = float(np.var(gaze_cx_list))
        var_cy  = float(np.var(gaze_cy_list))
        quality = var_cx + var_cy

        cp = _CalibPoint(
            screen_x_px  = float(target_x),
            screen_y_px  = float(target_y),
            gaze_cx_mean = float(np.mean(gaze_cx_list)),
            gaze_cy_mean = float(np.mean(gaze_cy_list)),
            n_samples    = n_ok,
            quality      = quality,
        )

        if quality > QUALITY_VAR_THRESH:
            log.warning("Point %d/%d: LOW QUALITY (var=%.4f). %s",
                        pt_idx + 1, n_total, quality,
                        "Re-collecting." if not retry else "Keeping anyway.")
        else:
            log.info("Point %d/%d  screen=(%d,%d)  gaze=(%.3f,%.3f)  n=%d  var=%.4f",
                     pt_idx + 1, n_total, target_x, target_y,
                     cp.gaze_cx_mean, cp.gaze_cy_mean, n_ok, quality)
        return cp

    try:
        for pt_idx, (nx, ny) in enumerate(points_norm):
            cp = _collect_point(pt_idx, nx, ny, retry=False)
            if cp is None:
                return None   # user aborted

            if cp.quality > QUALITY_VAR_THRESH:
                bad_points.add(pt_idx)
                # One automatic retry
                cp_retry = _collect_point(pt_idx, nx, ny, retry=True)
                if cp_retry is not None:
                    cp = cp_retry
                    if cp.quality <= QUALITY_VAR_THRESH:
                        bad_points.discard(pt_idx)

            completed.append(cp)

    finally:
        cv2.destroyWindow(WINDOW_NAME)

    # ── Fit transform ─────────────────────────────────────────────────────────
    if len(completed) < 4:
        log.error("Only %d/%d valid points. Need >= 4. Run calibration again.",
                  len(completed), n_total)
        return None

    gaze_arr   = np.array([[c.gaze_cx_mean, c.gaze_cy_mean] for c in completed])
    screen_arr = np.array([[c.screen_x_px,  c.screen_y_px]  for c in completed])
    quality_scores = [c.quality for c in completed]

    try:
        if calib_model == "polynomial" and len(completed) >= 6:
            M = fit_polynomial_transform(gaze_arr, screen_arr)
            log.info("Polynomial degree-2 transform fitted (%d points).", len(completed))
        else:
            if calib_model == "polynomial":
                log.warning("Not enough points for polynomial (%d < 6); "
                            "falling back to affine.", len(completed))
            M = fit_affine_transform(gaze_arr, screen_arr)
            calib_model = "affine"
    except ValueError as exc:
        log.error("Transform fit failed: %s", exc)
        return None

    # Residuals
    residuals = []
    for c, gp in zip(completed, gaze_arr):
        px, py = apply_calibration(gp[0], gp[1], M)
        err = float(np.sqrt((px - c.screen_x_px)**2 + (py - c.screen_y_px)**2))
        residuals.append(err)

    log.info("Calibration fit: %d pts  mean_err=%.1f px  max_err=%.1f px  "
             "bad_pts=%d  model=%s",
             len(completed), float(np.mean(residuals)), float(np.max(residuals)),
             len(bad_points), calib_model)

    # Store quality & model_type on the matrix object (carried to save_calibration)
    M._calib_quality    = quality_scores   # type: ignore[attr-defined]
    M._calib_model_type = calib_model      # type: ignore[attr-defined]
    return M


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_bg(w: int, h: int) -> np.ndarray:
    bg = np.full((h, w, 3), COLOR_BG, dtype=np.uint8)
    cv2.putText(
        bg,
        "Focus on each dot. Press Q to abort.",
        (w // 2 - 250, 30), FONT, FONT_SCALE, (180, 180, 180), FONT_THICK, cv2.LINE_AA,
    )
    return bg


def _draw_completed_dots(
    bg: np.ndarray,
    completed: list[_CalibPoint],
    bad_points: set[int],
    screen_w: int,
    screen_h: int,
) -> None:
    for i, cp in enumerate(completed):
        col = COLOR_BAD if i in bad_points else COLOR_DONE
        cv2.circle(bg,
                   (int(cp.screen_x_px), int(cp.screen_y_px)),
                   DOT_RADIUS_DONE, col, -1)


def _draw_progress(
    bg: np.ndarray,
    pt_idx: int,
    n_total: int,
    screen_w: int,
    screen_h: int,
    collecting: bool,
    remaining: float = 0.0,
    n_samples: int = 0,
    label: str = "",
) -> None:
    h = bg.shape[0]
    if collecting:
        status = f"Collecting... {remaining:.1f}s  ({n_samples} samples) {label}"
    else:
        status = f"Look at the dot  ({pt_idx + 1}/{n_total}) {label}"
    cv2.putText(
        bg, status, (20, h - 20),
        FONT, FONT_SCALE, COLOR_ACTIVE if collecting else COLOR_IDLE,
        FONT_THICK, cv2.LINE_AA,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run gaze calibration and save the transform to "
            "saved_models/calibration.npy."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=Path, required=True,
                        help="Project root directory.")
    parser.add_argument("--cam_index", type=int, default=0,
                        help="OpenCV VideoCapture device index.")
    parser.add_argument("--calibration_points", type=int, default=9,
                        choices=[9, 13],
                        help="Number of calibration targets (9 or 13).")
    parser.add_argument("--calibration_model", type=str, default="affine",
                        choices=["affine", "polynomial"],
                        help="Transform model: affine (2x3) or polynomial degree-2 (2x6).")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the interactive gaze calibration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    _infer_dir = Path(__file__).parent / "inference"
    sys.path.insert(0, str(_infer_dir))
    from eye_detector import EyeDetector
    from model_runner  import ModelRunner

    parser    = _build_arg_parser()
    args      = parser.parse_args(argv)
    data_path = args.data_path.resolve()
    model_dir = data_path / "saved_models"

    if not data_path.exists():
        log.error("data_path does not exist: %s", data_path)
        sys.exit(1)

    try:
        detector = EyeDetector.from_model_dir(model_dir)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    runner = ModelRunner(model_dir)
    if not runner._loaded.get("gaze", False):
        log.error("Gaze model not loaded. Train it first.")
        detector.close()
        sys.exit(1)

    # Open webcam
    cap = None
    for backend, bname in [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF,  "MSMF"),
        (cv2.CAP_ANY,   "ANY"),
    ]:
        c = cv2.VideoCapture(args.cam_index, backend)
        if c.isOpened():
            cap = c
            log.info("Webcam opened (index=%d, backend=%s)", args.cam_index, bname)
            break
        c.release()
    if cap is None or not cap.isOpened():
        log.error("Cannot open webcam at index %d.", args.cam_index)
        detector.close()
        sys.exit(1)

    screen_w, screen_h = get_screen_resolution()
    log.info("Screen resolution: %dx%d", screen_w, screen_h)
    log.info("Starting %d-point gaze calibration (model=%s).",
             args.calibration_points, args.calibration_model)
    log.info("Sit comfortably, look at each dot when it appears. Press Q to abort.")

    try:
        M = run_calibration(
            screen_w, screen_h, cap, runner, detector,
            n_points    = args.calibration_points,
            calib_model = args.calibration_model,
        )
    finally:
        cap.release()
        detector.close()

    if M is None:
        log.error("Calibration failed.")
        sys.exit(1)

    quality_scores = getattr(M, "_calib_quality",    None)
    model_type     = getattr(M, "_calib_model_type", args.calibration_model)

    save_calibration(
        M, (screen_w, screen_h), model_dir,
        quality_scores = quality_scores,
        model_type     = model_type,
    )
    log.info("Calibration complete. Run main.py to start the tracking session.")


if __name__ == "__main__":
    main()
