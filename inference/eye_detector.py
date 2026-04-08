"""inference/eye_detector.py
--------------------------------------------------------------------------------
Face landmark detection and eye-region extraction for the real-time pipeline.

Upgrades applied (v2):
  3 - CLAHE preprocessing, iris-landmark pupil centre, head-pose gating,
      no-face warning counter, confidence gating (head_turned flag)
  6 - Eyelid spline curves, iris circles, pupil dots, gaze-direction arrows,
      attention-state-coloured eye bounding box

MediaPipe model required:
  saved_models/face_landmarker.task  (~29 MB, downloaded once)

  Download:
      python inference/eye_detector.py --download_model \\
          --data_path "C:/Users/kunal/Desktop/Eye tracking Module"

Standalone live test:
    python inference/eye_detector.py \\
        --data_path "C:/Users/kunal/Desktop/Eye tracking Module"
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("eye_detector")

# ── Model artefact ────────────────────────────────────────────────────────────
MODEL_FILENAME      = "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ── Eye landmark indices (MediaPipe 478-point face mesh) ──────────────────────
LEFT_EYE_EAR_IDX:  list[int] = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX: list[int] = [263, 387, 385, 362, 380, 374]

LEFT_EYE_CONTOUR:  list[int] = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_CONTOUR: list[int] = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]

# ── Iris landmark indices (indices 468-477, iris-refinement model) ────────────
LEFT_IRIS:  list[int] = [468, 469, 470, 471, 472]
RIGHT_IRIS: list[int] = [473, 474, 475, 476, 477]

# ── Eyelid spline points (Upgrade 6) ─────────────────────────────────────────
LEFT_UPPER_LID:  list[int] = [33, 246, 161, 160, 159, 158, 157, 173, 133]
LEFT_LOWER_LID:  list[int] = [133, 155, 154, 153, 145, 144, 163, 7, 33]
RIGHT_UPPER_LID: list[int] = [263, 466, 388, 387, 386, 385, 384, 398, 362]
RIGHT_LOWER_LID: list[int] = [362, 382, 381, 380, 374, 373, 390, 249, 263]

# ── Head-pose 3-D reference points (approximate adult face, in mm) ────────────
# Index order matches HEAD_POSE_LM_IDX below.
_HEAD_3D = np.array([
    (  0.0,    0.0,    0.0),   # nose tip          - lm 1
    (  0.0, -130.0, -100.0),   # chin              - lm 152
    (-165.0,  170.0, -135.0),  # left eye corner   - lm 263
    ( 165.0,  170.0, -135.0),  # right eye corner  - lm 33
    (-150.0, -150.0, -125.0),  # left mouth corner - lm 287
    ( 150.0, -150.0, -125.0),  # right mouth corner- lm 57
], dtype=np.float64)
HEAD_POSE_LM_IDX: list[int] = [1, 152, 263, 33, 287, 57]
HEAD_TURN_THRESH_DEG: float  = 25.0   # yaw or pitch beyond this = head turned

# ── CLAHE settings (Upgrade 3) ────────────────────────────────────────────────
_CLAHE_CLIP    = 2.0
_CLAHE_TILE    = (8, 8)

# ── No-face warning threshold ─────────────────────────────────────────────────
NO_FACE_WARNING_FRAMES = 10   # show "Face not detected" after this many frames

# ── Detection thresholds ──────────────────────────────────────────────────────
DEFAULT_MIN_FACE_CONF     = 0.5
DEFAULT_MIN_PRESENCE_CONF = 0.5
DEFAULT_MIN_TRACK_CONF    = 0.5
CONFIDENCE_GATE_THRESH    = 0.65   # landmark confidence proxy (head-pose based)

# ── Output geometry ───────────────────────────────────────────────────────────
DEFAULT_EYE_CROP_SIZE = 64
DEFAULT_EYE_PAD_FRAC  = 0.20

# ── Visualisation colours (BGR) ───────────────────────────────────────────────
DRAW_DOT_RADIUS   = 2
DRAW_DOT_THICK    = -1
COLOR_LEFT_EYE    = (0, 220, 0)
COLOR_RIGHT_EYE   = (220, 0, 0)
COLOR_EAR_PT      = (0, 200, 220)
COLOR_IRIS        = (0, 180, 255)
COLOR_PUPIL       = (0, 255, 0)
COLOR_EYELID      = (0, 255, 200)
COLOR_GAZE_ARROW  = (0, 200, 255)

ATTENTION_BOX_COLORS: dict[str, tuple[int, int, int]] = {
    "focused":    (0,  210,  40),
    "distracted": (0,  210, 210),
    "off_task":   (50,  50, 220),
}
DEFAULT_BOX_COLOR = (140, 140, 140)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class EyeDetectionResult:
    """All per-frame outputs produced by EyeDetector.process().

    Original fields:
        face_detected, annotated_frame, landmarks_px,
        left_ear, right_ear, mean_ear, left_eye_crop, right_eye_crop

    New fields (Upgrade 3 + 6):
        left_pupil_center, right_pupil_center,
        iris_diameter_px, inter_ocular_dist_px,
        head_yaw_deg, head_pitch_deg, head_turned,
        no_face_frames, detection_confident
    """
    # ── Original fields ───────────────────────────────────────────────────────
    face_detected:   bool
    annotated_frame: np.ndarray
    landmarks_px:    Optional[list[tuple[int, int]]] = None
    left_ear:        Optional[float]                  = None
    right_ear:       Optional[float]                  = None
    mean_ear:        Optional[float]                  = None
    left_eye_crop:   Optional[np.ndarray]             = None
    right_eye_crop:  Optional[np.ndarray]             = None

    # ── New fields ────────────────────────────────────────────────────────────
    left_pupil_center:    Optional[tuple[int, int]] = None
    right_pupil_center:   Optional[tuple[int, int]] = None
    iris_diameter_px:     Optional[float]           = None
    inter_ocular_dist_px: Optional[float]           = None
    head_yaw_deg:         Optional[float]           = None
    head_pitch_deg:       Optional[float]           = None
    head_turned:          bool                      = False
    no_face_frames:       int                       = 0
    detection_confident:  bool                      = True   # False if head_turned


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_ear(
    landmarks_px: list[tuple[int, int]],
    eye_idx: list[int],
) -> float:
    """Compute Eye Aspect Ratio (Soukupova & Cech 2016).

    Returns EAR float. ~0.25-0.35 open eye, <0.20 blink.
    """
    pts   = [np.array(landmarks_px[i], dtype=np.float32) for i in eye_idx]
    horiz = np.linalg.norm(pts[0] - pts[3])
    vert1 = np.linalg.norm(pts[1] - pts[5])
    vert2 = np.linalg.norm(pts[2] - pts[4])
    return float((vert1 + vert2) / (2.0 * horiz + 1e-8))


def _pupil_center_from_iris(
    landmarks_px: list[tuple[int, int]],
    iris_idx: list[int],
) -> Optional[tuple[int, int]]:
    """Compute pupil centre as the mean of iris landmark coordinates."""
    if len(landmarks_px) < max(iris_idx) + 1:
        return None
    xs = [landmarks_px[i][0] for i in iris_idx]
    ys = [landmarks_px[i][1] for i in iris_idx]
    return int(round(float(np.mean(xs)))), int(round(float(np.mean(ys))))


def _iris_diameter(
    landmarks_px: list[tuple[int, int]],
    iris_idx: list[int],
) -> Optional[float]:
    """Estimate iris diameter from leftmost to rightmost iris landmark."""
    if len(landmarks_px) < max(iris_idx) + 1:
        return None
    pts = [np.array(landmarks_px[i], dtype=np.float32) for i in iris_idx]
    # Diameter = max pairwise distance among the 5 iris points
    diam = 0.0
    for a in pts:
        for b in pts:
            d = float(np.linalg.norm(a - b))
            if d > diam:
                diam = d
    return diam


def crop_eye(
    bgr_frame: np.ndarray,
    landmarks_px: list[tuple[int, int]],
    eye_contour_idx: list[int],
    out_size: int = DEFAULT_EYE_CROP_SIZE,
    pad_frac: float = DEFAULT_EYE_PAD_FRAC,
) -> Optional[np.ndarray]:
    """Extract a padded, resized eye crop from a BGR frame."""
    H, W  = bgr_frame.shape[:2]
    pts   = [landmarks_px[i] for i in eye_contour_idx]
    xs    = [p[0] for p in pts]
    ys    = [p[1] for p in pts]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pad_x = max(1, int((x_max - x_min) * pad_frac))
    pad_y = max(1, int((y_max - y_min) * pad_frac))

    x1 = max(0, x_min - pad_x)
    x2 = min(W, x_max + pad_x)
    y1 = max(0, y_min - pad_y)
    y2 = min(H, y_max + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = bgr_frame[y1:y2, x1:x2]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def _estimate_head_pose(
    landmarks_px: list[tuple[int, int]],
    frame_w: int,
    frame_h: int,
) -> tuple[Optional[float], Optional[float]]:
    """Estimate head yaw and pitch angles using solvePnP.

    Returns:
        (yaw_deg, pitch_deg) or (None, None) if solvePnP fails.
    """
    img_pts = np.array(
        [landmarks_px[i] for i in HEAD_POSE_LM_IDX],
        dtype=np.float64,
    )

    focal  = float(frame_w)
    cx     = frame_w / 2.0
    cy     = frame_h / 2.0
    cam_mat = np.array(
        [[focal,   0.0, cx],
         [  0.0, focal, cy],
         [  0.0,   0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coef = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        _HEAD_3D, img_pts, cam_mat, dist_coef,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None

    # Convert rotation vector to rotation matrix and extract Euler angles
    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose: pitch = rotation around X, yaw = rotation around Y
    pitch_rad = float(np.arctan2(-rmat[2, 1], rmat[2, 2]))
    yaw_rad   = float(np.arctan2( rmat[2, 0],
                                   np.sqrt(rmat[2, 1]**2 + rmat[2, 2]**2)))
    return float(np.degrees(yaw_rad)), float(np.degrees(pitch_rad))


# ─────────────────────────────────────────────────────────────────────────────
# Upgraded visualisation (Upgrade 6)
# ─────────────────────────────────────────────────────────────────────────────

def draw_eye_overlay(
    bgr_frame: np.ndarray,
    landmarks_px: list[tuple[int, int]],
    attention_name: Optional[str] = None,
    gaze_cx: Optional[float] = None,
    gaze_cy: Optional[float] = None,
    left_pupil_center: Optional[tuple[int, int]] = None,
    right_pupil_center: Optional[tuple[int, int]] = None,
    iris_diameter_px: float = 20.0,
) -> np.ndarray:
    """Draw eyelid curves, iris circles, pupil dots, gaze arrows, eye boxes.

    Upgrade 6 implementation:
    - Eyelid splines via cv2.polylines
    - Iris circles at detected centre + radius
    - Pupil dot (green filled circle)
    - Gaze direction arrow from pupil centre (when gaze_cx/cy available)
    - Attention-state-coloured bounding box around each eye region

    Args:
        bgr_frame:         Source BGR image.
        landmarks_px:      Full 478+ landmark list.
        attention_name:    "focused"/"distracted"/"off_task" for box colour.
        gaze_cx:           Normalised gaze X in [0, 1], or None.
        gaze_cy:           Normalised gaze Y in [0, 1], or None.
        left_pupil_center: Pixel coords of left pupil, or None.
        right_pupil_center: Pixel coords of right pupil, or None.
        iris_diameter_px:  Estimate of iris diameter in pixels.

    Returns:
        New BGR image with overlay drawn.
    """
    out      = bgr_frame.copy()
    box_col  = ATTENTION_BOX_COLORS.get(attention_name, DEFAULT_BOX_COLOR) if attention_name else DEFAULT_BOX_COLOR
    iris_r   = max(4, int(iris_diameter_px / 2))

    # ── Eyelid curves ─────────────────────────────────────────────────────────
    for lid_pts in (LEFT_UPPER_LID, LEFT_LOWER_LID):
        pts = np.array([landmarks_px[i] for i in lid_pts], dtype=np.int32)
        cv2.polylines(out, [pts], False, COLOR_EYELID, 1, cv2.LINE_AA)
    for lid_pts in (RIGHT_UPPER_LID, RIGHT_LOWER_LID):
        pts = np.array([landmarks_px[i] for i in lid_pts], dtype=np.int32)
        cv2.polylines(out, [pts], False, COLOR_EYELID, 1, cv2.LINE_AA)

    # ── Eye bounding boxes (attention-coloured) ────────────────────────────────
    H, W = out.shape[:2]
    for contour in (LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR):
        xs = [landmarks_px[i][0] for i in contour]
        ys = [landmarks_px[i][1] for i in contour]
        pad = 6
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(W, max(xs) + pad)
        y2 = min(H, max(ys) + pad)
        cv2.rectangle(out, (x1, y1), (x2, y2), box_col, 1)

    # ── Iris circles + pupil dots ──────────────────────────────────────────────
    for pupil, iris_pts in (
        (left_pupil_center,  LEFT_IRIS),
        (right_pupil_center, RIGHT_IRIS),
    ):
        if pupil is None:
            continue
        # Iris circle
        cv2.circle(out, pupil, iris_r, COLOR_IRIS, 1, cv2.LINE_AA)
        # Pupil dot
        cv2.circle(out, pupil, 3, COLOR_PUPIL, -1, cv2.LINE_AA)

        # Gaze-direction arrow (from pupil centre in direction of gaze offset)
        if gaze_cx is not None and gaze_cy is not None:
            dx = gaze_cx - 0.5
            dy = gaze_cy - 0.5
            length = np.sqrt(dx * dx + dy * dy + 1e-8)
            arrow_len = 20
            tip_x = int(pupil[0] + (dx / length) * arrow_len)
            tip_y = int(pupil[1] + (dy / length) * arrow_len)
            cv2.arrowedLine(
                out, pupil, (tip_x, tip_y),
                COLOR_GAZE_ARROW, 2, cv2.LINE_AA, tipLength=0.3,
            )

    return out


def draw_eye_landmarks(
    bgr_frame: np.ndarray,
    landmarks_px: list[tuple[int, int]],
) -> np.ndarray:
    """Legacy dot-based eye landmark overlay (kept for backward compatibility).

    Calls draw_eye_overlay with no extras for a minimal visualization.
    """
    return draw_eye_overlay(bgr_frame, landmarks_px)


# ─────────────────────────────────────────────────────────────────────────────
# Model download helper
# ─────────────────────────────────────────────────────────────────────────────

def download_face_landmarker_model(
    save_dir: Path,
    force: bool = False,
) -> Path:
    """Download face_landmarker.task (~29 MB) if not already present."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / MODEL_FILENAME

    if model_path.exists() and not force:
        log.info("Face landmarker model already present: %s", model_path)
        return model_path

    log.info("Downloading face_landmarker.task (~29 MB) ...")
    log.info("  URL         : %s", FACE_LANDMARKER_URL)
    log.info("  Destination : %s", model_path)

    def _reporthook(block_count: int, block_size: int, total_size: int) -> None:
        if total_size > 0 and block_count % 100 == 0:
            pct = min(100, int(block_count * block_size * 100 / total_size))
            log.info("  Downloading ... %d%%", pct)

    try:
        urllib.request.urlretrieve(
            FACE_LANDMARKER_URL, str(model_path), _reporthook,
        )
        size_mb = model_path.stat().st_size / 1_048_576
        log.info("Download complete: %s  (%.1f MB)", model_path, size_mb)
    except Exception as exc:
        if model_path.exists():
            model_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download face_landmarker.task: {exc}\n"
            "Please download it manually and place at: " + str(model_path)
        ) from exc

    return model_path


# ─────────────────────────────────────────────────────────────────────────────
# EyeDetector
# ─────────────────────────────────────────────────────────────────────────────

class EyeDetector:
    """MediaPipe FaceLandmarker-based eye detector.

    Upgrade 3 additions:
    - CLAHE preprocessing before detection
    - Iris pupil-centre extraction (landmarks 468-477)
    - Head-pose estimation via solvePnP
    - Consecutive no-face frame counter
    - detection_confident flag when head is turned > HEAD_TURN_THRESH_DEG

    Upgrade 6: rich overlay via draw_eye_overlay().
    """

    def __init__(
        self,
        model_path: Path,
        num_faces: int = 1,
        eye_crop_size: int = DEFAULT_EYE_CROP_SIZE,
        min_face_conf: float = DEFAULT_MIN_FACE_CONF,
        min_presence_conf: float = DEFAULT_MIN_PRESENCE_CONF,
        min_track_conf: float = DEFAULT_MIN_TRACK_CONF,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"FaceLandmarker model not found: {model_path}\n"
                "Download: python inference/eye_detector.py --download_model "
                "--data_path <project_root>"
            )

        base_opts = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options   = mp.tasks.vision.FaceLandmarkerOptions(
            base_options                          = base_opts,
            num_faces                             = num_faces,
            min_face_detection_confidence         = min_face_conf,
            min_face_presence_confidence          = min_presence_conf,
            min_tracking_confidence               = min_track_conf,
            output_face_blendshapes               = False,
            output_facial_transformation_matrixes = False,
        )
        self._landmarker    = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._eye_crop_size = eye_crop_size
        self._clahe         = cv2.createCLAHE(
            clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_TILE,
        )
        self._no_face_count: int = 0
        log.info("EyeDetector ready (model=%s)", model_path.name)

    @classmethod
    def from_model_dir(cls, model_dir: Path, **kwargs) -> "EyeDetector":
        return cls(model_dir / MODEL_FILENAME, **kwargs)

    def process(
        self,
        bgr_frame: np.ndarray,
        attention_name: Optional[str] = None,
        gaze_cx: Optional[float] = None,
        gaze_cy: Optional[float] = None,
    ) -> EyeDetectionResult:
        """Run landmark detection, EAR, iris extraction, head pose.

        Args:
            bgr_frame:      Camera frame in BGR format.
            attention_name: Current attention label for overlay box colour.
            gaze_cx:        Normalised gaze X (used for gaze arrow overlay).
            gaze_cy:        Normalised gaze Y.

        Returns:
            EyeDetectionResult with all fields populated.
        """
        H, W = bgr_frame.shape[:2]

        # ── CLAHE enhancement before MediaPipe detection ──────────────────────
        gray     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        enh_bgr  = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        rgb_enh  = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_enh)
        result   = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            self._no_face_count += 1
            return EyeDetectionResult(
                face_detected   = False,
                annotated_frame = bgr_frame.copy(),
                no_face_frames  = self._no_face_count,
            )

        self._no_face_count = 0

        # Convert first face's normalised landmarks to pixel coordinates
        lms    = result.face_landmarks[0]
        lms_px: list[tuple[int, int]] = [
            (int(lm.x * W), int(lm.y * H)) for lm in lms
        ]

        # ── EAR ───────────────────────────────────────────────────────────────
        left_ear  = compute_ear(lms_px, LEFT_EYE_EAR_IDX)
        right_ear = compute_ear(lms_px, RIGHT_EYE_EAR_IDX)
        mean_ear  = (left_ear + right_ear) / 2.0

        # ── Eye crops (use original frame, not enhanced) ───────────────────────
        left_crop  = crop_eye(bgr_frame, lms_px, LEFT_EYE_CONTOUR,  self._eye_crop_size)
        right_crop = crop_eye(bgr_frame, lms_px, RIGHT_EYE_CONTOUR, self._eye_crop_size)

        # ── Iris / pupil centres (iris landmarks present when model >= 478 pts)
        left_pupil  = _pupil_center_from_iris(lms_px, LEFT_IRIS)
        right_pupil = _pupil_center_from_iris(lms_px, RIGHT_IRIS)

        # Iris diameter (mean of left and right)
        l_diam = _iris_diameter(lms_px, LEFT_IRIS)
        r_diam = _iris_diameter(lms_px, RIGHT_IRIS)
        if l_diam and r_diam:
            iris_diam = (l_diam + r_diam) / 2.0
        else:
            iris_diam = l_diam or r_diam

        # Inter-ocular distance
        iod: Optional[float] = None
        if left_pupil and right_pupil:
            iod = float(np.linalg.norm(
                np.array(left_pupil) - np.array(right_pupil)
            ))

        # ── Head pose ─────────────────────────────────────────────────────────
        yaw_deg, pitch_deg = _estimate_head_pose(lms_px, W, H)
        head_turned = False
        if yaw_deg is not None and pitch_deg is not None:
            head_turned = (abs(yaw_deg) > HEAD_TURN_THRESH_DEG
                           or abs(pitch_deg) > HEAD_TURN_THRESH_DEG)

        # ── Rich overlay (Upgrade 6) ────────────────────────────────────────
        annotated = draw_eye_overlay(
            bgr_frame, lms_px,
            attention_name    = attention_name,
            gaze_cx           = gaze_cx,
            gaze_cy           = gaze_cy,
            left_pupil_center = left_pupil,
            right_pupil_center = right_pupil,
            iris_diameter_px  = float(iris_diam) if iris_diam else 20.0,
        )

        return EyeDetectionResult(
            face_detected        = True,
            annotated_frame      = annotated,
            landmarks_px         = lms_px,
            left_ear             = left_ear,
            right_ear            = right_ear,
            mean_ear             = mean_ear,
            left_eye_crop        = left_crop,
            right_eye_crop       = right_crop,
            left_pupil_center    = left_pupil,
            right_pupil_center   = right_pupil,
            iris_diameter_px     = float(iris_diam) if iris_diam else None,
            inter_ocular_dist_px = iod,
            head_yaw_deg         = yaw_deg,
            head_pitch_deg       = pitch_deg,
            head_turned          = head_turned,
            no_face_frames       = 0,
            detection_confident  = not head_turned,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Eye landmark detection and eye-crop extraction using MediaPipe "
            "FaceLandmarker. Use --download_model to fetch the model first."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=Path, required=True,
                        help="Project root directory.")
    parser.add_argument("--download_model", action="store_true",
                        help="Download face_landmarker.task to saved_models/.")
    parser.add_argument("--cam_index", type=int, default=0,
                        help="OpenCV VideoCapture device index.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Download model and/or run live eye-detection preview."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser    = _build_arg_parser()
    args      = parser.parse_args(argv)
    data_path = args.data_path.resolve()
    model_dir = data_path / "saved_models"

    if not data_path.exists():
        log.error("data_path does not exist: %s", data_path)
        sys.exit(1)

    if args.download_model:
        try:
            download_face_landmarker_model(model_dir)
        except RuntimeError as exc:
            log.error("%s", exc)
            sys.exit(1)

    try:
        detector = EyeDetector.from_model_dir(model_dir)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    # Try multiple backends for Windows
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

    log.info("Live eye detection running. Press Q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            result = detector.process(frame)
            vis    = result.annotated_frame

            if result.face_detected:
                ear_str = (
                    f"EAR L={result.left_ear:.3f}"
                    f"  R={result.right_ear:.3f}"
                    f"  mean={result.mean_ear:.3f}"
                )
                if result.head_yaw_deg is not None:
                    ear_str += f"  yaw={result.head_yaw_deg:.1f}"
                cv2.putText(vis, ear_str, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 220, 0), 1, cv2.LINE_AA)
                if result.left_eye_crop is not None:
                    thumb = cv2.resize(result.left_eye_crop, (96, 96))
                    h, w  = vis.shape[:2]
                    vis[8:104, w - 104:w - 8] = thumb
            else:
                msg = "No face detected" if result.no_face_frames < NO_FACE_WARNING_FRAMES else "FACE LOST"
                cv2.putText(vis, msg, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 220), 2, cv2.LINE_AA)

            cv2.imshow("Eye Detector", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
