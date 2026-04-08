"""collect_webcam_samples.py
--------------------------------------------------------------------------------
Interactive tool to collect labelled webcam frames for fine-tuning the
attention model (train_attention.py --webcam_finetune_only).

Controls (press key while the preview window is focused):
  f  - toggle recording as FOCUSED
  d  - toggle recording as DISTRACTED
  o  - toggle recording as OFF_TASK
  q  - quit

Frames are saved as JPEG to:
  <data_path>/data/webcam_finetune/focused/frame_NNNNN.jpg
  <data_path>/data/webcam_finetune/distracted/frame_NNNNN.jpg
  <data_path>/data/webcam_finetune/off_task/frame_NNNNN.jpg

Re-running the script RESUMES from the existing count (never overwrites).

Run:
    python tools/collect_webcam_samples.py \\
        --data_path "C:/Users/kunal/Desktop/Eye tracking Module"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect_webcam")

# ── Label definitions ─────────────────────────────────────────────────────────
LABEL_FOCUSED    = "focused"
LABEL_DISTRACTED = "distracted"
LABEL_OFF_TASK   = "off_task"
LABELS: list[str] = [LABEL_FOCUSED, LABEL_DISTRACTED, LABEL_OFF_TASK]

# Maps a keycode to a label (upper- and lower-case both accepted)
LABEL_KEY_MAP: dict[int, str] = {
    ord("f"): LABEL_FOCUSED,
    ord("d"): LABEL_DISTRACTED,
    ord("o"): LABEL_OFF_TASK,
    ord("F"): LABEL_FOCUSED,
    ord("D"): LABEL_DISTRACTED,
    ord("O"): LABEL_OFF_TASK,
}
QUIT_KEYS: frozenset[int] = frozenset({ord("q"), ord("Q")})

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TARGET      = 200
DEFAULT_CAM_INDEX   = 0
DEFAULT_JPG_QUALITY = 90

WEBCAM_FINETUNE_SUBPATH: tuple[str, str] = ("data", "webcam_finetune")

# ── Face-detection settings ───────────────────────────────────────────────────
# Uses OpenCV's bundled Haar cascade (no model download needed).
HAAR_SCALE_FACTOR  = 1.1
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE      = (60, 60)
FACE_CHECK_EVERY_N = 10    # run detection only once per N frames

# ── Overlay appearance ────────────────────────────────────────────────────────
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.65
FONT_THICK  = 2
COLOR_GREEN  = (0, 220, 0)
COLOR_YELLOW = (0, 220, 220)
COLOR_RED    = (0,   0, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (160, 160, 160)
OVERLAY_BG   = (20,  20,  20)
OVERLAY_ALPHA = 0.55

LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    LABEL_FOCUSED:    COLOR_GREEN,
    LABEL_DISTRACTED: COLOR_YELLOW,
    LABEL_OFF_TASK:   COLOR_RED,
}


# ─────────────────────────────────────────────────────────────────────────────
# Face detector wrapper
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """Face-presence detector backed by OpenCV's bundled Haar cascade.

    Uses :data:`cv2.data.haarcascades` + ``haarcascade_frontalface_default.xml``
    which ships with every opencv-python install — no model download required.
    Detection is throttled to every :data:`FACE_CHECK_EVERY_N` frames to keep
    the capture loop fast.
    """

    def __init__(self) -> None:
        """Load the Haar cascade from OpenCV's bundled data directory."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._classifier = cv2.CascadeClassifier(cascade_path)

    def face_present(self, bgr_frame: np.ndarray) -> bool:
        """Return True if at least one face is detected in the frame.

        Args:
            bgr_frame: BGR image from OpenCV VideoCapture.

        Returns:
            True if one or more faces are detected, False otherwise.
        """
        gray  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        faces = self._classifier.detectMultiScale(
            gray,
            scaleFactor  = HAAR_SCALE_FACTOR,
            minNeighbors = HAAR_MIN_NEIGHBORS,
            minSize      = HAAR_MIN_SIZE,
        )
        return len(faces) > 0

    def close(self) -> None:
        """No-op — Haar classifier holds no external resources."""


# ─────────────────────────────────────────────────────────────────────────────
# Overlay drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    counts: dict[str, int],
    target: int,
    current_mode: Optional[str],
    face_present: bool,
) -> np.ndarray:
    """Draw the status HUD on a copy of the input frame.

    Layout (top bar, ~130 px tall):
      Row 1 (y=26): recording-mode indicator
      Row 2 (y=60): per-label counters
      Row 3 (y=95, optional): no-face warning
      Top-right corner: coloured dot when recording

    Args:
        frame: BGR frame read from webcam.
        counts: Dict mapping label name to number of saved frames.
        target: Target number of frames per label.
        current_mode: Currently active recording label, or None when idle.
        face_present: Whether a face was detected in the current frame.

    Returns:
        New BGR frame with HUD overlaid.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Semi-transparent background bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), OVERLAY_BG, -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, out, 1.0 - OVERLAY_ALPHA, 0, out)

    # Row 1 – mode indicator
    if current_mode is None:
        mode_text  = "IDLE  (f=focused  d=distracted  o=off_task  q=quit)"
        mode_color = COLOR_WHITE
    else:
        pause_key  = current_mode[0]
        mode_text  = f"RECORDING: {current_mode.upper()}  (press {pause_key} to pause)"
        mode_color = LABEL_COLORS[current_mode]
    cv2.putText(out, mode_text, (10, 26),
                FONT, FONT_SCALE, mode_color, FONT_THICK, cv2.LINE_AA)

    # Row 2 – per-class counters
    col_w = w // len(LABELS)
    for i, label in enumerate(LABELS):
        cnt    = counts[label]
        done   = cnt >= target
        color  = COLOR_GRAY if done else LABEL_COLORS[label]
        suffix = " (done)" if done else ""
        text   = f"{label}: {cnt}/{target}{suffix}"
        cv2.putText(out, text, (10 + i * col_w, 60),
                    FONT, FONT_SCALE, color, FONT_THICK, cv2.LINE_AA)

    # Row 3 – face warning
    if not face_present:
        cv2.putText(out, "! NO FACE DETECTED", (10, 95),
                    FONT, FONT_SCALE, COLOR_RED, FONT_THICK, cv2.LINE_AA)

    # Recording indicator dot (top-right corner)
    if current_mode is not None:
        cv2.circle(out, (w - 22, 22), 10, LABEL_COLORS[current_mode], -1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_dirs(base_dir: Path) -> dict[str, Path]:
    """Create per-label subdirectories under base_dir if they do not exist.

    Args:
        base_dir: Parent directory for webcam_finetune data.

    Returns:
        Dict mapping label name to its Path.
    """
    dirs: dict[str, Path] = {}
    for label in LABELS:
        d = base_dir / label
        d.mkdir(parents=True, exist_ok=True)
        dirs[label] = d
    return dirs


def count_existing(dirs: dict[str, Path]) -> dict[str, int]:
    """Count existing JPEG files in each label directory (resume support).

    Args:
        dirs: Dict mapping label name to directory Path.

    Returns:
        Dict mapping label name to existing JPEG count.
    """
    return {label: len(list(d.glob("*.jpg"))) for label, d in dirs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main capture loop
# ─────────────────────────────────────────────────────────────────────────────

def run(
    data_path: Path,
    target: int = DEFAULT_TARGET,
    cam_index: int = DEFAULT_CAM_INDEX,
    jpg_quality: int = DEFAULT_JPG_QUALITY,
) -> None:
    """Run the interactive frame-collection loop.

    The loop shows a live webcam preview with a HUD overlay.  The user
    presses F/D/O to toggle recording for that label.  Frames are saved
    automatically every iteration while recording is active.  The loop
    terminates when all labels reach *target* frames or the user presses Q.

    Args:
        data_path: Project root directory.
        target: Maximum frames to collect per label.
        cam_index: OpenCV VideoCapture device index.
        jpg_quality: JPEG compression quality (1-100).

    Raises:
        RuntimeError: If the specified webcam cannot be opened.
    """
    base_dir    = data_path / Path(*WEBCAM_FINETUNE_SUBPATH)
    dirs        = setup_dirs(base_dir)
    counts      = count_existing(dirs)

    log.info("Output directory : %s", base_dir)
    for label in LABELS:
        status = "DONE" if counts[label] >= target else f"{counts[label]}/{target}"
        log.info("  %-12s  %s", label, status)

    # Open webcam
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam at device index {cam_index}. "
            "Try --cam_index 1 if multiple cameras are present."
        )
    log.info("Webcam opened (index=%d).", cam_index)
    log.info("Controls: F=focused  D=distracted  O=off_task  Q=quit")

    face_det    = FaceDetector()
    mode: Optional[str] = None
    save_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
    face_ok     = True
    frame_idx   = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                log.warning("Empty frame received; retrying.")
                time.sleep(0.01)
                continue

            # Run face detection every FACE_CHECK_EVERY_N frames (performance)
            if frame_idx % FACE_CHECK_EVERY_N == 0:
                face_ok = face_det.face_present(frame)
            frame_idx += 1

            # Save frame if currently recording and under the target
            if mode is not None and counts[mode] < target:
                save_index = counts[mode]
                filename   = dirs[mode] / f"frame_{save_index:05d}.jpg"
                cv2.imwrite(str(filename), frame, save_params)
                counts[mode] += 1

                # Auto-stop when this label hits target
                if counts[mode] >= target:
                    log.info(
                        "Target reached for label '%s' (%d frames collected).",
                        mode, target,
                    )
                    mode = None

            # Terminate when every label is complete
            if all(counts[lbl] >= target for lbl in LABELS):
                log.info("All labels complete (%d frames each).", target)
                break

            # Draw HUD and display
            vis = draw_overlay(frame, counts, target, mode, face_ok)
            cv2.imshow("Eye Tracking Sampler", vis)

            # Key handling (waitKey returns -1 if no key pressed)
            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                log.info("Quit key pressed.")
                break
            elif key in LABEL_KEY_MAP:
                pressed_label = LABEL_KEY_MAP[key]
                if counts[pressed_label] >= target:
                    log.info("Label '%s' is already complete.", pressed_label)
                elif mode == pressed_label:
                    # Toggle off: pause current recording
                    mode = None
                    log.info("Recording paused for '%s'.", pressed_label)
                else:
                    mode = pressed_label
                    log.info(
                        "Recording mode -> '%s'  (%d/%d)",
                        mode, counts[mode], target,
                    )
    finally:
        cap.release()
        face_det.close()
        cv2.destroyAllWindows()

    # Print summary
    log.info("=" * 60)
    log.info("Session summary:")
    for label in LABELS:
        pct = 100.0 * counts[label] / target
        log.info("  %-12s  %d / %d  (%.0f%%)", label, counts[label], target, pct)
    log.info("Total frames saved this session: %d", sum(counts.values()))
    log.info("Next step: python models/train_attention.py --webcam_finetune_only")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Collect labelled webcam frames for attention model fine-tuning. "
            "Controls: F=focused  D=distracted  O=off_task  Q=quit."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path", type=Path, required=True,
        help="Project root directory.",
    )
    parser.add_argument(
        "--target", type=int, default=DEFAULT_TARGET,
        help="Target number of frames per label class.",
    )
    parser.add_argument(
        "--cam_index", type=int, default=DEFAULT_CAM_INDEX,
        help="OpenCV VideoCapture device index.",
    )
    parser.add_argument(
        "--jpg_quality", type=int, default=DEFAULT_JPG_QUALITY,
        help="JPEG compression quality (1-100).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the collection loop.

    Args:
        argv: Argument list; defaults to sys.argv[1:] when None.
    """
    parser    = _build_arg_parser()
    args      = parser.parse_args(argv)
    data_path = args.data_path.resolve()

    if not data_path.exists():
        log.error("data_path does not exist: %s", data_path)
        sys.exit(1)

    try:
        run(
            data_path,
            target      = args.target,
            cam_index   = args.cam_index,
            jpg_quality = args.jpg_quality,
        )
    except RuntimeError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
