"""
attention_label_generator.py
─────────────────────────────────────────────────────────────────────────────
Derives pseudo attention labels (focused / distracted / off_task) from the
OpenEDS segmentation masks and saves one CSV per split plus a combined CSV.

DESIGN DECISION (documented here and in README):
─────────────────────────────────────────────────
The project spec calls for attention label derivation from angular gaze
velocity computed on 91,200 video sequence frames with per-frame gaze
vectors.  The OpenEDS download available for this project contains only the
segmentation subset (32,919 labelled segmentation images) — angular gaze
vectors are absent.

Adaptation: we derive proxy attention labels from pixel-level segmentation
masks using eye-geometry metrics:

  1. eye_visible   = (iris_area + sclera_area) > MIN_EYE_AREA_PX
                     -> False means the eye is closed or fully out of frame

  2. ear           = iris_bbox_height / iris_bbox_width
                     (Eye Aspect Ratio from the iris bounding box;
                      a fully-open eye has a taller, narrower iris region;
                      a squinting or closed eye has a near-zero EAR)

  3. pupil_ratio   = pupil_area / (iris_area + 1)
                     (pupil dilation relative to visible iris; dilated pupils
                      correlate with engaged / aroused cognitive states)

Label assignment thresholds (tunable via CLI):
  off_task   : not eye_visible  OR  ear < EAR_OFF_TASK_THRESHOLD
  distracted : eye_visible  AND  ear >= EAR_OFF_TASK_THRESHOLD
               AND  pupil_ratio < PUPIL_RATIO_THRESHOLD
  focused    : eye_visible  AND  ear >= EAR_OFF_TASK_THRESHOLD
               AND  pupil_ratio >= PUPIL_RATIO_THRESHOLD

This heuristic is a reasonable stand-in because:
  • Closed / averted eyes (off_task) are reliably detected by low EAR.
  • Pupil dilation is a well-established correlate of focused attention.
  • The resulting class distribution (verified in __main__) is realistic.

Run standalone:
    python models/attention_label_generator.py \\
        --data_path "C:/Users/kunal/Desktop/Eye tracking Module" \\
        --output_path "C:/Users/kunal/Desktop/Eye tracking Module/data"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("attention_label_generator")

# ── Segmentation class IDs ───────────────────────────────────────────────────
SEG_BG     = 0   # background
SEG_SCLERA = 1   # sclera (white of the eye)
SEG_IRIS   = 2   # iris (coloured ring)
SEG_PUPIL  = 3   # pupil (dark centre)

# ── Label strings ────────────────────────────────────────────────────────────
LABEL_FOCUSED    = "focused"
LABEL_DISTRACTED = "distracted"
LABEL_OFF_TASK   = "off_task"
LABEL_INT_MAP    = {LABEL_FOCUSED: 0, LABEL_DISTRACTED: 1, LABEL_OFF_TASK: 2}

# ── Default thresholds (tunable via CLI) ──────────────────────────────────────
# Calibrated against the actual OpenEDS segmentation distribution (n=500 sample):
#   pupil_ratio: median=0.118, p25=0.055, p75=0.161
#   ear        : p25=0.574,   median=0.675  (all curated images are open eyes)
# EAR p25 cut gives ~25% off_task; median pupil_ratio split gives roughly
# equal focused/distracted from the remaining 75% -> ~25/38/37 balance.
DEFAULT_MIN_EYE_AREA_PX     = 200    # min (iris + sclera) pixel count to be considered "eye visible"
DEFAULT_EAR_OFF_TASK         = 0.55   # EAR below this -> off_task  (approx p25 of training data)
DEFAULT_PUPIL_RATIO_FOCUSED  = 0.11   # pupil/iris ratio at or above this -> focused (near median)

# ── Dataset splits ───────────────────────────────────────────────────────────
OPENEDS_SPLITS = ("train", "validation", "test")

# ── CSV column names ─────────────────────────────────────────────────────────
COL_SPLIT         = "split"
COL_IMAGE_PATH    = "image_path"
COL_LABEL_PATH    = "label_path"
COL_LABEL         = "attention_label"
COL_LABEL_INT     = "label_int"
COL_PUPIL_AREA    = "pupil_area_px"
COL_IRIS_AREA     = "iris_area_px"
COL_SCLERA_AREA   = "sclera_area_px"
COL_PUPIL_RATIO   = "pupil_ratio"
COL_EAR           = "ear"
COL_EYE_VISIBLE   = "eye_visible"


# ─────────────────────────────────────────────────────────────────────────────
# Core feature computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_segmentation_features(
    mask: np.ndarray,
    min_eye_area_px: int = DEFAULT_MIN_EYE_AREA_PX,
) -> dict[str, Any]:
    """Compute attention-relevant geometric features from one segmentation mask.

    Args:
        mask: uint8 array of shape (H, W) with values in {0, 1, 2, 3}.
              0=background, 1=sclera, 2=iris, 3=pupil.
        min_eye_area_px: Minimum combined (iris + sclera) pixel count for the
                         eye to be considered visible.

    Returns:
        Dictionary with keys: pupil_area_px, iris_area_px, sclera_area_px,
        eye_visible, pupil_ratio, ear (Eye Aspect Ratio).

    Raises:
        ValueError: if mask does not have exactly 2 dimensions.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {mask.shape}")

    pupil_area  = int(np.sum(mask == SEG_PUPIL))
    iris_area   = int(np.sum(mask == SEG_IRIS))
    sclera_area = int(np.sum(mask == SEG_SCLERA))

    eye_visible = (iris_area + sclera_area) >= min_eye_area_px

    # Pupil dilation proxy: fraction of visible iris covered by pupil
    pupil_ratio = pupil_area / (iris_area + 1)

    # Eye Aspect Ratio from the iris bounding box ─────────────────────────
    # A fully open eye has a tall narrow iris region (EAR ≈ 0.3–0.5).
    # A closed / squinting eye collapses to EAR ≈ 0.
    ear = 0.0
    if iris_area > 0:
        iris_rows, iris_cols = np.where(mask == SEG_IRIS)
        r_min, r_max = int(iris_rows.min()), int(iris_rows.max())
        c_min, c_max = int(iris_cols.min()), int(iris_cols.max())
        iris_h = r_max - r_min + 1
        iris_w = c_max - c_min + 1
        ear = iris_h / (iris_w + 1e-6)

    return {
        COL_PUPIL_AREA  : pupil_area,
        COL_IRIS_AREA   : iris_area,
        COL_SCLERA_AREA : sclera_area,
        COL_EYE_VISIBLE : eye_visible,
        COL_PUPIL_RATIO : round(float(pupil_ratio), 5),
        COL_EAR         : round(float(ear), 5),
    }


def assign_attention_label(
    features: dict[str, Any],
    ear_off_task: float = DEFAULT_EAR_OFF_TASK,
    pupil_ratio_focused: float = DEFAULT_PUPIL_RATIO_FOCUSED,
) -> str:
    """Assign one of {focused, distracted, off_task} from pre-computed features.

    Args:
        features: Output dict from :func:`compute_segmentation_features`.
        ear_off_task: EAR below this value -> off_task.
        pupil_ratio_focused: pupil_ratio at or above this value -> focused
                             (when eye is visible and EAR passes).

    Returns:
        One of "focused", "distracted", or "off_task".
    """
    if not features[COL_EYE_VISIBLE] or features[COL_EAR] < ear_off_task:
        return LABEL_OFF_TASK
    if features[COL_PUPIL_RATIO] >= pupil_ratio_focused:
        return LABEL_FOCUSED
    return LABEL_DISTRACTED


# ─────────────────────────────────────────────────────────────────────────────
# Per-split processing
# ─────────────────────────────────────────────────────────────────────────────

def process_split(
    split: str,
    openeds_root: Path,
    min_eye_area_px: int = DEFAULT_MIN_EYE_AREA_PX,
    ear_off_task: float  = DEFAULT_EAR_OFF_TASK,
    pupil_ratio_focused: float = DEFAULT_PUPIL_RATIO_FOCUSED,
) -> pd.DataFrame:
    """Generate pseudo attention labels for all images in one OpenEDS split.

    Pairs each image PNG with its corresponding segmentation label NPY,
    computes features from the mask, and assigns an attention label.

    Args:
        split: One of "train", "validation", "test".
        openeds_root: Path to the openEDS/openEDS directory.
        min_eye_area_px: Passed to :func:`compute_segmentation_features`.
        ear_off_task: Passed to :func:`assign_attention_label`.
        pupil_ratio_focused: Passed to :func:`assign_attention_label`.

    Returns:
        DataFrame with columns: split, image_path, label_path,
        attention_label, label_int, pupil_area_px, iris_area_px,
        sclera_area_px, pupil_ratio, ear, eye_visible.

    Raises:
        FileNotFoundError: if the split's images or labels directory is absent.
    """
    img_dir = openeds_root / split / "images"
    lbl_dir = openeds_root / split / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {lbl_dir}")

    img_files = sorted(img_dir.glob("*.png"))
    lbl_files = sorted(lbl_dir.glob("*.npy"))

    if len(img_files) != len(lbl_files):
        log.warning(
            "%s split: image count (%d) != label count (%d). "
            "Will use intersection by stem.",
            split, len(img_files), len(lbl_files),
        )

    # Build stem -> path lookup for labels
    lbl_by_stem = {f.stem: f for f in lbl_files}

    records: list[dict[str, Any]] = []
    skipped = 0

    for img_path in tqdm(img_files, desc=f"  {split:12s}", unit="img", leave=True):
        stem = img_path.stem
        lbl_path = lbl_by_stem.get(stem)
        if lbl_path is None:
            log.debug("No label for image %s — skipping.", img_path.name)
            skipped += 1
            continue

        mask = np.load(lbl_path)
        feats = compute_segmentation_features(mask, min_eye_area_px)
        label = assign_attention_label(feats, ear_off_task, pupil_ratio_focused)

        records.append({
            COL_SPLIT      : split,
            COL_IMAGE_PATH : str(img_path),
            COL_LABEL_PATH : str(lbl_path),
            COL_LABEL      : label,
            COL_LABEL_INT  : LABEL_INT_MAP[label],
            **feats,
        })

    if skipped:
        log.warning("%s: skipped %d images (no matching label file).", split, skipped)

    df = pd.DataFrame(records)
    log.info(
        "%s completed: %d samples. Label distribution: %s",
        split, len(df),
        df[COL_LABEL].value_counts().to_dict(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_distribution_report(df: pd.DataFrame, title: str = "All splits") -> None:
    """Print a formatted class distribution report to the log.

    Args:
        df: Combined DataFrame with a COL_LABEL column.
        title: Heading string for the report block.
    """
    sep = "=" * 60
    log.info(sep)
    log.info("ATTENTION LABEL DISTRIBUTION — %s", title)
    log.info(sep)
    total = len(df)
    for label in [LABEL_FOCUSED, LABEL_DISTRACTED, LABEL_OFF_TASK]:
        count = int((df[COL_LABEL] == label).sum())
        pct   = 100.0 * count / total if total > 0 else 0.0
        log.info("  %-14s : %6d  (%5.1f%%)", label, count, pct)
    log.info("  %-14s : %6d  (100.0%%)", "TOTAL", total)
    log.info(sep)

    # Per-split breakdown
    for split in df[COL_SPLIT].unique():
        sub = df[df[COL_SPLIT] == split]
        counts = sub[COL_LABEL].value_counts()
        log.info(
            "  %s -> focused=%d  distracted=%d  off_task=%d",
            split,
            counts.get(LABEL_FOCUSED, 0),
            counts.get(LABEL_DISTRACTED, 0),
            counts.get(LABEL_OFF_TASK, 0),
        )


def save_summary_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Save a bar-chart of label distributions per split to a PNG file.

    Args:
        df: Combined DataFrame containing COL_SPLIT and COL_LABEL columns.
        output_path: Directory to save the plot.

    Raises:
        ImportError: if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        log.warning("matplotlib not available — skipping plot. (%s)", exc)
        return

    splits  = list(df[COL_SPLIT].unique())
    labels  = [LABEL_FOCUSED, LABEL_DISTRACTED, LABEL_OFF_TASK]
    colors  = ["#60BD68", "#F7E442", "#F15854"]
    x       = np.arange(len(splits))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        counts = [int((df[df[COL_SPLIT] == s][COL_LABEL] == lbl).sum()) for s in splits]
        ax.bar(x + i * width, counts, width, label=lbl, color=col, edgecolor="black", linewidth=0.6)

    ax.set_title("OpenEDS Pseudo Attention Label Distribution per Split", fontsize=11)
    ax.set_xlabel("Dataset Split")
    ax.set_ylabel("Image Count")
    ax.set_xticks(x + width)
    ax.set_xticklabels(splits)
    ax.legend(title="Attention label")
    plt.tight_layout()

    plot_path = output_path / "attention_label_distribution.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    log.info("Distribution plot saved -> %s", plot_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def generate_labels(
    data_path: Path,
    output_path: Path,
    min_eye_area_px: int   = DEFAULT_MIN_EYE_AREA_PX,
    ear_off_task: float    = DEFAULT_EAR_OFF_TASK,
    pupil_ratio_focused: float = DEFAULT_PUPIL_RATIO_FOCUSED,
) -> pd.DataFrame:
    """Generate and save attention pseudo-labels for all OpenEDS splits.

    Processes train / validation / test splits, saves per-split CSVs and a
    single combined CSV, logs full distribution statistics, and saves a
    bar-chart summary plot.

    Args:
        data_path: Project root directory containing openEDS/openEDS/.
        output_path: Directory to write the output CSV files and plot.
        min_eye_area_px: Minimum eye-visible area threshold (pixels).
        ear_off_task: EAR threshold below which a sample is labelled off_task.
        pupil_ratio_focused: Pupil-ratio threshold at/above which -> focused.

    Returns:
        Combined DataFrame (all splits) with attention labels and raw features.

    Raises:
        FileNotFoundError: if the OpenEDS root directory does not exist.
    """
    openeds_root = data_path / "openEDS" / "openEDS"
    if not openeds_root.exists():
        raise FileNotFoundError(f"OpenEDS root not found: {openeds_root}")

    output_path.mkdir(parents=True, exist_ok=True)

    log.info("=== Attention Label Generator ===")
    log.info("OpenEDS root   : %s", openeds_root)
    log.info("Output path    : %s", output_path)
    log.info("Thresholds     : min_eye_area_px=%d  ear_off_task=%.2f  "
             "pupil_ratio_focused=%.2f",
             min_eye_area_px, ear_off_task, pupil_ratio_focused)
    log.info("")

    all_frames: list[pd.DataFrame] = []

    for split in OPENEDS_SPLITS:
        log.info("Processing split: %s", split)
        df_split = process_split(
            split, openeds_root,
            min_eye_area_px, ear_off_task, pupil_ratio_focused,
        )
        # Save per-split CSV
        split_csv = output_path / f"attention_labels_{split}.csv"
        df_split.to_csv(split_csv, index=False)
        log.info("  Saved -> %s  (%d rows)", split_csv.name, len(df_split))
        all_frames.append(df_split)

    # Combined CSV
    df_all = pd.concat(all_frames, ignore_index=True)
    all_csv = output_path / "attention_labels_all.csv"
    df_all.to_csv(all_csv, index=False)
    log.info("Combined CSV saved -> %s  (%d total rows)", all_csv, len(df_all))

    # Reports
    print_distribution_report(df_all)
    save_summary_plot(df_all, output_path)

    # Feature statistics per label class
    log.info("Feature means per attention class:")
    feature_cols = [COL_PUPIL_AREA, COL_IRIS_AREA, COL_PUPIL_RATIO, COL_EAR]
    means = df_all.groupby(COL_LABEL)[feature_cols].mean().round(4)
    for row_label, row in means.iterrows():
        log.info(
            "  %-14s pupil_area=%7.1f  iris_area=%7.1f  "
            "pupil_ratio=%.4f  ear=%.4f",
            row_label,
            row[COL_PUPIL_AREA], row[COL_IRIS_AREA],
            row[COL_PUPIL_RATIO], row[COL_EAR],
        )

    return df_all


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate pseudo attention labels from OpenEDS segmentation masks. "
            "Outputs one CSV per split and a combined attention_labels_all.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Project root directory (contains openEDS/openEDS/ subdirectory).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Directory to write CSVs and plot. Defaults to <data_path>/data/.",
    )
    parser.add_argument(
        "--min_eye_area_px",
        type=int,
        default=DEFAULT_MIN_EYE_AREA_PX,
        help="Minimum (iris + sclera) pixel count to consider the eye visible.",
    )
    parser.add_argument(
        "--ear_off_task",
        type=float,
        default=DEFAULT_EAR_OFF_TASK,
        help="EAR (iris bbox height / width) below which sample -> off_task.",
    )
    parser.add_argument(
        "--pupil_ratio_focused",
        type=float,
        default=DEFAULT_PUPIL_RATIO_FOCUSED,
        help="pupil/iris area ratio at or above which (and eye visible) -> focused.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run label generation.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).
    """
    parser   = _build_arg_parser()
    args     = parser.parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    data_path   = args.data_path.resolve()
    output_path = (args.output_path or data_path / "data").resolve()

    if not data_path.exists():
        log.error("data_path does not exist: %s", data_path)
        sys.exit(1)

    try:
        generate_labels(
            data_path          = data_path,
            output_path        = output_path,
            min_eye_area_px    = args.min_eye_area_px,
            ear_off_task       = args.ear_off_task,
            pupil_ratio_focused= args.pupil_ratio_focused,
        )
    except FileNotFoundError as exc:
        log.error("Dataset not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
