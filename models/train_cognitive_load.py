"""
train_cognitive_load.py
────────────────────────────────────────────────────────────────────────────
Train a 3-class cognitive load classifier (low / medium / high) using
XGBoost on 4 eye-tracking features from the Multimodal Cognitive Load Dataset.

Output classes : 0=low (0-back), 1=medium (1-back), 2=high (2-back)
Input features : [Pupil_Dilation, Blink_Rate, Fixation_Duration, Saccade_Duration]
Architecture   : XGBoost gradient-boosted trees

Saved artefacts:
    saved_models/cogload_model.pkl    <- XGBoost model (pickled)
    saved_models/cogload_scaler.pkl   <- StandardScaler for 4 features

Run standalone:
    python models/train_cognitive_load.py \\
        --data_path "C:/Users/kunal/Desktop/Eye tracking Module" \\
        --epochs 300

Accuracy note (for the project report):
    Using ONLY the 4 eye-tracking features from a 414-feature dataset
    deliberately discards EEG (384 cols) and fNIRS (20 cols).
    State-of-the-art eye-only cognitive load classification achieves 58-68%
    on 3-class tasks.  The full multimodal reference is 85-92%.
    This gap is documented as a known trade-off: EEG/fNIRS cannot be
    collected from a standard webcam.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

# XGBoost 3.x: suppress the GPU->CPU device mismatch warning on predict().
# Predictions are always correct; this is a performance note only.
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_cognitive_load")

# ── Dataset ──────────────────────────────────────────────────────────────────
COGLOAD_CSV_NAME = "cognitive_load_dataset.csv"
LABEL_COL        = "Cognitive_Load"
FEATURE_COLS     = [
    "Pupil_Dilation",
    "Blink_Rate",
    "Fixation_Duration",
    "Saccade_Duration",
]
LABEL_NAMES      = {0: "low (0-back)", 1: "medium (1-back)", 2: "high (2-back)"}
NUM_CLASSES      = 3

# ── Saved artefact filenames ─────────────────────────────────────────────────
MODEL_FILENAME   = "cogload_model.pkl"
SCALER_FILENAME  = "cogload_scaler.pkl"

# ── XGBoost hyperparameters (spec-defined) ────────────────────────────────────
N_ESTIMATORS_MAX  = 300
MAX_DEPTH         = 6
LEARNING_RATE     = 0.05
SUBSAMPLE         = 0.8
COLSAMPLE_BYTREE  = 0.8
EARLY_STOP_ROUNDS = 10
MIN_CHILD_WEIGHT  = 1
GAMMA             = 0.0
RANDOM_SEED       = 42

# ── Split fractions ───────────────────────────────────────────────────────────
VAL_FRACTION  = 0.15
TEST_FRACTION = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_cogload(data_path: Path) -> pd.DataFrame:
    """Load the cognitive load dataset, keeping only eye features and label.

    Drops all 410 non-eye columns (EEG x384, fNIRS x20, driving x6).
    Converts the float label column to integer.

    Args:
        data_path: Project root directory containing cognitive_load_dataset.csv.

    Returns:
        DataFrame with columns FEATURE_COLS + [LABEL_COL] (integer label).

    Raises:
        FileNotFoundError: if the CSV is not found.
        KeyError: if expected columns are absent.
    """
    csv_path = data_path / COGLOAD_CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Cognitive load CSV not found: {csv_path}")

    log.info("Loading CSV: %s", csv_path)
    # Only load the columns we need -- much faster on 86k x 415 CSV
    usecols = FEATURE_COLS + [LABEL_COL]
    df = pd.read_csv(csv_path, usecols=usecols)
    log.info("Loaded shape: %s", df.shape)

    # Confirm columns
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing: {missing}")

    # Convert label to int
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    log.info("First 5 rows:\n%s", df.head().to_string())
    log.info("Label distribution:\n%s", df[LABEL_COL].value_counts().sort_index().to_string())
    log.info("Null values: %d", df.isnull().sum().sum())

    # Feature statistics
    log.info("Feature statistics:\n%s", df[FEATURE_COLS].describe().round(4).to_string())

    return df


def split_and_scale(
    df: pd.DataFrame,
    val_frac: float = VAL_FRACTION,
    test_frac: float = TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    StandardScaler,
]:
    """Stratified 70/15/15 split and StandardScaler normalisation.

    Args:
        df: Full DataFrame with FEATURE_COLS and LABEL_COL.
        val_frac: Validation fraction.
        test_frac: Test fraction.
        seed: Random state for reproducibility.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, scaler).
        X arrays are float32 numpy arrays scaled by the fitted StandardScaler.
        y arrays are int64 numpy arrays.
    """
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int64)

    # Split off test set first
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(X, y))

    X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
    X_test,     y_test     = X[test_idx],     y[test_idx]

    # Split val from trainval
    adjusted_val = val_frac / (1.0 - test_frac)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=seed)
    train_idx, val_idx = next(sss_val.split(X_trainval, y_trainval))

    X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
    X_val,   y_val   = X_trainval[val_idx],   y_trainval[val_idx]

    log.info(
        "Split sizes: train=%d  val=%d  test=%d",
        len(y_train), len(y_val), len(y_test),
    )
    for name, y_arr in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique, counts = np.unique(y_arr, return_counts=True)
        dist = {LABEL_NAMES[int(k)]: int(v) for k, v in zip(unique, counts)}
        log.info("  %s class distribution: %s", name, dist)

    # Fit scaler on training set only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    log.info(
        "Scaler fitted. Feature means: %s",
        dict(zip(FEATURE_COLS, scaler.mean_.round(4))),
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    n_estimators: int,
    use_gpu: bool = False,
) -> xgb.XGBClassifier:
    """Create the XGBoost classifier with spec-defined hyperparameters.

    Args:
        n_estimators: Maximum number of boosting rounds (trees).
        use_gpu: If True, attempt to use CUDA device.

    Returns:
        Configured XGBClassifier instance (not yet fitted).
    """
    device = "cuda" if use_gpu else "cpu"

    model = xgb.XGBClassifier(
        n_estimators        = n_estimators,
        max_depth           = MAX_DEPTH,
        learning_rate       = LEARNING_RATE,
        subsample           = SUBSAMPLE,
        colsample_bytree    = COLSAMPLE_BYTREE,
        min_child_weight    = MIN_CHILD_WEIGHT,
        gamma               = GAMMA,
        objective           = "multi:softprob",
        num_class           = NUM_CLASSES,
        eval_metric         = "mlogloss",
        early_stopping_rounds = EARLY_STOP_ROUNDS,
        tree_method         = "hist",
        device              = device,
        random_state        = RANDOM_SEED,
        verbosity           = 1,
    )
    log.info(
        "XGBoost model configured: n_estimators=%d  max_depth=%d  "
        "lr=%.3f  device=%s",
        n_estimators, MAX_DEPTH, LEARNING_RATE, device,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str = "test",
) -> None:
    """Log classification report and confusion matrix.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        split_name: Name of the evaluated split (for log headers).
    """
    target_names = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]
    acc = float((y_true == y_pred).mean())

    log.info("=" * 65)
    log.info("EVALUATION -- %s  (n=%d)", split_name.upper(), len(y_true))
    log.info("=" * 65)
    log.info("Accuracy: %.4f  (%.1f%%)", acc, 100.0 * acc)
    log.info(
        "Classification report:\n%s",
        classification_report(y_true, y_pred, target_names=target_names, zero_division=0),
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    log.info("Confusion matrix (rows=true, cols=pred):")
    header = "             " + "  ".join(f"pred_{LABEL_NAMES[i][:3]:>7}" for i in range(NUM_CLASSES))
    log.info(header)
    for i, row in enumerate(cm):
        log.info("true_%-8s  %s", LABEL_NAMES[i][:8], "  ".join(f"{v:>11}" for v in row))
    log.info("=" * 65)


def log_feature_importance(model: xgb.XGBClassifier) -> None:
    """Log the model's feature importance scores.

    Args:
        model: A fitted XGBClassifier.
    """
    importance = model.feature_importances_
    log.info("Feature importance (gain-based):")
    for col, score in sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1]):
        bar = "#" * int(score * 40)
        log.info("  %-22s : %.4f  %s", col, score, bar)


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_path: Path,
    output_path: Path,
    n_estimators: int = N_ESTIMATORS_MAX,
    debug: bool = False,
) -> None:
    """Full training pipeline: load -> split -> scale -> train -> evaluate -> save.

    Args:
        data_path: Project root directory (contains cognitive_load_dataset.csv).
        output_path: Directory to write model and scaler artefacts.
        n_estimators: Maximum boosting rounds (early stopping may reduce this).
        debug: If True, enable DEBUG logging.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    np.random.seed(RANDOM_SEED)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = load_cogload(data_path)

    # ── 2. Split + scale ─────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = split_and_scale(df)

    # ── 3. Detect GPU ─────────────────────────────────────────────────────────
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False
    log.info("GPU for XGBoost: %s", use_gpu)

    # ── 4. Build and fit ─────────────────────────────────────────────────────
    model = build_model(n_estimators=n_estimators, use_gpu=use_gpu)

    log.info("=" * 65)
    log.info(
        "Starting XGBoost training (max %d rounds, early_stopping=%d)",
        n_estimators, EARLY_STOP_ROUNDS,
    )
    log.info("=" * 65)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,   # print eval metric every 50 rounds
    )

    log.info(
        "Training complete. Best iteration: %d  best_val_mlogloss=%.6f",
        model.best_iteration,
        model.best_score,
    )

    # ── 5. Save artefacts ─────────────────────────────────────────────────────
    output_path.mkdir(parents=True, exist_ok=True)

    model_path  = output_path / MODEL_FILENAME
    scaler_path = output_path / SCALER_FILENAME

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("Model saved -> %s", model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info("Scaler saved -> %s", scaler_path)

    # ── 6. Validation evaluation ──────────────────────────────────────────────
    y_val_pred = model.predict(X_val)
    print_evaluation_report(y_val, y_val_pred, split_name="validation")

    # ── 7. Test evaluation ────────────────────────────────────────────────────
    y_test_pred = model.predict(X_test)
    print_evaluation_report(y_test, y_test_pred, split_name="test")

    # ── 8. Feature importance ─────────────────────────────────────────────────
    log_feature_importance(model)

    # ── 9. Summary ────────────────────────────────────────────────────────────
    test_acc = float((y_test == y_test_pred).mean())
    log.info("=" * 65)
    log.info("Target accuracy range (eye-only, 3-class): 58-68%%")
    log.info("Achieved: %.1f%%", 100.0 * test_acc)
    log.info("")
    log.info("Note: The full 414-feature dataset (with EEG + fNIRS) achieves")
    log.info("85-92%%.  Eye features alone cannot close this gap without")
    log.info("additional physiological sensors.  This is expected and documented")
    log.info("as a design decision -- only eye features are available from a")
    log.info("standard webcam at inference time.")
    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train an XGBoost 3-class cognitive load classifier on 4 eye features. "
            "Saves cogload_model.pkl and cogload_scaler.pkl to saved_models/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Project root directory (contains cognitive_load_dataset.csv).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=N_ESTIMATORS_MAX,
        help=(
            "Maximum XGBoost boosting rounds (n_estimators). "
            "Training will stop early if val mlogloss does not improve "
            f"for {EARLY_STOP_ROUNDS} consecutive rounds."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Directory to write artefacts. Defaults to <data_path>/saved_models/.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the training pipeline.

    Args:
        argv: Argument list; defaults to sys.argv[1:] when None.
    """
    parser = _build_arg_parser()
    args   = parser.parse_args(argv)

    data_path   = args.data_path.resolve()
    output_path = (args.output_path or data_path / "saved_models").resolve()

    if not data_path.exists():
        log.error("data_path does not exist: %s", data_path)
        sys.exit(1)

    log.info("data_path   = %s", data_path)
    log.info("output_path = %s", output_path)

    try:
        train(
            data_path    = data_path,
            output_path  = output_path,
            n_estimators = args.epochs,
            debug        = args.debug,
        )
    except (FileNotFoundError, KeyError) as exc:
        log.error("Setup error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
