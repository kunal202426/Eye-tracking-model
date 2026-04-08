"""
train_emotion.py
────────────────────────────────────────────────────────────────────────────
Train a 4-class emotion MLP on VREED eye-tracking features.

Output classes  : 0=sad, 1=calm, 2=angry, 3=happy  (Quad_Cat from VREED)
Input features  : [Num_of_Blink, Mean_Blink_Duration,
                   Mean_Fixation_Duration, Mean_Saccade_Amplitude]
Architecture    : MLP  Input(4) -> Dense(64,ReLU) -> Dropout(0.3)
                            -> Dense(32,ReLU) -> Dropout(0.2) -> Dense(4)
Saved artefacts : saved_models/emotion_model.pth
                  saved_models/emotion_scaler.pkl

Run standalone:
    python models/train_emotion.py \\
        --data_path "C:/Users/kunal/Desktop/Eye tracking Module" \\
        --epochs 100

Design note -- domain gap:
    VREED features are extracted from VR infrared eye-tracking at 200 Hz.
    Live features come from a webcam at 30 fps.  The StandardScaler saved
    here normalises based on the VREED distribution.  At inference time,
    feature_extractor.py performs an additional per-session baseline
    calibration that shifts live feature distributions before the scaler
    is applied.  The scaler must NOT be re-fitted on live data.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_emotion")

# ── Dataset ──────────────────────────────────────────────────────────────────
VREED_SUBPATH  = ("04 Eye Tracking Data",
                   "02 Eye Tracking Data (Features Extracted)",
                   "EyeTracking_FeaturesExtracted.csv")
LABEL_COL      = "Quad_Cat"

# The 4 input features mapped from VREED to the live-inference equivalents:
#   Num_of_Blink          -> pupil_dilation proxy  (arousal correlate)
#   Mean_Blink_Duration   -> blink_rate proxy       (ms per blink event)
#   Mean_Fixation_Duration -> fixation_duration      (ms)
#   Mean_Saccade_Amplitude -> saccade_amplitude      (degrees)
FEATURE_COLS   = [
    "Num_of_Blink",
    "Mean_Blink_Duration",
    "Mean_Fixation_Duration",
    "Mean_Saccade_Amplitude",
]

EMOTION_NAMES  = {0: "sad", 1: "calm", 2: "angry", 3: "happy"}
NUM_CLASSES    = 4

# ── Saved artefact filenames ─────────────────────────────────────────────────
MODEL_FILENAME  = "emotion_model.pth"
SCALER_FILENAME = "emotion_scaler.pkl"

# ── Architecture ─────────────────────────────────────────────────────────────
HIDDEN_1        = 64
HIDDEN_2        = 32
DROPOUT_1       = 0.3
DROPOUT_2       = 0.2

# ── Training defaults ─────────────────────────────────────────────────────────
DEFAULT_EPOCHS       = 100
DEFAULT_LR           = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE   = 32
EARLY_STOP_PATIENCE  = 10
VAL_FRACTION         = 0.15
TEST_FRACTION        = 0.15
RANDOM_SEED          = 42


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class EmotionMLP(nn.Module):
    """4-class emotion MLP trained on 4 eye-tracking feature scalars.

    Architecture:
        Input(4) -> Linear(64) -> BatchNorm -> ReLU -> Dropout(0.3)
                 -> Linear(32) -> BatchNorm -> ReLU -> Dropout(0.2)
                 -> Linear(4)

    Note: Softmax is NOT applied here; use nn.CrossEntropyLoss which
    includes log-softmax internally.  Apply softmax manually at inference.
    """

    def __init__(
        self,
        num_features: int = 4,
        hidden_1: int = HIDDEN_1,
        hidden_2: int = HIDDEN_2,
        num_classes: int = NUM_CLASSES,
        dropout_1: float = DROPOUT_1,
        dropout_2: float = DROPOUT_2,
    ) -> None:
        """Initialise the MLP.

        Args:
            num_features: Number of scalar input features.
            hidden_1: Width of the first hidden layer.
            hidden_2: Width of the second hidden layer.
            num_classes: Number of output emotion classes.
            dropout_1: Dropout probability after the first hidden layer.
            dropout_2: Dropout probability after the second hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_1),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_2),
            nn.Linear(hidden_2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape (batch, num_features).

        Returns:
            Raw logits of shape (batch, num_classes).
        """
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving.

    Saves the best model weights when a new minimum is found.
    """

    def __init__(self, patience: int = EARLY_STOP_PATIENCE, min_delta: float = 1e-5) -> None:
        """Initialise early stopping tracker.

        Args:
            patience: Number of epochs to wait after last improvement.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Update the tracker with the current validation loss.

        Args:
            val_loss: Current epoch validation loss.
            model: The model whose weights to snapshot on improvement.

        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log.debug("EarlyStopping: new best val_loss=%.6f", val_loss)
        else:
            self.counter += 1
            log.debug("EarlyStopping: no improvement %d/%d", self.counter, self.patience)
            if self.counter >= self.patience:
                return True
        return False

    def restore_best(self, model: nn.Module) -> None:
        """Load the best-seen weights back into the model.

        Args:
            model: Model to restore weights into.

        Raises:
            RuntimeError: if no best state has been recorded yet.
        """
        if self.best_state is None:
            raise RuntimeError("No best state recorded yet.")
        model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_vreed(data_path: Path) -> pd.DataFrame:
    """Load the VREED features CSV and log a summary.

    Performs median imputation for any nulls in the 4 feature columns
    and returns the DataFrame with only the feature + label columns.

    Args:
        data_path: Project root directory.

    Returns:
        Cleaned DataFrame with columns FEATURE_COLS + [LABEL_COL].

    Raises:
        FileNotFoundError: if the CSV is not found at the expected path.
    """
    csv_path = Path(data_path, *VREED_SUBPATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"VREED CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    log.info("Loaded VREED CSV: shape=%s  path=%s", df.shape, csv_path)
    log.info("All columns (%d): %s", len(df.columns), df.columns.tolist())

    # Confirm required columns exist
    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns not found in VREED CSV: {missing}")

    # ── Impute nulls with column median ─────────────────────────────────────
    null_counts = df[FEATURE_COLS].isnull().sum()
    if null_counts.sum() > 0:
        log.warning("Null values found -- imputing with column medians:")
        for col, n in null_counts[null_counts > 0].items():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            log.warning("  %s: %d nulls imputed with median=%.6f", col, n, med)

    df = df[FEATURE_COLS + [LABEL_COL]].copy()
    log.info("After imputation: shape=%s  nulls=%d", df.shape, df.isnull().sum().sum())
    log.info("First 5 rows:\n%s", df.head().to_string())
    log.info("Label distribution: %s", df[LABEL_COL].value_counts().sort_index().to_dict())
    return df


def split_dataset(
    df: pd.DataFrame,
    val_frac: float = VAL_FRACTION,
    test_frac: float = TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 train / val / test split.

    Args:
        df: Full DataFrame with feature and label columns.
        val_frac: Fraction for validation set.
        test_frac: Fraction for test set.
        seed: Random state for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    # First split off test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(X, y))

    df_trainval = df.iloc[train_val_idx].reset_index(drop=True)
    df_test     = df.iloc[test_idx].reset_index(drop=True)

    # Then split val from train_val
    adjusted_val = val_frac / (1.0 - test_frac)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=seed)
    train_idx, val_idx = next(sss_val.split(df_trainval[FEATURE_COLS].values,
                                             df_trainval[LABEL_COL].values))

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

    log.info(
        "Split sizes: train=%d  val=%d  test=%d",
        len(df_train), len(df_val), len(df_test),
    )
    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        dist = subset[LABEL_COL].value_counts().sort_index().to_dict()
        log.info("  %s class distribution: %s", name, dist)

    return df_train, df_val, df_test


def make_tensors(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    """Scale features and convert to tensors.

    Args:
        df: DataFrame with FEATURE_COLS and LABEL_COL.
        scaler: Pre-fitted StandardScaler, or None to create a new one.
        fit_scaler: If True, fit the scaler on this DataFrame's features.
                    Must be True for the training set, False for val/test.

    Returns:
        Tuple of (X_tensor, y_tensor, scaler).

    Raises:
        ValueError: if fit_scaler=False but scaler is None.
    """
    if not fit_scaler and scaler is None:
        raise ValueError("scaler must be provided when fit_scaler=False.")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int64)

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        log.info(
            "Scaler fitted on training set. Feature means: %s",
            dict(zip(FEATURE_COLS, scaler.mean_.round(4))),
        )
    else:
        X = scaler.transform(X).astype(np.float32)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        scaler,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        model: The EmotionMLP model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimiser: Gradient optimiser.
        device: Compute device.

    Returns:
        Tuple of (mean_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model for one epoch.

    Args:
        model: The EmotionMLP model.
        loader: Validation or test DataLoader.
        criterion: Loss function (must not modify gradients).
        device: Compute device.

    Returns:
        Tuple of (mean_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def train(
    data_path: Path,
    output_path: Path,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    debug: bool = False,
) -> None:
    """Full training pipeline: load -> split -> scale -> train -> evaluate -> save.

    Args:
        data_path: Project root directory (contains VREED CSV).
        output_path: Directory to write saved_models/ artefacts.
        epochs: Maximum training epochs.
        lr: Initial Adam learning rate.
        batch_size: Mini-batch size.
        weight_decay: L2 regularisation for Adam.
        debug: If True, log raw logits for every sample in the test set.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    df = load_vreed(data_path)

    # ── 2. Split ──────────────────────────────────────────────────────────────
    df_train, df_val, df_test = split_dataset(df)

    # ── 3. Scale and to tensors ───────────────────────────────────────────────
    X_train, y_train, scaler = make_tensors(df_train, fit_scaler=True)
    X_val,   y_val,   _      = make_tensors(df_val,   scaler=scaler, fit_scaler=False)
    X_test,  y_test,  _      = make_tensors(df_test,  scaler=scaler, fit_scaler=False)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False,
    )

    # ── 4. Build model ────────────────────────────────────────────────────────
    model     = EmotionMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5,
    )
    stopper   = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    log.info("Model architecture:\n%s", model)
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Total trainable parameters: %d", total_params)

    # ── 5. Training loop ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Starting training: epochs=%d  lr=%.5f  batch=%d", epochs, lr, batch_size)
    log.info("=" * 60)

    best_epoch = 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimiser, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step(vl_loss)

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  train_acc=%.3f  "
            "val_loss=%.4f  val_acc=%.3f  lr=%.6f",
            epoch, epochs, tr_loss, tr_acc, vl_loss, vl_acc,
            optimiser.param_groups[0]["lr"],
        )

        stopped = stopper.step(vl_loss, model)
        if stopper.best_loss == vl_loss:
            best_epoch = epoch
        if stopped:
            log.info("Early stopping triggered at epoch %d.", epoch)
            break

    stopper.restore_best(model)
    log.info("Best epoch: %d  best_val_loss=%.6f", best_epoch, stopper.best_loss)

    # ── 6. Save artefacts ─────────────────────────────────────────────────────
    output_path.mkdir(parents=True, exist_ok=True)

    model_path  = output_path / MODEL_FILENAME
    scaler_path = output_path / SCALER_FILENAME

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "emotion_names": EMOTION_NAMES,
            "feature_cols": FEATURE_COLS,
            "num_classes": NUM_CLASSES,
            "hidden_1": HIDDEN_1,
            "hidden_2": HIDDEN_2,
            "best_val_loss": stopper.best_loss,
            "best_epoch": best_epoch,
        },
        model_path,
    )
    log.info("Model saved -> %s", model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info("Scaler saved -> %s", scaler_path)

    # ── 7. Final evaluation ───────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("FINAL EVALUATION ON TEST SET")
    log.info("=" * 60)

    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    log.info("Test loss=%.4f  Test accuracy=%.4f  (%d samples)", test_loss, test_acc, len(y_test))

    # Gather all predictions for confusion matrix
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(device))
            all_logits.append(logits.cpu())
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y_batch)

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    logits = torch.cat(all_logits).numpy()

    target_names = [f"{EMOTION_NAMES[i]} (Q{i})" for i in range(NUM_CLASSES)]

    log.info("Classification report:\n%s",
             classification_report(labels, preds, target_names=target_names,
                                   zero_division=0))

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    log.info("Confusion matrix (rows=true, cols=pred):")
    header = "        " + "  ".join(f"pred_{EMOTION_NAMES[i][:3]:>5}" for i in range(NUM_CLASSES))
    log.info(header)
    for i, row in enumerate(cm):
        log.info("true_%-6s  %s", EMOTION_NAMES[i], "  ".join(f"{v:>8}" for v in row))

    if debug:
        log.debug("Per-sample raw logits (first 20 test samples):")
        for idx in range(min(20, len(labels))):
            lgts = logits[idx]
            log.debug(
                "  sample %3d  true=%s  pred=%s  logits=%s",
                idx,
                EMOTION_NAMES[labels[idx]],
                EMOTION_NAMES[preds[idx]],
                [f"{v:.3f}" for v in lgts],
            )

    log.info("=" * 60)
    log.info("Target accuracy range for 4-class eye-feature emotion: 60-72%%")
    log.info("Achieved: %.1f%%", 100.0 * test_acc)
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a 4-class emotion MLP on VREED eye-tracking features. "
            "Saves emotion_model.pth and emotion_scaler.pkl to saved_models/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Project root directory (contains '04 Eye Tracking Data/' subdirectory).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maximum number of training epochs (early stopping may stop earlier).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Initial Adam learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="L2 weight decay for Adam optimiser.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Directory to write saved_models/ artefacts. Defaults to <data_path>/saved_models/.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging and per-sample logit output for test set.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the full training pipeline.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).
    """
    parser = _build_arg_parser()
    args   = parser.parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

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
            epochs       = args.epochs,
            lr           = args.lr,
            batch_size   = args.batch_size,
            weight_decay = args.weight_decay,
            debug        = args.debug,
        )
    except (FileNotFoundError, KeyError) as exc:
        log.error("Setup error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected error during training: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
