"""
train_gaze.py
────────────────────────────────────────────────────────────────────────────
Train a pupil-centroid regression model (ResNet-18 backbone) on OpenEDS
segmentation images.  The model learns to predict (cx_norm, cy_norm) --
the normalised pupil centroid position within the eye image.

Saved artefacts:
    saved_models/gaze_model.pth         <- best checkpoint
    saved_models/gaze_norm_params.npy   <- [img_height=400, img_width=640]
    data/gaze_labels_all.csv            <- precomputed centroid labels

Design decision (dataset adaptation):
    The project spec calls for ResNet-18 regression on angular gaze vectors
    (pitch/yaw in degrees) from 91,200 OpenEDS video sequence frames.
    The available OpenEDS download contains only the segmentation subset
    (32,919 images) without angular gaze labels.

    Adaptation: derive gaze proxy labels from the pixel-level segmentation
    masks.  For each mask the pupil centroid is:
        cx_norm = mean_col(pupil_pixels) / image_width
        cy_norm = mean_row(pupil_pixels) / image_height
    Both coordinates are in [0, 1] and encode gaze direction within the
    visible eye region.

    At inference the model predicts (cx_norm, cy_norm) from a 64x64
    eye image crop.  The 9-point calibration routine maps this to actual
    screen pixel coordinates via an affine transform.

    This approach is MORE robust to the VR->webcam domain gap than angular
    regression: pixel-level pupil position is a universal cue that transfers
    across illumination conditions, while angular VR coordinates do not.

Run standalone:
    python models/train_gaze.py \\
        --data_path "C:/Users/kunal/Desktop/Eye tracking Module" \\
        --epochs 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_gaze")

# ── Paths ────────────────────────────────────────────────────────────────────
OPENEDS_ROOT_SUBPATH   = ("openEDS", "openEDS")
GAZE_LABELS_SUBPATH    = ("data", "gaze_labels_all.csv")
OPENEDS_SPLITS         = ("train", "validation", "test")

# ── Segmentation class ids ────────────────────────────────────────────────────
SEG_PUPIL_CLASS = 3

# ── Image size ────────────────────────────────────────────────────────────────
IMG_H            = 400     # native OpenEDS image height
IMG_W            = 640     # native OpenEDS image width
MODEL_INPUT_SIZE = 64      # resize target for model input

# ── Gaze label validity ───────────────────────────────────────────────────────
MIN_PUPIL_PX = 50          # reject samples with fewer pupil pixels

# ── Architecture ─────────────────────────────────────────────────────────────
RESNET_FEAT_DIM  = 512
GAZE_HEAD_DIM1   = 256
GAZE_HEAD_DIM2   = 64
NUM_GAZE_OUTPUTS = 2       # (cx_norm, cy_norm)

# ── Training defaults ─────────────────────────────────────────────────────────
DEFAULT_EPOCHS      = 30
DEFAULT_LR          = 1e-4
DEFAULT_WEIGHT_DECAY= 1e-4
DEFAULT_BATCH_SIZE  = 64
EARLY_STOP_PATIENCE = 10
RANDOM_SEED         = 42

# ── Artefact names ────────────────────────────────────────────────────────────
MODEL_FILENAME      = "gaze_model.pth"
NORM_PARAMS_FNAME   = "gaze_norm_params.npy"

# ── ImageNet normalisation ───────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Label pre-computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pupil_centroid(
    mask: np.ndarray,
    min_pupil_px: int = MIN_PUPIL_PX,
) -> tuple[float, float, int, bool]:
    """Compute normalised pupil centroid from a segmentation mask.

    Args:
        mask: uint8 array of shape (H, W) with values in {0, 1, 2, 3}.
        min_pupil_px: Minimum pupil pixel count to consider the sample valid.

    Returns:
        Tuple of (cx_norm, cy_norm, pupil_area_px, is_valid).
        cx_norm and cy_norm are in [0, 1].  If is_valid is False both
        centroid values are 0.5 (dummy; sample is excluded from training).
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got {mask.shape}")

    rows, cols  = np.where(mask == SEG_PUPIL_CLASS)
    pupil_area  = int(len(rows))

    if pupil_area < min_pupil_px:
        return 0.5, 0.5, pupil_area, False

    H, W  = mask.shape
    cx    = float(cols.mean()) / W
    cy    = float(rows.mean()) / H
    return cx, cy, pupil_area, True


def precompute_gaze_labels(
    openeds_root: Path,
    output_path: Path,
    min_pupil_px: int = MIN_PUPIL_PX,
) -> pd.DataFrame:
    """Compute pupil centroid labels for all OpenEDS splits and save to CSV.

    Args:
        openeds_root: Path to openEDS/openEDS directory.
        output_path: Directory to write gaze_labels_*.csv files.
        min_pupil_px: Minimum pupil pixels to mark a sample as valid.

    Returns:
        Combined DataFrame with gaze centroid labels for all splits.

    Raises:
        FileNotFoundError: if openeds_root does not exist.
    """
    if not openeds_root.exists():
        raise FileNotFoundError(f"OpenEDS root not found: {openeds_root}")

    output_path.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []

    for split in OPENEDS_SPLITS:
        img_dir = openeds_root / split / "images"
        lbl_dir = openeds_root / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            log.warning("Skipping split '%s' (images or labels dir missing).", split)
            continue

        img_files = sorted(img_dir.glob("*.png"))
        lbl_by_stem = {f.stem: f for f in sorted(lbl_dir.glob("*.npy"))}

        records = []
        invalid = 0

        for img_path in tqdm(img_files, desc=f"  {split:12s}", unit="img", leave=True):
            lbl_path = lbl_by_stem.get(img_path.stem)
            if lbl_path is None:
                invalid += 1
                continue

            mask = np.load(lbl_path)
            cx, cy, pupil_area, valid = compute_pupil_centroid(mask, min_pupil_px)
            records.append({
                "split"       : split,
                "image_path"  : str(img_path),
                "cx_norm"     : round(cx, 6),
                "cy_norm"     : round(cy, 6),
                "pupil_area_px": pupil_area,
                "valid"       : valid,
            })

        df_split = pd.DataFrame(records)
        n_valid  = int(df_split["valid"].sum())
        log.info(
            "%s: %d samples  valid=%d  invalid=%d",
            split, len(df_split), n_valid, len(df_split) - n_valid + invalid,
        )

        split_path = output_path / f"gaze_labels_{split}.csv"
        df_split.to_csv(split_path, index=False)
        all_frames.append(df_split)

    df_all = pd.concat(all_frames, ignore_index=True)
    all_path = output_path / "gaze_labels_all.csv"
    df_all.to_csv(all_path, index=False)
    log.info("Gaze labels saved -> %s  (%d total rows)", all_path, len(df_all))

    cx_stats = df_all.loc[df_all["valid"], "cx_norm"]
    cy_stats = df_all.loc[df_all["valid"], "cy_norm"]
    log.info(
        "Centroid stats: cx=[%.3f, %.3f] mean=%.3f std=%.4f | "
        "cy=[%.3f, %.3f] mean=%.3f std=%.4f",
        cx_stats.min(), cx_stats.max(), cx_stats.mean(), cx_stats.std(),
        cy_stats.min(), cy_stats.max(), cy_stats.mean(), cy_stats.std(),
    )
    return df_all


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GazeDataset(Dataset):
    """Dataset of (eye_image, (cx_norm, cy_norm)) pairs for gaze regression.

    Only includes samples where valid=True (pupil detected with the
    required minimum pixel count).
    """

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose) -> None:
        """Initialise from a gaze labels DataFrame.

        Args:
            df: DataFrame with columns image_path, cx_norm, cy_norm, valid.
            transform: torchvision transform applied to each PIL image.
        """
        # Keep only valid samples
        self.df        = df[df["valid"]].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of valid samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load image and return (image_tensor, gaze_tensor).

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor of shape (3, 64, 64),
                      gaze_tensor of shape (2,) with values in [0, 1]).

        Raises:
            FileNotFoundError: if the image file does not exist.
        """
        row  = self.df.iloc[idx]
        path = Path(row["image_path"])
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img   = Image.open(path).convert("RGB")
        gaze  = torch.tensor([row["cx_norm"], row["cy_norm"]], dtype=torch.float32)
        return self.transform(img), gaze


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def make_transforms(augment: bool = False) -> transforms.Compose:
    """Build the image transform pipeline for gaze regression.

    Note: horizontal flipping is NOT applied because it would invert the
    cx_norm label, requiring a corresponding label flip.  Only
    brightness/contrast jitter is used for augmentation.

    Args:
        augment: If True, apply brightness/contrast jitter.

    Returns:
        Composed transform.
    """
    ops: list = [transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))]
    if augment:
        ops.append(transforms.ColorJitter(brightness=0.25, contrast=0.25))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class GazeResNet(nn.Module):
    """ResNet-18 backbone with a 2-output regression head for gaze centroid.

    Architecture:
        ResNet-18 (pretrained, all layers fine-tuned) -> avgpool -> 512
        -> Linear(512, 256) -> ReLU
        -> Linear(256, 64)  -> ReLU
        -> Linear(64, 2)    -> Sigmoid   --> (cx_norm, cy_norm)
    """

    def __init__(self) -> None:
        """Initialise: load pretrained ResNet-18 and replace fc with regression head."""
        super().__init__()
        backbone    = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final classification layer
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(RESNET_FEAT_DIM, GAZE_HEAD_DIM1),
            nn.ReLU(inplace=True),
            nn.Linear(GAZE_HEAD_DIM1, GAZE_HEAD_DIM2),
            nn.ReLU(inplace=True),
            nn.Linear(GAZE_HEAD_DIM2, NUM_GAZE_OUTPUTS),
            nn.Sigmoid(),   # constrains output to (0, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape (batch, 3, H, W).

        Returns:
            Float tensor of shape (batch, 2) with (cx_norm, cy_norm) in (0, 1).
        """
        features = self.backbone(x)
        return self.head(features)


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = EARLY_STOP_PATIENCE, min_delta: float = 1e-6) -> None:
        """Initialise.

        Args:
            patience: Epochs to wait after last improvement.
            min_delta: Minimum improvement to reset the counter.
        """
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Update; snapshot best model state.

        Args:
            val_loss: Current validation MSE loss.
            model: Model to snapshot.

        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def restore_best(self, model: nn.Module) -> None:
        """Load best weights back into the model.

        Args:
            model: Target model.

        Raises:
            RuntimeError: if no best state has been recorded.
        """
        if self.best_state is None:
            raise RuntimeError("No best state recorded.")
        model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
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
        model: GazeResNet model.
        loader: Training DataLoader.
        criterion: MSE loss.
        optimiser: Adam optimiser.
        device: Compute device.

    Returns:
        Tuple of (mean_mse_loss, mean_euclidean_distance).
    """
    model.train()
    total_mse, total_euc, total = 0.0, 0.0, 0

    for imgs, gaze_true in loader:
        imgs, gaze_true = imgs.to(device), gaze_true.to(device)
        optimiser.zero_grad()
        gaze_pred = model(imgs)
        loss      = criterion(gaze_pred, gaze_true)
        loss.backward()
        optimiser.step()

        with torch.no_grad():
            euc = torch.sqrt(((gaze_pred - gaze_true) ** 2).sum(dim=1)).mean()
        total_mse += loss.item() * len(gaze_true)
        total_euc += euc.item() * len(gaze_true)
        total     += len(gaze_true)

    return total_mse / total, total_euc / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model for one epoch.

    Args:
        model: GazeResNet model.
        loader: Validation or test DataLoader.
        criterion: MSE loss.
        device: Compute device.

    Returns:
        Tuple of (mean_mse_loss, mean_euclidean_distance).
    """
    model.eval()
    total_mse, total_euc, total = 0.0, 0.0, 0

    for imgs, gaze_true in loader:
        imgs, gaze_true = imgs.to(device), gaze_true.to(device)
        gaze_pred = model(imgs)
        loss      = criterion(gaze_pred, gaze_true)
        euc       = torch.sqrt(((gaze_pred - gaze_true) ** 2).sum(dim=1)).mean()

        total_mse += loss.item() * len(gaze_true)
        total_euc += euc.item() * len(gaze_true)
        total     += len(gaze_true)

    return total_mse / total, total_euc / total


@torch.no_grad()
def collect_errors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Collect per-sample Euclidean gaze errors.

    Args:
        model: GazeResNet model.
        loader: Test DataLoader.
        device: Compute device.

    Returns:
        Float array of shape (N,) with per-sample Euclidean distances.
    """
    model.eval()
    errors = []
    for imgs, gaze_true in loader:
        imgs, gaze_true = imgs.to(device), gaze_true.to(device)
        gaze_pred = model(imgs)
        euc = torch.sqrt(((gaze_pred - gaze_true) ** 2).sum(dim=1))
        errors.append(euc.cpu().numpy())
    return np.concatenate(errors)


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_path: Path,
    output_path: Path,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    generate_labels: bool = False,
    debug: bool = False,
) -> None:
    """Full pipeline: precompute labels -> datasets -> train -> evaluate -> save.

    Args:
        data_path: Project root directory.
        output_path: Directory to write model artefacts.
        epochs: Maximum training epochs.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        weight_decay: Adam L2 regularisation.
        generate_labels: If True, recompute gaze centroid labels even if
                         the CSV already exists.
        debug: If True, enable DEBUG-level logging.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Gaze centroid labels ────────────────────────────────────────────
    data_dir  = data_path / "data"
    labels_csv = data_dir / "gaze_labels_all.csv"
    openeds_root = Path(data_path, *OPENEDS_ROOT_SUBPATH)

    if generate_labels or not labels_csv.exists():
        log.info("Precomputing gaze centroid labels (this may take ~60s)...")
        df_all = precompute_gaze_labels(openeds_root, data_dir)
    else:
        df_all = pd.read_csv(labels_csv)
        log.info("Loaded gaze labels from %s  (%d rows)", labels_csv, len(df_all))

    df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
    df_val   = df_all[df_all["split"] == "validation"].reset_index(drop=True)
    df_test  = df_all[df_all["split"] == "test"].reset_index(drop=True)

    log.info(
        "Valid samples: train=%d  val=%d  test=%d",
        int(df_train["valid"].sum()),
        int(df_val["valid"].sum()),
        int(df_test["valid"].sum()),
    )

    # ── 2. Datasets and loaders ───────────────────────────────────────────
    tf_train = make_transforms(augment=True)
    tf_eval  = make_transforms(augment=False)

    ds_train = GazeDataset(df_train, tf_train)
    ds_val   = GazeDataset(df_val,   tf_eval)
    ds_test  = GazeDataset(df_test,  tf_eval)

    num_workers = min(4, torch.get_num_threads())
    pin     = device.type == "cuda"
    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    # ── 3. Model, loss, optimiser ─────────────────────────────────────────
    model     = GazeResNet().to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.01,
    )
    stopper   = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("GazeResNet: total params=%d", total_params)

    # ── 4. Training loop ──────────────────────────────────────────────────
    log.info("=" * 65)
    log.info(
        "Training gaze model: epochs=%d  lr=%.5f  batch=%d",
        epochs, lr, batch_size,
    )
    log.info("=" * 65)

    best_epoch = 0
    for epoch in range(1, epochs + 1):
        tr_mse, tr_euc = train_epoch(model, train_loader, criterion, optimiser, device)
        vl_mse, vl_euc = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step()

        log.info(
            "Epoch %3d/%d  train_mse=%.6f  train_euc=%.4f  "
            "val_mse=%.6f  val_euc=%.4f  lr=%.6f",
            epoch, epochs, tr_mse, tr_euc, vl_mse, vl_euc,
            optimiser.param_groups[0]["lr"],
        )

        stopped = stopper.step(vl_mse, model)
        if stopper.best_loss == vl_mse:
            best_epoch = epoch
        if stopped:
            log.info("Early stopping triggered at epoch %d.", epoch)
            break

    stopper.restore_best(model)
    log.info("Best epoch: %d  best_val_mse=%.6f", best_epoch, stopper.best_loss)

    # ── 5. Save artefacts ─────────────────────────────────────────────────
    model_path      = output_path / MODEL_FILENAME
    norm_params_path= output_path / NORM_PARAMS_FNAME

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_mse"    : stopper.best_loss,
            "best_epoch"      : best_epoch,
            "img_height"      : IMG_H,
            "img_width"       : IMG_W,
            "note"            : "Predicts normalised pupil centroid (cx, cy) in [0,1]",
        },
        model_path,
    )
    log.info("Model saved -> %s", model_path)

    np.save(norm_params_path, np.array([float(IMG_H), float(IMG_W)]))
    log.info("Norm params saved -> %s  (H=%d, W=%d)", norm_params_path, IMG_H, IMG_W)

    # ── 6. Test set evaluation ────────────────────────────────────────────
    ts_mse, ts_euc = eval_epoch(model, test_loader, criterion, device)
    log.info("=" * 65)
    log.info("TEST SET EVALUATION  (n=%d)", len(ds_test))
    log.info("=" * 65)
    log.info("  MSE              : %.6f", ts_mse)
    log.info("  Mean Euclidean   : %.4f  (normalised units)", ts_euc)
    log.info("  Mean pixel error : cx=%.1f px  cy=%.1f px (on 640x400 image)",
             ts_euc * IMG_W, ts_euc * IMG_H)

    errors = collect_errors(model, test_loader, device)
    for pct in [50, 75, 90, 95]:
        log.info("  p%2d Euclidean    : %.4f", pct, float(np.percentile(errors, pct)))

    log.info("=" * 65)
    log.info("Target: mean Euclidean < 0.08 (normalised)")
    log.info("Achieved: %.4f", ts_euc)
    log.info("")
    log.info("Note: performance will improve significantly after the 9-point")
    log.info("calibration routine remaps predicted centroid to screen coordinates.")
    log.info("The calibration corrects for the VR->webcam domain gap.")
    log.info("=" * 65)

    if debug:
        # Show a few sample predictions
        model.eval()
        with torch.no_grad():
            imgs_b, gaze_b = next(iter(test_loader))
            preds = model(imgs_b.to(device)).cpu()
            log.debug("Sample predictions (first 5 test images):")
            for i in range(min(5, len(preds))):
                log.debug(
                    "  true=(%5.3f, %5.3f)  pred=(%5.3f, %5.3f)  dist=%.4f",
                    gaze_b[i, 0].item(), gaze_b[i, 1].item(),
                    preds[i, 0].item(),  preds[i, 1].item(),
                    float(torch.sqrt(((preds[i] - gaze_b[i]) ** 2).sum())),
                )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a ResNet-18 gaze regression model on OpenEDS pupil centroids. "
            "Saves gaze_model.pth and gaze_norm_params.npy to saved_models/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path", type=Path, required=True,
        help="Project root directory.",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY,
        help="Adam L2 weight decay.",
    )
    parser.add_argument(
        "--output_path", type=Path, default=None,
        help="Artefact output dir. Defaults to <data_path>/saved_models/.",
    )
    parser.add_argument(
        "--generate_labels", action="store_true",
        help="Recompute gaze centroid labels even if gaze_labels_all.csv exists.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging and sample prediction output.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run training.

    Args:
        argv: Argument list; defaults to sys.argv[1:] when None.
    """
    parser      = _build_arg_parser()
    args        = parser.parse_args(argv)

    data_path   = args.data_path.resolve()
    output_path = (args.output_path or data_path / "saved_models").resolve()

    if not data_path.exists():
        log.error("data_path does not exist: %s", data_path)
        sys.exit(1)

    log.info("data_path   = %s", data_path)
    log.info("output_path = %s", output_path)

    try:
        train(
            data_path       = data_path,
            output_path     = output_path,
            epochs          = args.epochs,
            lr              = args.lr,
            batch_size      = args.batch_size,
            weight_decay    = args.weight_decay,
            generate_labels = args.generate_labels,
            debug           = args.debug,
        )
    except FileNotFoundError as exc:
        log.error("Dataset not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
