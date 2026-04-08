"""
train_attention.py
────────────────────────────────────────────────────────────────────────────
Train a 3-class attention classifier (focused / distracted / off_task)
on OpenEDS eye images using MobileNetV2 with a frozen pretrained backbone.

Labels are derived from the segmentation masks by attention_label_generator.py.
Run that script first, or pass --generate_labels to run it automatically.

Saved artefacts:
    saved_models/attention_model.pth          <- after OpenEDS training
    saved_models/attention_model_finetuned.pth <- after webcam fine-tune
                                                  (only if webcam data found)

Run standalone:
    python models/train_attention.py \\
        --data_path "C:/Users/kunal/Desktop/Eye tracking Module" \\
        --epochs 20

Domain adaptation note:
    OpenEDS images are VR infrared at 400x640.  The model is trained on
    pseudo-labels derived from pixel-level segmentation geometry.
    At inference time the model sees RGB webcam images at 30 fps.
    The mandatory domain adaptation step is the webcam fine-tuning:

        python tools/collect_webcam_samples.py   # collect 200 labelled frames
        python models/train_attention.py \\
            --data_path ... --webcam_finetune_only

    This 10-epoch fine-tune of just the classifier head (last Dense layers)
    is what actually makes the attention model work reliably on a webcam.
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_attention")

# ── Label definitions ────────────────────────────────────────────────────────
ATTENTION_LABELS  = ["focused", "distracted", "off_task"]
LABEL_TO_INT      = {l: i for i, l in enumerate(ATTENTION_LABELS)}
NUM_CLASSES       = 3

# ── Dataset paths ────────────────────────────────────────────────────────────
LABELS_CSV_SUBPATH   = ("data", "attention_labels_all.csv")
WEBCAM_FINETUNE_DIR  = "data/webcam_finetune"
OPENEDS_IMAGES_ROOT  = ("openEDS", "openEDS")

# ── Image dimensions ─────────────────────────────────────────────────────────
IMG_SIZE             = 64    # model input: 64 x 64

# ── Architecture ─────────────────────────────────────────────────────────────
UNFREEZE_FROM_IDX    = 16    # unfreeze features[16], features[17], features[18]
CLASSIFIER_HIDDEN    = 128
CLASSIFIER_DROPOUT   = 0.4
MOBILENET_FEAT_DIM   = 1280  # output channels of MobileNetV2 features

# ── Training defaults ─────────────────────────────────────────────────────────
DEFAULT_EPOCHS         = 20
DEFAULT_LR             = 1e-4
DEFAULT_WEIGHT_DECAY   = 1e-4
DEFAULT_BATCH_SIZE     = 64
EARLY_STOP_PATIENCE    = 7
RANDOM_SEED            = 42

# ── Webcam fine-tune defaults ─────────────────────────────────────────────────
WEBCAM_EPOCHS          = 10
WEBCAM_LR              = 5e-4
WEBCAM_BATCH_SIZE      = 16
WEBCAM_MIN_SAMPLES     = 10   # skip fine-tune if fewer samples exist

# ── Artefact filenames ────────────────────────────────────────────────────────
MODEL_FILENAME         = "attention_model.pth"
MODEL_FINETUNED_FNAME  = "attention_model_finetuned.pth"

# ── ImageNet normalisation (used with pretrained MobileNetV2) ─────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AttentionDataset(Dataset):
    """PyTorch Dataset for OpenEDS eye images with pseudo attention labels.

    Each item is a (image_tensor, label_int) pair.  Images are loaded as
    grayscale, converted to 3-channel RGB by repeating the channel, then
    resized to 64x64 and normalised with ImageNet statistics.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: transforms.Compose,
        debug: bool = False,
    ) -> None:
        """Initialise the dataset from a labels DataFrame.

        Args:
            df: DataFrame with columns 'image_path' and 'attention_label'.
            transform: torchvision transform pipeline applied to each image.
            debug: If True, log every loaded image path.
        """
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.debug     = debug

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and transform one (image, label) pair.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor of shape (3, 64, 64), label_int).

        Raises:
            FileNotFoundError: if the image file does not exist on disk.
        """
        row   = self.df.iloc[idx]
        path  = Path(row["image_path"])
        label = LABEL_TO_INT[row["attention_label"]]

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Load as grayscale, convert to RGB (repeats channel x3)
        img = Image.open(path).convert("RGB")
        if self.debug:
            log.debug("Loaded %s  label=%s", path.name, row["attention_label"])

        return self.transform(img), label


class WebcamDataset(Dataset):
    """Dataset for webcam fine-tune samples stored as label-named subdirectories.

    Expected structure:
        data/webcam_finetune/focused/001.png
        data/webcam_finetune/distracted/002.png
        data/webcam_finetune/off_task/003.png
    """

    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        """Scan the label subdirectories and build a flat sample list.

        Args:
            root: Root directory of the webcam fine-tune data.
            transform: torchvision transform pipeline.

        Raises:
            FileNotFoundError: if root does not exist.
        """
        if not root.exists():
            raise FileNotFoundError(f"Webcam finetune directory not found: {root}")

        self.samples: list[tuple[Path, int]] = []
        self.transform = transform

        for label_name, label_int in LABEL_TO_INT.items():
            label_dir = root / label_name
            if not label_dir.exists():
                log.warning("Webcam finetune: no directory for class '%s'", label_name)
                continue
            imgs = sorted(label_dir.glob("*.png")) + sorted(label_dir.glob("*.jpg"))
            for p in imgs:
                self.samples.append((p, label_int))
            log.info("Webcam finetune: %s: %d images", label_name, len(imgs))

    def __len__(self) -> int:
        """Return the total number of webcam fine-tune samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and transform one sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label_int).
        """
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def make_transforms(augment: bool = False) -> transforms.Compose:
    """Build the image transform pipeline.

    Args:
        augment: If True, apply random horizontal flip and colour jitter
                 (used for training only).

    Returns:
        Composed transform pipeline.
    """
    ops: list = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build a MobileNetV2 with a custom 3-class head.

    Strategy:
      - Load MobileNetV2 pretrained on ImageNet.
      - Freeze features[0] through features[UNFREEZE_FROM_IDX - 1].
      - Leave features[UNFREEZE_FROM_IDX:] and the new classifier trainable.
      - Replace the classifier with:
            Linear(1280, 128) -> ReLU -> Dropout(0.4) -> Linear(128, 3)

    Args:
        num_classes: Number of output classes.

    Returns:
        Modified MobileNetV2 model with custom head.
    """
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze entire backbone first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 2 InvertedResidual blocks + final Conv2dNormActivation
    for layer in model.features[UNFREEZE_FROM_IDX:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Linear(MOBILENET_FEAT_DIM, CLASSIFIER_HIDDEN),
        nn.ReLU(inplace=True),
        nn.Dropout(p=CLASSIFIER_DROPOUT),
        nn.Linear(CLASSIFIER_HIDDEN, num_classes),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(
        "MobileNetV2 built: total params=%d  trainable=%d  frozen=%d",
        total, trainable, total - trainable,
    )
    return model


def freeze_backbone_for_finetune(model: nn.Module) -> None:
    """Freeze the entire backbone, leaving only the classifier trainable.

    Used for webcam domain adaptation fine-tuning so only the final
    Dense layers are updated on the small (<200 sample) webcam dataset.

    Args:
        model: The attention model (must have a .features and .classifier).
    """
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Webcam finetune mode: backbone frozen, %d classifier params trainable.", trainable)


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = EARLY_STOP_PATIENCE, min_delta: float = 1e-5) -> None:
        """Initialise.

        Args:
            patience: Epochs to wait after last improvement.
            min_delta: Minimum change to count as improvement.
        """
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Update with current val_loss; snapshot best weights.

        Args:
            val_loss: Validation loss for this epoch.
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
        """Restore the best-seen weights.

        Args:
            model: Model to restore.

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
        model: The attention model.
        loader: Training DataLoader.
        criterion: Weighted cross-entropy loss.
        optimiser: Gradient optimiser.
        device: Compute device.

    Returns:
        Tuple of (mean_loss, accuracy).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimiser.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model for one epoch without gradient computation.

    Args:
        model: The attention model.
        loader: Validation or test DataLoader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of (mean_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect all predictions and ground-truth labels for evaluation.

    Args:
        model: The attention model.
        loader: DataLoader (test split).
        device: Compute device.

    Returns:
        Tuple of (preds_array, labels_array) as numpy int arrays.
    """
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_labels_csv(
    data_path: Path,
    generate_if_missing: bool = False,
) -> pd.DataFrame:
    """Load the attention labels CSV generated by attention_label_generator.py.

    Args:
        data_path: Project root directory.
        generate_if_missing: If True, auto-run attention_label_generator to
                             create the CSV when it is absent.

    Returns:
        Combined labels DataFrame (all splits).

    Raises:
        FileNotFoundError: if the CSV is absent and generate_if_missing=False.
    """
    csv_path = Path(data_path, *LABELS_CSV_SUBPATH)

    if not csv_path.exists():
        if generate_if_missing:
            log.info("Labels CSV not found -- running attention_label_generator...")
            sys.path.insert(0, str(Path(__file__).parent))
            from attention_label_generator import generate_labels  # noqa: PLC0415
            generate_labels(
                data_path   = data_path,
                output_path = data_path / "data",
            )
        else:
            raise FileNotFoundError(
                f"Labels CSV not found: {csv_path}\n"
                "Run: python models/attention_label_generator.py "
                "--data_path <project_root>\n"
                "Or pass --generate_labels to this script."
            )

    df = pd.read_csv(csv_path)
    log.info("Loaded labels CSV: %d rows  path=%s", len(df), csv_path)
    log.info(
        "Label distribution: %s",
        df["attention_label"].value_counts().to_dict(),
    )
    return df


def get_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute balanced class weights for weighted cross-entropy loss.

    Args:
        labels: Integer label array for the training split.
        device: Target device for the weight tensor.

    Returns:
        Float tensor of shape (NUM_CLASSES,) on the specified device.
    """
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    log.info(
        "Class weights (balanced): %s",
        {ATTENTION_LABELS[i]: round(float(w), 4) for i, w in enumerate(weights)},
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Webcam fine-tune
# ─────────────────────────────────────────────────────────────────────────────

def webcam_finetune(
    model: nn.Module,
    data_path: Path,
    output_path: Path,
    device: torch.device,
    epochs: int = WEBCAM_EPOCHS,
    lr: float = WEBCAM_LR,
    batch_size: int = WEBCAM_BATCH_SIZE,
) -> bool:
    """Fine-tune only the classifier head on webcam-collected labelled frames.

    Loads samples from data/webcam_finetune/ and trains the top Dense
    layers only, with the entire MobileNetV2 backbone frozen.  Saves the
    fine-tuned checkpoint to attention_model_finetuned.pth.

    Args:
        model: Pre-trained attention model (from OpenEDS training).
        data_path: Project root directory.
        output_path: Directory to save the fine-tuned checkpoint.
        device: Compute device.
        epochs: Number of fine-tuning epochs.
        lr: Learning rate for the classifier.
        batch_size: Mini-batch size.

    Returns:
        True if fine-tuning was performed, False if skipped (not enough data).
    """
    webcam_root = data_path / WEBCAM_FINETUNE_DIR
    if not webcam_root.exists():
        log.info("Webcam finetune dir not found (%s) -- skipping.", webcam_root)
        return False

    tf = make_transforms(augment=True)
    try:
        webcam_ds = WebcamDataset(webcam_root, tf)
    except FileNotFoundError as exc:
        log.warning("Webcam finetune skipped: %s", exc)
        return False

    if len(webcam_ds) < WEBCAM_MIN_SAMPLES:
        log.warning(
            "Only %d webcam samples found (min=%d) -- skipping fine-tune.",
            len(webcam_ds), WEBCAM_MIN_SAMPLES,
        )
        return False

    log.info("Starting webcam fine-tune: %d samples  epochs=%d  lr=%.5f",
             len(webcam_ds), epochs, lr)

    loader = DataLoader(webcam_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    freeze_backbone_for_finetune(model)
    model.to(device)

    labels_arr = np.array([s[1] for s in webcam_ds.samples])
    weights    = get_class_weights(labels_arr, device)
    criterion  = nn.CrossEntropyLoss(weight=weights)
    optimiser  = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, loader, criterion, optimiser, device)
        log.info(
            "Finetune epoch %2d/%d  loss=%.4f  acc=%.3f",
            epoch, epochs, tr_loss, tr_acc,
        )

    finetune_path = output_path / MODEL_FINETUNED_FNAME
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_names": ATTENTION_LABELS,
            "num_classes": NUM_CLASSES,
            "finetuned_on_webcam": True,
            "webcam_samples": len(webcam_ds),
        },
        finetune_path,
    )
    log.info("Fine-tuned model saved -> %s", finetune_path)
    return True


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
    webcam_only: bool = False,
    debug: bool = False,
) -> None:
    """Full training pipeline: load labels -> datasets -> train -> evaluate -> save.

    Args:
        data_path: Project root directory.
        output_path: Directory to write model checkpoints.
        epochs: Maximum training epochs.
        lr: Initial Adam learning rate.
        batch_size: Mini-batch size for OpenEDS training.
        weight_decay: Adam L2 regularisation.
        generate_labels: If True, auto-run attention_label_generator if CSV missing.
        webcam_only: If True, skip OpenEDS training and only run webcam fine-tune
                     on a previously saved attention_model.pth.
        debug: If True, set logging to DEBUG level.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model()

    if webcam_only:
        # Load existing OpenEDS checkpoint, then fine-tune on webcam
        checkpoint_path = output_path / MODEL_FILENAME
        if not checkpoint_path.exists():
            log.error(
                "--webcam_finetune_only requires %s to exist. Train first.",
                checkpoint_path,
            )
            sys.exit(1)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("Loaded OpenEDS checkpoint from %s", checkpoint_path)
        model.to(device)
        webcam_finetune(model, data_path, output_path, device)
        return

    # ── Load labels CSV ───────────────────────────────────────────────────────
    df_all = load_labels_csv(data_path, generate_if_missing=generate_labels)

    df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
    df_val   = df_all[df_all["split"] == "validation"].reset_index(drop=True)
    df_test  = df_all[df_all["split"] == "test"].reset_index(drop=True)

    log.info(
        "Split sizes: train=%d  val=%d  test=%d",
        len(df_train), len(df_val), len(df_test),
    )

    # ── Datasets and loaders ──────────────────────────────────────────────────
    tf_train = make_transforms(augment=True)
    tf_eval  = make_transforms(augment=False)

    ds_train  = AttentionDataset(df_train, tf_train, debug=debug)
    ds_val    = AttentionDataset(df_val,   tf_eval)
    ds_test   = AttentionDataset(df_test,  tf_eval)

    num_workers = min(4, torch.get_num_threads())
    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda", drop_last=False,
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )

    # ── Loss with class weights ───────────────────────────────────────────────
    train_labels = np.array([LABEL_TO_INT[l] for l in df_train["attention_label"]])
    weights      = get_class_weights(train_labels, device)
    criterion    = nn.CrossEntropyLoss(weight=weights)

    # ── Optimiser and scheduler ───────────────────────────────────────────────
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.01,
    )
    stopper   = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info(
        "Training MobileNetV2 attention model: epochs=%d  lr=%.5f  batch=%d",
        epochs, lr, batch_size,
    )
    log.info("=" * 65)

    best_epoch = 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimiser, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step()

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

    # ── Save OpenEDS checkpoint ───────────────────────────────────────────────
    model_path = output_path / MODEL_FILENAME
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_names": ATTENTION_LABELS,
            "num_classes": NUM_CLASSES,
            "best_val_loss": stopper.best_loss,
            "best_epoch": best_epoch,
            "finetuned_on_webcam": False,
        },
        model_path,
    )
    log.info("Model saved -> %s", model_path)

    # ── Test set evaluation ───────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("FINAL EVALUATION ON TEST SET")
    log.info("=" * 65)

    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    log.info("Test loss=%.4f  Test accuracy=%.4f  (%d samples)", test_loss, test_acc, len(ds_test))

    preds, labels = collect_predictions(model, test_loader, device)
    log.info(
        "Classification report:\n%s",
        classification_report(labels, preds, target_names=ATTENTION_LABELS, zero_division=0),
    )

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    log.info("Confusion matrix (rows=true, cols=pred):")
    header = "           " + "  ".join(f"pred_{ATTENTION_LABELS[i][:4]:>6}" for i in range(NUM_CLASSES))
    log.info(header)
    for i, row in enumerate(cm):
        log.info("true_%-10s  %s", ATTENTION_LABELS[i], "  ".join(f"{v:>10}" for v in row))

    log.info("=" * 65)
    log.info("Target (before webcam finetune): 65-75%%")
    log.info("Target (after webcam finetune) : 75-82%%")
    log.info("Achieved on OpenEDS test set   : %.1f%%", 100.0 * test_acc)
    log.info("=" * 65)

    # ── Optional webcam fine-tune ─────────────────────────────────────────────
    finetuned = webcam_finetune(model, data_path, output_path, device)
    if finetuned:
        log.info("Webcam fine-tune complete. Use attention_model_finetuned.pth for inference.")
    else:
        log.info(
            "No webcam fine-tune data found.  To improve accuracy on your webcam:\n"
            "  1. Run: python tools/collect_webcam_samples.py --data_path <root>\n"
            "  2. Re-run this script with --webcam_finetune_only"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a 3-class attention CNN (MobileNetV2) on OpenEDS pseudo-labelled images."
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
        help="Initial Adam learning rate.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY,
        help="L2 weight decay for Adam.",
    )
    parser.add_argument(
        "--output_path", type=Path, default=None,
        help="Artefact output directory. Defaults to <data_path>/saved_models/.",
    )
    parser.add_argument(
        "--generate_labels", action="store_true",
        help="Auto-run attention_label_generator.py if labels CSV is missing.",
    )
    parser.add_argument(
        "--webcam_finetune_only", action="store_true",
        help=(
            "Skip OpenEDS training. Load saved attention_model.pth and "
            "run webcam fine-tune only."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run training.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).
    """
    parser  = _build_arg_parser()
    args    = parser.parse_args(argv)

    data_path   = args.data_path.resolve()
    output_path = (args.output_path or data_path / "saved_models").resolve()

    if not data_path.exists():
        log.error("data_path does not exist: %s", data_path)
        sys.exit(1)

    log.info("data_path   = %s", data_path)
    log.info("output_path = %s", output_path)

    try:
        train(
            data_path      = data_path,
            output_path    = output_path,
            epochs         = args.epochs,
            lr             = args.lr,
            batch_size     = args.batch_size,
            weight_decay   = args.weight_decay,
            generate_labels= args.generate_labels,
            webcam_only    = args.webcam_finetune_only,
            debug          = args.debug,
        )
    except FileNotFoundError as exc:
        log.error("Dataset not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
