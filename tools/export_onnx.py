"""tools/export_onnx.py
--------------------------------------------------------------------------------
Export the two CNN inference models to ONNX format.

ONNX models infer faster than PyTorch (.pth) at runtime, especially on CPU,
because ONNX Runtime applies additional graph optimisations.

Exported files (written to saved_models/):
    attention_model.onnx    <- MobileNetV2 attention classifier
    gaze_model.onnx         <- ResNet-18 pupil centroid regressor

Both models accept input shape  (batch, 3, 64, 64)  float32
with dynamic batch dimension.

Run:
    python tools/export_onnx.py \\
        --data_path "C:/Users/kunal/Desktop/Eye tracking Module"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    resnet18,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_onnx")

# ── Model artefact filenames ──────────────────────────────────────────────────
ATTENTION_PT_FILE       = "attention_model_finetuned.pth"
ATTENTION_FALLBACK_FILE = "attention_model.pth"
GAZE_PT_FILE            = "gaze_model.pth"

ATTENTION_ONNX_FILE     = "attention_model.onnx"
GAZE_ONNX_FILE          = "gaze_model.onnx"

# ── Export settings ───────────────────────────────────────────────────────────
OPSET_VERSION     = 17
MODEL_INPUT_SIZE  = 64
INPUT_NAME        = "eye_crop"
OUTPUT_NAME       = "output"
DYNAMIC_AXES      = {INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}}

# ── Architecture constants (must match model_runner.py) ───────────────────────
ATTENTION_NUM_CLASSES = 3
ATTENTION_FEAT_DIM    = 1280
ATTENTION_HIDDEN_DIM  = 128
GAZE_FEAT_DIM         = 512
GAZE_HEAD_DIM1        = 256
GAZE_HEAD_DIM2        = 64
GAZE_NUM_OUTPUTS      = 2


# ─────────────────────────────────────────────────────────────────────────────
# Architecture builders  (mirrors model_runner.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_attention() -> nn.Module:
    """Rebuild MobileNetV2 attention architecture (no pretrained weights)."""
    model = mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(ATTENTION_FEAT_DIM, ATTENTION_HIDDEN_DIM),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(ATTENTION_HIDDEN_DIM, ATTENTION_NUM_CLASSES),
    )
    return model


class _GazeResNet(nn.Module):
    """ResNet-18 gaze regression model (mirrors model_runner._GazeResNet)."""

    def __init__(self) -> None:
        super().__init__()
        backbone    = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(GAZE_FEAT_DIM,  GAZE_HEAD_DIM1), nn.ReLU(inplace=True),
            nn.Linear(GAZE_HEAD_DIM1, GAZE_HEAD_DIM2), nn.ReLU(inplace=True),
            nn.Linear(GAZE_HEAD_DIM2, GAZE_NUM_OUTPUTS), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    """Load model state_dict from a checkpoint file.

    Args:
        model:  Target nn.Module (architecture already built).
        path:   Path to .pth checkpoint.
        device: torch.device to load to.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()


def export_model(
    model: nn.Module,
    onnx_path: Path,
    device: torch.device,
    batch_size: int = 1,
) -> None:
    """Export a single model to ONNX, verify with onnx.checker, then
    validate inference numerics with onnxruntime.

    Args:
        model:      Loaded, eval-mode nn.Module.
        onnx_path:  Output .onnx file path.
        device:     Device the model currently lives on.
        batch_size: Batch size for the dummy input used during tracing.

    Raises:
        RuntimeError: If the ONNX check or ORT validation fails.
    """
    dummy = torch.randn(batch_size, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).to(device)

    log.info("Exporting %s ...", onnx_path.name)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params      = True,
        opset_version      = OPSET_VERSION,
        do_constant_folding= True,
        input_names        = [INPUT_NAME],
        output_names       = [OUTPUT_NAME],
        dynamic_axes       = DYNAMIC_AXES,
    )

    size_mb = onnx_path.stat().st_size / 1_048_576
    log.info("  File size        : %.1f MB", size_mb)

    # ── ONNX model check ──────────────────────────────────────────────────────
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    log.info("  ONNX check       : PASSED  (opset %d)", OPSET_VERSION)

    # ── ORT numeric validation ────────────────────────────────────────────────
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    input_np = dummy.cpu().numpy()

    # Warm-up
    sess.run([OUTPUT_NAME], {INPUT_NAME: input_np})

    # Timing comparison: PyTorch vs ORT (CPU)
    cpu_sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"],
    )
    dummy_cpu = dummy.cpu().numpy()

    # PyTorch CPU timing
    model_cpu = model.cpu()
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            model_cpu(torch.from_numpy(dummy_cpu))
    pt_ms = (time.perf_counter() - t0) * 10.0   # ms per call (/ 100 * 1e3)

    # ORT CPU timing
    t0 = time.perf_counter()
    for _ in range(100):
        cpu_sess.run([OUTPUT_NAME], {INPUT_NAME: dummy_cpu})
    ort_ms = (time.perf_counter() - t0) * 10.0

    # Numeric agreement
    pt_out  = model_cpu(torch.from_numpy(dummy_cpu)).detach().numpy()
    ort_out = cpu_sess.run([OUTPUT_NAME], {INPUT_NAME: dummy_cpu})[0]
    max_diff = float(np.abs(pt_out - ort_out).max())

    log.info(
        "  ORT validation   : max_diff=%.2e  PT=%.2f ms  ORT=%.2f ms  speedup=%.1fx",
        max_diff, pt_ms, ort_ms, pt_ms / ort_ms if ort_ms > 0 else 0.0,
    )

    if max_diff > 1e-4:
        raise RuntimeError(
            f"ORT output differs from PyTorch by {max_diff:.2e} (threshold 1e-4)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main export pipeline
# ─────────────────────────────────────────────────────────────────────────────

def export_all(model_dir: Path) -> None:
    """Export all available CNN models to ONNX.

    Skips a model with a warning if its checkpoint is not present.

    Args:
        model_dir: Directory containing .pth checkpoints; ONNX files are
                   also written here.
    """
    device = torch.device("cpu")   # ONNX export must be done on CPU for portability
    model_dir.mkdir(parents=True, exist_ok=True)

    exported = []

    # ── Attention CNN ─────────────────────────────────────────────────────────
    attn_pt = model_dir / ATTENTION_PT_FILE
    if not attn_pt.exists():
        attn_pt = model_dir / ATTENTION_FALLBACK_FILE
    if attn_pt.exists():
        model = _build_attention()
        _load_checkpoint(model, attn_pt, device)
        onnx_out = model_dir / ATTENTION_ONNX_FILE
        try:
            export_model(model, onnx_out, device)
            log.info("  Saved -> %s", onnx_out)
            exported.append(ATTENTION_ONNX_FILE)
        except Exception as exc:
            log.error("Attention ONNX export failed: %s", exc)
    else:
        log.warning(
            "Attention model checkpoint not found in %s -- skipping.", model_dir
        )

    # ── Gaze CNN ──────────────────────────────────────────────────────────────
    gaze_pt = model_dir / GAZE_PT_FILE
    if gaze_pt.exists():
        model = _GazeResNet()
        _load_checkpoint(model, gaze_pt, device)
        onnx_out = model_dir / GAZE_ONNX_FILE
        try:
            export_model(model, onnx_out, device)
            log.info("  Saved -> %s", onnx_out)
            exported.append(GAZE_ONNX_FILE)
        except Exception as exc:
            log.error("Gaze ONNX export failed: %s", exc)
    else:
        log.warning("Gaze model checkpoint not found: %s -- skipping.", gaze_pt)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    if exported:
        log.info("Exported %d ONNX model(s):", len(exported))
        for name in exported:
            p = model_dir / name
            log.info("  %s  (%.1f MB)", name, p.stat().st_size / 1_048_576)
    else:
        log.warning("No models were exported (no .pth checkpoints found).")
    log.info(
        "To use ONNX at runtime, replace ModelRunner with an onnxruntime "
        "InferenceSession loading these files."
    )
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Export trained CNN models to ONNX for faster inference. "
            "Writes attention_model.onnx and gaze_model.onnx to saved_models/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path", type=Path, required=True,
        help="Project root directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and export models to ONNX.

    Args:
        argv: Argument list; defaults to sys.argv[1:] when None.
    """
    parser    = _build_arg_parser()
    args      = parser.parse_args(argv)
    data_path = args.data_path.resolve()

    if not data_path.exists():
        log.error("data_path does not exist: %s", data_path)
        sys.exit(1)

    export_all(data_path / "saved_models")


if __name__ == "__main__":
    main()
