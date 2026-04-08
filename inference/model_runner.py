"""inference/model_runner.py
--------------------------------------------------------------------------------
Loads all four trained models and runs synchronous inference each webcam frame.

Upgrade 1 -- Kalman gaze smoothing:
  Gaze output is smoothed with a constant-velocity Kalman filter (filterpy).
  If the raw gaze jumps > 15% of normalised space in one frame (detection
  error), the Kalman update is skipped and the filter coasts on prediction.

Upgrade 2 -- TemporalVoter label stabilisation:
  Each of the 3 classifiers (emotion, attention, cogload) has a TemporalVoter
  that accumulates 15 frames of predictions with recency weighting.
  The stable (voted) prediction is what gets displayed.

Session scaler fallback:
  If a saved scaler file is missing, ModelRunner fits a sklearn StandardScaler
  on the last 300 frames of live features and uses it instead.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from filterpy.kalman import KalmanFilter
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torchvision.models import mobilenet_v2, resnet18

log = logging.getLogger("model_runner")

# ── Model artefact filenames ──────────────────────────────────────────────────
EMOTION_MODEL_FILE      = "emotion_model.pth"
EMOTION_SCALER_FILE     = "emotion_scaler.pkl"
ATTENTION_MODEL_FILE    = "attention_model_finetuned.pth"
ATTENTION_FALLBACK_FILE = "attention_model.pth"
COGLOAD_MODEL_FILE      = "cogload_model.pkl"
COGLOAD_SCALER_FILE     = "cogload_scaler.pkl"
GAZE_MODEL_FILE         = "gaze_model.pth"

# ── Label maps ────────────────────────────────────────────────────────────────
EMOTION_NAMES:   dict[int, str] = {0: "sad", 1: "calm", 2: "angry", 3: "happy"}
ATTENTION_NAMES: dict[int, str] = {0: "focused", 1: "distracted", 2: "off_task"}
COGLOAD_NAMES:   dict[int, str] = {0: "low", 1: "medium", 2: "high"}

# ── Image preprocessing ───────────────────────────────────────────────────────
MODEL_INPUT_SIZE = 64
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Architecture constants ────────────────────────────────────────────────────
EMOTION_IN_FEATURES     = 4
EMOTION_NUM_CLASSES     = 4
ATTENTION_NUM_CLASSES   = 3
ATTENTION_FEAT_DIM      = 1280
ATTENTION_HIDDEN_DIM    = 128
GAZE_FEAT_DIM           = 512
GAZE_HEAD_DIM1          = 256
GAZE_HEAD_DIM2          = 64
GAZE_NUM_OUTPUTS        = 2

# ── Kalman filter parameters (Upgrade 1) ─────────────────────────────────────
KALMAN_DT              = 1.0 / 30.0   # 30 fps
KALMAN_R_SCALE         = 50.0         # measurement noise
KALMAN_Q_SCALE         = 0.01         # process noise
KALMAN_P_SCALE         = 100.0        # initial state covariance
GAZE_JUMP_THRESHOLD    = 0.15         # max normalised jump before skipping update

# ── TemporalVoter parameters (Upgrade 2) ─────────────────────────────────────
VOTER_WINDOW           = 15
VOTER_MIN_CONFIDENCE   = 0.45

# ── Session scaler parameters ─────────────────────────────────────────────────
SESSION_SCALER_FIT_FRAMES = 300  # frames before fitting session scaler


# ─────────────────────────────────────────────────────────────────────────────
# Model architectures
# ─────────────────────────────────────────────────────────────────────────────

class _EmotionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMOTION_IN_FEATURES, 64),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(32, EMOTION_NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_attention_model() -> nn.Module:
    model = mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(ATTENTION_FEAT_DIM, ATTENTION_HIDDEN_DIM),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(ATTENTION_HIDDEN_DIM, ATTENTION_NUM_CLASSES),
    )
    return model


class _GazeResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone    = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(GAZE_FEAT_DIM, GAZE_HEAD_DIM1), nn.ReLU(inplace=True),
            nn.Linear(GAZE_HEAD_DIM1, GAZE_HEAD_DIM2), nn.ReLU(inplace=True),
            nn.Linear(GAZE_HEAD_DIM2, GAZE_NUM_OUTPUTS), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _build_gaze_model() -> nn.Module:
    return _GazeResNet()


# ─────────────────────────────────────────────────────────────────────────────
# Kalman gaze smoother (Upgrade 1)
# ─────────────────────────────────────────────────────────────────────────────

def build_gaze_kalman() -> KalmanFilter:
    """Build a constant-velocity Kalman filter for (cx, cy) gaze smoothing.

    State:  [cx, cy, vcx, vcy]
    Measurement: [cx, cy]
    """
    dt = KALMAN_DT
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=np.float64)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float64)
    kf.R *= KALMAN_R_SCALE
    kf.Q *= KALMAN_Q_SCALE
    kf.P *= KALMAN_P_SCALE
    kf.x = np.array([[0.5], [0.5], [0.0], [0.0]], dtype=np.float64)
    return kf


# ─────────────────────────────────────────────────────────────────────────────
# TemporalVoter (Upgrade 2)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalVoter:
    """Accumulates classification history and returns a stable voted prediction.

    Recent frames are weighted more heavily (linear recency weighting).
    If the winning class's weighted score < min_confidence, returns "uncertain".
    """

    def __init__(
        self,
        window: int          = VOTER_WINDOW,
        min_confidence: float = VOTER_MIN_CONFIDENCE,
    ) -> None:
        self.window         = window
        self.min_confidence = min_confidence
        self.history: list[tuple[int, float]] = []  # (class_idx, confidence)

    def update(self, predicted_class: int, confidence: float) -> None:
        """Add one frame's prediction to the history buffer."""
        self.history.append((predicted_class, confidence))
        if len(self.history) > self.window:
            self.history.pop(0)

    def get_stable_prediction(self) -> tuple[Optional[int], str, float]:
        """Return (best_class_idx, best_class_name_or_uncertain, score).

        Returns:
            (class_idx, class_name, norm_score) where class_name may be
            "uncertain" and class_idx None when confidence is too low.
        """
        if not self.history:
            return None, "uncertain", 0.0

        scores: dict[int, float] = defaultdict(float)
        total_weight = 0.0
        n = len(self.history)
        for i, (cls, conf) in enumerate(self.history):
            weight        = (i + 1) / n   # linear recency
            scores[cls]  += weight * conf
            total_weight += weight * conf

        if total_weight < 1e-8:
            return None, "uncertain", 0.0

        best_class = max(scores, key=lambda k: scores[k])
        best_score = scores[best_class] / total_weight

        if best_score < self.min_confidence:
            return None, "uncertain", float(best_score)

        return int(best_class), str(best_class), float(best_score)

    def reset(self) -> None:
        self.history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Predictions dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ModelPredictions:
    """All per-frame model output predictions.

    Original fields (raw per-frame):
        emotion_label/name/conf, attention_label/name/conf,
        cogload_label/name/conf, gaze_cx, gaze_cy

    New fields (Upgrade 1 + 2):
        *_probs:              Full softmax probability vectors (all classes).
        *_stable_name/conf:   TemporalVoter stable predictions.
        gaze_cx_smooth/cy_smooth: Kalman-filtered gaze position.
        kalman_state:         Full Kalman state [cx, cy, vcx, vcy].
    """
    # ── Raw per-frame predictions ─────────────────────────────────────────────
    emotion_label:   Optional[int]       = None
    emotion_name:    Optional[str]       = None
    emotion_conf:    Optional[float]     = None
    emotion_probs:   Optional[np.ndarray] = None   # shape (4,)

    attention_label: Optional[int]       = None
    attention_name:  Optional[str]       = None
    attention_conf:  Optional[float]     = None
    attention_probs: Optional[np.ndarray] = None   # shape (3,)

    cogload_label:   Optional[int]       = None
    cogload_name:    Optional[str]       = None
    cogload_conf:    Optional[float]     = None
    cogload_probs:   Optional[np.ndarray] = None   # shape (3,)

    gaze_cx:         Optional[float]     = None
    gaze_cy:         Optional[float]     = None

    # ── Stable TemporalVoter outputs (Upgrade 2) ──────────────────────────────
    emotion_stable_name:    Optional[str]   = None
    emotion_stable_conf:    Optional[float] = None
    attention_stable_name:  Optional[str]   = None
    attention_stable_conf:  Optional[float] = None
    cogload_stable_name:    Optional[str]   = None
    cogload_stable_conf:    Optional[float] = None

    # ── Kalman-smoothed gaze (Upgrade 1) ──────────────────────────────────────
    gaze_cx_smooth:  Optional[float]     = None
    gaze_cy_smooth:  Optional[float]     = None
    kalman_state:    Optional[np.ndarray] = None  # [cx, cy, vcx, vcy]

    models_loaded: dataclasses.field(default_factory=dict) = dataclasses.field(
        default_factory=dict,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helper
# ─────────────────────────────────────────────────────────────────────────────

def _bgr_crop_to_tensor(
    bgr_crop: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """BGR numpy eye crop -> (1, 3, 64, 64) float tensor."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return _EVAL_TRANSFORM(pil).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Session scaler
# ─────────────────────────────────────────────────────────────────────────────

class _SessionScaler:
    """Fits a StandardScaler on live session data as a fallback."""

    def __init__(self, name: str, fit_frames: int = SESSION_SCALER_FIT_FRAMES) -> None:
        self._name       = name
        self._fit_frames = fit_frames
        self._buf: Deque[np.ndarray] = collections.deque(maxlen=fit_frames)
        self._scaler: Optional[StandardScaler] = None
        self._fitted = False

    def push(self, x: np.ndarray) -> None:
        self._buf.append(x.copy())
        if len(self._buf) >= 30 and len(self._buf) % 30 == 0:
            X = np.stack(list(self._buf))
            sc = StandardScaler()
            sc.fit(X)
            self._scaler  = sc
            self._fitted  = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._fitted and self._scaler is not None:
            return self._scaler.transform(x)
        return x  # no scaling until enough data


# ─────────────────────────────────────────────────────────────────────────────
# ModelRunner
# ─────────────────────────────────────────────────────────────────────────────

class ModelRunner:
    """Loads all four trained models and runs synchronous per-frame inference.

    Upgrades:
      1 -- Kalman filter smoothing on gaze output.
      2 -- TemporalVoter for stable emotion / attention / cogload labels.
    """

    def __init__(
        self,
        model_dir: Path,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._loaded: dict[str, bool] = {}

        log.info("ModelRunner init (device=%s, model_dir=%s)", device, model_dir)

        self._emotion_model   = self._load_emotion(model_dir)
        self._emotion_scaler  = self._load_scaler(
            model_dir / EMOTION_SCALER_FILE, "emotion_scaler",
        )
        self._attention_model = self._load_attention(model_dir)
        self._cogload_model   = self._load_cogload(model_dir)
        self._cogload_scaler  = self._load_scaler(
            model_dir / COGLOAD_SCALER_FILE, "cogload_scaler",
        )
        self._gaze_model      = self._load_gaze(model_dir)

        # Session scalers (fallback when pickle files are missing)
        self._emotion_session_scaler  = _SessionScaler("emotion_session")
        self._cogload_session_scaler  = _SessionScaler("cogload_session")

        # Kalman filter (Upgrade 1)
        self._kf: KalmanFilter         = build_gaze_kalman()
        self._kf_initialised: bool     = False
        self._prev_gaze: Optional[tuple[float, float]] = None

        # TemporalVoters (Upgrade 2)
        self._emotion_voter   = TemporalVoter()
        self._attention_voter = TemporalVoter()
        self._cogload_voter   = TemporalVoter()

        loaded_str = "  ".join(
            f"{k}={'OK' if v else '--'}" for k, v in self._loaded.items()
        )
        log.info("Models: %s", loaded_str)

    # ── Public interface ──────────────────────────────────────────────────────

    @torch.no_grad()
    def run(
        self,
        eye_crop_bgr:   Optional[np.ndarray],
        emotion_feats:  Optional[np.ndarray],
        cogload_feats:  Optional[np.ndarray],
    ) -> ModelPredictions:
        """Run all loaded models; apply Kalman and TemporalVoter."""
        preds = ModelPredictions(models_loaded=dict(self._loaded))

        # ── 1. Emotion MLP ────────────────────────────────────────────────────
        if (self._emotion_model is not None
                and emotion_feats is not None):
            try:
                scaler = (self._emotion_scaler
                          or self._emotion_session_scaler)
                if isinstance(scaler, _SessionScaler):
                    self._emotion_session_scaler.push(emotion_feats)
                    x_scaled = self._emotion_session_scaler.transform(
                        emotion_feats.reshape(1, -1),
                    ).astype(np.float32)
                else:
                    x_scaled = scaler.transform(
                        emotion_feats.reshape(1, -1),
                    ).astype(np.float32)

                x_t    = torch.from_numpy(x_scaled).to(self.device)
                logits = self._emotion_model(x_t)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                label  = int(np.argmax(probs))
                conf   = float(probs[label])

                preds.emotion_label  = label
                preds.emotion_name   = EMOTION_NAMES.get(label, str(label))
                preds.emotion_conf   = conf
                preds.emotion_probs  = probs.copy()

                self._emotion_voter.update(label, conf)
                stable_idx, _, stable_score = self._emotion_voter.get_stable_prediction()
                preds.emotion_stable_name = (
                    EMOTION_NAMES.get(stable_idx, "uncertain")
                    if stable_idx is not None else "uncertain"
                )
                preds.emotion_stable_conf = stable_score
            except Exception as exc:
                log.warning("Emotion inference failed: %s", exc)

        # ── 2. Attention CNN ──────────────────────────────────────────────────
        if self._attention_model is not None and eye_crop_bgr is not None:
            try:
                x_t    = _bgr_crop_to_tensor(eye_crop_bgr, self.device)
                logits = self._attention_model(x_t)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                label  = int(np.argmax(probs))
                conf   = float(probs[label])

                preds.attention_label  = label
                preds.attention_name   = ATTENTION_NAMES.get(label, str(label))
                preds.attention_conf   = conf
                preds.attention_probs  = probs.copy()

                self._attention_voter.update(label, conf)
                stable_idx, _, stable_score = self._attention_voter.get_stable_prediction()
                preds.attention_stable_name = (
                    ATTENTION_NAMES.get(stable_idx, "uncertain")
                    if stable_idx is not None else "uncertain"
                )
                preds.attention_stable_conf = stable_score
            except Exception as exc:
                log.warning("Attention inference failed: %s", exc)

        # ── 3. Cognitive-load XGBoost ─────────────────────────────────────────
        if (self._cogload_model is not None and cogload_feats is not None):
            try:
                scaler = (self._cogload_scaler
                          or self._cogload_session_scaler)
                if isinstance(scaler, _SessionScaler):
                    self._cogload_session_scaler.push(cogload_feats)
                    x_scaled = self._cogload_session_scaler.transform(
                        cogload_feats.reshape(1, -1),
                    ).astype(np.float32)
                else:
                    x_scaled = scaler.transform(
                        cogload_feats.reshape(1, -1),
                    ).astype(np.float32)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    proba = self._cogload_model.predict_proba(x_scaled)[0]

                label = int(np.argmax(proba))
                conf  = float(proba[label])

                preds.cogload_label  = label
                preds.cogload_name   = COGLOAD_NAMES.get(label, str(label))
                preds.cogload_conf   = conf
                preds.cogload_probs  = proba.copy()

                self._cogload_voter.update(label, conf)
                stable_idx, _, stable_score = self._cogload_voter.get_stable_prediction()
                preds.cogload_stable_name = (
                    COGLOAD_NAMES.get(stable_idx, "uncertain")
                    if stable_idx is not None else "uncertain"
                )
                preds.cogload_stable_conf = stable_score
            except Exception as exc:
                log.warning("Cognitive-load inference failed: %s", exc)

        # ── 4. Gaze ResNet + Kalman filter (Upgrade 1) ───────────────────────
        if self._gaze_model is not None and eye_crop_bgr is not None:
            try:
                x_t = _bgr_crop_to_tensor(eye_crop_bgr, self.device)
                out = self._gaze_model(x_t).cpu().numpy()[0]
                raw_cx = float(out[0])
                raw_cy = float(out[1])

                preds.gaze_cx = raw_cx
                preds.gaze_cy = raw_cy

                # Kalman predict every frame
                self._kf.predict()

                # Jump detection: skip update if displacement > threshold
                do_update = True
                if self._prev_gaze is not None:
                    dx = abs(raw_cx - self._prev_gaze[0])
                    dy = abs(raw_cy - self._prev_gaze[1])
                    if dx > GAZE_JUMP_THRESHOLD or dy > GAZE_JUMP_THRESHOLD:
                        do_update = False

                if do_update:
                    if not self._kf_initialised:
                        self._kf.x = np.array(
                            [[raw_cx], [raw_cy], [0.0], [0.0]],
                            dtype=np.float64,
                        )
                        self._kf_initialised = True
                    else:
                        self._kf.update(np.array([[raw_cx], [raw_cy]]))

                self._prev_gaze = (raw_cx, raw_cy)

                preds.gaze_cx_smooth = float(self._kf.x[0, 0])
                preds.gaze_cy_smooth = float(self._kf.x[1, 0])
                preds.kalman_state   = self._kf.x.flatten().copy()

            except Exception as exc:
                log.warning("Gaze inference failed: %s", exc)

        return preds

    # ── Private loaders ───────────────────────────────────────────────────────

    def _load_emotion(self, model_dir: Path) -> Optional[nn.Module]:
        path = model_dir / EMOTION_MODEL_FILE
        if not path.exists():
            log.warning("Emotion model not found: %s", path)
            self._loaded["emotion"] = False
            return None
        try:
            ckpt  = torch.load(path, map_location=self.device, weights_only=False)
            model = _EmotionMLP().to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._loaded["emotion"] = True
            log.info("Emotion model loaded (val_acc=%.4f)",
                     ckpt.get("best_val_acc", float("nan")))
            return model
        except Exception as exc:
            log.warning("Failed to load emotion model: %s", exc)
            self._loaded["emotion"] = False
            return None

    def _load_attention(self, model_dir: Path) -> Optional[nn.Module]:
        path = model_dir / ATTENTION_MODEL_FILE
        if not path.exists():
            path = model_dir / ATTENTION_FALLBACK_FILE
        if not path.exists():
            log.warning("Attention model not found in: %s", model_dir)
            self._loaded["attention"] = False
            return None
        try:
            ckpt  = torch.load(path, map_location=self.device, weights_only=False)
            model = _build_attention_model().to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._loaded["attention"] = True
            log.info("Attention model loaded from %s", path.name)
            return model
        except Exception as exc:
            log.warning("Failed to load attention model: %s", exc)
            self._loaded["attention"] = False
            return None

    def _load_cogload(self, model_dir: Path) -> Optional[object]:
        path = model_dir / COGLOAD_MODEL_FILE
        if not path.exists():
            log.warning("Cognitive-load model not found: %s", path)
            self._loaded["cogload"] = False
            return None
        try:
            with open(path, "rb") as fh:
                model = pickle.load(fh)
            self._loaded["cogload"] = True
            log.info("Cognitive-load model loaded from %s", path.name)
            return model
        except Exception as exc:
            log.warning("Failed to load cognitive-load model: %s", exc)
            self._loaded["cogload"] = False
            return None

    def _load_gaze(self, model_dir: Path) -> Optional[nn.Module]:
        path = model_dir / GAZE_MODEL_FILE
        if not path.exists():
            log.warning("Gaze model not found: %s", path)
            self._loaded["gaze"] = False
            return None
        try:
            ckpt  = torch.load(path, map_location=self.device, weights_only=False)
            model = _build_gaze_model().to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._loaded["gaze"] = True
            log.info("Gaze model loaded (best_val_mse=%.6f)",
                     ckpt.get("best_val_mse", float("nan")))
            return model
        except Exception as exc:
            log.warning("Failed to load gaze model: %s", exc)
            self._loaded["gaze"] = False
            return None

    @staticmethod
    def _load_scaler(path: Path, name: str) -> Optional[object]:
        if not path.exists():
            log.warning("%s not found: %s -- will use session scaler", name, path)
            return None
        try:
            with open(path, "rb") as fh:
                scaler = pickle.load(fh)
            log.info("%s loaded.", name)
            return scaler
        except Exception as exc:
            log.warning("Failed to load %s: %s", name, exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load all model artefacts and run a synthetic forward-pass smoke "
            "test. No webcam required."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=Path, required=True,
                        help="Project root directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Load models and run synthetic forward-pass."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser    = _build_arg_parser()
    args      = parser.parse_args(argv)
    data_path = args.data_path.resolve()

    if not data_path.exists():
        log.error("data_path does not exist: %s", data_path)
        sys.exit(1)

    runner = ModelRunner(data_path / "saved_models")

    dummy_crop         = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    dummy_emotion_feats = np.array([12.0, 104.5, 300.0, 2.5], dtype=np.float32)
    dummy_cogload_feats = np.array([3.5, 18.0, 250.0, 120.0], dtype=np.float32)

    # Run several frames to exercise voters
    for _ in range(20):
        preds = runner.run(
            eye_crop_bgr  = dummy_crop,
            emotion_feats = dummy_emotion_feats,
            cogload_feats = dummy_cogload_feats,
        )

    log.info("=" * 60)
    log.info("Raw:    emotion=%s(%.2f)  attention=%s(%.2f)  cogload=%s(%.2f)",
             preds.emotion_name, preds.emotion_conf or 0,
             preds.attention_name, preds.attention_conf or 0,
             preds.cogload_name, preds.cogload_conf or 0)
    log.info("Stable: emotion=%s(%.2f)  attention=%s(%.2f)  cogload=%s(%.2f)",
             preds.emotion_stable_name, preds.emotion_stable_conf or 0,
             preds.attention_stable_name, preds.attention_stable_conf or 0,
             preds.cogload_stable_name, preds.cogload_stable_conf or 0)
    log.info("Gaze raw=(%.3f,%.3f) smooth=(%.3f,%.3f)",
             preds.gaze_cx or 0, preds.gaze_cy or 0,
             preds.gaze_cx_smooth or 0, preds.gaze_cy_smooth or 0)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
