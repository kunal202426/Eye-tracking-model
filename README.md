# Eye Tracking Capstone

Real-time webcam-based eye analysis pipeline that classifies **emotion** (4
classes), **attention** (3 classes), and **cognitive load** (3 classes) while
estimating **gaze position** on screen — all at ~30 fps on a consumer GPU.

---

## Hardware and Software Requirements

| Item | Requirement |
|------|-------------|
| GPU  | NVIDIA with CUDA (tested: RTX 3050 4 GB) |
| CPU  | Intel/AMD, 4+ cores (tested: i7-12650H) |
| RAM  | 8 GB+ |
| OS   | Windows 10/11 (64-bit) |
| Python | 3.11 (CPython) |

---

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install xgboost scikit-learn pandas numpy matplotlib tqdm pillow
pip install mediapipe opencv-python onnx onnxruntime
```

> **Note:** `onnxruntime-gpu` can replace `onnxruntime` for GPU-accelerated
> ONNX inference, but the CPU version already delivers significant speedups.

---

## Datasets

| Dataset | Location | Notes |
|---------|----------|-------|
| VREED features | `04 Eye Tracking Data/02 Eye Tracking Data (Features Extracted)/EyeTracking_FeaturesExtracted.csv` | 312 samples, label `Quad_Cat` 0-3 |
| OpenEDS segmentation | `openEDS/openEDS/train\|validation\|test/` | 32 919 images + `.npy` masks |
| Cognitive load | `cognitive_load_dataset.csv` | 86 435 rows, 3 balanced classes |

All three datasets must be present under the project root before training.

---

## Quick Start (after training)

```bash
# 1. Download the MediaPipe face landmark model (once)
python inference/eye_detector.py --download_model \
    --data_path "C:/Users/kunal/Desktop/Eye tracking Module"

# 2. Run gaze calibration
python calibration.py \
    --data_path "C:/Users/kunal/Desktop/Eye tracking Module"

# 3. Launch live tracking
python main.py \
    --data_path "C:/Users/kunal/Desktop/Eye tracking Module"
```

While the window is open:

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `R` | Re-run gaze calibration |

---

## Full Training Pipeline

Run scripts in this order from the project root.

### 1. Explore datasets (optional)

```bash
jupyter notebook data_exploration.ipynb
```

### 2. Generate attention pseudo-labels from OpenEDS segmentation masks

```bash
python models/attention_label_generator.py \
    --data_path "C:/Users/kunal/Desktop/Eye tracking Module"
```

Outputs: `data/attention_labels_all.csv`, `data/attention_label_distribution.png`

### 3. Train the Emotion MLP

```bash
python models/train_emotion.py \
    --data_path "C:/Users/kunal/Desktop/Eye tracking Module" \
    --epochs 100
```

Outputs: `saved_models/emotion_model.pth`, `saved_models/emotion_scaler.pkl`

### 4. Train the Attention CNN (MobileNetV2)

```bash
python models/train_attention.py \
    --data_path "C:/Users/kunal/Desktop/Eye tracking Module" \
    --epochs 30
```

Outputs: `saved_models/attention_model.pth`

### 5. (Optional) Collect webcam samples for domain adaptation

```bash
python tools/collect_webcam_samples.py \
    --data_path "C:/Users/kunal/Desktop/Eye tracking Module"
```

Controls: `F` = focused, `D` = distracted, `O` = off-task, `Q` = quit.
Saves 200 frames per class to `data/webcam_finetune/`.

### 6. (Optional) Fine-tune attention model on webcam data

```bash
python models/train_attention.py \
    --data_path "C:/Users/kunal/Desktop/Eye tracking Module" \
    --webcam_finetune_only
```

Outputs: `saved_models/attention_model_finetuned.pth`

### 7. Train the Cognitive Load XGBoost model

```bash
python models/train_cognitive_load.py \
    --data_path "C:/Users/kunal/Desktop/Eye tracking Module"
```

Outputs: `saved_models/cogload_model.pkl`, `saved_models/cogload_scaler.pkl`

### 8. Precompute gaze labels and train the Gaze ResNet-18

```bash
python models/train_gaze.py \
    --data_path "C:/Users/kunal/Desktop/Eye tracking Module" \
    --generate_labels \
    --epochs 30
```

Outputs: `saved_models/gaze_model.pth`, `saved_models/gaze_norm_params.npy`

### 9. (Optional) Export CNNs to ONNX for faster inference

```bash
python tools/export_onnx.py \
    --data_path "C:/Users/kunal/Desktop/Eye tracking Module"
```

Outputs: `saved_models/attention_model.onnx`, `saved_models/gaze_model.onnx`

Measured speedup over PyTorch (CPU): Attention ~30x, Gaze ~4x.

---

## Project Structure

```
Eye tracking Module/
|-- main.py                         Entry point -- fullscreen live tracking
|-- calibration.py                  9-point gaze calibration routine
|-- SESSION_STATE.md                Build-phase tracking (internal)
|-- data_exploration.ipynb          Dataset verification notebook
|
|-- models/
|   |-- attention_label_generator.py  Pseudo-labels from OpenEDS masks
|   |-- train_emotion.py              4-class Emotion MLP (VREED)
|   |-- train_attention.py            3-class Attention CNN (MobileNetV2)
|   |-- train_cognitive_load.py       3-class Cognitive Load XGBoost
|   `-- train_gaze.py                 Pupil centroid regression (ResNet-18)
|
|-- inference/
|   |-- eye_detector.py              MediaPipe FaceLandmarker wrapper
|   |-- feature_extractor.py         Rolling EAR-based feature computation
|   |-- model_runner.py              Multi-model inference orchestrator
|   `-- display_engine.py            Fullscreen HUD + gaze trail renderer
|
|-- tools/
|   |-- collect_webcam_samples.py    Interactive frame collection for fine-tuning
|   `-- export_onnx.py               CNN -> ONNX export and validation
|
|-- saved_models/                   (created at training time)
|   |-- emotion_model.pth           + emotion_scaler.pkl
|   |-- attention_model.pth         (+ attention_model_finetuned.pth)
|   |-- cogload_model.pkl           + cogload_scaler.pkl
|   |-- gaze_model.pth              + gaze_norm_params.npy
|   |-- face_landmarker.task        MediaPipe model (downloaded once)
|   |-- calibration.npy             Affine gaze-to-screen matrix
|   |-- calibration_screen.npy      Reference screen resolution
|   |-- attention_model.onnx        (optional, from export_onnx.py)
|   `-- gaze_model.onnx             (optional, from export_onnx.py)
|
`-- data/
    |-- attention_labels_all.csv    Generated by attention_label_generator.py
    |-- gaze_labels_all.csv         Generated by train_gaze.py --generate_labels
    `-- webcam_finetune/
        |-- focused/
        |-- distracted/
        `-- off_task/
```

---

## Model Architecture Summary

| Model | Backbone | Output | Training data |
|-------|----------|--------|---------------|
| Emotion MLP | 4-in -> 64 -> 32 -> 4 | Quad_Cat 0-3 (sad/calm/angry/happy) | VREED 312 samples |
| Attention CNN | MobileNetV2 (fine-tune last 3 layers) | 0=focused 1=distracted 2=off_task | OpenEDS 32k masks |
| Cognitive Load | XGBoost 300 trees | 0=low 1=medium 2=high | 86k balanced samples |
| Gaze ResNet | ResNet-18 + regression head | (cx_norm, cy_norm) in [0,1] | OpenEDS pupil centroids |

---

## Gaze Calibration

The system uses a 9-point affine calibration that maps the model's normalised
pupil centroid to screen pixel coordinates.  Calibration takes ~30 seconds and
must be repeated when the camera angle or seating position changes.

Run `calibration.py` once before each session, or press `R` during live
tracking to recalibrate without restarting.

---

## Known Limitations

**Domain gap (VR IR camera -> RGB webcam)**

All training datasets were collected with VR headset infrared eye trackers.
The webcam domain differs in lighting, resolution, and pupil visibility.
The fine-tuning step (`collect_webcam_samples.py` + `--webcam_finetune_only`)
compensates for this and is strongly recommended before deployment.

**Cognitive load accuracy**

The four available eye-tracking features (pupil dilation proxy, blink rate,
fixation duration, saccade duration) are weakly discriminative for cognitive
load without accompanying EEG or fNIRS signals.  Expect ~33-40% accuracy
(near chance for 3 balanced classes).  The model is included as a functional
placeholder; replacing the input features with physiological signals would
substantially improve performance.

**Emotion accuracy**

The VREED dataset contains 312 samples across 4 classes (~78 per class).
At 100 epochs the MLP typically converges to 30-45% validation accuracy.
The model performs better than chance but is not production-grade without
more data.

**MediaPipe model download**

`face_landmarker.task` (~29 MB) must be downloaded once:
```bash
python inference/eye_detector.py --download_model \
    --data_path "<project_root>"
```

---

## Dataset Adaptations vs Original Specification

The original project specification assumed dataset structures that differ from
the actual downloads.  The following adaptations were made:

| Spec assumption | Reality | Adaptation |
|-----------------|---------|------------|
| VREED: 6-class raw emotion | VREED: Quad_Cat 0-3 (4 classes) | MLP outputs 4 classes |
| OpenEDS: gaze angle vectors | OpenEDS: segmentation only | Pupil centroid regression |
| Attention: gaze velocity labels | No velocity data | EAR + pupil/iris ratio pseudo-labels |

All adaptations are documented in `data_exploration.ipynb` and
`SESSION_STATE.md`.
