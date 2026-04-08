# Eye Tracking Module - Friend Setup Guide (Windows)

This guide is for a first-time user who wants to run or retrain the project.
Follow the steps exactly in order.

## 1. What Your Friend Needs

- Windows 10 or 11 (64-bit)
- Python 3.11.x
- Webcam
- Optional but recommended for training: NVIDIA GPU with CUDA support

If they only want to run live inference with existing models, GPU is optional.

## 2. Share the Project Folder

Share the full project folder (zip or git clone) so this structure exists:

- `main.py`, `calibration.py`
- `models/`, `inference/`, `tools/`
- `saved_models/` (must include trained model files if they are not training)
- dataset folders/files described below

## 3. Open PowerShell in Project Root

In VS Code terminal or PowerShell:

```powershell
cd "C:\path\to\Eye tracking Module"
```

All commands below assume terminal is in this root folder.

## 4. Create and Activate Python Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

## 5. Install Libraries (Exact)

Install PyTorch first (CUDA 12.1 wheels):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install the rest:

```powershell
pip install xgboost scikit-learn pandas numpy matplotlib tqdm pillow
pip install mediapipe opencv-python onnx onnxruntime filterpy scipy jupyter
```

Notes:
- `onnxruntime-gpu` can be used instead of `onnxruntime`.
- `filterpy` is required by `inference/model_runner.py`.

## 6. Verify Basic Setup

```powershell
python -c "import torch,cv2,mediapipe,onnxruntime,filterpy,xgboost; print('setup ok')"
```

If this command fails, fix that package before moving on.

## 7. Required Data Locations

Keep these paths under the project root:

1. VREED features CSV:
- `04 Eye Tracking Data/02 Eye Tracking Data (Features Extracted)/EyeTracking_FeaturesExtracted.csv`

2. Cognitive load CSV:
- `cognitive_load_dataset.csv`

3. OpenEDS segmentation dataset (for attention/gaze training):
- `openEDS/openEDS/train/images/*.png`
- `openEDS/openEDS/train/labels/*.npy`
- `openEDS/openEDS/validation/images/*.png`
- `openEDS/openEDS/validation/labels/*.npy`
- `openEDS/openEDS/test/images/*.png`
- `openEDS/openEDS/test/labels/*.npy`

If OpenEDS is missing, attention and gaze training steps will fail.

## 8. Two Ways to Start

### Path A: Run Immediately (No Retraining)

Use this if `saved_models/` already contains:

- `emotion_model.pth`
- `emotion_scaler.pkl`
- `attention_model.pth` or `attention_model_finetuned.pth`
- `cogload_model.pkl`
- `cogload_scaler.pkl`
- `gaze_model.pth`
- `gaze_norm_params.npy`

Then run:

```powershell
python inference/eye_detector.py --download_model --data_path .
python calibration.py --data_path .
python main.py --data_path .
```

Controls during live run:
- `Q` or `ESC`: quit
- `R`: recalibrate
- `D`: toggle debug overlay

### Path B: Full Training from Scratch

Run in this exact order:

1. (Optional) Explore data
```powershell
jupyter notebook data_exploration.ipynb
```

2. Generate attention pseudo-labels
```powershell
python models/attention_label_generator.py --data_path .
```

3. Train emotion model
```powershell
python models/train_emotion.py --data_path . --epochs 100
```

4. Train attention model
```powershell
python models/train_attention.py --data_path . --epochs 30
```

5. (Optional but recommended) Collect webcam data for domain adaptation
```powershell
python tools/collect_webcam_samples.py --data_path .
```

6. (Optional but recommended) Fine-tune attention model on webcam data
```powershell
python models/train_attention.py --data_path . --webcam_finetune_only
```

7. Train cognitive load model
```powershell
python models/train_cognitive_load.py --data_path .
```

8. Generate gaze labels and train gaze model
```powershell
python models/train_gaze.py --data_path . --generate_labels --epochs 30
```

9. (Optional) Export ONNX models for faster runtime
```powershell
python tools/export_onnx.py --data_path .
```

10. Download MediaPipe model, calibrate, run main app
```powershell
python inference/eye_detector.py --download_model --data_path .
python calibration.py --data_path .
python main.py --data_path .
```

## 9. Training Outputs Checklist

After training, verify these files exist:

- `saved_models/emotion_model.pth`
- `saved_models/emotion_scaler.pkl`
- `saved_models/attention_model.pth`
- `saved_models/attention_model_finetuned.pth` (if webcam fine-tuned)
- `saved_models/cogload_model.pkl`
- `saved_models/cogload_scaler.pkl`
- `saved_models/gaze_model.pth`
- `saved_models/gaze_norm_params.npy`

Optional ONNX outputs:
- `saved_models/attention_model.onnx`
- `saved_models/gaze_model.onnx`

## 10. Common Issues and Fixes

1. `ModuleNotFoundError`:
- Environment not activated or package missing.
- Re-activate `.venv` and run install commands again.

2. Webcam not opening:
- Close other apps using camera (Zoom/Teams/browser).
- Retry with another camera index:
```powershell
python main.py --data_path . --cam_index 1
```

3. `face_landmarker.task` missing:
- Run:
```powershell
python inference/eye_detector.py --download_model --data_path .
```

4. OpenEDS-related file not found:
- Check folder is exactly `openEDS/openEDS/...` under project root.

5. Very low FPS:
- Use smaller camera resolution externally, close heavy apps, and prefer GPU.

## 11. Fastest "Just Run" Commands

If all models already exist and dependencies are installed:

```powershell
.\.venv\Scripts\Activate.ps1
python inference/eye_detector.py --download_model --data_path .
python calibration.py --data_path .
python main.py --data_path .
```
