---
LAST_COMPLETED: tools/accuracy_monitor.py
CURRENTLY_BUILDING: COMPLETE
PHASE: DONE

# Upgrade batch applied (all 10 upgrades from user prompt):
#   U1  model_runner.py     Kalman gaze smoothing (filterpy)
#   U2  model_runner.py     TemporalVoter 15-frame label stabilisation
#   U3  eye_detector.py     CLAHE preprocessing, iris-landmark pupil centre, solvePnP head pose
#   U4  feature_extractor.py  Two-threshold blink detector, iris dilation, gaze velocity
#   U5  display_engine.py   Full softmax bars, confidence colouring, eye health row
#   U6  eye_detector.py     Eyelid splines, iris circles, pupil dots, gaze arrows
#   U7  display_engine.py   Grid background, fixation heatmap, comet trail, timeline
#   U8  main.py             FPS watchdog, profiler, HD capture, graceful reinit
#   U9  tools/accuracy_monitor.py  Standalone diagnostic monitor + embedded AccuracyMonitor
#   U10 calibration.py      13-point grid, per-point quality scoring, polynomial regression

DECISIONS:
  - Project root: C:\Users\kunal\Desktop\Eye tracking Module
  - Python env: CPython 3.11 (Windows Store install)

  DATASET PATHS (confirmed by directory scan):
    VREED features:  "04 Eye Tracking Data/02 Eye Tracking Data (Features Extracted)/EyeTracking_FeaturesExtracted.csv"
    VREED raw .dat:  "04 Eye Tracking Data/01 Eye Tracking Data (Pre-Processed)/*.dat"
    VREED post-survey: "03 Self-Reported Questionnaires/02 Post Exposure Ratings.xlsx"
    OpenEDS images:  "openEDS/openEDS/train|validation|test/images/*.png"
    OpenEDS seg masks: "openEDS/openEDS/train|validation|test/labels/*.npy"   (uint8, values 0-3)
    OpenEDS bbox:    "bbox/bbox/S_*.txt"  (one row per image: x_start width y_start height)
    OpenEDS events:  "openEDS_events/openEDS_events/S_*/event_frames/*.npy"  (200x320 float32 event frames)
    Cognitive load:  "cognitive_load_dataset.csv"

  DATASET REALITY vs PROMPT ASSUMPTIONS (critical findings):
    VREED:
      - Label is Quad_Cat (0-3), NOT 6-class emotion → 4 classes: Q0=sad, Q1=calm, Q2=angry, Q3=happy
      - 312 samples x 50 features (not 4 raw features, but aggregated statistics)
      - No direct pupil_dilation column → proxy: Num_of_Blink as 4th feature
      - 96 total null values
      - Best 4 features for emotion MLP:
          [Num_of_Blink, Mean_Fixation_Duration, Mean_Saccade_Amplitude, Num_of_Fixations]
    OpenEDS:
      - SEGMENTATION subset only: 32,919 images with pixel-level masks (iris/pupil/sclera/bg)
      - NO angular gaze vectors present → cannot train gaze regression CNN as specified
      - NO gaze sequences for attention label derivation
      - ADAPTATION: Use segmentation masks to extract pupil center for gaze estimation
      - ADAPTATION: Derive pseudo attention labels from Eye Aspect Ratio + pupil visibility
      - Data counts: train=27,431 | val=2,744 | test=2,744
      - Segmentation classes: 0=background, 1=sclera, 2=iris, 3=pupil
    Cognitive Load:
      - Perfect match: 86,435 rows, columns Pupil_Dilation/Blink_Rate/Fixation_Duration/Saccade_Duration
      - Label: Cognitive_Load (0.0/1.0/2.0) → perfectly balanced (≈28,800 each)
      - Zero nulls → no preprocessing needed beyond scaling

  MODEL ADAPTATIONS:
    Model 1 - Emotion MLP:
      - 4 classes (Quad_Cat 0-3) instead of 6
      - Input: 4 features from VREED features CSV
      - Architecture unchanged (Input(4)->64->32->4 softmax)
    Model 2 - Attention CNN:
      - Training backbone on OpenEDS segmentation images (eye appearance features)
      - Pseudo-labels derived from: EAR (Eye Aspect Ratio from segmentation mask height/width ratio)
        + pupil-to-iris area ratio → focused(high ratio, normal EAR) / distracted(variable) / off_task(low EAR)
      - Fine-tuning via collect_webcam_samples.py remains the critical step
    Model 3 - Cognitive Load XGBoost:
      - No changes from spec, perfect data match
    Model 4 - Gaze Estimator:
      - CHANGED: Instead of CNN regression (no gaze labels), use segmentation-based pupil center detection
      - Train a segmentation model (lightweight U-Net or fine-tune MobileNetV2) on OpenEDS masks
      - At inference: extract pupil centroid from segmentation output → (x,y) in eye crop coords
      - Map pupil position to screen via 9-point affine calibration matrix
      - This is MORE robust than neural gaze regression given the domain gap

OPEN_ISSUES:
  - Verify Quad_Cat→emotion label mapping against questionnaire data
  - Confirm segmentation class order (0=bg/1=sclera/2=iris/3=pupil) from a visual check in notebook
  - VREED has 312 samples in features CSV but 408 rows in questionnaire — check ID alignment
  - bbox/bbox/S_*.txt format confirmed as: x_start width y_start height (4 integers per line)
  - Need to check if openEDS event frames are relevant (currently marked as domain adaptation only)
  - Requirements.txt needs segmentation-compatible libraries (already covered by torchvision)
---
