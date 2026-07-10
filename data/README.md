# Data

| Directory | Contents |
|-----------|----------|
| `public/gesture_dataset/gesture_dataset.csv` | Raw recorded MPU6050 samples (`label`, `sample_id`, `timestamp_ms`, `ax`, `ay`, `az`, `gx`, `gy`, `gz`), 50 samples per gesture at ~50 Hz |
| `processed/processed_dataset.csv` | Output of `src/preprocessing.py` — label-encoded and `StandardScaler`-normalized sensor values |
| `processed/features_dataset.csv` | Output of `src/feature_extraction.py` — 30 statistical features per gesture sample |
| `raw/`, `sample/`, `synthetic/` | Reserved for additional/future data sources (currently empty) |

Gesture classes: `flick_down`, `flick_up`, `idle`, `wave_left`, `wave_right`.
