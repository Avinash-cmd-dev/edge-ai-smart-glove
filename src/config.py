"""
Project Configuration File
Edge AI Smart Glove
"""

from pathlib import Path

# ==========================
# ROOT DIRECTORY
# ==========================

ROOT_DIR = Path(__file__).resolve().parent.parent

# ==========================
# DATA DIRECTORIES
# ==========================

DATA_DIR = ROOT_DIR / "data"

PUBLIC_DATA_DIR = DATA_DIR / "public" / "gesture_dataset"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ==========================
# DATA FILES
# ==========================

RAW_DATASET = PUBLIC_DATA_DIR / "gesture_dataset.csv"

PROCESSED_DATASET = PROCESSED_DATA_DIR / "processed_dataset.csv"

FEATURE_DATASET = PROCESSED_DATA_DIR / "features_dataset.csv"

# ==========================
# MODEL DIRECTORIES
# ==========================

MODEL_DIR = ROOT_DIR / "models"

TRAINED_MODEL_DIR = MODEL_DIR / "trained"

TFLITE_MODEL_DIR = MODEL_DIR / "tflite"

# ==========================
# MODELS
# ==========================

RANDOM_FOREST_MODEL = TRAINED_MODEL_DIR / "random_forest_model.pkl"

KERAS_MODEL = TRAINED_MODEL_DIR / "gesture_model.keras"

LABEL_ENCODER = TRAINED_MODEL_DIR / "label_encoder.pkl"

TFLITE_MODEL = TFLITE_MODEL_DIR / "gesture_model.tflite"

# ==========================
# RESULTS
# ==========================

RESULTS_DIR = ROOT_DIR / "results"

PLOTS_DIR = RESULTS_DIR / "plots"

METRICS_DIR = RESULTS_DIR / "metrics"