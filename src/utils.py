import joblib
import pandas as pd
import numpy as np

def load_model(model_path="models/random_forest_model.pkl"):
    return joblib.load(model_path)

def extract_features(sensor_dict):
    """Extract statistical features from sensor reading"""
    data = pd.DataFrame([sensor_dict])
    features = {}
    for axis in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
        col = data[axis]
        features.update({
            f'{axis}_mean': col.mean(),
            f'{axis}_std': col.std(),
            f'{axis}_min': col.min(),
            f'{axis}_max': col.max(),
            f'{axis}_rms': np.sqrt(np.mean(col**2))
        })
    return features