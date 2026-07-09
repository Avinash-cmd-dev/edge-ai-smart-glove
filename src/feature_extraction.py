"""
Feature Extraction Module
Edge AI Smart Glove

This script:
1. Loads the processed dataset
2. Groups data by gesture sample
3. Extracts statistical features
4. Saves the feature dataset
"""

import numpy as np
import pandas as pd

from config import (
    PROCESSED_DATASET,
    FEATURE_DATASET
)


def extract_features(group):
    """Extract statistical features from one gesture sample."""

    features = {}

    sensor_columns = ["ax", "ay", "az", "gx", "gy", "gz"]

    for col in sensor_columns:

        values = group[col]

        features[f"{col}_mean"] = values.mean()
        features[f"{col}_std"] = values.std()
        features[f"{col}_min"] = values.min()
        features[f"{col}_max"] = values.max()
        features[f"{col}_rms"] = np.sqrt(np.mean(values ** 2))

    return features


def main():

    print("=" * 60)
    print("EDGE AI SMART GLOVE")
    print("FEATURE EXTRACTION")
    print("=" * 60)

    print("\nLoading processed dataset...")

    df = pd.read_csv(PROCESSED_DATASET)

    feature_rows = []

    grouped = df.groupby(["label", "sample_id"])

    for (label, sample_id), group in grouped:

        feature = extract_features(group)

        feature["sample_id"] = sample_id
        feature["label"] = label

        feature_rows.append(feature)

    feature_df = pd.DataFrame(feature_rows)

    print("\nFeature Dataset Shape:")
    print(feature_df.shape)

    print("\nFirst Five Samples:")
    print(feature_df.head())

    feature_df.to_csv(FEATURE_DATASET, index=False)

    print("\nFeature dataset saved successfully!")
    print(FEATURE_DATASET)


if __name__ == "__main__":
    main()