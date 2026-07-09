"""
Preprocessing Module
Edge AI Smart Glove

This script:
1. Loads the raw gesture dataset
2. Preserves metadata (sample_id, timestamp_ms)
3. Encodes gesture labels
4. Normalizes sensor values
5. Saves the processed dataset
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

from config import (
    RAW_DATASET,
    PROCESSED_DATASET,
)

MODEL_DIR = "models/trained"


def load_dataset():
    """Load the raw dataset."""
    print("Loading dataset...")
    return pd.read_csv(RAW_DATASET)


def preprocess_data(df):
    """Preprocess the dataset."""

    # Preserve metadata
    sample_ids = df["sample_id"]
    timestamps = df["timestamp_ms"]

    # Sensor features
    X = df[["ax", "ay", "az", "gx", "gy", "gz"]]

    # Labels
    y = df["label"]

    print(f"\nFeature Shape : {X.shape}")
    print(f"Label Shape   : {y.shape}")

    # Encode gesture labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("\nGesture Classes")
    print("-" * 30)

    for idx, gesture in enumerate(encoder.classes_):
        print(f"{idx} -> {gesture}")

    # Normalize sensor values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler and encoder
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(encoder, f"{MODEL_DIR}/label_encoder.pkl")

    print("\nScaler saved successfully.")
    print("Label Encoder saved successfully.")

    # Build processed dataframe
    processed_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    # Preserve metadata
    processed_df["sample_id"] = sample_ids.values
    processed_df["timestamp_ms"] = timestamps.values

    # Add encoded labels
    processed_df["label"] = y_encoded

    # Arrange columns
    processed_df = processed_df[
        [
            "sample_id",
            "timestamp_ms",
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
            "label",
        ]
    ]

    return processed_df


def save_dataset(df):
    """Save processed dataset."""

    df.to_csv(PROCESSED_DATASET, index=False)

    print("\nProcessed dataset saved successfully!")
    print(PROCESSED_DATASET)


def main():

    print("=" * 60)
    print("EDGE AI SMART GLOVE")
    print("PREPROCESSING")
    print("=" * 60)

    dataset = load_dataset()

    processed = preprocess_data(dataset)

    print("\nFirst Five Processed Samples")
    print("-" * 60)
    print(processed.head())

    save_dataset(processed)


if __name__ == "__main__":
    main()