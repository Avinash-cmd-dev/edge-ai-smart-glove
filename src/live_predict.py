import time
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

print("=" * 60)
print("EDGE AI SMART GLOVE - LIVE AI PREDICTION")
print("=" * 60)

print("\nLoading Neural Network...")
model = tf.keras.models.load_model(
    "models/trained/gesture_model.keras"
)

print("Loading Label Encoder...")
label_encoder = joblib.load(
    "models/trained/label_encoder.pkl"
)

print("Loading Feature Dataset...")
df = pd.read_csv(
    "data/processed/features_dataset.csv"
)

print(f"Loaded {len(df)} gesture samples.")

print("\nStarting live prediction...\n")
print("Press Ctrl + C to stop.\n")

while True:

    sample = df.sample(1)

    actual_label = sample["label"].values[0]

    X = sample.drop(
        columns=["sample_id", "label"],
        errors="ignore"
    )

    prediction = model.predict(
        X.values.astype(np.float32),
        verbose=0
    )

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    actual_name = label_encoder.inverse_transform(
        [actual_label]
    )[0]

    predicted_name = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    print("-" * 60)
    print(f"Actual Gesture    : {actual_name}")
    print(f"Predicted Gesture : {predicted_name}")
    print(f"Confidence        : {confidence:.2f}%")

    if actual_name == predicted_name:
        print("Status            : ✅ Correct")
    else:
        print("Status            : ❌ Incorrect")

    time.sleep(2)