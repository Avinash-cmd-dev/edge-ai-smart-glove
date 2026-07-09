import random
import pandas as pd
import joblib

print("=" * 60)
print("EDGE AI SMART GLOVE - GESTURE PREDICTION")
print("=" * 60)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
print("\nLoading trained Random Forest model...")

model = joblib.load("models/random_forest_model.pkl")

print("Model loaded successfully!")

# -------------------------------------------------
# Load Feature Dataset
# -------------------------------------------------
print("\nLoading feature dataset...")

df = pd.read_csv("data/processed/features_dataset.csv")

print("Feature dataset loaded successfully!")

# -------------------------------------------------
# Pick a Random Sample
# -------------------------------------------------
sample_index = random.randint(0, len(df) - 1)

sample = df.iloc[sample_index]

print(f"\nTesting Sample Index : {sample_index}")

actual_label = sample["label"]

print(f"Actual Gesture       : {actual_label}")

# -------------------------------------------------
# Prepare Features
# -------------------------------------------------
X = sample.drop(["sample_id", "label"]).to_frame().T

# -------------------------------------------------
# Prediction
# -------------------------------------------------
prediction = model.predict(X)[0]

probabilities = model.predict_proba(X)[0]

confidence = probabilities.max() * 100

print(f"Predicted Gesture    : {prediction}")
print(f"Confidence           : {confidence:.2f}%")

# -------------------------------------------------
# Result
# -------------------------------------------------
print("\nPrediction Result")

if prediction == actual_label:
    print("Prediction Correct")
else:
    print("Prediction Incorrect")

# -------------------------------------------------
# Probability of Every Gesture
# -------------------------------------------------
print("\nClass Probabilities")
print("-" * 30)

for gesture, prob in zip(model.classes_, probabilities):
    print(f"{gesture:<12} : {prob*100:6.2f}%")

# -------------------------------------------------
# Top 3 Predictions
# -------------------------------------------------
print("\nTop Predictions")
print("-" * 30)

sorted_predictions = sorted(
    zip(model.classes_, probabilities),
    key=lambda x: x[1],
    reverse=True
)

for gesture, prob in sorted_predictions[:3]:
    print(f"{gesture:<12} : {prob*100:6.2f}%")

print("\n" + "=" * 60)
print("Prediction Completed Successfully")
print("=" * 60)