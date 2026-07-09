import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Load feature dataset
df = pd.read_csv("data/processed/features_dataset.csv")

# Features and labels
X = df.drop(columns=["sample_id", "label"])
y = df["label"]

# Same train-test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load saved model
model = joblib.load("models/random_forest_model.pkl")

# Predict
y_pred = model.predict(X_test)

print("\nAccuracy:")
print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))