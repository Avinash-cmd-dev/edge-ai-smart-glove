import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# Load feature dataset
DATASET_PATH = "data/processed/features_dataset.csv"

df = pd.read_csv(DATASET_PATH)

# Features
X = df.drop(columns=["sample_id", "label"])

# Labels
y = df["label"]

print("\nDataset Shape:")
print(df.shape)

print("\nFeature Shape:")
print(X.shape)

print("\nLabel Shape:")
print(y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nTraining Random Forest...")

# Train
model.fit(X_train, y_train)

print("Training Complete!")

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "models/random_forest_model.pkl")

print("\nModel saved successfully!")