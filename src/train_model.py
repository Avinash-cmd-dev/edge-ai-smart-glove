import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)

# Save model
joblib.dump(model, "models/random_forest_model.pkl")

print("\nModel saved successfully!")

# --------------------------------------------------
# Save confusion matrix plot and metrics
# --------------------------------------------------

class_names = sorted(y.unique())

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Random Forest Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig("results/plots/random_forest_confusion_matrix.png", dpi=150)
plt.close(fig)

with open("results/metrics/random_forest_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\nSaved confusion matrix plot and metrics to results/")