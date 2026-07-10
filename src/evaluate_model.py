import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)

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
ax.set_title(f"Evaluation Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig("results/plots/evaluation_confusion_matrix.png", dpi=150)
plt.close(fig)

with open("results/metrics/evaluation_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\nSaved confusion matrix plot and metrics to results/")