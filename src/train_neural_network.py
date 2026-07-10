"""
Train Neural Network
Edge AI Smart Glove

This script:
1. Loads extracted gesture features
2. Splits training/testing data
3. Builds a TensorFlow neural network
4. Trains the model
5. Evaluates performance
6. Saves the trained model
"""

import os
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from config import FEATURE_DATASET

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Model Save Path
# -----------------------------
MODEL_PATH = "models/trained/gesture_model.keras"

print("=" * 60)
print("EDGE AI SMART GLOVE")
print("NEURAL NETWORK TRAINING")
print("=" * 60)
# -----------------------------
# Load Dataset
# -----------------------------

print("\nLoading feature dataset...")

df = pd.read_csv(FEATURE_DATASET)

print("Dataset loaded successfully!")

print("\nDataset Shape:", df.shape)

# Features
X = df.drop(columns=["sample_id", "label"])

# Labels
y = df["label"]

print("Feature Shape:", X.shape)
print("Label Shape :", y.shape)
# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
# -----------------------------
# Build Neural Network
# -----------------------------

print("\nBuilding Neural Network...")

model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.30),

    Dense(64, activation="relu"),
    Dropout(0.20),

    Dense(32, activation="relu"),

    Dense(5, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# -----------------------------
# Train Neural Network
# -----------------------------

print("\nTraining Neural Network...")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

print("\nTraining Complete!")
# -----------------------------
# Evaluate Model
# -----------------------------

print("\nEvaluating Model...")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")
# -----------------------------
# Save Neural Network
# -----------------------------

print("\nSaving Neural Network...")

model.save(MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")

# -----------------------------
# Save training curves and metrics
# -----------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(history.history["accuracy"], label="train")
ax1.plot(history.history["val_accuracy"], label="validation")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()

ax2.plot(history.history["loss"], label="train")
ax2.plot(history.history["val_loss"], label="validation")
ax2.set_title("Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

fig.suptitle(f"Neural Network Training (Test Accuracy: {accuracy*100:.2f}%)")
fig.tight_layout()
fig.savefig("results/plots/neural_network_training_history.png", dpi=150)
plt.close(fig)

with open("results/metrics/neural_network_metrics.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Test Loss    : {loss:.4f}\n")
    f.write(f"Epochs Trained: {len(history.history['loss'])}\n")

print("\nSaved training curves and metrics to results/")