import csv
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow import keras

print("=" * 70)
print("EDGE AI SMART GLOVE")
print("MODEL BENCHMARK")
print("=" * 70)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("data/processed/features_dataset.csv")

X = df.drop(columns=["sample_id", "label"])
y = df["label"]

print("\nDataset Loaded")
print("Samples :", len(df))
print("Features:", X.shape[1])

# --------------------------------------------------
# Random Forest
# --------------------------------------------------

print("\nLoading Random Forest...")

rf = joblib.load("models/random_forest_model.pkl")

start = time.perf_counter()

rf.predict(X)

rf_time = time.perf_counter() - start

rf_avg = rf_time / len(X)

print("Random Forest Complete")

# --------------------------------------------------
# Neural Network
# --------------------------------------------------

print("\nLoading Neural Network...")

nn = keras.models.load_model(
    "models/trained/gesture_model.keras"
)

start = time.perf_counter()

nn.predict(X, verbose=0)

nn_time = time.perf_counter() - start

nn_avg = nn_time / len(X)

print("Neural Network Complete")

# --------------------------------------------------
# TensorFlow Lite
# --------------------------------------------------

print("\nLoading TensorFlow Lite...")

interpreter = tf.lite.Interpreter(
    model_path="models/trained/gesture_model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

start = time.perf_counter()

for sample in X.values.astype(np.float32):

    interpreter.set_tensor(
        input_details[0]["index"],
        sample.reshape(1, -1)
    )

    interpreter.invoke()

    interpreter.get_tensor(output_details[0]["index"])

tflite_time = time.perf_counter() - start

tflite_avg = tflite_time / len(X)

print("TensorFlow Lite Complete")

# --------------------------------------------------
# INT8 Model
# --------------------------------------------------

print("\nLoading INT8 Model...")

interpreter = tf.lite.Interpreter(
    model_path="models/trained/gesture_model_int8.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

scale, zero = input_details[0]["quantization"]

start = time.perf_counter()

for sample in X.values:

    quant = sample / scale + zero

    quant = np.clip(quant, -128, 127)

    quant = quant.astype(np.int8)

    interpreter.set_tensor(
        input_details[0]["index"],
        quant.reshape(1, -1)
    )

    interpreter.invoke()

    interpreter.get_tensor(output_details[0]["index"])

int8_time = time.perf_counter() - start

int8_avg = int8_time / len(X)

print("INT8 Model Complete")

# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n")
print("=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)

print(f"{'Model':20} {'Total(s)':12} {'Per Sample(ms)':15}")
print("-" * 70)

print(f"{'Random Forest':20} {rf_time:<12.4f} {rf_avg*1000:<15.4f}")

print(f"{'Neural Network':20} {nn_time:<12.4f} {nn_avg*1000:<15.4f}")

print(f"{'TensorFlow Lite':20} {tflite_time:<12.4f} {tflite_avg*1000:<15.4f}")

print(f"{'INT8 TinyML':20} {int8_time:<12.4f} {int8_avg*1000:<15.4f}")

print("=" * 70)

# --------------------------------------------------
# Save benchmark chart and metrics
# --------------------------------------------------

results = [
    ("Random Forest", rf_time, rf_avg * 1000),
    ("Neural Network", nn_time, nn_avg * 1000),
    ("TensorFlow Lite", tflite_time, tflite_avg * 1000),
    ("INT8 TinyML", int8_time, int8_avg * 1000),
]

names = [r[0] for r in results]
per_sample_ms = [r[2] for r in results]

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(names, per_sample_ms, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
ax.set_ylabel("Per Sample Inference Time (ms)")
ax.set_title("Model Inference Benchmark")
for i, v in enumerate(per_sample_ms):
    ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
fig.tight_layout()
fig.savefig("results/plots/benchmark_comparison.png", dpi=150)
plt.close(fig)

with open("results/metrics/benchmark_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "total_time_s", "per_sample_ms"])
    for name, total_time, avg_ms in results:
        writer.writerow([name, f"{total_time:.4f}", f"{avg_ms:.4f}"])

print("\nSaved benchmark chart and metrics to results/")

print("\nBenchmark Completed Successfully!")