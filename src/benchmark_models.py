import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

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

rf = joblib.load("models/trained/random_forest_model.pkl")

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

print("\nBenchmark Completed Successfully!")