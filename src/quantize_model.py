"""
Quantize TensorFlow Lite Model
Edge AI Smart Glove

Converts the trained Keras model into an INT8 TensorFlow Lite model
for TinyML deployment.
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

KERAS_MODEL = "models/trained/gesture_model.keras"
INT8_MODEL = "models/trained/gesture_model_int8.tflite"
FEATURE_DATASET = "data/processed/features_dataset.csv"

print("=" * 60)
print("EDGE AI SMART GLOVE")
print("INT8 QUANTIZATION")
print("=" * 60)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv(FEATURE_DATASET)

X = df.drop(columns=["sample_id", "label"]).astype(np.float32)

print(f"\nDataset Shape : {X.shape}")

# ---------------------------------------------------
# Representative Dataset
# ---------------------------------------------------

def representative_dataset():

    for i in range(min(100, len(X))):
        yield [X.iloc[i:i+1].values.astype(np.float32)]

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

model = tf.keras.models.load_model(KERAS_MODEL)

print("\nLoaded Keras Model")

# ---------------------------------------------------
# Converter
# ---------------------------------------------------

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Converting...")

quantized_model = converter.convert()

# ---------------------------------------------------
# Save
# ---------------------------------------------------

with open(INT8_MODEL, "wb") as f:
    f.write(quantized_model)

print("\nINT8 Model Saved!")

# ---------------------------------------------------
# Statistics
# ---------------------------------------------------

keras_size = os.path.getsize(KERAS_MODEL)/1024
int8_size = os.path.getsize(INT8_MODEL)/1024

print("\nModel Sizes")
print("-"*35)

print(f"Keras Model : {keras_size:.2f} KB")
print(f"INT8 Model  : {int8_size:.2f} KB")

reduction = (1-int8_size/keras_size)*100

print(f"Compression : {reduction:.2f}%")

# ---------------------------------------------------
# Verify
# ---------------------------------------------------

print("\nVerifying INT8 Model...")

interpreter = tf.lite.Interpreter(model_path=INT8_MODEL)

interpreter.allocate_tensors()

print("Verification Successful!")

print("\n" + "="*60)
print("TINYML MODEL READY")
print("="*60)