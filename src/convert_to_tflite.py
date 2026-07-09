"""
TensorFlow Lite Converter
Edge AI Smart Glove

Converts the trained Keras model into a TensorFlow Lite model
optimized for Edge AI deployment.
"""

import os
import tensorflow as tf

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

KERAS_MODEL = "models/trained/gesture_model.keras"
TFLITE_MODEL = "models/trained/gesture_model.tflite"

print("=" * 60)
print("EDGE AI SMART GLOVE")
print("TENSORFLOW LITE CONVERTER")
print("=" * 60)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

print("\nLoading Keras Model...")

model = tf.keras.models.load_model(KERAS_MODEL)

print("Model Loaded Successfully!")

# ---------------------------------------------------
# Convert to TensorFlow Lite
# ---------------------------------------------------

print("\nConverting Model...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

print("Conversion Successful!")

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------

with open(TFLITE_MODEL, "wb") as f:
    f.write(tflite_model)

print("\nTensorFlow Lite model saved successfully!")

# ---------------------------------------------------
# File Information
# ---------------------------------------------------

keras_size = os.path.getsize(KERAS_MODEL) / 1024
tflite_size = os.path.getsize(TFLITE_MODEL) / 1024

print("\nModel Size")
print("-" * 30)
print(f"Keras Model : {keras_size:.2f} KB")
print(f"TFLite Model: {tflite_size:.2f} KB")

reduction = (1 - (tflite_size / keras_size)) * 100

print(f"Size Reduction: {reduction:.2f}%")

# ---------------------------------------------------
# Verify TFLite Model
# ---------------------------------------------------

print("\nVerifying TensorFlow Lite Model...")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()

print("TensorFlow Lite Model Verified!")

print("\n" + "=" * 60)
print("EDGE AI MODEL READY")
print("=" * 60)