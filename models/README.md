# Models

All trained artifacts live in `trained/`:

| File | Produced by | Purpose |
|------|-------------|---------|
| `scaler.pkl` | `src/preprocessing.py` | `StandardScaler` fit on raw `ax..gz` values |
| `label_encoder.pkl` | `src/preprocessing.py` | Maps gesture names ↔ class indices (alphabetical) |
| `random_forest_model.pkl` | `src/train_model.py` | Random Forest baseline classifier |
| `gesture_model.keras` | `src/train_neural_network.py` | Trained Keras neural network (30 features → 5 classes) |
| `gesture_model.tflite` | `src/convert_to_tflite.py` | Float TensorFlow Lite conversion |
| `gesture_model_int8.tflite` | `src/quantize_model.py` | INT8-quantized model, embedded into `firmware/src/model_data.cpp` for on-device inference |

`models/random_forest_model.pkl` at the top level is a byte-for-byte
duplicate of `trained/random_forest_model.pkl`; `trained/` is the canonical
copy referenced by `src/config.py`.
