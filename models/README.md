# Models

| File | Produced by | Purpose |
|------|-------------|---------|
| `random_forest_model.pkl` | `src/train_model.py` | Random Forest baseline classifier — loaded directly from this top-level path by `src/evaluate_model.py`, `src/predict.py`, and `src/utils.py` |
| `trained/scaler.pkl` | `src/preprocessing.py` | `StandardScaler` fit on raw `ax..gz` values |
| `trained/label_encoder.pkl` | `src/preprocessing.py` | Maps gesture names ↔ class indices (alphabetical) |
| `trained/gesture_model.keras` | `src/train_neural_network.py` | Trained Keras neural network (30 features → 5 classes) |
| `trained/gesture_model.tflite` | `src/convert_to_tflite.py` | Float TensorFlow Lite conversion |
| `trained/gesture_model_int8.tflite` | `src/quantize_model.py` | INT8-quantized model, embedded into `firmware/src/model_data.cpp` for on-device inference |

Note: `src/config.py` points `TRAINED_MODEL_DIR` at `trained/` for the other
artifacts, but `train_model.py` / `evaluate_model.py` / `predict.py` hardcode
the top-level `models/random_forest_model.pkl` path instead — the two aren't
consistent, but that's how the scripts actually read/write it today.
