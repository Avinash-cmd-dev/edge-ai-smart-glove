![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn)
![TinyML](https://img.shields.io/badge/TinyML-Ready-success)
![ESP32](https://img.shields.io/badge/ESP32-Deployment-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

# 🧤 Edge AI Smart Glove - TinyML Gesture Recognition

An end-to-end TinyML project for real-time hand gesture recognition using IMU sensor data, TensorFlow Lite, and embedded AI techniques.

The project processes accelerometer and gyroscope data, extracts statistical features, trains machine learning models, converts the best-performing neural network to TensorFlow Lite, and prepares it for deployment on resource-constrained microcontrollers such as the ESP32.

---

## 🚀 Features

- End-to-end gesture recognition pipeline
- Data preprocessing and cleaning
- Statistical feature extraction (30 features)
- Random Forest baseline model
- TensorFlow Neural Network
- TensorFlow Lite conversion
- INT8 Quantization for TinyML
- Live gesture prediction (simulation)
- Real-time on-device inference on ESP32 + MPU6050 (TensorFlow Lite Micro)

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- TensorFlow Lite
- Scikit-learn
- NumPy
- Pandas
- Joblib

Target Hardware

- ESP32 (deployment target)
- MPU6050 IMU Sensor

---

## 📂 Project Structure

```
edge-ai-smart-glove/

├── data/
├── docs/
├── firmware/
├── models/
├── notebooks/
├── results/
├── src/
├── tests/
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🔄 Project Pipeline

```
Raw IMU Data
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Extraction
(30 Statistical Features)
        │
        ▼
Neural Network Training
        │
        ▼
Model Evaluation
        │
        ▼
TensorFlow Lite Conversion
        │
        ▼
INT8 Quantization
        │
        ▼
Live Prediction
        │
        ▼
ESP32 Deployment (firmware/)
```

---

## 📊 Results

| Metric | Value |
|---------|------:|
| Test Accuracy | **99.55%** |
| Feature Count | **30** |
| Model Format | TensorFlow Lite |
| Quantization | INT8 |
| Deployment | TinyML Ready |

---

## 📁 Main Scripts

| File | Purpose |
|------|---------|
| preprocessing.py | Dataset preprocessing |
| feature_extraction.py | Extract statistical features |
| train_model.py | Random Forest baseline |
| train_neural_network.py | TensorFlow model training |
| evaluate_model.py | Evaluate trained model |
| convert_to_tflite.py | Convert Keras → TFLite |
| quantize_model.py | INT8 quantization |
| live_predict.py | Live inference simulation |
| benchmark_models.py | Compare model performance |

---

## ⚙️ Installation

```bash
git clone <repository-url>

cd edge-ai-smart-glove

pip install -r requirements.txt
```

---

## ▶️ Run

Preprocess

```bash
python src/preprocessing.py
```

Extract Features

```bash
python src/feature_extraction.py
```

Train Neural Network

```bash
python src/train_neural_network.py
```

Evaluate

```bash
python src/evaluate_model.py
```

Convert to TensorFlow Lite

```bash
python src/convert_to_tflite.py
```

Quantize

```bash
python src/quantize_model.py
```

Live Prediction

```bash
python src/live_predict.py
```

---

## 🔌 Firmware (ESP32 + MPU6050)

The `firmware/` directory contains a PlatformIO project that runs the INT8
model directly on an ESP32 using TensorFlow Lite Micro: it samples the
MPU6050 over I2C, reproduces the 30-feature extraction from `src/` on-device,
and prints the predicted gesture and confidence over serial. See
[firmware/README.md](firmware/README.md) for wiring and build instructions.

## 🔮 Future Work

- Optimize inference latency
- Add Bluetooth gesture streaming
- Develop a wearable smart glove prototype

---

## 👨‍💻 Author

Avinash Singh

Electronics & Communication Engineering

Edge AI • TinyML • Embedded Systems • Machine Learning