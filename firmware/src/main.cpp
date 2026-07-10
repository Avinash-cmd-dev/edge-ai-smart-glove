#include <Arduino.h>
#include "imu.h"
#include "feature_extraction.h"
#include "predictor.h"
#include "model_config.h"

IMU imu;
FeatureExtractor featureExtractor;
Predictor predictor;

IMUSample window[WINDOW_SIZE];

void setup()
{
    Serial.begin(115200);
    while (!Serial)
        ;

    Serial.println();
    Serial.println("====================================");
    Serial.println("EDGE AI SMART GLOVE");
    Serial.println("ESP32 TinyML Firmware");
    Serial.println("====================================");

    Serial.println("Initializing MPU6050...");
    if (!imu.begin())
    {
        Serial.println("MPU6050 not detected! Check wiring (SDA/SCL) and power.");
        while (true)
            ;
    }
    Serial.println("MPU6050 ready.");

    Serial.println("Initializing TinyML Predictor...");
    if (!predictor.begin())
    {
        Serial.println("Predictor initialization failed!");
        while (true)
            ;
    }
    Serial.println("Predictor ready.");
}

void loop()
{
    Serial.println("--------------------------------");
    Serial.println("Reading IMU window...");

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        unsigned long sampleStart = millis();

        if (!imu.readSample(window[i]))
        {
            Serial.println("IMU read failed, retrying window...");
            delay(SAMPLE_INTERVAL_MS);
            return;
        }

        long remaining = SAMPLE_INTERVAL_MS - (millis() - sampleStart);
        if (remaining > 0)
        {
            delay(remaining);
        }
    }

    Serial.println("Extracting features...");
    float features[NUM_FEATURES];
    featureExtractor.computeFeatures(window, features);

    Serial.println("Running TinyML inference...");
    PredictionResult result = predictor.predict(features);

    Serial.print("Predicted Gesture : ");
    Serial.println(result.label);
    Serial.print("Confidence        : ");
    Serial.print(result.confidence, 1);
    Serial.println("%");

    delay(500);
}
