#include <Arduino.h>
#include <math.h>
#include "feature_extraction.h"

namespace
{
float channelValue(const IMUSample &sample, int channel)
{
    switch (channel)
    {
    case 0:
        return sample.ax;
    case 1:
        return sample.ay;
    case 2:
        return sample.az;
    case 3:
        return sample.gx;
    case 4:
        return sample.gy;
    default:
        return sample.gz;
    }
}
} // namespace

void FeatureExtractor::computeFeatures(const IMUSample window[WINDOW_SIZE], float outFeatures[NUM_FEATURES])
{
    for (int channel = 0; channel < NUM_CHANNELS; channel++)
    {
        // Standardize each raw sample the same way src/preprocessing.py's
        // StandardScaler does, before computing statistics over the window.
        float standardized[WINDOW_SIZE];
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            float raw = channelValue(window[i], channel);
            standardized[i] = (raw - SCALER_MEAN[channel]) / SCALER_SCALE[channel];
        }

        float sum = 0.0f;
        float sumSquares = 0.0f;
        float minVal = standardized[0];
        float maxVal = standardized[0];

        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            float v = standardized[i];
            sum += v;
            sumSquares += v * v;
            if (v < minVal)
                minVal = v;
            if (v > maxVal)
                maxVal = v;
        }

        float mean = sum / WINDOW_SIZE;

        float varianceSum = 0.0f;
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            float diff = standardized[i] - mean;
            varianceSum += diff * diff;
        }
        // Sample standard deviation (ddof=1), matching pandas' Series.std().
        float std = sqrtf(varianceSum / (WINDOW_SIZE - 1));

        float rms = sqrtf(sumSquares / WINDOW_SIZE);

        int base = channel * NUM_FEATURES_PER_CHANNEL;
        outFeatures[base + 0] = mean;
        outFeatures[base + 1] = std;
        outFeatures[base + 2] = minVal;
        outFeatures[base + 3] = maxVal;
        outFeatures[base + 4] = rms;
    }
}
