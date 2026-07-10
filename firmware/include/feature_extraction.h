#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include "model_config.h"
#include "imu.h"

class FeatureExtractor
{
public:
    // Computes NUM_FEATURES statistics (mean, std, min, max, rms per channel)
    // over a window of raw IMU samples, matching src/preprocessing.py
    // (StandardScaler) followed by src/feature_extraction.py.
    void computeFeatures(const IMUSample window[WINDOW_SIZE], float outFeatures[NUM_FEATURES]);
};

#endif
