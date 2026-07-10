#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "model_config.h"

struct PredictionResult
{
    const char *label;
    float confidence; // 0-100
};

class Predictor
{
public:
    bool begin();
    PredictionResult predict(const float features[NUM_FEATURES]);
};

#endif
