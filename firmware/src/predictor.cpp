#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "predictor.h"
#include "model_data.h"

namespace
{
constexpr int kTensorArenaSize = 24 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];

// The model (4x Dense + Softmax) only needs these two ops. Using a minimal
// resolver instead of AllOpsResolver keeps flash usage and compile time down.
constexpr int kNumOps = 2;
tflite::MicroMutableOpResolver<kNumOps> opsResolver;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *inputTensor = nullptr;
TfLiteTensor *outputTensor = nullptr;
} // namespace

bool Predictor::begin()
{
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.println("Model schema version mismatch!");
        return false;
    }

    opsResolver.AddFullyConnected();
    opsResolver.AddSoftmax();

    static tflite::MicroInterpreter staticInterpreter(model, opsResolver, tensorArena, kTensorArenaSize);
    interpreter = &staticInterpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        Serial.println("Failed to allocate tensors!");
        return false;
    }

    inputTensor = interpreter->input(0);
    outputTensor = interpreter->output(0);

    Serial.print("Embedded model size: ");
    Serial.print(model_data_len);
    Serial.println(" bytes");
    Serial.print("Tensor arena used: ");
    Serial.print(interpreter->arena_used_bytes());
    Serial.println(" bytes");

    return true;
}

PredictionResult Predictor::predict(const float features[NUM_FEATURES])
{
    float inputScale = inputTensor->params.scale;
    int32_t inputZeroPoint = inputTensor->params.zero_point;

    for (int i = 0; i < NUM_FEATURES; i++)
    {
        int32_t quantized = static_cast<int32_t>(roundf(features[i] / inputScale)) + inputZeroPoint;
        if (quantized < -128)
            quantized = -128;
        if (quantized > 127)
            quantized = 127;
        inputTensor->data.int8[i] = static_cast<int8_t>(quantized);
    }

    if (interpreter->Invoke() != kTfLiteOk)
    {
        return {"error", 0.0f};
    }

    float outputScale = outputTensor->params.scale;
    int32_t outputZeroPoint = outputTensor->params.zero_point;

    int bestIndex = 0;
    float bestProbability = -1.0f;

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        float probability = (outputTensor->data.int8[i] - outputZeroPoint) * outputScale;
        if (probability > bestProbability)
        {
            bestProbability = probability;
            bestIndex = i;
        }
    }

    return {CLASS_NAMES[bestIndex], bestProbability * 100.0f};
}
