// proto/model.cu - Protocol for Neural CA components
#pragma once
#include <cuda_runtime.h>

struct ModelProtocol {
    float* parameters_d;
    float* gradients_d;
    int num_parameters;

    virtual void forward(float* input, float* output) = 0;
    virtual float effective_rank() = 0;
    virtual float coherence() = 0;
    virtual void update_gradients(float learning_rate) = 0;
};