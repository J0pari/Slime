// proto/component.cu - Protocol for pooled components
#pragma once
#include <cuda_runtime.h>

struct ComponentProtocol {
    int id;
    float fitness;
    float coherence;
    float hunger;

    virtual void reset() = 0;
    virtual void to_dict(char* buffer) = 0;
    virtual void from_dict(const char* buffer) = 0;
    virtual float compute_fitness() = 0;
};