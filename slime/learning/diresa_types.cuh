
#ifndef DIRESA_TYPES_CUH
#define DIRESA_TYPES_CUH

#include "../core/organism.cu"
#include <cstdint>

struct DIRESAWeights {
    int input_dim;
    int output_dim;
    int hidden1;
    int hidden2;

    float* encoder_w1;
    float* encoder_b1;
    float* encoder_w2;
    float* encoder_b2;
    float* encoder_w3;
    float* encoder_b3;

    float* decoder_w1;
    float* decoder_b1;
    float* decoder_w2;
    float* decoder_b2;
    float* decoder_w3;
    float* decoder_b3;

    float cov_weight;
    float learning_rate;
    uint32_t training_step;

    float temperature;
    int replica_id;

    float distance_exponent;
    float quality_weight;
};

#endif
