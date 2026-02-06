#ifndef CA_STATE_CUH
#define CA_STATE_CUH

#include "../learning/autodiff.cu"
#include "../metrics/hardware_geometry.cu"
#include <cuda_fp16.h>

struct MultiHeadCAState {
    half* perception_weights;
    half* interaction_weights;
    half* value_weights;

    float* ca_concentration;
    float* ca_output;

    float* affinity_reduced;
    float* flow_field;
    float* reintegration_buffer;

    half* fp16_workspace;
    float* fp32_workspace;

    ADTape tape;

    TraceBuffer trace;

    float* perception_saved;
    float* interaction_saved;
    float* pre_gelu_saved;
};

#endif
