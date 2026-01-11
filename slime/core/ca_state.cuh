#ifndef CA_STATE_CUH
#define CA_STATE_CUH

#include "../learning/autodiff.cu"
#include "../metrics/hardware_geometry.cu"
#include <cuda_fp16.h>

struct MultiHeadCAState {
    // Weights (FP16 for tensor cores)
    half* perception_weights;
    half* interaction_weights;
    half* value_weights;

    // State fields
    float* ca_concentration;
    float* ca_output;

    // Flow Lenia transport
    float* affinity_reduced;
    float* flow_field;
    float* reintegration_buffer;

    // Tensor core workspaces
    half* fp16_workspace;
    float* fp32_workspace;

    // Autodiff tape (not a pointer - exists)
    ADTape tape;

    // Hardware trace buffer (not a pointer - exists)
    TraceBuffer trace;

    // Saved activations for backprop
    float* perception_saved;
    float* interaction_saved;
    float* pre_gelu_saved;
};

#endif
