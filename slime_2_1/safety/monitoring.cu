// Sheet S-001: Monitoring, Checkpointing & Resilience
//
// Unchanged from 2.0 in structure. Checkpoint state additionally includes:
//   * role tags of all organisms
//   * calibrated s_target value (frozen at first bootstrap crossing)
//   * rolling correlation window state for hybrid blending (A-601)
//
// CUSUM operates on blended surprise (A-601). A companion CUSUM on r itself
// (the hybrid blending weight) raises an alert if correlation collapses.

#ifndef SLIME_2_1_SAFETY_MONITORING_CU
#define SLIME_2_1_SAFETY_MONITORING_CU

#include "../config/constants.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace slime::safety {

struct CusumState {
    float upper;
    float lower;
    float reference;
    float allowance;
    float threshold;
    int   alert_count;
};

struct CheckpointHeader {
    int      generation;
    int      pool_size;
    int      archive_size;
    float    s_target;            // frozen at first bootstrap crossing
    bool     s_target_calibrated;
    int      bootstrap_generation; // generation at which the bootstrap fired
};

// Telemetry payload flushed per generation (host-side aggregation).
struct GenerationTelemetry {
    int   generation;
    float s_blended;
    float s_placeholder;
    float s_predictor;
    float r;                       // hybrid blending weight
    float rho;                     // s_avg / s_target
    int   classifier_count;
    int   predictor_count;
    float l_role_accuracy;
    float swap_accept_rate;
    int   stress_failure_lineages;
};

// CUSUM update used for blended-surprise drift detection and for r itself.
__host__ __device__ void cusum_update(CusumState* s, float x);

// Checkpoint write: writes header + entire population state, archive,
// placeholder regressor + AdamW state, correlation window, PT replica
// assignments, sentinel state. Atomic-replace on success.
void write_checkpoint(const CheckpointHeader& hdr,
                      const char* path);

// Restore inverse. Returns false if signature or schema mismatches.
bool load_checkpoint(CheckpointHeader* hdr_out,
                     const char* path);

}  // namespace slime::safety

#endif  // SLIME_2_1_SAFETY_MONITORING_CU
