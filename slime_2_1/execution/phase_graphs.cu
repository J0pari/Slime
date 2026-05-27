// Sheet A-102: Execution Backend — Phase-Major CUDA Graphs
//
// Unchanged from 2.0 in structure. Eight phase graphs replayed by host
// orchestration. Two 2.1-specific notes:
//
//   * forward_phase captures BTRAJ at four CA steps (16, 32, 48, 64) instead
//     of only the final step. The additional managed-memory writes are small:
//     4 * 32 * FP32 * POOL_SIZE = 24 KB per generation.
//
//   * world_train_phase covers both placeholder-regressor training and the
//     predictor-organisms' standard CAME path. Predictors train through the
//     same backward/optimizer phases as classifiers; only the loss differs.
//
// MPK experimental backend unchanged.

#ifndef SLIME_2_1_EXECUTION_PHASE_GRAPHS_CU
#define SLIME_2_1_EXECUTION_PHASE_GRAPHS_CU

#include "../config/constants.cuh"

#include <cuda_runtime.h>

namespace slime::execution {

// Eight phase graphs (names match A-102 from 2.0).
enum class Phase : int {
    Curriculum    = 0,
    Decode        = 1,
    Forward       = 2,    // BTRAJ capture inside
    Archive       = 3,
    WorldPredict  = 4,    // placeholder + predictor ensemble surprise
    Backward      = 5,
    Optimizer     = 6,
    WorldTrain    = 7,    // placeholder regressor only (predictors via main path)
    StressEval    = 8,    // S-004 SOT-density ladder, sampled
    Housekeeping  = 9,    // sentinels, lineage stats, hybrid r update
};

constexpr int PHASE_COUNT = 10;

struct PhaseGraphs {
    cudaGraph_t      graphs[PHASE_COUNT];
    cudaGraphExec_t  execs[PHASE_COUNT];
    cudaStream_t     stream;
};

// Build all phase graphs once at startup. Each graph is captured by replaying
// the corresponding kernel sequence in a capture stream.
void build_phase_graphs(PhaseGraphs* g);

// Replay a single phase. Host orchestration calls these in sequence each
// generation (see I-001).
inline cudaError_t launch_phase(PhaseGraphs* g, Phase p) {
    return cudaGraphLaunch(g->execs[static_cast<int>(p)], g->stream);
}

// Release exec + graph handles. Called on shutdown.
void destroy_phase_graphs(PhaseGraphs* g);

}  // namespace slime::execution

#endif  // SLIME_2_1_EXECUTION_PHASE_GRAPHS_CU
