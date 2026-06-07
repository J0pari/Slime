// Sheet A-102: Execution Backend — Phase-Major CUDA Graphs
//
// Eight phase graphs replayed by host orchestration. Notes:
//
//   * forward_phase captures BTRAJ at four CA steps (16, 32, 48, 64) instead
//     of only the final step. The additional managed-memory writes are small:
//     4 * 32 * FP32 * POOL_SIZE = 24 KB per generation.
//
//   * world_train_phase covers both placeholder-regressor training and the
//     predictor-organisms' standard CAME path. Predictors train through the
//     same backward/optimizer phases as classifiers; only the loss differs.
//
// The capture/build/launch/destroy helpers below are real CUDA graph API
// usage but capture NOTHING yet — the kernel sequence inside each phase is not
// populated. None of this has been compiled. An optional MPK backend is a
// possible alternative execution path, not implemented here.

#ifndef COEVO_EXECUTION_PHASE_GRAPHS_CU
#define COEVO_EXECUTION_PHASE_GRAPHS_CU

#include "../config/constants.cuh"

#include <cuda_runtime.h>

namespace slime::execution {

// Eight phase graphs (names match A-102).
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

// Capture a single phase from a callback that records device operations
// into the stream. Stream-capture begin/end is symmetric so callers can
// just supply the kernel-launch sequence.
template <typename F>
inline cudaError_t capture_phase(PhaseGraphs* g, Phase p, F&& record_ops) {
    int idx = static_cast<int>(p);
    cudaError_t err = cudaStreamBeginCapture(g->stream,
                                             cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) return err;
    record_ops(g->stream);
    err = cudaStreamEndCapture(g->stream, &g->graphs[idx]);
    if (err != cudaSuccess) return err;
    return cudaGraphInstantiate(&g->execs[idx], g->graphs[idx], nullptr, nullptr, 0);
}

// Build all phase graphs once at startup. Each graph is captured by replaying
// the corresponding kernel sequence in a capture stream. The actual kernel
// launches are wired in during C-001 phase 4; this initialiser allocates
// the stream and zeros the handle table.
inline cudaError_t build_phase_graphs(PhaseGraphs* g) {
    for (int i = 0; i < PHASE_COUNT; ++i) {
        g->graphs[i] = nullptr;
        g->execs[i]  = nullptr;
    }
    return cudaStreamCreate(&g->stream);
}

// Replay a single phase. Host orchestration calls these in sequence each
// generation (see I-001).
inline cudaError_t launch_phase(PhaseGraphs* g, Phase p) {
    cudaGraphExec_t exec = g->execs[static_cast<int>(p)];
    if (!exec) return cudaErrorInvalidValue;
    return cudaGraphLaunch(exec, g->stream);
}

// Release exec + graph handles. Called on shutdown.
inline void destroy_phase_graphs(PhaseGraphs* g) {
    for (int i = 0; i < PHASE_COUNT; ++i) {
        if (g->execs[i])  cudaGraphExecDestroy(g->execs[i]);
        if (g->graphs[i]) cudaGraphDestroy(g->graphs[i]);
    }
    if (g->stream) cudaStreamDestroy(g->stream);
}

}  // namespace slime::execution

#endif  // COEVO_EXECUTION_PHASE_GRAPHS_CU
