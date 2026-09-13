// Phase graphs (A-102, I7).
//
// Capturable phases are pure device launch sequences with stable arguments;
// host-interleaved phases (SOT reference, scoring, archive, telemetry
// readbacks) are not captured. Capture happens on the first execution of a
// phase (stream capture does not execute the launches, so the first
// generation captures and immediately replays); subsequent generations
// replay the instantiated graph.
//
// Debug mode (COEVO_PHASE_GRAPH_DEBUG=1) runs the first execution
// sequentially AND through the captured graph and compares an
// output checksum, so a capture that changes behavior fails loudly.

#ifndef COEVO_INTEGRATION_PHASE_GRAPH_CUH
#define COEVO_INTEGRATION_PHASE_GRAPH_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace slime::integration {

struct PhaseGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool captured = false;
    int  replays = 0;
};

inline bool phase_graph_debug() {
    const char* v = std::getenv("COEVO_PHASE_GRAPH_DEBUG");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

inline bool phase_begin_capture(cudaStream_t stream) {
    cudaError_t e = cudaStreamBeginCapture(stream,
                                           cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) {
        std::printf("[FATAL] CUDA phase capture begin failed: %s\n",
                    cudaGetErrorString(e));
        return false;
    }
    return true;
}

inline bool phase_end_capture(PhaseGraph* pg, cudaStream_t stream) {
    cudaError_t e = cudaStreamEndCapture(stream, &pg->graph);
    if (e == cudaSuccess) {
        e = cudaGraphInstantiate(&pg->exec, pg->graph, nullptr, nullptr, 0);
    }
    if (e != cudaSuccess) {
        std::printf("[FATAL] CUDA phase capture end failed: %s\n",
                    cudaGetErrorString(e));
        return false;
    }
    pg->captured = true;
    return true;
}

inline bool phase_replay(PhaseGraph* pg, cudaStream_t stream) {
    cudaError_t e = cudaGraphLaunch(pg->exec, stream);
    if (e != cudaSuccess) {
        std::printf("[FATAL] CUDA phase graph launch failed: %s\n",
                    cudaGetErrorString(e));
        return false;
    }
    pg->replays++;
    return true;
}

// Run a phase: capture-and-replay on first use, replay afterwards. The
// optional `validate` callback runs after each execution and may compare
// against state it keeps itself (used by the debug mode).
template <typename LaunchFn, typename ValidateFn>
inline bool phase_run(PhaseGraph* pg, cudaStream_t stream,
                      LaunchFn&& launches, ValidateFn&& validate) {
    if (pg->captured) {
        if (!phase_replay(pg, stream)) return false;
        if (phase_graph_debug()) validate();
        return true;
    }
    if (phase_graph_debug()) {
        // Sequential execution for the debug comparison, then capture (which
        // does not execute), then replay the captured graph.
        launches();
        validate();
    }
    if (!phase_begin_capture(stream)) return false;
    launches();
    if (!phase_end_capture(pg, stream)) return false;
    if (!phase_replay(pg, stream)) return false;
    if (phase_graph_debug()) validate();
    return true;
}

template <typename LaunchFn>
inline bool phase_run(PhaseGraph* pg, cudaStream_t stream,
                      LaunchFn&& launches) {
    return phase_run(pg, stream, launches, [] {});
}

}  // namespace slime::integration

#endif  // COEVO_INTEGRATION_PHASE_GRAPH_CUH
