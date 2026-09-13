// GPU authorization guard (GPU scheduling policy).
//
// Every GPU process must be launched by the scheduler path: a job submitted
// through `architecture/gpu_client.py` (the scheduler sets the marker in the
// job environment) or a `--direct` launch (the client sets the marker and
// holds the GPU lock). A bare launch would compete with a scheduled job and
// can kill it, so the GPU binaries refuse to start without the marker. The
// marker is set only by the client; see AGENTS.md "GPU scheduling".

#ifndef COEVO_CONFIG_GPU_AUTHORIZATION_CUH
#define COEVO_CONFIG_GPU_AUTHORIZATION_CUH

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace slime {

inline bool gpu_authorized() {
    const char* v = std::getenv("COEVO_GPU_AUTHORIZED");
    return v != nullptr && std::strcmp(v, "1") == 0;
}

inline void require_gpu_authorization(const char* program) {
    if (gpu_authorized()) return;
    std::printf(
        "[FATAL] %s refuses to run: GPU work goes through the scheduler "
        "(submit with `python architecture/gpu_client.py run ...`, or use "
        "`--direct`, which holds the GPU lock). A bare launch competes with "
        "scheduled jobs and can kill them.\n", program);
    std::fflush(stdout);
    std::exit(2);
}

}  // namespace slime

#endif  // COEVO_CONFIG_GPU_AUTHORIZATION_CUH
