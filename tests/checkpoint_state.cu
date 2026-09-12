// S-001 checkpoint state roundtrip (GPU component test).
//
// Build: make checkpoint-state-test
//
// Runs two generations, saves a checkpoint, loads it into a second World,
// and requires the loaded host state and device state (weights, CAME
// moments) to be byte-identical to the original. Then continues the loaded
// World one generation to prove a resumed run advances normally.

#define COEVO_NO_MAIN
#include "../integration/host_main.cu"

#include <cstdio>
#include <cstring>

using namespace slime;
using namespace slime::integration;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s (line %d)\n", msg, __LINE__); g_fail++; } \
    else { g_pass++; } \
} while (0)

#define CUDA_OK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::printf("CUDA FAIL: %s at line %d\n", cudaGetErrorString(_e), __LINE__); \
        return 1; \
    } \
} while (0)

static bool memeq(const void* a, const void* b, size_t n) {
    return std::memcmp(a, b, n) == 0;
}

int main() {
    // [claim:S001.checkpoint-roundtrip]
    // [claim:S002.operator-command-effective]
    std::printf("Checkpoint state roundtrip test\n");
    std::printf("================================\n");
    std::fflush(stdout);

    const char* ckpt = "checkpoints/test-ckpt.bin";
    std::remove(ckpt);

    World* a = new World;
    if (!initialize_world(a)) { std::printf("init A failed\n"); return 1; }
    a->checkpoint_path = ckpt;

    if (!step_generation(a)) { std::printf("step A gen 0 failed\n"); return 1; }
    if (!step_generation(a)) { std::printf("step A gen 1 failed\n"); return 1; }
    CHECK(a->generation == 2, "two generations advanced");

    // The per-generation save must exist after stepping.
    FILE* probe = std::fopen(ckpt, "rb");
    CHECK(probe != nullptr, "checkpoint file written");
    if (probe) std::fclose(probe);

    World* b = new World;
    if (!initialize_world(b)) { std::printf("init B failed\n"); return 1; }
    CHECK(load_checkpoint(b, ckpt), "checkpoint loads");

    // ---- Host state: byte-identical ----
    CHECK(memeq(&a->org_table, &b->org_table, sizeof(OrganismTable)),
          "organism table identical");
    CHECK(memeq(&a->archive, &b->archive, sizeof(archive::Archive)),
          "archive identical");
    CHECK(memeq(&a->mutation_ladder, &b->mutation_ladder, sizeof(a->mutation_ladder)),
          "mutation ladder identical");
    CHECK(memeq(&a->cusum_surprise, &b->cusum_surprise, sizeof(a->cusum_surprise)) &&
          memeq(&a->cusum_r, &b->cusum_r, sizeof(a->cusum_r)),
          "CUSUM states identical");
    CHECK(memeq(&a->placeholder_reg, &b->placeholder_reg, sizeof(a->placeholder_reg)),
          "placeholder regressor identical");
    CHECK(memeq(&a->replay_buffer, &b->replay_buffer, sizeof(a->replay_buffer)),
          "replay buffer identical");
    CHECK(memeq(&a->corr_window, &b->corr_window, sizeof(a->corr_window)),
          "correlation window identical");
    CHECK(memeq(&a->probe_set, &b->probe_set, sizeof(a->probe_set)),
          "probe set identical");
    CHECK(memeq(a->probe_fitness, b->probe_fitness, sizeof(a->probe_fitness)),
          "probe fitness identical");
    CHECK(memeq(&a->classifier_batch, &b->classifier_batch, sizeof(a->classifier_batch)),
          "classifier batch identical");
    CHECK(a->generation == b->generation &&
          a->bootstrap_fired == b->bootstrap_fired &&
          a->bootstrap_gen == b->bootstrap_gen &&
          a->s_target == b->s_target &&
          a->s_target_calibrated == b->s_target_calibrated &&
          a->host_sot_key == b->host_sot_key &&
          a->rng.state == b->rng.state && a->rng.inc == b->rng.inc,
          "scalars and RNG identical");
    CHECK(a->operator_state.n_pruned == b->operator_state.n_pruned &&
          memeq(a->operator_state.pruned_lineages, b->operator_state.pruned_lineages,
                sizeof(a->operator_state.pruned_lineages)),
          "operator state identical");

    // ---- Device state: byte-identical ----
    {
        float* wa = (float*)malloc(sizeof(float) * TOTAL_WEIGHTS);
        float* wb = (float*)malloc(sizeof(float) * TOTAL_WEIGHTS);
        CUDA_OK(cudaMemcpy(wa, a->d_weights, sizeof(float) * TOTAL_WEIGHTS,
                           cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(wb, b->d_weights, sizeof(float) * TOTAL_WEIGHTS,
                           cudaMemcpyDeviceToHost));
        CHECK(memeq(wa, wb, sizeof(float) * TOTAL_WEIGHTS), "weights identical");
        free(wa); free(wb);

        const char* came_names[4] = { "m", "v", "c", "prev_u" };
        float* ca = (float*)malloc(sizeof(float) * TOTAL_WEIGHTS);
        float* cb = (float*)malloc(sizeof(float) * TOTAL_WEIGHTS);
        float* src_a[4] = { a->d_came_m, a->d_came_v, a->d_came_c, a->d_came_prev_u };
        float* src_b[4] = { b->d_came_m, b->d_came_v, b->d_came_c, b->d_came_prev_u };
        for (int i = 0; i < 4; ++i) {
            CUDA_OK(cudaMemcpy(ca, src_a[i], sizeof(float) * TOTAL_WEIGHTS,
                               cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(cb, src_b[i], sizeof(float) * TOTAL_WEIGHTS,
                               cudaMemcpyDeviceToHost));
            char msg[64];
            std::snprintf(msg, sizeof(msg), "CAME %s identical", came_names[i]);
            CHECK(memeq(ca, cb, sizeof(float) * TOTAL_WEIGHTS), msg);
        }
        free(ca); free(cb);
    }

    // ---- Functional resume: the loaded World advances ----
    b->checkpoint_path = ckpt;
    bool continued = step_generation(b);
    CHECK(continued, "resumed run advances one generation");
    CHECK(b->generation == 3, "resumed generation counter continues");

    std::remove(ckpt);
    free_gpu_buffers(a);
    free_gpu_buffers(b);
    delete a;
    delete b;

    std::printf("\n================================\n");
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    if (g_fail > 0) {
        std::printf("CHECKPOINT STATE: FAIL\n");
        return 1;
    }
    std::printf("CHECKPOINT STATE: PASS\n");
    return 0;
}

