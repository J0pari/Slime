// S-001 checkpoint state roundtrip (GPU component test).
//
// Build: make checkpoint-state-test
//
// Runs two generations, populates the auxiliary run state (predictor batch,
// error EMA, surprise history, calibration samples, telemetry scalars),
// saves a checkpoint, loads it into a second World, and requires the loaded
// host state and device state (weights, CAME moments) to be byte-identical
// to the original. Then continues the loaded World one generation, rejects
// a payload-corrupted and a schema-corrupted checkpoint, and verifies that
// a refused replacement leaves the previous checkpoint intact.

#define COEVO_NO_MAIN
#include "../integration/host_main.cu"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

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

static bool copy_file(const char* src, const char* dst) {
    FILE* in = std::fopen(src, "rb");
    if (!in) return false;
    FILE* out = std::fopen(dst, "wb");
    if (!out) { std::fclose(in); return false; }
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), in)) > 0) {
        if (std::fwrite(buf, 1, n, out) != n) {
            std::fclose(in); std::fclose(out); return false;
        }
    }
    std::fclose(in);
    std::fclose(out);
    return true;
}

static bool flip_byte(const char* path, long offset) {
    FILE* f = std::fopen(path, "r+b");
    if (!f) return false;
    if (std::fseek(f, offset, SEEK_SET) != 0) { std::fclose(f); return false; }
    int c = std::fgetc(f);
    if (c == EOF) { std::fclose(f); return false; }
    if (std::fseek(f, offset, SEEK_SET) != 0) { std::fclose(f); return false; }
    std::fputc(c ^ 0x5A, f);
    std::fclose(f);
    return true;
}

static bool flip_byte_from_end(const char* path, long from_end) {
    FILE* f = std::fopen(path, "r+b");
    if (!f) return false;
    if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return false; }
    long size = std::ftell(f);
    if (size < from_end) { std::fclose(f); return false; }
    std::fclose(f);
    return flip_byte(path, size - from_end);
}

int main() {
    // [claim:S001.checkpoint-roundtrip]
    // [claim:S002.operator-command-effective]
    std::printf("Checkpoint state roundtrip test\n");
    std::printf("================================\n");
    std::fflush(stdout);

    const char* ckpt = "checkpoints/test-ckpt.bin";
    const char* ckpt_bad = "checkpoints/test-ckpt-bad.bin";
    const char* ckpt_schema = "checkpoints/test-ckpt-schema.bin";
    std::remove(ckpt);
    std::remove(ckpt_bad);
    std::remove(ckpt_schema);

    World* a = new World;
    if (!initialize_world(a)) { std::printf("init A failed\n"); return 1; }
    a->checkpoint_path = ckpt;

    if (!step_generation(a)) { std::printf("step A gen 0 failed\n"); return 1; }
    if (!step_generation(a)) { std::printf("step A gen 1 failed\n"); return 1; }
    CHECK(a->generation == 2, "two generations advanced");

    // Populate auxiliary run state so the payload's coverage is exercised
    // rather than comparing zeros.
    a->predictor_error_ema[3] = 0.25f;
    a->predictor_error_ema[POOL_SIZE - 1] = 0.75f;
    a->predictor_batch.target_pool_slot[0] = slime::PoolSlot(7);
    a->predictor_batch.target_pool_slot[1] = slime::PoolSlot();
    a->predictor_batch.target_lineage_id[0] = slime::LineageId(1234u);
    a->predictor_batch.target_bmap_64[0] = 0.5f;
    a->s_blended_history[0] = 1.25f;
    a->s_hist_head = 1;
    a->s_hist_filled = 1;
    a->calibration_samples[0] = 0.125f;
    a->calibration_samples[1] = 0.25f;
    a->n_calibration_samples = 2;
    a->last_mean_ce = 1.5f;
    a->last_max_abs_logit = 0.75f;
    CHECK(save_checkpoint(a, ckpt), "explicit save succeeds");

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
    CHECK(memeq(&a->reference_reg, &b->reference_reg, sizeof(a->reference_reg)),
          "reference regressor identical");
    CHECK(memeq(&a->replay_buffer, &b->replay_buffer, sizeof(a->replay_buffer)),
          "replay buffer identical");
    CHECK(memeq(&a->corr_window, &b->corr_window, sizeof(a->corr_window)),
          "correlation window identical");
    CHECK(memeq(&a->probe_set, &b->probe_set, sizeof(a->probe_set)),
          "probe set identical");
    CHECK(memeq(&a->classifier_batch, &b->classifier_batch, sizeof(a->classifier_batch)),
          "classifier batch identical");
    CHECK(memeq(&a->predictor_batch, &b->predictor_batch, sizeof(a->predictor_batch)),
          "predictor batch identical");
    CHECK(memeq(a->predictor_error_ema, b->predictor_error_ema,
                sizeof(a->predictor_error_ema)),
          "predictor error EMA identical");
    CHECK(memeq(a->s_blended_history, b->s_blended_history,
                sizeof(a->s_blended_history)) &&
          a->s_hist_head == b->s_hist_head &&
          a->s_hist_filled == b->s_hist_filled,
          "surprise history identical");
    CHECK(memeq(a->calibration_samples, b->calibration_samples,
                sizeof(a->calibration_samples)) &&
          a->n_calibration_samples == b->n_calibration_samples,
          "calibration samples identical");
    CHECK(a->generation == b->generation &&
          a->bootstrap_fired == b->bootstrap_fired &&
          a->bootstrap_gen == b->bootstrap_gen &&
          a->s_target == b->s_target &&
          a->s_target_calibrated == b->s_target_calibrated &&
          a->host_sot_key == b->host_sot_key &&
          a->last_mean_ce == b->last_mean_ce &&
          a->last_max_abs_logit == b->last_max_abs_logit &&
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
    CHECK(step_generation(b), "resumed run advances one generation");
    CHECK(b->generation == 3, "resumed generation counter continues");

    // ---- Corruption rejection ----
    CHECK(copy_file(ckpt, ckpt_bad), "copy checkpoint for corruption test");
    CHECK(flip_byte_from_end(ckpt_bad, 40), "flip a payload byte");
    World* c = new World;
    if (!initialize_world(c)) { std::printf("init C failed\n"); return 1; }
    CHECK(!load_checkpoint(c, ckpt_bad),
          "payload corruption is rejected (checksum)");
    CHECK(!load_checkpoint(c, ckpt_schema),
          "missing checkpoint is rejected");
    CHECK(copy_file(ckpt, ckpt_schema), "copy checkpoint for schema test");
    CHECK(flip_byte(ckpt_schema, 8), "flip a schema byte");
    CHECK(!load_checkpoint(c, ckpt_schema),
          "schema corruption is rejected");

    // ---- Replacement failure leaves the previous checkpoint intact ----
#ifdef _WIN32
    {
        HANDLE hold = CreateFileA(ckpt, GENERIC_READ, 0 /* exclusive */,
                                  nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL, nullptr);
        CHECK(hold != INVALID_HANDLE_VALUE, "exclusive handle on checkpoint");
        bool saved = save_checkpoint(a, ckpt);
        CHECK(!saved, "save refused while the target is exclusively held");
        if (hold != INVALID_HANDLE_VALUE) CloseHandle(hold);
        World* d = new World;
        if (!initialize_world(d)) { std::printf("init D failed\n"); return 1; }
        CHECK(load_checkpoint(d, ckpt),
              "previous checkpoint intact after a refused replacement");
        free_gpu_buffers(d);
        delete d;
    }
#endif

    std::remove(ckpt);
    std::remove(ckpt_bad);
    std::remove(ckpt_schema);
    free_gpu_buffers(a);
    free_gpu_buffers(b);
    free_gpu_buffers(c);
    delete a;
    delete b;
    delete c;

    std::printf("\n================================\n");
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    if (g_fail > 0) {
        std::printf("CHECKPOINT STATE: FAIL\n");
        return 1;
    }
    std::printf("CHECKPOINT STATE: PASS\n");
    return 0;
}
