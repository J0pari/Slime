// S-001: Full checkpoint serialization and resume.
//
// Included by integration/host_main.cu after main_loop.cu and alignment.cu.
// Every declared run-state field is serialized, the payload is
// checksummed (FNV-1a 64), the file is schema-hashed and version-checked,
// and the write replaces the previous checkpoint directly (MSVC
// std::filesystem::rename replaces an existing target via
// MoveFileEx(MOVEFILE_REPLACE_EXISTING)), so a crash never destroys the
// last good checkpoint.
//
// Device state saved: shared weights and CAME moments (m, v, c, prev_u).
// Organism grids and checkpoint buffers are NOT saved: the next forward
// re-seeds the grid from inputs, so they carry no state across generations.
//
// Layout (little-endian, raw POD):
//   uint32 magic, uint32 version, uint32 schema_hash
//   uint64 payload_len, uint64 payload_fnv1a64
//   CheckpointHeader (generation, pool_size, archive_size, s_target,
//                     s_target_calibrated, bootstrap_generation)
//   payload:
//     scalars: bootstrap_fired, bootstrap_gen, host_sot_key,
//              grad_health_warn_count, operator paused/requested/n_pruned,
//              pruned_lineages[TOTAL_ORG], rng, last_mean_ce,
//              last_max_abs_logit, s_hist_head, s_hist_filled,
//              s_blended_history[HYBRID_R_WINDOW], n_calibration_samples,
//              calibration_samples[..], predictor_error_ema[POOL_SIZE]
//     ClassifierBatch, PredictorBatch, ProbeSet (with signed tuples)
//     PlaceholderRegressor, PlaceholderReplayBuffer (with held-out flags),
//     CorrelationWindow, CusumState x2, MutationLadder
//     OrganismTable arrays
//     Archive
//     device weights + CAME moments (host-mirrored)

#ifndef COEVO_INTEGRATION_CHECKPOINTING_CU
#define COEVO_INTEGRATION_CHECKPOINTING_CU

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <vector>

namespace slime::integration {

// Schema hash over every serialized type and governing constant. Bump the
// checkpoint version (or this hash) whenever any of these change: a load
// with a mismatched schema refuses loudly instead of reading garbage.
// NOTE: this covers struct SIZES and governing constants; a same-size field
// reorder is invisible to it, so any layout edit requires a
// CHECKPOINT_VERSION bump as well.
inline uint32_t checkpoint_schema_hash_v1() {
    uint32_t h = 0x9E3779B9u;
    auto mix = [&h](uint32_t x) {
        h ^= x + 0x9E3779B9u + (h << 6) + (h >> 2);
    };
    mix(static_cast<uint32_t>(sizeof(genome::Genome)));
    mix(static_cast<uint32_t>(sizeof(genome::DeltaWeights)));
    mix(static_cast<uint32_t>(sizeof(curriculum::ClassifierBatch)));
    mix(static_cast<uint32_t>(sizeof(curriculum::PredictorBatch)));
    mix(static_cast<uint32_t>(sizeof(curriculum::ProbeSet)));
    mix(static_cast<uint32_t>(sizeof(predictor::PlaceholderRegressor)));
    mix(static_cast<uint32_t>(sizeof(predictor::PlaceholderReplayBuffer)));
    mix(static_cast<uint32_t>(sizeof(predictor::CorrelationWindow)));
    mix(static_cast<uint32_t>(sizeof(safety::CusumState)));
    mix(static_cast<uint32_t>(sizeof(safety::pt::MutationLadder)));
    mix(static_cast<uint32_t>(sizeof(archive::Archive)));
    mix(static_cast<uint32_t>(sizeof(nca::OrganismState)));
    mix(static_cast<uint32_t>(TOTAL_WEIGHTS));
    mix(static_cast<uint32_t>(TOTAL_ORG));
    mix(static_cast<uint32_t>(POOL_SIZE));
    mix(static_cast<uint32_t>(GENOME_BITS));
    mix(static_cast<uint32_t>(MAX_ARCHIVE));
    mix(static_cast<uint32_t>(CA_CHANNELS));
    mix(static_cast<uint32_t>(GRID_SIZE));
    mix(static_cast<uint32_t>(BMAP_DIM));
    mix(static_cast<uint32_t>(TASK_EMBED_DIM));
    mix(static_cast<uint32_t>(PROBE_BATCH));
    mix(static_cast<uint32_t>(PT_SWAP_INTERVAL));
    mix(static_cast<uint32_t>(HYBRID_R_WINDOW));
    mix(static_cast<uint32_t>(CALIBRATION_GEN_LO));
    mix(static_cast<uint32_t>(CALIBRATION_GEN_HI));
    return h;
}

namespace ckpt_detail {

inline uint64_t fnv1a64(const uint8_t* data, size_t n, uint64_t h = FNV1A64_OFFSET) {
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= FNV1A64_PRIME;
    }
    return h;
}

struct Buffer {
    std::vector<uint8_t> bytes;
    size_t read_pos = 0;
    bool ok = true;

    // Write path.
    void put(const void* p, size_t n) {
        const uint8_t* src = static_cast<const uint8_t*>(p);
        bytes.insert(bytes.end(), src, src + n);
    }
    template <typename T>
    void put_pod(const T& v) { put(&v, sizeof(T)); }
    template <typename T>
    void put_arr(const T* v, size_t n) { put(v, sizeof(T) * n); }

    // Read path.
    void get(void* p, size_t n) {
        if (!ok || read_pos + n > bytes.size()) { ok = false; return; }
        std::memcpy(p, bytes.data() + read_pos, n);
        read_pos += n;
    }
    template <typename T>
    void get_pod(T& v) { get(&v, sizeof(T)); }
    template <typename T>
    void get_arr(T* v, size_t n) { get(v, sizeof(T) * n); }
};

inline void write_run_state(Buffer& b, const World* w) {
    b.put_pod(static_cast<int32_t>(w->bootstrap_fired ? 1 : 0));
    b.put_pod(static_cast<int32_t>(w->bootstrap_gen));
    b.put_pod(w->host_sot_key);
    b.put_pod(static_cast<int32_t>(w->grad_health_warn_count));
    b.put_pod(static_cast<int32_t>(w->operator_state.paused ? 1 : 0));
    b.put_pod(static_cast<int32_t>(w->operator_state.checkpoint_requested ? 1 : 0));
    b.put_pod(static_cast<int32_t>(w->operator_state.n_pruned));
    b.put_arr(w->operator_state.pruned_lineages, TOTAL_ORG);
    b.put_pod(w->rng.state);
    b.put_pod(w->rng.inc);
    b.put_pod(w->last_mean_ce);
    b.put_pod(w->last_max_abs_logit);
    b.put_pod(static_cast<int32_t>(w->s_hist_head));
    b.put_pod(static_cast<int32_t>(w->s_hist_filled));
    b.put_arr(w->s_blended_history, HYBRID_R_WINDOW);
    b.put_pod(static_cast<int32_t>(w->n_calibration_samples));
    b.put_arr(w->calibration_samples,
              CALIBRATION_GEN_HI - CALIBRATION_GEN_LO + 1);
    b.put_arr(w->predictor_error_ema, POOL_SIZE);
    b.put_arr(w->predictor_loss_ema, POOL_SIZE);

    b.put_pod(w->classifier_batch);
    b.put_pod(w->predictor_batch);
    b.put_pod(w->probe_set);
    b.put_pod(w->placeholder_reg);
    b.put_pod(w->replay_buffer);
    b.put_pod(w->corr_window);
    b.put_pod(w->cusum_surprise);
    b.put_pod(w->cusum_r);
    b.put_pod(w->mutation_ladder);
    b.put_pod(w->stress_ladder);
    b.put_pod(w->sentinel_ens);
    b.put_pod(w->sentinel_history);

    const OrganismTable& t = w->org_table;
    b.put_arr(t.genomes, TOTAL_ORG);
    b.put_arr(t.deltas, TOTAL_ORG);
    b.put_arr(t.lineage_id, TOTAL_ORG);
    b.put_arr(t.parent_id, TOTAL_ORG);
    b.put_arr(t.spawn_gen, TOTAL_ORG);
    b.put_arr(t.replica_tag, TOTAL_ORG);
    b.put_arr(t.fitness, TOTAL_ORG);
    b.put_arr(t.f_raw, TOTAL_ORG);
    b.put_arr(t.f_sot, TOTAL_ORG);
    b.put_arr(t.last_loss, TOTAL_ORG);
    b.put_arr(t.role, TOTAL_ORG);
    b.put_arr(t.batch_sample_idx, POOL_SIZE);

    b.put_pod(w->archive);

    float weights_host[TOTAL_WEIGHTS];
    float came_host[4][TOTAL_WEIGHTS];
    cudaError_t ce = cudaMemcpy(weights_host, w->d_weights,
                                sizeof(float) * TOTAL_WEIGHTS,
                                cudaMemcpyDeviceToHost);
    if (ce == cudaSuccess) ce = cudaMemcpy(came_host[0], w->d_came_m, sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyDeviceToHost);
    if (ce == cudaSuccess) ce = cudaMemcpy(came_host[1], w->d_came_v, sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyDeviceToHost);
    if (ce == cudaSuccess) ce = cudaMemcpy(came_host[2], w->d_came_c, sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyDeviceToHost);
    if (ce == cudaSuccess) ce = cudaMemcpy(came_host[3], w->d_came_prev_u, sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
        std::printf("[FATAL] checkpoint: device readback failed: %s\n",
                    cudaGetErrorString(ce));
        b.ok = false;
        return;
    }
    b.put_arr(weights_host, TOTAL_WEIGHTS);
    for (int i = 0; i < 4; ++i) b.put_arr(came_host[i], TOTAL_WEIGHTS);
}

inline bool read_run_state(Buffer& b, World* w) {
    int32_t flag = 0;
    b.get_pod(flag); w->bootstrap_fired = (flag != 0);
    b.get_pod(w->bootstrap_gen);
    b.get_pod(w->host_sot_key);
    b.get_pod(w->grad_health_warn_count);
    b.get_pod(flag); w->operator_state.paused = (flag != 0);
    b.get_pod(flag); w->operator_state.checkpoint_requested = (flag != 0);
    b.get_pod(w->operator_state.n_pruned);
    if (w->operator_state.n_pruned < 0) w->operator_state.n_pruned = 0;
    if (w->operator_state.n_pruned > TOTAL_ORG) w->operator_state.n_pruned = TOTAL_ORG;
    b.get_arr(w->operator_state.pruned_lineages, TOTAL_ORG);
    b.get_pod(w->rng.state);
    b.get_pod(w->rng.inc);
    b.get_pod(w->last_mean_ce);
    b.get_pod(w->last_max_abs_logit);
    b.get_pod(w->s_hist_head);
    b.get_pod(w->s_hist_filled);
    if (w->s_hist_head < 0 || w->s_hist_head >= HYBRID_R_WINDOW) w->s_hist_head = 0;
    if (w->s_hist_filled < 0 || w->s_hist_filled > HYBRID_R_WINDOW) w->s_hist_filled = 0;
    b.get_arr(w->s_blended_history, HYBRID_R_WINDOW);
    b.get_pod(w->n_calibration_samples);
    if (w->n_calibration_samples < 0 ||
        w->n_calibration_samples > CALIBRATION_GEN_HI - CALIBRATION_GEN_LO + 1) {
        w->n_calibration_samples = 0;
    }
    b.get_arr(w->calibration_samples, CALIBRATION_GEN_HI - CALIBRATION_GEN_LO + 1);
    b.get_arr(w->predictor_error_ema, POOL_SIZE);
    b.get_arr(w->predictor_loss_ema, POOL_SIZE);

    b.get_pod(w->classifier_batch);
    b.get_pod(w->predictor_batch);
    b.get_pod(w->probe_set);
    b.get_pod(w->placeholder_reg);
    b.get_pod(w->replay_buffer);
    b.get_pod(w->corr_window);
    b.get_pod(w->cusum_surprise);
    b.get_pod(w->cusum_r);
    b.get_pod(w->mutation_ladder);
    b.get_pod(w->stress_ladder);
    b.get_pod(w->sentinel_ens);
    b.get_pod(w->sentinel_history);

    OrganismTable& t = w->org_table;
    b.get_arr(t.genomes, TOTAL_ORG);
    b.get_arr(t.deltas, TOTAL_ORG);
    b.get_arr(t.lineage_id, TOTAL_ORG);
    b.get_arr(t.parent_id, TOTAL_ORG);
    b.get_arr(t.spawn_gen, TOTAL_ORG);
    b.get_arr(t.replica_tag, TOTAL_ORG);
    b.get_arr(t.fitness, TOTAL_ORG);
    b.get_arr(t.f_raw, TOTAL_ORG);
    b.get_arr(t.f_sot, TOTAL_ORG);
    b.get_arr(t.last_loss, TOTAL_ORG);
    b.get_arr(t.role, TOTAL_ORG);
    b.get_arr(t.batch_sample_idx, POOL_SIZE);

    b.get_pod(w->archive);

    float weights_host[TOTAL_WEIGHTS];
    float came_host[4][TOTAL_WEIGHTS];
    b.get_arr(weights_host, TOTAL_WEIGHTS);
    for (int i = 0; i < 4; ++i) b.get_arr(came_host[i], TOTAL_WEIGHTS);
    if (!b.ok) return false;

    cudaError_t ce = cudaMemcpy(w->d_weights, weights_host,
                                sizeof(float) * TOTAL_WEIGHTS,
                                cudaMemcpyHostToDevice);
    if (ce == cudaSuccess) ce = cudaMemcpy(w->d_came_m, came_host[0], sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyHostToDevice);
    if (ce == cudaSuccess) ce = cudaMemcpy(w->d_came_v, came_host[1], sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyHostToDevice);
    if (ce == cudaSuccess) ce = cudaMemcpy(w->d_came_c, came_host[2], sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyHostToDevice);
    if (ce == cudaSuccess) ce = cudaMemcpy(w->d_came_prev_u, came_host[3], sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) {
        std::printf("[FATAL] resume: device restore failed: %s\n",
                    cudaGetErrorString(ce));
        return false;
    }
    return true;
}

}  // namespace ckpt_detail

// Save the full run state to `path`. The payload is checksummed; the file
// replaces the target directly (no remove window). Returns false (with a
// report) on any failure; the previous checkpoint is untouched on failure.
inline bool save_checkpoint(World* w, const char* path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path target(path);
    if (target.has_parent_path()) {
        fs::create_directories(target.parent_path(), ec);
        if (ec) {
            std::printf("[FATAL] checkpoint: cannot create %s: %s\n",
                        target.parent_path().string().c_str(), ec.message().c_str());
            return false;
        }
    }
    fs::path tmp = target;
    tmp += ".tmp";

    ckpt_detail::Buffer payload;
    ckpt_detail::write_run_state(payload, w);
    if (!payload.ok) {
        std::remove(tmp.string().c_str());
        return false;
    }
    uint64_t checksum = ckpt_detail::fnv1a64(payload.bytes.data(),
                                             payload.bytes.size());

    FILE* f = std::fopen(tmp.string().c_str(), "wb");
    if (!f) {
        std::printf("[FATAL] checkpoint: cannot open %s\n", tmp.string().c_str());
        return false;
    }
    safety::CheckpointHeader hdr{};
    hdr.generation = w->generation;
    hdr.pool_size = POOL_SIZE;
    hdr.archive_size = archive::archive_size(w->archive);
    hdr.s_target = w->s_target;
    hdr.s_target_calibrated = w->s_target_calibrated;
    hdr.bootstrap_generation = w->bootstrap_gen;

    uint32_t magic = safety::CHECKPOINT_MAGIC;
    uint32_t version = safety::CHECKPOINT_VERSION;
    uint32_t schema = checkpoint_schema_hash_v1();
    uint64_t payload_len = static_cast<uint64_t>(payload.bytes.size());

    bool ok = true;
    ok = ok && std::fwrite(&magic, sizeof(magic), 1, f) == 1;
    ok = ok && std::fwrite(&version, sizeof(version), 1, f) == 1;
    ok = ok && std::fwrite(&schema, sizeof(schema), 1, f) == 1;
    ok = ok && std::fwrite(&payload_len, sizeof(payload_len), 1, f) == 1;
    ok = ok && std::fwrite(&checksum, sizeof(checksum), 1, f) == 1;
    ok = ok && std::fwrite(&hdr, sizeof(hdr), 1, f) == 1;
    ok = ok && std::fwrite(payload.bytes.data(), 1, payload.bytes.size(), f)
               == payload.bytes.size();
    if (std::fflush(f) != 0) ok = false;
    if (std::fclose(f) != 0) ok = false;
    if (!ok) {
        std::printf("[FATAL] checkpoint: write failed for %s\n",
                    target.string().c_str());
        std::remove(tmp.string().c_str());
        return false;
    }

    // Direct replace: MSVC std::filesystem::rename replaces an existing
    // target (MoveFileEx with MOVEFILE_REPLACE_EXISTING). If that fails the
    // previous checkpoint is left untouched and the save is a loud failure:
    // there is no remove-then-rename fallback, because the window between a
    // remove and a rename can destroy the only good checkpoint.
    fs::rename(tmp, target, ec);
    if (ec) {
        std::printf("[FATAL] checkpoint: replace failed for %s: %s "
                    "(previous checkpoint left intact)\n",
                    target.string().c_str(), ec.message().c_str());
        std::remove(tmp.string().c_str());
        return false;
    }
    return true;
}

// Load a checkpoint into an initialized World. Returns false (with a report)
// on a missing/corrupt/incompatible file; the World must not be used after a
// failed load.
inline bool load_checkpoint(World* w, const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::printf("[FATAL] resume: checkpoint not found: %s\n", path);
        return false;
    }
    uint32_t magic = 0, version = 0, schema = 0;
    uint64_t payload_len = 0, checksum = 0;
    safety::CheckpointHeader hdr{};
    bool ok = true;
    ok = ok && std::fread(&magic, sizeof(magic), 1, f) == 1;
    ok = ok && std::fread(&version, sizeof(version), 1, f) == 1;
    ok = ok && std::fread(&schema, sizeof(schema), 1, f) == 1;
    ok = ok && std::fread(&payload_len, sizeof(payload_len), 1, f) == 1;
    ok = ok && std::fread(&checksum, sizeof(checksum), 1, f) == 1;
    ok = ok && std::fread(&hdr, sizeof(hdr), 1, f) == 1;
    if (!ok || magic != safety::CHECKPOINT_MAGIC) {
        std::printf("[FATAL] resume: %s is not a Slime checkpoint\n", path);
        std::fclose(f);
        return false;
    }
    if (version != safety::CHECKPOINT_VERSION) {
        std::printf("[FATAL] resume: checkpoint version %u, expected %u\n",
                    version, safety::CHECKPOINT_VERSION);
        std::fclose(f);
        return false;
    }
    if (schema != checkpoint_schema_hash_v1()) {
        std::printf("[FATAL] resume: checkpoint schema %08x, expected %08x "
                    "(struct or constant drift)\n",
                    schema, checkpoint_schema_hash_v1());
        std::fclose(f);
        return false;
    }
    if (hdr.pool_size != POOL_SIZE) {
        std::printf("[FATAL] resume: pool size mismatch (%d vs %d)\n",
                    hdr.pool_size, POOL_SIZE);
        std::fclose(f);
        return false;
    }

    ckpt_detail::Buffer payload;
    payload.bytes.resize(payload_len);
    if (payload_len > 0 &&
        std::fread(payload.bytes.data(), 1, payload_len, f) != payload_len) {
        std::printf("[FATAL] resume: truncated checkpoint %s\n", path);
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    uint64_t observed = ckpt_detail::fnv1a64(payload.bytes.data(), payload_len);
    if (observed != checksum) {
        std::printf("[FATAL] resume: checkpoint checksum mismatch "
                    "(expected %016llx, observed %016llx)\n",
                    static_cast<unsigned long long>(checksum),
                    static_cast<unsigned long long>(observed));
        return false;
    }

    if (!ckpt_detail::read_run_state(payload, w)) {
        std::printf("[FATAL] resume: corrupt checkpoint payload %s\n", path);
        return false;
    }

    w->generation = hdr.generation;
    w->s_target = hdr.s_target;
    w->s_target_calibrated = hdr.s_target_calibrated;
    w->bootstrap_gen = hdr.bootstrap_generation;
    // A resumed run must not start paused; a checkpoint request is consumed.
    w->operator_state.paused = false;
    w->operator_state.checkpoint_requested = false;
    w->grad_health_warn_count = 0;

    // The loaded archive must satisfy its own invariants.
    char err[256];
    if (!archive::archive_check_invariants(w->archive, err, sizeof(err))) {
        std::printf("[FATAL] resume: archive invariant violated after load: %s\n", err);
        return false;
    }
    return true;
}

}  // namespace slime::integration

#endif  // COEVO_INTEGRATION_CHECKPOINTING_CU

