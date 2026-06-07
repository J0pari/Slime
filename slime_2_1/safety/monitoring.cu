// Sheet S-001: Monitoring, Checkpointing & Resilience
//
// Checkpoint state includes, alongside the population/archive/optimizer state:
//   * role tags of all organisms
//   * calibrated s_target value (frozen at first bootstrap crossing)
//   * rolling correlation window state for hybrid blending (A-601)
//
// CUSUM operates on blended surprise (A-601). A companion CUSUM on r itself
// (the hybrid blending weight) raises an alert if correlation collapses.

#ifndef COEVO_SAFETY_MONITORING_CU
#define COEVO_SAFETY_MONITORING_CU

#include "../config/constants.cuh"

#include <cstdint>
#include <cstdio>
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
// Two-sided tabular CUSUM. On crossing the decision interval the accumulator
// that fired is reset to zero (standard practice): otherwise it stays latched
// above threshold and re-signals on every subsequent generation, inflating
// alert_count and masking the next genuine excursion.
__host__ __device__ inline void cusum_update(CusumState* s, float x) {
    float dev = x - s->reference;
    s->upper = fmaxf(0.f, s->upper + dev - s->allowance);
    s->lower = fmaxf(0.f, s->lower - dev - s->allowance);
    if (s->upper > s->threshold) {
        s->alert_count++;
        s->upper = 0.f;
    }
    if (s->lower > s->threshold) {
        s->alert_count++;
        s->lower = 0.f;
    }
}

// Checkpoint file format:
//   uint32_t magic            // 'S21C'
//   uint32_t version
//   uint32_t schema_hash      // catches struct drift
//   CheckpointHeader hdr
//   ...subsystem blobs (population, archive, placeholder, sentinels)
//
// write_checkpoint and load_checkpoint are host-side and operate on a single
// open() call. Atomic-replace via temp-file + rename.
constexpr uint32_t CHECKPOINT_MAGIC   = 0x53323143u;  // 'S' '2' '1' 'C'
constexpr uint32_t CHECKPOINT_VERSION = 1;

// Schema hash combines the sizes of the structures that flow through the
// checkpoint. Bump CHECKPOINT_VERSION if any of these change so old files
// fail fast in load_checkpoint.
__host__ inline uint32_t checkpoint_schema_hash() {
    uint32_t h = 0x9E3779B9u;
    auto mix = [&h](uint32_t x) {
        h ^= x + 0x9E3779B9u + (h << 6) + (h >> 2);
    };
    mix(static_cast<uint32_t>(sizeof(CheckpointHeader)));
    mix(static_cast<uint32_t>(GENOME_BITS));
    mix(static_cast<uint32_t>(MAX_ARCHIVE));
    mix(static_cast<uint32_t>(POOL_SIZE));
    mix(static_cast<uint32_t>(BMAP_DIM));
    mix(static_cast<uint32_t>(BTRAJ_SAMPLES));
    mix(static_cast<uint32_t>(CA_CHANNELS));
    mix(static_cast<uint32_t>(GRID_SIZE));
    return h;
}

// Header-only write/load. The full payload write/load is a thin wrapper that
// calls these and then dumps the remaining blobs via fwrite. Kept separate
// so the header can be sanity-checked without paying for the population
// deserialisation.
__host__ inline bool write_checkpoint_header(const CheckpointHeader& hdr,
                                             const char* path) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    uint32_t magic   = CHECKPOINT_MAGIC;
    uint32_t version = CHECKPOINT_VERSION;
    uint32_t schema  = checkpoint_schema_hash();
    bool ok = true;
    ok = ok && std::fwrite(&magic,   sizeof(magic),   1, f) == 1;
    ok = ok && std::fwrite(&version, sizeof(version), 1, f) == 1;
    ok = ok && std::fwrite(&schema,  sizeof(schema),  1, f) == 1;
    ok = ok && std::fwrite(&hdr,     sizeof(hdr),     1, f) == 1;
    std::fclose(f);
    return ok;
}

__host__ inline bool load_checkpoint_header(CheckpointHeader* hdr_out,
                                            const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    uint32_t magic, version, schema;
    bool ok = true;
    ok = ok && std::fread(&magic,   sizeof(magic),   1, f) == 1;
    ok = ok && std::fread(&version, sizeof(version), 1, f) == 1;
    ok = ok && std::fread(&schema,  sizeof(schema),  1, f) == 1;
    ok = ok && std::fread(hdr_out,  sizeof(*hdr_out), 1, f) == 1;
    std::fclose(f);
    if (!ok) return false;
    if (magic   != CHECKPOINT_MAGIC)        return false;
    if (version != CHECKPOINT_VERSION)      return false;
    if (schema  != checkpoint_schema_hash()) return false;
    return true;
}

// DECLARED ONLY — blueprint-in-place.
// write_checkpoint / load_checkpoint move the FULL run state, not just the
// header above. The header path is done; the payload is not, and a header-only
// checkpoint does not survive a restart. What the full payload must serialize,
// in order, after the header:
//   1. Organism table: for each of POOL_SIZE + STRESS_POOL_SIZE slots —
//      genome bits, delta weights (count + indices + values), role, lineage_id,
//      parent_id, spawn_gen, replica_tag. The CA grid is NOT serialized (it is
//      recomputed from the genome on reload); the CAME momentum buffers ARE
//      (m, v, c, prev_u), because PT swaps assume momentum follows the organism.
//   2. Archive: alive entries (descriptor, rff_proj, fitness, lineage, bin,
//      role) + per-role mu_rff vectors + inv_var_ema + bin caps/counts.
//   3. Placeholder regressor: all weights + AdamW moments + replay buffer.
//   4. Correlation window, both CUSUM states, calibrated s_target + its frozen
//      flag, mutation-ladder replica assignments + beta + accept EMA, stress
//      ladder state, sentinel ensemble + history, generation counter, RNG seeds.
// Pointers in CameState (m/v/c/prev_u) must be flattened to inline arrays on
// write and re-pointed on load — a raw fwrite(World) is wrong because of them.
// Write to a temp path then rename() for atomic replacement. load_checkpoint
// must re-decode every genome to rebuild CA grids before the first forward.
void write_checkpoint(const CheckpointHeader& hdr, const char* path);
bool load_checkpoint(CheckpointHeader* hdr_out,  const char* path);

}  // namespace slime::safety

#endif  // COEVO_SAFETY_MONITORING_CU
