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
#include <cstring>
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

// Startup-only CUDA telemetry. This records the device facts that determine
// whether the engineering target is actually available, plus transfer rates
// measured through the same CUDA runtime used by the integration loop.
struct CudaDiagnostics {
    char   device_name[256];
    int    device_ordinal;
    int    compute_major;
    int    compute_minor;
    int    multiprocessor_count;
    int    async_engine_count;
    int    memory_bus_width_bits;
    int    memory_clock_khz;
    int    max_threads_per_block;
    size_t total_global_memory_bytes;
    size_t free_memory_bytes;
    size_t total_memory_bytes;
    float  h2d_gbps;
    float  d2h_gbps;
    float  d2d_gbps;
    bool   concurrent_kernels;
    bool   unified_addressing;
    bool   managed_memory;
    bool   transfer_probe_completed;
};

inline bool cuda_diagnostics_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::printf("CUDA diagnostic failure (%s): %s\n", operation,
                cudaGetErrorString(status));
    return false;
}

inline bool collect_cuda_diagnostics(CudaDiagnostics* out) {
    std::memset(out, 0, sizeof(*out));

    int device_count = 0;
    if (!cuda_diagnostics_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) return false;
    if (device_count < 1) {
        std::printf("CUDA diagnostic failure: no CUDA-capable device found\n");
        return false;
    }
    if (!cuda_diagnostics_ok(cudaGetDevice(&out->device_ordinal), "cudaGetDevice")) return false;

    cudaDeviceProp prop{};
    if (!cuda_diagnostics_ok(cudaGetDeviceProperties(&prop, out->device_ordinal),
                             "cudaGetDeviceProperties")) return false;
    if (!cuda_diagnostics_ok(cudaMemGetInfo(&out->free_memory_bytes,
                                            &out->total_memory_bytes),
                             "cudaMemGetInfo")) return false;

    std::snprintf(out->device_name, sizeof(out->device_name), "%s", prop.name);
    out->compute_major = prop.major;
    out->compute_minor = prop.minor;
    out->multiprocessor_count = prop.multiProcessorCount;
    out->async_engine_count = prop.asyncEngineCount;
    out->memory_bus_width_bits = prop.memoryBusWidth;
    if (!cuda_diagnostics_ok(cudaDeviceGetAttribute(&out->memory_clock_khz,
                                                    cudaDevAttrMemoryClockRate,
                                                    out->device_ordinal),
                             "cudaDevAttrMemoryClockRate")) return false;
    out->max_threads_per_block = prop.maxThreadsPerBlock;
    out->total_global_memory_bytes = prop.totalGlobalMem;
    out->concurrent_kernels = prop.concurrentKernels != 0;
    out->unified_addressing = prop.unifiedAddressing != 0;
    out->managed_memory = prop.managedMemory != 0;
    return true;
}

inline bool benchmark_cuda_transfers(CudaDiagnostics* out, cudaStream_t stream) {
    constexpr size_t MAX_PROBE_BYTES = 64u * 1024u * 1024u;
    constexpr size_t MIN_PROBE_BYTES = 8u * 1024u * 1024u;
    constexpr int TRANSFER_ITERATIONS = 16;

    size_t probe_bytes = MAX_PROBE_BYTES;
    size_t memory_budget = out->free_memory_bytes / 16u;
    if (probe_bytes > memory_budget) probe_bytes = memory_budget;
    if (probe_bytes < MIN_PROBE_BYTES) {
        std::printf("CUDA diagnostic warning: only %zu MiB free; transfer probe skipped\n",
                    out->free_memory_bytes / (1024u * 1024u));
        return true;
    }

    void* h_buffer = nullptr;
    void* d_source = nullptr;
    void* d_destination = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    bool ok = cuda_diagnostics_ok(cudaMallocHost(&h_buffer, probe_bytes), "cudaMallocHost")
           && cuda_diagnostics_ok(cudaMalloc(&d_source, probe_bytes), "cudaMalloc source")
           && cuda_diagnostics_ok(cudaMalloc(&d_destination, probe_bytes), "cudaMalloc destination")
           && cuda_diagnostics_ok(cudaEventCreate(&start), "cudaEventCreate start")
           && cuda_diagnostics_ok(cudaEventCreate(&stop), "cudaEventCreate stop");
    if (!ok) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (d_source) cudaFree(d_source);
        if (d_destination) cudaFree(d_destination);
        if (h_buffer) cudaFreeHost(h_buffer);
        return false;
    }
    std::memset(h_buffer, 0, probe_bytes);

    auto measure = [&](cudaMemcpyKind kind, void* destination, const void* source,
                       float* gbps, const char* name) -> bool {
        if (!cuda_diagnostics_ok(cudaEventRecord(start, stream), name)) return false;
        for (int i = 0; i < TRANSFER_ITERATIONS; ++i) {
            if (!cuda_diagnostics_ok(cudaMemcpyAsync(destination, source, probe_bytes,
                                                      kind, stream), name)) return false;
        }
        if (!cuda_diagnostics_ok(cudaEventRecord(stop, stream), name)) return false;
        if (!cuda_diagnostics_ok(cudaEventSynchronize(stop), name)) return false;
        float milliseconds = 0.f;
        if (!cuda_diagnostics_ok(cudaEventElapsedTime(&milliseconds, start, stop), name)) return false;
        if (milliseconds <= 0.f) return false;
        double transferred = static_cast<double>(probe_bytes) * TRANSFER_ITERATIONS;
        *gbps = static_cast<float>(transferred / (static_cast<double>(milliseconds) * 1.0e6));
        return true;
    };

    ok = measure(cudaMemcpyHostToDevice, d_destination, h_buffer, &out->h2d_gbps, "H2D probe")
      && measure(cudaMemcpyDeviceToHost, h_buffer, d_destination, &out->d2h_gbps, "D2H probe")
      && measure(cudaMemcpyDeviceToDevice, d_source, d_destination, &out->d2d_gbps, "D2D probe");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_source);
    cudaFree(d_destination);
    cudaFreeHost(h_buffer);
    if (!ok) return false;
    out->transfer_probe_completed = true;
    return true;
}

inline bool emit_cuda_diagnostics(cudaStream_t stream) {
    CudaDiagnostics diagnostics{};
    if (!collect_cuda_diagnostics(&diagnostics)) return false;
    if (!benchmark_cuda_transfers(&diagnostics, stream)) return false;

    constexpr double BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0;
    std::printf("CUDA device: %s (device %d, sm_%d%d, %d SMs, %.2f GiB global, %.2f GiB free)\n",
                diagnostics.device_name, diagnostics.device_ordinal,
                diagnostics.compute_major, diagnostics.compute_minor,
                diagnostics.multiprocessor_count,
                diagnostics.total_global_memory_bytes / BYTES_PER_GIB,
                diagnostics.free_memory_bytes / BYTES_PER_GIB);
    std::printf("CUDA capabilities: %d-bit bus, %d MHz memory, %d async engines, max block %d, concurrent kernels=%s, UVA=%s, managed memory=%s\n",
                diagnostics.memory_bus_width_bits, diagnostics.memory_clock_khz / 1000,
                diagnostics.async_engine_count, diagnostics.max_threads_per_block,
                diagnostics.concurrent_kernels ? "yes" : "no",
                diagnostics.unified_addressing ? "yes" : "no",
                diagnostics.managed_memory ? "yes" : "no");
    if (diagnostics.transfer_probe_completed) {
        std::printf("CUDA transfer probe (64 MiB x 16): H2D %.2f GB/s, D2H %.2f GB/s, D2D %.2f GB/s\n",
                    diagnostics.h2d_gbps, diagnostics.d2h_gbps, diagnostics.d2d_gbps);
    }
    return true;
}

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
