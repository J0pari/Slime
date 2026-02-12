#ifndef PROVENANCE_CUH
#define PROVENANCE_CUH

#include "../config/config.cu"
#include "../core/organism.cu"
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <atomic>
#include <cstdio>

namespace StalenessThreshold {
    constexpr int FITNESS = 1;
    constexpr int LIFECYCLE_SIGNALS = 1;
    constexpr int BEHAVIORAL_COORDS = 1;
    constexpr int NICHE_ASSIGNMENT = 100;
    constexpr int SIGNAL_FLOW = 100;
    constexpr int ENCODER_WEIGHTS_EPOCH = 1;
}

// MeasuredValue helper functions (struct is in organism.cu)
template<typename T>
__device__ __host__ inline void measured_value_init(MeasuredValue<T>* mv) {
    mv->value = static_cast<T>(0);
    mv->state = ComputeState::UNCOMPUTED;
    mv->computed_at_generation = INT_MIN;
    mv->input_hash = UINT64_MAX;
}

template<>
__device__ __host__ inline void measured_value_init<float>(MeasuredValue<float>* mv) {
    mv->value = NAN;
    mv->state = ComputeState::UNCOMPUTED;
    mv->computed_at_generation = INT_MIN;
    mv->input_hash = UINT64_MAX;
}

template<typename T>
__device__ inline bool measured_value_is_valid(const MeasuredValue<T>* mv) {
    return mv->state == ComputeState::COMPUTED;
}

template<>
__device__ inline bool measured_value_is_valid<float>(const MeasuredValue<float>* mv) {
    return mv->state == ComputeState::COMPUTED && !isnan(mv->value);
}

template<typename T>
__device__ inline bool measured_value_is_stale(const MeasuredValue<T>* mv, int current_generation, int threshold) {
    if (mv->computed_at_generation == INT_MIN) return true;
    return (current_generation - mv->computed_at_generation) > threshold;
}

template<typename T>
__device__ inline void measured_value_set_computed(MeasuredValue<T>* mv, T v, int generation, uint64_t hash) {
    mv->value = v;
    mv->state = ComputeState::COMPUTED;
    mv->computed_at_generation = generation;
    mv->input_hash = hash;
}

template<typename T>
__device__ inline void measured_value_set_uncomputed(MeasuredValue<T>* mv) {
    mv->state = ComputeState::UNCOMPUTED;
    mv->computed_at_generation = INT_MIN;
    mv->input_hash = UINT64_MAX;
}

template<>
__device__ inline void measured_value_set_uncomputed<float>(MeasuredValue<float>* mv) {
    mv->value = NAN;
    mv->state = ComputeState::UNCOMPUTED;
    mv->computed_at_generation = INT_MIN;
    mv->input_hash = UINT64_MAX;
}

template<typename T>
__device__ inline void measured_value_mark_stale(MeasuredValue<T>* mv) { mv->state = ComputeState::STALE; }

template<typename T>
__device__ inline void measured_value_mark_invalid(MeasuredValue<T>* mv) { mv->state = ComputeState::INVALIDATED; }

template<typename T>
__device__ inline void measured_value_mark_computing(MeasuredValue<T>* mv) { mv->state = ComputeState::COMPUTING; }

template<typename T>
__device__ inline void measured_value_mark_failed(MeasuredValue<T>* mv) { mv->state = ComputeState::COMPUTATION_FAILED; }

// PhaseTransitionRecord helper
__device__ __host__ inline void phase_transition_record_init(PhaseTransitionRecord* ptr) {
    ptr->previous_phase = LifecyclePhase::DEAD;
    ptr->current_phase = LifecyclePhase::DEAD;
    ptr->transition_generation = INT_MIN;
    ptr->transition_count = 0;
}

__device__ __forceinline__ bool is_valid_phase_transition(LifecyclePhase from, LifecyclePhase to) {
    switch (from) {
        case LifecyclePhase::DEAD:
            return to == LifecyclePhase::ACTIVE;
        case LifecyclePhase::ACTIVE:
            return to == LifecyclePhase::DORMANT ||
                   to == LifecyclePhase::ARCHIVED ||
                   to == LifecyclePhase::DEAD;
        case LifecyclePhase::DORMANT:
            return to == LifecyclePhase::ACTIVE ||
                   to == LifecyclePhase::DEAD;
        case LifecyclePhase::ARCHIVED:
            return to == LifecyclePhase::ACTIVE;
    }
    return false;
}

__device__ __forceinline__ const char* phase_to_string(LifecyclePhase phase) {
    switch (phase) {
        case LifecyclePhase::DEAD: return "DEAD";
        case LifecyclePhase::ACTIVE: return "ACTIVE";
        case LifecyclePhase::DORMANT: return "DORMANT";
        case LifecyclePhase::ARCHIVED: return "ARCHIVED";
    }
    return "UNKNOWN";
}

__device__ __forceinline__ const char* compute_state_to_string(ComputeState state) {
    switch (state) {
        case ComputeState::UNCOMPUTED: return "UNCOMPUTED";
        case ComputeState::COMPUTING: return "COMPUTING";
        case ComputeState::COMPUTED: return "COMPUTED";
        case ComputeState::COMPUTATION_FAILED: return "FAILED";
        case ComputeState::STALE: return "STALE";
        case ComputeState::INVALIDATED: return "INVALIDATED";
    }
    return "UNKNOWN";
}

__device__ __host__ __forceinline__ uint32_t crc32_byte(uint32_t crc, uint8_t byte) {
    crc ^= byte;
    for (int i = 0; i < 8; i++) {
        crc = (crc >> 1) ^ ((crc & 1) ? CRC32_POLYNOMIAL : 0);
    }
    return crc;
}

__device__ __host__ __forceinline__ uint32_t crc32_compute(const void* data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < len; i++) {
        crc = crc32_byte(crc, bytes[i]);
    }
    return crc ^ 0xFFFFFFFF;
}

__device__ __forceinline__ void record_header_init(RecordHeader* hdr, uint64_t seq, uint32_t src, uint32_t size) {
    hdr->sequence_number = seq;
    hdr->source_id = src;
    hdr->block_id = blockIdx.x;
    hdr->thread_id = threadIdx.x;
    hdr->record_size = size;
    hdr->timestamp = clock64();
    hdr->checksum = 0;
    hdr->checksum_valid = 0;
}

__device__ __forceinline__ void record_header_finalize(RecordHeader* hdr, const void* payload, size_t payload_size) {
    hdr->checksum = crc32_compute(payload, payload_size);
    hdr->checksum_valid = 1;
    __threadfence_system();
}

__host__ __forceinline__ bool record_header_verify(const RecordHeader* hdr, const void* payload, size_t payload_size) {
    if (hdr->checksum_valid != 1) return false;
    uint32_t computed = crc32_compute(payload, payload_size);
    return computed == hdr->checksum;
}

template<typename T>
struct RingBufferSlot {
    RecordHeader header;
    T payload;
    volatile uint32_t committed;
};

template<typename T, int N>
struct RingBuffer {
    volatile uint64_t write_sequence;
    volatile uint64_t read_sequence;
    volatile uint64_t dropped_count;
    volatile uint64_t corrupted_count;
    RingBufferSlot<T> slots[N];

    __device__ void device_init() {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            write_sequence = 0;
            read_sequence = 0;
            dropped_count = 0;
            corrupted_count = 0;
            for (int i = 0; i < N; i++) {
                slots[i].committed = 0;
                slots[i].header.sequence_number = 0;
                slots[i].header.checksum_valid = 0;
            }
        }
        __threadfence_system();
    }

    __device__ T* acquire_write_slot(uint32_t source_id) {
        uint64_t seq = atomicAdd((unsigned long long*)&write_sequence, 1ULL);
        int slot_idx = seq % N;

        if (slots[slot_idx].committed && (seq - slots[slot_idx].header.sequence_number) < N) {
            atomicAdd((unsigned long long*)&dropped_count, 1ULL);
        }

        slots[slot_idx].committed = 0;
        __threadfence_system();

        record_header_init(&slots[slot_idx].header, seq, source_id, sizeof(T));
        return &slots[slot_idx].payload;
    }

    __device__ void commit_write_slot(int slot_idx) {
        record_header_finalize(&slots[slot_idx].header, &slots[slot_idx].payload, sizeof(T));
        slots[slot_idx].committed = 1;
        __threadfence_system();
    }

    __device__ void commit_write(T* payload_ptr) {
        int slot_idx = ((char*)payload_ptr - (char*)&slots[0].payload) / sizeof(RingBufferSlot<T>);
        commit_write_slot(slot_idx);
    }
};

template<typename T, int N>
struct HostRingBufferReader {
    RingBuffer<T, N>* buffer;
    uint64_t last_read_sequence;

    void init(RingBuffer<T, N>* buf) {
        buffer = buf;
        last_read_sequence = 0;
    }

    bool has_new_data() {
        std::atomic_thread_fence(std::memory_order_acquire);
        return buffer->write_sequence > last_read_sequence;
    }

    uint64_t get_dropped_count() {
        return buffer->dropped_count;
    }

    uint64_t get_corrupted_count() {
        return buffer->corrupted_count;
    }

    bool read_next(T* out, RecordHeader* hdr_out) {
        std::atomic_thread_fence(std::memory_order_acquire);

        if (buffer->write_sequence <= last_read_sequence) {
            return false;
        }

        int slot_idx = last_read_sequence % N;
        RingBufferSlot<T>* slot = &buffer->slots[slot_idx];

        if (!slot->committed) {
            return false;
        }

        if (slot->header.sequence_number != last_read_sequence) {
            uint64_t missed = slot->header.sequence_number - last_read_sequence;
            fprintf(stderr, "E_SEQ seq=%llu exp=%llu gap=%llu\n",
                    (unsigned long long)slot->header.sequence_number,
                    (unsigned long long)last_read_sequence,
                    (unsigned long long)missed);
            last_read_sequence = slot->header.sequence_number;
        }

        if (!record_header_verify(&slot->header, &slot->payload, sizeof(T))) {
            fprintf(stderr, "E_CRC seq=%llu\n", (unsigned long long)slot->header.sequence_number);
            ((RingBuffer<T, N>*)buffer)->corrupted_count++;
            last_read_sequence++;
            return false;
        }

        *out = slot->payload;
        if (hdr_out) *hdr_out = slot->header;
        last_read_sequence++;
        return true;
    }

    void read_all(void (*callback)(const T*, const RecordHeader*, void*), void* user_data) {
        T record;
        RecordHeader hdr;
        while (read_next(&record, &hdr)) {
            callback(&record, &hdr, user_data);
        }
    }
};

__device__ __forceinline__ bool is_uninitialized_int(int32_t val) {
    return val == PROVENANCE_UNINITIALIZED_INT;
}

__device__ __forceinline__ bool is_uninitialized_float(float val) {
    return isnan(val);
}

__host__ __forceinline__ bool host_is_uninitialized_int(int32_t val) {
    return val == PROVENANCE_UNINITIALIZED_INT;
}

__host__ __forceinline__ bool host_is_uninitialized_float(float val) {
    return std::isnan(val);
}

#define PROVENANCE_FATAL(msg) do { \
    printf("E_PROV %s:%d b%d t%d %s\n", __FILE__, __LINE__, blockIdx.x, threadIdx.x, msg); \
    __threadfence_system(); \
    asm("trap;"); \
} while(0)

#define PROVENANCE_FATAL_IF(cond, msg) do { \
    if (cond) { PROVENANCE_FATAL(msg); } \
} while(0)

#define PROVENANCE_ASSERT_INITIALIZED_INT(val, name) do { \
    if (is_uninitialized_int(val)) { \
        printf("E_UNINIT %s:%d b%d t%d %s\n", __FILE__, __LINE__, blockIdx.x, threadIdx.x, name); \
        __threadfence_system(); \
        asm("trap;"); \
    } \
} while(0)

#define PROVENANCE_ASSERT_INITIALIZED_FLOAT(val, name) do { \
    if (is_uninitialized_float(val)) { \
        printf("E_UNINIT %s:%d b%d t%d %s\n", __FILE__, __LINE__, blockIdx.x, threadIdx.x, name); \
        __threadfence_system(); \
        asm("trap;"); \
    } \
} while(0)

#define HOST_PROVENANCE_FATAL(msg) do { \
    fprintf(stderr, "E_HOST %s:%d %s\n", __FILE__, __LINE__, msg); \
    abort(); \
} while(0)

#define HOST_PROVENANCE_FATAL_IF(cond, msg) do { \
    if (cond) { HOST_PROVENANCE_FATAL(msg); } \
} while(0)

#define HOST_ASSERT_INITIALIZED_INT(val, name) do { \
    if (host_is_uninitialized_int(val)) { \
        fprintf(stderr, "E_UNINIT %s:%d %s\n", __FILE__, __LINE__, name); \
        abort(); \
    } \
} while(0)

#define HOST_ASSERT_INITIALIZED_FLOAT(val, name) do { \
    if (host_is_uninitialized_float(val)) { \
        fprintf(stderr, "E_UNINIT %s:%d %s\n", __FILE__, __LINE__, name); \
        abort(); \
    } \
} while(0)

#define HOST_REFUSE_ZERO_WITHOUT_PROVENANCE(val, source, name) do { \
    if ((val) == 0 && (source) == PROVENANCE_SOURCE_NONE) { \
        fprintf(stderr, "E_ZERO_NOPROV %s:%d %s\n", __FILE__, __LINE__, name); \
        abort(); \
    } \
} while(0)





enum class AuditEventType : uint16_t {
    SPAWN,
    CULL,
    PHASE_TRANSITION,
    ARCHIVE_INSERT,
    ARCHIVE_EVICT,
    FIELD_EPOCH_BOUNDARY,
    WEIGHT_UPDATE,
    GRADIENT_STEP,
    INTEGRITY_VIOLATION,
    STALENESS_DETECTED,
    SENTINEL_ACCESS_ATTEMPTED
};

struct SpawnData {
    int parent_idx;
    float mutation_rate;
    uint64_t parent_hash;
    uint64_t child_hash;
};

struct PhaseData {
    LifecyclePhase from;
    LifecyclePhase to;
    float trigger_value;
    int entry_idx;
};

struct ArchiveData {
    int niche_id;
    float distance_to_centroid;
    float fitness_at_insertion;
    int evicted_idx;  
};

struct EpochData {
    float signal_flow_sum;
    float reinforcement_applied;
    int epoch_number;
    int generation;
};

struct ViolationData {
    char message[120];  
    int severity;       
};


struct AuditEntry {
    uint64_t timestamp_ns;          
    uint64_t prev_entry_hash;       
    int generation;                 
    int entry_idx;                  
    AuditEventType event_type;
    uint16_t reserved;              

    union {
        SpawnData spawn_data;
        PhaseData phase_data;
        ArchiveData archive_data;
        EpochData epoch_data;
        ViolationData violation_data;
    };

    uint32_t entry_crc;             
};

__device__ __forceinline__ void audit_entry_init(AuditEntry* entry, AuditEventType type, int gen, int idx) {
    entry->timestamp_ns = clock64();
    entry->prev_entry_hash = PROVENANCE_UNINITIALIZED_HASH;  
    entry->generation = gen;
    entry->entry_idx = idx;
    entry->event_type = type;
    entry->reserved = 0;
    entry->entry_crc = 0;  
}

__device__ __forceinline__ void audit_entry_finalize(AuditEntry* entry, uint64_t prev_hash) {
    entry->prev_entry_hash = prev_hash;
    entry->entry_crc = crc32_compute(entry, sizeof(AuditEntry) - sizeof(uint32_t));
}


struct AuditRecord {
    RecordHeader header;

    int32_t generation;
    int32_t batch_size;
    int32_t num_classes;
    int32_t grid_size;
    int32_t correct_count;
    float loss;
    float accuracy;

    float train_accuracy;
    float test_accuracy;

    int32_t pool_alive_count;
    int32_t pool_capacity;
    int32_t pool_total_spawned;
    int32_t pool_total_culled;

    int32_t archive_occupied_cells;
    float elite_fitness_best;
    float elite_fitness_mean;

    float chemical_concentration_mean;
    float flow_lenia_mass_total;

    uint32_t fields_written_mask;
};

__device__ __forceinline__ void audit_record_init(AuditRecord* rec) {
    rec->generation = PROVENANCE_UNINITIALIZED_INT;
    rec->batch_size = PROVENANCE_UNINITIALIZED_INT;
    rec->num_classes = PROVENANCE_UNINITIALIZED_INT;
    rec->grid_size = PROVENANCE_UNINITIALIZED_INT;
    rec->correct_count = PROVENANCE_UNINITIALIZED_INT;
    rec->loss = PROVENANCE_UNINITIALIZED_FLOAT;
    rec->accuracy = PROVENANCE_UNINITIALIZED_FLOAT;
    rec->train_accuracy = PROVENANCE_UNINITIALIZED_FLOAT;
    rec->test_accuracy = PROVENANCE_UNINITIALIZED_FLOAT;
    rec->pool_alive_count = PROVENANCE_UNINITIALIZED_INT;
    rec->pool_capacity = PROVENANCE_UNINITIALIZED_INT;
    rec->pool_total_spawned = PROVENANCE_UNINITIALIZED_INT;
    rec->pool_total_culled = PROVENANCE_UNINITIALIZED_INT;
    rec->archive_occupied_cells = PROVENANCE_UNINITIALIZED_INT;
    rec->elite_fitness_best = PROVENANCE_UNINITIALIZED_FLOAT;
    rec->elite_fitness_mean = PROVENANCE_UNINITIALIZED_FLOAT;
    rec->chemical_concentration_mean = PROVENANCE_UNINITIALIZED_FLOAT;
    rec->flow_lenia_mass_total = PROVENANCE_UNINITIALIZED_FLOAT;
    rec->fields_written_mask = 0;
    rec->header.sequence_number = 0;
    rec->header.source_id = PROVENANCE_SOURCE_NONE;
    rec->header.checksum_valid = 0;
}

__host__ __forceinline__ bool audit_record_field_valid(const AuditRecord* rec, uint32_t field_mask) {
    return (rec->fields_written_mask & field_mask) == field_mask;
}

__host__ __forceinline__ void audit_record_refuse_invalid(const AuditRecord* rec, uint32_t field_mask, const char* field_name) {
    if (!audit_record_field_valid(rec, field_mask)) {
        fprintf(stderr, "E_FIELD %s m=0x%x\n", field_name, field_mask);
        abort();
    }
}

typedef RingBuffer<AuditRecord, RING_BUFFER_SLOTS> AuditRingBuffer;
typedef HostRingBufferReader<AuditRecord, RING_BUFFER_SLOTS> HostAuditReader;

constexpr uint32_t AUDIT_RING_SLOTS = 4;

typedef RingBuffer<StateExportEntry, AUDIT_RING_SLOTS> StateExportBuffer;
typedef HostRingBufferReader<StateExportEntry, AUDIT_RING_SLOTS> StateExportBufferReader;

typedef StateExportBuffer AuditBuffer;
typedef StateExportEntry TelemetryAuditEntry;

// StateExportEntry helper functions (moved from config.cu since struct is in organism.cu)
__host__ __forceinline__ void state_export_init_sentinel(StateExportEntry* buf) {
    buf->provenance_source = PROVENANCE_SOURCE_NONE;
    buf->fields_written_mask = 0;
    buf->sequence_number = 0;

    buf->generation = PROVENANCE_UNINITIALIZED_INT;
    buf->batch_size = PROVENANCE_UNINITIALIZED_INT;
    buf->num_classes = PROVENANCE_UNINITIALIZED_INT;
    buf->grid_size = PROVENANCE_UNINITIALIZED_INT;
    buf->correct_count = PROVENANCE_UNINITIALIZED_INT;
    buf->loss = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->accuracy = PROVENANCE_UNINITIALIZED_FLOAT;

    buf->train_accuracy = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->test_accuracy = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->generalization_gap = PROVENANCE_UNINITIALIZED_FLOAT;

    buf->pool_alive_count = PROVENANCE_UNINITIALIZED_INT;
    buf->pool_capacity = PROVENANCE_UNINITIALIZED_INT;

    buf->archive_occupied_cells = PROVENANCE_UNINITIALIZED_INT;
    buf->frontier_cells_gained = PROVENANCE_UNINITIALIZED_INT;
    buf->frontier_cells_lost = PROVENANCE_UNINITIALIZED_INT;
    buf->sparse_cell_count = PROVENANCE_UNINITIALIZED_INT;
    buf->niche_entropy = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->novelty_gradient = PROVENANCE_UNINITIALIZED_FLOAT;

    buf->elite_fitness_best = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->elite_fitness_mean = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->elite_fitness_delta = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->quality_floor = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->quality_mean = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->quality_range = PROVENANCE_UNINITIALIZED_FLOAT;

    buf->density_mean = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->density_max = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->density_variance = PROVENANCE_UNINITIALIZED_FLOAT;

    buf->hw_axis_min = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->hw_axis_max = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->hw_axis_mean = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->task_axis_min = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->task_axis_max = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->task_axis_mean = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->gen_axis_min = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->gen_axis_max = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->gen_axis_mean = PROVENANCE_UNINITIALIZED_FLOAT;

    buf->total_population = PROVENANCE_UNINITIALIZED_INT;
    buf->births_this_gen = PROVENANCE_UNINITIALIZED_INT;
    buf->deaths_this_gen = PROVENANCE_UNINITIALIZED_INT;

    buf->diresa_recon_loss_hw = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->diresa_recon_loss_task = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->diresa_recon_loss_gen = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->diresa_recon_loss_total = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->diresa_behavioral_drift = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->diresa_latent_utilization = PROVENANCE_UNINITIALIZED_FLOAT;

    buf->genome_unique_hashes = PROVENANCE_UNINITIALIZED_INT;
    buf->genome_hash_entropy = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->genome_avg_deltas = PROVENANCE_UNINITIALIZED_FLOAT;

    buf->axis_corr_hw_task = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->axis_corr_hw_gen = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->axis_corr_task_gen = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->hash_clustering_coefficient = PROVENANCE_UNINITIALIZED_FLOAT;

    buf->chemical_concentration_mean = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->chemical_concentration_max = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->chemical_gradient_magnitude_mean = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->chemical_source_activity = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->chemical_decay_rate_mean = PROVENANCE_UNINITIALIZED_FLOAT;

    buf->flow_lenia_mass_total = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->flow_lenia_mass_conservation_error = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->flow_lenia_affinity_mean = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->flow_lenia_flow_magnitude_mean = PROVENANCE_UNINITIALIZED_FLOAT;

    buf->fitness_alpha = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->fitness_beta = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->fitness_gamma = PROVENANCE_UNINITIALIZED_FLOAT;
    buf->fitness_delta = PROVENANCE_UNINITIALIZED_FLOAT;

    buf->memory_gpu_allocated = 0;
    buf->memory_gpu_free = 0;
    buf->memory_ca_state_size = 0;
    buf->memory_chemical_field_size = 0;
    buf->memory_archive_size = 0;

    buf->state_agent_count = PROVENANCE_UNINITIALIZED_INT;
    buf->state_voronoi_count = PROVENANCE_UNINITIALIZED_INT;
    buf->state_archive_count = PROVENANCE_UNINITIALIZED_INT;

    buf->pool_total_spawned = PROVENANCE_UNINITIALIZED_INT;
    buf->pool_total_culled = PROVENANCE_UNINITIALIZED_INT;

    for (int i = 0; i < POOL_CAPACITY_MAX; i++) {
        buf->pool_entry_alive[i] = PROVENANCE_UNINITIALIZED_INT;
        buf->pool_entry_fitness[i] = PROVENANCE_UNINITIALIZED_FLOAT;
        buf->pool_entry_hunger[i] = PROVENANCE_UNINITIALIZED_FLOAT;
        buf->pool_entry_age[i] = PROVENANCE_UNINITIALIZED_INT;
        buf->pool_entry_num_deltas[i] = PROVENANCE_UNINITIALIZED_INT;
        buf->pool_entry_genome_hash[i] = PROVENANCE_UNINITIALIZED_HASH;
    }

    for (int i = 0; i < NUM_CLASSES_MAX; i++) {
        buf->per_class_correct[i] = PROVENANCE_UNINITIALIZED_FLOAT;
        buf->per_class_total[i] = PROVENANCE_UNINITIALIZED_FLOAT;
    }

    for (int i = 0; i < AUDIT_SAMPLE_COUNT; i++) {
        buf->sample_labels[i] = PROVENANCE_UNINITIALIZED_INT;
        buf->sample_predictions[i] = PROVENANCE_UNINITIALIZED_INT;
        buf->sample_confidences[i] = PROVENANCE_UNINITIALIZED_FLOAT;
    }

    for (int i = 0; i < CA_FIELD_SIZE; i++) {
        buf->ca_snapshot[i] = PROVENANCE_UNINITIALIZED_FLOAT;
    }
}

__host__ __forceinline__ bool state_export_field_valid(const StateExportEntry* buf, uint32_t mask) {
    return (buf->fields_written_mask & mask) == mask;
}

__host__ __forceinline__ void state_export_refuse_invalid(const StateExportEntry* buf, uint32_t mask, const char* name) {
    if (!state_export_field_valid(buf, mask)) {
        fprintf(stderr, "E_AUDIT %s m=0x%x have=0x%x\n", name, mask, buf->fields_written_mask);
        abort();
    }
}

__host__ __forceinline__ void state_export_buffer_init_sentinel(StateExportBuffer* ring) {
    ring->write_sequence = 0;
    ring->read_sequence = 0;
    ring->dropped_count = 0;
    ring->corrupted_count = 0;
    for (uint32_t i = 0; i < AUDIT_RING_SLOTS; i++) {
        ring->slots[i].committed = 0;
        ring->slots[i].header.sequence_number = 0;
        ring->slots[i].header.checksum_valid = 0;
        state_export_init_sentinel(&ring->slots[i].payload);
    }
}

#endif
