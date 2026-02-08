#ifndef PROVENANCE_CUH
#define PROVENANCE_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

constexpr int32_t PROVENANCE_UNINITIALIZED_INT = -2147483647;
constexpr float PROVENANCE_UNINITIALIZED_FLOAT = -3.402823466e+38f;
constexpr uint64_t PROVENANCE_UNINITIALIZED_HASH = 0xDEADBEEFDEADBEEFULL;
constexpr int32_t PROVENANCE_INVALID_INDEX = -999999999;

constexpr uint32_t PROVENANCE_SOURCE_NONE = 0;
constexpr uint32_t PROVENANCE_SOURCE_INIT = 1;
constexpr uint32_t PROVENANCE_SOURCE_POOL = 2;
constexpr uint32_t PROVENANCE_SOURCE_ARCHIVE = 3;
constexpr uint32_t PROVENANCE_SOURCE_TRAINING = 4;
constexpr uint32_t PROVENANCE_SOURCE_TELEMETRY = 5;
constexpr uint32_t PROVENANCE_SOURCE_CLASSIFICATION = 6;
constexpr uint32_t PROVENANCE_SOURCE_FLOW = 7;
constexpr uint32_t PROVENANCE_SOURCE_CHEMOTAXIS = 8;
constexpr uint32_t PROVENANCE_SOURCE_LIFECYCLE = 9;
constexpr uint32_t PROVENANCE_SOURCE_DIRESA = 10;
constexpr uint32_t PROVENANCE_SOURCE_BACKWARD = 11;
constexpr uint32_t PROVENANCE_SOURCE_HOST = 12;

constexpr uint32_t RING_BUFFER_SLOTS = 8;
constexpr uint32_t CRC32_POLYNOMIAL = 0xEDB88320;

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

struct RecordHeader {
    uint64_t sequence_number;
    uint32_t source_id;
    uint32_t block_id;
    uint32_t thread_id;
    uint32_t record_size;
    uint64_t timestamp;
    uint32_t checksum;
    uint32_t checksum_valid;
};

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
    return val == PROVENANCE_UNINITIALIZED_FLOAT || isnan(val) || val < -1.0e+37f;
}

__host__ __forceinline__ bool host_is_uninitialized_int(int32_t val) {
    return val == PROVENANCE_UNINITIALIZED_INT;
}

__host__ __forceinline__ bool host_is_uninitialized_float(float val) {
    return val == PROVENANCE_UNINITIALIZED_FLOAT || std::isnan(val) || val < -1.0e+37f;
}

#define PROVENANCE_FATAL(msg) do { \
    printf("E_PROV %s:%d b%d t%d %s\n", __FILE__, __LINE__, blockIdx.x, threadIdx.x, msg); \
    asm("trap;"); \
} while(0)

#define PROVENANCE_FATAL_IF(cond, msg) do { \
    if (cond) { PROVENANCE_FATAL(msg); } \
} while(0)

#define PROVENANCE_ASSERT_INITIALIZED_INT(val, name) do { \
    if (is_uninitialized_int(val)) { \
        printf("E_UNINIT %s:%d b%d t%d %s\n", __FILE__, __LINE__, blockIdx.x, threadIdx.x, name); \
        asm("trap;"); \
    } \
} while(0)

#define PROVENANCE_ASSERT_INITIALIZED_FLOAT(val, name) do { \
    if (is_uninitialized_float(val)) { \
        printf("E_UNINIT %s:%d b%d t%d %s\n", __FILE__, __LINE__, blockIdx.x, threadIdx.x, name); \
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

constexpr uint32_t AUDIT_FIELD_GENERATION = (1 << 0);
constexpr uint32_t AUDIT_FIELD_BATCH = (1 << 1);
constexpr uint32_t AUDIT_FIELD_ACCURACY = (1 << 2);
constexpr uint32_t AUDIT_FIELD_POOL = (1 << 3);
constexpr uint32_t AUDIT_FIELD_ARCHIVE = (1 << 4);
constexpr uint32_t AUDIT_FIELD_CHEMICAL = (1 << 5);
constexpr uint32_t AUDIT_FIELD_FLOW = (1 << 6);

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

#endif
