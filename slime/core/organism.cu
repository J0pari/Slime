
#ifndef ORGANISM_CU
#define ORGANISM_CU

#include "../config/config.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <cuda/atomic>

// Forward declaration for RingBuffer template (defined in provenance.cuh)
// Used by AuditBuffer which is typedef'd to RingBuffer<StateExportEntry, 4>
template<typename T, int N> struct RingBuffer;
// AuditBuffer forward declaration - actual typedef in provenance.cuh
struct StateExportEntry;  // Already defined below, but forward declare for clarity
constexpr uint32_t ORGANISM_AUDIT_RING_SLOTS = 4;
typedef RingBuffer<StateExportEntry, ORGANISM_AUDIT_RING_SLOTS> AuditBuffer;

// ============================================================================
// Struct definitions - all types that flow through Organism
// ============================================================================

// From provenance.cuh - enums
enum class ComputeState : uint8_t {
    UNCOMPUTED,
    COMPUTING,
    COMPUTED,
    COMPUTATION_FAILED,
    STALE,
    INVALIDATED
};

enum class LifecyclePhase : uint8_t {
    DEAD = 0,
    ACTIVE = 1,
    STRESSED = 2,
    DORMANT = 3,
    REACTIVATING = 4,
    ARCHIVED = 5
};

enum class NicheState : uint8_t {
    UNASSIGNED,
    COMPUTING,
    ASSIGNED,
    STALE,
    EVICTED
};

enum class FieldEpochPhase : uint8_t {
    ACCUMULATING,
    REINFORCING,
    RESETTING,
    READY
};

enum class BufferEntryState : uint8_t {
    EMPTY,
    WRITING,
    VALID,
    CONSUMED,
    CORRUPT
};

// From provenance.cuh - MeasuredValue (data only, methods in provenance.cuh)
template<typename T>
struct MeasuredValue {
    T value;
    ComputeState state;
    int computed_at_generation;
    uint64_t input_hash;
};

// From provenance.cuh
struct PhaseTransitionRecord {
    LifecyclePhase previous_phase;
    LifecyclePhase current_phase;
    int transition_generation;
    int transition_count;
};

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

// From tubes.cu
struct MemoryEntry {
    float* data;
    int size;
    float timestamp;
    float decay_factor;
    float importance;
};

struct TemporalTube {
    MemoryEntry* entries;
    int capacity;
    int head;
    int count;
    float global_time;
    float decay_rate;
};

// From chemotaxis.cu
struct BehavioralInitSlots {
    int agent_embedding_scale;
    int init_exploration;
    int init_sensitivity;
    int levy_alpha;
    int ctx_metabolic;
    int ctx_stress;
    int ctx_morphogen;
};

struct ChemicalField {
    static constexpr int NUM_CHANNELS = 3;  // Match sample channels
    int channels;           // Runtime channel count
    float* concentration;   // [cells * channels] - multi-channel concentration
    float* gradient_x;      // [cells * channels] - per-channel x gradient
    float* gradient_y;      // [cells * channels] - per-channel y gradient
    float* laplacian;       // [cells * channels] - per-channel laplacian
    float* sources;         // [cells * channels] - per-channel sources
    float* decay_factors;   // [cells] - shared decay (same for all channels)
    TemporalTube* history;
    float cached_mean[NUM_CHANNELS];  // Per-channel mean
};

struct BehavioralState {
    float position[2];
    float velocity[2];
    float* hw_coords;
    float* task_coords;
    float* gen_coords;
    float gradient_memory[GRADIENT_HISTORY][2];
    float velocity_history[GRADIENT_HISTORY][2];
    float exploration_noise;
    float exploration;
    float sensitivity;
    int memory_index;
    uint64_t genome_hash;
    int organism_id;
};

// From telemetry_probes.cu
struct GenomeComplexityMetrics {
    float delta_diversity;
    float hash_entropy;
    float parameter_variance;
    int unique_hashes;
    float avg_deltas_per_genome;
};

struct ArchiveTopologyMetrics {
    int occupied_cells;
    int frontier_cells_gained;
    int frontier_cells_lost;
    int sparse_cell_count;
    float niche_entropy;
    float novelty_gradient;

    float elite_fitness_best;
    float elite_fitness_mean;
    float elite_fitness_delta;
    float quality_floor;
    float quality_mean;
    float quality_range;

    float density_mean;
    float density_max;
    float density_variance;

    float hw_axis_min, hw_axis_max, hw_axis_mean;
    float task_axis_min, task_axis_max, task_axis_mean;
    float gen_axis_min, gen_axis_max, gen_axis_mean;

    float axis_corr_hw_task;
    float axis_corr_hw_gen;
    float axis_corr_task_gen;

    int total_population;
    int births_since_checkpoint;
    int deaths_since_checkpoint;

    float hash_clustering_coefficient;
};

struct DIRESAEvolutionMetrics {
    float recon_loss_hw;
    float recon_loss_task;
    float recon_loss_gen;
    float recon_loss_total;
    float behavioral_drift_rate;
    float latent_utilization;
    float compression_ratio;
    float hardware_feature_correlation;
    float gradient_magnitude_avg;
    int archive_injections;
};

struct TaskPerformanceMetrics {
    float accuracy;
    float train_accuracy;
    float test_accuracy;
    float loss;
    float classification_stability;
    float avg_confidence;
    int correct_predictions;
    int total_predictions;
    int per_class_correct[NUM_CLASSES_MAX];
    int per_class_total[NUM_CLASSES_MAX];
};

struct PopulationMetrics {
    float total_accuracy;
    float total_generalization_gap;
    float total_hardware_efficiency;
    float total_fitness;
};

struct MemoryAllocationMetrics {
    size_t total_gpu_allocated;
    size_t total_gpu_free;
    size_t total_gpu_capacity;
    size_t unified_memory_allocated;
    size_t archive_pools_size;
    size_t training_pools_size;
    size_t ca_state_size;
    size_t chemical_field_size;
    size_t behavioral_pools_size;
    size_t diresa_weights_size;
    size_t autodiff_tape_size;
    size_t device_heap_limit;
    size_t device_heap_allocated;
};

struct TelemetryBuffer {
    GenomeComplexityMetrics genome_complexity;
    ArchiveTopologyMetrics archive_topology;
    DIRESAEvolutionMetrics diresa_evolution;
    TaskPerformanceMetrics task_performance;
    PopulationMetrics population_metrics;
    MemoryAllocationMetrics memory_allocation;
    int generation;
    bool valid;

    ArchiveTopologyMetrics last_checkpoint;
    int last_occupancy[MAX_CELLS];
    int last_total_spawned;
    int last_total_culled;
};

// From config.cu
struct StateExportEntry {
    uint32_t provenance_source;
    uint32_t fields_written_mask;
    uint64_t sequence_number;

    int generation;
    int batch_size;
    int num_classes;
    int grid_size;
    int correct_count;
    float loss;
    float accuracy;

    unsigned char sample_images[AUDIT_SAMPLE_COUNT * CA_FIELD_SIZE];
    int sample_labels[AUDIT_SAMPLE_COUNT];
    float sample_logits[AUDIT_SAMPLE_COUNT * NUM_CLASSES_MAX];
    int sample_predictions[AUDIT_SAMPLE_COUNT];
    float sample_confidences[AUDIT_SAMPLE_COUNT];

    float ca_snapshot[CA_FIELD_SIZE];

    float train_accuracy;
    float test_accuracy;
    float generalization_gap;

    int pool_alive_count;
    int pool_capacity;


    int archive_occupied_cells;
    int frontier_cells_gained;
    int frontier_cells_lost;
    int sparse_cell_count;
    float niche_entropy;
    float novelty_gradient;

    float elite_fitness_best;
    float elite_fitness_mean;
    float elite_fitness_delta;
    float quality_floor;
    float quality_mean;
    float quality_range;

    float density_mean;
    float density_max;
    float density_variance;

    float hw_axis_min, hw_axis_max, hw_axis_mean;
    float task_axis_min, task_axis_max, task_axis_mean;
    float gen_axis_min, gen_axis_max, gen_axis_mean;

    int total_population;
    int births_this_gen;
    int deaths_this_gen;

    float diresa_recon_loss_hw;
    float diresa_recon_loss_task;
    float diresa_recon_loss_gen;
    float diresa_recon_loss_total;
    float diresa_behavioral_drift;
    float diresa_latent_utilization;

    int genome_unique_hashes;
    float genome_hash_entropy;
    float genome_avg_deltas;

    float per_class_correct[NUM_CLASSES_MAX];
    float per_class_total[NUM_CLASSES_MAX];

    int pool_entry_alive[POOL_CAPACITY_MAX];
    float pool_entry_fitness[POOL_CAPACITY_MAX];
    float pool_entry_hunger[POOL_CAPACITY_MAX];
    int pool_entry_age[POOL_CAPACITY_MAX];
    int pool_entry_num_deltas[POOL_CAPACITY_MAX];
    uint64_t pool_entry_genome_hash[POOL_CAPACITY_MAX];

    float axis_corr_hw_task;
    float axis_corr_hw_gen;
    float axis_corr_task_gen;
    float hash_clustering_coefficient;

    float hw_warp_divergence_entropy;
    float hw_warp_convergence_rate;
    float hw_active_thread_fraction;
    float hw_memory_coalescing_efficiency;
    float hw_cache_line_utilization;
    float hw_tensor_core_usage;
    float hw_instruction_throughput;
    float hw_occupancy_variance;
    float hw_arithmetic_intensity;
    float hw_memory_bandwidth_saturation;

    float chemical_concentration_mean;
    float chemical_concentration_max;
    float chemical_gradient_magnitude_mean;
    float chemical_source_activity;
    float chemical_decay_rate_mean;

    float flow_lenia_mass_total;
    float flow_lenia_mass_conservation_error;
    float flow_lenia_affinity_mean;
    float flow_lenia_flow_magnitude_mean;

    float fitness_alpha;
    float fitness_beta;
    float fitness_gamma;
    float fitness_delta;

    size_t memory_gpu_allocated;
    size_t memory_gpu_free;
    size_t memory_ca_state_size;
    size_t memory_chemical_field_size;
    size_t memory_archive_size;

    int state_agent_count;
    float state_agent_pos_x[STATE_EXPORT_AGENT_COUNT];
    float state_agent_pos_y[STATE_EXPORT_AGENT_COUNT];
    float state_agent_vel_x[STATE_EXPORT_AGENT_COUNT];
    float state_agent_vel_y[STATE_EXPORT_AGENT_COUNT];
    float state_agent_exploration[STATE_EXPORT_AGENT_COUNT];
    float state_agent_sensitivity[STATE_EXPORT_AGENT_COUNT];

    int state_voronoi_count;
    int state_voronoi_density[STATE_EXPORT_VORONOI_COUNT];
    float state_voronoi_radius[STATE_EXPORT_VORONOI_COUNT];
    float state_voronoi_hw_centroid[STATE_EXPORT_VORONOI_COUNT * BEHAVIORAL_DIM_HW];
    float state_voronoi_task_centroid[STATE_EXPORT_VORONOI_COUNT * BEHAVIORAL_DIM_TASK];
    float state_voronoi_gen_centroid[STATE_EXPORT_VORONOI_COUNT * BEHAVIORAL_DIM_GEN];
    int state_voronoi_best_elite_idx[STATE_EXPORT_VORONOI_COUNT];

    int state_archive_count;
    float state_archive_fitness[STATE_EXPORT_ARCHIVE_COUNT];
    float state_archive_coherence[STATE_EXPORT_ARCHIVE_COUNT];
    float state_archive_effective_rank[STATE_EXPORT_ARCHIVE_COUNT];
    uint16_t state_archive_generation[STATE_EXPORT_ARCHIVE_COUNT];
    uint64_t state_archive_genome_hash[STATE_EXPORT_ARCHIVE_COUNT];
    uint32_t state_archive_parent_id_0[STATE_EXPORT_ARCHIVE_COUNT];
    uint32_t state_archive_parent_id_1[STATE_EXPORT_ARCHIVE_COUNT];
    float state_archive_hw_coords[STATE_EXPORT_ARCHIVE_COUNT * BEHAVIORAL_DIM_HW];
    float state_archive_task_coords[STATE_EXPORT_ARCHIVE_COUNT * BEHAVIORAL_DIM_TASK];
    float state_archive_gen_coords[STATE_EXPORT_ARCHIVE_COUNT * BEHAVIORAL_DIM_GEN];
    float state_archive_hardware_features[STATE_EXPORT_ARCHIVE_COUNT * HARDWARE_FEATURES_DIM];

    float state_chemical_sample[STATE_EXPORT_CHEM_SIZE * STATE_EXPORT_CHEM_SIZE];

    int pool_total_spawned;
    int pool_total_culled;
};

struct Architecture {
    int num_heads;
    int channels;
    int hidden_dim;
    int head_dim;
    int grid_size;

    __device__ __host__ static Architecture maxBounds() {
        Architecture arch;
        arch.num_heads = NUM_HEADS;
        arch.channels = CHANNELS;
        arch.hidden_dim = HIDDEN_DIM;
        arch.head_dim = HEAD_DIM;
        arch.grid_size = GRID_SIZE;
        return arch;
    }
};

// From dataset_loader.cu
enum DatasetFormat {
    FORMAT_IDX_UBYTE,
    FORMAT_NPZ,
    FORMAT_CIFAR_BIN,
    FORMAT_WAV_METADATA,
    FORMAT_WAV_DIRS,
    FORMAT_TXT_TIMESERIES,
    FORMAT_WFDB,
    FORMAT_DAT_BINARY,
    FORMAT_TARGZ_IMAGES
};

enum FeatureEncoding {
    ENCODING_SPATIAL_2D,
    ENCODING_SPECTRAL_AUDIO,
    ENCODING_TEMPORAL_1D
};

enum DatasetModality {
    MODALITY_VISION,
    MODALITY_AUDIO,
    MODALITY_TIMESERIES,
    MODALITY_MEDICAL
};

struct DatasetDescriptor {
    const char* name;
    DatasetFormat format;
    DatasetModality modality;
    FeatureEncoding encoding;
    const char* base_path;

    size_t sample_rows;
    size_t sample_cols;
    size_t channels;
    size_t num_classes;

    size_t num_train;
    size_t num_test;

    size_t train_size_bytes;
    size_t test_size_bytes;

    bool has_separate_test;
    bool needs_preprocessing;

    int n_fft;
    int hop_length;
    int n_mels;

    bool preserve_stereo;
    int bit_depth;
    bool use_multi_resolution;
    int pyramid_levels;
    int hilbert_order;
};

// From device_trace.cu
struct KernelLaunchInfo {
    const char* kernel_name;
    const char* file;
    int line;
    unsigned int grid_x, grid_y, grid_z;
    unsigned int block_x, block_y, block_z;
    size_t shared_mem;
};

// From autodiff.cu
enum TapeOp {
    OP_NONE,
    OP_ADD,
    OP_MUL,
    OP_TANH,
    OP_RELU,
    OP_EXP,
    OP_LOG,
    OP_SQRT,
    OP_SIN,
    OP_COS,
    OP_MATMUL,
    OP_REDUCE_SUM,
    OP_REDUCE_MAX
};

struct TapeEntry {
    TapeOp op;
    int output_idx;
    int input1_idx;
    int input2_idx;
    float aux_data;
    int level;
};

struct ADTape {
    TapeEntry* entries;
    int capacity;
    int current_size;
    float* value_buffer;
    float* grad_buffer;
    int* value_levels;
    int value_capacity;
    int current_value_idx;
    int max_level;

    int needs_weight_restore;
    int restore_elite_idx;
};

// From diresa.cu
struct DIRESABatch {
    int input_dim;
    int output_dim;
    int batch_size;

    float* features;
    float* features_shuffled;
    int* shuffle_indices;

    float* latent;
    float* latent_shuffled;

    float* reconstructed;

    float* orig_distances;
    float* latent_distances;

    float recon_loss;
    float dist_loss;
    float cov_loss;
};

// From lifecycle_stages.cu - data only, methods stay in lifecycle_stages.cu
template<int SECTION_SIZE>
struct LocalOrganismState {
    int organism_indices[SECTION_SIZE];
    float local_fitness[SECTION_SIZE];
    float local_coherence[SECTION_SIZE];
    float gradient_history[SECTION_SIZE][8];
    LifecyclePhase phases[SECTION_SIZE];
    float stress_accum[SECTION_SIZE];
};

// From archive.cu
struct GPUElite {
    float* fitness;
    float* coherence;
    float* effective_rank;
    uint64_t* genome_hash;
    uint32_t* parent_ids;
    uint16_t* generation;
    float* hw_coords;
    float* task_coords;
    float* gen_coords;
    float* latent_genome;
    float* hardware_features;
    float* task_performance;
    float* per_class_accuracy;
    int hw_dim;
    int task_dim;
    int gen_dim;

    uint64_t* fitness_input_hash;
    int* fitness_computed_at_generation;

    half* weight_deltas;
    uint32_t* weight_delta_indices;
    uint16_t* num_weight_deltas;

    int* archived_num_heads;
    int* archived_channels;
    int* archived_head_dim;

    uint64_t* hash_table_keys;
    int* hash_table_values;
};

struct VoronoiCell {
    float* hw_centroid;
    float* task_centroid;
    float* gen_centroid;
    float radius;
    int density;
    int density_prev;
    float density_fluctuation;
    int best_elite_idx;
    float quality_threshold;
};

// From pool.cu - data only, methods (next, levy_stable) stay in pool.cu
struct PRNGState {
    uint64_t s0;
    uint64_t s1;
};

// From hardware_geometry.cu
struct HardwareGeometry {
    float warp_divergence_entropy;
    float warp_convergence_rate;
    float active_thread_fraction;

    float memory_coalescing_efficiency;
    float cache_line_utilization;
    float memory_divergence_spread;
    float bank_conflict_density;

    float tensor_core_usage;
    float tensor_memory_bandwidth;

    float instruction_throughput;
    float pipeline_stall_fraction;
    float occupancy_variance;

    float arithmetic_intensity;
    float memory_bandwidth_saturation;
};

struct ExecutionTrace {
    unsigned long long active_warps;
    unsigned long long divergent_branches;
    unsigned long long total_branches;

    unsigned long long global_loads;
    unsigned long long global_stores;
    unsigned long long l2_transactions;
    unsigned long long dram_transactions;
    unsigned long long shared_loads;
    unsigned long long shared_stores;
    unsigned long long bank_conflicts;

    unsigned long long inst_executed;
    unsigned long long inst_issued;
    unsigned long long cycles_elapsed;
    unsigned long long tensor_core_cycles;

    float sm_occupancy;
    float achieved_bandwidth;
    float peak_bandwidth;
};

struct TraceBuffer {
    ExecutionTrace* traces;
    int capacity;
    int current_idx;
};

// From training_types.cu
struct ClassificationHead {
    float* pooling_weights;
    float* fc_weights;
    float* fc_bias;
    volatile int pointers_ready;
};

struct CAParameterMap {
    int perception_start[NUM_HEADS];
    int interaction_start[NUM_HEADS];
    int value_start[NUM_HEADS];

    int head_param_offsets[NUM_HEADS];
    int head_param_counts[NUM_HEADS];

    int perception_size;
    int interaction_size;
    int value_size;

    int total_params;
    int total_ca_params;

    int batch_size;
    int grid_size;
    int channels;
    int hidden_dim;
};

struct HybridTrainingMode {
    bool use_gradients;
    bool use_selection;
    float gradient_fitness_weight;
    float coherence_fitness_weight;
    float* batch_samples;
    int* batch_labels;
    int batch_size;
    ClassificationHead* classifier;
    float learning_rate;
    float gradient_clip_norm;
    float* adam_m;
    float* adam_v;
    int perception_size;
    int interaction_size;
    int value_size;
    int policy_size;
    int adam_timestep;
    bool is_train_batch;
};

struct Dataset {
    const DatasetDescriptor* descriptor;
    unsigned char* samples;
    unsigned char* labels;
    int num_samples;
    bool is_train;
};

struct DatasetStats {
    int dataset_id;
    float population_mean_accuracy;
    float population_best_accuracy;
    float population_accuracy_variance;
    float niche_diversity;
    int num_generations_trained;
    bool activation_threshold_met;
};

struct AdaptiveCurriculum {
    DatasetStats stats[NUM_ACTIVE_DATASETS];
    int current_dataset_idx;
    int num_datasets_completed;
    float curriculum_progress;

    float accuracy_threshold;
    float diversity_threshold;
    float min_generations_threshold;
};

struct UnifiedGradientBuffer {
    float* perception_grads;
    float* interaction_grads;
    float* value_grads;

    float* pooling_weight_grads;
    float* fc_weight_grads;
    float* fc_bias_grads;

    int has_autodiff_grads;
    int has_backprop_grads;

    int perception_size;
    int interaction_size;
    int value_size;
    int num_classes;
    int num_features;
};

// From hybrid_lifecycle.cu
struct WaveBufferOffsets {
    int ca_states_offset;
    int ca_output_offset;
    int activations_offset;
    int affinity_offset;
    int flow_offset;
    size_t backward_ws_offset;
};

struct BackwardWorkspaceLayout {
    size_t fp16_a_offset;
    size_t fp16_b_offset;
    size_t dW_offset;
    size_t dI_offset;
    size_t W_T_offset;
    size_t im2col_offset;
    size_t dpregelu_offset;
    size_t total_bytes;
};

// From ca_state.cuh - must be before PoolEntry which uses it
struct MultiHeadCAState {
    half* perception_weights;
    half* interaction_weights;
    half* value_weights;

    float* ca_concentration;
    float* ca_output;

    float* affinity_reduced;
    float* flow_field;
    float* reintegration_buffer;

    half* fp16_workspace;
    float* fp32_workspace;

    ADTape tape;

    TraceBuffer trace;

    float* perception_saved;
    float* interaction_saved;
    float* pre_gelu_saved;
};

// From diresa_types.cuh - must be before PoolEntry which uses it
struct DIRESAWeights {
    int input_dim;
    int output_dim;
    int hidden1;
    int hidden2;

    float* encoder_w1;
    float* encoder_b1;
    float* encoder_w2;
    float* encoder_b2;
    float* encoder_w3;
    float* encoder_b3;

    float* decoder_w1;
    float* decoder_b1;
    float* decoder_w2;
    float* decoder_b2;
    float* decoder_w3;
    float* decoder_b3;

    float cov_weight;
    float learning_rate;
    uint32_t training_step;

    float temperature;
    int replica_id;

    float distance_exponent;
    float quality_weight;
};

// From genome_ops.cuh
struct PoolEntry {
    int id;
    MeasuredValue<float> fitness;
    MeasuredValue<float> coherence;
    MeasuredValue<float> task_accuracy;
    MeasuredValue<float> train_accuracy;
    MeasuredValue<float> test_accuracy;
    MeasuredValue<float> task_loss;
    MeasuredValue<float> classification_stability;
    MeasuredValue<float> avg_confidence;
    int per_class_correct[NUM_CLASSES_MAX];
    int per_class_total[NUM_CLASSES_MAX];
    MeasuredValue<float> generalization_gap;
    MeasuredValue<float> hardware_efficiency;
    MeasuredValue<float> gradient_magnitude;
    MeasuredValue<float> effective_rank;
    MeasuredValue<float> recon_loss_hw;
    MeasuredValue<float> recon_loss_task;
    MeasuredValue<float> recon_loss_gen;
    MeasuredValue<float> recon_loss_total;
    MeasuredValue<float> behavioral_drift_rate;
    MeasuredValue<float> latent_utilization;
    MeasuredValue<float> compression_ratio;
    MeasuredValue<float> hardware_feature_correlation;
    MeasuredValue<float> hunger;
    int age;

    LifecyclePhase phase;
    MeasuredValue<float> stress;
    MeasuredValue<float> dormancy;
    MeasuredValue<float> reactivation;
    int field_epoch;
    int epoch_start_generation;
    float* signal_flow_accumulator;
    float* behavioral_coords;
    int niche_id;
    NicheState niche_state;
    float niche_rank;
    int last_archive_use;
    PhaseTransitionRecord phase_record;
    uint64_t genome_hash;
    int generation;
    float* gradients;
    uint64_t parent_hash;
    int parent_idx;
    uint16_t num_deltas;
    uint16_t max_deltas;
    uint16_t* delta_indices;
    float* delta_values;
    int num_heads;
    int channels;
    int hidden_dim;
    int head_dim;
    int grid_size;
    int num_tempering_replicas;
    int diresa_hidden1;
    int diresa_hidden2;
    int diresa_batch_size;
    float anneal_step;
    float cov_target;
    unsigned long long active_warps;
    unsigned long long divergent_branches;
    unsigned long long total_branches;
    unsigned long long global_loads;
    unsigned long long global_stores;
    unsigned long long l2_transactions;
    unsigned long long dram_transactions;
    unsigned long long inst_executed;
    unsigned long long inst_issued;
    unsigned long long cycles_elapsed;
    unsigned long long tensor_core_cycles;
    float dist_weight;
    float recon_weight;
    float distance_exponent;
    float quality_weight;
    float fitness_rank_exponent;
    float fitness_coherence_exponent;
    float fitness_coupling_exponent;
    float fitness_task_exponent;
    float fitness_gen_exponent;
    float fitness_efficiency_exponent;
    float baldwin_sensitivity;
    int coherence_window_size;
    float renyi_q;

    float flow_beta_A;
    float flow_n;
    float flow_s;
    float flow_alpha_min;
    float flow_alpha_max;
    float flow_sharpness;
    float flow_resource_dt;

    MultiHeadCAState* ca_state;

    DIRESAWeights* diresa_task_weights;
    DIRESAWeights* diresa_hw_weights;
    DIRESAWeights* diresa_gen_weights;
    int diresa_task_input_dim;
};

// From pool_types.cuh
struct ComponentPool {
    PoolEntry* entries;
    cuda::atomic<int, cuda::thread_scope_system> active_count;
    cuda::atomic<int, cuda::thread_scope_system> total_spawned;
    cuda::atomic<int, cuda::thread_scope_system> total_culled;
    int capacity;

    int* alive_indices;
    int alive_indices_count;

    bool* alive_flags;
    float* fitness_values;
};

// From parallel_compaction.cu
struct MemoryUpdateParams {
    float decay_threshold;
    float consolidation_threshold;
    float flow_lenia_dt;
    float fitness_trend;
    int old_count;
    int new_count;
};

struct Organism {

    ComponentPool* pool;
    GPUElite* archive;
    int archive_size;
    VoronoiCell* voronoi_cells;
    int num_voronoi_cells;
    float voronoi_correlation_exponent;
    MultiHeadCAState* ca_state_pool;
    BehavioralState* behavioral_agents;

    ChemicalField* chemical_field;

    float* fitness_history;
    float* effective_rank_history;
    float* coherence_history;
    int generation;
    int active_components;

    // Pool statistics (used by compute_pool_stats_device)
    float* stats_avg_fitness;
    float* stats_avg_coherence;
    float* stats_avg_age;
    float* stats_genetic_diversity;

    float* behavioral_field_pool;
    float* behavioral_gradient_pool;
    float* behavioral_coords_pool;
    float* coherence_workspace_pool;
    float* memory_data_pool;
    float* fitness_rank_pool;
    float* fitness_coherence_pool;
    float* correlation_matrix_pool;
    float* prediction_error_history;
    float* fitness_workspace_pool;

    
    DIRESAWeights* diresa_genome_weights;
    float* diresa_genome_weight_pool;
    
    DIRESAWeights* per_entry_diresa_task_weights;  
    DIRESAWeights* per_entry_diresa_hw_weights;    
    DIRESAWeights* per_entry_diresa_gen_weights;   
    float* per_entry_diresa_task_weight_pool;
    float* per_entry_diresa_hw_weight_pool;
    float* per_entry_diresa_gen_weight_pool;

    float* hw_coords_pool;     
    float* task_coords_pool;   
    float* gen_coords_pool;    

    uint16_t* delta_indices_pool;
    float* delta_values_pool;
    uint16_t* delta_counts_pool;

    float* latent_genome_pool;  

    HardwareGeometry* hardware_geom;

    CAParameterMap* param_map;

    int current_activation_grid_size;  

    HybridTrainingMode* training_mode;
    Dataset** dataset_array;
    Dataset* current_dataset;
    Dataset** test_dataset_array;
    Dataset* current_test_dataset;
    ClassificationHead* classifier;
    AdaptiveCurriculum* curriculum;
    float* voronoi_occupancy_histogram;
    float* pool_task_accuracies;

    void* lifecycle_states;

    TelemetryBuffer* telemetry;

    // Simulation state fields
    float dt;
    float global_time;
    int field_size;
    float embedding_learning_rate;

    // Loss/coherence tracking
    float* loss_history;
    int loss_history_length;
    float* coherence_output;

    // Behavioral gradients
    float* behavioral_gradients;
    float* features_buffer;

    // Attractors
    float* attractor_positions;
    float* attractor_strengths;
    int num_attractors;

    // Reduction workspace
    float* reduction_partial_sums;

    int* pool_compaction_flags;
    int* pool_compaction_scan;
    int* pool_compaction_recursive_workspace;
    int* pool_compaction_scan_recursive;  // Alias for pool_compaction_recursive_workspace

    float* resource_density;
    float* resource_next;
    float* fitness_landscape;
    float* resource_gradient_x;
    float* resource_gradient_y;

    int* lifecycle_phase_counts;

    float* reduction_workspace;  
    int reduction_num_blocks;    
    int reduction_total_cells;   

    float* gradient_features_pool;
    float* gradient_logits_pool;
    float* gradient_loss_pool;
    float* gradient_logit_grads_pool;
    float* gradient_magnitudes_pool;

    float* pooling_weights_grad;
    float* fc_weights_grad;
    float* fc_bias_grad;
    float* features_grad;

    float* adam_m_pooling;
    float* adam_v_pooling;
    float* adam_m_fc_weights;
    float* adam_v_fc_weights;
    float* adam_m_fc_bias;
    float* adam_v_fc_bias;

    uint8_t* elite_compressed_pool;
    uint32_t* elite_size_pool;

    float* adam_m_ca_pool;  
    float* adam_v_ca_pool;  

    float* batch_ca_states_pool;
    float* batch_ca_input_grads;
    int* batch_labels_pool;

    float* task_loss_pool;
    float* reg_loss_pool;
    float* rank_loss_pool;
    float* coherence_loss_pool;
    float* diversity_loss_pool;
    float* total_loss_pool;

    int* inherit_child_indices;
    int* inherit_parent_indices;
    int* num_pending_inherits;

    curandState* rng_states;

    volatile int* phase_barrier_counter;
    volatile int* phase_barrier_generation;
    int phase_barrier_num_blocks;

    int lifecycle_entry_idx;
    float* lifecycle_workspace_genomes;
    int lifecycle_wave_start;

    void* diresa_batch_context;
    curandState* diresa_rng_states;

    float* workspace_genomes;
    unsigned int init_seed;

    float* history_data_buffer;
    int tube_capacity;
    float tube_decay_rate;
    int tube_entry_size;

    int classifier_num_classes;
    unsigned int classifier_seed;

    BehavioralInitSlots behavioral_slots;

    // Single-kernel architecture runtime state
    AuditBuffer* audit_buffer;
    Architecture current_arch;
    int current_wave_start;
    int current_wave_size;
    float spawn_probability;
    float* spawn_workspace;
    float hunger_threshold;
    float diffusion_dt;
    int snapshot_field_size;
    float* attractor_field;
    int current_entry_idx;
    int init_pool_capacity;
    int chem_grid_size;
    float* clear_buffer_ptr;
    int clear_buffer_size;

    // Embedding/behavioral
    float* embedding_weights;
    int hw_dim;
    int task_dim;
    int gen_dim;

    // Autodiff tape fields
    ADTape* ad_tape;
    TapeEntry* ad_entries_pool;
    float* ad_values_pool;
    float* ad_grads_pool;
    int* ad_levels_pool;
    int ad_tape_capacity;
    int ad_value_capacity;
    int ad_output_idx;
    float ad_output_grad;

    // Genome gradient fields
    int* genome_param_indices;
    int num_genome_params;
    float* output_gradients;
    int genome_size;
    float learning_rate;
    float gradient_clip_norm;
    float* correlation_matrix;

    // Loss function fields
    float* loss_predictions;
    float* loss_targets;
    float* loss_out;
    int loss_batch_size;
    int loss_dim;
    float* loss_logits;
    int* loss_labels;
    float* loss_gradients;
    int loss_num_classes;
    float loss_smoothing;

    // Archive/behavioral dimension fields
    float* reconstruction_error;
    int* embedding_dim;
    TemporalTube* temporal_tube;
    int behavioral_dim_hw;
    int behavioral_dim_task;
    int behavioral_dim_gen;

    // Gradient fitness fields
    int* gf_param_start_indices;
    int* gf_param_counts;
    float* gf_gradient_magnitudes;
    int gf_num_heads;
    float* gf_effective_rank_out;
    float gf_renyi_order_q;
    float gf_task_accuracy;
    float gf_generalization_gap;
    float gf_effective_rank;
    float gf_hardware_efficiency;
    float gf_alpha;
    float gf_beta;
    float gf_gamma;
    float gf_delta;
    float* gf_fitness_out;
    float* gf_current_fitness;
    float* gf_fitness_ema;
    int gf_num_entries;
    float gf_ema_alpha;
    float* gf_absolute_fitness;
    float* gf_behavioral_coords;
    float* gf_relative_fitness;
    int gf_num_components;
    int gf_behavioral_dim;
    int gf_k_neighbors;

    // Tensor core fields
    float* tensor_fp32_data;
    half* tensor_fp16_data;
    int tensor_M;
    int tensor_N;
    half* tensor_A;
    half* tensor_B;
    float* tensor_C;
    int tensor_K;
    float* activation_data;
    int activation_size;
    half* tensor_neighborhood_fp16;
    MultiHeadCAState* multihead_ca_state;
    float* tensor_perception_out;
    int current_head_id;
    int max_grid_size;

    // Dataset loader fields
    float* dataset_waveform;
    float* dataset_windowed;
    int dataset_window_start;
    int dataset_window_size;
    cufftComplex* dataset_fft_out;
    float* dataset_magnitude;
    float* dataset_phase;
    int dataset_n_bins;
    float* dataset_phase_prev;
    float* dataset_phase_velocity;
    float dataset_hop_length;
    float dataset_sample_rate;
    float* dataset_mel_magnitude;
    float* dataset_mel_phase;
    float* dataset_mel_phase_velocity;
    int dataset_n_mels;
    int dataset_sample_rate_int;
    int dataset_n_fft;
    Dataset* dataset;
    int dataset_batch_size;
    int dataset_batch_offset;
    int dataset_grid_size;

    // CA state fields
    float* ca_state;
    int ca_channels;
    float* ca_prev_concentration;
    float* behavioral_field;

    // Reaction-diffusion fields
    float* rd_resource_density;
    float* rd_fitness_landscape;
    float* rd_resource_gradient_x;
    float* rd_resource_gradient_y;
    float* rd_resource_next;
    float rd_diffusivity;
    float rd_flow_strength;

    // Test output fields
    int* test_current_size_out;
    int* test_current_value_idx_out;
    float* test_first_grad_out;
    int* test_y_idx_out;

    // Parallel scan fields
    int* scan_input;
    int* scan_output;
    int* scan_block_sums;
    int* scan_block_prefixes;
    int scan_N;
    int scan_num_blocks;

    // Compaction fields
    int* compact_valid_flags;
    int* compact_write_indices;
    MemoryEntry* compact_temp_buffer;
    int compact_old_count;
    int compact_new_count;
    float compact_decay_threshold;

    // Memory update fields
    MemoryUpdateParams* memory_update_params;
    TemporalTube* memory_tubes;

    // Context fields
    float ctx_metabolic;
    float ctx_stress;
    float ctx_morphogen;
    float ctx_complexity;
    float ctx_niche;
    float ctx_learning;
    float ctx_performance;

    // Elite fields
    GPUElite* gpu_elite;
    int elite_idx;

    // Weight conversion fields (cuda_primitives)
    float* weights_fp32;
    half* weights_fp16;
    int weights_size;

    // Strided memory copy fields
    const float* strided_src_fp32;
    half* strided_dst_fp16;
    float* strided_dst_fp32;
    int strided_batch_size;
    int strided_slice_size;
    int strided_src_stride;
    int strided_dst_stride;

    // Batched GEMM fields
    const half* gemm_A;
    const half* gemm_B;
    float* gemm_C;
    int gemm_M;
    int gemm_N;
    int gemm_K;
    int gemm_A_head_stride;
    int gemm_B_head_stride;
    int gemm_C_head_stride;

    // Transpose fields
    const half* transpose_A;
    half* transpose_B;
    int transpose_M;
    int transpose_N;
    int transpose_A_head_stride;
    int transpose_B_head_stride;

    // Batched strided fields
    const float* batched_strided_src_fp32;
    half* batched_strided_dst_fp16;
    float* batched_strided_dst_fp32;
    int batched_src_head_stride;
    int batched_src_batch_stride;
    int batched_dst_head_stride;
    int batched_dst_batch_stride;
    int strided_batch_offset;

    // Weight gradient accumulation fields
    const float* weight_grad_src;
    float* grad_buffer;
    const int* head_offsets;
    int weight_size;
    int dW_head_stride;

    // Backward pass fields
    const float* backward_dL_dI;
    const float* backward_pre_gelu;
    float* backward_dL_dpregelu;
    const float* backward_dL_dP;
    const float* backward_P;
    float* backward_dL_dprerelu;
    int backward_elements_per_head;
    int backward_src_head_stride;
    int backward_dst_head_stride;

    // Im2col fields
    const float* im2col_input;
    float* im2col_col;
    int im2col_batch_size;
    int im2col_input_head_stride;
    int im2col_col_head_stride;

    // Col2im fields
    const float* col2im_col;
    float* col2im_output_grad;
    int col2im_batch_size;
    int col2im_col_head_stride;
    int col2im_output_head_stride;

    // FP32/FP16 conversion fields
    float* conv_fp32;
    half* conv_fp16;
    int conv_size;

    // Weight gradient accumulation fields (autodiff_integration)
    const float* weight_grads_src;
    float* weight_grads_dst;
    int weight_grads_offset;
    int weight_grads_size;

    // Training types fields
    UnifiedGradientBuffer* unified_grad_buffer;
    float* tt_perception_grads;
    float* tt_interaction_grads;
    float* tt_value_grads;
    float* tt_pooling_weight_grads;
    float* tt_fc_weight_grads;
    float* tt_fc_bias_grads;
    float* tt_pool_task_accuracies;
    float* tt_voronoi_occupancy_histogram;
    int tt_pool_size;
    int tt_num_voronoi_cells;

    // Classifier fields
    int cls_num_classes;
    float* classifier_workspace;

    // CA fields for autodiff
    CAParameterMap* ca_param_map;
    float* ca_output;
    float* perception_saved;
    float* interaction_saved;
    float* pre_gelu_saved;
    int ca_grid_size;
    int micro_batch_size;
    int micro_batch_offset;

    // Genome hash
    uint64_t genome_hash;

    // Float transpose fields (for autodiff_integration)
    const float* transpose_A_fp32;
    float* transpose_B_fp32;

    // Classifier fields
    int cls_batch_size;
    int cls_num_heads;
    int cls_num_features;
    float* cls_features;
    float* cls_fc_weights;
    float* cls_fc_bias;
    float* cls_logits;
    float* cls_probabilities;
    int* cls_labels;
    int* cls_correct_count;
    float* cls_loss_out;
    float* cls_logit_grads;
    float* cls_fc_weights_grad;
    float* cls_fc_bias_grad;
    float* cls_features_grad;
    float* cls_pooling_weights;
    float* cls_pooling_weights_grad;
    float* cls_ca_output;
    float* cls_ca_output_grad;

    // Delta indices buffer
    uint16_t* delta_indices_buffer;
    float* delta_values_buffer;

    // Adam optimizer fields
    float* adam_m;
    float* adam_v;
    float adam_beta1;
    float adam_beta2;
    float adam_epsilon;
    int adam_t;

    // Voronoi centroid fields
    float* voronoi_hw_centroids;
    float* voronoi_task_centroids;
    float* voronoi_gen_centroids;

    // Unified classifier Adam fields
    float* adam_m_classifier;
    float* adam_v_classifier;
    int adam_timestep;

    // Classification head pointer
    ClassificationHead* classification_head;

    // =========================================================================
    // Fields for main.cu allocation compatibility (OrganismPreallocatedBuffers)
    // =========================================================================

    // Pool sub-buffers (allocated separately, wired into pool struct on device)
    PoolEntry* pool_entries;
    int* pool_alive_indices;
    bool* pool_alive_flags;
    float* pool_fitness_values;

    // Archive hash table
    uint64_t* archive_hash_table_keys;
    int* archive_hash_table_values;

    // Archive data arrays
    float* archive_fitness;
    float* archive_coherence;
    float* archive_effective_rank;
    uint64_t* archive_genome_hash;
    uint32_t* archive_parent_ids;
    uint16_t* archive_generation;
    uint64_t* archive_fitness_input_hash;
    int* archive_fitness_computed_at_generation;
    float* archive_hw_coords;
    float* archive_task_coords;
    float* archive_gen_coords;
    float* archive_latent_genome;
    float* archive_hardware_features;
    float* archive_task_performance;
    float* archive_per_class_accuracy;

    // Behavioral coord buffers (aliases for hw_coords_pool etc.)
    float* behavioral_hw_coords_buffer;
    float* behavioral_task_coords_buffer;
    float* behavioral_gen_coords_buffer;

    // Voronoi centroid buffers (aliases)
    float* voronoi_hw_centroid_buffer;
    float* voronoi_task_centroid_buffer;
    float* voronoi_gen_centroid_buffer;

    // Chemical field history
    TemporalTube* chemical_field_history;
    MemoryEntry* chemical_field_history_entries;

    // Memory tubes (behavioral memory)
    MemoryEntry* memory_tubes_entries;
    float* memory_tubes_data;

    // Large workspace arrays
    half* all_ca_weights;
    float* all_ca_state;
    float* all_chem_fields;
    float* all_rd_fields;
    float* shared_workspace;

    // FP32/FP16 CA workspaces
    float* fp32_ca_workspace;
    half* fp16_ca_workspace;

    // Gradients buffer
    float* gradients_buffer;

    // Trace array
    ExecutionTrace* trace_array;

    // Autodiff tape pools
    TapeEntry* ad_tape_entries_pool;
    float* ad_tape_values_pool;
    float* ad_tape_grads_pool;
    int* ad_tape_levels_pool;

    // Activations saved
    float* perception_activations_saved;
    float* interaction_activations_saved;
    float* pre_gelu_values_saved;

    // Batched buffers
    float* batched_ca_output;
    float* batch_affinity_reduced;
    float* batch_flow_field;
    float* batch_reintegration_buffer;
    float* batch_prev_concentration;
    float* batch_samples_pool;

    // Behavioral buffers
    float* behavioral_features_buffer;
    float* behavioral_embedding_weights;
    float* behavioral_reconstruction_error;

    // Gradient buffers
    float* grad_concentration_buffer;
    float* ca_output_grad_buffer;
    float* dL_dperception_buffer;
    float* dL_dinteraction_buffer;

    // Workspace genome buffers
    float* component_workspace_genomes_buffer;
    float* behavioral_workspace_genomes_buffer;
    float* organism_workspace_genomes;

    // Backward workspace
    char* backward_workspace;  // Changed from void* for pointer arithmetic

    // Organism nested pointer (for device-side access pattern)
    Organism* organism;

    // Self-pointer for backward compatibility (organism->buffers->field == organism->field)
    Organism* buffers;

    // Flow gradient accumulators
    float flow_beta_A_grad;
    float flow_n_grad;
};

// Alias for backward compatibility with main.cu allocation code
typedef Organism OrganismPreallocatedBuffers;

#endif
