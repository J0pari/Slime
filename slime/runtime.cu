#ifndef RUNTIME_CU
#define RUNTIME_CU

#include "core/organism.cu"
#include "debug/device_trace.cu"

__device__ void init_behavioral_dimensions_device(Organism* organism) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Archive tessellation uses fixed dimensions (constants)
        organism->archive->hw_dim = BEHAVIORAL_DIM_HW;
        organism->archive->task_dim = BEHAVIORAL_DIM_TASK;
        organism->archive->gen_dim = BEHAVIORAL_DIM_GEN;
    }
}

// Initialize batch_prev_concentration from chemical/resource fields before first training iteration
__device__ void init_batch_prev_concentration_device(Organism* organism) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    ComponentPool* pool = organism->pool;
    ChemicalField* chem = organism->chemical_field;
    HybridTrainingMode* training_mode = organism->training_mode;

    if (!pool || !chem || !training_mode) return;

    const float* chem_concentration = chem->concentration;
    const float* chem_gradient_x = chem->gradient_x;
    const float* chem_gradient_y = chem->gradient_y;
    const float* chem_laplacian = chem->laplacian;
    const float* chem_sources = chem->sources;
    const float* chem_decay_factors = chem->decay_factors;
    int chem_channels = chem->channels;
    const float* resource_density = organism->resource_density;
    const float* fitness_landscape = organism->fitness_landscape;
    const float* resource_gradient_x = organism->resource_gradient_x;
    const float* resource_gradient_y = organism->resource_gradient_y;
    const float* attractor_field = organism->attractor_field;

    float* prev_conc = organism->buffers->batch_prev_concentration;
    if (!prev_conc) return;

    int batch_size = training_mode->batch_size;
    int alive_count = pool->alive_indices_count;

    // behavioral_dim for computing per-entry behavioral_field offset
    int behavioral_dim = organism->behavioral_dim_hw + organism->behavioral_dim_task + organism->behavioral_dim_gen;

    // Compute wave offsets for each alive entry
    size_t current_offset = 0;

    for (int compact_idx = 0; compact_idx < alive_count; compact_idx++) {
        int entry_idx = pool->alive_indices[compact_idx];
        PoolEntry* entry = &pool->entries[entry_idx];

        int grid_size = entry->grid_size;
        int ca_channels = entry->channels;
        int cells = grid_size * grid_size;
        int entry_buffer_size = batch_size * cells * ca_channels;

        float* entry_prev_conc = prev_conc + current_offset;

        // Compute per-entry behavioral_field from pool
        int behavioral_buffer_size = cells * behavioral_dim;
        const float* entry_behavioral_field = organism->behavioral_field_pool + entry_idx * behavioral_buffer_size;

        // Initialize all batch samples with the same field values
        for (int idx = global_tid; idx < entry_buffer_size; idx += total_threads) {
            int batch_idx = idx / (cells * ca_channels);
            int remainder = idx % (cells * ca_channels);
            int cell_idx = remainder / ca_channels;
            int c = remainder % ca_channels;

            float val;
            switch (c) {
                case 0: case 1: case 2: case 3: case 4: {
                    // Aggregate multi-channel chemical field into CA channels 0-4
                    float sum = 0.0f;
                    for (int cc = 0; cc < chem_channels; cc++) {
                        int field_idx = cc * cells + cell_idx;
                        switch (c) {
                            case 0: sum += chem_concentration[field_idx]; break;
                            case 1: sum += chem_gradient_x[field_idx]; break;
                            case 2: sum += chem_gradient_y[field_idx]; break;
                            case 3: sum += chem_laplacian[field_idx]; break;
                            case 4: sum += chem_sources[field_idx]; break;
                        }
                    }
                    val = sum / chem_channels;
                    break;
                }
                case 5:  val = chem_decay_factors[cell_idx]; break;
                case 6:  val = resource_density[cell_idx]; break;
                case 7:  val = fitness_landscape[cell_idx]; break;
                case 8:  val = resource_gradient_x[cell_idx]; break;
                case 9:  val = resource_gradient_y[cell_idx]; break;
                case 10: val = entry_behavioral_field[cell_idx]; break;
                case 11: {
                    int sample_base = batch_idx * cells * 3;
                    val = training_mode->batch_samples[sample_base + 0 * cells + cell_idx];
                    break;
                }
                case 12: {
                    int sample_base = batch_idx * cells * 3;
                    val = training_mode->batch_samples[sample_base + 1 * cells + cell_idx];
                    break;
                }
                case 13: {
                    int sample_base = batch_idx * cells * 3;
                    val = training_mode->batch_samples[sample_base + 2 * cells + cell_idx];
                    break;
                }
                case 14: {
                    // Bootstrap: recurrence starts as average concentration
                    float sum = 0.0f;
                    for (int cc = 0; cc < chem_channels; cc++) {
                        sum += chem_concentration[cc * cells + cell_idx];
                    }
                    val = sum / chem_channels;
                    break;
                }
                case 15: val = attractor_field[cell_idx]; break;
                default: DEVICE_FATAL("init_batch_prev_concentration: unhandled channel %d", c); break;
            }
            entry_prev_conc[idx] = val;
        }

        current_offset += entry_buffer_size;
    }
}

__device__ void wire_behavioral_agents_device(Organism* organism, int num_agents) {
    BehavioralState* agents = organism->behavioral_agents;
    float* hw_buffer = organism->hw_coords_pool;
    float* task_buffer = organism->task_coords_pool;
    float* gen_buffer = organism->gen_coords_pool;
    int hw_dim = organism->behavioral_dim_hw;
    int task_dim = organism->behavioral_dim_task;
    int gen_dim = organism->behavioral_dim_gen;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_agents) {
        agents[idx].hw_coords = &hw_buffer[idx * hw_dim];
        agents[idx].task_coords = &task_buffer[idx * task_dim];
        agents[idx].gen_coords = &gen_buffer[idx * gen_dim];
    }
}

__device__ void init_organism_device(Organism* organism) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    int pool_capacity = organism->init_pool_capacity;
    OrganismPreallocatedBuffers* buffers = organism->buffers;

    // Phase 1: Set base pointers that others depend on (single thread to avoid races)
    if (global_tid == 0) {
        organism->pool = buffers->pool;
        organism->archive = buffers->archive;
    }
    cg::this_grid().sync();

    // Phase 2: Set pool sub-fields and other dependent pointers (parallel safe now)
    int num_scalar_fields = 32;
    for (int field = global_tid; field < num_scalar_fields; field += total_threads) {
        switch (field) {
            case 0: organism->generation = 0; break;
            case 1: organism->pool->entries = buffers->pool_entries; break;
            case 2: organism->pool->alive_indices = buffers->pool_alive_indices; break;
            case 3: organism->pool->alive_indices_count = 0; break;
            case 4: organism->pool->alive_flags = buffers->pool_alive_flags; break;
            case 5: organism->pool->fitness_values = buffers->pool_fitness_values; break;
            case 6: organism->pool->capacity = pool_capacity; break;
            case 7: *((int*)&organism->pool->active_count) = POOL_CAPACITY_MIN; break;
            case 8: *((int*)&organism->pool->total_spawned) = 0; break;
            case 9: *((int*)&organism->pool->total_culled) = 0; break;
            case 10: organism->pool_compaction_flags = buffers->pool_compaction_flags; break;
            case 11: organism->pool_compaction_scan = buffers->pool_compaction_scan; break;
            case 12: organism->pool_compaction_recursive_workspace = buffers->pool_compaction_recursive_workspace; break;
            case 13: organism->pool_compaction_scan_recursive = buffers->pool_compaction_recursive_workspace; break;
            case 14: organism->inherit_child_indices = buffers->inherit_child_indices; break;
            case 15: organism->inherit_parent_indices = buffers->inherit_parent_indices; break;
            case 16: organism->num_pending_inherits = buffers->num_pending_inherits; break;
            case 17: organism->voronoi_cells = buffers->voronoi_cells; break;
            case 18: organism->behavioral_agents = buffers->behavioral_agents; break;
            case 19: organism->buffers = buffers; break;
            case 20: organism->phase_barrier_counter = buffers->phase_barrier_counter; break;
            case 21: organism->phase_barrier_generation = buffers->phase_barrier_generation; break;
            case 22: organism->phase_barrier_num_blocks = PROVENANCE_UNINITIALIZED_INT; break;
            case 23: *((volatile int*)organism->phase_barrier_counter) = 0; break;
            case 24: *((volatile int*)organism->phase_barrier_generation) = 0; break;
            case 25: organism->archive_size = 0; break;
            case 26: organism->num_voronoi_cells = pool_capacity; break;
            case 27: organism->active_components = POOL_CAPACITY_MIN; break;
            case 28: organism->delta_indices_buffer = buffers->delta_indices_buffer; break;
            case 29: organism->delta_values_buffer = buffers->delta_values_buffer; break;
            case 30: organism->gradients_buffer = buffers->gradients_buffer; break;
            case 31: organism->output_gradients = buffers->gradients_buffer; break;
        }
    }
    cg::this_grid().sync();

    // init_pool_device uses all threads for parallel work - each thread handles one pool entry
    init_pool_device(organism);
    cg::this_grid().sync();

    // Initialize wave state - derived from alive_indices_count set by init_pool_device
    if (global_tid == 0) {
        int alive_count = organism->pool->alive_indices_count;
        organism->current_wave_start = 0;  // First wave starts at index 0
        organism->current_wave_size = min(WAVE_SIZE, alive_count);
    }
    cg::this_grid().sync();
}

__device__ void init_voronoi_pointers_device(Organism* organism) {
    VoronoiCell* cells = organism->voronoi_cells;
    int num_cells = organism->num_voronoi_cells;
    float* hw_buffer = organism->voronoi_hw_centroids;
    float* task_buffer = organism->voronoi_task_centroids;
    float* gen_buffer = organism->voronoi_gen_centroids;
    int hw_dim = organism->archive->hw_dim;
    int task_dim = organism->archive->task_dim;
    int gen_dim = organism->archive->gen_dim;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        cells[idx].hw_centroid = &hw_buffer[idx * hw_dim];
        cells[idx].task_centroid = &task_buffer[idx * task_dim];
        cells[idx].gen_centroid = &gen_buffer[idx * gen_dim];
    }
}

__device__ void init_organism_phase2_device(Organism* organism) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    Dataset** dataset_array = organism->dataset_array;
    Dataset** test_dataset_array = organism->test_dataset_array;
    unsigned int seed = organism->init_seed;
    OrganismPreallocatedBuffers* buffers = organism->buffers;

    cudaError_t err;

    {
        organism->diresa_genome_weights = buffers->diresa_genome_weights;
        organism->diresa_genome_weight_pool = buffers->diresa_genome_weight_pool;
        organism->diresa_genome_grad_pool = buffers->diresa_genome_grad_pool;
        
        organism->per_entry_diresa_task_weights = buffers->per_entry_diresa_task_weights;
        organism->per_entry_diresa_hw_weights = buffers->per_entry_diresa_hw_weights;
        organism->per_entry_diresa_gen_weights = buffers->per_entry_diresa_gen_weights;
        organism->per_entry_diresa_task_weight_pool = buffers->per_entry_diresa_task_weight_pool;
        organism->per_entry_diresa_hw_weight_pool = buffers->per_entry_diresa_hw_weight_pool;
        organism->per_entry_diresa_gen_weight_pool = buffers->per_entry_diresa_gen_weight_pool;
        organism->per_entry_diresa_task_grad_pool = buffers->per_entry_diresa_task_grad_pool;
        organism->per_entry_diresa_hw_grad_pool = buffers->per_entry_diresa_hw_grad_pool;
        organism->per_entry_diresa_gen_grad_pool = buffers->per_entry_diresa_gen_grad_pool;

        float* primary_genome = &organism->workspace_genomes[GENOME_SIZE * 2];

        // Find first alive entry for initial genome copy
        int init_entry_idx = -1;
        int pool_capacity = organism->pool->capacity;
        for (int i = 0; i < pool_capacity; i++) {
            if (organism->pool->alive_flags[i]) {
                init_entry_idx = i;
                break;
            }
        }

        // At init, root entries have their genome directly in delta_values (no parent exists yet)
        if (init_entry_idx >= 0) {
            PoolEntry* entry = &organism->pool->entries[init_entry_idx];
            for (int i = 0; i < GENOME_SIZE; i++) {
                primary_genome[i] = entry->delta_values[i];
            }
        }

        // Behavioral dimensions are environmental constants (fixed for all entries)
        organism->archive->hw_dim = BEHAVIORAL_DIM_HW;
        organism->archive->task_dim = BEHAVIORAL_DIM_TASK;
        organism->archive->gen_dim = BEHAVIORAL_DIM_GEN;

        organism->hw_coords_pool = buffers->behavioral_hw_coords_buffer;
        organism->task_coords_pool = buffers->behavioral_task_coords_buffer;
        organism->gen_coords_pool = buffers->behavioral_gen_coords_buffer;
        wire_behavioral_agents_device(organism, POOL_CAPACITY_MAX);

        organism->archive->fitness = buffers->archive_fitness;
        organism->archive->coherence = buffers->archive_coherence;
        organism->archive->effective_rank = buffers->archive_effective_rank;
        organism->archive->genome_hash = buffers->archive_genome_hash;
        organism->archive->parent_ids = buffers->archive_parent_ids;
        organism->archive->generation = buffers->archive_generation;
        organism->archive->fitness_input_hash = buffers->archive_fitness_input_hash;
        organism->archive->fitness_computed_at_generation = buffers->archive_fitness_computed_at_generation;
        organism->archive->hw_coords = buffers->archive_hw_coords;
        organism->archive->task_coords = buffers->archive_task_coords;
        organism->archive->gen_coords = buffers->archive_gen_coords;
        organism->archive->latent_genome = buffers->archive_latent_genome;
        organism->archive->hardware_features = buffers->archive_hardware_features;
        organism->archive->task_performance = buffers->archive_task_performance;
        organism->archive->per_class_accuracy = buffers->archive_per_class_accuracy;
        organism->archive->hash_table_keys = buffers->archive_hash_table_keys;
        organism->archive->hash_table_values = buffers->archive_hash_table_values;

        init_hash_table_device(organism);

        organism->hw_coords_pool = buffers->hw_coords_pool;
        organism->task_coords_pool = buffers->task_coords_pool;
        organism->gen_coords_pool = buffers->gen_coords_pool;
        organism->prev_hw_coords_pool = buffers->prev_hw_coords_pool;
        organism->prev_task_coords_pool = buffers->prev_task_coords_pool;
        organism->prev_gen_coords_pool = buffers->prev_gen_coords_pool;

        organism->voronoi_hw_centroids = buffers->voronoi_hw_centroid_buffer;
        organism->voronoi_task_centroids = buffers->voronoi_task_centroid_buffer;
        organism->voronoi_gen_centroids = buffers->voronoi_gen_centroid_buffer;

        init_voronoi_pointers_device(organism);

        // Voronoi cells are synced from archive each generation via sync_voronoi_from_archive_device
        // At init time, archive is empty so num_voronoi_cells = 0
        organism->num_voronoi_cells = 0;

        int default_decay_rate_slot = GenomeParamTable::memory_default_decay_rate;
        float default_decay_rate_norm = genome_slot_to_unit(primary_genome, default_decay_rate_slot);
        float default_decay_rate = DEFAULT_DECAY_RATE_MIN + default_decay_rate_norm * (DEFAULT_DECAY_RATE_MAX - DEFAULT_DECAY_RATE_MIN);

        Architecture arch = get_arch_from_pool(organism, 0);
        int field_size = arch.grid_size * arch.grid_size;

        organism->telemetry = buffers->telemetry;
        organism->telemetry->valid = false;
        organism->telemetry->generation = 0;

        organism->telemetry->memory_allocation.device_heap_limit = DEVICE_MALLOC_HEAP_MB * BYTES_PER_MB;
        organism->telemetry->memory_allocation.device_heap_allocated = 0;

        organism->ca_state_pool = buffers->ca_state_pool;
        organism->chemical_field = buffers->chemical_field;

        organism->chemical_field->history = buffers->chemical_field_history;
        organism->chemical_field->history->entries = buffers->chemical_field_history_entries;

        organism->history_data_buffer = buffers->history_data_buffer;
        organism->tube_capacity = MAX_HISTORY_LENGTH;
        organism->tube_decay_rate = default_decay_rate;
        organism->tube_entry_size = field_size;

        organism->temporal_tube = organism->chemical_field->history;
        init_tube_device(organism->temporal_tube, organism->history_data_buffer,
                         organism->tube_capacity, organism->tube_entry_size, organism->tube_decay_rate);

        int perception_size = arch.num_heads * arch.channels * arch.head_dim;
        int interaction_size = arch.num_heads * arch.head_dim * arch.head_dim;
        int value_size = arch.num_heads * arch.head_dim * arch.channels;
        int total_weights_size = perception_size + interaction_size + value_size;

        dim3 weight_init_grid((total_weights_size + BLOCK_SIZE - 1) / BLOCK_SIZE, pool_capacity);
        dim3 weight_init_block(BLOCK_SIZE);

        float* all_ca_state = buffers->all_ca_state;
        MultiHeadCAState* ca_state_pool = buffers->ca_state_pool;

        for (int entry_idx = 0; entry_idx < pool_capacity; entry_idx++) {
            PoolEntry* entry = &organism->pool->entries[entry_idx];
            MultiHeadCAState* entry_ca_state = &ca_state_pool[entry_idx];

            float* entry_ca_base = all_ca_state + entry_idx * CA_STATE_STRIDE;

            entry_ca_state->ca_concentration = entry_ca_base;
            entry_ca_state->ca_output = entry_ca_base + CA_CONCENTRATION_SIZE;
            entry_ca_state->affinity_reduced = entry_ca_base + CA_CONCENTRATION_SIZE + CA_OUTPUT_SIZE;
            entry_ca_state->flow_field = entry_ca_base + CA_CONCENTRATION_SIZE + CA_OUTPUT_SIZE + CA_AFFINITY_SIZE;
            entry_ca_state->reintegration_buffer = entry_ca_base + CA_CONCENTRATION_SIZE + CA_OUTPUT_SIZE + CA_AFFINITY_SIZE + CA_FLOW_SIZE;

            half* entry_weights_base = buffers->all_ca_weights + entry_idx * CA_WEIGHTS_PER_ENTRY_STRIDE;
            entry_ca_state->perception_weights = entry_weights_base;
            entry_ca_state->interaction_weights = entry_weights_base + CA_PERCEPTION_WEIGHT_SIZE;
            entry_ca_state->value_weights = entry_weights_base + CA_PERCEPTION_WEIGHT_SIZE + CA_INTERACTION_WEIGHT_SIZE;

            // Accumulate workspace offset based on dimensions of all previous entries
            size_t fp32_ws_offset = 0;
            size_t fp16_ws_offset = 0;
            for (int prev_idx = 0; prev_idx < entry_idx; prev_idx++) {
                PoolEntry* prev = &organism->pool->entries[prev_idx];
                int cells = prev->grid_size * prev->grid_size;
                fp32_ws_offset += cells * (prev->num_heads + 1) * prev->head_dim;
                fp16_ws_offset += cells * (prev->channels + prev->head_dim);
            }
            entry_ca_state->fp32_workspace = buffers->fp32_ca_workspace + fp32_ws_offset;
            entry_ca_state->fp16_workspace = buffers->fp16_ca_workspace + fp16_ws_offset;

            entry_ca_state->tape.entries = buffers->ad_tape_entries_pool + entry_idx * TAPE_ENTRIES_PER_ENTRY;
            entry_ca_state->tape.capacity = TAPE_ENTRIES_PER_ENTRY;
            entry_ca_state->tape.current_size = 0;
            entry_ca_state->tape.value_buffer = buffers->ad_tape_values_pool + entry_idx * TAPE_VALUES_PER_ENTRY;
            entry_ca_state->tape.grad_buffer = buffers->ad_tape_grads_pool + entry_idx * TAPE_VALUES_PER_ENTRY;
            entry_ca_state->tape.value_levels = buffers->ad_tape_levels_pool + entry_idx * TAPE_VALUES_PER_ENTRY;
            entry_ca_state->tape.value_capacity = TAPE_VALUES_PER_ENTRY;
            entry_ca_state->tape.current_value_idx = 0;
            entry_ca_state->tape.max_level = 0;
            entry_ca_state->tape.needs_weight_restore = 0;
            entry_ca_state->tape.restore_elite_idx = INT_MAX;

            entry_ca_state->trace.traces = buffers->trace_array + entry_idx * TRACE_CAPACITY;
            entry_ca_state->trace.capacity = TRACE_CAPACITY;
            entry_ca_state->trace.current_idx = 0;

            // Accumulate saved activation offsets based on dimensions of all previous entries
            size_t perception_saved_offset = 0;
            size_t interaction_saved_offset = 0;
            size_t pre_gelu_saved_offset = 0;

            for (int prev_idx = 0; prev_idx < entry_idx; prev_idx++) {
                PoolEntry* prev = &organism->pool->entries[prev_idx];
                int cells = prev->grid_size * prev->grid_size;
                perception_saved_offset += BATCH_SIZE * prev->num_heads * cells * prev->head_dim;
                interaction_saved_offset += BATCH_SIZE * prev->num_heads * cells * prev->head_dim;
                pre_gelu_saved_offset += BATCH_SIZE * prev->num_heads * cells * prev->head_dim;
            }

            entry_ca_state->perception_saved = buffers->perception_activations_saved + perception_saved_offset;
            entry_ca_state->interaction_saved = buffers->interaction_activations_saved + interaction_saved_offset;
            entry_ca_state->pre_gelu_saved = buffers->pre_gelu_values_saved + pre_gelu_saved_offset;

            entry->ca_state = entry_ca_state;
        }

        init_trace_buffer_device(organism, TRACE_CAPACITY);
        cg::this_grid().sync();

        init_organism_ca_weights_device(organism);

        float* all_chem_fields = buffers->all_chem_fields;
        // Multi-channel fields: CA_FIELD_SIZE * CHEM_CHANNELS each
        organism->chemical_field->channels = CHEM_CHANNELS;
        organism->chemical_field->concentration = all_chem_fields + CA_FIELD_SIZE * CHEM_CHANNELS * 0;
        organism->chemical_field->gradient_x = all_chem_fields + CA_FIELD_SIZE * CHEM_CHANNELS * 1;
        organism->chemical_field->gradient_y = all_chem_fields + CA_FIELD_SIZE * CHEM_CHANNELS * 2;
        organism->chemical_field->laplacian = all_chem_fields + CA_FIELD_SIZE * CHEM_CHANNELS * 3;
        organism->chemical_field->sources = all_chem_fields + CA_FIELD_SIZE * CHEM_CHANNELS * 4;
        // Single-channel decay_factors (shared across all channels)
        organism->chemical_field->decay_factors = all_chem_fields + CA_FIELD_SIZE * CHEM_CHANNELS * 5;

        organism->fitness_history = buffers->fitness_history;
        organism->effective_rank_history = buffers->effective_rank_history;
        organism->coherence_history = buffers->coherence_history;

        float* all_rd_fields = buffers->all_rd_fields;
        organism->resource_density = all_rd_fields + CA_FIELD_SIZE * RD_RESOURCE_DENSITY;
        organism->resource_next = all_rd_fields + CA_FIELD_SIZE * RD_RESOURCE_NEXT;
        organism->fitness_landscape = all_rd_fields + CA_FIELD_SIZE * RD_FITNESS_LANDSCAPE;
        organism->resource_gradient_x = all_rd_fields + CA_FIELD_SIZE * RD_RESOURCE_GRADIENT_X;
        organism->resource_gradient_y = all_rd_fields + CA_FIELD_SIZE * RD_RESOURCE_GRADIENT_Y;

        init_resource_fields_device(organism);
        cg::this_grid().sync();
        if (global_tid == 0) printf("V:p2_sync2 grid_size=%d\n", arch.grid_size);
        cg::this_grid().sync();

        // Partition shared workspace into three separate pools based on maximum entry requirements
        size_t coherence_ws_size = 0;
        size_t correlation_ws_size = 0;
        size_t fitness_ws_size = 0;

        for (int i = 0; i < pool_capacity; i++) {
            PoolEntry* entry = &organism->pool->entries[i];
            int cells = entry->grid_size * entry->grid_size;
            size_t entry_coh_size = cells * entry->num_heads * entry->head_dim;
            size_t entry_corr_size = entry->num_heads * entry->num_heads;
            size_t entry_fit_size = cells * entry->channels;
            coherence_ws_size = (entry_coh_size > coherence_ws_size) ? entry_coh_size : coherence_ws_size;
            correlation_ws_size = (entry_corr_size > correlation_ws_size) ? entry_corr_size : correlation_ws_size;
            fitness_ws_size = (entry_fit_size > fitness_ws_size) ? entry_fit_size : fitness_ws_size;
        }

        float* shared_workspace = buffers->shared_workspace;
        organism->coherence_workspace_pool = shared_workspace;
        organism->correlation_matrix_pool = shared_workspace + coherence_ws_size;
        organism->fitness_workspace_pool = shared_workspace + coherence_ws_size + correlation_ws_size;

        organism->lifecycle_states = buffers->lifecycle_states;

        // Find first alive entry for genome DIRESA initialization
        int first_alive_idx = -1;
        for (int i = 0; i < pool_capacity; i++) {
            if (organism->pool->alive_flags[i]) {
                first_alive_idx = i;
                break;
            }
        }
        PoolEntry* first_entry = (first_alive_idx >= 0) ? &organism->pool->entries[first_alive_idx] : nullptr;

        int max_num_classes = 0;
        for (int i = 0; i < NUM_ACTIVE_DATASETS; i++) {
            int nc = dataset_array[i]->descriptor->num_classes;
            if (nc > max_num_classes) {
                max_num_classes = nc;
            }
        }
        int num_classes = max_num_classes;

        organism->telemetry->memory_allocation.total_gpu_allocated = 0;
        organism->telemetry->memory_allocation.archive_pools_size = 0;
        organism->telemetry->memory_allocation.training_pools_size = 0;
        organism->telemetry->memory_allocation.ca_state_size = 0;
        organism->telemetry->memory_allocation.behavioral_pools_size = 0;
        organism->telemetry->memory_allocation.diresa_weights_size = 0;
        organism->telemetry->memory_allocation.autodiff_tape_size = 0;
        organism->telemetry->memory_allocation.device_heap_limit = DEVICE_MALLOC_HEAP_MB * BYTES_PER_MB;
        organism->telemetry->memory_allocation.device_heap_allocated = 0;

        if (global_tid == 0) printf("V:p2_pre_diresa_genome\n");
        cg::this_grid().sync();

        // Genome DIRESA - block 0 handles it
        if (blockIdx.x == 0 && first_entry != nullptr) {
            size_t genome_stride = GENOME_SIZE * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX +
                                   DIRESA_HIDDEN1_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX +
                                   DIRESA_HIDDEN2_MAX * GENOME_LATENT_DIM_MAX + GENOME_LATENT_DIM_MAX +
                                   GENOME_LATENT_DIM_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX +
                                   DIRESA_HIDDEN2_MAX * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX +
                                   DIRESA_HIDDEN1_MAX * GENOME_SIZE + GENOME_SIZE;
            init_diresa_entry_device(
                organism->diresa_genome_weights,
                organism->diresa_genome_weight_pool,
                organism->diresa_genome_grad_pool,
                genome_stride,
                GENOME_SIZE,
                GENOME_LATENT_DIM_MAX,
                first_entry->diresa_hidden1,
                first_entry->diresa_hidden2,
                first_entry->distance_exponent,
                first_entry->quality_weight,
                primary_genome,
                first_entry->gradients,
                seed + 666666
            );
        }
        cg::this_grid().sync();
        if (global_tid == 0) printf("V:p2_post_diresa_genome pool_cap=%d\n", pool_capacity);

        // Per-entry DIRESA - each block handles entries[blockIdx.x]
        int my_entry = blockIdx.x;
        if (my_entry < pool_capacity && organism->pool->alive_flags[my_entry]) {
            PoolEntry* e = &organism->pool->entries[my_entry];

            int entry_task_input_dim = e->num_heads * e->channels;
            e->diresa_task_input_dim = entry_task_input_dim;

            // Set up weight pointers for this entry
            e->diresa_task_weights = &organism->per_entry_diresa_task_weights[my_entry];
            e->diresa_hw_weights = &organism->per_entry_diresa_hw_weights[my_entry];
            e->diresa_gen_weights = &organism->per_entry_diresa_gen_weights[my_entry];

            float* entry_task_pool = organism->per_entry_diresa_task_weight_pool + my_entry * DIRESA_TASK_STRIDE_PER_ENTRY;
            float* entry_hw_pool = organism->per_entry_diresa_hw_weight_pool + my_entry * DIRESA_HW_STRIDE;
            float* entry_gen_pool = organism->per_entry_diresa_gen_weight_pool + my_entry * DIRESA_GEN_STRIDE;
            float* entry_task_grad = organism->per_entry_diresa_task_grad_pool + my_entry * DIRESA_TASK_STRIDE_PER_ENTRY;
            float* entry_hw_grad = organism->per_entry_diresa_hw_grad_pool + my_entry * DIRESA_HW_STRIDE;
            float* entry_gen_grad = organism->per_entry_diresa_gen_grad_pool + my_entry * DIRESA_GEN_STRIDE;

            // Task DIRESA
            init_diresa_entry_device(
                e->diresa_task_weights,
                entry_task_pool,
                entry_task_grad,
                0,
                entry_task_input_dim,
                BEHAVIORAL_DIM_TASK,
                e->diresa_hidden1,
                e->diresa_hidden2,
                e->distance_exponent,
                e->quality_weight,
                primary_genome,
                e->gradients,
                seed + 888888 + my_entry
            );

            // HW DIRESA
            init_diresa_entry_device(
                e->diresa_hw_weights,
                entry_hw_pool,
                entry_hw_grad,
                0,
                HARDWARE_FEATURES_DIM,
                BEHAVIORAL_DIM_HW,
                e->diresa_hidden1,
                e->diresa_hidden2,
                e->distance_exponent,
                e->quality_weight,
                primary_genome,
                e->gradients,
                seed + 999999 + my_entry
            );

            // Gen DIRESA
            init_diresa_entry_device(
                e->diresa_gen_weights,
                entry_gen_pool,
                entry_gen_grad,
                0,
                1,
                BEHAVIORAL_DIM_GEN,
                e->diresa_hidden1,
                e->diresa_hidden2,
                e->distance_exponent,
                e->quality_weight,
                primary_genome,
                e->gradients,
                seed + 777777 + my_entry
            );
        }
        cg::this_grid().sync();

        if (global_tid == 0) printf("V:p2_archive_init archive_size=%d\n", organism->archive_size);

        organism->latent_genome_pool = buffers->latent_genome_pool;

        organism->behavioral_field_pool = buffers->behavioral_field_pool;
        organism->behavioral_gradient_pool = buffers->behavioral_gradient_pool;
        organism->memory_data_pool = buffers->memory_data_pool;

        // Wire memory tubes and sub-buffers
        organism->memory_tubes = buffers->memory_tubes;
        organism->memory_tubes->entries = buffers->memory_tubes_entries;
        organism->memory_update_params = buffers->memory_update_params;

        init_tube_device(organism->memory_tubes, buffers->memory_tubes_data,
                         organism->tube_capacity, BEHAVIORAL_DIM_TOTAL + AGENT_SPATIAL_DIMS,
                         organism->tube_decay_rate);
        cg::this_grid().sync();

        organism->prediction_error_history = buffers->prediction_error_history;
        organism->hardware_geom = buffers->hardware_geom;
        organism->delta_indices_pool = buffers->delta_indices_pool;
        organism->delta_values_pool = buffers->delta_values_pool;
        organism->delta_counts_pool = buffers->delta_counts_pool;
        organism->compact_valid_flags = buffers->compact_valid_flags;
        organism->scan_output = buffers->scan_output;
        organism->scan_block_sums = buffers->scan_block_sums;
        organism->compact_temp_buffer = buffers->compact_temp_buffer;

        organism->fitness_rank_pool = buffers->fitness_rank_pool;
        organism->fitness_coherence_pool = buffers->fitness_coherence_pool;
        organism->fitness_history = buffers->fitness_history;
        organism->coherence_history = buffers->coherence_history;
        organism->effective_rank_history = buffers->effective_rank_history;

        organism->rng_states = buffers->rng_states;
        organism->init_seed = CURAND_DEFAULT_SEED;
        init_rng_states_device(organism);
        cg::this_grid().sync();
        if (global_tid == 0) printf("V:p2_sync3 rng\n");

        organism->param_map = buffers->param_map;
        init_ca_param_map_device(organism);

        organism->current_activation_grid_size = arch.grid_size;  

        organism->lifecycle_phase_counts = buffers->lifecycle_phase_counts;

        organism->reduction_workspace = buffers->reduction_workspace;
        organism->reduction_partial_sums = buffers->reduction_partial_sums;
        organism->correlation_matrix = buffers->correlation_matrix;
        // Spatial cells only - reduce_concentration_mean loops over CHEM_CHANNELS separately
        int total_cells = arch.grid_size * arch.grid_size;
        organism->reduction_total_cells = total_cells;
        organism->reduction_num_blocks = (total_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;

        organism->gradient_features_pool = buffers->gradient_features_pool;
        organism->gradient_logits_pool = buffers->gradient_logits_pool;
        organism->gradient_loss_pool = buffers->gradient_loss_pool;
        organism->gradient_logit_grads_pool = buffers->gradient_logit_grads_pool;
        organism->gradient_magnitudes_pool = buffers->gradient_magnitudes_pool;

        organism->pooling_weights_grad = buffers->pooling_weights_grad;
        organism->fc_weights_grad = buffers->fc_weights_grad;
        organism->fc_bias_grad = buffers->fc_bias_grad;
        organism->features_grad = buffers->features_grad;

        organism->adam_m_ca_pool = buffers->adam_m_ca_pool;
        organism->adam_v_ca_pool = buffers->adam_v_ca_pool;

        organism->adam_m_pooling = buffers->adam_m_pooling;
        organism->adam_v_pooling = buffers->adam_v_pooling;
        organism->adam_m_fc_weights = buffers->adam_m_fc_weights;
        organism->adam_v_fc_weights = buffers->adam_v_fc_weights;
        organism->adam_m_fc_bias = buffers->adam_m_fc_bias;
        organism->adam_v_fc_bias = buffers->adam_v_fc_bias;

        organism->batch_ca_states_pool = buffers->batch_ca_states_pool;
        organism->batch_ca_input_grads = buffers->batch_ca_input_grads;
        organism->batch_labels_pool = buffers->batch_labels_pool;
        organism->task_loss_pool = buffers->task_loss_pool;
        organism->reg_loss_pool = buffers->reg_loss_pool;
        organism->rank_loss_pool = buffers->rank_loss_pool;
        organism->coherence_loss_pool = buffers->coherence_loss_pool;
        organism->diversity_loss_pool = buffers->diversity_loss_pool;
        organism->total_loss_pool = buffers->total_loss_pool;

        organism->training_mode = buffers->training_mode;

        organism->classifier = buffers->classifier;
        organism->classifier_workspace = buffers->classifier_workspace;
        organism->classifier_num_classes = num_classes;
        organism->classifier_seed = seed + 777777;
        cg::this_grid().sync();
        if (global_tid == 0) printf("V:p2_pre_classifier\n");
        init_classifier_device(organism);
        cg::this_grid().sync();
        if (global_tid == 0) printf("V:p2_post_classifier\n");

        organism->training_mode->classifier = organism->classifier;
        organism->training_mode->batch_samples = buffers->batch_samples_pool;
        organism->training_mode->batch_labels = buffers->batch_labels_pool;

        organism->training_mode->adam_m = organism->adam_m_ca_pool;
        organism->training_mode->adam_v = organism->adam_v_ca_pool;

        organism->curriculum = buffers->curriculum;

        organism->voronoi_occupancy_histogram = buffers->voronoi_occupancy_histogram;

        organism->pool_task_accuracies = buffers->pool_task_accuracies;

        organism->dataset_array = dataset_array;
        organism->test_dataset_array = test_dataset_array;

        organism->behavioral_slots.agent_embedding_scale = GenomeParamTable::chemotaxis_agent_embedding_scale;
        organism->behavioral_slots.init_exploration = GenomeParamTable::chemotaxis_init_exploration;
        organism->behavioral_slots.init_sensitivity = GenomeParamTable::chemotaxis_init_sensitivity;
        organism->behavioral_slots.ctx_metabolic = GenomeParamTable::init_context_metabolic;
        organism->behavioral_slots.ctx_stress = GenomeParamTable::init_context_stress;
        organism->behavioral_slots.ctx_morphogen = GenomeParamTable::init_context_morphogen;

        cg::this_grid().sync();
        if (global_tid == 0) printf("V:p2_pre_behavioral\n");
        init_behavioral_state_device(organism);
        cg::this_grid().sync();
        if (global_tid == 0) printf("V:p2_post_behavioral\n");

        int behavioral_dim = BEHAVIORAL_DIM_TOTAL;
        organism->embedding_weights = buffers->behavioral_embedding_weights;
        organism->init_seed = seed + 1;
        if (global_tid == 0) printf("V:p2_pre_embedding\n");
        init_embedding_weights_device(organism);
        cg::this_grid().sync();
        if (global_tid == 0) printf("V:p2_post_embedding\n");

        organism->chem_grid_size = arch.grid_size;

        organism->diffusion_dt = CHEMICAL_DIFFUSION_DT_MAX;
        init_training_mode_device(organism);
        cg::this_grid().sync();

        // Initialize curriculum and set current_dataset before loading samples
        init_curriculum_device(organism);
        cg::this_grid().sync();
        if (global_tid == 0) {
            int dataset_idx = organism->curriculum->current_dataset_idx;
            organism->current_dataset = organism->dataset_array[dataset_idx];
        }
        cg::this_grid().sync();

        // Load initial batch samples before init_chemical_field_device
        // (init_chemical_field_device seeds concentration from samples)
        if (global_tid == 0) printf("V:p2_pre_initial_batch\n");
        load_batch_device(organism);
        cg::this_grid().sync();
        if (global_tid == 0) printf("V:p2_post_initial_batch\n");

        if (global_tid == 0) printf("V:p2_pre_chemfield\n");
        init_chemical_field_device(organism);
        cg::this_grid().sync();
        if (global_tid == 0) printf("V:p2_post_chemfield\n");

        cg::this_grid().sync();
        mark_checkpoint(12);  // pre_chemsources - after grid sync
        set_chemical_sources_from_agents_device(organism);
        mark_checkpoint(13);  // function returned
        __syncthreads();
        mark_checkpoint(14);  // block sync done, about to grid sync
        cg::this_grid().sync();
        mark_checkpoint(15);  // post_chemsources - grid sync completed

        diffusion_reaction_device(organism);
        cg::this_grid().sync();

        organism->snapshot_field_size = field_size;
        store_chemical_snapshot_device(organism);
        cg::this_grid().sync();

        // Bootstrap attractor_field from first history entry
        if (global_tid == 0) {
            int hist_count = organism->chemical_field->history->count;
            DEVICE_FATAL_IF(hist_count <= 0, "init_organism_phase2: chemical history empty after first snapshot");
            int history_idx = (organism->chemical_field->history->head + hist_count - 1)
                  % organism->chemical_field->history->capacity;
            organism->attractor_field = organism->chemical_field->history->entries[history_idx].data;
        }
        cg::this_grid().sync();

        // curriculum and current_dataset already initialized earlier before load_batch_device

        organism->clear_buffer_ptr = organism->fitness_history;
        organism->clear_buffer_size = 2 * POOL_CAPACITY_MAX;
        clear_buffer_device(organism);
        cg::this_grid().sync();

        organism->clear_buffer_ptr = organism->coherence_history;
        clear_buffer_device(organism);
        cg::this_grid().sync();

        organism->clear_buffer_ptr = organism->effective_rank_history;
        organism->clear_buffer_size = 2;
        clear_buffer_device(organism);
        cg::this_grid().sync();
    }
}


__global__ void persistent_evolution_kernel(
    unsigned int seed,
    Dataset** dataset_array,
    Dataset** test_dataset_array,
    Organism* organism,
    AuditBuffer* audit
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    if (global_tid == 0) printf("V:kernel_entry blocks=%d threads=%d\n", gridDim.x, blockDim.x);

    // Parallel genome initialization - each thread handles a slice
    for (int i = global_tid; i < GENOME_SIZE; i += total_threads) {
        PRNGState rng;
        rng.s0 = (seed + i) * XORSHIFT_GOLDEN_RATIO_A;
        rng.s1 = (seed + i) * XORSHIFT_GOLDEN_RATIO_B;
        organism->organism_workspace_genomes[i] = prng_next(&rng) * GENOME_RANGE_SCALE + GENOME_VALUE_MIN;
    }
    cg::this_grid().sync();

    // Parallel scalar init - distribute field writes across threads
    int num_init_fields = 8;
    for (int field = global_tid; field < num_init_fields; field += total_threads) {
        switch (field) {
            case 0: {
                int pool_capacity_slot = GenomeParamTable::pool_capacity;
                float pool_capacity_norm = fmaxf(0.0f, fminf(1.0f, genome_slot_to_unit(organism->organism_workspace_genomes, pool_capacity_slot)));
                int pool_capacity = POOL_CAPACITY_MIN + (int)(pool_capacity_norm * (POOL_CAPACITY_MAX - POOL_CAPACITY_MIN));
                organism->init_pool_capacity = pool_capacity;
            } break;
            case 1: organism->dataset_array = dataset_array; break;
            case 2: organism->test_dataset_array = test_dataset_array; break;
            case 3: organism->workspace_genomes = organism->organism_workspace_genomes; break;
            case 4: organism->buffers = organism; break;
            case 5: organism->init_seed = seed; break;
            case 6: organism->audit_buffer = audit; break;
            case 7: break; // Reserved
        }
    }
    cg::this_grid().sync();

    init_organism_device(organism);
    cg::this_grid().sync();

    init_organism_phase2_device(organism);
    cg::this_grid().sync();

    int capacity = organism->pool->capacity;

    // Parallel spawn config init - all threads compute, distribute writes
    int init_active = Atomics::load_int(organism->pool->active_count);
    float init_spawn_prob = SPAWN_RATE_MAX * expf(-init_active / (float)capacity);

    for (int field = global_tid; field < 4; field += total_threads) {
        switch (field) {
            case 0: organism->spawn_probability = init_spawn_prob; break;
            case 1: organism->spawn_workspace = organism->organism_workspace_genomes + 4 * GENOME_SIZE + 2 * capacity * GENOME_SIZE; break;
            case 2: organism->hunger_threshold = HUNGER_THRESHOLD_MAX; break;
            case 3: organism->diffusion_dt = CHEMICAL_DIFFUSION_DT_MAX; break;
        }
    }
    cg::this_grid().sync();

    // load_batch_device handles generation 0 bootstrap internally
    // (initializes from fields at gen 0, from prev_concentration at gen > 0)
    load_batch_device(organism);
    cg::this_grid().sync();

    // Initialize per-entry CA state from chemical field BEFORE main loop
    // This ensures ca_concentration is valid when hybrid_organism_lifecycle_device
    // reads it on the first tick (generation 0)
    {
        int init_alive_count = organism->pool->alive_indices_count;
        for (int compact = blockIdx.x; compact < init_alive_count; compact += gridDim.x) {
            int entry_idx = organism->pool->alive_indices[compact];
            if (global_tid == 0) {
                organism->current_entry_idx = entry_idx;
            }
            cg::this_grid().sync();
            initialize_ca_from_field_device(organism);
            cg::this_grid().sync();
        }
    }

    unsigned long long tick = 0;

    while (true) {
        // Parallel arch setup - each block handles its own entry
        int alive_count_local = organism->pool->alive_indices_count;
        int my_entry = blockIdx.x % max(1, alive_count_local);
        Architecture arch_local = get_arch_from_pool(organism, my_entry);

        // Block 0 writes shared arch state, others use local
        if (global_tid == 0) {
            organism->current_arch = arch_local;
            organism->snapshot_field_size = arch_local.grid_size * arch_local.grid_size;
        }
        cg::this_grid().sync();

        if (organism->archive_size > 0) {
            sync_voronoi_from_archive_device(organism);
            cg::this_grid().sync();
            update_voronoi_density_device(organism);
            cg::this_grid().sync();
        }

        // Derive organism params from density statistics
        derive_organism_params_device(organism);
        cg::this_grid().sync();

        reset_trace_buffer_device(organism);
        cg::this_grid().sync();

        int alive_count = organism->pool->alive_indices_count;
        int num_waves = (alive_count + WAVE_SIZE - 1) / WAVE_SIZE;
        for (int wave = 0; wave < num_waves; wave++) {
            // Wave setup - use atomics to avoid single-thread bottleneck
            int wave_start = wave * WAVE_SIZE;
            int wave_size = min(WAVE_SIZE, alive_count - wave_start);
            if (global_tid == 0) {
                organism->current_wave_start = wave_start;
                organism->current_wave_size = wave_size;
            }
            cg::this_grid().sync();

            load_batch_device(organism);
            cg::this_grid().sync();

            hybrid_organism_lifecycle_device(organism);
            cg::this_grid().sync();
        }

        aggregate_hardware_geometry_device(organism);
        cg::this_grid().sync();

        reduce_concentration_mean_device(organism);
        cg::this_grid().sync();
        finalize_concentration_mean_device(organism);
        cg::this_grid().sync();

        selection_device(organism);
        cg::this_grid().sync();

        // Voronoi sync moved to start of loop (before training) so telemetry probes have valid cells
        // Selection updates archive here; voronoi will sync from it at start of next iteration

        component_evolution_device(organism);
        cg::this_grid().sync();

        compute_fitness_from_diresa_device(organism);
        cg::this_grid().sync();

        // Spawn probability - all threads compute same value, one writes
        int active = Atomics::load_int(organism->pool->active_count);
        float spawn_prob = SPAWN_RATE_MAX * expf(-active / (float)capacity);
        if (global_tid == 0) {
            organism->spawn_probability = spawn_prob;
        }
        cg::this_grid().sync();

        if (spawn_prob > SPAWN_PROBABILITY_MIN_MIN) {
            spawn_wave_device(organism);
            cg::this_grid().sync();
        }

        if (organism->archive_size > 0 && organism->num_voronoi_cells > 0) {
            archive_driven_lifecycle_device(organism, organism->hunger_threshold);
        }
        cg::this_grid().sync();

        if (organism->generation >= 1) {
            memory_update_device(organism);
            cg::this_grid().sync();
        }

        diffusion_reaction_device(organism);
        cg::this_grid().sync();

        store_chemical_snapshot_device(organism);
        cg::this_grid().sync();

        // Attractor field update - single write but check is parallel
        int hist_count = organism->chemical_field->history->count;
        DEVICE_FATAL_IF(hist_count <= 0, "persistent_evolution: chemical history empty");
        if (global_tid == 0) {
            int history_idx = (organism->chemical_field->history->head + hist_count - 1)
                  % organism->chemical_field->history->capacity;
            organism->attractor_field = organism->chemical_field->history->entries[history_idx].data;
        }
        cg::this_grid().sync();

        // Process alive entries - each block handles different entries in parallel
        int p2_alive_count = organism->pool->alive_indices_count;
        for (int compact = blockIdx.x; compact < p2_alive_count; compact += gridDim.x) {
            // All threads in block read the entry index (no guard needed for reads)
            int entry_idx = organism->pool->alive_indices[compact];
            if (global_tid == 0) {
                organism->current_entry_idx = entry_idx;
            }
            cg::this_grid().sync();
            initialize_ca_from_field_device(organism);
            cg::this_grid().sync();
        }

        for (int compact = blockIdx.x; compact < p2_alive_count; compact += gridDim.x) {
            int entry_idx = organism->pool->alive_indices[compact];
            if (global_tid == 0) {
                organism->current_entry_idx = entry_idx;
            }
            cg::this_grid().sync();
            update_field_from_ca_device(organism);
            cg::this_grid().sync();
        }

        if (organism->archive_size > 0) {
            behavioral_update_device(organism);
        }
        cg::this_grid().sync();

        // Generation increment - atomic ensures correctness, all threads can observe
        if (global_tid == 0) {
            int old_gen = atomicAdd(&organism->generation, 1);
            if (old_gen % 10 == 0) {
                printf("V:gen=%d\n", old_gen);
            }
        }
        cg::this_grid().sync();
        tick++;
    }
}


#endif
