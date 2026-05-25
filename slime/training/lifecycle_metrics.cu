__device__ void lifecycle_metrics_device(Organism* organism) {
    HybridTrainingMode* training_mode = organism->training_mode;
    CAParameterMap* param_map = organism->param_map;
    int generation = organism->generation;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_device_error_generation = generation;
    }
    float* workspace_genomes = organism->workspace_genomes;
    AuditBuffer* audit = organism->audit_buffer;
    int wave_start = organism->current_wave_start;

    extern __shared__ float sdata[];
    int compact_idx = wave_start + blockIdx.x;
    int wave_position = blockIdx.x;
    int tid = threadIdx.x;
    ComponentPool* pool = organism->pool;

    bool has_work = compact_idx < pool->alive_indices_count && blockIdx.x < WAVE_SIZE;

    int entry_idx;
    PoolEntry* entry;
    if (has_work) {
        entry_idx = pool->alive_indices[compact_idx];
        entry = &pool->entries[entry_idx];
    }

    cg::this_grid().sync();

    __shared__ WaveBufferOffsets s_wave_offsets;
    if (has_work && tid == 0) {
        s_wave_offsets = compute_wave_offsets(pool, wave_start, wave_position, training_mode->batch_size);
    }
    cg::this_grid().sync();

    MultiHeadCAState* ca_state;
    if (has_work) {
        ca_state = entry->ca_state;
    }

    __shared__ float* s_primary_genome;
    __shared__ int s_num_classes;
    __shared__ int s_behavioral_dim;
    __shared__ int s_num_features;
    __shared__ Architecture s_arch;
    __shared__ float s_task_accuracy;
    __shared__ dim3 s_component_grid;
    __shared__ dim3 s_component_block;
    __shared__ dim3 s_ca_grid;
    __shared__ dim3 s_ca_block;
    __shared__ dim3 s_field_grid;
    __shared__ dim3 s_field_block;

    float* primary_genome;
    float* primary_parent_temp;
    int num_classes;
    int behavioral_dim;
    int num_features;
    Architecture arch;
    dim3 component_grid;
    dim3 component_block;
    dim3 ca_grid;
    dim3 ca_block;
    dim3 field_grid;
    dim3 field_block;

    if (has_work && tid == 0) {
        primary_genome = &workspace_genomes[entry_idx * GENOME_SIZE * 2];
        primary_parent_temp = &workspace_genomes[entry_idx * GENOME_SIZE * 2 + GENOME_SIZE];

        reconstruct_genome_from_archive(entry->parent_hash, (GPUElite*)organism->archive, organism->archive_size,
            entry->delta_indices, entry->delta_values, entry->num_deltas,
            entry->max_deltas, primary_genome, GENOME_SIZE, primary_parent_temp, organism->diresa_genome_weights);

        num_classes = organism->current_dataset->descriptor->num_classes;

        BehavioralDimensions dims;
        dims.derive_from_genome();
        behavioral_dim = dims.total();

        arch.num_heads = entry->num_heads;
        arch.channels = entry->channels;
        arch.hidden_dim = entry->hidden_dim;
        arch.head_dim = entry->head_dim;
        arch.grid_size = entry->grid_size;
        s_task_accuracy = fminf(fmaxf(entry->task_accuracy.value, 0.0f), 1.0f);

        num_features = arch.num_heads * POOLING_NUM_TILES * arch.channels;

        component_grid = dim3(POOL_CAPACITY_MAX);
        component_block = dim3(WARP_SIZE);
        ca_grid = dim3(arch.grid_size / WMMA_TILE_DIM, arch.num_heads, 1);
        ca_block = dim3(WMMA_TILE_DIM, WMMA_TILE_DIM, 1);
        field_grid = dim3((arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM, (arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM);
        field_block = dim3(WMMA_TILE_DIM, WMMA_TILE_DIM);

        s_primary_genome = primary_genome;
        s_num_classes = num_classes;
        s_behavioral_dim = behavioral_dim;
        s_num_features = num_features;
        s_arch = arch;
        s_component_grid = component_grid;
        s_component_block = component_block;
        s_ca_grid = ca_grid;
        s_ca_block = ca_block;
        s_field_grid = field_grid;
        s_field_block = field_block;
    }
    cg::this_grid().sync();

    primary_genome = s_primary_genome;
    primary_parent_temp = primary_genome + GENOME_SIZE;
    num_classes = s_num_classes;
    behavioral_dim = s_behavioral_dim;
    num_features = s_num_features;
    arch = s_arch;
    component_grid = s_component_grid;
    component_block = s_component_block;
    ca_grid = s_ca_grid;
    ca_block = s_ca_block;
    field_grid = s_field_grid;
    field_block = s_field_block;

    float* ca_output_grad = organism->buffers->ca_output_grad_buffer + s_wave_offsets.ca_output_offset;
    float* wave_prev_conc = organism->buffers->batch_prev_concentration + s_wave_offsets.ca_states_offset;

    int alive_ct = pool->alive_indices_count;
    int wave_end_compact = min(wave_start + (int)gridDim.x, alive_ct);
    bool is_last_wave = (wave_end_compact >= alive_ct);

    grid_barrier(gridDim.x);
    if (tid == 0) atomicAdd(&g_v_post_bwd_barrier_count, 1);

    float* component_workspace_genomes = organism->buffers->component_workspace_genomes_buffer;
    GPUElite* archive = organism->archive;
    int archive_size_val = organism->archive_size;

    // CHECKPOINT 2: verify trace buffer state before aggregation
    if (has_work && tid == 0 && blockIdx.x == 0) {
        TraceBuffer* chk_tb2 = &organism->ca_state_pool[pool->alive_indices[wave_start]].trace;
        printf("TRACE_CHK2 eid=%d cidx=%d cyc0=%llu br0=%llu\n",
            pool->alive_indices[wave_start], chk_tb2->current_idx,
            chk_tb2->traces[0].cycles_elapsed, chk_tb2->traces[0].total_branches);
    }

    // Aggregate per-entry trace buffers into PoolEntry HW counters
    // Wave-scoped: only aggregate entries processed by this wave
    if (has_work && blockIdx.x == 0) {
        for (int compact = wave_start + tid; compact < wave_end_compact; compact += blockDim.x) {
            int eid = pool->alive_indices[compact];
            PoolEntry* ent = &pool->entries[eid];
            TraceBuffer* tb = &organism->ca_state_pool[eid].trace;
            int trace_count = tb->current_idx;
            if (trace_count > 0) {
                ent->cycles_elapsed = 0;
                ent->inst_executed = 0;
                ent->inst_issued = 0;
                ent->tensor_core_cycles = 0;
                ent->divergent_branches = 0;
                ent->total_branches = 0;
                for (int i = 0; i < trace_count; i++) {
                    ExecutionTrace* t = &tb->traces[i];
                    ent->cycles_elapsed += t->cycles_elapsed;
                    ent->inst_executed += t->inst_executed;
                    ent->inst_issued += t->inst_issued;
                    ent->tensor_core_cycles += t->tensor_core_cycles;
                    ent->divergent_branches += t->divergent_branches;
                    ent->total_branches += t->total_branches;
                }
            }
        }
    }
    cg::this_grid().sync();

    // Wave-scoped: compute metrics for entries processed by this wave
    if (has_work && blockIdx.x == 0) {
        for (int compact = wave_start + tid; compact < wave_end_compact; compact += blockDim.x) {
            int eid = pool->alive_indices[compact];
            DEVICE_FATAL_IF(!pool->alive_flags[eid], "hybrid_lifecycle: dead entry in alive_indices (metrics loop)");

            PoolEntry* ent = &pool->entries[eid];
            float* eid_primary_genome = &component_workspace_genomes[eid * 2 * GENOME_SIZE];
            float* eid_parent_temp = &component_workspace_genomes[eid * 2 * GENOME_SIZE + GENOME_SIZE];

            reconstruct_genome_from_archive(ent->parent_hash, archive, archive_size_val,
                ent->delta_indices, ent->delta_values, ent->num_deltas,
                ent->max_deltas, eid_primary_genome, GENOME_SIZE, eid_parent_temp, organism->diresa_genome_weights);

            float gen_gap_val = fabsf(ent->train_accuracy.value - ent->test_accuracy.value);
            measured_value_set_computed(&ent->generalization_gap, gen_gap_val, generation, ent->genome_hash);

            TraceBuffer* tb = &organism->ca_state_pool[eid].trace;
            int trace_count = tb->current_idx;
            if (trace_count > 0) {
                float ipc = (float)ent->inst_executed / (float)ent->cycles_elapsed;
                float tensor_util = (float)ent->tensor_core_cycles / (float)ent->cycles_elapsed;
                float branch_efficiency = (float)(ent->total_branches - ent->divergent_branches) / (float)ent->total_branches;
                float hw_eff_val = ipc * tensor_util * branch_efficiency;
                measured_value_set_computed(&ent->hardware_efficiency, hw_eff_val, generation, ent->genome_hash);
                DEVICE_VALIDATE_HW_COUNTER(ent->cycles_elapsed, 1ULL, 0xFFFFFFFFFFFFULL);
                DEVICE_VALIDATE_HW_COUNTER(ent->inst_executed, 1ULL, 0xFFFFFFFFFFFFULL);
                DEVICE_VALIDATE_HW_COUNTER(ent->tensor_core_cycles, 0ULL, ent->cycles_elapsed);
            }

            DEVICE_VALIDATE_FINITE(ent->task_accuracy.value);
            DEVICE_VALIDATE_FINITE(ent->hardware_efficiency.value);
            DEVICE_VALIDATE_FINITE(ent->generalization_gap.value);
            DEVICE_VALIDATE_PROBABILITY(ent->task_accuracy.value);

            organism->fitness_history[(generation % 2) * POOL_CAPACITY_MAX + eid] = ent->task_accuracy.value;

            if (generation > 0) {
                float prev_acc = organism->fitness_history[((generation - 1) % 2) * POOL_CAPACITY_MAX + eid];
                float coherence_val = ent->task_accuracy.value - prev_acc;
                measured_value_set_computed(&ent->coherence, coherence_val, generation, ent->genome_hash);
                DEVICE_VALIDATE_FINITE(ent->coherence.value);
                device_validate_fitness_components(pool->fitness_values[eid], ent->coherence.value, ent->effective_rank.value, "pool_entry_fitness");
                organism->coherence_history[(generation % 2) * POOL_CAPACITY_MAX + eid] = ent->coherence.value;
            }
        }
    }
    cg::this_grid().sync();

    {
        // Population-level reduction: must sum ALL entries, so only run on last wave
        // (all per-entry metrics have been computed by now across all waves)
        float local_acc = 0.0f, local_gap = 0.0f, local_hw = 0.0f, local_fit = 0.0f;
        if (is_last_wave && has_work && blockIdx.x == 0) {
            for (int compact = tid; compact < alive_ct; compact += blockDim.x) {
                int eid = pool->alive_indices[compact];
                local_acc += pool->entries[eid].task_accuracy.value;
                local_gap += pool->entries[eid].generalization_gap.value;
                local_hw += pool->entries[eid].hardware_efficiency.value;
                local_fit += pool->fitness_values[eid];
            }
        }
        sdata[tid] = local_acc;
        cg::this_grid().sync();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            cg::this_grid().sync();
        }
        float total_acc = sdata[0];

        sdata[tid] = local_gap;
        cg::this_grid().sync();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            cg::this_grid().sync();
        }
        float total_gap = sdata[0];

        sdata[tid] = local_hw;
        cg::this_grid().sync();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            cg::this_grid().sync();
        }
        float total_hw = sdata[0];

        sdata[tid] = local_fit;
        cg::this_grid().sync();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            cg::this_grid().sync();
        }
        float total_fit = sdata[0];

        if (tid == 0 && is_last_wave && has_work && blockIdx.x == 0) {
            organism->telemetry->population_metrics.total_accuracy = total_acc;
            organism->telemetry->population_metrics.total_generalization_gap = total_gap;
            organism->telemetry->population_metrics.total_hardware_efficiency = total_hw;
            organism->telemetry->population_metrics.total_fitness = total_fit;
        }
    }
    cg::this_grid().sync();

    // Wave-scoped: Baldwin learning for entries in this wave
    if (has_work && blockIdx.x == 0 && generation > 0) {
        for (int compact = wave_start + tid; compact < wave_end_compact; compact += blockDim.x) {
            int eid = pool->alive_indices[compact];
            DEVICE_FATAL_IF(!pool->alive_flags[eid], "hybrid_lifecycle: dead entry in alive_indices (baldwin loop)");

            float prev_task_accuracy = organism->fitness_history[((generation - 1) % 2) * POOL_CAPACITY_MAX + eid];
            float current_task_accuracy = pool->entries[eid].task_accuracy.value;
            float learning_success = current_task_accuracy - prev_task_accuracy;

            if (is_meaningful(learning_success, 1.0f)) {
                PoolEntry* ent = &pool->entries[eid];
                float baldwin_sensitivity = ent->baldwin_sensitivity;
                float scale = learning_success * baldwin_sensitivity;
                float* grads = ent->gradients;
                float* eid_primary_genome = &component_workspace_genomes[eid * 2 * GENOME_SIZE];

                for (int g = 0; g < GENOME_SIZE; g++) {
                    float val = grads[g] + scale * eid_primary_genome[g];
                    grads[g] = fmaxf(GENOME_VALUE_MIN, fminf(GENOME_VALUE_MAX, val));
                }
            }
        }
    }
    cg::this_grid().sync();

    cg::this_grid().sync();

    if (training_mode->use_gradients) {
        {
            int weight_count = arch.num_heads * arch.channels * arch.head_dim;
            half* weights_fp16 = ca_state->perception_weights;
            float* weights_fp32 = ca_state->fp32_workspace;

            for (int idx = tid; idx < weight_count; idx += blockDim.x) {
                weights_fp32[idx] = __half2float(weights_fp16[idx]);
            }
        }
    }
    cg::this_grid().sync();

    float* behavioral_workspace_genomes = organism->buffers->behavioral_workspace_genomes_buffer;


    // Behavioral embedding: runs once over all agents, so last wave only
    if (is_last_wave && has_work && blockIdx.x == 0 && generation % EMBEDDING_UPDATE_FREQ == 0) {
        if (tid == 0) {
            *organism->buffers->behavioral_reconstruction_error = 0.0f;
        }
    }
    cg::this_grid().sync();

    if (is_last_wave && has_work && blockIdx.x == 0 && generation % EMBEDDING_UPDATE_FREQ == 0) {
        int hw_dim = BEHAVIORAL_DIM_HW;
        int task_dim = BEHAVIORAL_DIM_TASK;
        int gen_dim = BEHAVIORAL_DIM_GEN;
        int embed_behavioral_dim = hw_dim + task_dim + gen_dim;

        {
            BehavioralState* agents = organism->behavioral_agents;
            float* embedding_weights = organism->buffers->behavioral_embedding_weights;
            float* reconstruction_error = organism->buffers->behavioral_reconstruction_error;
            int num_agents = POOL_CAPACITY_MAX;
            float* features_buffer = organism->buffers->behavioral_features_buffer;

            float ctx_complexity = organism->telemetry->genome_complexity.hash_entropy;
            float ctx_niche = organism->telemetry->archive_topology.novelty_gradient;
            float ctx_learning = organism->telemetry->diresa_evolution.behavioral_drift_rate;
            float ctx_performance = organism->telemetry->task_performance.accuracy;

            int embed_ctx_metabolic_slot = GenomeParamTable::embed_ctx_metabolic;
            int embed_ctx_stress_slot = GenomeParamTable::embed_ctx_stress;
            int embed_ctx_morphogen_slot = GenomeParamTable::embed_ctx_morphogen;
            float embed_ctx_metabolic = genome_slot_to_unit(primary_genome, embed_ctx_metabolic_slot);
            float embed_ctx_stress = genome_slot_to_unit(primary_genome, embed_ctx_stress_slot);
            float embed_ctx_morphogen = genome_slot_to_unit(primary_genome, embed_ctx_morphogen_slot);

            TrainingParams embed_training_params;
            float learning_rate = embed_training_params.get_behavioral_learning_rate(
                primary_genome, entry->gradients,
                embed_ctx_metabolic, embed_ctx_stress, embed_ctx_morphogen,
                ctx_complexity, ctx_niche, ctx_learning, ctx_performance
            );

            float fourier_base_freq = FOURIER_BASE_FREQ;
            int fourier_num_octaves = min(FOURIER_NUM_OCTAVES, embed_behavioral_dim - 4);
            float fourier_spectrum_exponent = FOURIER_SPECTRUM_EXPONENT;

            for (int agent_id = tid; agent_id < num_agents; agent_id += blockDim.x) {
                BehavioralState* agent = &agents[agent_id];
                float* features = &features_buffer[agent_id * embed_behavioral_dim];

                features[0] = sqrtf(agent->velocity[0] * agent->velocity[0] +
                                   agent->velocity[1] * agent->velocity[1]);

                float turn_rate = 0.0f;
                for (int i = 1; i < GRADIENT_HISTORY; i++) {
                    float dx = agent->gradient_memory[i][0] - agent->gradient_memory[i-1][0];
                    float dy = agent->gradient_memory[i][1] - agent->gradient_memory[i-1][1];
                    turn_rate += sqrtf(dx * dx + dy * dy);
                }
                features[1] = turn_rate / GRADIENT_HISTORY;
                features[2] = agent->exploration_noise;
                features[3] = agent->sensitivity;

                for (int k = 0; k < fourier_num_octaves; k++) {
                    float freq = fourier_base_freq * powf(OCTAVE_MULTIPLIER, (float)k);
                    float cos_sum = 0.0f;
                    float sin_sum = 0.0f;

                    for (int i = 0; i < GRADIENT_HISTORY; i++) {
                        cos_sum += agent->gradient_memory[i][0] * cosf(freq * i);
                        sin_sum += agent->gradient_memory[i][1] * sinf(freq * i);
                    }

                    float magnitude = sqrtf(cos_sum * cos_sum + sin_sum * sin_sum) / GRADIENT_HISTORY;
                    if (freq > 0.0f) {
                        float amplitude_weight = powf(freq, -fourier_spectrum_exponent);
                        features[BASE_FEATURES_COUNT + k] = magnitude * amplitude_weight;
                    }
                }

                for (int d = 0; d < embed_behavioral_dim; d++) {
                    float reconstruction = 0.0f;
                    for (int f = 0; f < embed_behavioral_dim; f++) {
                        reconstruction += features[f] * embedding_weights[f * embed_behavioral_dim + d];
                    }

                    float* target_coord;
                    int local_idx;
                    if (d < hw_dim) {
                        target_coord = agent->hw_coords;
                        local_idx = d;
                    } else if (d < hw_dim + task_dim) {
                        target_coord = agent->task_coords;
                        local_idx = d - hw_dim;
                    } else {
                        target_coord = agent->gen_coords;
                        local_idx = d - hw_dim - task_dim;
                    }

                    float error = reconstruction - target_coord[local_idx];
                    target_coord[local_idx] += learning_rate * error;
                    atomicAdd(reconstruction_error, error * error);
                }
            }
        }
    }
    cg::this_grid().sync();

    // Delta deposition: deposit only (ca_output_ch0 - ca_input_ch0) to conserve mass
    if (has_work && organism->buffers->batch_prev_concentration && organism->chemical_field) {
        ChemicalField* field = organism->chemical_field;
        int grid_size_dep = arch.grid_size;
        int cells_dep = grid_size_dep * grid_size_dep;
        int channels_dep = arch.channels;
        int num_heads_dep = arch.num_heads;
        int batch_sz = training_mode->batch_size;
        int head_stride = cells_dep * channels_dep;
        float* field_coords = organism->sample_field_coords + entry_idx * batch_sz * 2;
        float* ca_input = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;

        int field_elements = channels_dep * cells_dep;
        for (int i = tid; i < field_elements; i += blockDim.x) {
            field->concentration[i] *= entry->field_decay_rate;
        }

        for (int s = 0; s < batch_sz; s++) {
            int center_x = (int)field_coords[s * 2];
            int center_y = (int)field_coords[s * 2 + 1];
            int batch_base = s * num_heads_dep * head_stride;
            for (int cell = tid; cell < cells_dep; cell += blockDim.x) {
                int cx = cell % grid_size_dep;
                int cy = cell / grid_size_dep;
                int field_x = (cx + center_x) % grid_size_dep;
                int field_y = (cy + center_y) % grid_size_dep;
                int field_cell = field_y * grid_size_dep + field_x;

                float delta_ch0 = 0.0f;
                for (int h = 0; h < num_heads_dep; h++) {
                    int idx = batch_base + h * head_stride + cell * channels_dep;
                    delta_ch0 += wave_prev_conc[idx] - ca_input[idx];
                }
                atomicAdd(&field->concentration[field_cell], delta_ch0);
            }
        }
    }
    cg::this_grid().sync();

    if (tid == 0 && blockIdx.x == 0) {
        run_telemetry_probes(organism, generation);
    }
}
