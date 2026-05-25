__device__ void lifecycle_forward_device(Organism* organism) {
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
    if (blockIdx.x == 0 && tid == 0) {
        g_block_counter = 0;
        g_blocks_entered = 0;
        g_blocks_grad = 0;
        g_blocks_ca_fwd = 0;
        g_blocks_flow = 0;
        g_blocks_bwd = 0;
        g_blocks_complete = 0;
    }
    ComponentPool* pool = organism->pool;

    bool has_work = compact_idx < pool->alive_indices_count && blockIdx.x < WAVE_SIZE;

    int entry_idx;
    PoolEntry* entry;
    if (has_work) {
        entry_idx = pool->alive_indices[compact_idx];
        entry = &pool->entries[entry_idx];
    }

    if (has_work && tid == 0) {
        organism->lifecycle_entry_idx = entry_idx;
        organism->lifecycle_workspace_genomes = workspace_genomes;
        organism->lifecycle_wave_start = wave_start;
    }
    cg::this_grid().sync();

    __shared__ WaveBufferOffsets s_wave_offsets;
    if (has_work && tid == 0) {
        s_wave_offsets = compute_wave_offsets(pool, wave_start, wave_position, training_mode->batch_size);
    }
    cg::this_grid().sync();

    __shared__ bool s_entry_alive;
    if (has_work && tid == 0) {
        s_entry_alive = pool->alive_flags[entry_idx];
    }
    cg::this_grid().sync();
    if (has_work) {
        DEVICE_FATAL_IF(!s_entry_alive, "hybrid_organism_lifecycle_kernel: dead entry in alive_indices");
        if (tid == 0) atomicAdd(&g_block_counter, 1);
        if (tid == 0) atomicAdd(&g_blocks_entered, 1);
    }

    MultiHeadCAState* ca_state;
    int local_cells;
    float thread_sum = 0.0f;

    if (has_work) {
        ca_state = entry->ca_state;
        local_cells = entry->channels * entry->grid_size * entry->grid_size;
        for (int i = tid; i < local_cells; i += blockDim.x) {
            thread_sum += ca_state->ca_concentration[i];
        }
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float local_ca_mean;
    if (has_work) {
        local_ca_mean = sdata[0] / (float)local_cells;
        if (tid == 0) sdata[0] = local_ca_mean;
    }
    __syncthreads();
    if (has_work) {
        local_ca_mean = sdata[0];
    }

    __shared__ int s_error_flag;
    __shared__ bool s_use_gradients;
    if (tid == 0) {
        s_error_flag = 0;
        s_use_gradients = training_mode->use_gradients;
    }
    cg::this_grid().sync();

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
    cudaError_t err;

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

    if (has_work && tid == 0 && organism->current_activation_grid_size == 0) {
        organism->current_activation_grid_size = arch.grid_size;
    }

    if (has_work && tid == 0 && arch.grid_size != organism->current_activation_grid_size) {
        organism->current_activation_grid_size = arch.grid_size;
    }

    float* ca_output_grad = organism->buffers->ca_output_grad_buffer + s_wave_offsets.ca_output_offset;
    float* wave_prev_conc = organism->buffers->batch_prev_concentration + s_wave_offsets.ca_states_offset;

    if (s_use_gradients) {
        cg::this_grid().sync();

        if (has_work) {
            reset_tape_device(&ca_state->tape, tid);
        }
        cg::this_grid().sync();

        if (has_work) {
            float* ca_input = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;
            float* ca_output = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
            multi_head_ca_tensor_device(entry, training_mode->batch_size, ca_input, ca_output, s_task_accuracy);
        }

        if (has_work && tid == 0) {
            atomicAdd(&g_blocks_ca_fwd, 1);
        }

        cg::this_grid().sync();
        if (has_work) {
            DEVICE_FATAL_IF(s_error_flag, "hybrid_lifecycle: error flag set after CA forward");
        }

        // Persist per-head state: ca_output → prev_concentration for next generation
        if (has_work) {
            int state_size = training_mode->batch_size * arch.num_heads *
                             arch.grid_size * arch.grid_size * arch.channels;
            float* wave_ca_output = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;

            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                wave_prev_conc[idx] = wave_ca_output[idx];
            }
        }
        cg::this_grid().sync();

        if (has_work && tid == 0) {
            atomicAdd(&g_v_flow_done_count, 1);
        }

        grid_barrier(gridDim.x);

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr && has_work) {
            float* ca_out = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
            float* features = organism->gradient_features_pool + entry_idx * training_mode->batch_size * num_features;
            float* pooling_weights = training_mode->classifier[entry_idx].pooling_weights;
            spatial_pooling_device(
                ca_out, pooling_weights, features,
                training_mode->batch_size, arch.grid_size, arch.num_heads, arch.channels,
                tid, blockDim.x);
        }
        cg::this_grid().sync();

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr) {
            if (has_work && tid == 0) {
                BehavioralDimensions dims;
                dims.derive_from_genome();
                int task_dim = dims.task_dim;

                DEVICE_FATAL_IF(entry->diresa_task_weights->input_dim != num_features,
                    "hybrid_lifecycle: diresa_task_weights->input_dim mismatch with num_features (entry %d)", entry_idx);

                // Per-sample DIRESA encode → fractal fold → 2D field coords
                int batch_size_local = training_mode->batch_size;
                float* features_base = organism->gradient_features_pool + entry_idx * batch_size_local * num_features;
                float* field_coords = organism->sample_field_coords + entry_idx * batch_size_local * 2;
                float sample_latent[BEHAVIORAL_DIM_TASK];
                for (int s = 0; s < batch_size_local; s++) {
                    diresa_encode(features_base + s * num_features, sample_latent, entry->diresa_task_weights);
                    fractal_fold_2d(sample_latent, task_dim, arch.grid_size, &field_coords[s * 2], &field_coords[s * 2 + 1]);
                }
            }
        }
        cg::this_grid().sync();

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr && has_work) {
            int batch_size = training_mode->batch_size;
            float* features = organism->gradient_features_pool + entry_idx * batch_size * num_features;
            float* logits = organism->gradient_logits_pool + entry_idx * batch_size * num_classes;
            float* fc_weights = training_mode->classifier[entry_idx].fc_weights;
            float* fc_bias = training_mode->classifier[entry_idx].fc_bias;
            classification_head_device(
                features, fc_weights, fc_bias, logits,
                batch_size, num_features, num_classes,
                tid, blockDim.x);
        }
        cg::this_grid().sync();

        // Per-block loss pointer - uninitialized, only valid when has_work
        float* loss_out;
        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr && has_work) {
            loss_out = &organism->gradient_loss_pool[entry_idx];
            if (tid == 0) {
                *loss_out = 0.0f;
            }
        }
        cg::this_grid().sync();

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr && has_work) {
            int batch_size = training_mode->batch_size;
            float* logits = organism->gradient_logits_pool + entry_idx * batch_size * num_classes;
            float* logit_grads = organism->gradient_logit_grads_pool + entry_idx * batch_size * num_classes;
            int* batch_labels = training_mode->batch_labels;
            cross_entropy_label_smoothing_device(
                logits, batch_labels, logit_grads, loss_out,
                batch_size, num_classes, LABEL_SMOOTHING,
                tid, blockDim.x);
        }
        cg::this_grid().sync();

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr && has_work && tid == 0) {
                int batch_size = training_mode->batch_size;
                float* logits = organism->gradient_logits_pool + entry_idx * batch_size * num_classes;
                int* batch_labels = training_mode->batch_labels;

                int correct = 0;
                float avg_confidence = 0.0f;
                int local_per_class_correct[NUM_CLASSES_MAX];
                int local_per_class_total[NUM_CLASSES_MAX];
                for (int c = 0; c < NUM_CLASSES_MAX; c++) {
                    local_per_class_correct[c] = 0;
                    local_per_class_total[c] = 0;
                }

                int grid_size = entry->grid_size;

                int pool_capacity = organism->pool->capacity;
                int entries_per_side = (int)ceilf(sqrtf((float)pool_capacity));
                int entry_region_size = grid_size / entries_per_side;

                int entry_tile_y = entry_idx / entries_per_side;
                int entry_tile_x = entry_idx % entries_per_side;
                int entry_offset_y = entry_tile_y * entry_region_size;
                int entry_offset_x = entry_tile_x * entry_region_size;

                int tiles_per_side = (int)ceilf(sqrtf((float)batch_size));
                int tile_size = entry_region_size / tiles_per_side;

                for (int b = 0; b < batch_size; b++) {
                    float* batch_logits = &logits[b * num_classes];
                    int true_label = batch_labels[b];

                    int pred_class = 0;
                    float max_logit = batch_logits[0];
                    for (int c = 1; c < num_classes; c++) {
                        if (batch_logits[c] > max_logit) {
                            max_logit = batch_logits[c];
                            pred_class = c;
                        }
                    }

                    bool is_correct = (pred_class == true_label);
                    if (is_correct) correct++;
                    local_per_class_total[true_label]++;
                    if (is_correct) local_per_class_correct[true_label]++;

                    int tile_y = b / tiles_per_side;
                    int tile_x = b % tiles_per_side;
                    float sample_accuracy = is_correct ? 1.0f : 0.0f;

                    for (int dy = 0; dy < tile_size; dy++) {
                        for (int dx = 0; dx < tile_size; dx++) {
                            int y = entry_offset_y + tile_y * tile_size + dy;
                            int x = entry_offset_x + tile_x * tile_size + dx;
                            if (y < grid_size && x < grid_size) {
                                int pos = y * grid_size + x;
                                int cells = grid_size * grid_size;
                                int chem_ch = organism->chemical_field->channels;
                                for (int cc = 0; cc < chem_ch; cc++) {
                                    organism->chemical_field->concentration[cc * cells + pos] = sample_accuracy;
                                }
                            }
                        }
                    }

                    float sum_exp = 0.0f;
                    for (int c = 0; c < num_classes; c++) {
                        sum_exp += expf(batch_logits[c] - max_logit);
                    }
                    float confidence = expf(batch_logits[pred_class] - max_logit) / sum_exp;
                    avg_confidence += confidence;
                }

                float accuracy = (float)correct / batch_size;
                avg_confidence /= batch_size;

                int ema_slot = GenomeParamTable::accuracy_ema_smoothing;
                float ema_smoothing = genome_slot_to_unit(primary_genome, ema_slot);
                ema_smoothing = EMA_SMOOTHING_MIN + ema_smoothing * (EMA_SMOOTHING_MAX - EMA_SMOOTHING_MIN);

                int current_gen = organism->generation;
                if (training_mode->is_train_batch) {
                    float prior = measured_value_is_valid(&entry->train_accuracy) ? entry->train_accuracy.value : accuracy;
                    float smoothed = ema_smoothing * prior + (1.0f - ema_smoothing) * accuracy;
                    measured_value_set_computed(&entry->train_accuracy, smoothed, current_gen, entry->genome_hash);
                } else {
                    measured_value_set_computed(&entry->test_accuracy, accuracy, current_gen, entry->genome_hash);
                }
                measured_value_set_computed(&entry->task_accuracy, accuracy, current_gen, entry->genome_hash);
                measured_value_set_computed(&entry->avg_confidence, avg_confidence, current_gen, entry->genome_hash);
                measured_value_set_computed(&entry->task_loss, *loss_out, current_gen, entry->genome_hash);

                float conf_var = 0.0f;
                for (int b = 0; b < batch_size; b++) {
                    float* bl = &logits[b * num_classes];
                    float ml = bl[0];
                    for (int c = 1; c < num_classes; c++) { if (bl[c] > ml) ml = bl[c]; }
                    float se = 0.0f;
                    int pc = 0;
                    for (int c = 0; c < num_classes; c++) { float ev = expf(bl[c] - ml); se += ev; if (bl[c] > bl[pc]) pc = c; }
                    float conf = expf(bl[pc] - ml) / se;
                    float diff = conf - avg_confidence;
                    conf_var += diff * diff;
                }
                float stability = 1.0f - sqrtf(conf_var / batch_size);
                measured_value_set_computed(&entry->classification_stability, stability, current_gen, entry->genome_hash);

                for (int c = 0; c < NUM_CLASSES_MAX; c++) {
                    entry->per_class_correct[c] = local_per_class_correct[c];
                    entry->per_class_total[c] = local_per_class_total[c];
                }
            }
        }
        cg::this_grid().sync();
}
