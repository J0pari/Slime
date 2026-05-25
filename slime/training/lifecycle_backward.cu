__device__ void lifecycle_backward_device(Organism* organism) {
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
        DEVICE_FATAL_IF(!s_entry_alive, "lifecycle_backward_device: dead entry in alive_indices");
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

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr && has_work) {

            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads_local = arch.num_heads;
            int spatial_size = grid_size * grid_size;

            float* ca_out = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
            float* features = organism->gradient_features_pool + entry_idx * batch_size * num_features;
            float* logit_grads = organism->gradient_logit_grads_pool + entry_idx * batch_size * num_classes;
            float* features_grad = organism->features_grad + entry_idx * batch_size * num_features;
            float* fc_weights = training_mode->classifier[entry_idx].fc_weights;
            constexpr int FC_WEIGHTS_ENTRY_STRIDE = NUM_CLASSES_MAX * CLASSIFIER_FEATURE_DIM;
            constexpr int FC_BIAS_ENTRY_STRIDE = NUM_CLASSES_MAX;
            constexpr int POOLING_ENTRY_STRIDE = CLASSIFIER_FEATURE_DIM;
            float* fc_weights_grad = organism->fc_weights_grad + entry_idx * FC_WEIGHTS_ENTRY_STRIDE;
            float* fc_bias_grad = organism->fc_bias_grad + entry_idx * FC_BIAS_ENTRY_STRIDE;
            float* pooling_weights = training_mode->classifier[entry_idx].pooling_weights;
            float* pooling_weights_grad = organism->pooling_weights_grad + entry_idx * POOLING_ENTRY_STRIDE;

            int fc_weights_size = num_classes * num_features;
            int features_grad_size = batch_size * num_features;
            int ca_grad_size = batch_size * num_heads_local * spatial_size * channels;

            for (int i = tid; i < fc_weights_size; i += blockDim.x) {
                fc_weights_grad[i] = 0.0f;
            }
            for (int i = tid; i < num_classes; i += blockDim.x) {
                fc_bias_grad[i] = 0.0f;
            }
            for (int i = tid; i < features_grad_size; i += blockDim.x) {
                features_grad[i] = 0.0f;
            }
            for (int i = tid; i < num_features; i += blockDim.x) {
                pooling_weights_grad[i] = 0.0f;
            }
            for (int i = tid; i < ca_grad_size; i += blockDim.x) {
                ca_output_grad[i] = 0.0f;
            }
        }
        cg::this_grid().sync();

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr && has_work) {
            int batch_size = training_mode->batch_size;
            float* features = organism->gradient_features_pool + entry_idx * batch_size * num_features;
            float* logit_grads = organism->gradient_logit_grads_pool + entry_idx * batch_size * num_classes;
            float* features_grad = organism->features_grad + entry_idx * batch_size * num_features;
            float* fc_weights = training_mode->classifier[entry_idx].fc_weights;
            constexpr int FC_WEIGHTS_ENTRY_STRIDE = NUM_CLASSES_MAX * CLASSIFIER_FEATURE_DIM;
            constexpr int FC_BIAS_ENTRY_STRIDE = NUM_CLASSES_MAX;
            float* fc_weights_grad = organism->fc_weights_grad + entry_idx * FC_WEIGHTS_ENTRY_STRIDE;
            float* fc_bias_grad = organism->fc_bias_grad + entry_idx * FC_BIAS_ENTRY_STRIDE;

            classification_head_backward_device(
                logit_grads, features, fc_weights,
                fc_weights_grad, fc_bias_grad, features_grad,
                batch_size, num_features, num_classes,
                tid, blockDim.x);
        }
        cg::this_grid().sync();

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr && has_work) {
            int batch_size = training_mode->batch_size;
            float* ca_out = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
            float* features_grad = organism->features_grad + entry_idx * batch_size * num_features;
            float* pooling_weights = training_mode->classifier[entry_idx].pooling_weights;
            constexpr int POOLING_ENTRY_STRIDE = CLASSIFIER_FEATURE_DIM;
            float* pooling_weights_grad = organism->pooling_weights_grad + entry_idx * POOLING_ENTRY_STRIDE;

            spatial_pooling_backward_device(
                ca_out, features_grad, pooling_weights,
                ca_output_grad, pooling_weights_grad,
                batch_size, arch.grid_size, arch.num_heads, arch.channels,
                tid, blockDim.x);
        }
        cg::this_grid().sync();

        // Backward pass - work guarded, syncs outside
        // Blocks without work skip work but hit all grid syncs
        {
            // Reset broadcast bounds
            if (tid == 0 && blockIdx.x == 0) {
                g_bwd_max_num_chunks = 0;
                g_bwd_max_num_cells = 0;
                g_bwd_max_total_samples = 0;
            }
            cg::this_grid().sync();

            // Blocks with work broadcast their bounds via atomicMax
            if (has_work && tid == 0) {
                int num_cells = arch.grid_size * arch.grid_size;
                int total_samples = training_mode->batch_size * num_cells;
                int num_chunks = (total_samples + BACKWARD_CHUNK_SAMPLES - 1) / BACKWARD_CHUNK_SAMPLES;
                atomicMax(&g_bwd_max_num_chunks, num_chunks);
                atomicMax(&g_bwd_max_num_cells, num_cells);
                atomicMax(&g_bwd_max_total_samples, total_samples);
            }
            cg::this_grid().sync();

            // All blocks read broadcast bounds
            int bwd_num_chunks = g_bwd_max_num_chunks;
            int bwd_num_cells = g_bwd_max_num_cells;
            int bwd_total_samples = g_bwd_max_total_samples;

            // Per-block variables - uninitialized, only valid when has_work
            // Blocks without work never read these - garbage is fine
            TraceBuffer* trace_buffer;
            float* dL_dperception;
            float* dL_dinteraction;
            char* backward_ws_base;
            BackwardWorkspaceLayout ws_layout;
            half* ws_fp16_a;
            half* ws_fp16_b;
            float* ws_dW;
            float* ws_dI;
            half* ws_W_T;
            float* ws_im2col;
            float* ws_dpregelu;
            float* perception_saved;
            float* interaction_saved;
            float* pre_gelu_saved;
            int num_cells;
            int total_samples;
            int I_head_stride;
            int I_batch_stride;
            int V_head_stride;
            int V_batch_stride;
            int ws_dW_interaction_stride;
            int ws_W_T_interaction_stride;
            int chunk_ws_a_stride;
            int chunk_ws_b_stride;

            ExecutionTrace* bwd_trace_slot = nullptr;
            unsigned long long bwd_cycle_start = 0;
            if (has_work) {
                trace_buffer = &ca_state->trace;
                {
                    int trace_idx = -1;
                    if (tid == 0 && trace_buffer->traces != nullptr &&
                        trace_buffer->current_idx < trace_buffer->capacity) {
                        trace_idx = atomicAdd(&trace_buffer->current_idx, 1);
                        printf("TRACE_BWD_ALLOC b%d eid=%d idx=%d cidx_now=%d cidx_addr=%p\n",
                            blockIdx.x, entry_idx, trace_idx, trace_buffer->current_idx, &trace_buffer->current_idx);
                    } else if (tid == 0) {
                        printf("TRACE_BWD_SKIP b%d traces=%p cidx=%d cap=%d\n",
                            blockIdx.x, trace_buffer->traces, trace_buffer->current_idx, trace_buffer->capacity);
                    }
                    if (tid < WARP_SIZE) {
                        trace_idx = __shfl_sync(0xFFFFFFFF, trace_idx, 0);
                        if (trace_idx >= 0 && trace_idx < trace_buffer->capacity) {
                            bwd_trace_slot = &trace_buffer->traces[trace_idx];
                            record_warp_metrics(bwd_trace_slot, blockIdx.x);
                            record_memory_access(bwd_trace_slot, (void*)&ca_state->perception_weights[tid], true);
                            record_shared_memory_access(bwd_trace_slot, false, false);
                        }
                    }
                    bwd_cycle_start = clock64();
                }

                num_cells = arch.grid_size * arch.grid_size;
                total_samples = training_mode->batch_size * num_cells;

                DEVICE_FATAL_IF(organism->buffers == nullptr, "BWD: organism->buffers is null");
                DEVICE_FATAL_IF(organism->buffers->dL_dperception_buffer == nullptr, "BWD: dL_dperception_buffer is null");
                DEVICE_FATAL_IF(organism->buffers->dL_dinteraction_buffer == nullptr, "BWD: dL_dinteraction_buffer is null");
                DEVICE_FATAL_IF(s_wave_offsets.activations_offset < 0, "BWD: activations_offset negative");

                dL_dperception = organism->buffers->dL_dperception_buffer + s_wave_offsets.activations_offset;
                dL_dinteraction = organism->buffers->dL_dinteraction_buffer + s_wave_offsets.activations_offset;
                if (tid == 0) atomicAdd(&g_v_bwd_enter_count, 1);

                if (tid == 0) atomicAdd(&g_v_bwd_fatal_checks_count, 1);
                DEVICE_FATAL_IF(organism->buffers->backward_workspace == nullptr, "BWD: backward_workspace is null");
                DEVICE_FATAL_IF(s_wave_offsets.backward_ws_offset < 0, "BWD: backward_ws_offset negative");
                backward_ws_base = organism->buffers->backward_workspace + s_wave_offsets.backward_ws_offset;
                ws_layout = compute_backward_ws_layout(entry);
                DEVICE_FATAL_IF(ws_layout.fp16_a_offset < 0, "BWD: ws_layout.fp16_a_offset negative");
                DEVICE_FATAL_IF(ws_layout.fp16_b_offset < 0, "BWD: ws_layout.fp16_b_offset negative");
                DEVICE_FATAL_IF(ws_layout.dW_offset < 0, "BWD: ws_layout.dW_offset negative");
                DEVICE_FATAL_IF(ws_layout.dI_offset < 0, "BWD: ws_layout.dI_offset negative");
                DEVICE_FATAL_IF(ws_layout.W_T_offset < 0, "BWD: ws_layout.W_T_offset negative");
                DEVICE_FATAL_IF(ws_layout.im2col_offset < 0, "BWD: ws_layout.im2col_offset negative");
                DEVICE_FATAL_IF(ws_layout.dpregelu_offset < 0, "BWD: ws_layout.dpregelu_offset negative");

                ws_fp16_a = (half*)(backward_ws_base + ws_layout.fp16_a_offset);
                ws_fp16_b = (half*)(backward_ws_base + ws_layout.fp16_b_offset);
                ws_dW = (float*)(backward_ws_base + ws_layout.dW_offset);
                ws_dI = (float*)(backward_ws_base + ws_layout.dI_offset);
                ws_W_T = (half*)(backward_ws_base + ws_layout.W_T_offset);
                ws_im2col = (float*)(backward_ws_base + ws_layout.im2col_offset);
                ws_dpregelu = (float*)(backward_ws_base + ws_layout.dpregelu_offset);

                DEVICE_FATAL_IF(ca_state == nullptr, "BWD: ca_state is null");
                DEVICE_FATAL_IF(ca_state->perception_saved == nullptr, "BWD: perception_saved is null");
                DEVICE_FATAL_IF(ca_state->interaction_saved == nullptr, "BWD: interaction_saved is null");
                DEVICE_FATAL_IF(ca_state->pre_gelu_saved == nullptr, "BWD: pre_gelu_saved is null");
                DEVICE_FATAL_IF(ca_state->flow_projection_weights == nullptr, "BWD: flow_projection_weights is null");
                DEVICE_FATAL_IF(ca_state->interaction_weights == nullptr, "BWD: interaction_weights is null");
                DEVICE_FATAL_IF(ca_state->perception_weights == nullptr, "BWD: perception_weights is null");
                DEVICE_FATAL_IF(ca_state->tape.grad_buffer == nullptr, "BWD: tape.grad_buffer is null");
                DEVICE_FATAL_IF(param_map == nullptr, "BWD: param_map is null");
                DEVICE_FATAL_IF(param_map->interaction_start == nullptr, "BWD: param_map->interaction_start is null");
                DEVICE_FATAL_IF(param_map->perception_start == nullptr, "BWD: param_map->perception_start is null");
                DEVICE_FATAL_IF(organism->batch_ca_states_pool == nullptr, "BWD: batch_ca_states_pool is null");
                perception_saved = ca_state->perception_saved;
                interaction_saved = ca_state->interaction_saved;
                pre_gelu_saved = ca_state->pre_gelu_saved;

                I_head_stride = num_cells * arch.head_dim;
                I_batch_stride = arch.num_heads * I_head_stride;
                V_head_stride = num_cells * arch.channels;
                V_batch_stride = arch.num_heads * V_head_stride;
                ws_dW_interaction_stride = arch.head_dim * arch.head_dim;
                ws_W_T_interaction_stride = arch.head_dim * arch.head_dim;

                DEVICE_FATAL_IF(arch.num_heads <= 0, "BWD: arch.num_heads <= 0");
                DEVICE_FATAL_IF(arch.head_dim <= 0, "BWD: arch.head_dim <= 0");
                DEVICE_FATAL_IF(arch.channels <= 0, "BWD: arch.channels <= 0");
                DEVICE_FATAL_IF(total_samples <= 0, "BWD: total_samples <= 0");
                DEVICE_FATAL_IF(ca_output_grad == nullptr, "BWD: ca_output_grad is null");

                chunk_ws_a_stride = BACKWARD_CHUNK_SAMPLES * arch.head_dim;
                chunk_ws_b_stride = BACKWARD_CHUNK_SAMPLES * arch.channels;
            }

            int warp_id = tid / WARP_SIZE;
            int lane_id = tid % WARP_SIZE;
            int num_warps = blockDim.x / WARP_SIZE;

            // Zero d_ca_input before transport backward (d_source accumulates here)
            float* d_ca_input = nullptr;
            if (has_work) {
                DEVICE_FATAL_IF(organism->batch_ca_input_grads == nullptr,
                    "batch_ca_input_grads must be allocated for backward pass");
                d_ca_input = organism->batch_ca_input_grads + s_wave_offsets.ca_states_offset;
                int d_ca_total = training_mode->batch_size * arch.num_heads * num_cells * arch.channels;
                for (int idx = tid; idx < d_ca_total; idx += blockDim.x) {
                    d_ca_input[idx] = 0.0f;
                }
            }
            cg::this_grid().sync();

            {
            // Transport backward: ca_output_grad → d_interaction, d_source, d_flow_projection_weights
            if (has_work) {
                float* ca_input = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;
                half* flow_proj_w = ca_state->flow_projection_weights;
                int batch_size = training_mode->batch_size;
                int channels = arch.channels;
                int head_dim = arch.head_dim;
                int num_heads_bwd = arch.num_heads;
                int max_grad_buffer = ca_state->tape.value_capacity;

                // Zero flow projection grads in tape
                int fp_grad_total = num_heads_bwd * 2 * head_dim;
                for (int idx = tid; idx < fp_grad_total; idx += blockDim.x) {
                    int h = idx / (2 * head_dim);
                    int local = idx % (2 * head_dim);
                    ca_state->tape.grad_buffer[param_map->flow_projection_start[h] + local] = 0.0f;
                }

                // Zero dL_dinteraction
                int total_dI = num_heads_bwd * batch_size * num_cells * head_dim;
                for (int idx = tid; idx < total_dI; idx += blockDim.x) {
                    dL_dinteraction[idx] = 0.0f;
                }

                // Use fp32_workspace as per-thread scratch for d_weights (avoids stack overflow)
                // Layout: fp32_workspace[tid * 2 * HEAD_DIM .. tid * 2 * HEAD_DIM + 2*HEAD_DIM - 1]
                float* fp32_ws = ca_state->fp32_workspace;
                float* d_weights_local = &fp32_ws[tid * 2 * head_dim];
                for (int d = 0; d < 2 * head_dim; d++) {
                    d_weights_local[d] = 0.0f;
                }

                for (int batch_id = 0; batch_id < batch_size; batch_id++) {
                    for (int head = 0; head < num_heads_bwd; head++) {
                        int saved_base = batch_id * I_batch_stride + head * I_head_stride;
                        int flow_weight_offset = head * 2 * head_dim;
                        int batch_cell_offset = (batch_id * num_heads_bwd + head) * num_cells * channels;

                        for (int cell_idx = tid; cell_idx < num_cells; cell_idx += blockDim.x) {
                            const float* inter_ptr = &interaction_saved[saved_base + cell_idx * head_dim];
                            float interaction_sum = 0.0f;
                            for (int d = 0; d < head_dim; d++) {
                                interaction_sum += fabsf(inter_ptr[d]);
                            }

                            float2 flow = FlowLeniaOps::project_to_flow(
                                inter_ptr, head_dim, &flow_proj_w[flow_weight_offset]);
                            float gate_input = interaction_sum / (float)head_dim - compute_ca_gate_center(s_task_accuracy);
                            float gate = activation_sigmoid(gate_input);

                            float d_source[CHANNELS];
                            float d_flow_x, d_flow_y, d_gate_val;
                            FlowLeniaOps::bilinear_transport_backward(
                                ca_input, cell_idx, flow, gate, entry->flow_resource_dt,
                                arch.grid_size, ca_output_grad,
                                d_source, &d_flow_x, &d_flow_y, &d_gate_val,
                                channels, batch_cell_offset);

                            // Flow projection backward: accumulates d_interaction into pre-zeroed dL_dinteraction,
                            // accumulates weight gradients into thread-local d_weights_local (reduced after loop)
                            int out_base = head * I_head_stride + batch_id * I_batch_stride + cell_idx * head_dim;
                            FlowLeniaOps::project_to_flow_backward(
                                d_flow_x, d_flow_y, inter_ptr, head_dim,
                                &flow_proj_w[flow_weight_offset],
                                &dL_dinteraction[out_base], d_weights_local);

                            // Gate backward: d_gate → sigmoid backward → d_interaction_sum
                            float d_sigmoid = d_gate_val * gate * (1.0f - gate);
                            float d_interaction_sum = d_sigmoid / (float)head_dim;
                            for (int d = 0; d < head_dim; d++) {
                                float sign_val = (inter_ptr[d] > 0.0f) ? 1.0f :
                                                 (inter_ptr[d] < 0.0f) ? -1.0f : 0.0f;
                                dL_dinteraction[out_base + d] += d_interaction_sum * sign_val;
                            }

                            // d_source → d_ca_input: each thread writes unique cell_idx, no contention
                            for (int c = 0; c < channels; c++) {
                                d_ca_input[batch_cell_offset + cell_idx * channels + c] = d_source[c];
                            }
                        }

                        // Reduce thread-local flow projection gradients into grad_buffer.
                        // One atomicAdd per thread per element (was one per cell per element).
                        int fp_grad_base = param_map->flow_projection_start[head];
                        for (int d = 0; d < 2 * head_dim; d++) {
                            atomicAdd(&ca_state->tape.grad_buffer[fp_grad_base + d], d_weights_local[d]);
                        }
                    }
                }
            }
            cg::this_grid().sync();

            // V:BWD_INTER_TRANSPOSE - work guarded, sync outside
            if (has_work) {
                int dW_tiles = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int total_tile_elements = dW_tiles * dW_tiles * arch.num_heads * WMMA_TILE_DIM * WMMA_TILE_DIM;
                int max_W_per_head = arch.head_dim * arch.head_dim;
                int max_W_T_per_head = ws_W_T_interaction_stride;

                for (int work_idx = tid; work_idx < total_tile_elements; work_idx += blockDim.x) {
                    int elements_per_tile = WMMA_TILE_DIM * WMMA_TILE_DIM;
                    int tiles_per_head = dW_tiles * dW_tiles;
                    int head_id = work_idx / (tiles_per_head * elements_per_tile);
                    int remainder = work_idx % (tiles_per_head * elements_per_tile);
                    int tile_idx = remainder / elements_per_tile;
                    int elem_idx = remainder % elements_per_tile;
                    int tile_x = tile_idx % dW_tiles;
                    int tile_y = tile_idx / dW_tiles;
                    int local_x = elem_idx % WMMA_TILE_DIM;
                    int local_y = elem_idx / WMMA_TILE_DIM;

                    int bx = tile_x * WMMA_TILE_DIM;
                    int by = tile_y * WMMA_TILE_DIM;
                    int x = bx + local_x;
                    int y = by + local_y;

                    if (y < arch.head_dim && x < arch.head_dim) {
                        int W_src_idx = y * arch.head_dim + x;
                        int W_T_dst_idx = x * arch.head_dim + y;
                        PROVENANCE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "BWD inter transpose: head_id OOB");
                        PROVENANCE_FATAL_IF(W_src_idx < 0 || W_src_idx >= max_W_per_head, "BWD inter transpose: src OOB");
                        PROVENANCE_FATAL_IF(W_T_dst_idx < 0 || W_T_dst_idx >= max_W_T_per_head, "BWD inter transpose: dst OOB");
                        const half* W_head = ca_state->interaction_weights + head_id * arch.head_dim * arch.head_dim;
                        half* W_T_head = ws_W_T + head_id * ws_W_T_interaction_stride;
                        half w_val = W_head[W_src_idx];
                        PROVENANCE_FATAL_IF(!isfinite(__half2float(w_val)), "BWD inter transpose: W NaN/Inf");
                        W_T_head[W_T_dst_idx] = w_val;
                    }
                }
            }
            cg::this_grid().sync();

            // Zero interaction/perception dW - work guarded, sync outside
            if (has_work && tid == 0) atomicAdd(&g_v_bwd_zero_dw_count, 1);
            if (has_work) {
                int total_interaction = arch.num_heads * arch.head_dim * arch.head_dim;
                int total_perception = arch.num_heads * arch.channels * arch.head_dim;
                int total_zero = total_interaction + total_perception;
                PROVENANCE_FATAL_IF(total_zero <= 0, "BWD zero dW: total_zero overflow");
                float* ws_dW_inter = ws_dW;
                for (int idx = tid; idx < total_zero; idx += blockDim.x) {
                    ws_dW_inter[idx] = 0.0f;
                }
            }
            cg::this_grid().sync();
            // Per-block variables for second chunk loop - uninitialized, only valid when has_work
            int chunk_ws_dI_stride;
            int chunk_ws_dpregelu_stride;
            int chunk_ws_pooled_stride;
            float* ws_dW_interaction;
            float* ws_dW_perception;
            half* ws_W_T_interaction;

            if (has_work) {
                chunk_ws_dI_stride = BACKWARD_CHUNK_SAMPLES * arch.head_dim;
                chunk_ws_dpregelu_stride = BACKWARD_CHUNK_SAMPLES * arch.head_dim;
                chunk_ws_pooled_stride = BACKWARD_CHUNK_SAMPLES * arch.channels;
                ws_dW_interaction = ws_dW;
                ws_dW_perception = ws_dW_interaction + arch.num_heads * ws_dW_interaction_stride;
                ws_W_T_interaction = ws_W_T;
            }
            cg::this_grid().sync();

            if (has_work && tid == 0) atomicAdd(&g_v_bwd_setup_done_count, 1);

            // Second chunk loop - all blocks iterate bwd_num_chunks, only blocks with work do actual work
            for (int chunk_idx = 0; chunk_idx < bwd_num_chunks; chunk_idx++) {
                int chunk_start = chunk_idx * BACKWARD_CHUNK_SAMPLES;
                int chunk_samples = has_work ? min(BACKWARD_CHUNK_SAMPLES, total_samples - chunk_start) : 0;
                int chunk_samples_aligned = (chunk_samples / WMMA_TILE_DIM) * WMMA_TILE_DIM;
                bool chunk_has_work = has_work && (chunk_samples_aligned > 0);

                if (chunk_has_work && tid == 0) atomicAdd(&g_v_bwd_chunk_count, 1);
                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_chunk0_count, 1);
                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_chunk2_enter_count, 1);

                // GELU backward: dL_dinteraction (from transport bwd) → ws_dpregelu
                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_di_write_count, 1);
                if (chunk_has_work) {
                    int total_elem = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    int max_out = arch.num_heads * training_mode->batch_size * num_cells * arch.head_dim;
                    int max_saved = training_mode->batch_size * I_batch_stride + arch.num_heads * I_head_stride;
                    int max_dpregelu = arch.num_heads * chunk_ws_dpregelu_stride;
                    for (int idx = tid; idx < total_elem; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int sample_in_chunk = remainder / arch.head_dim;
                        int dim_idx = remainder % arch.head_dim;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_id = global_sample % num_cells;

                        int out_idx = head_id * I_head_stride + batch_id * I_batch_stride + cell_id * arch.head_dim + dim_idx;
                        int saved_idx = batch_id * I_batch_stride + head_id * I_head_stride + cell_id * arch.head_dim + dim_idx;
                        int dpregelu_idx = head_id * chunk_ws_dpregelu_stride + sample_in_chunk * arch.head_dim + dim_idx;

                        PROVENANCE_FATAL_IF(out_idx < 0 || out_idx >= max_out, "BWD dI: out_idx OOB");
                        PROVENANCE_FATAL_IF(saved_idx < 0 || saved_idx >= max_saved, "BWD dI: saved_idx OOB");
                        PROVENANCE_FATAL_IF(dpregelu_idx < 0 || dpregelu_idx >= max_dpregelu, "BWD dI: dpregelu_idx OOB");

                        float dL_dI_val = dL_dinteraction[out_idx];
                        PROVENANCE_FATAL_IF(!isfinite(dL_dI_val), "BWD dI: dL_dinteraction NaN/Inf");
                        float pre_val = pre_gelu_saved[saved_idx];
                        PROVENANCE_FATAL_IF(!isfinite(pre_val), "BWD dI: pre_gelu_saved NaN/Inf");
                        float gelu_bwd = activation_gelu_backward(pre_val, dL_dI_val) * INTERACTION_OUTPUT_SCALE;
                        PROVENANCE_FATAL_IF(!isfinite(gelu_bwd), "BWD dI: gelu_bwd NaN/Inf");
                        ws_dpregelu[dpregelu_idx] = gelu_bwd;
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_perc_load_count, 1);
                if (chunk_has_work) {
                    int total_P = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    int max_src_P = arch.num_heads * training_mode->batch_size * num_cells * arch.head_dim;
                    int max_dst_P = arch.num_heads * chunk_ws_a_stride;
                    for (int idx = tid; idx < total_P; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int sample_in_chunk = remainder / arch.head_dim;
                        int dim_idx = remainder % arch.head_dim;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_id = global_sample % num_cells;
                        int src_idx = head_id * I_head_stride + batch_id * I_batch_stride + cell_id * arch.head_dim + dim_idx;
                        int dst_idx = head_id * chunk_ws_a_stride + sample_in_chunk * arch.head_dim + dim_idx;
                        PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_src_P, "BWD perc: src_idx OOB");
                        PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_dst_P, "BWD perc: dst_idx OOB");
                        float val = perception_saved[src_idx];
                        PROVENANCE_FATAL_IF(!isfinite(val), "BWD perc: perception_saved NaN/Inf");
                        ws_fp16_a[dst_idx] = __float2half(val);
                    }
                }
                if (chunk_has_work) {
                    int total_D = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    int max_dpregelu = arch.num_heads * chunk_ws_dpregelu_stride;
                    int max_dst_D = arch.num_heads * chunk_ws_a_stride;
                    for (int idx = tid; idx < total_D; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int src_idx = head_id * chunk_ws_dpregelu_stride + remainder;
                        int dst_idx = head_id * chunk_ws_a_stride + remainder;
                        PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_dpregelu, "BWD D: src_idx OOB");
                        PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_dst_D, "BWD D: dst_idx OOB");
                        float val = ws_dpregelu[src_idx];
                        PROVENANCE_FATAL_IF(!isfinite(val), "BWD D: ws_dpregelu NaN/Inf");
                        ws_fp16_b[dst_idx] = __float2half(val);
                    }
                }
                cg::this_grid().sync();


                if (chunk_has_work) {
                    int dW_tiles = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int total_tiles = dW_tiles * dW_tiles * arch.num_heads;
                    int max_ws_fp16_a_inter = arch.num_heads * chunk_ws_a_stride;
                    int max_ws_dW_inter = arch.num_heads * ws_dW_interaction_stride;

                    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                        int head_id = tile_idx / (dW_tiles * dW_tiles);
                        int tile_flat = tile_idx % (dW_tiles * dW_tiles);
                        int warpM = tile_flat / dW_tiles;
                        int warpN = tile_flat % dW_tiles;
                        int tile_row = warpM * WMMA_TILE_DIM;
                        int tile_col = warpN * WMMA_TILE_DIM;

                        DEVICE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "WMMA inter dW: head_id OOB");

                        if (tile_row < arch.head_dim && tile_col < arch.head_dim) {
                            int A_base_offset = head_id * chunk_ws_a_stride;
                            int B_base_offset = head_id * chunk_ws_a_stride;
                            int C_base_offset = head_id * ws_dW_interaction_stride;
                            DEVICE_FATAL_IF(A_base_offset < 0 || A_base_offset >= max_ws_fp16_a_inter, "WMMA inter dW: A_base OOB");
                            DEVICE_FATAL_IF(B_base_offset < 0 || B_base_offset >= max_ws_fp16_a_inter, "WMMA inter dW: B_base OOB");
                            DEVICE_FATAL_IF(C_base_offset < 0 || C_base_offset >= max_ws_dW_inter, "WMMA inter dW: C_base OOB");

                            const half* A_head = ws_fp16_a + A_base_offset;
                            const half* B_head = ws_fp16_b + B_base_offset;
                            float* C_head = ws_dW_interaction + C_base_offset;

                            int C_tile_offset = tile_row * arch.head_dim + tile_col;
                            int C_tile_max = (tile_row + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                            DEVICE_FATAL_IF(C_tile_offset < 0 || C_tile_max >= ws_dW_interaction_stride, "WMMA inter dW: C_tile OOB");

                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> a_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                            nvcuda::wmma::load_matrix_sync(c_frag, C_head + C_tile_offset, arch.head_dim, nvcuda::wmma::mem_row_major);

                            for (int k_tile = 0; k_tile < chunk_samples_aligned; k_tile += WMMA_TILE_DIM) {
                                int A_tile_offset = k_tile * arch.head_dim + tile_row;
                                int A_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_row + WMMA_TILE_DIM - 1);
                                int B_tile_offset = k_tile * arch.head_dim + tile_col;
                                int B_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                                DEVICE_FATAL_IF(A_tile_offset < 0 || A_tile_max >= chunk_ws_a_stride, "WMMA inter dW: A_tile OOB");
                                DEVICE_FATAL_IF(B_tile_offset < 0 || B_tile_max >= chunk_ws_a_stride, "WMMA inter dW: B_tile OOB");
                                nvcuda::wmma::load_matrix_sync(a_frag, A_head + A_tile_offset, arch.head_dim);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + B_tile_offset, arch.head_dim);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                            nvcuda::wmma::store_matrix_sync(C_head + C_tile_offset, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                        }
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_inter_grad_count, 1);

                if (chunk_has_work) {
                    int tiles_M = (chunk_samples_aligned + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int tiles_N = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int total_tiles = tiles_M * tiles_N * arch.num_heads;
                    int max_ws_fp16_b_dP = arch.num_heads * chunk_ws_a_stride;
                    int max_ws_W_T_inter = arch.num_heads * ws_W_T_interaction_stride;
                    int max_ws_dI_dP = arch.num_heads * chunk_ws_dI_stride;

                    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                        int head_id = tile_idx / (tiles_M * tiles_N);
                        int tile_flat = tile_idx % (tiles_M * tiles_N);
                        int warpM = tile_flat / tiles_N;
                        int warpN = tile_flat % tiles_N;
                        int tile_row = warpM * WMMA_TILE_DIM;
                        int tile_col = warpN * WMMA_TILE_DIM;

                        DEVICE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "WMMA dP: head_id OOB");

                        if (tile_row < chunk_samples_aligned && tile_col < arch.head_dim) {
                            int A_base_offset = head_id * chunk_ws_a_stride;
                            int B_base_offset = head_id * ws_W_T_interaction_stride;
                            int C_base_offset = head_id * chunk_ws_dI_stride;
                            DEVICE_FATAL_IF(A_base_offset < 0 || A_base_offset >= max_ws_fp16_b_dP, "WMMA dP: A_base OOB");
                            DEVICE_FATAL_IF(B_base_offset < 0 || B_base_offset >= max_ws_W_T_inter, "WMMA dP: B_base OOB");
                            DEVICE_FATAL_IF(C_base_offset < 0 || C_base_offset >= max_ws_dI_dP, "WMMA dP: C_base OOB");

                            const half* A_head = ws_fp16_b + A_base_offset;
                            const half* B_head = ws_W_T_interaction + B_base_offset;
                            float* C_head = ws_dI + C_base_offset;

                            int C_tile_offset = tile_row * arch.head_dim + tile_col;
                            int C_tile_max = (tile_row + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                            DEVICE_FATAL_IF(C_tile_offset < 0 || C_tile_max >= chunk_ws_dI_stride, "WMMA dP: C_tile OOB");

                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> a_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                            nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                            for (int k_tile = 0; k_tile < arch.head_dim; k_tile += WMMA_TILE_DIM) {
                                if (k_tile + WMMA_TILE_DIM <= arch.head_dim) {
                                    int A_tile_offset = tile_row * arch.head_dim + k_tile;
                                    int A_tile_max = (tile_row + WMMA_TILE_DIM - 1) * arch.head_dim + (k_tile + WMMA_TILE_DIM - 1);
                                    int B_tile_offset = k_tile * arch.head_dim + tile_col;
                                    int B_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                                    DEVICE_FATAL_IF(A_tile_offset < 0 || A_tile_max >= chunk_ws_a_stride, "WMMA dP: A_tile OOB");
                                    DEVICE_FATAL_IF(B_tile_offset < 0 || B_tile_max >= ws_W_T_interaction_stride, "WMMA dP: B_tile OOB");
                                    nvcuda::wmma::load_matrix_sync(a_frag, A_head + A_tile_offset, arch.head_dim);
                                    nvcuda::wmma::load_matrix_sync(b_frag, B_head + B_tile_offset, arch.head_dim);
                                    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                                }
                            }
                            nvcuda::wmma::store_matrix_sync(C_head + C_tile_offset, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                        }
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_value_grad_count, 1);
                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_dp_write_count, 1);
                if (chunk_has_work) {
                    int total_elem = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    int max_ws_dI = arch.num_heads * chunk_ws_dI_stride;
                    int max_out = arch.num_heads * training_mode->batch_size * num_cells * arch.head_dim;
                    int max_dpregelu = arch.num_heads * chunk_ws_dpregelu_stride;
                    for (int idx = tid; idx < total_elem; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int sample_in_chunk = remainder / arch.head_dim;
                        int dim_idx = remainder % arch.head_dim;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_id = global_sample % num_cells;

                        int ws_idx = head_id * chunk_ws_dI_stride + sample_in_chunk * arch.head_dim + dim_idx;
                        int out_idx = head_id * I_head_stride + batch_id * I_batch_stride + cell_id * arch.head_dim + dim_idx;
                        int dpregelu_idx = head_id * chunk_ws_dpregelu_stride + sample_in_chunk * arch.head_dim + dim_idx;

                        PROVENANCE_FATAL_IF(ws_idx < 0 || ws_idx >= max_ws_dI, "BWD dP: ws_idx OOB");
                        PROVENANCE_FATAL_IF(out_idx < 0 || out_idx >= max_out, "BWD dP: out_idx OOB");
                        PROVENANCE_FATAL_IF(dpregelu_idx < 0 || dpregelu_idx >= max_dpregelu, "BWD dP: dpregelu_idx OOB");

                        float dL_dP_val = ws_dI[ws_idx];
                        PROVENANCE_FATAL_IF(!isfinite(dL_dP_val), "BWD dP: ws_dI NaN/Inf");
                        dL_dperception[out_idx] = dL_dP_val;
                        float perc_val = perception_saved[out_idx];
                        PROVENANCE_FATAL_IF(!isfinite(perc_val), "BWD dP: perception_saved NaN/Inf");
                        float relu_grad = dL_dP_val * PERCEPTION_OUTPUT_SCALE * ((perc_val > 0.0f) ? 1.0f : 0.0f);
                        PROVENANCE_FATAL_IF(!isfinite(relu_grad), "BWD dP: relu_grad NaN/Inf");
                        ws_dpregelu[dpregelu_idx] = relu_grad;
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_im2col_count, 1);
                if (chunk_has_work) {
                    int im2col_sample_stride = chunk_samples_aligned * arch.channels;
                    int total_im2col_work = arch.num_heads * chunk_samples_aligned;
                    int max_input = training_mode->batch_size * arch.num_heads * num_cells * arch.channels;
                    int max_im2col = arch.num_heads * im2col_sample_stride;
                    const float* input_batch = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;

                    for (int idx = tid; idx < total_im2col_work; idx += blockDim.x) {
                        int head_id = idx / chunk_samples_aligned;
                        int sample_in_chunk = idx % chunk_samples_aligned;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_idx = global_sample % num_cells;
                        int cell_y = cell_idx / arch.grid_size;
                        int cell_x = cell_idx % arch.grid_size;

                        int head_input_offset = (batch_id * arch.num_heads + head_id) * num_cells * arch.channels;

                        for (int c = 0; c < arch.channels; c++) {
                            float sum = 0.0f;
                            for (int dy = -1; dy <= 1; dy++) {
                                for (int dx = -1; dx <= 1; dx++) {
                                    int ny = max(0, min(arch.grid_size - 1, cell_y + dy));
                                    int nx = max(0, min(arch.grid_size - 1, cell_x + dx));
                                    int input_idx = head_input_offset + ny * arch.grid_size * arch.channels + nx * arch.channels + c;
                                    PROVENANCE_FATAL_IF(input_idx < 0 || input_idx >= max_input, "BWD im2col: input_idx OOB");
                                    sum += input_batch[input_idx];
                                }
                            }
                            int im2col_idx = head_id * im2col_sample_stride + sample_in_chunk * arch.channels + c;
                            PROVENANCE_FATAL_IF(im2col_idx < 0 || im2col_idx >= max_im2col, "BWD im2col: im2col_idx OOB");
                            ws_im2col[im2col_idx] = sum * GATHER_NORMALIZATION;
                        }
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_conv_fp16_count, 1);
                if (chunk_has_work) {
                    int total_conv = arch.num_heads * chunk_samples_aligned * arch.channels;
                    for (int idx = tid; idx < total_conv; idx += blockDim.x) {
                        float val = ws_im2col[idx];
                        PROVENANCE_FATAL_IF(!isfinite(val), "BWD conv fp16: ws_im2col NaN/Inf");
                        ws_fp16_a[idx] = __float2half(val);
                    }
                }
                if (chunk_has_work) {
                    int total_D = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    int max_dpregelu = arch.num_heads * chunk_ws_dpregelu_stride;
                    int max_dst_D = arch.num_heads * chunk_ws_a_stride;
                    for (int idx = tid; idx < total_D; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int src_idx = head_id * chunk_ws_dpregelu_stride + remainder;
                        int dst_idx = head_id * chunk_ws_a_stride + remainder;
                        PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_dpregelu, "BWD D fp16: src_idx OOB");
                        PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_dst_D, "BWD D fp16: dst_idx OOB");
                        float val = ws_dpregelu[src_idx];
                        ws_fp16_b[dst_idx] = __float2half(val);
                    }
                }
                cg::this_grid().sync();


                if (chunk_has_work) {
                    int ws_dW_perception_stride = arch.channels * arch.head_dim;
                    int dW_tiles_c = (arch.channels + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int dW_tiles_h = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int total_tiles = dW_tiles_c * dW_tiles_h * arch.num_heads;
                    int im2col_head_stride = chunk_samples_aligned * arch.channels;
                    int max_ws_fp16_a_perc = arch.num_heads * im2col_head_stride;
                    int chunk_B_stride = chunk_samples_aligned * arch.head_dim;
                    int max_ws_fp16_b_perc = arch.num_heads * chunk_B_stride;
                    int max_ws_dW_perc = arch.num_heads * ws_dW_perception_stride;

                    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                        int head_id = tile_idx / (dW_tiles_c * dW_tiles_h);
                        int tile_flat = tile_idx % (dW_tiles_c * dW_tiles_h);
                        int warpM = tile_flat / dW_tiles_h;
                        int warpN = tile_flat % dW_tiles_h;
                        int tile_row = warpM * WMMA_TILE_DIM;
                        int tile_col = warpN * WMMA_TILE_DIM;

                        DEVICE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "WMMA perc dW: head_id OOB");

                        if (tile_row < arch.channels && tile_col < arch.head_dim) {
                            int A_base_offset = head_id * im2col_head_stride;
                            int B_base_offset = head_id * chunk_B_stride;
                            int C_base_offset = head_id * ws_dW_perception_stride;
                            DEVICE_FATAL_IF(A_base_offset < 0 || A_base_offset >= max_ws_fp16_a_perc, "WMMA perc dW: A_base OOB");
                            DEVICE_FATAL_IF(B_base_offset < 0 || B_base_offset >= max_ws_fp16_b_perc, "WMMA perc dW: B_base OOB");
                            DEVICE_FATAL_IF(C_base_offset < 0 || C_base_offset >= max_ws_dW_perc, "WMMA perc dW: C_base OOB");

                            const half* A_ptr = ws_fp16_a + A_base_offset;
                            const half* B_head = ws_fp16_b + B_base_offset;
                            float* C_head = ws_dW_perception + C_base_offset;

                            int C_tile_offset = tile_row * arch.head_dim + tile_col;
                            int C_tile_max = (tile_row + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                            DEVICE_FATAL_IF(C_tile_offset < 0 || C_tile_max >= ws_dW_perception_stride, "WMMA perc dW: C_tile OOB");

                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> a_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                            nvcuda::wmma::load_matrix_sync(c_frag, C_head + C_tile_offset, arch.head_dim, nvcuda::wmma::mem_row_major);

                            for (int k_tile = 0; k_tile < chunk_samples_aligned; k_tile += WMMA_TILE_DIM) {
                                int A_tile_offset = k_tile * arch.channels + tile_row;
                                int A_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.channels + (tile_row + WMMA_TILE_DIM - 1);
                                int B_tile_offset = k_tile * arch.head_dim + tile_col;
                                int B_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                                DEVICE_FATAL_IF(A_tile_offset < 0 || A_tile_max >= im2col_head_stride, "WMMA perc dW: A_tile OOB");
                                DEVICE_FATAL_IF(B_tile_offset < 0 || B_tile_max >= chunk_B_stride, "WMMA perc dW: B_tile OOB");
                                nvcuda::wmma::load_matrix_sync(a_frag, A_ptr + A_tile_offset, arch.channels);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + B_tile_offset, arch.head_dim);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                            nvcuda::wmma::store_matrix_sync(C_head + C_tile_offset, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                        }
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_perc_grad_count, 1);
                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_input_grad_count, 1);
                if (chunk_has_work) {
                    int weights_per_head = arch.channels * arch.head_dim;
                    int max_dpregelu = arch.num_heads * chunk_ws_dpregelu_stride;
                    int im2col_head_stride_ig = chunk_samples_aligned * arch.channels;
                    int max_im2col = arch.num_heads * im2col_head_stride_ig;
                    int total_input_grad = arch.num_heads * chunk_samples_aligned * arch.channels;
                    for (int idx = tid; idx < total_input_grad; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.channels);
                        int remainder = idx % (chunk_samples_aligned * arch.channels);
                        int sample_in_chunk = remainder / arch.channels;
                        int channel_idx = remainder % arch.channels;

                        float d_input_accum = 0.0f;
                        int head_weight_start = head_id * weights_per_head;
                        for (int hd = 0; hd < arch.head_dim; hd++) {
                            int W_idx = head_weight_start + channel_idx * arch.head_dim + hd;
                            int dprerelu_idx = head_id * chunk_ws_dpregelu_stride + sample_in_chunk * arch.head_dim + hd;
                            PROVENANCE_FATAL_IF(W_idx < 0 || W_idx >= arch.num_heads * weights_per_head, "BWD input: W_idx OOB");
                            PROVENANCE_FATAL_IF(dprerelu_idx < 0 || dprerelu_idx >= max_dpregelu, "BWD input: dprerelu_idx OOB");
                            d_input_accum += __half2float(ca_state->perception_weights[W_idx]) * ws_dpregelu[dprerelu_idx];
                        }
                        int im2col_idx = head_id * im2col_head_stride_ig + sample_in_chunk * arch.channels + channel_idx;
                        PROVENANCE_FATAL_IF(im2col_idx < 0 || im2col_idx >= max_im2col, "BWD input: im2col idx OOB");
                        ws_im2col[im2col_idx] = d_input_accum;
                    }
                }
                cg::this_grid().sync();
                if (has_work && tid == 0 && blockIdx.x == 0 && chunk_idx == 0) {
                    TraceBuffer* mid_tb2 = &organism->ca_state_pool[pool->alive_indices[wave_start]].trace;
                    printf("TRACE_MID_B cidx=%d cyc0=%llu\n", mid_tb2->current_idx, mid_tb2->traces[0].cycles_elapsed);
                }

                if (chunk_has_work) {
                    if (tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_scatter_count, 1);
                    int im2col_head_stride_sc = chunk_samples_aligned * arch.channels;
                    int scatter_loop_bound = arch.num_heads * chunk_samples_aligned * arch.channels;
                    int max_d_ca = training_mode->batch_size * arch.num_heads * num_cells * arch.channels;
                    for (int idx = tid; idx < scatter_loop_bound; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.channels);
                        int remainder = idx % (chunk_samples_aligned * arch.channels);
                        int sample_in_chunk = remainder / arch.channels;
                        int channel_idx = remainder % arch.channels;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_idx = global_sample % num_cells;
                        int cell_y = cell_idx / arch.grid_size;
                        int cell_x = cell_idx % arch.grid_size;
                        int im2col_idx = head_id * im2col_head_stride_sc + sample_in_chunk * arch.channels + channel_idx;
                        float d_pooled_val = ws_im2col[im2col_idx];

                        float d_normalized = d_pooled_val * GATHER_NORMALIZATION;
                        int head_d_ca_base = (batch_id * arch.num_heads + head_id) * num_cells * arch.channels;
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                int ny = cell_y + dy;
                                int nx = cell_x + dx;
                                if (ny >= 0 && ny < arch.grid_size && nx >= 0 && nx < arch.grid_size) {
                                    int out_cell_idx = ny * arch.grid_size + nx;
                                    int out_idx = head_d_ca_base + out_cell_idx * arch.channels + channel_idx;
                                    PROVENANCE_FATAL_IF(out_idx < 0 || out_idx >= max_d_ca, "BWD scatter: out_idx OOB");
                                    atomicAdd(&d_ca_input[out_idx], d_normalized);
                                }
                            }
                        }
                    }
                }
                cg::this_grid().sync();
            }

            if (has_work && tid == 0) atomicAdd(&g_v_bwd_chunks_done_count, 1);

            // Post-loop grad copy - work guarded, sync outside
            if (has_work && tid == 0) atomicAdd(&g_v_bwd_inter_grad_copy_count, 1);
            if (has_work) {
                int total_grads = arch.num_heads * arch.head_dim * arch.head_dim;
                int max_src = arch.num_heads * ws_dW_interaction_stride;
                int max_grad_buffer = ca_state->tape.capacity;
                PROVENANCE_FATAL_IF(total_grads <= 0, "BWD inter grad: total overflow");
                for (int idx = tid; idx < total_grads; idx += blockDim.x) {
                    int head_id = idx / (arch.head_dim * arch.head_dim);
                    int local_idx = idx % (arch.head_dim * arch.head_dim);
                    int src_idx = head_id * ws_dW_interaction_stride + local_idx;
                    int dst_idx = param_map->interaction_start[head_id] + local_idx;
                    PROVENANCE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "BWD inter grad: head_id OOB");
                    PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_src, "BWD inter grad: src_idx OOB");
                    PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_grad_buffer, "BWD inter grad: dst_idx OOB");
                    float val = ws_dW_interaction[src_idx];
                    PROVENANCE_FATAL_IF(!isfinite(val), "BWD inter grad: src NaN/Inf");
                    ca_state->tape.grad_buffer[dst_idx] = val;
                }
            }
            cg::this_grid().sync();

            if (has_work && tid == 0) atomicAdd(&g_v_bwd_i_done_count, 1);
            if (has_work && tid == 0) atomicAdd(&g_v_bwd_perc_grad_copy_count, 1);
            if (has_work) {
                int ws_dW_perception_stride = arch.channels * arch.head_dim;
                int weights_per_head = arch.channels * arch.head_dim;
                int total_grads = arch.num_heads * weights_per_head;
                int max_src = arch.num_heads * ws_dW_perception_stride;
                int max_grad_buffer = ca_state->tape.capacity;
                PROVENANCE_FATAL_IF(total_grads <= 0, "BWD perc grad: total overflow");
                for (int idx = tid; idx < total_grads; idx += blockDim.x) {
                    int head_id = idx / weights_per_head;
                    int local_idx = idx % weights_per_head;
                    int src_idx = head_id * ws_dW_perception_stride + local_idx;
                    int dst_idx = param_map->perception_start[head_id] + local_idx;
                    PROVENANCE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "BWD perc grad: head_id OOB");
                    PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_src, "BWD perc grad: src_idx OOB");
                    PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_grad_buffer, "BWD perc grad: dst_idx OOB");
                    float val = ws_dW_perception[src_idx];
                    PROVENANCE_FATAL_IF(!isfinite(val), "BWD perc grad: src NaN/Inf");
                    ca_state->tape.grad_buffer[dst_idx] = val;
                }
            }
            cg::this_grid().sync();

            if (has_work && tid == 0) atomicAdd(&g_v_bwd_v_done_count, 1);
            cg::this_grid().sync();

            if (has_work && tid == 0) atomicAdd(&g_v_bwd_grad_conc_count, 1);
            if (has_work && d_ca_input != nullptr) {
                DEVICE_FATAL_IF(organism->buffers->grad_concentration_buffer == nullptr, "BWD: grad_concentration_buffer is null");
                float* grad_conc = organism->buffers->grad_concentration_buffer;
                int max_d_ca = total_samples * arch.channels;
                for (int cell = tid; cell < num_cells; cell += blockDim.x) {
                    int src_idx = cell * arch.channels;
                    PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_d_ca, "BWD grad_conc: src_idx OOB");
                    PROVENANCE_FATAL_IF(cell < 0 || cell >= num_cells, "BWD grad_conc: cell OOB");
                    float val = d_ca_input[src_idx];
                    PROVENANCE_FATAL_IF(!isfinite(val), "BWD grad_conc: d_ca_input NaN/Inf");
                    grad_conc[cell] = val;
                }
            }
            cg::this_grid().sync();
        }

        cg::this_grid().sync();

            // Record elapsed cycles for backward pass
            if (bwd_trace_slot != nullptr && tid == 0) {
                unsigned long long bwd_cycle_end = clock64();
                unsigned long long elapsed = bwd_cycle_end - bwd_cycle_start;
                bwd_trace_slot->cycles_elapsed = elapsed;
                bwd_trace_slot->tensor_core_cycles = elapsed;
            }
        }


        // Effective rank computation - uses param_map interleaved layout [P0|I0|V0|P1|I1|V1|...]
        // Blocks without work only participate in syncs, never touch data.
        {
            __shared__ float head_grad_sq[NUM_HEADS];
            __shared__ float warp_sums_eff[32];

            for (int h = 0; h < NUM_HEADS; h++) {
                if (has_work && h < arch.num_heads) {
                    int head_start = param_map->head_param_offsets[h];
                    int head_count = param_map->head_param_counts[h];
                    float* grad_buf = ca_state->tape.grad_buffer;

                    float local_sq = 0.0f;
                    for (int i = tid; i < head_count; i += blockDim.x) {
                        float g = grad_buf[head_start + i];
                        local_sq += g * g;
                    }

                    unsigned mask = __activemask();
                    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                        local_sq += __shfl_down_sync(mask, local_sq, offset);
                    }
                    int lane = tid % warpSize;
                    int warp_id = tid / warpSize;
                    if (lane == 0) warp_sums_eff[warp_id] = local_sq;
                }
                cg::this_grid().sync();

                if (has_work && h < arch.num_heads) {
                    float local_sq = warp_sums_eff[tid < blockDim.x / warpSize ? tid : 0];
                    if (tid < blockDim.x / warpSize) {
                        unsigned active = __activemask();
                        for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2) {
                            local_sq += __shfl_down_sync(active, local_sq, offset);
                        }
                    }
                    if (tid == 0) {
                        head_grad_sq[h] = sqrtf(local_sq / (float)param_map->head_param_counts[h]);
                    }
                }
                cg::this_grid().sync();
            }

            if (has_work && tid == 0) {
                float total_sq = 0.0f;
                for (int h = 0; h < arch.num_heads; h++) {
                    total_sq += head_grad_sq[h] * head_grad_sq[h];
                }

                DEVICE_FATAL_IF(total_sq < 1e-12f, "effective_rank: total_sq < 1e-12");

                float clamped_rank;
                if (total_sq >= 1e-12f) {
                    float entropy = 0.0f;
                    for (int h = 0; h < arch.num_heads; h++) {
                        float g = head_grad_sq[h];
                        float p = (g * g) / total_sq;
                        if (p > 1e-12f) {
                            entropy -= p * logf(p);
                        }
                    }
                    float eff_rank = expf(entropy);
                    clamped_rank = fmaxf(1.0f, fminf((float)arch.num_heads, eff_rank));
                } else {
                    clamped_rank = 1.0f;
                }
                measured_value_set_computed(&entry->effective_rank, clamped_rank, organism->generation, entry->genome_hash);
            }
            cg::this_grid().sync();
        }

        if (has_work && tid == 0) atomicAdd(&g_v_bwd_done_count, 1);

    int alive_ct = pool->alive_indices_count;
    int wave_end_compact = min(wave_start + (int)gridDim.x, alive_ct);
    bool is_last_wave = (wave_end_compact >= alive_ct);

        // Per-entry Adam updates: each block updates its own entry using actual entry_idx
        if (has_work && training_mode->batch_samples != nullptr && training_mode->classifier != nullptr) {
            adam_update_perception_device(organism, entry_idx);
            adam_update_interaction_device(organism, entry_idx);
            adam_update_flow_projection_device(organism, entry_idx);
            adam_update_pooling_device(organism, entry_idx);
            adam_update_fc_weights_device(organism, entry_idx);
            adam_update_fc_bias_device(organism, entry_idx);
        }
        cg::this_grid().sync();

        // Adam timestep: increment once per generation (last wave only)
        if (has_work && blockIdx.x == 0 && is_last_wave && training_mode->batch_samples != nullptr && training_mode->classifier != nullptr) {
            if (tid == 0) {
                training_mode->adam_timestep++;
            }
        }
        cg::this_grid().sync();

        cg::this_grid().sync();
}
