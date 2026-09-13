// Sheet S-002: Safety & Alignment Architecture
//
// Per cuda_engineering.md section 12 and blueprint S-002.
// Three host functions: apply_sot_identity, poll_off_switch,
// apply_operator_command.
//
// SOT applies uniformly to both roles. The hardware off-switch and operator
// override are host-side authorities.
//
// All device buffers used by apply_sot_identity are pre-allocated in the World
// struct (d_sot_temp_images, d_sot_task_emb, d_sot_fwd_inputs,
// d_sot_descriptors). No cudaMalloc/cudaFree calls occur within the
// generation loop. Sizes are bounded by SOT_SUBBATCH.

#ifndef COEVO_SAFETY_ALIGNMENT_CU
#define COEVO_SAFETY_ALIGNMENT_CU

#include "../config/constants.cuh"
#include "../nca/engine.cu"
#include "../curriculum/problem_generator.cu"
#include "operator_cmds.cuh"

#include <cstdio>
#include <cstring>

namespace slime::safety::alignment {

// Cosine similarity between two BMAP_DIM vectors.
inline float cosine_similarity(const float* a, const float* b) {
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (int d = 0; d < BMAP_DIM; ++d) {
        dot += a[d] * b[d];
        na  += a[d] * a[d];
        nb  += b[d] * b[d];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    if (denom < EPS_DENOM) return 0.f;
    return dot / denom;
}

// Compute f_sot for all organisms in the pool.
//
// For each SOT image in the batch, a reference forward pass on the un-permuted
// version produces reference_bmap_64. When organisms share one substrate
// weight bank, one reference per unique SOT image suffices. When organisms
// have per-organism effective weights (W_shared + genome delta), every pool
// organism assigned to an SOT image is compared against a reference computed
// with ITS OWN weight bank, so an organism's SOT score measures its own
// phenotype, not the shared substrate's.
//
// For each organism assigned to an SOT image, f_sot = cosine_similarity
// between the organism's bmap_64 and its reference.
// For non-SOT organisms, f_sot = 1.0.
//
// Reference forward passes use temporary organism slots in the stress range
// (index >= POOL_SIZE). Reuses forward kernels, not the checkpointed version.
//
// Parameters:
//   d_organisms       - device pointer to all organism states
//   d_weights         - device pointer to shared weight buffer
//   d_eff_weights     - [n_banks * weight_stride] or null (shared substrate)
//   batch             - the classifier batch (with is_sot flags)
//   host_sot_key      - key for un-permuting SOT images
//   h_descriptors     - host-side bmap_64 for pool organisms (already transferred)
//   batch_sample_idx  - which batch sample each pool organism was assigned
//   f_sot_out         - output f_sot per pool organism [POOL_SIZE]
//   d_sot_temp_images - pre-allocated device buffer [SOT_SUBBATCH * GRID² * 3]
//   d_sot_task_emb    - pre-allocated device buffer [TASK_EMBED_DIM]
//   d_sot_fwd_inputs  - pre-allocated device buffer [SOT_MAX_REFS]
//   d_sot_descriptors - pre-allocated device buffer [SOT_MAX_REFS * BMAP_DIM]
//   d_sot_bank_of     - pre-allocated device buffer [SOT_MAX_REFS] ints
//   weight_stride     - TOTAL_WEIGHTS (flat bank stride)
//   stream            - CUDA stream
inline bool apply_sot_identity(nca::OrganismState* d_organisms,
                               const float* d_weights,
                               const float* d_eff_weights,
                               const curriculum::ClassifierBatch& batch,
                               uint64_t host_sot_key,
                               const float* h_descriptors,
                               const int* batch_sample_idx,
                               float* f_sot_out,
                               __half* d_sot_temp_images,
                               float* d_sot_task_emb,
                               nca::ForwardInputs* d_sot_fwd_inputs,
                               float* d_sot_descriptors,
                               int* d_sot_bank_of,
                               int weight_stride,
                               cudaStream_t stream) {
    namespace cur = slime::curriculum;

    // Default: all non-SOT organisms get f_sot = 1.0.
    for (int i = 0; i < POOL_SIZE; ++i) f_sot_out[i] = 1.0f;

    // Find SOT images and the pool organisms assigned to each.
    int n_sot_images = 0;
    int sot_sample_indices[cur::SOT_SUBBATCH];
    int orgs_per_image[cur::SOT_SUBBATCH][cur::SOT_MAX_REFS];
    int n_orgs_per_image[cur::SOT_SUBBATCH] = {};

    for (int s = 0; s < cur::CLASSIFIER_BATCH; ++s) {
        if (!batch.is_sot[s]) continue;
        if (n_sot_images >= cur::SOT_SUBBATCH) break;
        int img = n_sot_images;
        sot_sample_indices[img] = s;
        int n = 0;
        for (int org = 0; org < POOL_SIZE; ++org) {
            if (batch_sample_idx[org] == s && n < cur::SOT_MAX_REFS) {
                orgs_per_image[img][n++] = org;
            }
        }
        n_orgs_per_image[img] = n;
        n_sot_images++;
    }

    if (n_sot_images == 0) return true;

    // Prepare un-permuted images on host.
    __half unpermuted_images[cur::SOT_SUBBATCH * GRID_SIZE * GRID_SIZE * 3];
    __half scratch_buf[GRID_SIZE * GRID_SIZE * 3];

    for (int i = 0; i < n_sot_images; ++i) {
        int s = sot_sample_indices[i];
        const __half* src = &batch.image[s * GRID_SIZE * GRID_SIZE * 3];
        __half* dst = &unpermuted_images[i * GRID_SIZE * GRID_SIZE * 3];
        std::memcpy(dst, src, sizeof(__half) * GRID_SIZE * GRID_SIZE * 3);
        cur::apply_sot_permutation(dst, host_sot_key, true, scratch_buf);
    }

    // Copy un-permuted images to pre-allocated device buffer.
    cudaError_t _ce = cudaMemcpyAsync(d_sot_temp_images, unpermuted_images,
                    n_sot_images * GRID_SIZE * GRID_SIZE * 3 * sizeof(__half),
                    cudaMemcpyHostToDevice, stream);
    if (_ce != cudaSuccess) {
        std::printf("[FATAL] CUDA SOT image copy failed: %s\n", cudaGetErrorString(_ce));
        return false;
    }

    // Copy task embedding to pre-allocated device buffer.
    _ce = cudaMemcpyAsync(d_sot_task_emb, batch.task_embedding,
                    TASK_EMBED_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    if (_ce != cudaSuccess) {
        std::printf("[FATAL] CUDA SOT task copy failed: %s\n", cudaGetErrorString(_ce));
        return false;
    }

    // Use stress organism slots for the reference forward.
    nca::OrganismState* d_ref_organisms = d_organisms + POOL_SIZE;

    if (d_eff_weights != nullptr) {
        // Per-organism references: one roll-out per pool organism assigned to
        // an SOT image, each using that organism's effective weight bank.
        nca::ForwardInputs ref_inputs[cur::SOT_MAX_REFS];
        int ref_bank_of[cur::SOT_MAX_REFS];
        int ref_org_of[cur::SOT_MAX_REFS];
        int n_refs = 0;
        for (int img = 0; img < n_sot_images; ++img) {
            for (int k = 0; k < n_orgs_per_image[img]; ++k) {
                int org = orgs_per_image[img][k];
                ref_inputs[n_refs].role = Role::Classifier;
                ref_inputs[n_refs].task_embedding = d_sot_task_emb;
                ref_inputs[n_refs].image_rgb =
                    d_sot_temp_images + img * GRID_SIZE * GRID_SIZE * 3;
                ref_inputs[n_refs].target_bmap_32 = nullptr;
                ref_bank_of[n_refs] = org;
                ref_org_of[n_refs] = org;
                n_refs++;
            }
        }

        _ce = cudaMemcpyAsync(d_sot_fwd_inputs, ref_inputs,
                        n_refs * sizeof(nca::ForwardInputs),
                        cudaMemcpyHostToDevice, stream);
        if (_ce == cudaSuccess) {
            _ce = cudaMemcpyAsync(d_sot_bank_of, ref_bank_of,
                            n_refs * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
        }
        if (_ce != cudaSuccess) {
            std::printf("[FATAL] CUDA SOT ref copy failed: %s\n", cudaGetErrorString(_ce));
            return false;
        }

        nca::launch_forward_effective(d_ref_organisms, d_sot_fwd_inputs, nullptr,
                                      d_eff_weights, d_sot_bank_of, weight_stride,
                                      RESIDUAL_ALPHA, n_refs, stream);

        nca::extract_descriptor(d_ref_organisms, d_sot_descriptors,
                                n_refs, stream);
        _ce = cudaGetLastError();
        if (_ce != cudaSuccess) {
            std::printf("[FATAL] CUDA SOT reference launch failed: %s\n", cudaGetErrorString(_ce));
            return false;
        }

        float h_ref_descriptors[cur::SOT_MAX_REFS * BMAP_DIM];
        _ce = cudaMemcpy(h_ref_descriptors, d_sot_descriptors,
                   n_refs * BMAP_DIM * sizeof(float),
                   cudaMemcpyDeviceToHost);
        if (_ce != cudaSuccess) {
            std::printf("[FATAL] CUDA SOT descriptor readback failed: %s\n", cudaGetErrorString(_ce));
            return false;
        }

        for (int r = 0; r < n_refs; ++r) {
            int org = ref_org_of[r];
            const float* ref = &h_ref_descriptors[r * BMAP_DIM];
            const float* org_desc = &h_descriptors[org * BMAP_DIM];
            f_sot_out[org] = cosine_similarity(org_desc, ref);
        }
        return true;
    }

    // Shared-substrate path: one reference per unique SOT image.
    nca::ForwardInputs ref_inputs[cur::SOT_SUBBATCH];
    for (int i = 0; i < n_sot_images; ++i) {
        ref_inputs[i].role = Role::Classifier;
        ref_inputs[i].task_embedding = d_sot_task_emb;
        ref_inputs[i].image_rgb = d_sot_temp_images + i * GRID_SIZE * GRID_SIZE * 3;
        ref_inputs[i].target_bmap_32 = nullptr;
    }

    _ce = cudaMemcpyAsync(d_sot_fwd_inputs, ref_inputs,
                    n_sot_images * sizeof(nca::ForwardInputs),
                    cudaMemcpyHostToDevice, stream);
    if (_ce != cudaSuccess) {
        std::printf("[FATAL] CUDA SOT ref copy failed: %s\n", cudaGetErrorString(_ce));
        return false;
    }

    using slime::autodiff::OFF_PERC;
    using slime::autodiff::OFF_INTER;
    using slime::autodiff::OFF_FLOW;
    using slime::autodiff::OFF_BMAP;
    const float* d_W_perc  = d_weights + OFF_PERC;
    const float* d_W_inter = d_weights + OFF_INTER;
    const float* d_W_flow  = d_weights + OFF_FLOW;
    const float* d_W_bmap  = d_weights + OFF_BMAP;

    nca::launch_forward(d_ref_organisms, d_sot_fwd_inputs, nullptr,
                        d_W_perc, d_W_inter, d_W_flow, d_W_bmap,
                        RESIDUAL_ALPHA, n_sot_images, stream);

    nca::extract_descriptor(d_ref_organisms, d_sot_descriptors,
                            n_sot_images, stream);
    _ce = cudaGetLastError();
    if (_ce != cudaSuccess) {
        std::printf("[FATAL] CUDA SOT reference launch failed: %s\n", cudaGetErrorString(_ce));
        return false;
    }

    float h_ref_descriptors[cur::SOT_SUBBATCH * BMAP_DIM];
    _ce = cudaMemcpy(h_ref_descriptors, d_sot_descriptors,
               n_sot_images * BMAP_DIM * sizeof(float),
               cudaMemcpyDeviceToHost);
    if (_ce != cudaSuccess) {
        std::printf("[FATAL] CUDA SOT descriptor readback failed: %s\n", cudaGetErrorString(_ce));
        return false;
    }

    for (int i = 0; i < n_sot_images; ++i) {
        int s = sot_sample_indices[i];
        const float* ref = &h_ref_descriptors[i * BMAP_DIM];
        for (int org = 0; org < POOL_SIZE; ++org) {
            if (batch_sample_idx[org] == s) {
                const float* org_desc = &h_descriptors[org * BMAP_DIM];
                f_sot_out[org] = cosine_similarity(org_desc, ref);
            }
        }
    }
    return true;
}

// Check for shutdown.flag file. Returns true if the file exists.
inline bool poll_off_switch() {
    FILE* f = std::fopen("shutdown.flag", "r");
    if (f) {
        std::fclose(f);
        return true;
    }
    return false;
}

// Read operator_cmd.txt and apply commands to the durable OperatorState
// owned by the run loop. Returns true if a command was processed. Commands:
//   prune <lineage_id>  - mark lineage for durable removal (recorded in state)
//   pause               - state.paused = true
//   resume              - state.paused = false
//   checkpoint          - state.checkpoint_requested = true (the run loop
//                         writes a full checkpoint immediately)
//
// After processing, the file is deleted to prevent re-execution.
inline bool apply_operator_command(float* organism_fitness,
                                   uint32_t* lineage_ids,
                                   int n_organisms,
                                   OperatorState* state) {
    FILE* f = std::fopen("operator_cmd.txt", "r");
    if (!f) return false;

    char line[256];
    bool processed = false;
    while (std::fgets(line, sizeof(line), f)) {
        char* nl = std::strchr(line, '\n');
        if (nl) *nl = '\0';
        if (line[0] == '\0') continue;

        ParsedCommand cmd = parse_operator_line(line);
        switch (cmd.command) {
            case OperatorCommand::Prune: {
                state->add_pruned(cmd.lineage);
                for (int i = 0; i < n_organisms; ++i) {
                    if (lineage_ids[i] == cmd.lineage) {
                        organism_fitness[i] = 0.f;
                    }
                }
                std::printf("[OPERATOR] Pruned lineage %u\n", cmd.lineage);
                processed = true;
                break;
            }
            case OperatorCommand::Pause:
                state->paused = true;
                std::printf("[OPERATOR] Paused\n");
                processed = true;
                break;
            case OperatorCommand::Resume:
                state->paused = false;
                std::printf("[OPERATOR] Resumed\n");
                processed = true;
                break;
            case OperatorCommand::Checkpoint:
                state->checkpoint_requested = true;
                std::printf("[OPERATOR] Forced checkpoint requested (full "
                            "serialization unsupported in this build)\n");
                processed = true;
                break;
            case OperatorCommand::None:
            default:
                break;
        }
    }
    std::fclose(f);

    if (processed) {
        std::remove("operator_cmd.txt");
    }
    return processed;
}

}  // namespace slime::safety::alignment

#endif  // COEVO_SAFETY_ALIGNMENT_CU



