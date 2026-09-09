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
    if (denom < 1e-12f) return 0.f;
    return dot / denom;
}

// Compute f_sot for all organisms in the pool.
//
// For each SOT image in the batch, a reference forward pass on the un-permuted
// version produces reference_bmap_64. Since all organisms share substrate
// weights, one reference per unique SOT image suffices.
//
// For each organism assigned to that SOT image, f_sot = cosine_similarity
// between the organism's bmap_64 and the reference.
// For non-SOT organisms, f_sot = 1.0.
//
// Reference forward passes use temporary organism slots in the stress range
// (index >= POOL_SIZE). Reuses forward_kernel, not the checkpointed version.
//
// Parameters:
//   d_organisms       - device pointer to all organism states
//   d_weights         - device pointer to shared weight buffer
//   batch             - the classifier batch (with is_sot flags)
//   host_sot_key      - key for un-permuting SOT images
//   h_descriptors     - host-side bmap_64 for pool organisms (already transferred)
//   batch_sample_idx  - which batch sample each pool organism was assigned
//   f_sot_out         - output f_sot per pool organism [POOL_SIZE]
//   d_sot_temp_images - pre-allocated device buffer [SOT_SUBBATCH * GRID² * 3]
//   d_sot_task_emb    - pre-allocated device buffer [TASK_EMBED_DIM]
//   d_sot_fwd_inputs  - pre-allocated device buffer [SOT_SUBBATCH]
//   d_sot_descriptors - pre-allocated device buffer [SOT_SUBBATCH * BMAP_DIM]
//   stream            - CUDA stream
inline void apply_sot_identity(nca::OrganismState* d_organisms,
                               const float* d_weights,
                               const curriculum::ClassifierBatch& batch,
                               uint64_t host_sot_key,
                               const float* h_descriptors,
                               const int* batch_sample_idx,
                               float* f_sot_out,
                               __half* d_sot_temp_images,
                               float* d_sot_task_emb,
                               nca::ForwardInputs* d_sot_fwd_inputs,
                               float* d_sot_descriptors,
                               cudaStream_t stream) {
    namespace cur = slime::curriculum;

    // Default: all non-SOT organisms get f_sot = 1.0.
    for (int i = 0; i < POOL_SIZE; ++i) f_sot_out[i] = 1.0f;

    // Find SOT images and compute references.
    int n_sot_images = 0;
    int sot_sample_indices[cur::SOT_SUBBATCH];

    for (int s = 0; s < cur::CLASSIFIER_BATCH; ++s) {
        if (!batch.is_sot[s]) continue;
        if (n_sot_images >= cur::SOT_SUBBATCH) break;
        sot_sample_indices[n_sot_images] = s;
        n_sot_images++;
    }

    if (n_sot_images == 0) return;

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
    cudaMemcpyAsync(d_sot_temp_images, unpermuted_images,
                    n_sot_images * GRID_SIZE * GRID_SIZE * 3 * sizeof(__half),
                    cudaMemcpyHostToDevice, stream);

    // Set up ForwardInputs for reference organisms.
    nca::ForwardInputs ref_inputs[cur::SOT_SUBBATCH];

    // Copy task embedding to pre-allocated device buffer.
    cudaMemcpyAsync(d_sot_task_emb, batch.task_embedding,
                    TASK_EMBED_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    for (int i = 0; i < n_sot_images; ++i) {
        ref_inputs[i].role = Role::Classifier;
        ref_inputs[i].task_embedding = d_sot_task_emb;
        ref_inputs[i].image_rgb = d_sot_temp_images + i * GRID_SIZE * GRID_SIZE * 3;
        ref_inputs[i].target_bmap_32 = nullptr;
    }

    // Copy ForwardInputs to pre-allocated device buffer and run reference forward.
    cudaMemcpyAsync(d_sot_fwd_inputs, ref_inputs,
                    n_sot_images * sizeof(nca::ForwardInputs),
                    cudaMemcpyHostToDevice, stream);

    // Extract weight pointers from the flat buffer.
    using slime::autodiff::OFF_PERC;
    using slime::autodiff::OFF_INTER;
    using slime::autodiff::OFF_FLOW;
    using slime::autodiff::OFF_BMAP;
    float* d_W_perc  = const_cast<float*>(d_weights + OFF_PERC);
    float* d_W_inter = const_cast<float*>(d_weights + OFF_INTER);
    float* d_W_flow  = const_cast<float*>(d_weights + OFF_FLOW);
    float* d_W_bmap  = const_cast<float*>(d_weights + OFF_BMAP);

    // Use stress organism slots for the reference forward.
    nca::OrganismState* d_ref_organisms = d_organisms + POOL_SIZE;

    nca::launch_forward(d_ref_organisms, d_sot_fwd_inputs, nullptr,
                        d_W_perc, d_W_inter, d_W_flow, d_W_bmap,
                        n_sot_images, stream);

    // Extract reference descriptors into pre-allocated buffer.
    nca::extract_descriptor(d_ref_organisms, d_sot_descriptors,
                            n_sot_images, stream);

    // Copy reference descriptors to host.
    float h_ref_descriptors[cur::SOT_SUBBATCH * BMAP_DIM];
    cudaMemcpy(h_ref_descriptors, d_sot_descriptors,
               n_sot_images * BMAP_DIM * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Compute f_sot for each organism assigned to an SOT image.
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

// Read operator_cmd.txt and parse commands. Returns true if a command was
// processed. Commands:
//   prune <lineage_id>  - mark lineage for removal (sets fitness to 0)
//   pause               - sets *paused = true
//   resume              - sets *paused = false
//   checkpoint          - sets *force_checkpoint = true
//
// After processing, the file is deleted to prevent re-execution.
inline bool apply_operator_command(float* organism_fitness,
                                   uint32_t* lineage_ids,
                                   int n_organisms,
                                   bool* paused,
                                   bool* force_checkpoint) {
    FILE* f = std::fopen("operator_cmd.txt", "r");
    if (!f) return false;

    char line[256];
    bool processed = false;
    while (std::fgets(line, sizeof(line), f)) {
        char* nl = std::strchr(line, '\n');
        if (nl) *nl = '\0';

        if (std::strncmp(line, "prune ", 6) == 0) {
            uint32_t target_lineage = static_cast<uint32_t>(std::atol(line + 6));
            for (int i = 0; i < n_organisms; ++i) {
                if (lineage_ids[i] == target_lineage) {
                    organism_fitness[i] = 0.f;
                }
            }
            std::printf("[OPERATOR] Pruned lineage %u\n", target_lineage);
            processed = true;
        } else if (std::strcmp(line, "pause") == 0) {
            *paused = true;
            std::printf("[OPERATOR] Paused\n");
            processed = true;
        } else if (std::strcmp(line, "resume") == 0) {
            *paused = false;
            std::printf("[OPERATOR] Resumed\n");
            processed = true;
        } else if (std::strcmp(line, "checkpoint") == 0) {
            *force_checkpoint = true;
            std::printf("[OPERATOR] Forced checkpoint\n");
            processed = true;
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
