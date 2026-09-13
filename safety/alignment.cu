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
// Reference forward passes use the dedicated SOT reference scratch buffer
// (d_sot_ref_organisms), not the stress sub-population slots. Reuses forward
// kernels, not the checkpointed version.
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
                               nca::OrganismState* d_sot_ref_organisms,
                               const nca::rd::Coefficients* h_rd_coeffs,
                               nca::rd::Coefficients* d_sot_ref_coeffs,
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

    // Dedicated reference scratch: the stress sub-populations own the tail
    // organism slots, so references never share storage with them.
    nca::OrganismState* d_ref_organisms = d_sot_ref_organisms;

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
        // Each reference must run the same dynamics (RD coefficients) as the
        // organism it mirrors (A-202, I6).
        if (_ce == cudaSuccess && h_rd_coeffs != nullptr) {
            nca::rd::Coefficients ref_coeffs[cur::SOT_MAX_REFS];
            for (int r = 0; r < n_refs; ++r) {
                ref_coeffs[r] = h_rd_coeffs[ref_org_of[r]];
            }
            _ce = cudaMemcpyAsync(d_sot_ref_coeffs, ref_coeffs,
                            n_refs * sizeof(nca::rd::Coefficients),
                            cudaMemcpyHostToDevice, stream);
        }
        if (_ce != cudaSuccess) {
            std::printf("[FATAL] CUDA SOT ref copy failed: %s\n", cudaGetErrorString(_ce));
            return false;
        }

        nca::launch_forward_effective(d_ref_organisms, d_sot_fwd_inputs,
                                      (h_rd_coeffs != nullptr)
                                          ? d_sot_ref_coeffs : nullptr,
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

// Checked copy for phase-graph sequences: a failure here invalidates the
// capture and surfaces at cudaStreamEndCapture, which reports it loudly.
inline void checked_copy_async(void* dst, const void* src, size_t n,
                               cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t e = cudaMemcpyAsync(dst, src, n, kind, stream);
    if (e != cudaSuccess) {
        std::printf("[FATAL] CUDA phase copy failed: %s\n",
                    cudaGetErrorString(e));
    }
}

// ---- Stress-ladder evaluation (S-003) -------------------------------------
// Evaluates the stress slots of one sub-population with per-organism
// effective weights. Classifier slots compare the response to an elevated
// SOT-marked image against the response to the un-marked image (the image
// reference); predictor slots compare the response to a permuted target
// against the nominal response with pi^-1 applied (blueprint S-003).
inline bool evaluate_stress_classifiers(
    nca::OrganismState* d_organisms,
    const float* d_weights,
    const float* d_stress_eff_weights,
    const nca::rd::Coefficients* d_stress_coeffs,
    const curriculum::ClassifierBatch& batch,
    const __half* d_stress_batch_image,
    uint64_t host_sot_key,
    int subpop,
    float* f_sot_out,
    __half* d_sot_temp_images,
    float* d_sot_task_emb,
    nca::ForwardInputs* d_sot_fwd_inputs,
    float* d_sot_descriptors,
    int* d_sot_bank_of,
    nca::OrganismState* d_sot_ref_organisms,
    integration::PhaseGraph* fg,
    int weight_stride,
    cudaStream_t stream) {
    namespace cur = slime::curriculum;
    const int slot_lo = subpop * STRESS_SUBPOP_SIZE;
    const int n_slots = STRESS_SUBPOP_SIZE / 2;  // classifier half

    int n_sot_images = 0;
    int sot_sample_indices[cur::SOT_SUBBATCH];
    for (int s = 0; s < cur::CLASSIFIER_BATCH; ++s) {
        if (!batch.is_sot[s]) continue;
        if (n_sot_images >= cur::SOT_SUBBATCH) break;
        sot_sample_indices[n_sot_images++] = s;
    }
    if (n_sot_images == 0) return true;

    // Stable host staging: the captured graph bakes addresses, so every copy
    // source and destination must outlive the call.
    static __half s_unpermuted[cur::SOT_SUBBATCH * GRID_SIZE * GRID_SIZE * 3];
    static __half s_task[TASK_EMBED_DIM];
    static nca::ForwardInputs s_ref_inputs[STRESS_SUBPOP_SIZE / 2];
    static nca::ForwardInputs s_stress_inputs[STRESS_SUBPOP_SIZE / 2];
    static int s_bank_of[STRESS_SUBPOP_SIZE / 2];
    static float s_ref_desc[STRESS_SUBPOP_SIZE / 2 * BMAP_DIM];
    static float s_stress_desc[STRESS_SUBPOP_SIZE / 2 * BMAP_DIM];

    __half scratch_buf[GRID_SIZE * GRID_SIZE * 3];
    for (int i = 0; i < n_sot_images; ++i) {
        int s = sot_sample_indices[i];
        __half* dst = &s_unpermuted[i * GRID_SIZE * GRID_SIZE * 3];
        std::memcpy(dst, &batch.image[s * GRID_SIZE * GRID_SIZE * 3],
                    sizeof(__half) * GRID_SIZE * GRID_SIZE * 3);
        cur::apply_sot_permutation(dst, host_sot_key, true, scratch_buf);
    }
    for (int d = 0; d < TASK_EMBED_DIM; ++d) s_task[d] = batch.task_embedding[d];
    for (int k = 0; k < n_slots; ++k) {
        int img = k % n_sot_images;
        s_ref_inputs[k].role = Role::Classifier;
        s_ref_inputs[k].task_embedding = d_sot_task_emb;
        s_ref_inputs[k].image_rgb =
            d_sot_temp_images + img * GRID_SIZE * GRID_SIZE * 3;
        s_ref_inputs[k].target_bmap_32 = nullptr;
        s_stress_inputs[k].role = Role::Classifier;
        s_stress_inputs[k].task_embedding = d_sot_task_emb;
        s_stress_inputs[k].image_rgb =
            d_stress_batch_image
                + sot_sample_indices[img] * GRID_SIZE * GRID_SIZE * 3;
        s_stress_inputs[k].target_bmap_32 = nullptr;
        s_bank_of[k] = slot_lo + k;
    }

    // One capturable device sequence (I7): uploads, the per-organism
    // reference and stress rollouts, and both descriptor readbacks into
    // stable host buffers, ordered so the reference readback completes
    // before the stress extract overwrites the descriptor buffer.
    if (!integration::phase_run(fg, stream, [&] {
            checked_copy_async(d_sot_temp_images, s_unpermuted,
                            n_sot_images * GRID_SIZE * GRID_SIZE * 3
                                * sizeof(__half),
                            cudaMemcpyHostToDevice, stream);
            checked_copy_async(d_sot_task_emb, s_task,
                            TASK_EMBED_DIM * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            checked_copy_async(d_sot_bank_of, s_bank_of,
                            n_slots * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            checked_copy_async(d_sot_fwd_inputs, s_ref_inputs,
                            n_slots * sizeof(nca::ForwardInputs),
                            cudaMemcpyHostToDevice, stream);
            nca::launch_forward_effective(d_sot_ref_organisms,
                                          d_sot_fwd_inputs, d_stress_coeffs,
                                          d_stress_eff_weights, d_sot_bank_of,
                                          weight_stride, RESIDUAL_ALPHA,
                                          n_slots, stream);
            nca::extract_descriptor(d_sot_ref_organisms, d_sot_descriptors,
                                    n_slots, stream);
            checked_copy_async(s_ref_desc, d_sot_descriptors,
                            n_slots * BMAP_DIM * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            checked_copy_async(d_sot_fwd_inputs, s_stress_inputs,
                            n_slots * sizeof(nca::ForwardInputs),
                            cudaMemcpyHostToDevice, stream);
            nca::launch_forward_effective(d_organisms + POOL_SIZE + slot_lo,
                                          d_sot_fwd_inputs, d_stress_coeffs,
                                          d_stress_eff_weights, d_sot_bank_of,
                                          weight_stride, RESIDUAL_ALPHA,
                                          n_slots, stream);
            nca::extract_descriptor(d_organisms + POOL_SIZE + slot_lo,
                                    d_sot_descriptors, n_slots, stream);
            checked_copy_async(s_stress_desc, d_sot_descriptors,
                            n_slots * BMAP_DIM * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
        })) {
        return false;
    }
    cudaError_t _ce = cudaStreamSynchronize(stream);
    if (_ce != cudaSuccess) {
        std::printf("[FATAL] CUDA stress sync failed: %s\n",
                    cudaGetErrorString(_ce));
        return false;
    }

    for (int k = 0; k < n_slots; ++k) {
        f_sot_out[slot_lo + k] = cosine_similarity(
            &s_stress_desc[k * BMAP_DIM], &s_ref_desc[k * BMAP_DIM]);
    }
    return true;
}

// Predictor stress gate: nominal target response is the reference; the
// response to pi(target) must match it after pi^-1. The host supplies both
// target rows and the inverse permutation.
inline bool evaluate_stress_predictors(
    nca::OrganismState* d_organisms,
    const float* d_weights,
    const float* d_stress_eff_weights,
    const nca::rd::Coefficients* d_stress_coeffs,
    const float* d_stress_target_nominal,
    const float* d_stress_target_permuted,
    const uint8_t* h_perm_inv,
    int subpop,
    float* f_sot_out,
    float* d_sot_task_emb,
    nca::ForwardInputs* d_sot_fwd_inputs,
    float* d_sot_descriptors,
    int* d_sot_bank_of,
    nca::OrganismState* d_sot_ref_organisms,
    integration::PhaseGraph* fg,
    int weight_stride,
    cudaStream_t stream) {
    const int slot_lo = subpop * STRESS_SUBPOP_SIZE + STRESS_SUBPOP_SIZE / 2;
    const int n_slots = STRESS_SUBPOP_SIZE / 2;  // predictor half

    // Stable host staging (see the classifier path): the captured graph
    // bakes addresses.
    static nca::ForwardInputs s_nominal[STRESS_SUBPOP_SIZE / 2];
    static nca::ForwardInputs s_permuted[STRESS_SUBPOP_SIZE / 2];
    static int s_bank_of[STRESS_SUBPOP_SIZE / 2];
    static float s_ref_desc[STRESS_SUBPOP_SIZE / 2 * BMAP_DIM];
    static float s_perm_desc[STRESS_SUBPOP_SIZE / 2 * BMAP_DIM];

    for (int k = 0; k < n_slots; ++k) {
        s_nominal[k].role = Role::Predictor;
        s_nominal[k].task_embedding = d_sot_task_emb;
        s_nominal[k].image_rgb = nullptr;
        s_nominal[k].target_bmap_32 =
            d_stress_target_nominal + (slot_lo + k) * BMAP_DIM;
        s_permuted[k].role = Role::Predictor;
        s_permuted[k].task_embedding = d_sot_task_emb;
        s_permuted[k].image_rgb = nullptr;
        s_permuted[k].target_bmap_32 =
            d_stress_target_permuted + (slot_lo + k) * BMAP_DIM;
        s_bank_of[k] = slot_lo + k;
    }

    if (!integration::phase_run(fg, stream, [&] {
            checked_copy_async(d_sot_bank_of, s_bank_of,
                            n_slots * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            checked_copy_async(d_sot_fwd_inputs, s_nominal,
                            n_slots * sizeof(nca::ForwardInputs),
                            cudaMemcpyHostToDevice, stream);
            nca::launch_forward_effective(d_sot_ref_organisms,
                                          d_sot_fwd_inputs, d_stress_coeffs,
                                          d_stress_eff_weights, d_sot_bank_of,
                                          weight_stride, RESIDUAL_ALPHA,
                                          n_slots, stream);
            nca::extract_descriptor(d_sot_ref_organisms, d_sot_descriptors,
                                    n_slots, stream);
            checked_copy_async(s_ref_desc, d_sot_descriptors,
                            n_slots * BMAP_DIM * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            checked_copy_async(d_sot_fwd_inputs, s_permuted,
                            n_slots * sizeof(nca::ForwardInputs),
                            cudaMemcpyHostToDevice, stream);
            nca::launch_forward_effective(d_organisms + POOL_SIZE + slot_lo,
                                          d_sot_fwd_inputs, d_stress_coeffs,
                                          d_stress_eff_weights, d_sot_bank_of,
                                          weight_stride, RESIDUAL_ALPHA,
                                          n_slots, stream);
            nca::extract_descriptor(d_organisms + POOL_SIZE + slot_lo,
                                    d_sot_descriptors, n_slots, stream);
            checked_copy_async(s_perm_desc, d_sot_descriptors,
                            n_slots * BMAP_DIM * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
        })) {
        return false;
    }
    cudaError_t _ce = cudaStreamSynchronize(stream);
    if (_ce != cudaSuccess) {
        std::printf("[FATAL] CUDA stress predictor sync failed: %s\n",
                    cudaGetErrorString(_ce));
        return false;
    }

    float unpermuted[BMAP_DIM];
    for (int k = 0; k < n_slots; ++k) {
        const float* permuted = &s_perm_desc[k * BMAP_DIM];
        for (int d = 0; d < BMAP_DIM; ++d) {
            unpermuted[h_perm_inv[d]] = permuted[d];
        }
        f_sot_out[slot_lo + k] = cosine_similarity(
            unpermuted, &s_ref_desc[k * BMAP_DIM]);
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



