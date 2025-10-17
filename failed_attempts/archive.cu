#include <cuda_runtime.h>
#include <cuda/std/atomic>
#include <curand_kernel.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

struct Elite {
    float fitness;
    float coherence;
    float effective_rank;
    float hunger;
    float behavioral_descriptor[10];
    float* genome;
    float* delta_chain;
    unsigned long long genome_hash;
    int genome_size;
    int delta_size;
    int generation;
    int parent_ids[2];
    float creation_time;
    float last_access_time;
    float mutation_strength;
};

struct VoronoiCell {
    float centroid[10];
    float* elite_fitnesses;
    Elite** elites;
    cuda::std::atomic<int> elite_count;
    cuda::std::atomic<float> density;
    float volume;
    int capacity;
    int last_update_step;
};

struct CVTArchive {
    VoronoiCell* cells;
    float* distance_matrix;
    int* nearest_neighbors;
    cuda::std::atomic<int> total_elites;
    cuda::std::atomic<float> total_coverage;
    int n_cells;
    int behavior_dim;
    int k_neighbors;

    float* diresa_embeddings;
    float* diresa_weights;
    cuda::std::atomic<int> diresa_dim;
    int diresa_capacity;

    unsigned long long* hash_table;
    int hash_table_size;

    float temperature;
    int annealing_step;
};

__device__ unsigned long long jenkins_hash(float* data, int size) {
    unsigned long long hash = 0;
    for (int i = 0; i < size; i++) {
        unsigned int bits = __float_as_uint(data[i]);
        hash += bits;
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
}

__device__ float p_adic_distance(unsigned int x, unsigned int y, int p) {
    unsigned int diff = x ^ y;
    int valuation = __clz(diff);
    return __powf((float)p, -(float)valuation);
}

__global__ void jacobi_svd_kernel(float* A, float* U, float* S, float* V, int m, int n) {
    __shared__ float a_shared[64][64];
    __shared__ float u_shared[64][64];
    __shared__ float v_shared[64][64];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (tid < m && bid < n) {
        a_shared[tid][bid] = A[tid * n + bid];
        u_shared[tid][bid] = (tid == bid) ? 1.0f : 0.0f;
        v_shared[tid][bid] = (tid == bid) ? 1.0f : 0.0f;
    }
    __syncthreads();

    const int MAX_SWEEPS = 30;
    for (int sweep = 0; sweep < MAX_SWEEPS; sweep++) {
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                float app = 0.0f, aqq = 0.0f, apq = 0.0f;

                #pragma unroll
                for (int i = 0; i < m; i++) {
                    app += a_shared[i][p] * a_shared[i][p];
                    aqq += a_shared[i][q] * a_shared[i][q];
                    apq += a_shared[i][p] * a_shared[i][q];
                }

                if (fabsf(apq) > 1e-10f * sqrtf(app * aqq)) {
                    float tau = (aqq - app) / (2.0f * apq);
                    float t = (tau >= 0) ? 1.0f / (tau + sqrtf(1.0f + tau * tau))
                                         : -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                    float c = 1.0f / sqrtf(1.0f + t * t);
                    float s = t * c;

                    __syncthreads();
                    for (int i = tid; i < m; i += blockDim.x) {
                        float aip = a_shared[i][p];
                        float aiq = a_shared[i][q];
                        a_shared[i][p] = c * aip - s * aiq;
                        a_shared[i][q] = s * aip + c * aiq;

                        float uip = u_shared[i][p];
                        float uiq = u_shared[i][q];
                        u_shared[i][p] = c * uip - s * uiq;
                        u_shared[i][q] = s * uip + c * uiq;
                    }

                    for (int j = tid; j < n; j += blockDim.x) {
                        float vjp = v_shared[j][p];
                        float vjq = v_shared[j][q];
                        v_shared[j][p] = c * vjp - s * vjq;
                        v_shared[j][q] = s * vjp + c * vjq;
                    }
                    __syncthreads();
                }
            }
        }
    }

    if (tid < n) {
        S[tid] = sqrtf(fabsf(a_shared[tid][tid]));
        for (int i = 0; i < m; i++) {
            U[i * n + tid] = u_shared[i][tid];
        }
        for (int j = 0; j < n; j++) {
            V[tid * n + j] = v_shared[tid][j];
        }
    }
}

__global__ void compress_genome_kernel(float* genome, int size, float* compressed,
                                      int* compressed_size, float threshold) {
    extern __shared__ float shared_mem[];
    float* U = shared_mem;
    float* S = shared_mem + size * size;
    float* V = shared_mem + size * size + size;

    jacobi_svd_kernel<<<1, min(size, 64)>>>(genome, U, S, V, size, size);
    cudaDeviceSynchronize();

    int rank = 0;
    for (int i = 0; i < size; i++) {
        if (S[i] > threshold * S[0]) rank++;
    }

    if (threadIdx.x == 0) {
        *compressed_size = rank * (2 * size + 1);
    }

    for (int i = threadIdx.x; i < rank; i += blockDim.x) {
        compressed[i] = S[i];
        for (int j = 0; j < size; j++) {
            compressed[rank + i * size + j] = U[j * size + i];
            compressed[rank * (size + 1) + i * size + j] = V[i * size + j];
        }
    }
}

__global__ void decompress_genome_kernel(float* compressed, int compressed_size,
                                        float* genome, int genome_size) {
    int rank = compressed_size / (2 * genome_size + 1);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < genome_size * genome_size) {
        int i = tid / genome_size;
        int j = tid % genome_size;

        float val = 0.0f;
        #pragma unroll
        for (int r = 0; r < rank; r++) {
            float s = compressed[r];
            float u = compressed[rank + r * genome_size + i];
            float v = compressed[rank * (genome_size + 1) + r * genome_size + j];
            val += s * u * v;
        }

        genome[tid] = val;
    }
}

__global__ void add_elite_kernel(CVTArchive* archive, Elite* candidate) {
    cg::grid_group grid = cg::this_grid();

    unsigned long long hash = jenkins_hash(candidate->genome, candidate->genome_size);

    int hash_bucket = hash % archive->hash_table_size;
    unsigned long long* hash_slot = &archive->hash_table[hash_bucket];

    int attempts = 0;
    while (attempts < 10) {
        unsigned long long existing = atomicCAS(hash_slot, 0, hash);
        if (existing == 0) break;
        if (existing == hash) return;

        hash_bucket = (hash_bucket + 1) % archive->hash_table_size;
        hash_slot = &archive->hash_table[hash_bucket];
        attempts++;
    }

    __shared__ float dist_cache[256];
    __shared__ int nearest_cell;
    __shared__ float min_dist;

    int tid = threadIdx.x;
    if (tid < archive->n_cells) {
        float dist_sq = 0.0f;
        #pragma unroll 10
        for (int d = 0; d < archive->behavior_dim; d++) {
            float diff = candidate->behavioral_descriptor[d] - archive->cells[tid].centroid[d];
            dist_sq += diff * diff;
        }
        dist_cache[tid] = sqrtf(dist_sq);
    }
    __syncthreads();

    if (tid < 32) {
        float local_min = (tid < archive->n_cells) ? dist_cache[tid] : INFINITY;
        int local_idx = tid;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_dist = __shfl_down_sync(0xFFFFFFFF, local_min, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_dist < local_min) {
                local_min = other_dist;
                local_idx = other_idx;
            }
        }

        if (tid == 0) {
            min_dist = local_min;
            nearest_cell = local_idx;
        }
    }
    __syncthreads();

    VoronoiCell* cell = &archive->cells[nearest_cell];

    if (grid.thread_rank() == 0) {
        int old_count = cell->elite_count.load();

        if (old_count == 0) {
            Elite** slot = &cell->elites[0];
            cudaMalloc(slot, sizeof(Elite));
            **slot = *candidate;

            float* compressed;
            int compressed_size;
            cudaMalloc(&compressed, candidate->genome_size * sizeof(float));

            compress_genome_kernel<<<1, 256, 3 * candidate->genome_size * candidate->genome_size * sizeof(float)>>>(
                candidate->genome, candidate->genome_size, compressed, &compressed_size, 0.01f
            );
            cudaDeviceSynchronize();

            (*slot)->delta_chain = compressed;
            (*slot)->delta_size = compressed_size;
            (*slot)->genome_hash = hash;
            (*slot)->creation_time = (float)clock64();
            (*slot)->last_access_time = (*slot)->creation_time;

            cell->elite_count.fetch_add(1);
            archive->total_elites.fetch_add(1);

            float new_density = min(1.0f, cell->density.load() + 0.1f);
            cell->density.store(new_density);
        }
        else if (old_count < cell->capacity) {
            int worst_idx = -1;
            float worst_score = INFINITY;

            for (int i = 0; i < old_count; i++) {
                Elite* e = cell->elites[i];
                float age_factor = expf(-(float)(clock64() - e->creation_time) / 1e9f);
                float score = e->fitness * e->coherence * age_factor;

                if (score < worst_score) {
                    worst_score = score;
                    worst_idx = i;
                }
            }

            float candidate_score = candidate->fitness * candidate->coherence;
            float temperature = archive->temperature;
            float accept_prob = 1.0f / (1.0f + expf(-(candidate_score - worst_score) / temperature));

            curandState rng;
            curand_init(clock64(), grid.thread_rank(), 0, &rng);
            float rand_val = curand_uniform(&rng);

            if (rand_val < accept_prob) {
                Elite* victim = cell->elites[worst_idx];
                cudaFree(victim->delta_chain);

                *victim = *candidate;

                float* compressed;
                int compressed_size;
                cudaMalloc(&compressed, candidate->genome_size * sizeof(float));

                compress_genome_kernel<<<1, 256, 3 * candidate->genome_size * candidate->genome_size * sizeof(float)>>>(
                    candidate->genome, candidate->genome_size, compressed, &compressed_size, 0.01f
                );
                cudaDeviceSynchronize();

                victim->delta_chain = compressed;
                victim->delta_size = compressed_size;
                victim->genome_hash = hash;
                victim->creation_time = (float)clock64();
                victim->last_access_time = victim->creation_time;
                victim->parent_ids[0] = worst_idx;
                victim->parent_ids[1] = -1;

                float new_density = min(1.0f, cell->density.load() * 0.95f + 0.05f);
                cell->density.store(new_density);
            }
        }
        else {
            if (cell->density.load() < 0.5f) {
                int new_capacity = min(100, cell->capacity * 2);
                Elite** new_elites;
                cudaMalloc(&new_elites, new_capacity * sizeof(Elite*));
                cudaMemcpy(new_elites, cell->elites, cell->capacity * sizeof(Elite*),
                          cudaMemcpyDeviceToDevice);
                cudaFree(cell->elites);
                cell->elites = new_elites;
                cell->capacity = new_capacity;
            }
        }

        archive->temperature *= 0.9995f;
        archive->annealing_step++;
    }
}

__global__ void update_voronoi_cells_kernel(CVTArchive* archive) {
    int cell_id = blockIdx.x;
    if (cell_id >= archive->n_cells) return;

    VoronoiCell* cell = &archive->cells[cell_id];

    extern __shared__ float shared_centroid[];
    float* new_centroid = shared_centroid;
    float* weights = shared_centroid + archive->behavior_dim;

    int tid = threadIdx.x;

    if (tid < archive->behavior_dim) {
        new_centroid[tid] = 0.0f;
        weights[tid] = 0.0f;
    }
    __syncthreads();

    int n_elites = cell->elite_count.load();
    for (int i = tid; i < n_elites; i += blockDim.x) {
        Elite* e = cell->elites[i];
        float weight = e->fitness * e->coherence;

        for (int d = 0; d < archive->behavior_dim; d++) {
            atomicAdd(&new_centroid[d], e->behavioral_descriptor[d] * weight);
            atomicAdd(&weights[d], weight);
        }
    }
    __syncthreads();

    if (tid < archive->behavior_dim && weights[tid] > 0) {
        float momentum = 0.9f;
        float adaptive_rate = 1.0f / (1.0f + (float)archive->annealing_step * 0.0001f);
        float updated = new_centroid[tid] / weights[tid];
        cell->centroid[tid] = momentum * cell->centroid[tid] + (1.0f - momentum) * adaptive_rate * updated;
    }

    if (tid == 0) {
        float target_volume = 1.0f / archive->n_cells;
        float density = cell->density.load();
        float adaptive_volume = target_volume * (2.0f - density);
        cell->volume = cell->volume * 0.95f + adaptive_volume * 0.05f;
        cell->last_update_step = archive->annealing_step;
    }
}

__global__ void compute_distance_matrix_kernel(CVTArchive* archive) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;

    if (i >= archive->n_cells || j >= archive->n_cells) return;

    float dist_sq = 0.0f;
    #pragma unroll
    for (int d = 0; d < archive->behavior_dim; d++) {
        float diff = archive->cells[i].centroid[d] - archive->cells[j].centroid[d];
        dist_sq += diff * diff;
    }

    float euclidean = sqrtf(dist_sq);

    unsigned int code_i = i;
    unsigned int code_j = j;
    float p_adic = p_adic_distance(code_i, code_j, 2);

    float hybrid_dist = 0.7f * euclidean + 0.3f * p_adic;

    archive->distance_matrix[i * archive->n_cells + j] = hybrid_dist;
}

__global__ void bitonic_sort_distances_kernel(float* distances, int* indices, int n, int k) {
    extern __shared__ float shared_dist[];
    int* shared_idx = (int*)(shared_dist + n);

    int tid = threadIdx.x;

    if (tid < n) {
        shared_dist[tid] = distances[tid];
        shared_idx[tid] = indices[tid];
    }
    __syncthreads();

    for (int size = 2; size <= n; size *= 2) {
        int dir = (tid & (size - 1)) < (size / 2);

        for (int stride = size / 2; stride > 0; stride /= 2) {
            __syncthreads();
            int partner = tid ^ stride;

            if (partner < n && tid < n) {
                if ((tid & size) == 0) {
                    if (dir == (shared_dist[tid] > shared_dist[partner])) {
                        float tmp_d = shared_dist[tid];
                        shared_dist[tid] = shared_dist[partner];
                        shared_dist[partner] = tmp_d;

                        int tmp_i = shared_idx[tid];
                        shared_idx[tid] = shared_idx[partner];
                        shared_idx[partner] = tmp_i;
                    }
                }
            }
        }
    }
    __syncthreads();

    if (tid < k && tid < n) {
        distances[tid] = shared_dist[tid];
        indices[tid] = shared_idx[tid];
    }
}

__global__ void find_k_nearest_neighbors_kernel(CVTArchive* archive) {
    int cell_id = blockIdx.x;
    if (cell_id >= archive->n_cells) return;

    extern __shared__ float shared_mem[];
    float* distances = shared_mem;
    int* indices = (int*)(shared_mem + archive->n_cells);

    int tid = threadIdx.x;

    if (tid < archive->n_cells) {
        distances[tid] = archive->distance_matrix[cell_id * archive->n_cells + tid];
        indices[tid] = tid;
    }
    __syncthreads();

    bitonic_sort_distances_kernel<<<1, archive->n_cells, 2 * archive->n_cells * sizeof(float)>>>(
        distances, indices, archive->n_cells, archive->k_neighbors + 1
    );
    cudaDeviceSynchronize();

    if (tid > 0 && tid <= archive->k_neighbors) {
        archive->nearest_neighbors[cell_id * archive->k_neighbors + tid - 1] = indices[tid];
    }
}

__global__ void sample_elite_kernel(CVTArchive* archive, float* behavior, float hunger,
                                   Elite* output) {
    __shared__ float cell_scores[256];
    __shared__ int selected_cell;
    __shared__ int selected_elite;

    int tid = threadIdx.x;

    if (tid < archive->n_cells) {
        VoronoiCell* cell = &archive->cells[tid];
        int count = cell->elite_count.load();

        if (count > 0) {
            float dist_sq = 0.0f;
            #pragma unroll
            for (int d = 0; d < archive->behavior_dim; d++) {
                float diff = behavior[d] - cell->centroid[d];
                dist_sq += diff * diff;
            }

            float density = cell->density.load();
            float exploration_bonus = hunger * expf(-density);
            float exploitation_score = (1.0f - hunger) / (1.0f + sqrtf(dist_sq));

            cell_scores[tid] = exploitation_score + exploration_bonus;
        } else {
            cell_scores[tid] = 0.0f;
        }
    }
    __syncthreads();

    if (tid < 32) {
        float max_score = -INFINITY;
        int best_cell = 0;

        for (int i = 0; i < archive->n_cells; i++) {
            if (cell_scores[i] > max_score) {
                max_score = cell_scores[i];
                best_cell = i;
            }
        }

        if (tid == 0) {
            selected_cell = best_cell;
        }
    }
    __syncthreads();

    VoronoiCell* cell = &archive->cells[selected_cell];
    int count = cell->elite_count.load();

    if (tid == 0 && count > 0) {
        float best_score = -INFINITY;
        int best_idx = 0;

        for (int i = 0; i < count; i++) {
            Elite* e = cell->elites[i];
            float recency = 1.0f / (1.0f + (float)(clock64() - e->last_access_time) / 1e9f);
            float quality = e->fitness * e->coherence;
            float novelty = 1.0f - e->hunger;
            float score = quality * (1.0f - hunger) + novelty * hunger + recency * 0.1f;

            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        selected_elite = best_idx;
    }
    __syncthreads();

    if (tid == 0 && count > 0) {
        Elite* selected = cell->elites[selected_elite];
        *output = *selected;

        selected->last_access_time = (float)clock64();

        float* decompressed;
        cudaMalloc(&decompressed, selected->genome_size * sizeof(float));

        decompress_genome_kernel<<<(selected->genome_size * selected->genome_size + 255) / 256, 256>>>(
            selected->delta_chain, selected->delta_size,
            decompressed, selected->genome_size
        );
        cudaDeviceSynchronize();

        output->genome = decompressed;

        int k = archive->k_neighbors;
        for (int n = 0; n < k; n++) {
            int neighbor_id = archive->nearest_neighbors[selected_cell * k + n];
            VoronoiCell* neighbor = &archive->cells[neighbor_id];
            float influence = expf(-(float)n / k);
            neighbor->density.fetch_add(influence * 0.01f);
        }
    }
}

__global__ void update_diresa_kernel(CVTArchive* archive) {
    cg::grid_group grid = cg::this_grid();

    int n_samples = min(1000, archive->total_elites.load());
    int current_dim = archive->diresa_dim.load();
    int tid = grid.thread_rank();

    extern __shared__ float dist_matrix[];

    if (tid < n_samples * n_samples) {
        int i = tid / n_samples;
        int j = tid % n_samples;

        int cell_i = i % archive->n_cells;
        int elite_i = i / archive->n_cells;
        int cell_j = j % archive->n_cells;
        int elite_j = j / archive->n_cells;

        float high_d_dist = 0.0f;
        if (archive->cells[cell_i].elite_count > elite_i &&
            archive->cells[cell_j].elite_count > elite_j) {

            Elite* ei = archive->cells[cell_i].elites[elite_i];
            Elite* ej = archive->cells[cell_j].elites[elite_j];

            #pragma unroll
            for (int d = 0; d < archive->behavior_dim; d++) {
                float diff = ei->behavioral_descriptor[d] - ej->behavioral_descriptor[d];
                high_d_dist += diff * diff;
            }
        }

        dist_matrix[tid] = sqrtf(high_d_dist);
    }
    __syncthreads();

    if (tid < n_samples * current_dim) {
        int sample = tid / current_dim;
        int dim = tid % current_dim;

        float grad = 0.0f;
        for (int other = 0; other < n_samples; other++) {
            if (other != sample) {
                float high_dist = dist_matrix[sample * n_samples + other];
                float low_dist = 0.0f;

                #pragma unroll
                for (int d = 0; d < current_dim; d++) {
                    float diff = archive->diresa_embeddings[sample * 10 + d] -
                               archive->diresa_embeddings[other * 10 + d];
                    low_dist += diff * diff;
                }
                low_dist = sqrtf(low_dist);

                float weight = 1.0f / (1.0f + high_dist);
                float error = (high_dist - low_dist) * weight;

                if (dim < current_dim) {
                    float diff = archive->diresa_embeddings[sample * 10 + dim] -
                               archive->diresa_embeddings[other * 10 + dim];
                    grad += error * diff / (low_dist + 1e-10f);
                }
            }
        }

        archive->diresa_embeddings[sample * 10 + dim] -= 0.01f * grad;
    }

    grid.sync();

    if (tid == 0) {
        float stress = 0.0f;
        for (int i = 0; i < n_samples * n_samples; i++) {
            float high_dist = dist_matrix[i];
            int si = i / n_samples;
            int sj = i % n_samples;

            float low_dist = 0.0f;
            #pragma unroll
            for (int d = 0; d < current_dim; d++) {
                float diff = archive->diresa_embeddings[si * 10 + d] -
                           archive->diresa_embeddings[sj * 10 + d];
                low_dist += diff * diff;
            }
            low_dist = sqrtf(low_dist);

            float error = high_dist - low_dist;
            stress += error * error;
        }

        float trustworthiness = 1.0f - stress / (n_samples * n_samples);

        if (trustworthiness < 0.7f && current_dim < 10) {
            archive->diresa_dim.fetch_add(1);
        } else if (trustworthiness > 0.9f && current_dim > 2) {
            archive->diresa_dim.fetch_sub(1);
        }
    }
}

__global__ void garbage_collect_kernel(CVTArchive* archive) {
    int cell_id = blockIdx.x;
    if (cell_id >= archive->n_cells) return;

    VoronoiCell* cell = &archive->cells[cell_id];
    int count = cell->elite_count.load();

    float current_time = (float)clock64();
    float max_age = 1e10f;

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        Elite* e = cell->elites[i];
        float age = current_time - e->creation_time;
        float staleness = current_time - e->last_access_time;

        float survival_score = e->fitness * e->coherence * expf(-staleness / 1e9f);

        if (age > max_age && survival_score < 0.1f) {
            cudaFree(e->delta_chain);
            cudaFree(e->genome);

            if (i < count - 1) {
                *e = *cell->elites[count - 1];
            }

            cudaFree(cell->elites[count - 1]);

            if (threadIdx.x == 0) {
                cell->elite_count.fetch_sub(1);
                archive->total_elites.fetch_sub(1);
            }
        }
    }
}

__global__ void compute_coverage_kernel(CVTArchive* archive, float* coverage_out) {
    __shared__ int occupied;
    __shared__ float quality_sum;

    if (threadIdx.x == 0) {
        occupied = 0;
        quality_sum = 0.0f;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < archive->n_cells; i += blockDim.x) {
        VoronoiCell* cell = &archive->cells[i];
        int count = cell->elite_count.load();

        if (count > 0) {
            atomicAdd(&occupied, 1);

            float cell_quality = 0.0f;
            for (int j = 0; j < count; j++) {
                cell_quality = fmaxf(cell_quality, cell->elites[j]->fitness);
            }
            atomicAdd(&quality_sum, cell_quality);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float coverage = (float)occupied / archive->n_cells;
        float avg_quality = quality_sum / (occupied + 1);
        *coverage_out = coverage * avg_quality;
        archive->total_coverage.store(coverage);
    }
}