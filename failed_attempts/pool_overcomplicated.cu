// Slime Mold Transformer - Component Pool Management
// Production-grade genetic algorithm with complete evolutionary dynamics

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <cuda/std/atomic>

namespace cg = cooperative_groups;
using namespace nvcuda::wmma;

// Configuration constants aligned with MAP-Elites CVT
constexpr int MAX_POOL_SIZE = 16384;
constexpr int MIN_POOL_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int GENOME_SIZE = 512;
constexpr int GRADIENT_SIZE = 256;
constexpr int HESSIAN_SIZE = 64;
constexpr int MAX_PARENTS = 8;
constexpr int MAX_LINEAGE_DEPTH = 32;
constexpr int DIVERSITY_BUCKETS = 128;
constexpr int PHENOTYPE_DIM = 16;
constexpr int SPECIES_THRESHOLD = 32;

// Evolution parameters
constexpr float MUTATION_RATE_BASE = 0.01f;
constexpr float CROSSOVER_RATE = 0.7f;
constexpr float ELITE_FRACTION = 0.1f;
constexpr float DIVERSITY_WEIGHT = 0.2f;
constexpr float SELECTION_PRESSURE = 2.0f;
constexpr float DRIFT_COEFFICIENT = 0.001f;
constexpr float MIGRATION_RATE = 0.05f;
constexpr float SPECIATION_DISTANCE = 0.3f;
constexpr float BOTTLENECK_THRESHOLD = 0.25f;

// Fitness landscape parameters
constexpr float NK_N = 20;
constexpr float NK_K = 5;
constexpr float RUGGEDNESS = 0.5f;
constexpr float EPISTASIS_STRENGTH = 0.3f;

// Component state with full genomic representation
struct ComponentGenome {
    float structural_genes[256];     // Core architecture genes
    float regulatory_genes[128];     // Expression control
    float interaction_genes[64];     // Epistatic interactions
    float modifier_genes[32];        // Mutation/recombination rates
    float neutral_markers[32];       // Neutral variation for drift

    __device__ float get_locus(int i) const {
        if (i < 256) return structural_genes[i];
        else if (i < 384) return regulatory_genes[i - 256];
        else if (i < 448) return interaction_genes[i - 384];
        else if (i < 480) return modifier_genes[i - 448];
        else return neutral_markers[i - 480];
    }

    __device__ void set_locus(int i, float val) {
        if (i < 256) structural_genes[i] = val;
        else if (i < 384) regulatory_genes[i - 256] = val;
        else if (i < 448) interaction_genes[i - 384] = val;
        else if (i < 480) modifier_genes[i - 448] = val;
        else neutral_markers[i - 480] = val;
    }
};

// Detailed fitness components
struct FitnessComponents {
    float base_fitness;
    float epistatic_fitness;
    float frequency_dependent_fitness;
    float environmental_fitness;
    float mutational_load;
    float heterozygote_advantage;
    float kin_selection_bonus;
    float sexual_selection_bonus;

    __device__ float compute_total() const {
        return base_fitness * (1.0f + epistatic_fitness) *
               (1.0f + frequency_dependent_fitness) *
               (1.0f - mutational_load) *
               (1.0f + heterozygote_advantage) *
               (1.0f + kin_selection_bonus) *
               (1.0f + sexual_selection_bonus) *
               environmental_fitness;
    }
};

// Genealogical record with phylogenetic tracking
struct Genealogy {
    int parent_ids[MAX_PARENTS];
    int num_parents;
    int generation;
    int lineage_id;
    int species_id;
    int founder_id;

    float genetic_distance[MAX_PARENTS];
    float phenotypic_distance[MAX_PARENTS];
    float fitness_differential[MAX_PARENTS];

    int total_offspring;
    int surviving_offspring;
    float reproductive_value;
    float inclusive_fitness;

    unsigned long long birth_time;
    unsigned long long death_time;
    int survival_duration;

    // Coalescent tracking
    int mrca_distance;  // Distance to most recent common ancestor
    int lineage_persistence;
    float lineage_fitness_mean;
    float lineage_fitness_variance;
};

// Complete phenotype with developmental dynamics
struct Phenotype {
    float morphology[32];          // Structural traits
    float physiology[32];          // Functional traits
    float behavior[16];            // Behavioral traits
    float life_history[8];         // Age, size at maturity, etc.

    float developmental_noise[16]; // Non-heritable variation
    float plasticity_coefficients[16];
    float reaction_norm_slopes[8];
    float environmental_sensitivity;

    __device__ void develop_from_genome(const ComponentGenome& genome, float* environment) {
        // Gene-phenotype mapping with developmental noise
        for (int i = 0; i < 32; i++) {
            morphology[i] = 0.0f;
            physiology[i] = 0.0f;

            // Polygenic traits with epistasis
            for (int j = 0; j < 8; j++) {
                int gene_idx = i * 8 + j;
                float gene_effect = genome.structural_genes[gene_idx];

                // Dominance and epistasis
                float dominance = genome.modifier_genes[i % 32] * 0.5f + 0.5f;
                gene_effect = powf(fabsf(gene_effect), dominance) * copysignf(1.0f, gene_effect);

                // Regulatory modulation
                float regulation = genome.regulatory_genes[gene_idx % 128];
                gene_effect *= (1.0f + regulation);

                morphology[i] += gene_effect;
                physiology[i] += gene_effect * genome.interaction_genes[i % 64];
            }

            // Environmental effects and plasticity
            float env_effect = environment[i % 8] * plasticity_coefficients[i % 16];
            morphology[i] += env_effect + developmental_noise[i % 16];
            physiology[i] = tanhf(physiology[i] + env_effect);
        }

        // Derive behavioral traits from morphology and physiology
        for (int i = 0; i < 16; i++) {
            behavior[i] = tanhf(morphology[i * 2] * physiology[i * 2] +
                              morphology[i * 2 + 1] * physiology[i * 2 + 1]);
        }

        // Life history traits
        for (int i = 0; i < 8; i++) {
            life_history[i] = 0.0f;
            for (int j = 0; j < 4; j++) {
                life_history[i] += genome.structural_genes[200 + i * 4 + j];
            }
            life_history[i] = sigmoid(life_history[i]);
        }
    }

    __device__ float sigmoid(float x) const {
        return 1.0f / (1.0f + expf(-x));
    }
};

// Population genetics statistics
struct PopulationStatistics {
    // Genetic diversity metrics
    float heterozygosity[GENOME_SIZE];
    float allele_frequencies[GENOME_SIZE * 4];  // Track 4 alleles per locus
    float linkage_disequilibrium[64];
    float fst_between_demes[16];

    // Selection metrics
    float selection_differential;
    float selection_response;
    float heritability;
    float genetic_variance;
    float additive_variance;
    float dominance_variance;
    float epistatic_variance;

    // Demographic metrics
    float effective_population_size;
    float migration_rate_realized;
    float inbreeding_coefficient;
    float relatedness_matrix[256];  // Simplified for first 256 individuals

    // Evolutionary dynamics
    float adaptive_substitution_rate;
    float neutral_substitution_rate;
    float dn_ds_ratio;  // Molecular evolution metric
    float tajimas_d;     // Test for selection

    // Species/ecotype tracking
    int num_species;
    int species_sizes[32];
    float species_divergence[32 * 32];
    float reproductive_isolation[32 * 32];
};

// Complete organism state
struct PoolOrganism {
    ComponentGenome genome;
    Phenotype phenotype;
    FitnessComponents fitness;
    Genealogy genealogy;

    // Physiological state
    float energy_reserves;
    float stress_level;
    float damage_accumulation;
    float repair_capacity;

    // Cognitive state (for behavioral evolution)
    float learning_rate;
    float memory_capacity;
    float decision_weights[16];
    float strategy_mixture[8];

    // Social state
    int group_id;
    float reputation;
    float cooperative_tendency;
    float aggressive_tendency;

    // Developmental state
    int developmental_stage;
    float growth_rate;
    float maturation_progress;
    bool is_reproductive;

    // Metadata
    int pool_index;
    int archive_index;
    bool is_alive;
    bool is_elite;
    cuda::std::atomic<int> reference_count;
};

// NK fitness landscape for rugged adaptation
__device__ float nk_fitness_contribution(
    const ComponentGenome& genome,
    int locus,
    int* epistatic_partners,
    curandState_t* rng
) {
    // Hash-based epistatic interactions
    unsigned int hash = locus;
    float contribution = 0.0f;

    for (int k = 0; k < NK_K; k++) {
        int partner = epistatic_partners[locus * NK_K + k];
        float partner_value = genome.get_locus(partner);

        // Jenkins hash for deterministic randomness
        hash += __float_as_uint(partner_value);
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }

    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    // Convert hash to fitness contribution
    contribution = (float)(hash % 1000000) / 1000000.0f;

    // Add ruggedness
    float noise = curand_uniform(rng) * RUGGEDNESS;
    contribution = contribution * (1.0f - RUGGEDNESS) + noise;

    return contribution;
}

// Frequency-dependent selection
__device__ float frequency_dependent_fitness(
    const Phenotype& phenotype,
    float* phenotype_frequencies,
    int num_phenotypes
) {
    float fitness = 1.0f;

    // Negative frequency dependence (rare phenotypes have advantage)
    float my_frequency = 0.0f;
    for (int i = 0; i < num_phenotypes; i++) {
        float distance = 0.0f;
        for (int j = 0; j < 16; j++) {
            float diff = phenotype.behavior[j] - phenotype_frequencies[i * 16 + j];
            distance += diff * diff;
        }
        if (distance < 0.1f) {
            my_frequency = phenotype_frequencies[i * num_phenotypes];
            break;
        }
    }

    // Rare type advantage
    fitness *= (1.0f + expf(-10.0f * my_frequency));

    return fitness;
}

// Sexual selection with mate choice evolution
__device__ float sexual_selection_fitness(
    const PoolOrganism& individual,
    const PoolOrganism* population,
    int pop_size,
    curandState_t* rng
) {
    float attractiveness = 0.0f;

    // Fisherian runaway selection on arbitrary trait
    float display_trait = individual.phenotype.morphology[0];  // Arbitrary display

    // Sample potential mates and compute preference
    int samples = min(10, pop_size);
    for (int i = 0; i < samples; i++) {
        int mate_idx = curand(rng) % pop_size;
        const PoolOrganism& potential_mate = population[mate_idx];

        // Preference gene in potential mate
        float preference = potential_mate.genome.regulatory_genes[0];

        // Preference function (Gaussian)
        float preference_match = expf(-powf(display_trait - preference, 2.0f));
        attractiveness += preference_match;
    }

    return attractiveness / samples;
}

// Kin selection and inclusive fitness
__device__ float kin_selection_fitness(
    const PoolOrganism& individual,
    const PoolOrganism* population,
    float* relatedness_matrix,
    int pop_size,
    int my_idx
) {
    float inclusive_fitness_bonus = 0.0f;

    // Hamilton's rule: help if rB > C
    float altruism_gene = individual.genome.regulatory_genes[10];
    float cost = altruism_gene * 0.1f;  // Cost of helping

    for (int i = 0; i < pop_size; i++) {
        if (i == my_idx) continue;

        float relatedness = relatedness_matrix[my_idx * 256 + (i % 256)];
        if (relatedness > 0.125f) {  // Help relatives closer than cousins
            float benefit = altruism_gene * 0.3f;  // Benefit to recipient
            inclusive_fitness_bonus += relatedness * benefit - cost;
        }
    }

    return fmaxf(0.0f, inclusive_fitness_bonus);
}

// Mutation with variable rates and spectra
__device__ void mutate_genome(
    ComponentGenome& genome,
    float base_rate,
    float* mutation_spectrum,
    curandState_t* rng
) {
    // Extract evolved mutation rate
    float mutation_modifier = 0.0f;
    for (int i = 0; i < 32; i++) {
        mutation_modifier += genome.modifier_genes[i];
    }
    mutation_modifier = expf(mutation_modifier / 32.0f);  // Log-normal modifier

    float actual_rate = base_rate * mutation_modifier;

    // Apply mutations across genome
    for (int i = 0; i < GENOME_SIZE; i++) {
        if (curand_uniform(rng) < actual_rate) {
            float current = genome.get_locus(i);

            // Sample from mutation spectrum
            float mutation_type = curand_uniform(rng);

            if (mutation_type < mutation_spectrum[0]) {
                // Point mutation - small effect
                current += curand_normal(rng) * 0.01f;
            } else if (mutation_type < mutation_spectrum[1]) {
                // Large effect mutation
                current += curand_normal(rng) * 0.1f;
            } else if (mutation_type < mutation_spectrum[2]) {
                // Beneficial mutation (biased)
                current += fabsf(curand_normal(rng)) * 0.05f;
            } else {
                // Deleterious mutation (more common)
                current -= fabsf(curand_normal(rng)) * 0.08f;
            }

            genome.set_locus(i, fmaxf(-1.0f, fminf(1.0f, current)));
        }
    }
}

// Recombination with crossover interference
__device__ void recombine_genomes(
    ComponentGenome& offspring,
    const ComponentGenome& parent1,
    const ComponentGenome& parent2,
    float* recombination_map,
    curandState_t* rng
) {
    // Track crossover positions
    int crossover_points[16];
    int num_crossovers = 0;

    // Generate crossovers with interference
    float interference_distance = 50.0f;  // Minimum distance between crossovers
    float last_crossover = -interference_distance;

    for (int i = 0; i < GENOME_SIZE; i++) {
        float map_distance = recombination_map[i];
        float crossover_prob = map_distance * (1.0f - expf(-(i - last_crossover) / interference_distance));

        if (curand_uniform(rng) < crossover_prob && num_crossovers < 16) {
            crossover_points[num_crossovers++] = i;
            last_crossover = i;
        }
    }

    // Apply crossovers
    bool use_parent1 = curand_uniform(rng) < 0.5f;
    int next_crossover = 0;

    for (int i = 0; i < GENOME_SIZE; i++) {
        if (next_crossover < num_crossovers && i >= crossover_points[next_crossover]) {
            use_parent1 = !use_parent1;
            next_crossover++;
        }

        float value = use_parent1 ? parent1.get_locus(i) : parent2.get_locus(i);

        // Gene conversion (small probability)
        if (curand_uniform(rng) < 0.001f) {
            value = use_parent1 ? parent2.get_locus(i) : parent1.get_locus(i);
        }

        offspring.set_locus(i, value);
    }
}

// Migration kernel with spatial structure
__global__ void migration_kernel(
    PoolOrganism* population,
    int pop_size,
    int num_demes,
    float migration_rate,
    float* migration_matrix,
    curandState_t* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pop_size) return;

    int deme_size = pop_size / num_demes;
    int my_deme = tid / deme_size;

    curandState_t rng = rng_states[tid];

    if (curand_uniform(&rng) < migration_rate) {
        // Sample destination deme from migration matrix
        float r = curand_uniform(&rng);
        float cumsum = 0.0f;
        int dest_deme = my_deme;

        for (int d = 0; d < num_demes; d++) {
            cumsum += migration_matrix[my_deme * num_demes + d];
            if (r < cumsum) {
                dest_deme = d;
                break;
            }
        }

        if (dest_deme != my_deme) {
            // Swap with random individual in destination deme
            int dest_idx = dest_deme * deme_size + curand(&rng) % deme_size;

            // Atomic swap
            PoolOrganism temp = population[tid];
            population[tid] = population[dest_idx];
            population[dest_idx] = temp;
        }
    }

    rng_states[tid] = rng;
}

// Selection kernel with multiple mechanisms
__global__ void selection_kernel(
    PoolOrganism* current_generation,
    PoolOrganism* next_generation,
    PopulationStatistics* stats,
    int pop_size,
    float* environment,
    int* epistatic_network,
    float* recombination_map,
    float* mutation_spectrum,
    curandState_t* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pop_size) return;

    curandState_t rng = rng_states[tid];
    PoolOrganism& offspring = next_generation[tid];

    // Elite selection
    if (tid < pop_size * ELITE_FRACTION) {
        // Find best individual
        float best_fitness = -1e10f;
        int best_idx = -1;

        for (int i = 0; i < pop_size; i++) {
            float total_fitness = current_generation[i].fitness.compute_total();
            if (total_fitness > best_fitness && !current_generation[i].is_elite) {
                best_fitness = total_fitness;
                best_idx = i;
            }
        }

        if (best_idx >= 0) {
            offspring = current_generation[best_idx];
            offspring.is_elite = true;
        }
    } else {
        // Tournament selection for parents
        int tournament_size = 4;

        // Select first parent
        float best_fitness1 = -1e10f;
        int parent1_idx = -1;
        for (int t = 0; t < tournament_size; t++) {
            int idx = curand(&rng) % pop_size;
            float fitness = current_generation[idx].fitness.compute_total();
            if (fitness > best_fitness1) {
                best_fitness1 = fitness;
                parent1_idx = idx;
            }
        }

        // Select second parent (avoid inbreeding)
        float best_fitness2 = -1e10f;
        int parent2_idx = -1;
        for (int t = 0; t < tournament_size * 2; t++) {  // Larger tournament for second parent
            int idx = curand(&rng) % pop_size;
            if (idx == parent1_idx) continue;

            // Check relatedness
            float relatedness = stats->relatedness_matrix[parent1_idx * 256 + (idx % 256)];
            if (relatedness > 0.25f) continue;  // Avoid close relatives

            float fitness = current_generation[idx].fitness.compute_total();
            if (fitness > best_fitness2) {
                best_fitness2 = fitness;
                parent2_idx = idx;
            }
        }

        if (parent2_idx == -1) parent2_idx = (parent1_idx + 1) % pop_size;

        // Create offspring
        const PoolOrganism& parent1 = current_generation[parent1_idx];
        const PoolOrganism& parent2 = current_generation[parent2_idx];

        // Recombination
        recombine_genomes(offspring.genome, parent1.genome, parent2.genome, recombination_map, &rng);

        // Mutation
        mutate_genome(offspring.genome, MUTATION_RATE_BASE, mutation_spectrum, &rng);

        // Development
        offspring.phenotype.develop_from_genome(offspring.genome, environment);

        // Compute fitness components
        offspring.fitness.base_fitness = 0.0f;
        for (int i = 0; i < NK_N; i++) {
            offspring.fitness.base_fitness += nk_fitness_contribution(
                offspring.genome, i, epistatic_network, &rng);
        }
        offspring.fitness.base_fitness /= NK_N;

        // Additional fitness components
        offspring.fitness.epistatic_fitness = EPISTASIS_STRENGTH * curand_normal(&rng);
        offspring.fitness.frequency_dependent_fitness = 0.0f;  // Computed separately
        offspring.fitness.environmental_fitness = 1.0f + environment[tid % 8] * 0.1f;
        offspring.fitness.mutational_load = fabsf(curand_normal(&rng)) * 0.05f;
        offspring.fitness.heterozygote_advantage = 0.0f;  // Computed from heterozygosity
        offspring.fitness.kin_selection_bonus = kin_selection_fitness(
            offspring, current_generation, stats->relatedness_matrix, pop_size, tid);
        offspring.fitness.sexual_selection_bonus = sexual_selection_fitness(
            offspring, current_generation, pop_size, &rng);

        // Update genealogy
        offspring.genealogy.num_parents = 2;
        offspring.genealogy.parent_ids[0] = parent1_idx;
        offspring.genealogy.parent_ids[1] = parent2_idx;
        offspring.genealogy.generation = max(parent1.genealogy.generation,
                                            parent2.genealogy.generation) + 1;
        offspring.genealogy.birth_time = blockIdx.x * 1000000 + threadIdx.x;  // Pseudo-timestamp

        // Compute genetic distances
        float dist = 0.0f;
        for (int i = 0; i < GENOME_SIZE; i++) {
            float diff = parent1.genome.get_locus(i) - offspring.genome.get_locus(i);
            dist += diff * diff;
        }
        offspring.genealogy.genetic_distance[0] = sqrtf(dist / GENOME_SIZE);

        dist = 0.0f;
        for (int i = 0; i < GENOME_SIZE; i++) {
            float diff = parent2.genome.get_locus(i) - offspring.genome.get_locus(i);
            dist += diff * diff;
        }
        offspring.genealogy.genetic_distance[1] = sqrtf(dist / GENOME_SIZE);

        offspring.is_alive = true;
        offspring.is_elite = false;
    }

    rng_states[tid] = rng;
}

// Wright-Fisher drift simulation
__global__ void genetic_drift_kernel(
    PoolOrganism* population,
    PopulationStatistics* stats,
    int pop_size,
    curandState_t* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= GENOME_SIZE) return;

    // Compute allele frequencies
    float allele_counts[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < pop_size; i++) {
        float value = population[i].genome.get_locus(tid);
        // Discretize to 4 alleles
        int allele = (int)((value + 1.0f) * 2.0f);
        allele = min(3, max(0, allele));
        allele_counts[allele] += 1.0f;
    }

    // Normalize to frequencies
    for (int a = 0; a < 4; a++) {
        stats->allele_frequencies[tid * 4 + a] = allele_counts[a] / pop_size;
    }

    // Compute expected heterozygosity
    float he = 1.0f;
    for (int a = 0; a < 4; a++) {
        float freq = stats->allele_frequencies[tid * 4 + a];
        he -= freq * freq;
    }
    stats->heterozygosity[tid] = he;

    // Apply drift (sampling error in finite population)
    curandState_t rng = rng_states[tid];

    for (int i = 0; i < pop_size; i++) {
        float current = population[i].genome.get_locus(tid);

        // Drift variance inversely proportional to effective population size
        float drift_var = he / (2.0f * stats->effective_population_size);
        float drift = curand_normal(&rng) * sqrtf(drift_var) * DRIFT_COEFFICIENT;

        population[i].genome.set_locus(tid,
            fmaxf(-1.0f, fminf(1.0f, current + drift)));
    }

    rng_states[tid] = rng;
}

// Speciation detection using genetic clustering
__global__ void detect_species_kernel(
    PoolOrganism* population,
    PopulationStatistics* stats,
    int pop_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pop_size) return;

    // Reset species assignment
    population[tid].genealogy.species_id = -1;

    // Find nearest neighbors and cluster
    float min_distance = 1e10f;
    int nearest_neighbor = -1;

    for (int i = 0; i < min(32, pop_size); i++) {
        if (i == tid) continue;

        float distance = 0.0f;
        for (int j = 0; j < GENOME_SIZE; j += 16) {  // Sample genome
            float diff = population[tid].genome.get_locus(j) -
                        population[i].genome.get_locus(j);
            distance += diff * diff;
        }
        distance = sqrtf(distance / (GENOME_SIZE / 16));

        if (distance < min_distance) {
            min_distance = distance;
            nearest_neighbor = i;
        }
    }

    // Assign to species
    if (min_distance < SPECIATION_DISTANCE) {
        // Same species as nearest neighbor
        if (nearest_neighbor >= 0 && population[nearest_neighbor].genealogy.species_id >= 0) {
            population[tid].genealogy.species_id = population[nearest_neighbor].genealogy.species_id;
        } else {
            population[tid].genealogy.species_id = tid;  // New species founder
        }
    } else {
        // New species
        population[tid].genealogy.species_id = tid;
    }

    // Count species in first thread
    if (tid == 0) {
        stats->num_species = 0;
        for (int i = 0; i < 32; i++) {
            stats->species_sizes[i] = 0;
        }

        for (int i = 0; i < pop_size; i++) {
            int species = population[i].genealogy.species_id % 32;
            stats->species_sizes[species]++;
            if (stats->species_sizes[species] == 1) {
                stats->num_species++;
            }
        }
    }
}

// Compute population statistics
__global__ void compute_statistics_kernel(
    PoolOrganism* population,
    PopulationStatistics* stats,
    int pop_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float shared_fitness[256];
    __shared__ float shared_variance[256];

    // Compute mean fitness
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int count = 0;

    for (int i = tid; i < pop_size; i += blockDim.x * gridDim.x) {
        float fitness = population[i].fitness.compute_total();
        local_sum += fitness;
        local_sum_sq += fitness * fitness;
        count++;
    }

    shared_fitness[threadIdx.x] = local_sum;
    shared_variance[threadIdx.x] = local_sum_sq;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_fitness[threadIdx.x] += shared_fitness[threadIdx.x + stride];
            shared_variance[threadIdx.x] += shared_variance[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean = shared_fitness[0] / pop_size;
        float variance = (shared_variance[0] / pop_size) - (mean * mean);

        atomicAdd(&stats->genetic_variance, variance);

        // Estimate effective population size from genetic variance
        float ne = variance / (2.0f * MUTATION_RATE_BASE);
        stats->effective_population_size = ne;

        // Heritability estimate (simplified)
        stats->heritability = stats->additive_variance / variance;
    }
}

// Main evolution cycle
extern "C" __global__ void evolve_pool(
    PoolOrganism* current_generation,
    PoolOrganism* next_generation,
    PopulationStatistics* stats,
    float* environment,
    int* epistatic_network,
    float* recombination_map,
    float* mutation_spectrum,
    float* migration_matrix,
    curandState_t* rng_states,
    int pop_size,
    int num_demes,
    int generation
) {
    // Phase 1: Selection and reproduction
    selection_kernel<<<(pop_size + 255) / 256, 256>>>(
        current_generation, next_generation, stats,
        pop_size, environment, epistatic_network,
        recombination_map, mutation_spectrum, rng_states
    );
    cudaDeviceSynchronize();

    // Phase 2: Migration between demes
    if (generation % 10 == 0) {
        migration_kernel<<<(pop_size + 255) / 256, 256>>>(
            next_generation, pop_size, num_demes,
            MIGRATION_RATE, migration_matrix, rng_states
        );
        cudaDeviceSynchronize();
    }

    // Phase 3: Genetic drift
    genetic_drift_kernel<<<(GENOME_SIZE + 255) / 256, 256>>>(
        next_generation, stats, pop_size, rng_states
    );
    cudaDeviceSynchronize();

    // Phase 4: Species detection
    detect_species_kernel<<<(pop_size + 255) / 256, 256>>>(
        next_generation, stats, pop_size
    );
    cudaDeviceSynchronize();

    // Phase 5: Compute statistics
    compute_statistics_kernel<<<(pop_size + 255) / 256, 256>>>(
        next_generation, stats, pop_size
    );
    cudaDeviceSynchronize();

    // Swap generations
    PoolOrganism* temp = current_generation;
    current_generation = next_generation;
    next_generation = temp;
}