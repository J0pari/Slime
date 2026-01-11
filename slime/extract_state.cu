
#ifndef EXTRACT_STATE_CU
#define EXTRACT_STATE_CU

#include "config/config.cu"
#include <stdio.h>
#include <cuda_runtime.h>
#include "core/organism.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
\
            return; \
        } \
    } while(0)

struct SystemSnapshot {

    float* chemical_concentration;
    float* chemical_gradient_x;
    float* chemical_gradient_y;
    float* chemical_laplacian;

    float* ca_concentration;
    float* ca_output;

    float* agent_positions;
    float* agent_velocities;
    float* agent_behavioral_coords;
    int* agent_alive;

    float* voronoi_centroids;
    int* voronoi_density;
    float* voronoi_radius;
    int* voronoi_best_elite;

    int archive_size;
    float* archive_fitness;
    float* archive_coherence;
    uint64_t* archive_hashes;
    float* archive_behavioral_coords;
    float* archive_raw_metrics;
    uint32_t* archive_parent_ids;
    uint32_t* archive_compressed_size;
    uint16_t* archive_generation;

    int memory_count;
    float* memory_timestamps;
    float* memory_decay;

    int pool_active;
    float* pool_fitness;
    float* pool_coherence;
    float* pool_hunger;
    float* pool_genome_sample;
};

extern "C" void extract_system_state(Organism* d_organism, FILE* json_file, int generation, double elapsed_time) {
    Organism h_organism;
    CUDA_CHECK(cudaMemcpy(&h_organism, d_organism, sizeof(Organism), cudaMemcpyDeviceToHost));

    ComponentPool h_pool;
    CUDA_CHECK(cudaMemcpy(&h_pool, h_organism.pool, sizeof(ComponentPool), cudaMemcpyDeviceToHost));

    int grid_size = h_pool.entries[0].grid_size;

    fprintf(json_file, "{\"gen\":%d,\"elapsed\":%.2f,", generation, elapsed_time);

    int sample_size = WMMA_TILE_DIM;
    int offset = (grid_size - sample_size) / 2;
    float* h_chemical = new float[sample_size * sample_size];

    ChemicalField h_chem_field;
    CUDA_CHECK(cudaMemcpy(&h_chem_field, h_organism.chemical_field, sizeof(ChemicalField), cudaMemcpyDeviceToHost));

    fprintf(json_file, "\"chemical\":{\"concentration\":[");
    for (int y = 0; y < sample_size; y++) {
        for (int x = 0; x < sample_size; x++) {
            float val;
            CUDA_CHECK(cudaMemcpy(&val, h_chem_field.concentration + (offset + y) * grid_size + (offset + x), sizeof(float), cudaMemcpyDeviceToHost));
            fprintf(json_file, "%.4f", val);
            if (y < sample_size - 1 || x < sample_size - 1) fprintf(json_file, ",");
        }
    }
    fprintf(json_file, "],\"size\":%d},", sample_size);


    BehavioralState* h_agents = new BehavioralState[h_pool.active_count.load()];
    CUDA_CHECK(cudaMemcpy(h_agents, h_organism.behavioral_agents, h_pool.active_count.load() * sizeof(BehavioralState), cudaMemcpyDeviceToHost));

    fprintf(json_file, "\"agents\":[");
    for (int i = 0; i < h_pool.active_count.load(); i++) {
        fprintf(json_file, "{\"pos\":[%.3f,%.3f],\"vel\":[%.3f,%.3f],\"bc\":[",
                h_agents[i].position[0], h_agents[i].position[1],
                h_agents[i].velocity[0], h_agents[i].velocity[1]);
        for (int j = 0; j < BEHAVIORAL_DIM; j++) {
            fprintf(json_file, "%.3f", h_agents[i].behavioral_coords[j]);
            if (j < BEHAVIORAL_DIM - 1) fprintf(json_file, ",");
        }
        fprintf(json_file, "]}");
        if (i < h_pool.active_count.load() - 1) fprintf(json_file, ",");
    }
    fprintf(json_file, "],");
    delete[] h_agents;

    int voronoi_sample = min(VORONOI_EXPORT_LIMIT, h_organism.num_voronoi_cells);
    VoronoiCell* h_voronoi = new VoronoiCell[voronoi_sample];
    CUDA_CHECK(cudaMemcpy(h_voronoi, h_organism.voronoi_cells, voronoi_sample * sizeof(VoronoiCell), cudaMemcpyDeviceToHost));

    fprintf(json_file, "\"voronoi\":[");
    for (int i = 0; i < voronoi_sample; i++) {
        fprintf(json_file, "{\"density\":%d,\"radius\":%.3f,\"centroid\":[",
                h_voronoi[i].density, h_voronoi[i].radius);
        for (int j = 0; j < 3; j++) {
            fprintf(json_file, "%.3f", h_voronoi[i].centroid[j]);
            if (j < 2) fprintf(json_file, ",");
        }
        fprintf(json_file, "]}");
        if (i < voronoi_sample - 1) fprintf(json_file, ",");
    }
    fprintf(json_file, "],");
    delete[] h_voronoi;

    int archive_sample = min(BEHAVIORAL_DIM, h_organism.archive_size);
    if (archive_sample > 0) {
        GPUElite* h_archive = new GPUElite[archive_sample];
        CUDA_CHECK(cudaMemcpy(h_archive, h_organism.archive, archive_sample * sizeof(GPUElite), cudaMemcpyDeviceToHost));

        fprintf(json_file, "\"archive\":{\"size\":%d,\"elites\":[", h_organism.archive_size);
        for (int i = 0; i < archive_sample; i++) {
            fprintf(json_file, "{\"f\":%.4f,\"c\":%.4f,\"rank\":%.4f,\"gen\":%d,\"hash\":%llu,\"parents\":[%u,%u],\"compressed_bytes\":%u,\"bc\":[",
                    h_archive[i].fitness,
                    h_archive[i].coherence,
                    h_archive[i].effective_rank,
                    (int)h_archive[i].generation,
                    (unsigned long long)h_archive[i].genome_hash,
                    h_archive[i].parent_ids[0],
                    h_archive[i].parent_ids[1],
                    h_archive[i].compressed_size);
            for (int j = 0; j < BEHAVIORAL_DIM; j++) {
                fprintf(json_file, "%.3f", h_archive[i].behavioral_coords[j]);
                if (j < BEHAVIORAL_DIM - 1) fprintf(json_file, ",");
            }
            fprintf(json_file, "],\"hardware_features\":[");
            for (int j = 0; j < (WMMA_TILE_DIM - 1); j++) {
                fprintf(json_file, "%.3f", h_archive[i].hardware_features[j]);
                if (j < (HARDWARE_FEATURES_DIM - 1)) fprintf(json_file, ",");
            }
            fprintf(json_file, "]}");
            if (i < archive_sample - 1) fprintf(json_file, ",");
        }
        fprintf(json_file, "]}},");
        delete[] h_archive;
    } else {
        fprintf(json_file, "\"archive\":{\"size\":0,\"elites\":[]}},");
    }

    fprintf(json_file, "\"pool\":{\"active\":%d,\"capacity\":%d,\"spawned\":%d,\"culled\":%d}",
            h_pool.active_count.load(), h_pool.capacity,
            h_pool.total_spawned.load(), h_pool.total_culled.load());

    fprintf(json_file, "}\n");
    fflush(json_file);

    delete[] h_chemical;
}

#endif
