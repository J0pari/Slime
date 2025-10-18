// Extract actual GPU system state to JSON
#ifndef EXTRACT_STATE_CU
#define EXTRACT_STATE_CU

#include <stdio.h>
#include <cuda_runtime.h>
#include "core/organism.cu"

struct SystemSnapshot {
    // Chemical field spatial patterns
    float* chemical_concentration;  // GRID_SIZE * GRID_SIZE
    float* chemical_gradient_x;
    float* chemical_gradient_y;
    float* chemical_laplacian;
    
    // Multi-head CA state
    float* ca_input;                // GRID_SIZE * GRID_SIZE * CHANNELS
    float* ca_output;
    float* flow_kernels;            // NUM_HEADS * 9
    float* mass_buffer;             // NUM_HEADS
    
    // Behavioral agents
    float* agent_positions;         // MAX_COMPONENTS * 2
    float* agent_velocities;        // MAX_COMPONENTS * 2
    float* agent_behavioral_coords; // MAX_COMPONENTS * 10
    int* agent_alive;               // MAX_COMPONENTS
    
    // Voronoi tessellation
    float* voronoi_centroids;       // MAX_CELLS * 10
    int* voronoi_density;           // MAX_CELLS
    float* voronoi_radius;          // MAX_CELLS
    int* voronoi_best_elite;        // MAX_CELLS
    
    // Archive
    int archive_size;
    float* archive_fitness;         // archive_size
    float* archive_coherence;       // archive_size
    uint64_t* archive_hashes;       // archive_size
    float* archive_behavioral_coords; // archive_size * 10
    float* archive_raw_metrics;     // archive_size * 75
    uint32_t* archive_parent_ids;   // archive_size * 2
    uint32_t* archive_compressed_size; // archive_size
    uint16_t* archive_generation;   // archive_size
    
    // Memory tubes
    int memory_count;
    float* memory_timestamps;       // memory_count
    float* memory_decay;            // memory_count
    
    // Component pool
    int pool_active;
    float* pool_fitness;            // pool_active
    float* pool_coherence;          // pool_active
    float* pool_hunger;             // pool_active
    float* pool_genome_sample;      // pool_active * 32 (first 32 genome elements)
};

extern "C" void extract_system_state(Organism* d_organism, FILE* json_file, int generation, double elapsed_time) {
    Organism h_organism;
    cudaMemcpy(&h_organism, d_organism, sizeof(Organism), cudaMemcpyDeviceToHost);
    
    ComponentPool h_pool;
    cudaMemcpy(&h_pool, h_organism.pool, sizeof(ComponentPool), cudaMemcpyDeviceToHost);
    
    fprintf(json_file, "{\"gen\":%d,\"elapsed\":%.2f,", generation, elapsed_time);
    
    // Chemical field - sample 16x16 grid from center
    int sample_size = 16;
    int offset = (GRID_SIZE - sample_size) / 2;
    float* h_chemical = new float[sample_size * sample_size];
    
    ChemicalField h_chem_field;
    cudaMemcpy(&h_chem_field, h_organism.chemical_field, sizeof(ChemicalField), cudaMemcpyDeviceToHost);
    
    fprintf(json_file, "\"chemical\":{\"concentration\":[");
    for (int y = 0; y < sample_size; y++) {
        for (int x = 0; x < sample_size; x++) {
            float val;
            cudaMemcpy(&val, h_chem_field.concentration + (offset + y) * GRID_SIZE + (offset + x), sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(json_file, "%.4f", val);
            if (y < sample_size - 1 || x < sample_size - 1) fprintf(json_file, ",");
        }
    }
    fprintf(json_file, "],\"size\":%d},", sample_size);
    
    // Multi-head CA state
    MultiHeadCAState h_ca;
    cudaMemcpy(&h_ca, h_organism.ca_state, sizeof(MultiHeadCAState), cudaMemcpyDeviceToHost);
    
    float* h_flow = new float[NUM_HEADS * 9];
    cudaMemcpy(h_flow, h_ca.flow_kernels, NUM_HEADS * 9 * sizeof(float), cudaMemcpyDeviceToHost);
    
    fprintf(json_file, "\"ca\":{\"flow_kernels\":[");
    for (int h = 0; h < NUM_HEADS; h++) {
        fprintf(json_file, "[");
        for (int k = 0; k < 9; k++) {
            fprintf(json_file, "%.4f", h_flow[h * 9 + k]);
            if (k < 8) fprintf(json_file, ",");
        }
        fprintf(json_file, "]");
        if (h < NUM_HEADS - 1) fprintf(json_file, ",");
    }
    fprintf(json_file, "]},");
    delete[] h_flow;
    
    // Behavioral agents
    BehavioralState* h_agents = new BehavioralState[h_pool.active_count.load()];
    cudaMemcpy(h_agents, h_organism.behavioral_agents, h_pool.active_count.load() * sizeof(BehavioralState), cudaMemcpyDeviceToHost);
    
    fprintf(json_file, "\"agents\":[");
    for (int i = 0; i < h_pool.active_count.load(); i++) {
        fprintf(json_file, "{\"pos\":[%.3f,%.3f],\"vel\":[%.3f,%.3f],\"bc\":[", 
                h_agents[i].position[0], h_agents[i].position[1],
                h_agents[i].velocity[0], h_agents[i].velocity[1]);
        for (int j = 0; j < 10; j++) {
            fprintf(json_file, "%.3f", h_agents[i].behavioral_coords[j]);
            if (j < 9) fprintf(json_file, ",");
        }
        fprintf(json_file, "]}");
        if (i < h_pool.active_count.load() - 1) fprintf(json_file, ",");
    }
    fprintf(json_file, "],");
    delete[] h_agents;
    
    // Voronoi cells - sample first 20
    int voronoi_sample = min(20, h_organism.num_voronoi_cells);
    VoronoiCell* h_voronoi = new VoronoiCell[voronoi_sample];
    cudaMemcpy(h_voronoi, h_organism.voronoi_cells, voronoi_sample * sizeof(VoronoiCell), cudaMemcpyDeviceToHost);
    
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
    
    // Archive - sample first 10
    int archive_sample = min(10, h_organism.archive_size);
    if (archive_sample > 0) {
        GPUElite* h_archive = new GPUElite[archive_sample];
        cudaMemcpy(h_archive, h_organism.archive, archive_sample * sizeof(GPUElite), cudaMemcpyDeviceToHost);
        
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
            for (int j = 0; j < 10; j++) {
                fprintf(json_file, "%.3f", h_archive[i].behavioral_coords[j]);
                if (j < 9) fprintf(json_file, ",");
            }
            fprintf(json_file, "],\"metrics\":[");
            for (int j = 0; j < 10; j++) {
                fprintf(json_file, "%.3f", h_archive[i].raw_metrics[j]);
                if (j < 9) fprintf(json_file, ",");
            }
            fprintf(json_file, "]}");
            if (i < archive_sample - 1) fprintf(json_file, ",");
        }
        fprintf(json_file, "]}},");
        delete[] h_archive;
    } else {
        fprintf(json_file, "\"archive\":{\"size\":0,\"elites\":[]}},");
    }
    
    // Pool state
    fprintf(json_file, "\"pool\":{\"active\":%d,\"capacity\":%d,\"spawned\":%d,\"culled\":%d}",
            h_pool.active_count.load(), h_pool.capacity, 
            h_pool.total_spawned.load(), h_pool.total_culled.load());
    
    fprintf(json_file, "}\n");
    fflush(json_file);
    
    delete[] h_chemical;
}

#endif
