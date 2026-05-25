#ifndef AUDIT_WRITER_CU
#define AUDIT_WRITER_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include "../debug/provenance.cuh"
extern "C" int stbi_write_png(char const *filename, int x, int y, int comp, const void *data, int stride_bytes);
#include <cstdio>
#include <cmath>
#include <cstring>

int write_sample_images(const char* session_dir, int gen, TelemetryAuditEntry* audit) {
    int img_size = audit->grid_size * audit->grid_size;
    for (int s = 0; s < AUDIT_SAMPLE_COUNT && s < audit->batch_size; s++) {
        char png_path[256];
        snprintf(png_path, sizeof(png_path), "%s/samples/gen%04d_s%d.png", session_dir, gen, s);
        if (!stbi_write_png(png_path, audit->grid_size, audit->grid_size, 1, &audit->sample_images[s * img_size], audit->grid_size)) {
            fprintf(stderr, "FATAL: Cannot create sample %s\n", png_path);
            return 1;
        }
    }
    return 0;
}

int write_ca_snapshot(const char* path, int gen, TelemetryAuditEntry* audit) {
    int ca_size = audit->grid_size * audit->grid_size;
    unsigned char* pixels = (unsigned char*)malloc(ca_size);
    if (!pixels) {
        fprintf(stderr, "FATAL: Cannot allocate pixels for CA snapshot\n");
        return 1;
    }
    for (int i = 0; i < ca_size; i++) {
        float val = audit->ca_snapshot[i];
        if (std::isnan(val)) {
            fprintf(stderr, "FATAL: NAN in CA snapshot at index %d, gen %d\n", i, gen);
            free(pixels);
            return 1;
        }
        val = (val < 0.0f) ? 0.0f : ((val > 1.0f) ? 1.0f : val);
        pixels[i] = (unsigned char)(val * 255.0f);
    }
    char png_path[256];
    snprintf(png_path, sizeof(png_path), "%s", path);
    char* ext = strstr(png_path, ".pgm");
    if (ext) { strcpy(ext, ".png"); }
    if (!stbi_write_png(png_path, audit->grid_size, audit->grid_size, 1, pixels, audit->grid_size)) {
        fprintf(stderr, "FATAL: Cannot create CA state %s\n", png_path);
        free(pixels);
        return 1;
    }
    free(pixels);
    return 0;
}

int write_predictions_csv(const char* path, int gen, TelemetryAuditEntry* audit) {
    FILE* f = fopen(path, gen == 0 ? "w" : "a");
    if (!f) {
        fprintf(stderr, "FATAL: Cannot create predictions %s\n", path);
        return 1;
    }
    const char* mode = audit->is_train_batch ? "train" : "test";
    if (gen == 0) {
        fprintf(f, "gen,sample,mode,label,prediction,confidence,correct\n");
    }
    for (int s = 0; s < AUDIT_SAMPLE_COUNT && s < audit->batch_size; s++) {
        if (std::isnan(audit->sample_confidences[s])) {
            fprintf(stderr, "FATAL: NAN confidence for sample %d, gen %d\n", s, gen);
            fclose(f);
            return 1;
        }
        if (audit->sample_labels[s] < 0 || audit->sample_predictions[s] < 0) {
            fprintf(stderr, "FATAL: Invalid label/prediction for sample %d, gen %d\n", s, gen);
            fclose(f);
            return 1;
        }
        fprintf(f, "%d,%d,%s,%d,%d,%.6f,%d\n",
                gen, s, mode, audit->sample_labels[s], audit->sample_predictions[s],
                audit->sample_confidences[s],
                (audit->sample_labels[s] == audit->sample_predictions[s]) ? 1 : 0);
    }
    fclose(f);
    return 0;
}

void append_to_manifest(const char* manifest_path, const char* file_path, double elapsed_sec) {
    FILE* mf = fopen(manifest_path, "r");
    bool need_header = (mf == NULL);
    if (mf) fclose(mf);

    mf = fopen(manifest_path, "a");
    if (mf) {
        if (need_header) {
            fprintf(mf, "file,elapsed_sec\n");
        }
        fprintf(mf, "%s,%.2f\n", file_path, elapsed_sec);
        fclose(mf);
    }
}

int write_generation_summary(const char* session_dir, int gen, TelemetryAuditEntry* audit) {
    char path[256];
    snprintf(path, sizeof(path), "%s/metrics.csv", session_dir);

    FILE* f = fopen(path, gen == 0 ? "w" : "a");
    if (!f) return 1;

    if (gen == 0) {
        fprintf(f,
            "gen,mode,accuracy,loss,train_acc,test_acc,gen_gap,"
            "avg_confidence,stability,"
            "pool_alive,pool_capacity,"
            "occupied_cells,frontier_gained,frontier_lost,sparse_cells,niche_entropy,novelty_gradient,"
            "fitness_best,fitness_mean,fitness_delta,quality_floor,quality_mean,quality_range,"
            "density_mean,density_max,density_var,"
            "hw_min,hw_max,hw_mean,"
            "task_min,task_max,task_mean,"
            "gen_min,gen_max,gen_mean,"
            "total_pop,births,deaths,"
            "diresa_loss_hw,diresa_loss_task,diresa_loss_gen,diresa_loss_total,diresa_drift,diresa_utilization,"
            "unique_hashes,hash_entropy,avg_deltas,"
            "correct,batch_size,error_count\n");
    }

    const char* mode = audit->is_train_batch ? "train" : "test";
    fprintf(f,
        "%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,"
        "%d,%d,"
        "%d,%d,%d,%d,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,"
        "%d,%d,%d,"
        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%d,%.6f,%.6f,"
        "%d,%d,%d\n",
        gen, mode, audit->accuracy, audit->loss, audit->train_accuracy, audit->test_accuracy, audit->generalization_gap,
        audit->avg_confidence, audit->classification_stability,
        audit->pool_alive_count, audit->pool_capacity,
        audit->archive_occupied_cells, audit->frontier_cells_gained, audit->frontier_cells_lost, audit->sparse_cell_count, audit->niche_entropy, audit->novelty_gradient,
        audit->elite_fitness_best, audit->elite_fitness_mean, audit->elite_fitness_delta, audit->quality_floor, audit->quality_mean, audit->quality_range,
        audit->density_mean, audit->density_max, audit->density_variance,
        audit->hw_axis_min, audit->hw_axis_max, audit->hw_axis_mean,
        audit->task_axis_min, audit->task_axis_max, audit->task_axis_mean,
        audit->gen_axis_min, audit->gen_axis_max, audit->gen_axis_mean,
        audit->total_population, audit->births_this_gen, audit->deaths_this_gen,
        audit->diresa_recon_loss_hw, audit->diresa_recon_loss_task, audit->diresa_recon_loss_gen, audit->diresa_recon_loss_total, audit->diresa_behavioral_drift, audit->diresa_latent_utilization,
        audit->genome_unique_hashes, audit->genome_hash_entropy, audit->genome_avg_deltas,
        audit->correct_count, audit->batch_size, audit->error_count);

    fclose(f);
    return 0;
}

int write_pool_state(const char* session_dir, int gen, TelemetryAuditEntry* audit) {
    if (!audit) return 1;

    char path[256];
    snprintf(path, sizeof(path), "%s/pool_states.csv", session_dir);

    FILE* f = fopen(path, gen == 0 ? "w" : "a");
    if (!f) return 1;

    if (gen == 0) {
        fprintf(f, "gen,entry,fitness,hunger,age,num_deltas,genome_hash\n");
    }
    for (int i = 0; i < audit->pool_capacity && i < POOL_CAPACITY_MAX; i++) {
        if (!audit->pool_entry_alive[i]) continue;
        fprintf(f, "%d,%d,%.6f,%.6f,%d,%d,%llu\n",
                gen, i,
                audit->pool_entry_fitness[i],
                audit->pool_entry_hunger[i],
                audit->pool_entry_age[i],
                audit->pool_entry_num_deltas[i],
                (unsigned long long)audit->pool_entry_genome_hash[i]);
    }

    fclose(f);
    return 0;
}

int write_class_accuracy(const char* session_dir, int gen, float* per_class_correct,
                         float* per_class_total, int num_classes) {
    char path[256];
    snprintf(path, sizeof(path), "%s/class_accuracy.csv", session_dir);

    FILE* f = fopen(path, gen == 0 ? "w" : "a");
    if (!f) return 1;

    if (gen == 0) {
        fprintf(f, "gen");
        for (int c = 0; c < num_classes; c++) {
            fprintf(f, ",class%d_acc", c);
        }
        fprintf(f, "\n");
    }

    fprintf(f, "%d", gen);
    for (int c = 0; c < num_classes; c++) {
        float acc = (per_class_total && per_class_total[c] > 0) ?
                    per_class_correct[c] / per_class_total[c] : 0.0f;
        fprintf(f, ",%.6f", acc);
    }
    fprintf(f, "\n");

    fclose(f);
    return 0;
}

int write_chemical_field(const char* session_dir, int gen, float* concentration, int grid_size) {
    if (!concentration) return 1;

    char path[256];
    snprintf(path, sizeof(path), "%s/chemical_fields.csv", session_dir);

    FILE* f = fopen(path, gen == 0 ? "w" : "a");
    if (!f) return 1;

    if (gen == 0) {
        fprintf(f, "gen,y,x,concentration\n");
    }
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            fprintf(f, "%d,%d,%d,%.6f\n", gen, y, x, concentration[y * grid_size + x]);
        }
    }

    fclose(f);
    return 0;
}

int write_error_log(const char* session_dir, int gen, TelemetryAuditEntry* audit) {
    if (audit->error_count == 0) return 0;

    char path[256];
    snprintf(path, sizeof(path), "%s/errors.csv", session_dir);

    FILE* f = fopen(path, gen == 0 ? "w" : "a");
    if (!f) return 1;

    if (gen == 0) {
        fprintf(f, "gen,error_gen,block,thread,line,message\n");
    }
    for (int i = 0; i < audit->error_count && i < DEVICE_ERROR_LOG_CAPACITY; i++) {
        DeviceErrorEntry* e = &audit->error_log[i];
        fprintf(f, "%d,%d,%d,%d,%d,%s\n",
                gen, e->generation, e->block_id, e->thread_id, e->source_line, e->message);
    }
    fclose(f);
    return 0;
}

static const char* KERNEL_PHASE_NAMES[KERNEL_PHASE_COUNTER_COUNT] = {
    "blocks_ca_fwd", "blocks_entered",
    "flow_done", "bwd_enter", "bwd_fatal_checks", "bwd_chunk",
    "bwd_value_grad", "bwd_inter_grad", "bwd_perc_grad", "bwd_done",
    "bwd_zero_dw", "bwd_setup_done", "bwd_chunks_done",
    "bwd_inter_grad_copy", "bwd_perc_grad_copy", "bwd_grad_conc",
    "bwd_chunk0", "bwd_i_done", "bwd_v_done",
    "bwd_chunk2_enter", "bwd_di_write", "bwd_perc_load", "bwd_dp_write",
    "bwd_im2col", "bwd_conv_fp16", "bwd_input_grad", "bwd_scatter",
    "post_bwd_barrier"
};

int write_kernel_diagnostics(const char* session_dir, int gen, TelemetryAuditEntry* audit) {
    char path[256];
    snprintf(path, sizeof(path), "%s/kernel_diagnostics.csv", session_dir);

    FILE* f = fopen(path, gen == 0 ? "w" : "a");
    if (!f) return 1;

    if (gen == 0) {
        fprintf(f, "gen,num_heads,head_dim,channels,num_blocks");
        for (int i = 0; i < KERNEL_PHASE_COUNTER_COUNT; i++) {
            fprintf(f, ",%s", KERNEL_PHASE_NAMES[i]);
        }
        fprintf(f, "\n");
    }

    fprintf(f, "%d,%d,%d,%d,%d",
            gen, audit->num_heads, audit->head_dim, audit->channels, audit->num_blocks);
    for (int i = 0; i < KERNEL_PHASE_COUNTER_COUNT; i++) {
        fprintf(f, ",%d", audit->kernel_phase_counts[i]);
    }
    fprintf(f, "\n");

    fclose(f);
    return 0;
}

int write_state_json(FILE* json_file, double elapsed_time, TelemetryAuditEntry* audit) {
    if (!json_file || !audit) return 1;

    if (audit->provenance_source == PROVENANCE_SOURCE_NONE) {
        return 0;
    }

    if (host_is_uninitialized_int(audit->generation)) {
        fprintf(stderr, "E_UNINIT audit.generation\n");
        return 1;
    }

    if (!(audit->fields_written_mask & AUDIT_MASK_POOL)) {
        fprintf(stderr, "E_NOPROV audit.pool m=0x%x\n", audit->fields_written_mask);
        return 1;
    }

    HOST_ASSERT_INITIALIZED_INT(audit->pool_alive_count, "pool_alive_count");
    HOST_ASSERT_INITIALIZED_INT(audit->pool_capacity, "pool_capacity");

    fprintf(json_file, "{\"gen\":%d,\"elapsed\":%.2f,\"seq\":%llu,",
            audit->generation, elapsed_time, (unsigned long long)audit->sequence_number);

    fprintf(json_file, "\"chemical\":{\"concentration\":[");
    for (int i = 0; i < STATE_EXPORT_CHEM_SIZE * STATE_EXPORT_CHEM_SIZE; i++) {
        fprintf(json_file, "%.4f", audit->state_chemical_sample[i]);
        if (i < STATE_EXPORT_CHEM_SIZE * STATE_EXPORT_CHEM_SIZE - 1) fprintf(json_file, ",");
    }
    fprintf(json_file, "],\"size\":%d},", STATE_EXPORT_CHEM_SIZE);

    fprintf(json_file, "\"agents\":[");
    for (int i = 0; i < audit->state_agent_count && i < STATE_EXPORT_AGENT_COUNT; i++) {
        fprintf(json_file, "{\"pos\":[%.3f,%.3f],\"vel\":[%.3f,%.3f],\"exploration\":%.3f,\"sensitivity\":%.3f}",
                audit->state_agent_pos_x[i], audit->state_agent_pos_y[i],
                audit->state_agent_vel_x[i], audit->state_agent_vel_y[i],
                audit->state_agent_exploration[i], audit->state_agent_sensitivity[i]);
        if (i < audit->state_agent_count - 1 && i < STATE_EXPORT_AGENT_COUNT - 1) fprintf(json_file, ",");
    }
    fprintf(json_file, "],");

    fprintf(json_file, "\"voronoi\":[");
    for (int i = 0; i < audit->state_voronoi_count && i < STATE_EXPORT_VORONOI_COUNT; i++) {
        fprintf(json_file, "{\"density\":%d,\"radius\":%.3f,\"best_elite\":%d,\"hw_centroid\":[",
                audit->state_voronoi_density[i], audit->state_voronoi_radius[i],
                audit->state_voronoi_best_elite_idx[i]);
        for (int j = 0; j < BEHAVIORAL_DIM_HW; j++) {
            fprintf(json_file, "%.3f", audit->state_voronoi_hw_centroid[i * BEHAVIORAL_DIM_HW + j]);
            if (j < BEHAVIORAL_DIM_HW - 1) fprintf(json_file, ",");
        }
        fprintf(json_file, "],\"task_centroid\":[");
        for (int j = 0; j < BEHAVIORAL_DIM_TASK; j++) {
            fprintf(json_file, "%.3f", audit->state_voronoi_task_centroid[i * BEHAVIORAL_DIM_TASK + j]);
            if (j < BEHAVIORAL_DIM_TASK - 1) fprintf(json_file, ",");
        }
        fprintf(json_file, "],\"gen_centroid\":[");
        for (int j = 0; j < BEHAVIORAL_DIM_GEN; j++) {
            fprintf(json_file, "%.3f", audit->state_voronoi_gen_centroid[i * BEHAVIORAL_DIM_GEN + j]);
            if (j < BEHAVIORAL_DIM_GEN - 1) fprintf(json_file, ",");
        }
        fprintf(json_file, "]}");
        if (i < audit->state_voronoi_count - 1 && i < STATE_EXPORT_VORONOI_COUNT - 1) fprintf(json_file, ",");
    }
    fprintf(json_file, "],");

    fprintf(json_file, "\"archive\":{\"size\":%d,\"elites\":[", audit->state_archive_count);
    for (int i = 0; i < audit->state_archive_count && i < STATE_EXPORT_ARCHIVE_COUNT; i++) {
        fprintf(json_file, "{\"f\":%.4f,\"c\":%.4f,\"rank\":%.4f,\"gen\":%d,\"hash\":%llu,\"parents\":[%u,%u],\"hw_coords\":[",
                audit->state_archive_fitness[i],
                audit->state_archive_coherence[i],
                audit->state_archive_effective_rank[i],
                (int)audit->state_archive_generation[i],
                (unsigned long long)audit->state_archive_genome_hash[i],
                audit->state_archive_parent_id_0[i],
                audit->state_archive_parent_id_1[i]);
        for (int j = 0; j < BEHAVIORAL_DIM_HW; j++) {
            fprintf(json_file, "%.3f", audit->state_archive_hw_coords[i * BEHAVIORAL_DIM_HW + j]);
            if (j < BEHAVIORAL_DIM_HW - 1) fprintf(json_file, ",");
        }
        fprintf(json_file, "],\"task_coords\":[");
        for (int j = 0; j < BEHAVIORAL_DIM_TASK; j++) {
            fprintf(json_file, "%.3f", audit->state_archive_task_coords[i * BEHAVIORAL_DIM_TASK + j]);
            if (j < BEHAVIORAL_DIM_TASK - 1) fprintf(json_file, ",");
        }
        fprintf(json_file, "],\"gen_coords\":[");
        for (int j = 0; j < BEHAVIORAL_DIM_GEN; j++) {
            fprintf(json_file, "%.3f", audit->state_archive_gen_coords[i * BEHAVIORAL_DIM_GEN + j]);
            if (j < BEHAVIORAL_DIM_GEN - 1) fprintf(json_file, ",");
        }
        fprintf(json_file, "],\"hardware_features\":[");
        for (int j = 0; j < HARDWARE_FEATURES_DIM; j++) {
            fprintf(json_file, "%.3f", audit->state_archive_hardware_features[i * HARDWARE_FEATURES_DIM + j]);
            if (j < HARDWARE_FEATURES_DIM - 1) fprintf(json_file, ",");
        }
        fprintf(json_file, "]}");
        if (i < audit->state_archive_count - 1 && i < STATE_EXPORT_ARCHIVE_COUNT - 1) fprintf(json_file, ",");
    }
    fprintf(json_file, "]},");

    fprintf(json_file, "\"pool\":{\"active\":%d,\"capacity\":%d,\"spawned\":%d,\"culled\":%d}",
            audit->pool_alive_count, audit->pool_capacity,
            audit->pool_total_spawned, audit->pool_total_culled);

    fprintf(json_file, "}\n");
    fflush(json_file);
    return 0;
}

#endif
