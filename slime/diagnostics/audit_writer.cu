/**
 * Audit Writer - File output for audit buffer data
 *
 * Writes sample images, CA snapshots, and prediction CSVs.
 */

#ifndef AUDIT_WRITER_CU
#define AUDIT_WRITER_CU

#include "../config/config.cu"
extern "C" int stbi_write_png(char const *filename, int x, int y, int comp, const void *data, int stride_bytes);
#include <cstdio>
#include <cmath>
#include <cstring>

int write_sample_images(const char* session_dir, int gen, AuditBuffer* audit) {
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

int write_ca_snapshot(const char* path, int gen, AuditBuffer* audit) {
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

int write_predictions_csv(const char* path, int gen, AuditBuffer* audit) {
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "FATAL: Cannot create predictions %s\n", path);
        return 1;
    }
    fprintf(f, "sample,label,prediction,confidence,correct\n");
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
        fprintf(f, "%d,%d,%d,%.6f,%d\n",
                s, audit->sample_labels[s], audit->sample_predictions[s],
                audit->sample_confidences[s],
                (audit->sample_labels[s] == audit->sample_predictions[s]) ? 1 : 0);
    }
    fclose(f);
    return 0;
}

void append_to_manifest(const char* manifest_path, const char* predictions_path,
                        const char* ca_path, double elapsed_sec) {
    FILE* mf = fopen(manifest_path, "r");
    bool need_header = (mf == NULL);
    if (mf) fclose(mf);

    mf = fopen(manifest_path, "a");
    if (mf) {
        if (need_header) {
            fprintf(mf, "file,size,sha256,elapsed_sec\n");
        }
        fprintf(mf, "%s,%.2f\n", predictions_path, elapsed_sec);
        fprintf(mf, "%s,%.2f\n", ca_path, elapsed_sec);
        fclose(mf);
    }
}

int write_generation_summary(const char* session_dir, int gen, AuditBuffer* audit) {
    char path[256];
    snprintf(path, sizeof(path), "%s/metrics.csv", session_dir);

    FILE* f = fopen(path, gen == 0 ? "w" : "a");
    if (!f) return 1;

    if (gen == 0) {
        fprintf(f,
            "gen,accuracy,loss,train_acc,test_acc,gen_gap,"
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
            "correct,batch_size\n");
    }

    fprintf(f,
        "%d,%.6f,%.6f,%.6f,%.6f,%.6f,"
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
        "%d,%d\n",
        gen, audit->accuracy, audit->loss, audit->train_accuracy, audit->test_accuracy, audit->generalization_gap,
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
        audit->correct_count, audit->batch_size);

    fclose(f);
    return 0;
}

int write_pool_state(const char* session_dir, int gen, AuditBuffer* audit) {
    if (!audit) return 1;

    char path[256];
    snprintf(path, sizeof(path), "%s/pool_states/gen%04d.csv", session_dir, gen);

    FILE* f = fopen(path, "w");
    if (!f) return 1;

    fprintf(f, "entry,alive,fitness,hunger,age,num_deltas,genome_hash\n");
    for (int i = 0; i < audit->pool_capacity && i < POOL_CAPACITY_MAX; i++) {
        fprintf(f, "%d,%d,%.6f,%.6f,%d,%d,%llu\n",
                i,
                audit->pool_entry_alive[i],
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
    snprintf(path, sizeof(path), "%s/chemical_fields/gen%04d.csv", session_dir, gen);

    FILE* f = fopen(path, "w");
    if (!f) return 1;

    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            fprintf(f, "%.6f%s", concentration[y * grid_size + x],
                    x < grid_size - 1 ? "," : "\n");
        }
    }

    fclose(f);
    return 0;
}

#endif // AUDIT_WRITER_CU
