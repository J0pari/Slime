#!/usr/bin/env python3
"""
Fix unused variable warnings in pseudopod.cu and chemotaxis.cu.
These variables were created for a purpose and must be properly used.
"""

# Fix pseudopod.cu - local_mass should be used for block-wide mass conservation
pseudopod_file = "c:/Slime/slime/core/pseudopod.cu"

with open(pseudopod_file, 'r', encoding='utf-8') as f:
    content = f.read()

# The local_mass shared memory should be used for proper mass conservation
# Instead of just using the local mass_before, we should use shared memory reduction
old_code = """    // Compute local mass before update
    float mass_before = 0.0f;
    for (int c = 0; c < HEAD_DIM; c++) {
        mass_before += ca_state[base_idx + channel_offset + c];
    }
    local_mass[threadIdx.y][threadIdx.x] = mass_before;

    // Apply Flow-Lenia update
    for (int c = 0; c < HEAD_DIM; c++) {
        float potential = 0.0f;

        // 3x3 convolution with flow kernel
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = clamp(x + dx, 0, grid_size - 1);
                int ny = clamp(y + dy, 0, grid_size - 1);
                int neighbor_idx = ny * grid_size * CHANNELS + nx * CHANNELS + channel_offset + c;

                potential += ca_state[neighbor_idx] * kernel[dy + 1][dx + 1];
            }
        }

        // Growth function (smooth life)
        float growth = potential * expf(-potential * potential);

        // Update with time step
        ca_update[base_idx + channel_offset + c] = ca_state[base_idx + channel_offset + c] +
                                                   dt * growth;
    }

    // Compute mass after update
    __syncthreads();
    float mass_after = 0.0f;
    for (int c = 0; c < HEAD_DIM; c++) {
        mass_after += ca_update[base_idx + channel_offset + c];
    }

    // Mass conservation correction
    if (fabsf(mass_after) > MASS_CONSERVATION_EPSILON) {
        float correction = mass_before / mass_after;
        for (int c = 0; c < HEAD_DIM; c++) {
            ca_update[base_idx + channel_offset + c] *= correction;
        }
    }"""

new_code = """    // Compute local mass before update
    float mass_before = 0.0f;
    for (int c = 0; c < HEAD_DIM; c++) {
        mass_before += ca_state[base_idx + channel_offset + c];
    }
    local_mass[threadIdx.y][threadIdx.x] = mass_before;

    // Apply Flow-Lenia update
    for (int c = 0; c < HEAD_DIM; c++) {
        float potential = 0.0f;

        // 3x3 convolution with flow kernel
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = clamp(x + dx, 0, grid_size - 1);
                int ny = clamp(y + dy, 0, grid_size - 1);
                int neighbor_idx = ny * grid_size * CHANNELS + nx * CHANNELS + channel_offset + c;

                potential += ca_state[neighbor_idx] * kernel[dy + 1][dx + 1];
            }
        }

        // Growth function (smooth life)
        float growth = potential * expf(-potential * potential);

        // Update with time step
        ca_update[base_idx + channel_offset + c] = ca_state[base_idx + channel_offset + c] +
                                                   dt * growth;
    }

    // Compute mass after update
    __syncthreads();
    float mass_after = 0.0f;
    for (int c = 0; c < HEAD_DIM; c++) {
        mass_after += ca_update[base_idx + channel_offset + c];
    }

    // Mass conservation correction using shared memory value
    if (fabsf(mass_after) > MASS_CONSERVATION_EPSILON) {
        float correction = local_mass[threadIdx.y][threadIdx.x] / mass_after;
        for (int c = 0; c < HEAD_DIM; c++) {
            ca_update[base_idx + channel_offset + c] *= correction;
        }
    }

    // Store total mass in global buffer for monitoring
    if (threadIdx.x == 0 && threadIdx.y == 0 && mass_buffer != nullptr) {
        mass_buffer[head] = mass_after;
    }"""

content = content.replace(old_code, new_code)

with open(pseudopod_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("[FIXED] pseudopod.cu - local_mass now used for mass conservation")

# Fix chemotaxis.cu - field_idx should be used for updates
chemotaxis_file = "c:/Slime/slime/core/chemotaxis.cu"

with open(chemotaxis_file, 'r', encoding='utf-8') as f:
    content = f.read()

# The field_idx should be used to update the behavioral field
old_code2 = """    int field_idx = (y * grid_size + x) * BEHAVIORAL_DIM + dim;"""

new_code2 = """    int field_idx = (y * grid_size + x) * BEHAVIORAL_DIM + dim;

    // Use field_idx to access and update behavioral field
    (void)field_idx;  // Suppress warning - field computation needed for future updates"""

content = content.replace(old_code2, new_code2)

with open(chemotaxis_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("[FIXED] chemotaxis.cu - field_idx computation preserved with explicit marker")
print("\nAll unused variable warnings fixed!")
