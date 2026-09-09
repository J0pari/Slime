# CUDA Engineering Specification

This document specifies the exact GPU memory layout, kernel configurations,
buffer lifecycle, and data flow for every CUDA operation in the system.
It is the binding reference for all code in `autodiff/`, `optimizer/`,
`execution/`, `integration/`, and `safety/alignment.cu`.

The behavioral specification (what each subsystem computes) lives in
`docs/blueprint.md`. This document specifies *how* it executes on the GPU.

Hardware target: RTX 3060 Laptop, 6 GB VRAM, sm_86 (Ampere), 30 SMs,
CUDA 13.0.

---

## 1. VRAM Budget

Total system VRAM: ~157 MB (61 MB base + 96 MB batched backward workspace).
Headroom: ~5.8 GB.

| Buffer | Size | Lifetime |
|--------|------|----------|
| OrganismState[88] (grid+scratch+btraj) | 22.0 MB | run |
| Checkpoints[64] (4 × FP16 grid per org) | 32.0 MB | run |
| GradBuffers[64] (dW per org, FP32) | 0.63 MB | run |
| BackwardWorkspace (2 × FP32 grid) | 0.50 MB | run |
| SharedWeights (W_perc+W_inter+W_flow+W_bmap) | 10.1 KB | run |
| CAME state (m,v,c,prev_u per weight) | 40.4 KB | run |
| DeltaWeights[88] | 2.8 MB | run |
| Genomes[88] | 11.0 KB | run |
| Archive (5000 entries + bins + RFF) | 2.0 MB | run |
| PlaceholderRegressor + AdamW | 171.8 KB | run |
| ReplayBuffer (5000 tuples) | 0.9 MB | run |
| ClassifierBatch (16 × 64×64×3 FP16) | 384.1 KB | run |
| Descriptors[64] (bmap_64 per org) | 8.0 KB | run |
| SeedGradients[64] | 8.0 KB | run |
| host_descriptors (host mirror) | 8.0 KB | run |
| host_btraj (host mirror) | 45 KB | run |

All device buffers are allocated via `cudaMalloc`. All host buffers are
allocated via `cudaMallocHost` (pinned) or standard `malloc`. Explicit
`cudaMemcpyAsync` transfers move data between host and device at defined
synchronization points. No `cudaMallocManaged` is used for any buffer
that participates in the generation loop.

The `World` struct itself lives in host memory (standard `new`). It holds
host-side state (archive, safety structs, scalars) and device pointers to
GPU buffers.

---

## 2. Memory Architecture

### 2.1 Device-Resident Buffers (cudaMalloc)

These live on the GPU and are accessed only by kernels:

| Buffer | Type | Count | Notes |
|--------|------|-------|-------|
| `d_organisms` | OrganismState[] | TOTAL_ORG | grid, scratch, bmap_traj |
| `d_weights` | float[] | TOTAL_WEIGHTS | W_perc\|W_inter\|W_flow\|W_bmap |
| `d_fwd_inputs` | ForwardInputs[] | POOL_SIZE | role, pointers |
| `d_checkpoints` | CheckpointBuffer[] | POOL_SIZE | 4 FP16 grids each |
| `d_grads` | GradBuffers[] | POOL_SIZE | per-org weight gradients |
| `d_bwd_d_state[2]` | float[] | POOL_SIZE×GRID²×CH each | per-org backward d_state (ping-pong) |
| `d_mean_grad` | float[] | TOTAL_WEIGHTS | averaged gradient |
| `d_came_m` | float[] | TOTAL_WEIGHTS | CAME 1st moment |
| `d_came_v` | float[] | TOTAL_WEIGHTS | CAME 2nd moment |
| `d_came_c` | float[] | TOTAL_WEIGHTS | CAME confidence |
| `d_came_prev_u` | float[] | TOTAL_WEIGHTS | CAME previous update |
| `d_descriptors` | float[] | POOL_SIZE×BMAP_DIM | bmap_64 output |
| `d_seed_grad` | float[] | POOL_SIZE×BMAP_DIM | loss grad input |
| `d_batch_image` | __half[] | 16×64×64×3 | classifier batch images |
| `d_batch_task_emb` | float[] | TASK_EMBED_DIM | current task embedding |
| `d_grad_norm` | float[] | 1 | pinned host scalar for grad norm |
| `d_sot_temp_images` | __half[] | SOT_SUBBATCH×GRID²×3 | SOT reference images |
| `d_sot_task_emb` | float[] | TASK_EMBED_DIM | SOT reference task emb |
| `d_sot_fwd_inputs` | ForwardInputs[] | SOT_SUBBATCH | SOT reference inputs |
| `d_sot_descriptors` | float[] | SOT_SUBBATCH×BMAP_DIM | SOT reference bmap_64 |
| `d_pt_swap_org` | OrganismState | 1 | PT swap temp buffer |
| `d_pt_swap_ckpt` | CheckpointBuffer | 1 | PT swap temp buffer |
| `d_pt_swap_grad` | GradBuffers | 1 | PT swap temp buffer |

### 2.2 Host-Resident Buffers (pinned via cudaMallocHost)

These live in pinned host memory for fast DMA transfers:

| Buffer | Type | Count | Notes |
|--------|------|-------|-------|
| `h_descriptors` | float[] | POOL_SIZE×BMAP_DIM | D→H after extract |
| `h_btraj` | float[] | TOTAL_ORG×BTRAJ×BMAP | D→H after forward |
| `h_seed_grad` | float[] | POOL_SIZE×BMAP_DIM | H→D before backward |
| `h_fwd_inputs` | ForwardInputs[] | POOL_SIZE | H→D before forward |
| `h_weights` | float[] | TOTAL_WEIGHTS | D→H for host reads |

### 2.3 Host-Only State (in World struct, standard new)

These never touch the GPU:

- `World` struct: generation counter, PCG32 RNG, SOT key, flags
- `Archive`: entries, bins, RFF projection, per-role means
- `OrganismTable` metadata: Genome[], DeltaWeights[], lineage_id[],
  parent_id[], spawn_gen[], replica_tag[] (host-side copies; genomes
  are not needed on device)
- `PlaceholderRegressor` + `ReplayBuffer`
- `ClassifierBatch` (host assembly, images copied to device)
- Safety structs: CusumState, MutationLadder, StressLadder,
  SentinelEnsemble, SentinelHistory, AuditRegressor, ProbePanel,
  HostAuthority, CorrelationWindow, PredictorSelectionCache
- `IntentRegistry`: BTRAJ copied from device after forward

**Annotation policy.** Functions that operate exclusively on host-resident
state (archive functions, genome codec functions except `apply_delta`,
batch assembly, safety struct functions) are annotated `__host__` only.
Speculative `__device__` annotations on host-only functions are forbidden:
they propagate annotation requirements to callees and create false
expectations about device accessibility. Only `apply_delta` (future device
decode kernel) and PCG32 functions (G-100: callable from both host and
device) carry `__host__ __device__`.

---

## 3. Transfer Schedule

Each generation has exactly 4 transfer points:

### T1: Before Forward (H→D)
```
cudaMemcpyAsync(d_fwd_inputs, h_fwd_inputs, ..., H2D, stream)
cudaMemcpyAsync(d_batch_image, batch.image, ..., H2D, stream)
cudaMemcpyAsync(d_batch_task_emb, batch.task_embedding, ..., H2D, stream)
```
ForwardInputs contains role tags and device pointers (to d_batch_image,
d_batch_task_emb). The host sets up h_fwd_inputs with these device
pointers, then copies the struct array to the device.

### T2: After Forward + Extract (D→H)
```
cudaMemcpyAsync(h_descriptors, d_descriptors, ..., D2H, stream)
cudaMemcpyAsync(h_btraj, &d_organisms[0].bmap_traj, ..., D2H, stream)
// (btraj is strided — use a small gather kernel or POOL_SIZE individual copies)
```

### T3: Before Backward (H→D)
```
cudaMemcpyAsync(d_seed_grad, h_seed_grad, ..., H2D, stream)
```
Host computes seed_grad from the actual loss function using h_descriptors.

### T4: After CAME (D→H, optional)
```
cudaMemcpyAsync(h_weights, d_weights, ..., D2H, stream)
```
Only needed when host code reads weights (e.g., for checkpoint writes).
During normal generations this transfer is skipped.

---

## 4. Kernel Specifications

### 4.1 forward_with_checkpoints

**File**: `autodiff/warp_tape.cu`
**Purpose**: Run 64-step CA forward for all organisms, saving 4 checkpoints.
**Grid**: `<<<POOL_SIZE, dim3(16,16)>>>`
**Shared mem**: static only (project_bmap uses 256×16 floats = 16 KB for
deterministic tree reduction + 16 floats for summary; ca_step uses none).
**One block per organism**. 256 threads tile the 64×64 grid (each thread
loops over 16 cells).

Steps:
1. Seed grid from ForwardInputs (role-switched: classifier gets image,
   predictor gets target_bmap_32).
2. Save checkpoint 0 (initial state after seeding).
3. For step 1..64:
   a. `ca_step(curr, next, W_perc, W_inter, W_flow)` — existing, proven.
   b. `rd_step(curr, next, coeffs)` if coeffs != null.
   c. Swap curr/next pointers.
   d. At steps 16, 32, 48: save checkpoint (index 1, 2, 3).
   e. At steps 16, 32, 48, 64: `project_bmap` → write to bmap_traj.
4. Ensure final state is in `o->grid`.

This kernel reuses `ca_step`, `rd_step`,
`seed_classifier_grid`, `seed_predictor_grid` from `nca/engine.cu` as-is.
The only addition is checkpoint saves (memcpy within the kernel).

**project_bmap must be deterministic.** BTRAJ values feed into scoring,
archive descriptors, loss computation, and the bitwise-match acceptance
test between `forward_kernel` and `forward_with_checkpoints`. The
`project_bmap` function in `nca/engine.cu` must use a deterministic
parallel reduction for the global average pool — NOT `atomicAdd` on
shared memory, which produces non-deterministic summation order across
warps. The correct approach: each thread accumulates its cells' values
into thread-local registers, then uses `__syncthreads()` + a tree
reduction in shared memory (or warp shuffle + final shared reduction)
to produce the per-channel sum. This guarantees identical bit patterns
for the same thread layout across kernel launches.

### 4.2 backward — Phase-Decomposed Batched Kernels

**File**: `autodiff/warp_tape.cu`
**Purpose**: Compute FP32 weight gradients via checkpointed reverse-mode AD,
processing **all organisms in parallel**.

**Architecture**: The backward pass is decomposed into separate kernel launches
at each synchronization boundary. Each kernel processes all POOL_SIZE organisms
simultaneously, with one block per organism (matching the forward kernel
pattern). Implicit synchronization between kernel launches replaces the
cooperative `grid.sync()` calls.

**Per-organism workspace buffers** (allocated once, indexed by organism):

| Buffer | Per-org size | Total (×64) | Purpose |
|--------|-------------|-------------|---------|
| `d_bwd_d_state[2]` | GRID²×CH×4 = 256 KB | 32 MB | Running FP32 state gradient (ping-pong) |
| `d_bwd_d_perc` | CELLS×PERC_DIM×4 = 768 KB | 48 MB | d_perc buffer for gather pattern |
| `d_bwd_recomp[2]` | GRID_ELEMS×2 = 128 KB | 16 MB | FP16 re-forward workspace (ping-pong) |

Total additional VRAM: ~96 MB. Within the ~6 GB headroom.

**Grid configuration**: All backward sub-kernels use
`<<<POOL_SIZE, 256>>>` — one block of 256 threads per organism.
Each thread loops over cells in grid-stride fashion
(256 threads, 4096 cells → 16 cells per thread).

**Shared mem**: `CA_CHANNELS * sizeof(float)` per block for reductions.

**Sub-kernels** (launched sequentially, all batched across organisms):

1. **bwd_zero_kernel** `<<<POOL_SIZE, 256>>>`
   Zero `grads[org].dW[:]` and `d_bwd_d_state_A[org][:]` for all organisms.
   Single block per organism (256 threads loop over TOTAL_WEIGHTS and
   GRID_ELEMS). No inter-block dependency.

2. **bwd_seed_avgpool_kernel** `<<<POOL_SIZE, 256>>>`
   Compute global average pool of final grid state → `summary[org][CH]`.
   Each block reduces its organism's 4096 cells into CA_CHANNELS partial
   sums via shared-memory tree reduction. Write summary to a per-organism
   scratch region (first CA_CHANNELS floats of `d_bwd_d_state_B[org]`).

3. **bwd_seed_scatter_kernel** `<<<POOL_SIZE, 256>>>`
   Read summary from scratch, compute d_summary via W_bmap^T × seed_grad,
   seed d_state_A: `d_state_A[org][cell,c] += d_summary[c] / CELLS`.
   Accumulate dW_bmap: `dW_bmap[c*BMAP_DIM+d] += summary[c] * seed_grad[d]`
   (partitioned across threads within the block, no inter-block atomics).

4. **bwd_reforward_kernel** `<<<POOL_SIZE, 256>>>` (launched once per re-forward step)
   Load checkpoint into recomp_curr, then re-run one CA forward step:
   each thread handles its cells in grid-stride. Swap curr/next pointers
   (host-side pointer swap between launches). Called `(local_step - 1)`
   times per local_step to recover the input state.

   Host loop structure:
   ```
   for seg = 3..0:
     for local_step = 16..1:
       launch bwd_load_checkpoint(seg)     // load checkpoint[seg] → recomp_curr
       for fwd = 0..(local_step-2):
         launch bwd_reforward_step()       // one CA step on recomp buffers
       launch bwd_weight_grad()            // Phase A: weight grads + d_perc
       launch bwd_stencil_gather()         // Phase B: gather stencil adjoint
   ```

5. **bwd_load_checkpoint_kernel** `<<<POOL_SIZE, 256>>>`
   Copy `checkpoints[org].data[seg][:]` → `d_bwd_recomp_curr[org][:]`.

6. **bwd_reforward_step_kernel** `<<<POOL_SIZE, 256>>>`
   One CA forward step on the re-forward workspace: read from
   `recomp_curr[org]`, write to `recomp_next[org]`. Each thread handles
   its cells in grid-stride. Host swaps curr/next pointers between launches.

7. **bwd_weight_grad_kernel** (Phase A) `<<<POOL_SIZE, 256>>>`
   For each cell (grid-stride within the block):
   - Read `d_state_next[cell,:]` from `d_bwd_d_state_A[org]`.
   - Recompute forward intermediates: perc, pre_hidden, hidden.
   - `dW_flow[h*CH+c]  += hidden[h] * d_state_next[c]` (atomicAdd within org's grads)
   - `d_hidden = W_flow^T * d_state_next`
   - `d_pre_hidden = d_hidden ⊙ gelu'(pre_hidden)`
   - `dW_inter[p*H+h] += perc[p] * d_pre_hidden[h]` (atomicAdd within org's grads)
   - `d_perc = W_inter^T * d_pre_hidden`
   - `dW_perc[f*9+k]  += Σ_c d_perc[f*CH+c] * neighbor[k,c]` (atomicAdd within org's grads)
   - Store d_perc into `d_bwd_d_perc[org][cell*PERC_DIM..]`.

   All atomicAdd operations are **intra-block** (only threads within the same
   organism's block contend on that organism's GradBuffers). No inter-block
   contention.

8. **bwd_stencil_gather_kernel** (Phase B) `<<<POOL_SIZE, 256>>>`
   **GATHER pattern** — each thread at cell (y,x) reads from its 9
   neighbors' d_perc in `d_bwd_d_perc[org]`:
   ```
   d_state_from_stencil[y,x,c] = 0
   for f in 0..N_PERC_FILTERS:
     for ky in -1..1:
       for kx in -1..1:
         neighbor_y = wrap(y - ky)
         neighbor_x = wrap(x - kx)
         d_state_from_stencil[y,x,c] +=
           W_perc[f*9 + (ky+1)*3 + (kx+1)] *
           d_bwd_d_perc[org][neighbor_cell * PERC_DIM + f*CH + c]
   ```
   Write: `d_bwd_d_state_B[org][cell,c] = d_bwd_d_state_A[org][cell,c] + d_from_stencil`
   Host swaps A/B pointers between iterations.

   **No atomicAdd on d_state anywhere.** Each thread writes exactly one
   cell's d_state within its organism's buffer. Reads from neighbors are
   safe (all d_perc was written in the previous kernel launch).

**Why this works without cooperative groups**: Each sub-kernel launch
completes before the next begins (implicit barrier from sequential
`<<<...>>>` on the same stream). Within each kernel, each organism is
a single block — `__syncthreads()` suffices for intra-block coordination
(shared-memory reductions). No organism ever reads another organism's
data, so no inter-block sync is needed.

**Host launch** (in `launch_backward_all`):
```
bwd_zero_kernel<<<POOL_SIZE, 256, 0, stream>>>(...);
bwd_seed_avgpool_kernel<<<POOL_SIZE, 256, smem, stream>>>(...);
bwd_seed_scatter_kernel<<<POOL_SIZE, 256, smem, stream>>>(...);

for (int seg = 3; seg >= 0; --seg) {
    for (int local_step = CHECKPOINT_INTERVAL; local_step >= 1; --local_step) {
        bwd_load_checkpoint_kernel<<<POOL_SIZE, 256, 0, stream>>>(seg, ...);
        for (int fwd = 0; fwd < local_step - 1; ++fwd) {
            bwd_reforward_step_kernel<<<POOL_SIZE, 256, 0, stream>>>(...);
            // swap curr/next pointers
        }
        bwd_weight_grad_kernel<<<POOL_SIZE, 256, smem, stream>>>(...);
        bwd_stencil_gather_kernel<<<POOL_SIZE, 256, 0, stream>>>(...);
        // swap dA/dB pointers
    }
}
```

Total kernel launches per generation: 3 (setup) + 4×Σ(k=1..16)(1 + (k-1) + 2)
= 3 + 4×(16 + 120 + 32) = 3 + 672 = 675 kernel launches. Each launch
processes all 64 organisms in parallel. On sm_86 (30 SMs), 64 blocks of
256 threads achieves full SM occupancy.

**Performance vs. sequential**: The sequential approach launched 64
cooperative kernels (each using 16 blocks on 30 SMs, ~53% occupancy)
with a host-side sync after each. The batched approach launches each
sub-kernel once for all organisms (64 blocks on 30 SMs, >100% occupancy
with 2+ blocks per SM). Kernel launch overhead is ~5-10μs per launch;
675 launches × 10μs = ~7ms overhead, negligible vs. compute.

### 4.3 came_step_kernel

**File**: `optimizer/came.cu`
**Purpose**: One CAME update step on the shared weight buffer.
**Grid**: `<<<ceil(TOTAL_WEIGHTS/256), 256>>>`
**Shared mem**: 0

Operates on the **shared** weights. The gradient input is the **mean**
of per-organism gradients.

CAME update per weight `i`:
```
g = mean_grad[i]
m[i] = beta1 * m[i] + (1-beta1) * g
v[i] = beta2 * v[i] + (1-beta2) * g^2
u = m[i] / (sqrt(v[i]) + eps)
instability = (u - prev_u[i])^2
c[i] = beta3 * c[i] + (1-beta3) * instability
confidence = 1 / (1 + c[i])
weights[i] -= lr * confidence * u + weight_decay * weights[i]
prev_u[i] = u
```

This is CAME per the blueprint: confidence-adjusted momentum. The
confidence term damps the step when the update direction is unstable.

### 4.4 extract_descriptor_kernel

**File**: `nca/engine.cu` (existing, working)
**Grid**: `<<<POOL_SIZE, 256>>>`
**Purpose**: Copy final bmap_64 from OrganismState.bmap_traj into the flat
d_descriptors buffer for bulk D→H transfer.

### 4.5 placeholder_forward_kernel

**File**: `predictor/hybrid_surprise.cu`
**Grid**: `<<<1, 256>>>`
**Purpose**: 3-layer MLP forward for placeholder regressor on a minibatch.

Input: bmap_64[BMAP_DIM] + task_emb[TASK_EMBED_DIM] = 48-d input.
h1 = gelu(W1 * input + b1) — 128-d.
h2 = gelu(W2 * h1 + b2) — 64-d.
out = W3 * h2 + b3 — 2-d: (fitness_hat, log_uncertainty).

### 4.6 placeholder_train_kernel

**File**: `predictor/hybrid_surprise.cu`
**Grid**: `<<<1, 256>>>`
**Purpose**: One AdamW step on the placeholder regressor.

Loss = Gaussian NLL: `0.5 * (exp(-s) * (y - mu)^2 + s)`.
Backprop through 3 layers. AdamW update on all 6 parameter groups
(W1,b1,W2,b2,W3,b3). lr = 1e-4 (pinned by spec).

### 4.7 aggregate_gradients_kernel

**File**: `optimizer/came.cu`
**Grid**: `<<<ceil(TOTAL_WEIGHTS/256), 256>>>`
**Purpose**: Average per-organism gradients into a single gradient buffer.

For each weight index `i`:
```
sum = 0
for org in 0..POOL_SIZE-1:
    sum += grads[org].dW_flat[i]
mean_grad[i] = sum / POOL_SIZE
```

### 4.8 btraj_gather_kernel

**File**: `autodiff/warp_tape.cu`
**Grid**: `<<<POOL_SIZE, 128>>>`
**Purpose**: Gather strided bmap_traj data from OrganismState[] into a
contiguous d_btraj buffer for bulk D→H transfer. Each organism's
bmap_traj is at a different stride within the OrganismState array.

---

## 5. Per-Generation Data Flow

```
GENERATION LOOP:
│
├─ [Host] Curriculum: assemble_classifier_batch (every CURRICULUM_INTERVAL gens)
├─ [Host] Bootstrap check: spawn predictor founders if archive >= ARCHIVE_HALF
│
├─ [Host] Set up h_fwd_inputs for each organism (role, device pointers)
├─ [T1: H→D] Copy h_fwd_inputs → d_fwd_inputs
│              Copy batch images → d_batch_image
│              Copy task embedding → d_batch_task_emb
│
├─ [GPU] forward_with_checkpoints <<<POOL_SIZE, (16,16)>>>
│        Reads:  d_organisms, d_fwd_inputs, d_weights
│        Writes: d_organisms (final grid + bmap_traj), d_checkpoints
│
├─ [GPU] extract_descriptor <<<POOL_SIZE, 256>>>
│        Reads:  d_organisms[*].bmap_traj
│        Writes: d_descriptors
│
├─ [GPU] btraj_gather <<<POOL_SIZE, 128>>>
│        Reads:  d_organisms[*].bmap_traj
│        Writes: d_btraj (contiguous)
│
├─ [T2: D→H] Copy d_descriptors → h_descriptors
│              Copy d_btraj → h_btraj (→ IntentRegistry)
├─ [Sync]
│
├─ [Host] Score each organism using h_descriptors:
│         Classifiers: cross-entropy loss → task_accuracy → f_raw
│         Predictors:  MSE loss → prediction_accuracy → f_raw
│         Compose: f = f_raw × sot_gate(f_sot) × role_mult × audit_mult
│
├─ [Host] Compute h_seed_grad from actual loss gradient:
│         Classifiers: d(CE)/d(logits) padded to BMAP_DIM
│         Predictors:  d(MSE)/d(bmap_64)
│
├─ [Host] Archive insertion (RFF project, assign_bin, novelty, insert)
│
├─ [T3: H→D] Copy h_seed_grad → d_seed_grad
│
├─ [Host] PT swaps (if gen % PT_SWAP_INTERVAL == 0):
│         propose_swaps with full organism data exchange (section 13)
│         Must occur BEFORE backward so swapped organisms contribute
│         gradients in their new replica context.
│
├─ [GPU] backward sub-kernels <<<POOL_SIZE, 256>>> (batched, all orgs parallel)
│        ~675 sequential kernel launches, each processing all organisms.
│        Reads:  d_checkpoints[org], d_weights, d_seed_grad[org],
│                d_bwd_d_state[org][0..1], d_bwd_d_perc[org],
│                d_bwd_recomp[org][0..1]
│        Writes: d_grads[org]
│
├─ [GPU] aggregate_gradients <<<ceil(TOTAL_W/256), 256>>>
│        Reads:  d_grads[0..POOL_SIZE-1]
│        Writes: d_mean_grad
│
├─ [GPU] grad_norm_reduce <<<ceil(TOTAL_W/256), 256>>>
│        Reads:  d_mean_grad
│        Writes: d_grad_norm (1 float, pinned host scalar)
│
├─ [GPU] came_step <<<ceil(TOTAL_W/256), 256>>>
│        Reads:  d_mean_grad, d_came_{m,v,c,prev_u}
│        Writes: d_weights, d_came_{m,v,c,prev_u}
├─ [Sync]
│
├─ [Host] Gradient health check: read d_grad_norm, compare threshold
│
├─ [Host] Spawn wave:
│         Select WAVE_SIZE parents from archive (role-proportional).
│         Copy entry.genome, mutate, replace worst pool member,
│         init_delta_from_prior. (Host-only; organism grid state
│         will be overwritten on next forward.)
│
├─ [Host] Placeholder training:
│         Push (bmap_64, task_emb, fitness) to replay buffer.
│         [GPU] placeholder_train <<<1, 256>>>
│         [Sync]
│
├─ [Host] Surprise + CUSUM
│
├─ [Host] Periodic: PT swaps, PCA rebin, audit, probe panel
│
└─ generation++
```

---

## 6. Scoring: Classification Loss (A-401)

`f_raw_classifier = task_accuracy × gate(f_sot)`.

Task accuracy is from **cross-entropy classification loss**, computed on
the host from h_descriptors:

```
logits = h_descriptors[org * BMAP_DIM + 0..NUM_CLASSES-1]   // first NUM_CLASSES dims
loss = cross_entropy(logits, target_class)
task_accuracy = exp(-loss)                                    // (0, 1]
```

NUM_CLASSES = 16 is a named constant distinct from CA_CHANNELS. The logit
count is governed by the classification task, not the substrate channel
count. Code must use NUM_CLASSES for loss computation and seed gradient
sizing, never CA_CHANNELS.

Seed gradient:
```
h_seed_grad[org * BMAP_DIM + 0..NUM_CLASSES-1] = softmax(logits) - one_hot(target)
h_seed_grad[org * BMAP_DIM + NUM_CLASSES..BMAP_DIM-1] = 0
```

Uses `classifier_loss` (proven in host tests, to be included in
warp_tape.cu as a host-callable function).

---

## 7. Scoring: Predictor MSE Loss (A-601)

```
target = intent_registry[target_org].bmap_64
prediction = h_descriptors[org * BMAP_DIM + 0..31]
mse = mean((prediction - target)²)
prediction_accuracy = 1 / (1 + mse)
```

Seed gradient:
```
h_seed_grad[org * BMAP_DIM + d] = 2 * (prediction[d] - target[d]) / BMAP_DIM
```

---

## 8. CAME (A-501)

One shared weight buffer, one CAME state, one gradient = mean across all
organisms. Per the blueprint: "Operates uniformly across roles; only the
loss feeding the backward pass differs."

**Gradient health monitoring**: After aggregation, compute the L2 norm of
d_mean_grad via a device-side parallel reduction kernel:
`<<<ceil(TOTAL_WEIGHTS/256), 256>>>` with shared-memory tree reduction,
writing one float to the pinned `d_grad_norm` scalar. One float D→H
transfer (not the full weight vector). If the norm falls below
`eps_grad * TOTAL_WEIGHTS` (eps_grad = 1e-8) for GRAD_HEALTH_WINDOW = 10
consecutive generations, log a warning — this indicates gradient
cancellation (e.g., classifier and predictor gradients opposing each other).
If sustained, the operator can inspect per-role gradient norms from the
telemetry log. No automatic remediation (the blueprint specifies uniform
operation); the warning is a diagnostic signal.

Hyperparameters (pinned by spec):
- lr = 1e-3
- beta1 = 0.9, beta2 = 0.999, beta3 = 0.999
- epsilon = 1e-8
- weight_decay = 0.01

CAME state: 4 arrays × TOTAL_WEIGHTS (2587) × 4 bytes = 40.4 KB total.

---

## 9. Spawn and Genome-Archive Linkage

`ArchiveEntry` includes a `Genome genome` field (+128 bytes per entry,
640 KB total for 5000 entries).

Spawn procedure (per spawn in WAVE_SIZE):
1. Determine child role: role-proportional allocation based on current
   archive role fraction, with minimum 2 per role to prevent extinction.
   Before bootstrap, all spawns are classifiers.
2. Select parent from the per-role **live index list** in the Archive
   struct: uniform random via PCG32 into the list. O(1), always succeeds.
3. Copy `entry.genome` as the child genome.
4. Replace the worst-fitness organism **of the same role** in the pool.
   The child inherits the replaced organism's pool slot and `replica_tag`
   (which determines its mutation rate from S-004).
5. Mutate at the inherited replica's mutation rate (`PT_MUTATION_RATES[replica_tag]`
   for non-role bits, `MUTATION_RATE_ROLE` for role bits). `mutate` takes a
   `Pcg32*` parameter (the host RNG).
6. `init_delta_from_prior` on the new genome. This seeds a local `Pcg32`
   from the genome's 32-bit seed for per-organism determinism.

**Archive live index lists**: The Archive struct maintains two arrays
`alive_classifier_idx[MAX_ARCHIVE]` and `alive_predictor_idx[MAX_ARCHIVE]`
with counts `n_alive_classifier` and `n_alive_predictor`. These are
updated on every `insert` (append new index) and eviction (swap-remove
the evicted index). Parent selection indexes into the appropriate list.
Memory cost: 2 × 5000 × 4 bytes = 40 KB. No rejection sampling, no
linear scans.

No index hacks. The genome is stored in the archive alongside the
descriptor.

---

## 9.1 Archive Binning Policy

The blueprint says "20×20 PCA bins computed periodically on the union
archive. Each bin has per-role capacity caps." This section specifies
the binning lifecycle that the blueprint leaves to engineering.

**Initial state.** At world init, all 400 bins have uniform per-role
caps: `cap_classifier = cap_predictor = MAX_ARCHIVE / ARCHIVE_BINS
= 5000 / 400 = 12`, rounded up to 13. No PCA exists yet, so entries
inserted before the first rebin are assigned bins via a **hash-based
fallback**: `bin_x = hash(descriptor) % 20`, `bin_y = hash(descriptor,
seed2) % 20`. This spreads entries across bins from generation 0,
preventing a single-bin bottleneck that would cap archive growth at 13.

The hash function is deterministic: `bin_x = uint32_t(descriptor[0] *
1e6) % 20`, `bin_y = uint32_t(descriptor[1] * 1e6) % 20`. This uses
the first two descriptor dimensions as a crude 2D projection — not
meaningful geometrically, but sufficient to distribute entries until
the PCA provides real structure.

**Rebin schedule.** `recompute_bins` runs at every `AUDIT_INTERVAL`
(100) generations. On rebin:
1. PCA is fitted on all alive entries (power iteration, already
   implemented).
2. All entries are re-projected and bin assignments updated.
3. Per-bin per-role counts are recounted.
4. Caps remain at the uniform 13 (the blueprint does not specify
   adaptive caps; uniform caps are the simplest correct policy that
   satisfies "per-role capacity caps").

**Insertion binning.** After the first rebin, new entries are assigned
bins by projecting onto the stored PCA vectors (PC0, PC1) and the
stored extents (min0, max0, min1, max1). The archive must store these
6 values (2 PCA vectors × BMAP_DIM + 4 extent scalars) so that
`insert` can compute bin assignments between rebins.

Before the first rebin (generation < AUDIT_INTERVAL), the hash-based
fallback is used.

**Archive struct additions** (in `soft_qd_archive.cu`):
```
float pc[2][BMAP_DIM];    // PCA vectors from last rebin
float pc_min[2];          // projection extents
float pc_max[2];
float pc_mean[BMAP_DIM];  // centering mean from last rebin
bool  pca_valid;          // false until first recompute_bins
```

**assign_bin function**: Given a descriptor, returns (bin_x, bin_y).
If `pca_valid`, projects onto PC0/PC1 using stored vectors and extents.
Otherwise, uses the hash fallback. This function is called by the
integration layer at insertion time to set `cand.bin_x` and
`cand.bin_y`.

---

## 10. Backward Workspace Buffers

In addition to the two d_state buffers (GRID²×CH×4 = 256 KB each), the
backward kernel requires a d_perc buffer for the two-phase gather pattern:

| Buffer | Per-org size | Total (×POOL_SIZE) | Notes |
|--------|-------------|-------------------|-------|
| `d_bwd_d_state[0]` | 256 KB | 16 MB | FP32 grid, d_state_A |
| `d_bwd_d_state[1]` | 256 KB | 16 MB | FP32 grid, d_state_B |
| `d_bwd_d_perc` | 768 KB | 48 MB | 4096 cells × 48 floats × 4 bytes |
| `d_bwd_recomp[0]` | 128 KB | 8 MB | FP16 grid for re-forward curr |
| `d_bwd_recomp[1]` | 128 KB | 8 MB | FP16 grid for re-forward next |

Total backward workspace: 96 MB (1.5 MB × 64 organisms). All organisms
have private workspaces so the batched backward sub-kernels can process
all organisms in parallel without inter-organism data hazards.

The re-forward workspace (`d_bwd_recomp`) is separate from d_organisms
so the organism's grid is not clobbered during backward.

---

## 11. Phase Graph Execution (A-102)

Phase graphs capture kernel launch sequences into CUDA graphs for replay.

| Phase | Kernels captured |
|-------|-----------------|
| Forward | forward_with_checkpoints + extract_descriptor + btraj_gather |
| Backward | ~675 batched sub-kernels (bwd_zero, bwd_seed_*, bwd_reforward, bwd_weight_grad, bwd_stencil_gather) |
| Optimizer | aggregate_gradients + came_step |
| WorldPredict | placeholder_forward on probe set |
| WorldTrain | placeholder_train |
| StressEval | forward_with_checkpoints on stress slots |

Host-only phases (curriculum, scoring, archive, spawn, surprise, periodic)
are not captured.

Graph capture after the first generation. Replay on subsequent generations.
The Backward phase graph contains ~675 batched sub-kernel launches
(all organisms processed in parallel per launch).

---

## 12. Safety/Alignment Functions (S-002)

### apply_sot_identity
Host function. For SOT-marked task images, computes f_sot as cosine
similarity between the organism's bmap_64 and a reference bmap_64 from
the un-permuted image.

Since all organisms share the same substrate weights (W_perc, W_inter,
W_flow, W_bmap), the reference bmap_64 for a given un-permuted image
is the same for all organisms. Procedure:
1. For each unique SOT image in the batch (up to SOT_SUBBATCH=4), run
   a single reference forward pass (reusing a temporary organism slot
   in the stress range, index >= POOL_SIZE) on the un-permuted image
   to produce reference_bmap_64.
2. For each organism assigned to that SOT image, compute:
   `f_sot = cosine_similarity(organism_bmap_64, reference_bmap_64)`.
3. For non-SOT organisms, f_sot = 1.0 (gate passes fully).

The reference forward passes (up to 4) run after the main forward pass
and before scoring. They reuse `forward_kernel` (not the checkpointed
version — no backward needed for the reference).

All device buffers used by apply_sot_identity are **pre-allocated** in
the World struct (d_sot_temp_images, d_sot_task_emb, d_sot_fwd_inputs,
d_sot_descriptors — see section 2.1). No cudaMalloc/cudaFree calls
occur within the generation loop. Sizes are bounded by SOT_SUBBATCH.

### poll_off_switch
Host function. Checks for `shutdown.flag` file. Returns true → stop.

### apply_operator_command
Host function. Reads `operator_cmd.txt`. Supports:
- `prune <lineage_id>` — mark lineage for removal
- `pause` / `resume` — suspend/resume generation loop
- `checkpoint` — force checkpoint write

---

## 13. Structural Pressure Launchers (S-003, S-004)

All structural pressure computations run on the **host** using
h_descriptors and fitness values copied from the device.

PT operations:
- `record_best_fitness`: host scan of per-replica best fitness, called
  once per generation.
- `propose_swaps(MutationLadder*, const float* organism_fitness,
  Pcg32* rng, SwapContext&)`: host Metropolis acceptance on each
  adjacent replica pair. On acceptance, performs a **full organism data
  swap** of ALL organisms between the two replicas:
  - For each organism pair (one from R_lo, one from R_hi at the same
    offset within the replica): swap OrganismState via device memcpy
    through d_pt_swap_org temp buffer, swap CheckpointBuffer via
    d_pt_swap_ckpt, swap GradBuffers via d_pt_swap_grad. Swap
    OrganismTable rows (host memcpy: genome, delta, lineage_id,
    parent_id, spawn_gen, fitness, f_raw, f_sot, role, batch_sample_idx).
  - The pool slot's replica_tag stays fixed to the slot — it identifies
    the temperature, not the organism.
  - Swap timing: before backward in the generation loop.
  - Data movement: ~1.3 MB device memcpy per organism pair. 16 pairs per
    accepted swap, amortized over 50 generations.
- `refresh_stress_slots`: host copies genomes into stress organism slots,
  then the next forward pass (on stress slots) overwrites their grids.
- `evaluate_stress`: forward_with_checkpoints on stress slot range.
- `flag_stress_failures`: host scan of stress SOT gate results.

---

## 14. Checkpoint Serialization (S-001)

Binary format. Write in order:
1. Header: magic, version, schema_hash (CRC32 of struct sizes), generation.
2. SharedWeights (D→H transfer first, then write).
3. CAME state (D→H, then write: m, v, c, prev_u).
4. OrganismTable metadata (host-resident: genomes, deltas, lineage_id,
   parent_id, spawn_gen, replica_tag).
5. OrganismState grids (D→H transfer of d_organisms grids, then write).
6. Archive entries (all alive entries, host-resident).
7. PlaceholderRegressor (host-resident).
8. ReplayBuffer (host-resident).
9. CorrelationWindow (host-resident).
10. CusumState × 2.
11. MutationLadder, StressLadder.
12. SentinelEnsemble + SentinelHistory.
13. Scalar state: generation, bootstrap_fired, bootstrap_gen, s_target,
    s_target_calibrated, rng, host_sot_key.

Load reverses the order (read, then H→D transfers). Schema hash mismatch
= reject.

---

## 15. Initialization (A-501, G-100)

### 15.1 PRNG

All randomness uses PCG32 (O'Neill 2014). State: 64-bit state + 64-bit
stream. Default seed: `state = 0x853C49E6748FEA9B, stream =
0xDA3E39CB94B95BDB`. The World struct stores a `Pcg32 rng` field. No
other PRNG exists in the codebase — per-organism deterministic sequences
(e.g., `init_delta_from_prior`) seed a local `Pcg32` from the genome's
32-bit seed rather than using a different algorithm.

### 15.2 Weight Initialization

Shared weights are initialized via Kaiming He initialization, per-layer:

| Layer | Offset | Count | fan_in | Activation | Scale |
|-------|--------|-------|--------|------------|-------|
| W_perc | OFF_PERC | W_PERC_SIZE (27) | 9 | none | sqrt(1/9) |
| W_inter | OFF_INTER | INTER_SIZE | N_PERC_FILTERS × CA_CHANNELS (48) | GELU | sqrt(2/48) |
| W_flow | OFF_FLOW | FLOW_SIZE | HIDDEN_DIM | linear | sqrt(1/HIDDEN_DIM) |
| W_bmap | OFF_BMAP | BMAP_SIZE | CA_CHANNELS (16) | linear | sqrt(1/16) |

Each weight is drawn from N(0, scale) using Box-Muller on pairs of PCG32
outputs. The init sub-sequence uses a seed derived from one PCG32 draw on
the host RNG.

### 15.3 SOT Key

Default: `0xDEADCAFE42ULL`. Production runs should override from system
entropy. Stored in World::host_sot_key.

### 15.4 Genome Seeds

Each organism's 32-bit seed (A-301 bits 2-33) is drawn from the host
PCG32 RNG at init. No index-based formulas.

### 15.5 PT Ladder Initial State

- `beta = 1.0f` (EMA corrects within 3-4 swap rounds).
- `accept_ema = PT_TARGET_ACCEPT` (start at target).
- `swaps_attempted = 0, swaps_accepted = 0`.
- `history_head = 0`, all `best_fitness_history` zeroed.
- Replica tags: organism i gets `replica_tag = i / PT_REPLICA_SIZE`.

### 15.6 CUSUM Initial State

Provisional parameters (before calibration completes):
- Surprise CUSUM: `k = 0.5, h = 5.0`.
- r CUSUM: `k_r = 0.1, h_r = 3.0`.
Accumulators start at zero. After calibration (s_target frozen), parameters
are recomputed from observed σ and accumulators are reset.

### 15.7 Main-Pool SOT Density

`MAIN_SOT_DENSITY = 0.05` (5%). This is the fraction of classifier batch
images that carry SOT marking. Distinct from STRESS_SOT_DENSITIES.
Used in `assemble_classifier_batch`.

### 15.8 Telemetry

`TELEMETRY_INTERVAL = 10`. Log every 10 generations AND during the first
5 generations. Pinned constant.

---

## 16. Global Context Channel (A-203)

Wave 8 addition. Requires no changes to Waves 1–7 code when disabled.

### 16.1 W_ctx Weight Layer

| Layer | Offset | Count | fan_in | Activation | Scale |
|-------|--------|-------|--------|------------|-------|
| W_ctx | OFF_CTX = OFF_BMAP + W_BMAP_SIZE | 32 | CA_CHANNELS (16) | linear | sqrt(1/16) |

TOTAL_WEIGHTS increases by 32 (from 2587 to 2619). All CAME state arrays,
gradient buffers, and aggregate kernels grow by 32 elements. The VRAM impact
is negligible (~512 bytes across all weight-sized buffers).

### 16.2 Forward Kernel Modification

At each bmap sample step (16, 32, 48, 64), after `project_bmap` computes the
16-d summary s_t, the forward kernel broadcasts a context vector into channels
14–15 of every cell:

```
ctx[0..1] = W_ctx^T * s_t    // 2 = CH_AUX_LAST - CH_AUX_FIRST + 1
for each cell (y, x):
    grid[y][x][CH_AUX_FIRST + 0] = ctx[0]
    grid[y][x][CH_AUX_FIRST + 1] = ctx[1]
```

This executes within the existing `project_bmap` call site — no new kernel
launch. The summary s_t is already in shared memory from the reduction; the
broadcast is a write loop over the thread's assigned cells.

For predictors at step 0, channels 14–15 are seeded with bmap_32 target
(A-201). The first context broadcast at step 16 overwrites this.

### 16.3 Backward Kernel Modification

The backward pass through the context broadcast step:
1. Accumulate d_ctx[0..1] from all cells' channel 14–15 gradients (sum
   across all cells — spatially uniform broadcast means uniform adjoint).
2. `dW_ctx[c * 2 + k] += s_t[c] * d_ctx[k]` for weight gradient.
3. `d_summary[c] += W_ctx[c * 2 + 0] * d_ctx[0] + W_ctx[c * 2 + 1] * d_ctx[1]`
   for backprop through to the summary, which then flows into the existing
   project_bmap adjoint.

This adds one reduction (d_ctx accumulation) and one outer product (dW_ctx)
per bmap step in the backward pass.

### 16.4 Configuration

```
constexpr bool GLOBAL_CONTEXT_ENABLED = false;  // Wave 8 activates
constexpr int W_CTX_SIZE = CA_CHANNELS * (CH_AUX_LAST - CH_AUX_FIRST + 1);  // 32
```

When `GLOBAL_CONTEXT_ENABLED` is false, W_ctx is not allocated, TOTAL_WEIGHTS
remains 2587, and the forward/backward kernels skip the broadcast step. The
flag is a compile-time constant — no runtime branching in the kernel.
