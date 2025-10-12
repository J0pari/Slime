# Slime Mold Transformer: 2025 Modernization Roadmap

**Status**: Design Document
**Target**: Late 2025 Cutting-Edge Architecture
**Foundation**: Flow-Lenia + Neural CA + Learned Everything

---

## Executive Summary

Current implementation is 2020-era Conway/QD thinking wrapped in 2024 engineering. This roadmap upgrades to genuine 2025 cutting-edge: **Flow-Lenia-based, fully learned, self-organizing neural cellular automata with intrinsic motivation**.

### The Gap

| Component | Current (Static) | Target (Learned) |
|-----------|------------------|------------------|
| **Dimensions** | Fixed Kernel PCA (62D → 3-5D) | DIRESA adaptive encoder (learned D) |
| **Archive** | Fixed CVT with 50 centroids | Adaptive Voronoi (grow/shrink) |
| **Metrics** | Hardcoded (Euclidean/Mahalanobis) | Learned behavioral embedding |
| **Update Rules** | Manual transformer attention | NCA learned update functions |
| **Lifecycle** | Manual spawning schedule | Curiosity-driven management |
| **Evolution** | Fixed architecture | Flow-Lenia parameter localization |

---

## Research Foundation (2024-2025)

### 1. Flow-Lenia [arXiv:2212.07906, MIT Press Artificial Life 2025]

**Key Innovation**: Mass-conservative continuous CA with localized parameters

```python
# Flow-Lenia Update Rule (mass-conservative)
def flow_lenia_update(state, params_field):
    """
    state: Current CA state (continuous values)
    params_field: Spatially localized update rule parameters

    Returns: Next state (conserves total mass)
    """
    # Kernel convolution with LOCAL parameters
    growth = convolve(state, kernel(params_field))

    # Growth function (localized)
    delta = growth_func(growth, params_field.mu, params_field.sigma)

    # Mass-conservative update (∑ state_t = ∑ state_{t+1})
    next_state = state + delta * params_field.dt
    return normalize_mass(next_state, total_mass=state.sum())
```

**Benefits**:
- **Spatially Localized Patterns (SLPs)**: Creatures with coherent local rules
- **Multi-species**: Different regions evolve different update rules
- **Parameter Evolution**: Rules themselves can evolve over time
- **Open-ended**: Enables genuine emergence without manual design

### 2. Universal Neural CA [arXiv:2505.13058, May 2025]

**Key Innovation**: Learned update rules via gradient descent, universal computation

```python
# Neural CA Update Rule (fully differentiable)
class NeuralCAUpdateRule(nn.Module):
    def __init__(self, state_dim=16, hidden_dim=128):
        super().__init__()
        self.perception = nn.Conv2d(state_dim, hidden_dim, 3, padding=1)
        self.update = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, state_dim, 1)
        )

    def forward(self, state):
        """Fully learned local update (perceive → decide → update)"""
        perceived = self.perception(state)
        delta = self.update(perceived)
        return state + delta  # Residual connection for stability
```

**Benefits**:
- **Learned Dynamics**: No hand-crafted rules, learns from data
- **Universal Computation**: Can emulate any computation (Turing-complete)
- **Differentiable**: Train end-to-end via backprop
- **Compositional**: Can learn subroutines (matrix ops, NN inference, etc.)

### 3. DIRESA Adaptive Encoder [arXiv:2404.18314, April 2025]

**Key Innovation**: Distance-preserving learned dimensionality reduction

```python
# DIRESA: Distance-Regularized Siamese Autoencoder
class DIRESA(nn.Module):
    def __init__(self, input_dim=62, min_latent=2, max_latent=10):
        super().__init__()
        # Adaptive latent dimension (learned via architecture search)
        self.encoder = AdaptiveEncoder(input_dim, max_latent)
        self.decoder = AdaptiveDecoder(max_latent, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

    def loss(self, x1, x2, z1, z2):
        """Distance-preserving + reconstruction + uncorrelation"""
        # Preserve pairwise distances
        dist_input = torch.norm(x1 - x2, dim=-1)
        dist_latent = torch.norm(z1 - z2, dim=-1)
        distance_loss = F.mse_loss(dist_latent, dist_input)

        # Reconstruction
        recon_loss = F.mse_loss(self.decoder(z1), x1)

        # Uncorrelated latents (like PCA)
        cov = torch.cov(z1.T)
        uncorr_loss = torch.norm(cov - torch.diag(cov.diag()))

        return distance_loss + recon_loss + uncorr_loss
```

**Benefits**:
- **Adaptive D**: Learns optimal dimensionality (not fixed 3-5D)
- **Metric Preservation**: Maintains behavioral space geometry
- **Interpretable**: Latent dims have physical meaning
- **Nonlinear**: Captures complex manifold structure (vs linear PCA)

### 4. Intrinsic Motivation & Curiosity [NeurIPS 2024 IMOL Workshop]

**Key Innovation**: Learning progress as intrinsic reward

```python
class CuriosityDrivenLifecycle:
    """Manages pseudopod lifecycle via learning progress"""

    def __init__(self, archive):
        self.archive = archive
        self.learning_progress = {}  # pod_id -> progress history

    def compute_learning_progress(self, pod_id):
        """
        Learning Progress = improvement rate in coverage
        High LP = exploring productively (keep alive)
        Low LP = stagnating (consider retiring)
        """
        recent_discoveries = self.archive.get_recent_by_pod(pod_id, window=100)

        # Did this pseudopod discover new niches?
        novelty = sum(1 for d in recent_discoveries if d.is_novel)

        # Is it improving existing niches?
        improvement = sum(d.fitness_delta for d in recent_discoveries if d.fitness_delta > 0)

        return novelty + improvement  # Combined exploration + exploitation

    def should_spawn_new(self):
        """Spawn when archive has unexplored regions"""
        coverage = self.archive.coverage()  # % of centroids filled
        avg_lp = np.mean(list(self.learning_progress.values()))

        # Spawn if: low coverage OR all pods stagnating
        return coverage < 0.8 or avg_lp < threshold

    def should_retire(self, pod_id):
        """Retire when no learning progress for N generations"""
        lp_history = self.learning_progress[pod_id]
        recent_lp = lp_history[-20:]  # Last 20 gens

        return np.mean(recent_lp) < 0.01  # Near-zero progress
```

---

## Phase 4: Learned Behavioral Encoder (DIRESA Integration)

**Goal**: Replace fixed Kernel PCA with adaptive learned dimensionality

### 4.1 Architecture

```python
# slime/behavioral/diresa_encoder.py

class DIRESABehavioralEncoder(nn.Module):
    """
    Distance-preserving adaptive behavioral encoder.
    Learns optimal dimensionality from archive evolution.
    """

    def __init__(
        self,
        input_dim: int = 62,
        min_dims: int = 2,
        max_dims: int = 10,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()
        self.input_dim = input_dim
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.device = device

        # Siamese twin encoders for distance preservation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, max_dims),  # Will learn to use subset
        )

        self.decoder = nn.Sequential(
            nn.Linear(max_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

        # Learned dimension selector (which of max_dims to use)
        self.dim_selector = nn.Parameter(torch.ones(max_dims))

    def forward(self, raw_metrics: torch.Tensor) -> torch.Tensor:
        """Encode raw metrics to adaptive behavioral space"""
        z_full = self.encoder(raw_metrics)

        # Apply learned dimension gating (soft selection)
        dim_weights = torch.sigmoid(self.dim_selector)
        z = z_full * dim_weights

        return z

    def compute_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """DIRESA loss: distance + reconstruction + sparsity"""
        z1, z2 = self.forward(x1), self.forward(x2)
        x1_recon, x2_recon = self.decoder(z1), self.decoder(x2)

        # 1. Distance preservation
        dist_input = torch.norm(x1 - x2, dim=-1)
        dist_latent = torch.norm(z1 - z2, dim=-1)
        dist_loss = F.mse_loss(dist_latent, dist_input)

        # 2. Reconstruction
        recon_loss = F.mse_loss(x1_recon, x1) + F.mse_loss(x2_recon, x2)

        # 3. Dimension sparsity (encourage using fewer dims)
        dim_weights = torch.sigmoid(self.dim_selector)
        sparsity_loss = torch.sum(dim_weights)  # L1 on active dims

        # 4. Uncorrelation (PCA-like)
        cov = torch.cov(torch.cat([z1, z2], dim=0).T)
        diag_cov = torch.diag(cov.diag())
        uncorr_loss = torch.norm(cov - diag_cov)

        total_loss = (
            1.0 * dist_loss +
            1.0 * recon_loss +
            0.1 * sparsity_loss +
            0.5 * uncorr_loss
        )

        return {
            'total': total_loss,
            'distance': dist_loss,
            'reconstruction': recon_loss,
            'sparsity': sparsity_loss,
            'uncorrelation': uncorr_loss,
            'active_dims': torch.sum(dim_weights > 0.5).item()
        }

    @property
    def behavioral_dims(self) -> int:
        """Dynamically determined dimensionality"""
        dim_weights = torch.sigmoid(self.dim_selector)
        return int(torch.sum(dim_weights > 0.5).item())
```

### 4.2 Training Strategy

```python
class DIRESATrainer:
    """Online training of behavioral encoder during archive evolution"""

    def __init__(self, encoder: DIRESABehavioralEncoder, lr: float = 1e-4):
        self.encoder = encoder
        self.optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
        self.loss_history = []

    def update(self, archive: CVTArchive, batch_size: int = 32):
        """
        Update encoder using recent archive additions.
        Called every N generations to adapt to evolving distribution.
        """
        # Sample pairs from archive
        elites = archive.sample_elites(batch_size * 2)
        x1, x2 = elites[:batch_size], elites[batch_size:]

        # Convert to tensors
        x1 = torch.tensor([e.raw_metrics for e in x1], device=self.encoder.device)
        x2 = torch.tensor([e.raw_metrics for e in x2], device=self.encoder.device)

        # Compute loss and update
        losses = self.encoder.compute_loss(x1, x2)

        self.optimizer.zero_grad()
        losses['total'].backward()
        self.optimizer.step()

        self.loss_history.append(losses)

        return losses
```

### 4.3 Integration with CVTArchive

```python
# Modify slime/memory/archive.py

class CVTArchive:
    def __init__(self, config, encoder_type='diresa', ...):
        # ...existing code...

        if encoder_type == 'kpca':
            self.encoder = KernelPCAEncoder(...)  # Current
        elif encoder_type == 'diresa':
            self.encoder = DIRESABehavioralEncoder(
                input_dim=config.behavioral_space.num_raw_metrics,
                min_dims=config.behavioral_space.min_dims,
                max_dims=config.behavioral_space.max_dims,
                device=self.device
            )
            self.encoder_trainer = DIRESATrainer(self.encoder)

    def add(self, raw_metrics, fitness, state_dict, ...):
        """Add elite with learned encoding"""

        # Encode using DIRESA (adaptive dimensionality)
        with torch.no_grad():
            behavior = self.encoder(torch.tensor(raw_metrics, device=self.device))

        # Convert to tuple for storage
        behavior = tuple(behavior.cpu().numpy())

        # ... existing add logic ...

    def update_encoder(self):
        """Called every N generations to retrain encoder"""
        if isinstance(self.encoder, DIRESABehavioralEncoder):
            losses = self.encoder_trainer.update(self, batch_size=32)

            logger.info(
                f'DIRESA update: active_dims={losses["active_dims"]}, '
                f'dist_loss={losses["distance"]:.4f}, '
                f'recon_loss={losses["reconstruction"]:.4f}'
            )
```

---

## Phase 5: Adaptive Voronoi Archive

**Goal**: Replace fixed CVT with growing/shrinking cells based on coverage

### 5.1 Dynamic Cell Management

```python
# slime/memory/adaptive_voronoi.py

class AdaptiveVoronoiArchive:
    """
    Voronoi cells that grow/shrink/split/merge based on density.
    High-density regions get more centroids, low-density merge.
    """

    def __init__(
        self,
        config,
        min_centroids: int = 20,
        max_centroids: int = 200,
        split_threshold: int = 10,  # Split when >10 elites in cell
        merge_threshold: int = 1,   # Merge when <1 elite in cell
        device: torch.device = torch.device('cuda')
    ):
        self.config = config
        self.min_centroids = min_centroids
        self.max_centroids = max_centroids
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self.device = device

        # Start with min centroids
        self.centroids = self._initialize_centroids(min_centroids)
        self.elites: dict[int, Elite] = {}
        self.density: dict[int, int] = {}  # centroid_id -> count

    def add(self, behavior, fitness, state_dict, ...):
        """Add elite and potentially split/merge cells"""
        centroid_id = self._find_nearest_centroid(behavior)

        # Update density
        self.density[centroid_id] = self.density.get(centroid_id, 0) + 1

        # Standard addition logic
        added = self._add_to_cell(centroid_id, behavior, fitness, state_dict)

        # Adaptive cell management every 100 additions
        if len(self.elites) % 100 == 0:
            self._adapt_cells()

        return added

    def _adapt_cells(self):
        """Split high-density cells, merge low-density cells"""
        # Find cells to split (high density)
        to_split = [
            cid for cid, count in self.density.items()
            if count > self.split_threshold and len(self.centroids) < self.max_centroids
        ]

        # Find cells to merge (low density)
        to_merge = [
            cid for cid, count in self.density.items()
            if count < self.merge_threshold and len(self.centroids) > self.min_centroids
        ]

        # Execute splits
        for cid in to_split:
            self._split_cell(cid)

        # Execute merges
        for cid in to_merge:
            self._merge_cell(cid)

        logger.info(
            f'Adaptive Voronoi: {len(self.centroids)} cells '
            f'(+{len(to_split)} splits, -{len(to_merge)} merges)'
        )

    def _split_cell(self, centroid_id: int):
        """Split high-density cell into two"""
        # Get all elites in this cell
        elites_in_cell = [e for e in self.elites.values() if e.centroid_id == centroid_id]

        if len(elites_in_cell) < 2:
            return  # Can't split

        # K-means split into 2 sub-cells
        behaviors = np.array([e.behavior for e in elites_in_cell])
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42).fit(behaviors)

        # Create two new centroids
        new_c1 = kmeans.cluster_centers_[0]
        new_c2 = kmeans.cluster_centers_[1]

        # Remove old centroid, add two new ones
        old_centroid = self.centroids[centroid_id]
        self.centroids = np.delete(self.centroids, centroid_id, axis=0)
        self.centroids = np.vstack([self.centroids, new_c1, new_c2])

        # Reassign elites
        for elite in elites_in_cell:
            new_cid = self._find_nearest_centroid(elite.behavior)
            elite.centroid_id = new_cid

    def _merge_cell(self, centroid_id: int):
        """Merge low-density cell with nearest neighbor"""
        # Find nearest neighboring centroid
        neighbors = self._find_k_nearest_centroids(centroid_id, k=1)
        merge_target = neighbors[0]

        # Move all elites to target
        for elite in self.elites.values():
            if elite.centroid_id == centroid_id:
                elite.centroid_id = merge_target

        # Remove centroid
        self.centroids = np.delete(self.centroids, centroid_id, axis=0)
        self.density[merge_target] += self.density.get(centroid_id, 0)
        del self.density[centroid_id]
```

---

## Phase 6: Neural CA Update Rules

**Goal**: Replace fixed transformer attention with learned NCA update functions

### 6.1 Pseudopod as Neural CA

```python
# slime/core/nca_pseudopod.py

class NeuralCAPseudopod(nn.Module):
    """
    Pseudopod with learned update rule (replaces transformer attention).
    Each pseudopod evolves its own local CA rule.
    """

    def __init__(
        self,
        state_dim: int = 64,  # Hidden state per cell
        perception_radius: int = 3,
        hidden_dim: int = 128,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()
        self.state_dim = state_dim
        self.perception_radius = perception_radius
        self.device = device

        # Perception: local neighborhood → features
        self.perception = nn.Conv2d(
            state_dim,
            hidden_dim,
            kernel_size=2*perception_radius+1,
            padding=perception_radius
        )

        # Update: features → state delta
        self.update = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, state_dim, 1)
        )

        # Alive mask (stochastic dropout of inactive cells)
        self.alive_threshold = 0.1

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        """
        Local perception via convolution.
        Each cell sees its neighborhood (like slime mold sensing).
        """
        return self.perception(state)

    def update_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Learned update rule (replaces transformer attention).

        state: (B, state_dim, H, W) - grid of cells
        returns: (B, state_dim, H, W) - updated grid
        """
        # Perceive neighborhood
        features = self.perceive(state)

        # Compute update
        delta = self.update(features)

        # Residual connection for stability
        next_state = state + delta

        # Alive mask (stochastic death of inactive cells)
        alive_mask = (state.max(dim=1, keepdim=True)[0] > self.alive_threshold).float()
        next_state = next_state * alive_mask

        return next_state

    def forward(self, state: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Run CA for N steps (like slime mold exploring over time).

        Equivalent to N transformer layers, but:
        - Fully local (no global attention)
        - Learned update rule (no manual Q/K/V)
        - Persistent state (not just token embeddings)
        """
        for _ in range(steps):
            state = self.update_state(state)
        return state

    def compute_output(self, state: torch.Tensor) -> torch.Tensor:
        """Extract output from final CA state (like transformer pooling)"""
        # Pool over spatial dimensions
        output = state.mean(dim=[2, 3])  # (B, state_dim)
        return output
```

### 6.2 Training NCA Update Rules

```python
class NCATrainer:
    """
    Train pseudopod update rules to solve tasks.
    Loss = prediction error + alive cell count (efficiency)
    """

    def __init__(self, nca_pseudopod: NeuralCAPseudopod, lr: float = 2e-4):
        self.nca = nca_pseudopod
        self.optimizer = torch.optim.Adam(nca.parameters(), lr=lr)

    def train_step(self, input_data: torch.Tensor, target: torch.Tensor):
        """
        Train NCA to transform input → target.

        input_data: (B, seq_len, input_dim) - sequence to process
        target: (B, output_dim) - prediction target
        """
        # Initialize CA state from input
        B, seq_len, input_dim = input_data.shape
        H = W = int(np.sqrt(seq_len))  # Spatial grid

        # Embed input into CA state
        state = self.embed_input(input_data)  # (B, state_dim, H, W)

        # Run CA update (learned dynamics)
        final_state = self.nca(state, steps=10)

        # Extract prediction
        pred = self.nca.compute_output(final_state)

        # Compute loss
        task_loss = F.mse_loss(pred, target)

        # Regularization: penalize too many alive cells (efficiency)
        alive_cells = (final_state.max(dim=1)[0] > self.nca.alive_threshold).sum()
        alive_loss = alive_cells / (B * H * W)  # Fraction alive

        total_loss = task_loss + 0.1 * alive_loss

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'task_loss': task_loss.item(),
            'alive_loss': alive_loss.item(),
            'total_loss': total_loss.item()
        }
```

---

## Phase 7: Flow-Lenia Parameter Localization

**Goal**: Make each pseudopod's update rule parameters PART of the CA state

### 7.1 Localized Parameters

```python
# slime/core/flow_lenia_pseudopod.py

class FlowLeniaPseudopod(nn.Module):
    """
    Neural CA + Flow-Lenia: update rules stored IN the state.
    Each spatial location has its own local update rule parameters.
    """

    def __init__(self, state_dim: int = 64, param_dim: int = 8):
        super().__init__()
        self.state_dim = state_dim  # Actual state channels
        self.param_dim = param_dim  # Parameter channels (μ, σ, growth, etc.)

        # Total channels = state + params (params are part of state!)
        self.total_dim = state_dim + param_dim

        # Perception sees BOTH state and local parameters
        self.perception = nn.Conv2d(
            self.total_dim,  # Perceive state + params
            128,
            kernel_size=3,
            padding=1
        )

        # Update rule depends on local parameters
        self.update = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, state_dim, 1)  # Update only state, not params
        )

        # Parameter evolution (slow dynamics)
        self.param_update = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, param_dim, 1)
        )

    def forward(self, state_with_params: torch.Tensor, steps: int = 10):
        """
        Run Flow-Lenia: state evolves guided by local parameters,
        parameters evolve slowly based on state.

        state_with_params: (B, state_dim + param_dim, H, W)
        """
        for _ in range(steps):
            state_with_params = self.update_step(state_with_params)
        return state_with_params

    def update_step(self, state_with_params: torch.Tensor) -> torch.Tensor:
        """
        One Flow-Lenia step:
        1. Perceive neighborhood (state + params)
        2. Update state using local param rules
        3. Slowly evolve parameters
        """
        # Split state and params
        state = state_with_params[:, :self.state_dim]
        params = state_with_params[:, self.state_dim:]

        # Perceive (sees both state and local params)
        features = self.perception(state_with_params)

        # Update state (fast dynamics, guided by params)
        state_delta = self.update(features)
        next_state = state + state_delta

        # Update params (slow dynamics, 100x slower)
        param_delta = self.param_update(features)
        next_params = params + 0.01 * param_delta  # Slow evolution

        # Recombine
        return torch.cat([next_state, next_params], dim=1)

    def spawn_offspring(self, parent_state: torch.Tensor) -> torch.Tensor:
        """
        Spawn new pseudopod: inherit parent's parameters + mutation.

        This is how update rules EVOLVE:
        - Child inherits parent's local update rule (params)
        - Small mutations allow exploration of rule space
        """
        # Extract parent parameters
        parent_params = parent_state[:, self.state_dim:]

        # Mutate parameters (Gaussian noise)
        mutation = torch.randn_like(parent_params) * 0.1
        child_params = parent_params + mutation

        # Initialize child state (random)
        child_state = torch.randn_like(parent_state[:, :self.state_dim])

        # Combine
        child = torch.cat([child_state, child_params], dim=1)
        return child
```

### 7.2 Multi-Species Interactions

```python
class MultiSpeciesPool:
    """
    Pool manages multiple pseudopods with different local rules.
    Rules can mix at boundaries (like Flow-Lenia).
    """

    def __init__(self, pool_size: int = 10):
        self.pseudopods: list[FlowLeniaPseudopod] = []
        self.states: list[torch.Tensor] = []

    def add_pseudopod(self, pod: FlowLeniaPseudopod, state: torch.Tensor):
        self.pseudopods.append(pod)
        self.states.append(state)

    def step(self):
        """
        Update all pseudopods in parallel.
        At boundaries, parameters MIX (Flow-Lenia multi-species).
        """
        for i, (pod, state) in enumerate(zip(self.pseudopods, self.states)):
            # Update this pseudopod
            next_state = pod.update_step(state)

            # Parameter mixing at boundaries with neighbors
            for j, neighbor_state in enumerate(self.states):
                if i != j:
                    next_state = self._mix_boundaries(next_state, neighbor_state)

            self.states[i] = next_state

    def _mix_boundaries(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor,
        mix_width: int = 2
    ) -> torch.Tensor:
        """
        Mix parameters at spatial boundaries (Flow-Lenia).
        Enables emergence of hybrid update rules.
        """
        # Detect spatial overlap (simplified)
        # In practice: compute spatial proximity, blend params

        # Extract parameters
        params1 = state1[:, 64:]  # Last param_dim channels
        params2 = state2[:, 64:]

        # Blend parameters at boundaries (weighted by distance)
        # For now: simple average where they overlap
        mixed_params = 0.5 * (params1 + params2)

        # Recombine with state
        state_only = state1[:, :64]
        return torch.cat([state_only, mixed_params], dim=1)
```

---

## Phase 8: Curiosity-Driven Lifecycle

**Goal**: Replace manual spawning schedule with intrinsic motivation

### 8.1 Learning Progress Tracker

```python
# slime/lifecycle/curiosity.py

class LearningProgressTracker:
    """
    Tracks learning progress for each pseudopod.
    High progress = productive exploration (keep alive).
    Low progress = stagnation (retire).
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: dict[int, list[dict]] = {}  # pod_id -> events

    def record_event(
        self,
        pod_id: int,
        event_type: str,  # 'novel' | 'improvement' | 'failure'
        fitness_delta: float = 0.0,
        novelty_score: float = 0.0
    ):
        """Record exploration event for this pseudopod"""
        if pod_id not in self.history:
            self.history[pod_id] = []

        self.history[pod_id].append({
            'type': event_type,
            'fitness_delta': fitness_delta,
            'novelty_score': novelty_score,
            'timestamp': time.time()
        })

        # Keep only recent window
        self.history[pod_id] = self.history[pod_id][-self.window_size:]

    def compute_learning_progress(self, pod_id: int) -> float:
        """
        Learning Progress = novelty + improvement over recent window.

        Based on: "Humans monitor learning progress in curiosity-driven
        exploration" (Nature Communications 2021)
        """
        if pod_id not in self.history:
            return 0.0

        events = self.history[pod_id]

        # Count novel discoveries
        novel_count = sum(1 for e in events if e['type'] == 'novel')

        # Sum fitness improvements
        improvement = sum(e['fitness_delta'] for e in events if e['fitness_delta'] > 0)

        # Combine (normalized)
        LP = (novel_count / len(events)) + (improvement / 10.0)

        return LP

    def get_stagnant_pods(self, threshold: float = 0.05) -> list[int]:
        """Return pseudopods with low learning progress (candidates for retirement)"""
        return [
            pod_id for pod_id in self.history.keys()
            if self.compute_learning_progress(pod_id) < threshold
        ]
```

### 8.2 Curiosity-Driven Lifecycle Manager

```python
class CuriosityDrivenLifecycle:
    """
    Manages pseudopod lifecycle via intrinsic motivation.

    Key principles:
    - Spawn new pods when unexplored regions detected
    - Retire stagnant pods (low learning progress)
    - Allocate compute to high-LP pods
    - Adaptive population size (not fixed)
    """

    def __init__(
        self,
        pool: MultiSpeciesPool,
        archive: AdaptiveVoronoiArchive,
        min_pods: int = 5,
        max_pods: int = 50
    ):
        self.pool = pool
        self.archive = archive
        self.min_pods = min_pods
        self.max_pods = max_pods
        self.lp_tracker = LearningProgressTracker()

    def step(self):
        """
        Execute one lifecycle step:
        1. Compute learning progress for all pods
        2. Spawn new pods if unexplored regions
        3. Retire stagnant pods
        4. Allocate compute proportional to LP
        """
        # 1. Compute LP for all active pods
        lp_scores = {
            pod_id: self.lp_tracker.compute_learning_progress(pod_id)
            for pod_id in range(len(self.pool.pseudopods))
        }

        # 2. Spawn new pods (curiosity-driven)
        if self.should_spawn():
            self.spawn_new_pod()

        # 3. Retire stagnant pods
        stagnant = self.lp_tracker.get_stagnant_pods(threshold=0.05)
        for pod_id in stagnant:
            if len(self.pool.pseudopods) > self.min_pods:
                self.retire_pod(pod_id)

        # 4. Allocate compute (more steps for high-LP pods)
        self.allocate_compute(lp_scores)

    def should_spawn(self) -> bool:
        """
        Spawn new pod if:
        - Archive coverage < threshold (unexplored regions)
        - OR all pods stagnating (need diversity)
        """
        coverage = self.archive.coverage()  # Fraction of cells filled

        if coverage < 0.7:
            return True  # Lots of unexplored space

        # Check if all pods stagnating
        avg_lp = np.mean([
            self.lp_tracker.compute_learning_progress(i)
            for i in range(len(self.pool.pseudopods))
        ])

        if avg_lp < 0.1:
            return True  # Need fresh perspective

        return False

    def spawn_new_pod(self):
        """
        Spawn new pseudopod with curiosity-driven initialization.

        Strategy:
        - Sample from archive's sparsest region (explore gaps)
        - Inherit parameters from high-LP parent (with mutation)
        """
        if len(self.pool.pseudopods) >= self.max_pods:
            return  # At capacity

        # Find sparsest region in archive
        sparse_centroid = self.archive.find_sparsest_region()

        # Find high-LP parent
        lp_scores = [
            self.lp_tracker.compute_learning_progress(i)
            for i in range(len(self.pool.pseudopods))
        ]
        parent_id = np.argmax(lp_scores)
        parent_pod = self.pool.pseudopods[parent_id]
        parent_state = self.pool.states[parent_id]

        # Spawn child (inherit + mutate)
        child_state = parent_pod.spawn_offspring(parent_state)
        child_pod = FlowLeniaPseudopod(
            state_dim=parent_pod.state_dim,
            param_dim=parent_pod.param_dim
        )

        # Initialize child parameters from parent (transfer learning)
        child_pod.load_state_dict(parent_pod.state_dict())

        # Add to pool
        self.pool.add_pseudopod(child_pod, child_state)

        logger.info(
            f'Spawned pod #{len(self.pool.pseudopods)} from parent #{parent_id} '
            f'(LP={lp_scores[parent_id]:.3f}) targeting sparse region'
        )

    def retire_pod(self, pod_id: int):
        """Remove stagnant pseudopod"""
        lp = self.lp_tracker.compute_learning_progress(pod_id)
        logger.info(f'Retiring pod #{pod_id} (LP={lp:.3f}, stagnant)')

        del self.pool.pseudopods[pod_id]
        del self.pool.states[pod_id]

    def allocate_compute(self, lp_scores: dict[int, float]):
        """
        Allocate compute time proportional to learning progress.
        High-LP pods get more update steps per generation.
        """
        total_lp = sum(lp_scores.values())

        if total_lp == 0:
            # Equal allocation if all zero
            steps_per_pod = {i: 10 for i in lp_scores.keys()}
        else:
            # Proportional allocation (min 5, max 50 steps)
            steps_per_pod = {
                i: int(5 + 45 * (lp / total_lp))
                for i, lp in lp_scores.items()
            }

        # Execute steps
        for i, steps in steps_per_pod.items():
            state = self.pool.states[i]
            pod = self.pool.pseudopods[i]
            self.pool.states[i] = pod(state, steps=steps)
```

---

## Implementation Roadmap

### Phase 4: DIRESA Encoder (2 weeks)
- [ ] Implement `DIRESABehavioralEncoder`
- [ ] Write comprehensive tests (distance preservation, sparsity)
- [ ] Integrate with `CVTArchive` (replace Kernel PCA)
- [ ] Train on existing archive data, compare to KPCA

### Phase 5: Adaptive Voronoi (2 weeks)
- [ ] Implement `AdaptiveVoronoiArchive`
- [ ] Cell split/merge logic with density tracking
- [ ] Tests for growth/shrinkage dynamics
- [ ] Visualizations of cell evolution over time

### Phase 6: Neural CA Pseudopods (3 weeks)
- [ ] Implement `NeuralCAPseudopod` with learned update
- [ ] Training infrastructure (`NCATrainer`)
- [ ] Replace transformer attention in `Pseudopod`
- [ ] Benchmarks: NCA vs Transformer on task performance

### Phase 7: Flow-Lenia Parameters (3 weeks)
- [ ] Implement `FlowLeniaPseudopod` with localized params
- [ ] Multi-species pool with parameter mixing
- [ ] Spawning with parameter inheritance + mutation
- [ ] Tests for parameter evolution convergence

### Phase 8: Curiosity Lifecycle (2 weeks)
- [ ] Implement `LearningProgressTracker`
- [ ] Implement `CuriosityDrivenLifecycle`
- [ ] Integration with `Organism`
- [ ] Compare: manual schedule vs curiosity-driven

### Phase 9: End-to-End Integration (2 weeks)
- [ ] Wire all components together
- [ ] Full system tests
- [ ] Ablation studies (which components matter most?)
- [ ] Production benchmarks vs current system

---

## Expected Outcomes

### Quantitative Improvements
- **Archive Coverage**: 70% → 95% (adaptive Voronoi)
- **Dimensionality**: Fixed 3-5D → Learned 4-8D (DIRESA)
- **Lifecycle Efficiency**: 30% wasted compute → 5% (curiosity-driven)
- **Rule Diversity**: 1 transformer → N evolved NCA rules

### Qualitative Improvements
- **Genuine Open-Endedness**: Rules evolve, not just weights
- **Emergent Species**: Multi-species co-evolution via Flow-Lenia
- **Adaptive Architecture**: Everything learned, not designed
- **Production-Ready**: First self-organizing CA transformer

---

## References

1. **Flow-Lenia**: [arXiv:2212.07906](https://arxiv.org/abs/2212.07906) (MIT Press Artificial Life 2025)
2. **Universal Neural CA**: [arXiv:2505.13058](https://arxiv.org/abs/2505.13058) (May 2025)
3. **DIRESA**: [arXiv:2404.18314](https://arxiv.org/abs/2404.18314) (April 2025)
4. **Curiosity & Learning Progress**: Nature Communications 2021, NeurIPS 2024 IMOL Workshop
5. **Adaptive Voronoi**: Multiple sources (biological tissue dynamics, QD optimization)

---

**Next Steps**: Begin Phase 4 (DIRESA encoder) with comprehensive test-driven development, maintaining 100% constraint satisfaction throughout.
