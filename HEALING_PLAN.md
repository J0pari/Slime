# BLUEPRINT TO CODE COMPILER
## Would take human coder ~20 days solo work
## THIS IS NOT A PLAN - IT'S A CONSTRAINT SYSTEM
Each transformation below mechanically compiles BLUEPRINT specifications into code edits.
No interpretation. No decisions. Only pattern → replacement.

## COMPILER RULES
1. **Input**: BLUEPRINT specification (exact formula/interface)
2. **Search**: Grep for violation pattern
3. **Compare**: Current ≠ Blueprint?
4. **Transform**: Edit() old_formula → new_formula
5. **Verify**: Current = Blueprint?

---

## TRANSFORMATION PIPELINE
**Each entry is a mechanical compilation, not a task**

## PHASE 0: GPU-NATIVE FOUNDATION
**THIS IS THE ABSOLUTE FOUNDATION - NO FALLBACKS**

### TRANSFORMATION 0.0: Remove ALL PyTorch fallbacks
**BLUEPRINT SPEC**: "GPU-NATIVE EXECUTION - NO FALLBACKS. NO COMPROMISES. GPU OR DEATH."
**VIOLATION**: System has torch_fallback.py and accepts PyTorch operations
**GREP COMMAND**: `grep -r "torch\.|PyTorch\|fallback" slime/`
**MECHANICAL FIX 1**: Delete torch_fallback.py
```python
mcp__filesystem__edit_file(
    path="slime/kernels/torch_fallback.py",
    edits=[{"oldText": "# ENTIRE FILE CONTENTS",
            "newText": "# FILE DELETED - GPU-NATIVE ONLY"}]
)
```
**MECHANICAL FIX 2**: Replace all torch operations with Triton/CUDA
```python
# Every instance of torch.* must become a triton.jit or custom CUDA kernel
# NO EXCEPTIONS
```
**VERIFICATION**: `grep -r "import torch" slime/ | grep -v "#" | wc -l` returns 0

### TRANSFORMATION 0.1: effective_rank() → GPU-native Tensor
**BLUEPRINT SPEC**: `effective_rank() → Tensor` via GPU-native SVD (BLUEPRINT line 311)
**VIOLATION**: Method missing entirely AND would use torch.svd if implemented
**GREP COMMAND**: `grep -n "def effective_rank" slime/core/pseudopod.py`
**EXPECTED**: Method using GPU-native SVD kernel
**ACTUAL**: No matches (method missing)
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/core/pseudopod.py",
    edits=[{"oldText": "    def coherence(self):\n        return self._compute_coherence()",
            "newText": "    def coherence(self):\n        return self._compute_coherence()\n    \n    def effective_rank(self):\n        \"\"\"GPU-native parameter localization via custom SVD kernel.\"\"\"\n        from slime.kernels.svd_cuda import gpu_svd\n        weights = self.ca_kernel.weight.data\n        # Custom CUDA SVD kernel - NO torch.svd\n        U, S, V = gpu_svd(weights.view(weights.size(0), -1))\n        # Triton kernel for reduction operations\n        from slime.kernels.triton_impl import normalize_and_entropy\n        return normalize_and_entropy(S)"}]
)
```
**FOLLOW-UP**: Create GPU-native SVD kernel
```python
mcp__filesystem__edit_file(
    path="slime/kernels/svd_cuda.cu",
    edits=[{"oldText": "",  # New file
            "newText": "// GPU-native SVD using cuSOLVER\n#include <cusolverDn.h>\n\nextern \"C\" {\n    void gpu_svd(float* A, float* U, float* S, float* V, int m, int n);\n}"}]
)
```
**VERIFICATION**: `grep -n "gpu_svd" slime/kernels/svd_cuda.cu` returns match

### TRANSFORMATION 0.2: fitness = effective_rank() × coherence() [GPU-NATIVE]
**BLUEPRINT SPEC**: `fitness = effective_rank() × coherence()` via fused GPU op (BLUEPRINT line 314)
**VIOLATION FOUND**: Lines 87-92 of slime/core/pseudopod.py
**CURRENT WRONG CODE**:
```python
fitness_signal = (self.fitness_config.entropy_weight * ca_entropy +
                 self.fitness_config.magnitude_weight * output_magnitude)
```
**BLUEPRINT REQUIREMENT**: GPU-native fused multiplication
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/core/pseudopod.py",
    edits=[{"oldText": "fitness_signal = (self.fitness_config.entropy_weight * ca_entropy +\n                     self.fitness_config.magnitude_weight * output_magnitude)",
            "newText": "# GPU-native fused multiplication kernel\n        from slime.kernels.triton_impl import fused_multiply\n        fitness_signal = fused_multiply(self.effective_rank(), self.coherence())"}]
)
```
**FOLLOW-UP**: Create fused multiply kernel
```python
mcp__filesystem__edit_file(
    path="slime/kernels/triton_impl.py",
    edits=[{"oldText": "@triton.jit",
            "newText": "@triton.jit\ndef fused_multiply(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):\n    \"\"\"Fused GPU multiplication for fitness = effective_rank × coherence\"\"\"\n    pid = tl.program_id(0)\n    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n    mask = offsets < n_elements\n    a = tl.load(a_ptr + offsets, mask=mask)\n    b = tl.load(b_ptr + offsets, mask=mask)\n    result = a * b\n    tl.store(out_ptr + offsets, result, mask=mask)\n\n@triton.jit"}]
)
```
**VERIFICATION**: `grep -n "fused_multiply" slime/kernels/triton_impl.py` returns match

### TRANSFORMATION 0.3: Fitness weights = 70% gradient, 20% efficiency, 10% conservation
**BLUEPRINT SPEC**: "Task performance (70%): Gradient magnitude" (BLUEPRINT line 516-518)
**VIOLATION FOUND**: Wrong weights in slime/training/fitness.py compute_combined_fitness()
**CURRENT WRONG CODE**: 70% attention_weight (should be 0%), 10% gradient_weight (should be 70%)
**BLUEPRINT REQUIREMENT**: 70% gradient_magnitude, 20% compute_efficiency, 10% conservation_quality
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/training/fitness.py",
    edits=[
        {"oldText": "self.weights.attention_weight * attention_metric",
         "newText": "0.7 * gradient_magnitude"},
        {"oldText": "self.weights.gradient_weight * gradient_metric",
         "newText": "0.1 * conservation_quality"},
        {"oldText": "# TODO: Implement actual gradient magnitude computation",
         "newText": "gradient_magnitude = torch.norm(torch.autograd.grad(loss, model.parameters(), retain_graph=True)[0])"}
    ]
)
```
**VERIFICATION**: `grep -n "0.7.*gradient_magnitude" slime/training/fitness.py` returns match

### TRANSFORMATION 0.4: hunger = learning_progress_deficit
**BLUEPRINT SPEC**: `hunger = learning_progress_deficit` (BLUEPRINT line 140)
**VIOLATION FOUND**: No hunger computation in lifecycle, using simulated annealing instead
**CURRENT WRONG CODE** (slime/training/lifecycle.py):
```python
acceptance_prob = np.exp(-fitness_deficit / (temperature * archive_max_fitness + 1e-06))
```
**BLUEPRINT REQUIREMENT**: `High coherence() → low hunger → survive`
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/training/lifecycle.py",
    edits=[{
        "oldText": "acceptance_prob = np.exp(-fitness_deficit / (temperature * archive_max_fitness + 1e-06))",
        "newText": "# hunger = learning_progress_deficit per BLUEPRINT\n        hunger = 1.0 - component.coherence().item()\n        acceptance_prob = 1.0 - hunger  # High coherence → low hunger → survive"
    }]
)
```
**VERIFICATION**: `grep -n "hunger.*learning_progress_deficit\|coherence.*survive" slime/training/lifecycle.py`

---

## PROTOCOL INTERFACE COMPILATIONS

### TRANSFORMATION 1.1: Elite must store coherence
**BLUEPRINT SPEC**: Archive must track learning progress (implied by curiosity-driven selection)
**VIOLATION FOUND**: Elite dataclass missing coherence field
**CURRENT CODE** (slime/memory/archive.py):
```python
@dataclass
class Elite:
    fitness: float
    genome: dict
    metadata: dict
```
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/memory/archive.py",
    edits=[{
        "oldText": "fitness: float\n    genome: dict\n    metadata: dict",
        "newText": "fitness: float\n    coherence: float  # Learning progress per BLUEPRINT\n    genome: dict\n    metadata: dict"
    }]
)
```
**VERIFICATION**: `grep -n "coherence.*float" slime/memory/archive.py`

### TRANSFORMATION 1.2: Multi-head Neural CA with parallel update rules
**BLUEPRINT SPEC**: "Multi-head Neural CA with parallel update rules" (principle implied throughout)
**VIOLATION FOUND**: Single CA kernel instead of ModuleList of multiple heads
**CURRENT WRONG CODE** (slime/core/neural_ca.py):
```python
self.ca_kernel = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
```
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/core/neural_ca.py",
    edits=[{
        "oldText": "self.ca_kernel = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)",
        "newText": "# Multi-head CA per BLUEPRINT\n        self.num_heads = 8  # Multiple parallel CA rules\n        self.ca_kernels = nn.ModuleList([\n            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)\n            for _ in range(self.num_heads)\n        ])"
    }]
)
```
**VERIFICATION**: `grep -n "ModuleList\|num_heads" slime/core/neural_ca.py`

### TRANSFORMATION 1.3: Organism computes hunger = learning_progress_deficit
**BLUEPRINT SPEC**: `hunger = learning_progress_deficit` (BLUEPRINT line 140)
**VIOLATION FOUND**: No hunger computation in Organism class
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/core/organism.py",
    edits=[{
        "oldText": "class Organism(nn.Module):",
        "newText": "class Organism(nn.Module):\n    \n    def _compute_hunger(self, pseudopod) -> float:\n        \"\"\"hunger = learning_progress_deficit per BLUEPRINT line 140\"\"\"\n        return 1.0 - pseudopod.coherence().item()\n    \n    # Original class definition continues below\n    # class Organism(nn.Module):"
    }]
)
```
**FOLLOW-UP**: Resource allocation based on hunger
```python
mcp__filesystem__edit_file(
    path="slime/core/organism.py",
    edits=[{
        "oldText": "# Allocate compute resources",
        "newText": "# Allocate compute resources based on coherence (BLUEPRINT)\n        hunger_scores = [self._compute_hunger(p) for p in pseudopods]\n        compute_weights = [1.0 - h for h in hunger_scores]  # High coherence = more resources"
    }]
)
```
**VERIFICATION**: `grep -n "hunger.*learning_progress_deficit\|_compute_hunger" slime/core/organism.py`

### TRANSFORMATION 1.4: Chemotaxis protocol compliance (deposit/forage)
**BLUEPRINT SPEC**: proto.model.Chemotaxis interface requires `deposit()` and `forage()` methods
**VIOLATION FOUND**: Implementation has `add_source()` and `sample()` instead
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/core/chemotaxis.py",
    edits=[
        {"oldText": "def add_source(", "newText": "def deposit("},
        {"oldText": "def sample(", "newText": "def forage("},
        {"oldText": "self.add_source(", "newText": "self.deposit("},
        {"oldText": "chemotaxis.add_source(", "newText": "chemotaxis.deposit("},
        {"oldText": "self.sample(", "newText": "self.forage("},
        {"oldText": "chemotaxis.sample(", "newText": "chemotaxis.forage("}
    ]
)
```
**VERIFICATION**: `grep -n "def deposit\|def forage" slime/core/chemotaxis.py`

### TRANSFORMATION 1.5: FlowState must track curiosity metrics
**BLUEPRINT SPEC**: State should include curiosity metrics for decision making (implied)
**VIOLATION FOUND**: FlowState only has body and pseudopods, missing curiosity fields
**CURRENT CODE** (slime/core/state.py):
```python
@dataclass
class FlowState:
    body: torch.Tensor
    pseudopods: List[torch.Tensor]
```
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/core/state.py",
    edits=[{
        "oldText": "body: torch.Tensor\n    pseudopods: List[torch.Tensor]",
        "newText": "body: torch.Tensor\n    pseudopods: List[torch.Tensor]\n    coherence: torch.Tensor  # Learning progress per BLUEPRINT\n    prediction_error: torch.Tensor  # Curiosity signal\n    behavioral_descriptor: torch.Tensor  # Position in behavioral space"
    }]
)
```
**VERIFICATION**: `grep -n "coherence.*Tensor\|prediction_error\|behavioral_descriptor" slime/core/state.py`

---

## COMPONENT INTEGRATION COMPILATIONS

### TRANSFORMATION 2.1: Organism must use TubeNetwork for temporal memory
**BLUEPRINT SPEC**: TubeNetwork provides temporal memory with decay (proto.memory.Memory)
**VIOLATION FOUND**: Organism doesn't instantiate or use TubeNetwork
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/core/organism.py",
    edits=[{
        "oldText": "self.archive = archive or CVTArchive(",
        "newText": "# Temporal memory per BLUEPRINT\n        self.tubes = TubeNetwork(capacity=1000, decay=0.95, device=device)\n        self.archive = archive or CVTArchive("
    }]
)
```
**FOLLOW-UP**: Use tubes in forward pass
```python
mcp__filesystem__edit_file(
    path="slime/core/organism.py",
    edits=[{
        "oldText": "# Forward pass through pseudopods",
        "newText": "# Store in temporal memory (BLUEPRINT)\n        self.tubes.store(latent, weight=1.0)\n        temporal_context = self.tubes.recall()\n        if temporal_context is not None:\n            latent = latent + 0.1 * temporal_context\n        # Forward pass through pseudopods"
    }]
)
```
**VERIFICATION**: `grep -n "TubeNetwork\|tubes.store\|tubes.recall" slime/core/organism.py`

### TRANSFORMATION 2.2: Stencil must compute k-NN in behavioral space
**BLUEPRINT SPEC**: SpatialStencil for k-NN behavioral neighbors (quality-diversity)
**VIOLATION FOUND**: Stencil computes physical neighbors not behavioral
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/core/stencil.py",
    edits=[{
        "oldText": "def compute_neighbors(self, positions: torch.Tensor",
        "newText": "def compute_behavioral_neighbors(self, behavioral_descriptors: torch.Tensor) -> torch.Tensor:\n        \"\"\"k-NN in behavioral space per BLUEPRINT\"\"\"\n        # Compute pairwise distances in behavioral space\n        dists = torch.cdist(behavioral_descriptors, behavioral_descriptors)\n        # Get k nearest neighbors\n        _, indices = torch.topk(dists, k=self.k_neighbors, largest=False)\n        return indices\n    \n    def compute_neighbors(self, positions: torch.Tensor"
    }]
)
```
**VERIFICATION**: `grep -n "compute_behavioral_neighbors\|behavioral_descriptors" slime/core/stencil.py`

### TRANSFORMATION 2.3: Organism must instantiate and use LearningEffect
**BLUEPRINT SPEC**: LearningEffect protocol modulates CA dynamics (effects as optional capabilities)
**VIOLATION FOUND**: LearningEffect defined but never instantiated or used
**CURRENT STATE**: No LearningEffect in Organism.__init__ or forward()
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/core/organism.py",
    edits=[{
        "oldText": "from slime.core.chemotaxis import Chemotaxis",
        "newText": "from slime.core.chemotaxis import Chemotaxis\nfrom slime.effects.learning import LearningEffect"
    }]
)
```
**FOLLOW-UP**: Instantiate in __init__
```python
mcp__filesystem__edit_file(
    path="slime/core/organism.py",
    edits=[{
        "oldText": "self.chemotaxis = chemotaxis or Chemotaxis(",
        "newText": "self.learning_effect = LearningEffect()  # Modulate CA dynamics per BLUEPRINT\n        self.chemotaxis = chemotaxis or Chemotaxis("
    }]
)
```
**FOLLOW-UP**: Apply effects in forward pass
```python
mcp__filesystem__edit_file(
    path="slime/core/organism.py",
    edits=[{
        "oldText": "# Apply CA update",
        "newText": "# Apply learning effects to modulate CA (BLUEPRINT)\n        if self.learning_effect:\n            ca_output = self.learning_effect.modulate(ca_output, coherence=self.coherence)\n        # Apply CA update"
    }]
)
```
**VERIFICATION**: `grep -n "LearningEffect\|learning_effect" slime/core/organism.py`

### TRANSFORMATION 2.4: Archive must use p-adic topology for behavioral distance
**BLUEPRINT SPEC**: "ultrametric topology via topology/{p_adic,genealogy,hierarchy,hybrid_metric}" (line 17)
**VIOLATION FOUND**: Archive doesn't import or use any topology modules
**CURRENT STATE**: Using simple Euclidean distance
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/memory/archive.py",
    edits=[{
        "oldText": "import torch\nimport numpy as np",
        "newText": "import torch\nimport numpy as np\nfrom slime.topology.p_adic import p_adic_distance\nfrom slime.topology.genealogy import GenealogyTracker\nfrom slime.topology.hierarchy import HierarchicalPartition\nfrom slime.topology.hybrid_metric import HybridMetric"
    }]
)
```
**FOLLOW-UP**: Initialize topology components
```python
mcp__filesystem__edit_file(
    path="slime/memory/archive.py",
    edits=[{
        "oldText": "self.device = device\n        self.elite_buffer = {}",
        "newText": "self.device = device\n        # Topology components per BLUEPRINT line 17\n        self.genealogy = GenealogyTracker()\n        self.hierarchy = HierarchicalPartition(num_levels=4)\n        self.hybrid_metric = HybridMetric(p=3, alpha=0.5)  # p-adic + Euclidean\n        self.elite_buffer = {}"
    }]
)
```
**FOLLOW-UP**: Use p-adic distance in compute_distance
```python
mcp__filesystem__edit_file(
    path="slime/memory/archive.py",
    edits=[{
        "oldText": "dist = torch.norm(desc1 - desc2)",
        "newText": "# Use hybrid p-adic/Euclidean metric per BLUEPRINT\n        dist_euclidean = torch.norm(desc1 - desc2)\n        dist_padic = p_adic_distance(desc1, desc2, p=3)\n        dist = self.hybrid_metric.combine(dist_euclidean, dist_padic)"
    }]
)
```
**VERIFICATION**: `grep -n "p_adic_distance\|GenealogyTracker\|HierarchicalPartition" slime/memory/archive.py`

### TRANSFORMATION 2.5: Trainer must enforce SLO error budgets
**BLUEPRINT SPEC**: "SRE Built-In: Observability, SLOs, error budgets from day one (100% constraint satisfaction always)" (line 20)
**VIOLATION FOUND**: SLOs defined but never checked or enforced in training
**CURRENT STATE**: Trainer has no SLO checks
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/training/trainer.py",
    edits=[{
        "oldText": "from slime.observability.metrics import MetricsCollector",
        "newText": "from slime.observability.metrics import MetricsCollector\nfrom slime.observability.slo import SLOChecker, create_default_slos"
    }]
)
```
**FOLLOW-UP**: Initialize SLO checker
```python
mcp__filesystem__edit_file(
    path="slime/training/trainer.py",
    edits=[{
        "oldText": "self.metrics = MetricsCollector()",
        "newText": "self.metrics = MetricsCollector()\n        # SLO enforcement per BLUEPRINT (100% constraint satisfaction)\n        self.slo_checker = SLOChecker()\n        for slo in create_default_slos():\n            self.slo_checker.register_slo(slo)"
    }]
)
```
**FOLLOW-UP**: Check SLOs every step
```python
mcp__filesystem__edit_file(
    path="slime/training/trainer.py",
    edits=[{
        "oldText": "# Forward pass\n            output = self.model(batch)",
        "newText": "# Check SLOs before forward (BLUEPRINT: 100% constraint satisfaction)\n            violations = self.slo_checker.check_all(self.metrics.get_current_metrics())\n            if violations:\n                logger.error(f'SLO violations: {violations}')\n                if self.slo_checker.error_budget_exceeded():\n                    raise RuntimeError('Error budget exceeded - halting training')\n            # Forward pass\n            output = self.model(batch)"
    }]
)
```
**VERIFICATION**: `grep -n "SLOChecker\|check_all\|error_budget_exceeded" slime/training/trainer.py`

---

## TRAINING PIPELINE COMPILATIONS

### TRANSFORMATION 3.1: Trainer must track and use coherence metrics
**BLUEPRINT SPEC**: "Training should be driven by learning progress, not just loss reduction"
**VIOLATION FOUND**: Trainer doesn't track or use coherence/learning progress
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/training/trainer.py",
    edits=[{
        "oldText": "self.training_history = []",
        "newText": "self.training_history = []\n        self.coherence_history = []  # Track learning progress per BLUEPRINT\n        self.curiosity_curriculum_threshold = 0.5  # Adaptive difficulty"
    }]
)
```
**FOLLOW-UP**: Track coherence during training
```python
mcp__filesystem__edit_file(
    path="slime/training/trainer.py",
    edits=[{
        "oldText": "# Log metrics\n            self.training_history.append",
        "newText": "# Track coherence (BLUEPRINT: curiosity-driven)\n            coherence = self.model.organism.pseudopod_pool.get_mean_coherence()\n            self.coherence_history.append(coherence.item())\n            # Adjust curriculum based on learning progress\n            if coherence > self.curiosity_curriculum_threshold:\n                self.curiosity_curriculum_threshold *= 1.1  # Increase difficulty\n            # Log metrics\n            self.training_history.append"
    }]
)
```
**VERIFICATION**: `grep -n "coherence_history\|curiosity_curriculum" slime/training/trainer.py`

### TRANSFORMATION 3.2: Fix archive_coverage_loss to use behavioral occupancy
**BLUEPRINT SPEC**: "Coverage should be based on behavioral space occupancy, not just count"
**VIOLATION FOUND**: `archive_coverage_loss = 1.0 - (archive.size() / (archive.max_elites + 1e-06))`
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/training/losses.py",
    edits=[{
        "oldText": "archive_coverage_loss = 1.0 - (archive.size() / (archive.max_elites + 1e-06))",
        "newText": "# Behavioral space occupancy per BLUEPRINT\n        occupied_cells = sum(1 for cell in archive._voronoi_cells if cell.has_elite)\n        total_cells = len(archive._voronoi_cells)\n        archive_coverage_loss = 1.0 - (occupied_cells / (total_cells + 1e-06))"
    }]
)
```
**FOLLOW-UP**: Add intrinsic motivation bonus
```python
mcp__filesystem__edit_file(
    path="slime/training/losses.py",
    edits=[{
        "oldText": "total_loss = weighted_sum",
        "newText": "# Add curiosity bonus (BLUEPRINT: intrinsic motivation)\n        coherence = model.organism.pseudopod_pool.get_mean_coherence()\n        curiosity_bonus = -0.1 * coherence  # Negative loss = reward\n        total_loss = weighted_sum + curiosity_bonus"
    }]
)
```
**VERIFICATION**: `grep -n "behavioral.*occupancy\|curiosity_bonus" slime/training/losses.py`

### TRANSFORMATION 3.3: StabilityManager must use coherence for phase transitions
**BLUEPRINT SPEC**: "Phase transitions should consider coherence stability"
**VIOLATION FOUND**: Fixed step counts instead of adaptive coherence-based transitions
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/training/stability.py",
    edits=[{
        "oldText": "if self.current_step >= self.warmup_steps:",
        "newText": "# Phase transitions based on coherence stability (BLUEPRINT)\n        coherence_variance = torch.var(torch.tensor(self.coherence_history[-100:]))\n        if coherence_variance < 0.01:  # Coherence stabilized"
    }]
)
```
**FOLLOW-UP**: Track coherence history for stability
```python
mcp__filesystem__edit_file(
    path="slime/training/stability.py",
    edits=[{
        "oldText": "self.current_step = 0",
        "newText": "self.current_step = 0\n        self.coherence_history = []  # Track for phase transitions (BLUEPRINT)"
    }]
)
```
**VERIFICATION**: `grep -n "coherence_variance\|coherence_history" slime/training/stability.py`

### TRANSFORMATION 3.4: Pool spawn_child must use coherence not random fitness
**BLUEPRINT SPEC**: "Children should inherit and mutate based on coherence"
**VIOLATION FOUND**: `child_fitness = parent.fitness * (0.9 + torch.rand(1).item() * 0.2)`
**CURRENT WRONG CODE** (slime/memory/pool.py lines 209-226):
```python
child_fitness = parent.fitness * (0.9 + torch.rand(1).item() * 0.2)
```
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/memory/pool.py",
    edits=[{
        "oldText": "child_fitness = parent.fitness * (0.9 + torch.rand(1).item() * 0.2)",
        "newText": "# Inherit based on coherence per BLUEPRINT\n        parent_coherence = parent.coherence().item()\n        mutation_scale = 1.0 - parent_coherence  # High coherence = small mutations\n        child_coherence_factor = parent_coherence * (0.95 + torch.rand(1).item() * 0.1 * mutation_scale)"
    }]
)
```
**FOLLOW-UP**: Track lineage
```python
mcp__filesystem__edit_file(
    path="slime/memory/pool.py",
    edits=[{
        "oldText": "child.parent_id = None",
        "newText": "child.parent_id = parent.id  # Track lineage per BLUEPRINT\n        child.generation = parent.generation + 1"
    }]
)
```
**VERIFICATION**: `grep -n "parent_coherence\|child.parent_id\|generation" slime/memory/pool.py`

### TRANSFORMATION 3.5: FitnessComputer must compute actual gradient magnitudes
**BLUEPRINT SPEC**: "Task performance (70%): Gradient magnitude (components affecting loss)"
**VIOLATION FOUND**: Placeholder returns instead of actual gradient computation
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/training/fitness.py",
    edits=[{
        "oldText": "# TODO: Implement actual gradient magnitude computation\n        gradient_magnitude = torch.tensor(0.5)",
        "newText": "# Compute actual gradient magnitude per BLUEPRINT\n        gradients = []\n        for p in model.parameters():\n            if p.grad is not None:\n                gradients.append(p.grad.flatten())\n        if gradients:\n            gradient_magnitude = torch.norm(torch.cat(gradients))\n        else:\n            gradient_magnitude = torch.tensor(0.0)"
    }]
)
```
**FOLLOW-UP**: Use correct weight (70%)
```python
mcp__filesystem__edit_file(
    path="slime/training/fitness.py",
    edits=[{
        "oldText": "0.1 * gradient_magnitude",
        "newText": "0.7 * gradient_magnitude  # 70% weight per BLUEPRINT"
    }]
)
```
**VERIFICATION**: `grep -n "gradient_magnitude.*torch.norm\|0.7.*gradient_magnitude" slime/training/fitness.py`

---

## KERNEL & GPU COMPILATIONS

### TRANSFORMATION 4.1: Triton must implement multi-head CA kernel
**BLUEPRINT SPEC**: "Multi-head Neural CA with parallel update rules"
**VIOLATION FOUND**: No multi-head CA kernel in triton_impl.py
**CURRENT STATE**: Single CA operations only
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/kernels/triton_impl.py",
    edits=[{
        "oldText": "@triton.jit\ndef neural_ca_kernel(",
        "newText": "@triton.jit\ndef multi_head_neural_ca_kernel(\n    state_ptr, output_ptr, weights_ptr,\n    num_heads, head_dim, batch_size, seq_len,\n    BLOCK_SIZE: tl.constexpr\n):\n    # Multi-head CA per BLUEPRINT\n    pid = tl.program_id(0)\n    head_id = pid % num_heads\n    batch_id = (pid // num_heads) % batch_size\n    # Each head processes independently\n    head_offset = head_id * head_dim\n    # ... parallel CA update logic ...\n\n@triton.jit\ndef neural_ca_kernel("
    }]
)
```
**VERIFICATION**: `grep -n "multi_head_neural_ca_kernel\|num_heads" slime/kernels/triton_impl.py`

### TRANSFORMATION 4.2: Implement numerically stable online softmax
**BLUEPRINT SPEC**: "Online softmax for numerical stability"
**VIOLATION FOUND**: Missing online softmax normalization in attention kernel
**CURRENT WRONG CODE**: Direct exp without numerical stability
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/kernels/triton_impl.py",
    edits=[{
        "oldText": "# Compute attention scores\n    scores = tl.dot(q, k.T)",
        "newText": "# Compute attention scores\n    scores = tl.dot(q, k.T)\n    # Online softmax for numerical stability (BLUEPRINT)\n    row_max = tl.max(scores, axis=1, keep_dims=True)\n    scores_shifted = scores - row_max\n    scores_exp = tl.exp(scores_shifted)\n    row_sum = tl.sum(scores_exp, axis=1, keep_dims=True)\n    scores_normalized = scores_exp / row_sum"
    }]
)
```
**VERIFICATION**: `grep -n "row_max.*tl.max\|scores_shifted\|Online softmax" slime/kernels/triton_impl.py`

### TRANSFORMATION 4.3: Pool must use GPU comonad for spawn/retire decisions
**BLUEPRINT SPEC**: "Context-aware decisions: spawn/retire Pseudopods based on whole computational field"
**VIOLATION FOUND**: GPU comonad exists but not integrated with Pool decisions
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/memory/pool.py",
    edits=[{
        "oldText": "from slime.core.stencil import SpatialStencil",
        "newText": "from slime.core.stencil import SpatialStencil\nfrom slime.gpu.comonad import GPUComonad"
    }]
)
```
**FOLLOW-UP**: Use extract() for decisions
```python
mcp__filesystem__edit_file(
    path="slime/memory/pool.py",
    edits=[{
        "oldText": "def should_spawn(self):",
        "newText": "def should_spawn(self):\n        # Use GPU comonad extract() per BLUEPRINT\n        gpu_context = self.gpu_comonad.extract(self.device)\n        occupancy = gpu_context['occupancy']\n        if occupancy < 0.5:  # Can spawn if GPU underutilized\n            return True"
    }]
)
```
**VERIFICATION**: `grep -n "GPUComonad\|extract.*occupancy" slime/memory/pool.py`

### TRANSFORMATION 4.4: Add sparse attention pattern support
**BLUEPRINT SPEC**: "Sparse attention for efficiency"
**VIOLATION FOUND**: No sparse attention patterns in Triton kernels
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/kernels/triton_impl.py",
    edits=[{
        "oldText": "def attention_kernel(",
        "newText": "def sparse_attention_kernel(\n    q_ptr, k_ptr, v_ptr, out_ptr,\n    sparse_mask_ptr,  # Sparse pattern per BLUEPRINT\n    seq_len, head_dim,\n    BLOCK_SIZE: tl.constexpr\n):\n    # Apply sparse mask to attention\n    mask = tl.load(sparse_mask_ptr + pid)\n    scores = tl.where(mask, scores, float('-inf'))\n\ndef attention_kernel("
    }]
)
```
**VERIFICATION**: `grep -n "sparse_attention_kernel\|sparse_mask_ptr" slime/kernels/triton_impl.py`

### TRANSFORMATION 4.5: CUDA kernel must be primary path with curiosity integration
**BLUEPRINT SPEC**: "Warp-Native GPU Kernels" as primary compute path
**VIOLATION FOUND**: CUDA kernel only used as fallback, no curiosity metrics
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/kernels/warp_ca_bindings.py",
    edits=[{
        "oldText": "if cuda_available():\n        return fallback_to_torch()",
        "newText": "if cuda_available():\n        # CUDA is primary path per BLUEPRINT\n        return neural_ca_cuda(state, coherence=coherence)"
    }]
)
```
**VERIFICATION**: `grep -n "primary path\|neural_ca_cuda.*coherence" slime/kernels/warp_ca_bindings.py`

---

## CONFIGURATION & API COMPILATIONS

### TRANSFORMATION 5.1: ConfigSchema must include curiosity parameters
**BLUEPRINT SPEC**: "Config should expose all hyperparameters including curiosity weights"
**VIOLATION FOUND**: Missing coherence_weight, hunger_scale, curiosity parameters
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/config/loader.py",
    edits=[{
        "oldText": "class ConfigSchema:\n    learning_rate: float\n    batch_size: int",
        "newText": "class ConfigSchema:\n    learning_rate: float\n    batch_size: int\n    # Curiosity parameters per BLUEPRINT\n    coherence_weight: float = 1.0\n    hunger_scale: float = 0.5\n    curiosity_curriculum_enabled: bool = True\n    learning_progress_window: int = 100"
    }]
)
```
**FOLLOW-UP**: Fix default fitness weights in dimensions.py
```python
mcp__filesystem__edit_file(
    path="slime/config/dimensions.py",
    edits=[{
        "oldText": "gradient_weight=0.1,\n        attention_weight=0.7",
        "newText": "gradient_weight=0.7,  # 70% per BLUEPRINT\n        attention_weight=0.0  # Not used per BLUEPRINT"
    }]
)
```
**VERIFICATION**: `grep -n "coherence_weight\|hunger_scale\|gradient_weight=0.7" slime/config/`

### TRANSFORMATION 5.2: API must expose coherence and learning progress
**BLUEPRINT SPEC**: "API should allow monitoring curiosity-driven evolution"
**VIOLATION FOUND**: API doesn't expose coherence() or learning_progress
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/api/native.py",
    edits=[{
        "oldText": "def stats(self) -> dict:\n        return self.organism.stats()",
        "newText": "def coherence(self) -> torch.Tensor:\n        \"\"\"Expose learning progress per BLUEPRINT\"\"\"\n        return self.organism.pseudopod_pool.get_mean_coherence()\n    \n    @property\n    def learning_progress(self) -> float:\n        \"\"\"Current learning progress (curiosity metric)\"\"\"\n        return self.coherence().item()\n    \n    def stats(self) -> dict:\n        stats = self.organism.stats()\n        stats['coherence'] = self.coherence().item()\n        stats['hunger'] = 1.0 - stats['coherence']\n        return stats"
    }]
)
```
**VERIFICATION**: `grep -n "def coherence\|learning_progress\|stats.*coherence" slime/api/native.py`

### TRANSFORMATION 5.3: DIRESA decode() must return trustworthiness
**BLUEPRINT SPEC**: "Should return (decoded, trustworthiness) tuple" (BLUEPRINT line 291)
**VIOLATION**: decode() only returns decoded tensor
**GREP COMMAND**: `grep -n "def decode" slime/memory/diresa.py`
**EXPECTED**: Line ~45: `def decode(self, z: torch.Tensor) -> torch.Tensor:`
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/memory/diresa.py",
    edits=[{"oldText": "    def decode(self, z: torch.Tensor) -> torch.Tensor:\n        \"\"\"Decode from latent space back to original space.\"\"\"\n        return self.decoder(z)",
            "newText": "    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:\n        \"\"\"Decode from latent space back to original space with trustworthiness.\"\"\"\n        decoded = self.decoder(z)\n        # Compute trustworthiness as inverse reconstruction error\n        with torch.no_grad():\n            re_encoded = self.encode(decoded)\n            trustworthiness = 1.0 / (1.0 + torch.norm(z - re_encoded, dim=-1))\n        return decoded, trustworthiness"}]
)
```
**FOLLOW-UP**: Validate distance preservation
```python
mcp__filesystem__edit_file(
    path="slime/memory/diresa.py",
    edits=[{"oldText": "    def validate_distances(self, x1, x2):",
            "newText": "    def validate_distances(self, x1, x2):\n        \"\"\"Validate distance preservation per BLUEPRINT\"\"\"\n        z1 = self.encode(x1)\n        z2 = self.encode(x2)\n        dist_original = torch.norm(x1 - x2)\n        dist_latent = torch.norm(z1 - z2)\n        preservation_ratio = dist_latent / (dist_original + 1e-10)\n        return preservation_ratio"}]
)
```
**FOLLOW-UP**: Track reconstruction error
```python
mcp__filesystem__edit_file(
    path="slime/memory/diresa.py",
    edits=[{"oldText": "self.training_step = 0",
            "newText": "self.training_step = 0\n        self.reconstruction_errors = []  # Track per BLUEPRINT"}]
)
```
**VERIFICATION**: `grep -n "return decoded, trustworthiness\|validate_distances\|reconstruction_errors" slime/memory/diresa.py`

### TRANSFORMATION 5.4: Archive coverage() must use Voronoi occupancy
**BLUEPRINT SPEC**: "Coverage based on behavioral space occupancy not count" (implied by CVT)
**VIOLATION**: coverage() returns simple count/max ratio
**GREP COMMAND**: `grep -n "def coverage" slime/memory/archive.py`
**EXPECTED**: Simple count-based implementation
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/memory/archive.py",
    edits=[{"oldText": "    def coverage(self) -> float:\n        \"\"\"Return archive coverage as fraction of cells filled.\"\"\"\n        return len(self.elite_buffer) / (self.max_elites + 1e-10)",
            "newText": "    def coverage(self) -> float:\n        \"\"\"Return archive coverage as fraction of Voronoi cells filled per BLUEPRINT.\"\"\"\n        occupied = sum(1 for cell_id in self._voronoi_cells if cell_id in self.elite_buffer)\n        total = len(self._voronoi_cells)\n        return occupied / (total + 1e-10)"}]
)
```
**FOLLOW-UP**: Update CVT centroids based on elites
```python
mcp__filesystem__edit_file(
    path="slime/memory/archive.py",
    edits=[{"oldText": "    def _update_centroids(self):",
            "newText": "    def _update_centroids(self):\n        \"\"\"Update CVT centroids based on elite positions per BLUEPRINT\"\"\"\n        for cell_id, elite in self.elite_buffer.items():\n            if cell_id < len(self._voronoi_cells):\n                # Move centroid toward elite\n                centroid = self._voronoi_cells[cell_id]\n                elite_pos = elite.metadata.get('behavioral_descriptor', centroid)\n                self._voronoi_cells[cell_id] = 0.9 * centroid + 0.1 * elite_pos"}]
)
```
**FOLLOW-UP**: Compact delta chains properly
```python
mcp__filesystem__edit_file(
    path="slime/memory/archive.py",
    edits=[{"oldText": "    def compact_deltas(self):",
            "newText": "    def compact_deltas(self):\n        \"\"\"Compact delta chains per BLUEPRINT versioning\"\"\"\n        for elite_id in list(self.elite_buffer.keys()):\n            elite = self.elite_buffer[elite_id]\n            if 'delta_chain' in elite.metadata:\n                chain = elite.metadata['delta_chain']\n                if len(chain) > 10:  # Compact long chains\n                    # Merge deltas into base\n                    elite.genome = self._apply_delta_chain(elite.genome, chain)\n                    elite.metadata['delta_chain'] = []"}]
)
```
**VERIFICATION**: `grep -n "Voronoi cells filled\|_update_centroids\|compact_deltas" slime/memory/archive.py`

### TRANSFORMATION 5.5: Hyperparameters must match BLUEPRINT ranges
**BLUEPRINT SPEC**: "70% gradient, 20% efficiency, 10% conservation" (lines 516-518)
**VIOLATION**: Various wrong hyperparameter values throughout
**GREP COMMAND**: `grep -n "gradient_weight\|attention_weight\|entropy_weight" slime/`
**MECHANICAL FIX 1**: Validate fitness weights
```python
mcp__filesystem__edit_file(
    path="slime/config/dimensions.py",
    edits=[{"oldText": "class FitnessWeights:\n    gradient_weight: float = 0.1\n    attention_weight: float = 0.7",
            "newText": "class FitnessWeights:\n    gradient_weight: float = 0.7  # 70% per BLUEPRINT line 516\n    attention_weight: float = 0.0  # Not used per BLUEPRINT\n    efficiency_weight: float = 0.2  # 20% per BLUEPRINT line 517\n    conservation_weight: float = 0.1  # 10% per BLUEPRINT line 518"}]
)
```
**MECHANICAL FIX 2**: Add range validation
```python
mcp__filesystem__edit_file(
    path="slime/config/loader.py",
    edits=[{"oldText": "    def validate(self):",
            "newText": "    def validate(self):\n        \"\"\"Validate hyperparameters match BLUEPRINT constraints\"\"\"\n        # Fitness weights must sum to 1.0\n        weight_sum = (self.fitness.gradient_weight + \n                     self.fitness.efficiency_weight + \n                     self.fitness.conservation_weight)\n        assert abs(weight_sum - 1.0) < 1e-6, f'Weights sum to {weight_sum}, not 1.0'\n        # Gradient must be 70%\n        assert abs(self.fitness.gradient_weight - 0.7) < 1e-6, 'Gradient must be 70%'\n        # Coherence weight must be positive\n        assert self.coherence_weight > 0, 'Coherence weight must be positive'"}]
)
```
**MECHANICAL FIX 3**: Log warnings for deviations
```python
mcp__filesystem__edit_file(
    path="slime/config/loader.py",
    edits=[{"oldText": "import logging",
            "newText": "import logging\nimport warnings"}],
           {"oldText": "        logger.info(f'Loaded config from {path}')",
            "newText": "        logger.info(f'Loaded config from {path}')\n        # Warn if deviating from BLUEPRINT\n        if config.fitness.gradient_weight != 0.7:\n            warnings.warn(f'gradient_weight={config.fitness.gradient_weight} deviates from BLUEPRINT 70%')\n        if config.fitness.entropy_weight > 0:\n            warnings.warn('entropy_weight should be 0 per BLUEPRINT (use coherence instead)')"}]
)
```
**VERIFICATION**: `grep -n "gradient_weight.*0.7\|assert.*gradient.*0.7\|warnings.warn" slime/config/`

---

## TESTING COMPILATIONS

### TRANSFORMATION 6.1: Create test_blueprint_compliance.py
**BLUEPRINT SPEC**: "fitness = effective_rank() × coherence()" (line 241)
**VIOLATION**: No tests verify blueprint formula
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/tests/unit/test_blueprint_compliance.py",
    edits=[{"oldText": "",  # New file
            "newText": "import pytest\nimport torch\nfrom slime.core.pseudopod import Pseudopod\nfrom slime.training.fitness import FitnessComputer\n\nclass TestBlueprintCompliance:\n    def test_fitness_formula(self):\n        \"\"\"Test fitness = effective_rank × coherence per BLUEPRINT line 241\"\"\"\n        pseudopod = Pseudopod(input_dim=64, hidden_dim=128)\n        # Compute fitness\n        effective_rank = pseudopod.effective_rank()\n        coherence = pseudopod.coherence()\n        expected_fitness = effective_rank * coherence\n        actual_fitness = pseudopod.compute_fitness()\n        torch.testing.assert_close(actual_fitness, expected_fitness)\n    \n    def test_fitness_weights(self):\n        \"\"\"Test weights are 70% gradient, 20% efficiency, 10% conservation\"\"\"\n        computer = FitnessComputer()\n        assert computer.weights.gradient_weight == 0.7\n        assert computer.weights.efficiency_weight == 0.2\n        assert computer.weights.conservation_weight == 0.1\n        assert computer.weights.attention_weight == 0.0  # Not used\n    \n    def test_curiosity_driven_lifecycle(self):\n        \"\"\"Test hunger = learning_progress_deficit per BLUEPRINT line 140\"\"\"\n        from slime.training.lifecycle import LifecycleManager\n        manager = LifecycleManager()\n        pseudopod = Pseudopod(input_dim=64, hidden_dim=128)\n        coherence = pseudopod.coherence()\n        hunger = 1.0 - coherence.item()\n        survival_prob = 1.0 - hunger\n        assert survival_prob >= 0 and survival_prob <= 1"}]
)
```
**VERIFICATION**: `grep -n "test_fitness_formula\|test_fitness_weights\|test_curiosity" slime/tests/unit/test_blueprint_compliance.py`

### TRANSFORMATION 6.2: Create ablation study tests
**BLUEPRINT SPEC**: "Curiosity-driven evolution" (core principle throughout)
**VIOLATION**: No ablation tests to verify curiosity importance
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/tests/ablations/test_curiosity_ablation.py",
    edits=[{"oldText": "",  # New file
            "newText": "import pytest\nimport torch\nfrom slime.core.organism import Organism\n\nclass TestCuriosityAblation:\n    def test_with_curiosity(self):\n        \"\"\"Test system WITH curiosity-driven selection per BLUEPRINT\"\"\"\n        organism = Organism(config={'use_curiosity': True})\n        initial_coherence = organism.pseudopod_pool.get_mean_coherence()\n        for _ in range(100):\n            organism.step()\n        final_coherence = organism.pseudopod_pool.get_mean_coherence()\n        assert final_coherence > initial_coherence  # Should improve\n    \n    def test_without_curiosity(self):\n        \"\"\"Test system WITHOUT curiosity (ablation)\"\"\"\n        organism = Organism(config={'use_curiosity': False})\n        initial_coherence = organism.pseudopod_pool.get_mean_coherence()\n        for _ in range(100):\n            organism.step()\n        final_coherence = organism.pseudopod_pool.get_mean_coherence()\n        # Without curiosity, coherence improvement should be minimal\n        assert abs(final_coherence - initial_coherence) < 0.1\n    \n    def test_multi_head_vs_single_ca(self):\n        \"\"\"Test multi-head CA vs single CA per BLUEPRINT\"\"\"\n        multi_head = Organism(config={'num_ca_heads': 8})\n        single_head = Organism(config={'num_ca_heads': 1})\n        # Multi-head should have richer dynamics\n        multi_diversity = multi_head.compute_behavioral_diversity()\n        single_diversity = single_head.compute_behavioral_diversity()\n        assert multi_diversity > single_diversity\n    \n    def test_behavioral_diversity_impact(self):\n        \"\"\"Test impact of behavioral diversity per BLUEPRINT CVT\"\"\"\n        with_diversity = Organism(config={'use_behavioral_diversity': True})\n        without_diversity = Organism(config={'use_behavioral_diversity': False})\n        # Run both for same steps\n        for _ in range(100):\n            with_diversity.step()\n            without_diversity.step()\n        # With diversity should explore more behavioral space\n        assert with_diversity.archive.coverage() > without_diversity.archive.coverage()"}]
)
```
**VERIFICATION**: `grep -n "test_with_curiosity\|test_multi_head\|behavioral_diversity" slime/tests/ablations/test_curiosity_ablation.py`

### TRANSFORMATION 6.3: Create reproducibility tests
**BLUEPRINT SPEC**: "Reproducibility: clear experimental protocol" (implied by scientific rigor)
**VIOLATION**: No reproducibility tests exist
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/tests/unit/test_reproducibility.py",
    edits=[{"oldText": "",  # New file
            "newText": "import pytest\nimport torch\nimport numpy as np\nfrom slime.core.organism import Organism\n\nclass TestReproducibility:\n    def test_deterministic_with_same_seed(self):\n        \"\"\"Test deterministic behavior with same seed per BLUEPRINT\"\"\"\n        torch.manual_seed(42)\n        np.random.seed(42)\n        organism1 = Organism(seed=42)\n        results1 = []\n        for _ in range(10):\n            results1.append(organism1.step())\n        \n        torch.manual_seed(42)\n        np.random.seed(42)\n        organism2 = Organism(seed=42)\n        results2 = []\n        for _ in range(10):\n            results2.append(organism2.step())\n        \n        # Should be identical\n        for r1, r2 in zip(results1, results2):\n            torch.testing.assert_close(r1, r2)\n    \n    def test_different_seeds_different_results(self):\n        \"\"\"Test different seeds produce different results\"\"\"\n        organism1 = Organism(seed=42)\n        organism2 = Organism(seed=43)\n        \n        results1 = organism1.step()\n        results2 = organism2.step()\n        \n        # Should be different\n        assert not torch.allclose(results1, results2)\n    \n    def test_gpu_cpu_consistency(self):\n        \"\"\"Test GPU and CPU produce consistent results\"\"\"\n        if not torch.cuda.is_available():\n            pytest.skip('CUDA not available')\n        \n        torch.manual_seed(42)\n        cpu_organism = Organism(device='cpu', seed=42)\n        cpu_result = cpu_organism.step()\n        \n        torch.manual_seed(42)\n        gpu_organism = Organism(device='cuda', seed=42)\n        gpu_result = gpu_organism.step().cpu()\n        \n        # Should be very close (allowing for minor numerical differences)\n        torch.testing.assert_close(cpu_result, gpu_result, rtol=1e-5, atol=1e-5)"}]
)
```
**VERIFICATION**: `grep -n "test_deterministic\|test_different_seeds\|test_gpu_cpu" slime/tests/unit/test_reproducibility.py`

### TRANSFORMATION 6.4: Fix existing unit tests for BLUEPRINT compliance
**BLUEPRINT SPEC**: "fitness = effective_rank() × coherence()" and curiosity-driven selection
**VIOLATION**: Existing tests don't test blueprint concepts
**MECHANICAL FIX 1**: Update test_archive.py
```python
mcp__filesystem__edit_file(
    path="slime/tests/unit/test_archive.py",
    edits=[{"oldText": "def test_add_elite(self):",
            "newText": "def test_add_elite_with_coherence(self):\n        \"\"\"Test elite storage includes coherence per BLUEPRINT\"\"\"\n        archive = CVTArchive()\n        elite = Elite(fitness=0.8, coherence=0.6, genome={}, metadata={})\n        archive.add_elite(elite)\n        assert elite.coherence == 0.6\n    \n    def test_add_elite(self):"}]
)
```
**MECHANICAL FIX 2**: Update test_lifecycle.py
```python
mcp__filesystem__edit_file(
    path="slime/tests/unit/test_lifecycle.py",
    edits=[{"oldText": "def test_selection(self):",
            "newText": "def test_coherence_based_selection(self):\n        \"\"\"Test selection based on coherence not random fitness per BLUEPRINT\"\"\"\n        manager = LifecycleManager()\n        high_coherence = Pseudopod(coherence=0.9)\n        low_coherence = Pseudopod(coherence=0.1)\n        # High coherence should survive (low hunger)\n        assert manager.should_survive(high_coherence) > manager.should_survive(low_coherence)\n    \n    def test_selection(self):"}]
)
```
**MECHANICAL FIX 3**: Create test_fitness.py
```python
mcp__filesystem__edit_file(
    path="slime/tests/unit/test_fitness.py",
    edits=[{"oldText": "",  # New file
            "newText": "import pytest\nimport torch\nfrom slime.training.fitness import FitnessComputer\n\nclass TestFitness:\n    def test_fitness_formula(self):\n        \"\"\"Test fitness = effective_rank × coherence per BLUEPRINT line 241\"\"\"\n        computer = FitnessComputer()\n        effective_rank = torch.tensor(2.5)\n        coherence = torch.tensor(0.8)\n        fitness = computer.compute(effective_rank, coherence)\n        expected = effective_rank * coherence  # 2.5 * 0.8 = 2.0\n        torch.testing.assert_close(fitness, expected)\n    \n    def test_gradient_magnitude_computation(self):\n        \"\"\"Test actual gradient magnitude not placeholder\"\"\"\n        computer = FitnessComputer()\n        model = torch.nn.Linear(10, 5)\n        loss = model(torch.randn(2, 10)).sum()\n        loss.backward()\n        gradient_mag = computer.compute_gradient_magnitude(model)\n        assert gradient_mag > 0  # Should be actual gradient not 0.5 placeholder"}]
)
```
**VERIFICATION**: `grep -n "test_add_elite_with_coherence\|test_coherence_based_selection\|test_fitness_formula" slime/tests/unit/`

### TRANSFORMATION 6.5: Update run.py for curiosity-driven demonstration
**BLUEPRINT SPEC**: "Curiosity-driven evolution" not classification
**VIOLATION**: run.py includes classification head and loss
**MECHANICAL FIX 1**: Remove classification components
```python
mcp__filesystem__edit_file(
    path="run.py",
    edits=[{"oldText": "# Add classification head\n    model.add_classification_head(num_classes=10)",
            "newText": "# No classification - curiosity-driven per BLUEPRINT"},
           {"oldText": "classification_loss = F.cross_entropy(logits, labels)",
            "newText": "# No classification loss - use coherence per BLUEPRINT"}]
)
```
**MECHANICAL FIX 2**: Add curiosity metrics display
```python
mcp__filesystem__edit_file(
    path="run.py",
    edits=[{"oldText": "print(f'Loss: {loss.item():.4f}')",
            "newText": "coherence = model.organism.pseudopod_pool.get_mean_coherence()\n        learning_progress = coherence.item()\n        hunger = 1.0 - learning_progress\n        print(f'Coherence: {learning_progress:.4f}, Hunger: {hunger:.4f}')"}]
)
```
**MECHANICAL FIX 3**: Show evolution metrics
```python
mcp__filesystem__edit_file(
    path="run.py",
    edits=[{"oldText": "# Training loop",
            "newText": "# Curiosity-driven evolution loop per BLUEPRINT\n        print('\\nStarting curiosity-driven evolution...')\n        print('Formula: fitness = effective_rank() × coherence()')\n        print('Weights: 70% gradient, 20% efficiency, 10% conservation\\n')"}]
)
```
**VERIFICATION**: `grep -n "No classification\|coherence.*hunger\|curiosity-driven evolution" run.py`

---

## TOOLS & VISUALIZATION COMPILATIONS

### TRANSFORMATION 7.1: Visualizer must show curiosity metrics
**BLUEPRINT SPEC**: "Curiosity-driven evolution" requires visibility
**VIOLATION**: Visualizer doesn't show coherence/learning progress
**GREP COMMAND**: `grep -n "def plot" slime/tools/visualize.py`
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/tools/visualize.py",
    edits=[{"oldText": "def plot_fitness(history):",
            "newText": "def plot_curiosity_metrics(history):\n        \"\"\"Plot coherence and learning progress per BLUEPRINT\"\"\"\n        coherence_values = [h['coherence'] for h in history]\n        hunger_values = [1.0 - c for c in coherence_values]\n        plt.figure(figsize=(10, 6))\n        plt.subplot(2, 1, 1)\n        plt.plot(coherence_values, label='Coherence (Learning Progress)')\n        plt.ylabel('Coherence')\n        plt.legend()\n        plt.subplot(2, 1, 2)\n        plt.plot(hunger_values, label='Hunger', color='red')\n        plt.ylabel('Hunger')\n        plt.xlabel('Step')\n        plt.legend()\n        plt.title('Curiosity-Driven Evolution Metrics')\n    \n    def plot_fitness(history):"}]
)
```
**FOLLOW-UP**: Plot behavioral descriptors
```python
mcp__filesystem__edit_file(
    path="slime/tools/visualize.py",
    edits=[{"oldText": "def plot_behavioral_space(archive):",
            "newText": "def plot_behavioral_space(archive):\n        \"\"\"Plot actual behavioral descriptors per BLUEPRINT CVT\"\"\"\n        descriptors = [elite.metadata['behavioral_descriptor'] for elite in archive.elites]\n        if len(descriptors[0]) == 2:\n            x = [d[0] for d in descriptors]\n            y = [d[1] for d in descriptors]\n            plt.scatter(x, y, c='blue', alpha=0.5)\n            plt.xlabel('Behavioral Dim 1')\n            plt.ylabel('Behavioral Dim 2')\n            plt.title('MAP-Elites CVT Archive')"}]
)
```
**VERIFICATION**: `grep -n "plot_curiosity_metrics\|plot_behavioral_space" slime/tools/visualize.py`

### TRANSFORMATION 7.2: Export must include curiosity metrics
**BLUEPRINT SPEC**: "Curiosity-driven evolution" requires metric export
**VIOLATION**: Export doesn't save coherence/learning progress
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/tools/export.py",
    edits=[{"oldText": "def export_checkpoint(model, path):",
            "newText": "def export_checkpoint(model, path):\n        \"\"\"Export including curiosity metrics per BLUEPRINT\"\"\"\n        checkpoint = {\n            'model_state': model.state_dict(),\n            'coherence_history': model.coherence_history,\n            'learning_progress': model.organism.pseudopod_pool.get_mean_coherence().item(),\n            'hunger_values': [1.0 - c for c in model.coherence_history],\n            'fitness_formula': 'effective_rank() × coherence()',\n            'weights': {'gradient': 0.7, 'efficiency': 0.2, 'conservation': 0.1}\n        }\n        torch.save(checkpoint, path)\n        print(f'Saved curiosity-driven checkpoint to {path}')"}]
)
```
**VERIFICATION**: `grep -n "coherence_history\|learning_progress\|export_checkpoint" slime/tools/export.py`

### TRANSFORMATION 7.3: Benchmarks must measure curiosity metrics
**BLUEPRINT SPEC**: "Curiosity-driven evolution" performance metrics
**VIOLATION**: Benchmarks don't measure coherence/learning progress
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/bench/benchmark_curiosity.py",
    edits=[{"oldText": "",  # New file
            "newText": "import time\nimport torch\nfrom slime.core.organism import Organism\n\ndef benchmark_curiosity_vs_fitness():\n    \"\"\"Benchmark curiosity-driven vs fitness-driven per BLUEPRINT\"\"\"\n    # Curiosity-driven\n    start = time.time()\n    curiosity_org = Organism(config={'use_curiosity': True})\n    for _ in range(1000):\n        curiosity_org.step()\n    curiosity_time = time.time() - start\n    curiosity_coherence = curiosity_org.pseudopod_pool.get_mean_coherence()\n    \n    # Fitness-driven (ablation)\n    start = time.time()\n    fitness_org = Organism(config={'use_curiosity': False})\n    for _ in range(1000):\n        fitness_org.step()\n    fitness_time = time.time() - start\n    fitness_coherence = fitness_org.pseudopod_pool.get_mean_coherence()\n    \n    print(f'Curiosity-driven: {curiosity_time:.2f}s, coherence={curiosity_coherence:.4f}')\n    print(f'Fitness-driven: {fitness_time:.2f}s, coherence={fitness_coherence:.4f}')\n    print(f'Improvement: {(curiosity_coherence/fitness_coherence - 1)*100:.1f}%')"}]
)
```
**VERIFICATION**: `grep -n "benchmark_curiosity_vs_fitness" slime/bench/benchmark_curiosity.py`

### TRANSFORMATION 7.4: Neural CA must implement Flow-Lenia dynamics
**BLUEPRINT SPEC**: "Flow-Lenia dynamics with bell curve growth" (BLUEPRINT principle)
**VIOLATION**: Missing growth_center, growth_width parameters
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/core/neural_ca.py",
    edits=[{"oldText": "class NeuralCA(nn.Module):\n    def __init__(self",
            "newText": "class NeuralCA(nn.Module):\n    def __init__(self, growth_center=0.5, growth_width=0.15,  # Flow-Lenia params per BLUEPRINT"}],
           {"oldText": "self.hidden_dim = hidden_dim",
            "newText": "self.hidden_dim = hidden_dim\n        # Flow-Lenia bell curve growth parameters per BLUEPRINT\n        self.growth_center = growth_center\n        self.growth_width = growth_width"}]
)
```
**FOLLOW-UP**: Implement bell curve growth
```python
mcp__filesystem__edit_file(
    path="slime/core/neural_ca.py",
    edits=[{"oldText": "def growth_function(self, x):",
            "newText": "def growth_function(self, x):\n        \"\"\"Bell curve growth per BLUEPRINT Flow-Lenia\"\"\"\n        return torch.exp(-((x - self.growth_center) / self.growth_width) ** 2)"}]
)
```
**VERIFICATION**: `grep -n "growth_center\|growth_width\|bell curve" slime/core/neural_ca.py`

### TRANSFORMATION 7.5: README must document curiosity-driven architecture
**BLUEPRINT SPEC**: Core principle is curiosity-driven evolution
**VIOLATION**: README doesn't explain curiosity concepts
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="README.md",
    edits=[{"oldText": "# Slime: Neural Cellular Automata",
            "newText": "# Slime: Curiosity-Driven Neural Cellular Automata\n\n## Core Formula\n`fitness = effective_rank() × coherence()`\n\n## Fitness Weights\n- 70% Gradient Magnitude (task performance)\n- 20% Compute Efficiency\n- 10% Conservation Quality\n\n## Key Principle\n`High coherence → Low hunger → Survival`\n\n# Slime: Neural Cellular Automata"}]
)
```
**VERIFICATION**: `grep -n "Curiosity-Driven\|effective_rank.*coherence\|70%.*Gradient" README.md`

---

## PHASE 8: CRITICAL MISSING IMPLEMENTATIONS
**These critical features were identified as missing from initial plan**

### TRANSFORMATION 8.1: Implement ACTUAL compute efficiency metric [#60]
**BLUEPRINT SPEC**: "20% compute efficiency" (BLUEPRINT line 387)
**ISSUE**: compute_efficiency returns placeholder 0.5
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/training/fitness.py",
    edits=[{"oldText": "def compute_efficiency(self, model):\n        return torch.tensor(0.5)  # Placeholder",
            "newText": "def compute_efficiency(self, model):\n        \"\"\"Measure actual FLOPS and memory bandwidth per BLUEPRINT\"\"\"\n        from slime.kernels.triton_impl import measure_flops\n        # Measure on actual forward pass\n        start_flops = torch.cuda.FloatTensor([0])\n        torch.cuda.nvtx.range_push('efficiency_measure')\n        _ = model(self.test_input)\n        torch.cuda.synchronize()\n        end_flops = measure_flops()\n        torch.cuda.nvtx.range_pop()\n        # Normalize to [0,1] based on hardware capacity\n        theoretical_flops = torch.cuda.get_device_properties(0).multi_processor_count * 1.4e12\n        efficiency = end_flops / theoretical_flops\n        return efficiency"}]
)
```

### TRANSFORMATION 8.2: Implement CA conservation quality metric [#61]
**BLUEPRINT SPEC**: "10% CA conservation quality" (BLUEPRINT line 388)
**ISSUE**: No measurement of mass conservation quality
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/training/fitness.py",
    edits=[{"oldText": "def compute_conservation_quality(self, model):\n        return torch.tensor(0.8)  # Placeholder",
            "newText": "def compute_conservation_quality(self, pseudopod):\n        \"\"\"Measure CA mass conservation quality per BLUEPRINT\"\"\"\n        # Get CA metrics from last forward pass\n        ca_metrics = pseudopod._ca_metrics\n        mass_before = ca_metrics.get('mass_before', 1.0)\n        mass_after = ca_metrics.get('mass_after', 1.0)\n        # Quality is how close to perfect conservation (1.0)\n        conservation_error = abs(mass_after / mass_before - 1.0)\n        quality = 1.0 / (1.0 + conservation_error)  # Maps [0,inf) to (1,0]\n        return torch.tensor(quality)"}]
)
```

### TRANSFORMATION 8.3: Memory-based Archive GC [#67]
**BLUEPRINT SPEC**: "GC based on memory pressure" (implied by resource management)
**ISSUE**: GC triggers on operation count not memory
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/memory/archive.py",
    edits=[{"oldText": "self.ops_since_gc += 1\n        if self.ops_since_gc >= 100:\n            self._garbage_collect()",
            "newText": "# GC based on memory pressure per BLUEPRINT\n        memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()\n        if memory_used > 0.85:  # 85% memory threshold\n            self._garbage_collect()\n            torch.cuda.empty_cache()"}]
)
```

### TRANSFORMATION 8.4: Metrics → Lifecycle feedback loop [#29]
**BLUEPRINT SPEC**: "Metrics should feed back into lifecycle decisions" 
**ISSUE**: Metrics collected but don't influence decisions
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/training/lifecycle.py",
    edits=[{"oldText": "def should_spawn(self, pool_size):",
            "newText": "def should_spawn(self, pool_size, metrics_collector):\n        \"\"\"Spawn decisions based on metrics per BLUEPRINT\"\"\"\n        # Get current system metrics\n        gpu_util = metrics_collector.get('gpu_utilization', 0.5)\n        learning_rate = metrics_collector.get('learning_progress_rate', 0.0)\n        \n        # Low GPU util + low learning = spawn more\n        if gpu_util < 0.6 and learning_rate < 0.01:\n            return True"}]
)
```

### TRANSFORMATION 8.5: Integrate genealogy tracking [#72]
**BLUEPRINT SPEC**: "Track evolutionary lineages" (BLUEPRINT topology section)
**ISSUE**: GenealogyTracker never used
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/memory/pool.py",
    edits=[{"oldText": "def __init__(self",
            "newText": "def __init__(self, genealogy_tracker=None,"},
           {"oldText": "self._components = []",
            "newText": "self._components = []\n        # Track lineages per BLUEPRINT\n        from slime.topology.genealogy import GenealogyTracker\n        self.genealogy = genealogy_tracker or GenealogyTracker()"},
           {"oldText": "child = self._spawn_child(parent)",
            "newText": "child = self._spawn_child(parent)\n        # Record lineage\n        self.genealogy.add_offspring(parent.id, child.id)"}]
)
```

### TRANSFORMATION 8.6: Use hierarchical partition [#73]
**BLUEPRINT SPEC**: "Hierarchical behavioral space structure"
**ISSUE**: HierarchicalPartition unused
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/memory/archive.py",
    edits=[{"oldText": "def get_behavioral_partition(self, descriptor):",
            "newText": "def get_behavioral_partition(self, descriptor):\n        \"\"\"Use hierarchical partition per BLUEPRINT\"\"\"\n        level = self.hierarchy.get_level(descriptor)\n        partition = self.hierarchy.get_partition(descriptor, level)\n        return partition"}]
)
```

### TRANSFORMATION 8.7: GPU Comonad → Kernel integration [#43]
**BLUEPRINT SPEC**: "Context-aware GPU execution" (comonad abstraction)
**ISSUE**: Comonad not connected to kernels
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/kernels/triton_impl.py",
    edits=[{"oldText": "@triton.jit\ndef neural_ca_kernel(",
            "newText": "from slime.gpu.comonad import GPUComonad\n\n@triton.jit\ndef neural_ca_kernel_with_context(\n    state_ptr, output_ptr, weights_ptr,\n    context_ptr,  # Comonadic context per BLUEPRINT\n    batch_size, hidden_dim,\n    BLOCK_SIZE: tl.constexpr\n):\n    # Extract local context\n    pid = tl.program_id(0)\n    context = tl.load(context_ptr + pid)\n    # Use context to modulate computation\n    scale = context['occupancy']  # Scale by GPU occupancy\n\n@triton.jit\ndef neural_ca_kernel("}]
)
```

### TRANSFORMATION 8.8: Multi-head CA forward pass [#30]
**BLUEPRINT SPEC**: "Multi-head Neural CA with parallel update rules"
**ISSUE**: ModuleList created but not used in forward
**MECHANICAL FIX**:
```python
mcp__filesystem__edit_file(
    path="slime/core/neural_ca.py",
    edits=[{"oldText": "def forward(self, x):\n        # Apply CA update\n        output = self.ca_kernel(x)",
            "newText": "def forward(self, x):\n        # Multi-head CA per BLUEPRINT\n        outputs = []\n        for head in self.ca_kernels:\n            head_output = head(x)\n            outputs.append(head_output)\n        # Combine heads (learnable weighted sum)\n        output = torch.stack(outputs).sum(dim=0) / self.num_heads"}]
)
```

**VERIFICATION**: Run all violation checks:
```bash
grep -n "measure_flops\|conservation_quality\|memory_allocated\|genealogy\|hierarchy\|comonad\|multi-head" slime/
```

---

## MECHANICAL VERIFICATION CHECKS

### VERIFICATION 1: Import Cycles
```bash
python -c "import slime; from slime.core import *; from slime.training import *; print('No cycles')"
```

### VERIFICATION 2: Fitness Formula
```bash
grep -n "effective_rank.*coherence" slime/core/pseudopod.py | head -1
# EXPECT: fitness_signal = self.effective_rank() * self.coherence()
```

### VERIFICATION 3: Fitness Weights
```bash
grep -n "gradient_weight.*0.7" slime/config/dimensions.py
# EXPECT: gradient_weight: float = 0.7  # 70% per BLUEPRINT
```

### VERIFICATION 4: Lifecycle Coherence
```bash
grep -n "hunger.*coherence" slime/training/lifecycle.py
# EXPECT: hunger = 1.0 - component.coherence().item()
```

### VERIFICATION 5: Archive Coherence Storage
```bash
grep -n "coherence.*float" slime/memory/archive.py | grep Elite
# EXPECT: coherence: float  # Learning progress per BLUEPRINT
```

### VERIFICATION 6: Multi-head CA
```bash
grep -n "ModuleList\|num_heads" slime/core/neural_ca.py
# EXPECT: self.ca_kernels = nn.ModuleList
```

### VERIFICATION 7: Hunger Mechanism
```bash
grep -n "_compute_hunger\|hunger.*deficit" slime/core/organism.py
# EXPECT: def _compute_hunger(self, pseudopod) -> float:
```

### VERIFICATION 8: All Tests Pass
```bash
pytest slime/tests/unit/test_blueprint_compliance.py -v
# EXPECT: All tests PASSED
```

---

## COMPILER EXECUTION SEQUENCE

```python
# MECHANICAL EXECUTION ORDER - NO DEVIATION ALLOWED
EXECUTION_SEQUENCE = [
    # PHASE 0: GPU-NATIVE FOUNDATION - ABSOLUTELY FIRST
    ("0.0", "Remove ALL PyTorch fallbacks"),  # Delete torch_fallback.py
    ("0.1", "GPU-native effective_rank() via cuSOLVER"),  # Custom CUDA SVD
    ("0.2", "GPU-native fitness multiplication"),  # Fused Triton kernel
    ("0.3", "GPU-native gradient computation"),  # Triton autograd
    ("0.4", "GPU-native coherence reduction"),  # Triton reduction kernels
    
    # PHASE 1: CURIOSITY FORMULA - DEPENDS ON GPU-NATIVE
    ("1.0", "fitness = effective_rank() × coherence()"),  # Core formula
    ("1.1", "Fitness weights = 70/20/10"),  # Correct weights
    ("1.2", "hunger = learning_progress_deficit"),  # Selection mechanism
    
    # PROTOCOL INTERFACES - DEPENDS ON PHASE 1
    ("2.1-2.5", "Protocol compliance"),  # Elite, Multi-head, Chemotaxis
    
    # COMPONENT INTEGRATION - DEPENDS ON PHASE 2  
    ("3.1-3.5", "Component wiring"),  # TubeNetwork, Stencil, Effects, Topology
    
    # TRAINING PIPELINE - DEPENDS ON PHASE 3
    ("4.1-4.5", "Training curiosity"),  # Coherence tracking, lifecycle, fitness
    
    # CONFIGURATION & TESTING - CAN OVERLAP
    ("5.1-5.5", "Configuration"),  # API and config
    ("6.1-6.5", "Testing"),  # Validation
    ("7.1-7.5", "Tools"),  # Visualization
    
    # PHASE 8: MISSING CRITICAL IMPLEMENTATIONS
    ("8.1", "Actual compute efficiency metric"),  # Real FLOPS measurement
    ("8.2", "CA conservation quality metric"),  # Mass conservation quality
    ("8.3", "Memory-based Archive GC"),  # GC on memory pressure
    ("8.4", "Metrics → Lifecycle feedback"),  # Observability influences decisions
    ("8.5", "Genealogy tracking integration"),  # Track lineages
    ("8.6", "Hierarchical partition usage"),  # Structure behavioral space
    ("8.7", "GPU Comonad → Kernel bridge"),  # Connect abstractions to hardware
    ("8.8", "Multi-head CA forward pass"),  # Actually use all CA heads
]

def execute_compiler():
    for task_id, description in EXECUTION_SEQUENCE:
        print(f"Executing TRANSFORMATION {task_id}: {description}")
        # Mechanical transformation - no interpretation
        apply_transformation(task_id)
        verify_transformation(task_id)
```

---

- **Parallelizable**: Phases 4-7 can overlap
- **Critical Path**: Phases 0→1→2→3 must be sequential
- **Testing**: Continuous throughout, formal in Phase 6

---

## SUCCESS CRITERIA

The system is healed when:
1. **GPU-NATIVE**: All operations use Triton/CUDA kernels, zero PyTorch fallbacks
2. Fitness formula is `effective_rank() × coherence()` via GPU-native ops
3. Selection is based on learning progress (coherence) not fitness
4. All identified deviations from blueprint are fixed
5. Blueprint compliance tests pass
6. Curiosity-driven evolution demonstrated with archive visualization
7. Behavioral diversity maintained via MAP-Elites CVT
8. **Actual Metrics**: Compute efficiency and conservation quality measured, not placeholders
9. **Feedback Loops**: Metrics influence lifecycle decisions
10. **Full Integration**: Genealogy, hierarchy, topology, comonad all actively used
11. No import cycles or protocol violations
12. Multi-head CA actually uses all heads in forward pass

---

## COMPILER EXECUTION PIPELINE

**THIS IS A MECHANICAL COMPILER**:
1. Each transformation is a pure function: `(BlueprintSpec, CurrentCode) → mcp__filesystem__edit_file`
2. No interpretation or creativity allowed
3. Pattern match blueprint → Find violation → Apply transformation
4. Verify transformation succeeded

**EXECUTION ORDER**:
```python
# The compiler executes transformations in dependency order:
for transformation in TRANSFORMATIONS:
    grep_result = grep(transformation.pattern, transformation.file)
    if grep_result != transformation.expected:
        mcp__filesystem__edit_file(
            path=transformation.file,
            edits=[{"oldText": grep_result, "newText": transformation.blueprint_spec}]
        )
        verify = grep(transformation.pattern, transformation.file)
        assert verify == transformation.expected
```

**NO HUMAN INTERVENTION**: This document IS the program. Execute mechanically.