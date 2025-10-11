import torch
import torch.nn as nn
import triton
import triton.language as tl
from dataclasses import dataclass
from typing import Optional
from collections import deque


@triton.jit
def effective_rank_kernel(
    correlation_ptr,
    rank_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Compute effective rank from eigenvalues via entropy on GPU."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    eigenvals = tl.load(correlation_ptr + offsets, mask=mask, other=0.0)
    eigenvals = tl.where(eigenvals > 1e-10, eigenvals, 0.0)
    
    total = tl.sum(eigenvals)
    probs = eigenvals / (total + 1e-10)
    log_probs = tl.log(probs + 1e-10)
    entropy = -tl.sum(probs * log_probs)
    
    rank = tl.exp(entropy)
    tl.store(rank_ptr, rank)


@triton.jit
def pseudopod_correlation_kernel(
    key_ptr, value_ptr, corr_ptr,
    batch_size, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Compute normalized correlation between key and value tensors."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    k_ptrs = key_ptr + offs_m[:, None] * head_dim + tl.arange(0, head_dim)[None, :]
    v_ptrs = value_ptr + offs_n[:, None] * head_dim + tl.arange(0, head_dim)[None, :]
    
    k = tl.load(k_ptrs, mask=offs_m[:, None] < batch_size, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < batch_size, other=0.0)
    
    # Normalize vectors
    k_norm = tl.sqrt(tl.sum(k * k, axis=1, keep_dims=True) + 1e-10)
    v_norm = tl.sqrt(tl.sum(v * v, axis=1, keep_dims=True) + 1e-10)
    
    k = k / k_norm
    v = v / v_norm
    
    # Compute correlation
    corr = tl.dot(k, tl.trans(v))
    
    corr_ptrs = corr_ptr + offs_m[:, None] * batch_size + offs_n[None, :]
    tl.store(corr_ptrs, corr, mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < batch_size))


@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    seq_len, head_dim, temperature,
    BLOCK_SIZE: tl.constexpr
):
    """Fused attention with metabolic temperature modulation."""
    row_idx = tl.program_id(0)
    
    # Load query vector
    q_offs = tl.arange(0, head_dim)
    q = tl.load(q_ptr + row_idx * head_dim + q_offs)
    
    # Normalize query
    q_norm = tl.sqrt(tl.sum(q * q) + 1e-10)
    q = q / q_norm
    
    # Accumulator for weighted values
    acc = tl.zeros([head_dim], dtype=tl.float32)
    scale = 1.0 / (tl.sqrt(float(head_dim)) * temperature)
    
    # Max value for numerical stability
    max_score = float('-inf')
    
    # First pass: find max score for numerical stability
    for col_start in range(0, seq_len, BLOCK_SIZE):
        col_offs = col_start + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offs < seq_len
        
        k = tl.load(k_ptr + col_offs[:, None] * head_dim + q_offs[None, :], 
                   mask=col_mask[:, None], other=0.0)
        k_norm = tl.sqrt(tl.sum(k * k, axis=1, keep_dims=True) + 1e-10)
        k = k / k_norm
        
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(col_mask, scores, float('-inf'))
        
        block_max = tl.max(scores)
        max_score = tl.maximum(max_score, block_max)
    
    # Second pass: compute softmax and accumulate
    sum_exp = 0.0
    for col_start in range(0, seq_len, BLOCK_SIZE):
        col_offs = col_start + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offs < seq_len
        
        k = tl.load(k_ptr + col_offs[:, None] * head_dim + q_offs[None, :],
                   mask=col_mask[:, None], other=0.0)
        k_norm = tl.sqrt(tl.sum(k * k, axis=1, keep_dims=True) + 1e-10)
        k = k / k_norm
        
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(col_mask, scores, float('-inf'))
        
        # Numerically stable softmax
        scores_exp = tl.exp(scores - max_score)
        scores_exp = tl.where(col_mask, scores_exp, 0.0)
        
        sum_exp += tl.sum(scores_exp)
        
        v = tl.load(v_ptr + col_offs[:, None] * head_dim + q_offs[None, :],
                   mask=col_mask[:, None], other=0.0)
        
        acc += tl.sum(scores_exp[:, None] * v, axis=0)
    
    # Normalize by sum of exponentials
    acc = acc / (sum_exp + 1e-10)
    
    tl.store(out_ptr + row_idx * head_dim + q_offs, acc)


@dataclass
class FlowState:
    """State of the slime mold's exploration through information space."""
    pseudopodia: torch.Tensor
    tubes: deque
    chemotaxis: dict
    dimensionality: Optional[torch.Tensor]
    
    def entropy(self):
        """Shannon entropy of the dimensionality wavefunction."""
        if self.dimensionality is None:
            return torch.tensor(float('inf'))
        probs = self.dimensionality ** 2
        probs = probs / (probs.sum() + 1e-10)
        return -torch.sum(probs * torch.log(probs + 1e-10))
    
    def mutual_information_with(self, flow_memory):
        """Compute mutual information between current state and historical flow."""
        if not flow_memory or self.dimensionality is None:
            return torch.tensor(0.0, device=self.dimensionality.device if self.dimensionality is not None else 'cpu')
        
        history = torch.stack(list(flow_memory))
        h_flat = history.flatten(-2)
        d_expanded = self.dimensionality.unsqueeze(0).expand(h_flat.shape[0], -1)
        
        min_dim = min(h_flat.shape[-1], d_expanded.shape[-1])
        h_flat = h_flat[..., :min_dim]
        d_expanded = d_expanded[..., :min_dim]
        
        # Compute joint probabilities
        joint_probs = torch.abs(h_flat * d_expanded)
        joint_probs = joint_probs / (joint_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Marginal probabilities
        marginal_h = joint_probs.mean(dim=0) + 1e-10
        marginal_d = self.dimensionality[:min_dim] ** 2
        marginal_d = marginal_d / (marginal_d.sum() + 1e-10)
        
        # Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        h_marginal = -torch.sum(marginal_h * torch.log(marginal_h))
        d_marginal = -torch.sum(marginal_d * torch.log(marginal_d + 1e-10))
        h_joint = -torch.mean(torch.sum(joint_probs * torch.log(joint_probs + 1e-10), dim=-1))
        
        return h_marginal + d_marginal - h_joint


class Pseudopodium:
    """A sensory extension probing the information landscape."""
    
    def __init__(self, key, value, query):
        self.key = key
        self.value = value
        self.query = query
        self.correlation = self._compute_correlation(key, value)
    
    @staticmethod
    def _compute_correlation(key, value):
        """Compute correlation using optimized Triton kernel."""
        batch_size = key.shape[0]
        head_dim = key.shape[1]
        
        correlation = torch.empty(batch_size, batch_size, device=key.device, dtype=key.dtype)
        
        BLOCK_M = min(32, triton.next_power_of_2(batch_size))
        BLOCK_N = min(32, triton.next_power_of_2(batch_size))
        
        grid = (triton.cdiv(batch_size, BLOCK_M), triton.cdiv(batch_size, BLOCK_N))
        
        pseudopod_correlation_kernel[grid](
            key, value, correlation,
            batch_size, head_dim,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2
        )
        
        return correlation
    
    def effective_rank(self):
        """Compute effective rank via exponential of entropy of singular values."""
        s = torch.linalg.svdvals(self.correlation)
        s = s[s > 1e-6]
        if s.numel() == 0:
            return torch.tensor(1.0, device=self.correlation.device)
        s_norm = s / (s.sum() + 1e-10)
        return torch.exp(-torch.sum(s_norm * torch.log(s_norm + 1e-10)))
    
    def coherence(self):
        """Measure of how well the correlation structure is preserved under inversion."""
        eye = torch.eye(self.correlation.shape[0], device=self.correlation.device)
        partial = torch.linalg.solve(self.correlation + eye * 1e-3, eye)
        
        corr_sq = torch.sum(self.correlation ** 2)
        partial_sq = torch.sum(partial ** 2)
        
        return corr_sq / (corr_sq + partial_sq + 1e-10)


class TubeNetwork:
    """Temporal memory structure with exponential decay."""
    
    def __init__(self, decay=0.95, capacity=100):
        self.decay = decay
        self.capacity = capacity
        self.tubes = deque(maxlen=capacity)
        self._discount_cache = None
    
    def flow(self, correlation, conductance):
        """Add new correlation pattern weighted by conductance."""
        self.tubes.append(correlation * conductance)
        self._discount_cache = None  # Invalidate cache
        return self
    
    def potential_field(self):
        """Compute discounted potential field from tube history."""
        if not self.tubes:
            return None
        
        tube_stack = torch.stack(list(self.tubes))
        device = tube_stack.device
        
        # Lazy computation of discount factors
        if self._discount_cache is None or len(self._discount_cache) != len(self.tubes):
            self._discount_cache = torch.tensor(
                [self.decay ** i for i in range(len(self.tubes))],
                device=device,
                dtype=tube_stack.dtype
            )
        
        return tube_stack * self._discount_cache.view(-1, 1, 1)


class FoodSource:
    """Spatial map of nutrients in behavioral (rank, coherence) space."""
    
    def __init__(self, behavioral_dims=2, resolution=50):
        self.resolution = resolution
        self.sources = {}
    
    def deposit(self, nutrient, location, concentration):
        """Deposit nutrient at behavioral coordinates with given concentration."""
        idx = tuple(int(coord * self.resolution) % self.resolution for coord in location)
        
        concentration_value = concentration.item() if torch.is_tensor(concentration) else concentration
        
        if idx not in self.sources or self.sources[idx]['concentration'] < concentration_value:
            self.sources[idx] = {
                'nutrient': nutrient.detach() if torch.is_tensor(nutrient) else nutrient,
                'location': location,
                'concentration': concentration_value
            }
    
    def forage(self, metabolic_rate=1.0, hunger=0.0):
        """Sample nutrient proportional to concentration, modulated by metabolism."""
        if not self.sources:
            return None
        
        concentrations = torch.tensor([s['concentration'] for s in self.sources.values()])
        device = list(self.sources.values())[0]['nutrient'].device
        
        # Numerically stable softmax with log-sum-exp trick
        logits = (concentrations.to(device) + hunger * 0.1) / (metabolic_rate + 1e-10)
        logits = logits - logits.max()  # Prevent overflow
        weights = torch.softmax(logits, dim=0)
        
        nutrients = torch.stack([s['nutrient'] for s in self.sources.values()])
        return torch.sum(nutrients * weights.view(-1, 1, 1), dim=0)


class Plasmodium(nn.Module):
    """Slime mold intelligence: self-organizing neural architecture with organic memory."""
    
    def __init__(self, sensory_dim, latent_dim, n_pseudopodia=4):
        super().__init__()
        
        assert latent_dim % n_pseudopodia == 0, "latent_dim must be divisible by n_pseudopodia"
        assert latent_dim % 32 == 0, "latent_dim must be divisible by 32 for Triton efficiency"
        
        self.encode = nn.Sequential(nn.Linear(sensory_dim, latent_dim), nn.Tanh())
        self.decode = nn.Sequential(nn.Linear(latent_dim, sensory_dim), nn.Tanh())
        self.project_query = nn.Linear(sensory_dim, latent_dim)
        self.project_key = nn.Linear(latent_dim, latent_dim)
        self.project_value = nn.Linear(latent_dim, latent_dim)
        self.normalize = nn.LayerNorm(latent_dim)
        self.n_pseudopodia = n_pseudopodia
        self.head_dim = latent_dim // n_pseudopodia
        self.predict_coherence = nn.Linear(latent_dim, 1)
        self.predict_rank = nn.Linear(latent_dim, 1)
    
    def metabolic_rate(self, step, period=1000):
        """Circadian rhythm for exploration-exploitation balance."""
        phase = torch.tensor(step * 3.14159 / period)
        return 0.5 * (1 + torch.cos(phase))
    
    def extend_pseudopodia(self, body, stimulus):
        """Extend sensory probes into information space."""
        body = body + self.normalize(self.encode(stimulus))
        k = self.project_key(body)
        v = self.project_value(body)
        q = self.project_query(stimulus)
        
        pseudopodia = []
        for i in range(self.n_pseudopodia):
            start = i * self.head_dim
            end = start + self.head_dim
            pseudopodia.append(Pseudopodium(
                k[:, start:end].contiguous(),
                v[:, start:end].contiguous(),
                q[:, start:end].contiguous()
            ))
        
        return pseudopodia
    
    def merge_pseudopodia(self, pseudopodia, temperature):
        """Merge multiple pseudopodial observations weighted by coherence."""
        correlations = torch.stack([p.correlation for p in pseudopodia])
        coherences = torch.stack([p.coherence() for p in pseudopodia])
        
        # Temperature-modulated softmax over coherences
        logits = torch.log(coherences + 1e-10) / (temperature + 1e-10)
        weights = torch.softmax(logits, dim=0)
        
        merged_correlation = torch.einsum('h...,h->...', correlations, weights)
        merged_rank = torch.stack([p.effective_rank() for p in pseudopodia]).mean()
        merged_coherence = coherences.min()
        
        return merged_correlation, merged_rank, merged_coherence
    
    def attend(self, pseudopodia, temperature):
        """Compute attention with metabolic temperature modulation via Triton kernel."""
        batch_size = pseudopodia[0].key.shape[0]
        latent_dim = self.n_pseudopodia * self.head_dim
        
        keys = torch.cat([p.key for p in pseudopodia], dim=-1).contiguous()
        values = torch.cat([p.value for p in pseudopodia], dim=-1).contiguous()
        queries = torch.cat([p.query for p in pseudopodia], dim=-1).contiguous()
        
        output = torch.empty_like(queries)
        
        BLOCK_SIZE = 32
        grid = (batch_size,)
        
        attention_kernel[grid](
            queries, keys, values, output,
            batch_size, latent_dim, temperature.item(),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2
        )
        
        return output
    
    def grow(self, stimulus, body, tubes, food, flow_state, step):
        """Single growth step: extend, merge, flow, adapt."""
        temperature = self.metabolic_rate(step)
        
        # Extend pseudopodia to probe environment
        pseudopodia = self.extend_pseudopodia(body, stimulus)
        correlation, rank, coherence = self.merge_pseudopodia(pseudopodia, temperature)
        
        # Flow correlation through tube network
        tubes.flow(correlation, temperature)
        
        # Collapse wavefunction to adjust dimensionality
        flow_state.dimensionality = self._collapse_wavefunction(
            flow_state.dimensionality, rank, body.device
        )
        
        # Compute attended nutrient
        nutrient = self.attend(pseudopodia, temperature)
        flow_field = tubes.potential_field()
        
        # Absorb nutrients modulated by flow field
        if flow_field is not None:
            flow_influence = flow_field.mean(0).flatten()
            min_dim = min(body.shape[-1], flow_influence.shape[0], nutrient.shape[-1])
            body = body.clone()
            body[:, :min_dim] = body[:, :min_dim] + 0.1 * flow_influence[:min_dim].unsqueeze(0) * nutrient[:, :min_dim]
        
        # Information-theoretic fitness
        mi = flow_state.mutual_information_with(tubes.tubes)
        uncertainty = flow_state.entropy()
        fitness = coherence * rank / (1 + uncertainty)
        
        # Behavioral coordinates
        behavior = (
            torch.clamp(rank, 0, 1).item(),
            torch.clamp(coherence, 0, 1).item()
        )
        
        # Deposit and forage nutrients
        food.deposit(nutrient, behavior, fitness)
        foraged_nutrient = food.forage(temperature, uncertainty)
        
        # Update body with exponential moving average
        body = 0.9 * body + 0.1 * (foraged_nutrient if foraged_nutrient is not None else nutrient)
        
        return FlowState(
            body,
            tubes.tubes.copy(),
            food.sources.copy(),
            flow_state.dimensionality
        ), {
            'reconstruction': self.decode(body),
            'predicted_coherence': self.predict_coherence(body),
            'predicted_rank': self.predict_rank(body),
            'mutual_information': mi,
            'uncertainty': uncertainty,
            'fitness': fitness,
            'temperature': temperature,
            'rank': rank,
            'coherence': coherence
        }
    
    def _collapse_wavefunction(self, state, basis_dim, device):
        """Quantum-inspired dimensionality adaptation based on effective rank."""
        target_dim = int(torch.clamp(basis_dim, 1, 1000).item())
        target_dim = (target_dim // 32) * 32  # Align to 32 for GPU efficiency
        target_dim = max(32, target_dim)  # Minimum dimension
        
        if state is None:
            state = torch.randn(target_dim, device=device)
            return torch.nn.functional.normalize(state, p=2, dim=0)
        
        if target_dim > state.shape[0]:
            # Expand state space
            state = torch.cat([state, torch.randn(target_dim - state.shape[0], device=device)])
        elif target_dim < state.shape[0]:
            # Contract state space
            state = state[:target_dim]
        
        return torch.nn.functional.normalize(state, p=2, dim=0)
    
    def loss(self, stimulus, diagnostics):
        """Multi-objective loss: reconstruction + prediction + exploration."""
        reconstruction_loss = torch.nn.functional.mse_loss(
            diagnostics['reconstruction'], stimulus
        )
        
        coherence_loss = torch.nn.functional.mse_loss(
            diagnostics['predicted_coherence'],
            diagnostics['coherence'].detach().unsqueeze(0).unsqueeze(0)
        )
        
        rank_loss = torch.nn.functional.mse_loss(
            diagnostics['predicted_rank'],
            diagnostics['rank'].detach().unsqueeze(0).unsqueeze(0)
        )
        
        # Exploration bonus: high MI, low uncertainty
        exploration_bonus = diagnostics['mutual_information'] / (1 + diagnostics['uncertainty'])
        
        total_loss = reconstruction_loss + 0.1 * coherence_loss + 0.1 * rank_loss - 0.01 * exploration_bonus
        
        return total_loss, {
            'reconstruction': reconstruction_loss.item(),
            'coherence_prediction': coherence_loss.item(),
            'rank_prediction': rank_loss.item(),
            'exploration': exploration_bonus.item()
        }
    
    def train_step(self, stimulus, optimizer, n_steps=10):
        """Single training step with multiple growth iterations."""
        body = self.encode(stimulus)
        tubes = TubeNetwork()
        food = FoodSource()
        flow_state = FlowState(body, deque(), {}, None)
        
        total_losses = {
            'reconstruction': 0,
            'coherence_prediction': 0,
            'rank_prediction': 0,
            'exploration': 0
        }
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            flow_state, diagnostics = self.grow(
                stimulus, flow_state.pseudopodia, tubes, food, flow_state, step
            )
            loss, loss_components = self.loss(stimulus, diagnostics)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            
            for k, v in loss_components.items():
                total_losses[k] += v
            
            # Detach for next iteration
            flow_state = FlowState(
                flow_state.pseudopodia.detach(),
                deque(),
                {},
                flow_state.dimensionality.detach() if flow_state.dimensionality is not None else None
            )
        
        return {k: v / n_steps for k, v in total_losses.items()}
    
    @torch.no_grad()
    def explore(self, stimulus, convergence_threshold=1e-4, max_steps=1000):
        """Explore information landscape until convergence via MI gradient."""
        body = self.encode(stimulus)
        tubes = TubeNetwork()
        food = FoodSource()
        flow_state = FlowState(body, deque(), {}, None)
        mi_history = deque(maxlen=100)
        
        for step in range(max_steps):
            flow_state, aux = self.grow(
                stimulus, flow_state.pseudopodia, tubes, food, flow_state, step
            )
            
            mi = aux['mutual_information']
            mi_history.append(mi.item() if torch.is_tensor(mi) else mi)
            
            # Convergence detection via MI gradient
            if len(mi_history) >= 20:
                recent_mi = sum(list(mi_history)[-10:]) / 10
                old_mi = sum(list(mi_history)[-20:-10]) / 10
                
                if abs(recent_mi - old_mi) < convergence_threshold:
                    break
            
            yield {
                'body': flow_state.pseudopodia,
                'network': tubes,
                'food_sources': food,
                'flow': flow_state,
                'diagnostics': aux,
                'step': step
            }


# Mixed precision training support
def create_model_with_amp(sensory_dim, latent_dim, n_pseudopodia=4, device='cuda'):
    """Create model with automatic mixed precision support."""
    model = Plasmodium(sensory_dim, latent_dim, n_pseudopodia).to(device)
    scaler = torch.cuda.amp.GradScaler()
    return model, scaler


def train_step_amp(model, stimulus, optimizer, scaler, n_steps=10):
    """Training step with automatic mixed precision."""
    with torch.cuda.amp.autocast():
        body = model.encode(stimulus)
        tubes = TubeNetwork()
        food = FoodSource()
        flow_state = FlowState(body, deque(), {}, None)
        
        total_losses = {
            'reconstruction': 0,
            'coherence_prediction': 0,
            'rank_prediction': 0,
            'exploration': 0
        }
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            flow_state, diagnostics = model.grow(
                stimulus, flow_state.pseudopodia, tubes, food, flow_state, step
            )
            loss, loss_components = model.loss(stimulus, diagnostics)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            for k, v in loss_components.items():
                total_losses[k] += v
            
            flow_state = FlowState(
                flow_state.pseudopodia.detach(),
                deque(),
                {},
                flow_state.dimensionality.detach() if flow_state.dimensionality is not None else None
            )
    
    return {k: v / n_steps for k, v in total_losses.items()}
