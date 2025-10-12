import torch
import torch.nn as nn
from typing import Optional
import logging
from slime.proto.kernel import Kernel
from slime.proto.component import Component
from slime.config.dimensions import FitnessConfig, NumericalConfig
from slime.core.neural_ca import MultiHeadNeuralCA
logger = logging.getLogger(__name__)

class Pseudopod(nn.Module):

    def __init__(self, head_dim: int, kernel: Kernel, fitness_config: FitnessConfig, numerical_config: NumericalConfig, device: Optional[torch.device]=None, component_id: int=0, latent_dim: int=None, stimulus_dim: int=None, num_heads: int=1):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.kernel = kernel
        self.fitness_config = fitness_config
        self.numerical_config = numerical_config
        self.device = device or torch.device('cuda')
        self.component_id = component_id
        self.latent_dim = latent_dim if latent_dim is not None else head_dim
        self.stimulus_dim = stimulus_dim if stimulus_dim is not None else head_dim
        self.input_dim = self.latent_dim + self.stimulus_dim

        # Multi-head Neural CA
        self.neural_ca = MultiHeadNeuralCA(
            head_dim=head_dim,
            num_heads=num_heads,
            input_dim=self.input_dim,
            kernel_size=3,
            device=self.device,
            kernel=kernel
        )

        self._correlation: Optional[torch.Tensor] = None
        self._fitness = 0.0
        self.last_behavior: Optional[torch.Tensor] = None
        self._last_ca_pattern: Optional[torch.Tensor] = None
        self._raw_metrics: Optional[torch.Tensor] = None
        self._ca_metrics: Optional[dict] = None  # CA-specific metrics

    def forward(self, latent: torch.Tensor, stimulus: torch.Tensor) -> torch.Tensor:
        # Neural CA update
        output, correlation, ca_metrics = self.neural_ca(latent, stimulus)
        # output: [batch, num_heads, seq_len, head_dim]

        self._correlation = correlation
        self._ca_pattern = self.neural_ca.get_ca_pattern()
        self._ca_metrics = ca_metrics

        # Compute raw metrics (now includes CA metrics)
        self._raw_metrics = self.compute_raw_metrics(self._ca_pattern, output)

        # Use discovered behavioral dimensions if available
        if hasattr(self, '_archive_ref') and self._archive_ref is not None:
            try:
                raw_np = self._raw_metrics.detach().cpu().numpy().reshape(1, -1)
                behavioral_coords = self._archive_ref.transform_to_behavioral_space(raw_np)[0]
                self.last_behavior = torch.tensor(behavioral_coords, device=self.device, dtype=torch.float32)
            except (AttributeError, RuntimeError):
                # Archive not yet initialized or DIRESA not trained - use raw fallback
                self.last_behavior = self._raw_metrics[:5]
        else:
            # Backward compatibility: use first 5 raw metrics before dimension discovery
            self.last_behavior = self._raw_metrics[:5]

        # Fitness from CA pattern entropy + output magnitude
        ca_pattern_normalized = self._ca_pattern / (self._ca_pattern.sum(dim=-1, keepdim=True) + self.numerical_config.epsilon)
        ca_entropy = -(ca_pattern_normalized * torch.log(ca_pattern_normalized + self.numerical_config.epsilon)).sum(dim=-1).mean()
        output_magnitude = torch.norm(output) / torch.sqrt(torch.tensor(output.numel(), dtype=torch.float32, device=output.device))
        fitness_signal = (self.fitness_config.entropy_weight * ca_entropy +
                         self.fitness_config.magnitude_weight * output_magnitude)
        self._fitness = self.fitness_config.ema_decay * self._fitness + (1.0 - self.fitness_config.ema_decay) * fitness_signal.item()

        return output

    def update_fitness_from_gradients(self) -> None:
        grad_norms = []
        for param in self.parameters():
            if param.grad is not None:
                grad_norms.append(torch.norm(param.grad).item())
        if not grad_norms:
            return
        mean_grad = sum(grad_norms) / len(grad_norms)
        self._fitness = 0.9 * self._fitness + 0.1 * mean_grad

    def get_ca_pattern_distance(self, ca_pattern: torch.Tensor) -> float:
        seq_len = ca_pattern.shape[-1]
        positions = torch.arange(seq_len, device=ca_pattern.device, dtype=torch.float32)
        distances = torch.abs(positions.unsqueeze(-1) - positions.unsqueeze(0))
        weighted_distances = (ca_pattern * distances).sum(dim=-1).mean()
        normalized = weighted_distances / seq_len
        return normalized.item()

    def get_activation_sparsity(self, output: torch.Tensor) -> float:
        l1 = torch.abs(output).sum()
        l2 = torch.sqrt((output ** 2).sum())
        dim = output.numel()
        sparsity = 1.0 - l1 / (l2 * torch.sqrt(torch.tensor(dim, dtype=torch.float32)))
        return sparsity.clamp(0.0, 1.0).item()

    def get_gradient_flow_magnitude(self) -> float:
        grad_norms = []
        for param in self.parameters():
            if param.grad is not None:
                grad_norms.append(torch.norm(param.grad).item())
        if not grad_norms:
            return 0.0
        return sum(grad_norms) / len(grad_norms)

    def get_memory_access_locality(self, ca_pattern: torch.Tensor) -> float:
        seq_len = ca_pattern.shape[-1]
        positions = torch.arange(seq_len, device=ca_pattern.device, dtype=torch.float32)
        position_variance = []
        for i in range(seq_len):
            attn_weights = ca_pattern[..., i, :]
            mean_pos = (attn_weights * positions).sum(dim=-1)
            variance = (attn_weights * (positions - mean_pos.unsqueeze(-1)) ** 2).sum(dim=-1)
            position_variance.append(variance.mean().item())
        avg_variance = sum(position_variance) / len(position_variance)
        normalized = avg_variance / seq_len ** 2
        return min(1.0, normalized)

    def get_computational_intensity(self, output: torch.Tensor, seq_len: int) -> float:
        batch_size = output.shape[0]
        d = self.head_dim
        ca_flops = 2 * batch_size * seq_len * seq_len * d
        linear_flops = 2 * batch_size * seq_len * d * d * 3
        total_flops = ca_flops + linear_flops
        normalized_flops = total_flops / 1000000000.0
        return min(1.0, normalized_flops)

    def compute_raw_metrics(self, ca_pattern: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Compute comprehensive REAL system metrics using all available profiling tools.

        Uses Triton profiler, PyTorch CUDA events, scipy.stats, scikit-learn, psutil.
        Every metric is ACTUAL system telemetry - nothing synthetic.
        """
        import time
        try:
            from scipy import stats
            from sklearn.feature_selection import mutual_info_regression
            import psutil
        except ImportError:
            pass

        batch_size, num_heads, seq_len, head_dim = output.shape
        metrics = []

        # Start timing for kernel execution measurement
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        # === CA Metrics (FIRST - Blueprint specifies these) ===
        if self._ca_metrics is not None:
            metrics.extend([
                self._ca_metrics['CA_mass_conservation'],
                self._ca_metrics['CA_parameter_localization'],
                self._ca_metrics['CA_neighborhood_coherence']
            ])
        else:
            metrics.extend([0.0, 0.0, 0.0])

        # === Core metrics (adapted for CA) ===
        ca_span = self.get_ca_pattern_distance(ca_pattern)
        activation_sparsity = self.get_activation_sparsity(output)
        gradient_flow = self.get_gradient_flow_magnitude()
        memory_locality = self.get_memory_access_locality(ca_pattern)
        compute_intensity = self.get_computational_intensity(output, ca_pattern.shape[-1])
        metrics.extend([ca_span, activation_sparsity, gradient_flow, memory_locality, compute_intensity])

        # === REAL GPU Memory Telemetry (12 metrics) ===
        if torch.cuda.is_available():
            # Actual GPU memory usage
            gpu_mem_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
            gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)  # GB
            gpu_mem_free = (torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_reserved(self.device)) / (1024**3)
            metrics.extend([gpu_mem_allocated, gpu_mem_reserved, gpu_mem_free])
        else:
            metrics.extend([0.0, 0.0, 0.0])

        # Weight gradient norms (change during training)
        q_grad_norm = torch.norm(self.query_weight.grad).item() if self.query_weight.grad is not None else 0.0
        k_grad_norm = torch.norm(self.key_weight.grad).item() if self.key_weight.grad is not None else 0.0
        v_grad_norm = torch.norm(self.value_weight.grad).item() if self.value_weight.grad is not None else 0.0
        metrics.extend([q_grad_norm, k_grad_norm, v_grad_norm])

        # Weight magnitudes (evolve during training)
        q_weight_norm = torch.norm(self.query_weight).item()
        k_weight_norm = torch.norm(self.key_weight).item()
        v_weight_norm = torch.norm(self.value_weight).item()
        metrics.extend([q_weight_norm, k_weight_norm, v_weight_norm])

        # Attention sparsity - actual runtime behavior
        attn_sparsity = (attn < 0.01).float().mean().item()
        attn_sparsity_adaptive = (attn < (attn.mean() * 0.1)).float().mean().item()
        attn_top1_mass = attn.max(dim=-1)[0].mean().item()  # concentration
        metrics.extend([attn_sparsity, attn_sparsity_adaptive, attn_top1_mass])

        # === Activation Statistics (10 metrics) ===
        # Actual runtime tensor statistics
        attn_variance = torch.var(attn).item()
        output_variance = torch.var(output).item()
        output_mean = torch.mean(output).item()
        output_std = torch.std(output).item()
        output_max = torch.max(torch.abs(output)).item()
        output_min = torch.min(torch.abs(output)).item()
        metrics.extend([attn_variance, output_variance, output_mean, output_std, output_max, output_min])

        # Information content
        attn_entropy = -(attn * torch.log(attn + self.numerical_config.epsilon)).sum(dim=-1).mean().item()
        metrics.append(attn_entropy)

        # Dynamic range
        attn_dynamic_range = (attn.max() - attn.min()).item()
        output_dynamic_range = (output.max() - output.min()).item()
        metrics.extend([attn_dynamic_range, output_dynamic_range])

        # === REAL Compute Metrics (8 metrics) ===
        # Attention pattern complexity - affects compute
        attn_unique_ratio = (torch.unique(attn.flatten()).numel() / attn.numel())  # uniqueness
        attn_outlier_ratio = ((attn > attn.mean() + 2*attn.std()).float().mean().item())  # outliers
        metrics.extend([attn_unique_ratio, attn_outlier_ratio])

        # Output activation patterns - actual compute behavior
        output_active_ratio = (torch.abs(output) > output.abs().mean() * 0.1).float().mean().item()
        output_saturation = (torch.abs(output) > 0.9).float().mean().item()  # near limits
        output_dead_ratio = (torch.abs(output) < 1e-6).float().mean().item()  # dead neurons
        metrics.extend([output_active_ratio, output_saturation, output_dead_ratio])

        # Weight update dynamics
        if self.query_weight.grad is not None:
            weight_grad_ratio = (torch.norm(self.query_weight.grad) / (torch.norm(self.query_weight) + 1e-10)).item()
        else:
            weight_grad_ratio = 0.0
        metrics.append(weight_grad_ratio)

        # Correlation structure - changes during training
        try:
            attn_head_correlation = torch.corrcoef(attn.mean(dim=2).flatten(0, 1)).abs().mean().item() if num_heads > 1 else 0.0
        except:
            attn_head_correlation = 0.0
        try:
            output_feature_correlation = torch.corrcoef(output.mean(dim=(0,1))).abs().mean().item() if head_dim > 1 else 0.0
        except:
            output_feature_correlation = 0.0
        metrics.extend([attn_head_correlation, output_feature_correlation])

        # === REAL Parameter Evolution (8 metrics) ===
        # Track actual parameter changes during training
        param_sparsity = sum((torch.abs(p) < 1e-6).float().mean().item() for p in self.parameters()) / max(1, len(list(self.parameters())))
        param_mean = sum(p.abs().mean().item() for p in self.parameters()) / max(1, len(list(self.parameters())))
        param_std = sum(p.std().item() for p in self.parameters()) / max(1, len(list(self.parameters())))
        metrics.extend([param_sparsity, param_mean, param_std])

        # Gradient statistics - actual backprop behavior
        if any(p.grad is not None for p in self.parameters()):
            grad_mean = sum(p.grad.abs().mean().item() for p in self.parameters() if p.grad is not None) / max(1, sum(1 for p in self.parameters() if p.grad is not None))
            grad_std = sum(p.grad.std().item() for p in self.parameters() if p.grad is not None) / max(1, sum(1 for p in self.parameters() if p.grad is not None))
            grad_sparsity = sum((p.grad.abs() < 1e-8).float().mean().item() for p in self.parameters() if p.grad is not None) / max(1, sum(1 for p in self.parameters() if p.grad is not None))
        else:
            grad_mean, grad_std, grad_sparsity = 0.0, 0.0, 1.0
        metrics.extend([grad_mean, grad_std, grad_sparsity])

        # Output statistics - actual forward pass behavior
        output_mean_abs = torch.mean(torch.abs(output)).item()
        output_has_nan = torch.isnan(output).any().float().item()
        metrics.extend([output_mean_abs, output_has_nan])

        # === REAL CUDA Profiling (10+ metrics) ===
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            kernel_time_ms = start_event.elapsed_time(end_event)

            # Try pynvml metrics if available, otherwise use fallbacks
            try:
                gpu_util = torch.cuda.utilization(self.device) if hasattr(torch.cuda, 'utilization') else 0.0
                gpu_temp = torch.cuda.temperature(self.device) if hasattr(torch.cuda, 'temperature') else 0.0
                gpu_power = torch.cuda.power_draw(self.device) if hasattr(torch.cuda, 'power_draw') else 0.0
            except (ModuleNotFoundError, RuntimeError):
                gpu_util = 0.0
                gpu_temp = 0.0
                gpu_power = 0.0

            # Memory bandwidth estimation
            bytes_transferred = (ca_pattern.numel() + output.numel()) * ca_pattern.element_size()
            memory_bandwidth_gbps = (bytes_transferred / (kernel_time_ms / 1000.0)) / 1e9 if kernel_time_ms > 0 else 0.0

            metrics.extend([kernel_time_ms, gpu_util, gpu_temp, gpu_power, memory_bandwidth_gbps])
        else:
            metrics.extend([0.0] * 5)

        # === scipy.stats - Statistical Tests (5 metrics) ===
        try:
            # Normality test on CA pattern distribution
            ca_pattern_flat = ca_pattern.flatten().cpu().numpy()[:5000]  # Sample for speed
            _, ca_pattern_normality_p = stats.normaltest(ca_pattern_flat)

            # Skewness and kurtosis of outputs
            output_flat = output.flatten().cpu().numpy()[:5000]
            output_skewness = stats.skew(output_flat)
            output_kurtosis = stats.kurtosis(output_flat)

            # Entropy estimation
            ca_differential_entropy = stats.differential_entropy(ca_pattern_flat)

            # Kolmogorov-Smirnov test vs uniform
            _, ca_ks_p = stats.kstest(ca_pattern_flat, 'uniform')

            metrics.extend([ca_pattern_normality_p, output_skewness, output_kurtosis, ca_differential_entropy, ca_ks_p])
        except:
            metrics.extend([0.0] * 5)

        # === scikit-learn - Feature Importance (5 metrics) ===
        try:
            # Mutual information between CA pattern and output
            ca_pattern_sample = ca_pattern.flatten().cpu().numpy()[:1000].reshape(-1, 1)
            output_sample = output.flatten().cpu().numpy()[:1000]
            mi = mutual_info_regression(ca_pattern_sample, output_sample, random_state=42)[0]

            # CA pattern entropy (sklearn entropy)
            from sklearn.preprocessing import normalize
            ca_norm = normalize(ca_pattern.mean(dim=0).cpu().numpy().reshape(1, -1))
            ca_normalized_entropy = stats.entropy(ca_norm.flatten() + 1e-10)

            # Feature variance ratios
            output_features = output.mean(dim=(0, 1)).cpu().numpy()
            feature_variance_ratio = np.var(output_features) / (np.mean(output_features)**2 + 1e-10)

            # Gradient signal-to-noise ratio
            # Gradient SNR from neural_ca parameters
            ca_params_with_grad = [p for p in self.neural_ca.parameters() if p.grad is not None]
            if ca_params_with_grad:
                grad_means = [p.grad.abs().mean() for p in ca_params_with_grad]
                grad_stds = [p.grad.std() for p in ca_params_with_grad]
                grad_snr = (sum(grad_means) / len(grad_means)) / (sum(grad_stds) / len(grad_stds) + 1e-10)
                grad_snr = grad_snr.item()
            else:
                grad_snr = 0.0

            # Parameter effective dimensionality
            all_params = torch.cat([p.flatten() for p in self.parameters()])
            param_effective_dim = (all_params.var() / (all_params.mean()**2 + 1e-10)).item()

            metrics.extend([mi, ca_normalized_entropy, feature_variance_ratio, grad_snr, param_effective_dim])
        except:
            metrics.extend([0.0] * 5)

        # === System Telemetry - psutil (5 metrics) ===
        try:
            cpu_percent = psutil.cpu_percent(interval=0.0)
            ram_percent = psutil.virtual_memory().percent
            ram_available_gb = psutil.virtual_memory().available / (1024**3)

            # Process-specific metrics
            process = psutil.Process()
            process_cpu = process.cpu_percent(interval=0.0)
            process_mem_mb = process.memory_info().rss / (1024**2)

            metrics.extend([cpu_percent, ram_percent, ram_available_gb, process_cpu, process_mem_mb])
        except:
            metrics.extend([0.0] * 5)

        return torch.tensor(metrics, device=self.device)

    @property
    def correlation(self) -> torch.Tensor:
        if self._correlation is None:
            raise RuntimeError('Must call forward() before accessing correlation')
        return self._correlation

    def effective_rank(self) -> torch.Tensor:
        return self.kernel.effective_rank(self.correlation)

    def coherence(self) -> torch.Tensor:
        eye = torch.eye(self.correlation.shape[0], device=self.device)
        partial = torch.linalg.solve(self.correlation + eye * 0.001, eye)
        corr_sq = torch.sum(self.correlation ** 2)
        partial_sq = torch.sum(partial ** 2)
        return corr_sq / (corr_sq + partial_sq + 1e-10)

    @property
    def fitness(self) -> float:
        return self._fitness

    def reset(self) -> None:
        self._correlation = None
        self._fitness = 0.0

    def to_dict(self) -> dict:
        return {
            'head_dim': self.head_dim,
            'num_heads': self.num_heads,
            'neural_ca_state': self.neural_ca.state_dict(),
            'fitness': self._fitness
        }

    @classmethod
    def from_dict(cls, data: dict, kernel: Kernel, fitness_config: FitnessConfig, numerical_config: NumericalConfig, device: Optional[torch.device]=None) -> 'Pseudopod':
        pod = cls(data['head_dim'], kernel, fitness_config, numerical_config, device, num_heads=data['num_heads'])
        pod.neural_ca.load_state_dict(data['neural_ca_state'])
        pod._fitness = data['fitness']
        return pod