"""Gradient-based fitness computation for component lifecycle"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FitnessComputer:
    """Compute fitness metrics based on gradient magnitudes AND compute efficiency.

    Critical insight (Decision #10): Fitness must include efficiency signals.

    Without efficiency:
    - Slow components survive if they help task accuracy
    - No evolutionary pressure for hardware-friendly patterns
    - Diversity without utility

    With efficiency:
    - Cross-GPU communication → slower → lower fitness → culled
    - Poor cache behavior → slower → lower fitness → culled
    - Hardware-optimal patterns emerge from selection pressure

    Fitness = task_performance * 0.7 + compute_efficiency * 0.2 + gradient_magnitude * 0.1
    """

    def __init__(self, ema_alpha: float = 0.1, include_efficiency: bool = True):
        """Initialize fitness computer.

        Args:
            ema_alpha: Exponential moving average smoothing factor
            include_efficiency: Include compute efficiency in fitness (Decision #10)
        """
        self.ema_alpha = ema_alpha
        self.include_efficiency = include_efficiency
        self._fitness_history: Dict[int, List[float]] = {}
        self._latency_history: Dict[int, List[float]] = {}

    def compute_gradient_based_fitness(
        self,
        component: nn.Module,
        component_id: int,
    ) -> float:
        """Compute fitness from gradient magnitude.

        Args:
            component: Component to compute fitness for
            component_id: Unique component identifier

        Returns:
            Fitness score in [0, inf), higher is better
        """
        grad_norms = []

        for name, param in component.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                grad_norms.append(grad_norm)

        if not grad_norms:
            return 0.0

        mean_grad_norm = sum(grad_norms) / len(grad_norms)
        max_grad_norm = max(grad_norms)

        raw_fitness = (mean_grad_norm + max_grad_norm) / 2.0

        if component_id not in self._fitness_history:
            self._fitness_history[component_id] = []

        self._fitness_history[component_id].append(raw_fitness)

        if len(self._fitness_history[component_id]) == 1:
            smoothed_fitness = raw_fitness
        else:
            prev_fitness = self._fitness_history[component_id][-2]
            smoothed_fitness = (
                self.ema_alpha * raw_fitness +
                (1 - self.ema_alpha) * prev_fitness
            )

        return smoothed_fitness

    def compute_attention_alignment_fitness(
        self,
        component: nn.Module,
        attention_weights: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> float:
        """Compute fitness from attention alignment with targets.

        Args:
            component: Component to compute fitness for
            attention_weights: Attention weights [batch, seq, seq]
            target_mask: Binary mask of target positions [batch, seq]

        Returns:
            Fitness score based on attention to targets
        """
        if attention_weights.numel() == 0 or target_mask.numel() == 0:
            return 0.0

        target_mask = target_mask.unsqueeze(1).expand_as(attention_weights)

        attention_to_targets = (attention_weights * target_mask).sum() / (target_mask.sum() + 1e-10)

        return attention_to_targets.item()

    def compute_information_bottleneck_fitness(
        self,
        component: nn.Module,
        input_activations: torch.Tensor,
        output_activations: torch.Tensor,
    ) -> float:
        """Compute fitness from information bottleneck metrics.

        Args:
            component: Component to compute fitness for
            input_activations: Input activations
            output_activations: Output activations

        Returns:
            Fitness based on mutual information
        """
        if input_activations.numel() == 0 or output_activations.numel() == 0:
            return 0.0

        input_entropy = self._compute_entropy(input_activations)
        output_entropy = self._compute_entropy(output_activations)

        mutual_info = min(input_entropy, output_entropy)

        return mutual_info

    def _compute_entropy(self, activations: torch.Tensor) -> float:
        """Compute entropy of activations"""
        activations_flat = activations.flatten()

        if activations_flat.numel() == 0:
            return 0.0

        probs = torch.softmax(activations_flat, dim=0)

        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        return entropy.item()

    def compute_compute_efficiency(
        self,
        component: nn.Module,
        component_id: int,
        forward_time_ms: float,
        baseline_time_ms: float = 1.0,
    ) -> float:
        """Compute hardware efficiency metric.

        Faster components get higher fitness.
        This creates evolutionary pressure for:
        - Cache-friendly memory access patterns
        - Sparse computations
        - Efficient attention mechanisms

        Args:
            component: Component to evaluate
            component_id: Component identifier
            forward_time_ms: Measured forward pass time in milliseconds
            baseline_time_ms: Baseline time for normalization

        Returns:
            Efficiency score in [0, 1], higher is better
        """
        if component_id not in self._latency_history:
            self._latency_history[component_id] = []

        self._latency_history[component_id].append(forward_time_ms)

        # EMA of latency
        if len(self._latency_history[component_id]) == 1:
            smoothed_latency = forward_time_ms
        else:
            prev_latency = self._latency_history[component_id][-2]
            smoothed_latency = (
                self.ema_alpha * forward_time_ms +
                (1 - self.ema_alpha) * prev_latency
            )

        # Convert latency to efficiency: faster = higher score
        efficiency = baseline_time_ms / (smoothed_latency + 1e-6)
        return min(efficiency, 1.0)  # Cap at 1.0

    def compute_combined_fitness(
        self,
        component: nn.Module,
        component_id: int,
        gradient_weight: float = 0.1,
        attention_weights: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        attention_weight: float = 0.7,
        forward_time_ms: Optional[float] = None,
        efficiency_weight: float = 0.2,
        input_activations: Optional[torch.Tensor] = None,
        output_activations: Optional[torch.Tensor] = None,
        information_weight: float = 0.0,
    ) -> float:
        """Compute weighted combination of fitness metrics.

        Default weights follow Decision #10:
        - attention_weight: 0.7 (task performance)
        - efficiency_weight: 0.2 (compute efficiency)
        - gradient_weight: 0.1 (gradient magnitude)

        Args:
            component: Component to compute fitness for
            component_id: Component identifier
            gradient_weight: Weight for gradient-based fitness
            attention_weights: Optional attention weights
            target_mask: Optional target mask
            attention_weight: Weight for attention fitness (task performance)
            forward_time_ms: Optional forward pass time for efficiency
            efficiency_weight: Weight for compute efficiency
            input_activations: Optional input activations
            output_activations: Optional output activations
            information_weight: Weight for information fitness

        Returns:
            Combined fitness score
        """
        total_weight = gradient_weight + attention_weight + efficiency_weight + information_weight

        if total_weight == 0:
            return 0.0

        fitness = 0.0

        # Task performance (highest weight)
        if attention_weight > 0 and attention_weights is not None and target_mask is not None:
            attn_fitness = self.compute_attention_alignment_fitness(
                component, attention_weights, target_mask
            )
            fitness += attention_weight * attn_fitness

        # Compute efficiency (hardware awareness)
        if efficiency_weight > 0 and forward_time_ms is not None and self.include_efficiency:
            eff_fitness = self.compute_compute_efficiency(
                component, component_id, forward_time_ms
            )
            fitness += efficiency_weight * eff_fitness

        # Gradient magnitude (relevance)
        if gradient_weight > 0:
            grad_fitness = self.compute_gradient_based_fitness(component, component_id)
            fitness += gradient_weight * grad_fitness

        # Information bottleneck (optional)
        if information_weight > 0 and input_activations is not None and output_activations is not None:
            info_fitness = self.compute_information_bottleneck_fitness(
                component, input_activations, output_activations
            )
            fitness += information_weight * info_fitness

        return fitness / total_weight

    def get_fitness_history(self, component_id: int) -> List[float]:
        """Get fitness history for component"""
        return self._fitness_history.get(component_id, [])

    def clear_history(self, component_id: Optional[int] = None) -> None:
        """Clear fitness history"""
        if component_id is None:
            self._fitness_history.clear()
        elif component_id in self._fitness_history:
            del self._fitness_history[component_id]

    def stats(self) -> Dict:
        """Get statistics about fitness computation"""
        if not self._fitness_history:
            return {
                'num_components': 0,
                'avg_fitness': 0.0,
                'max_fitness': 0.0,
                'min_fitness': 0.0,
            }

        all_fitness = [
            f for history in self._fitness_history.values()
            for f in history
        ]

        return {
            'num_components': len(self._fitness_history),
            'total_computations': len(all_fitness),
            'avg_fitness': sum(all_fitness) / len(all_fitness),
            'max_fitness': max(all_fitness),
            'min_fitness': min(all_fitness),
        }
