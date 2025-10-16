"""
Rigorous statistical geometry for model comparison.

Implements:
1. Fisher-Rao metric on model manifold (information geometry)
2. AICc with Jacobian correction for change of basis (coordinate-free comparison)
3. Semantic JSON distillation from single run (actionable insights)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import logging
from slime.config.dimensions import StatisticalGeometryConfig, NumericalConfig

logger = logging.getLogger(__name__)


@dataclass
class GeometricComparison:
    """
    Geometric comparison between two models on statistical manifold.

    Uses Fisher information metric (Riemannian metric on probability distributions)
    to measure distance between models in a coordinate-free way.
    """
    model_a_name: str
    model_b_name: str

    # Fisher-Rao distance (geodesic on statistical manifold)
    fisher_rao_distance: float

    # Relative Fisher information (directional divergence)
    relative_info_a_to_b: float  # How much info does A have that B lacks?
    relative_info_b_to_a: float  # How much info does B have that A lacks?

    # Model complexity (AICc with Jacobian correction)
    aicc_a: float
    aicc_b: float
    aicc_delta: float  # Positive = A is worse (more complex for same fit)

    # Semantic interpretation
    interpretation: str


class StatisticalGeometry:
    """
    Compute geometric properties of models on statistical manifold.

    Fisher information matrix defines Riemannian metric:
        g_ij(θ) = E[∂log p(x|θ)/∂θ_i * ∂log p(x|θ)/∂θ_j]

    Fisher-Rao distance is geodesic distance on this manifold.
    """

    def __init__(self, device: torch.device, config: StatisticalGeometryConfig, numerical_config: NumericalConfig):
        self.device = device
        self.config = config
        self.numerical_config = numerical_config

    def compare_models(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> GeometricComparison:
        """
        Compute geometric comparison between two models.

        Args:
            model_a, model_b: Models to compare
            data_loader: Data for computing Fisher information
            model_a_name, model_b_name: Names for reporting

        Returns:
            GeometricComparison with Fisher-Rao distance and AICc
        """
        logger.info(f"Computing geometric comparison: {model_a_name} vs {model_b_name}")

        # Compute Fisher information matrices
        fisher_a = self._compute_fisher_information(model_a, data_loader)
        fisher_b = self._compute_fisher_information(model_b, data_loader)

        # Fisher-Rao distance (geodesic on statistical manifold)
        fr_distance = self._fisher_rao_distance(fisher_a, fisher_b)

        # Relative information (directional divergence)
        rel_info_a_to_b = self._relative_information(fisher_a, fisher_b)
        rel_info_b_to_a = self._relative_information(fisher_b, fisher_a)

        # AICc with Jacobian correction
        aicc_a = self._aicc_with_jacobian(model_a, data_loader, fisher_a)
        aicc_b = self._aicc_with_jacobian(model_b, data_loader, fisher_b)
        aicc_delta = aicc_a - aicc_b

        # Semantic interpretation
        interpretation = self._interpret_comparison(
            fr_distance, rel_info_a_to_b, rel_info_b_to_a, aicc_delta
        )

        return GeometricComparison(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            fisher_rao_distance=fr_distance,
            relative_info_a_to_b=rel_info_a_to_b,
            relative_info_b_to_a=rel_info_b_to_a,
            aicc_a=aicc_a,
            aicc_b=aicc_b,
            aicc_delta=aicc_delta,
            interpretation=interpretation
        )

    def _compute_fisher_information(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """
        Compute empirical Fisher information matrix.

        Fisher information: I(θ) = E[∇log p(x|θ) * ∇log p(x|θ)^T]

        Empirical estimate: I ≈ (1/N) Σ g_i * g_i^T where g_i = ∇log p(x_i|θ)

        Returns:
            Fisher information matrix (flattened parameter space)
        """
        model.eval()

        # Collect gradients
        gradients = []

        samples_collected = 0
        for inputs, targets in data_loader:
            if samples_collected >= self.config.fisher_num_samples:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Log probability
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
            log_prob = log_probs[range(len(targets)), targets].sum()

            # Gradient of log probability
            model.zero_grad()
            log_prob.backward()

            # Collect gradient vector (flattened)
            grad_vec = torch.cat([
                p.grad.flatten()
                for p in model.parameters()
                if p.grad is not None
            ])

            gradients.append(grad_vec)
            samples_collected += len(inputs)

        # Stack gradients: (N, d) where d = total parameters
        gradients = torch.stack(gradients)

        # Fisher information: (1/N) * G^T @ G where G is gradient matrix
        fisher = (gradients.T @ gradients) / len(gradients)

        return fisher

    def _fisher_rao_distance(
        self,
        fisher_a: torch.Tensor,
        fisher_b: torch.Tensor
    ) -> float:
        """
        Compute Fisher-Rao distance between two models.

        Approximation: d_FR(A, B) ≈ ||log(Σ_A) - log(Σ_B)||_F

        where Σ is Fisher information matrix (interpreted as covariance of gradients)

        This is coordinate-free distance on statistical manifold.
        """
        # Eigenvalue decomposition
        reg = self.config.fisher_regularization
        eigvals_a, eigvecs_a = torch.linalg.eigh(fisher_a + reg * torch.eye(fisher_a.shape[0], device=fisher_a.device))
        eigvals_b, eigvecs_b = torch.linalg.eigh(fisher_b + reg * torch.eye(fisher_b.shape[0], device=fisher_b.device))

        # Log eigenvalues
        log_eigvals_a = torch.log(eigvals_a + reg)
        log_eigvals_b = torch.log(eigvals_b + reg)

        # Frobenius norm of difference
        fr_distance = torch.norm(log_eigvals_a - log_eigvals_b).item()

        return fr_distance

    def _relative_information(
        self,
        fisher_source: torch.Tensor,
        fisher_target: torch.Tensor
    ) -> float:
        """
        Compute relative information from source to target.

        Measures: How much information does source have that target lacks?

        Formula: I_rel(A → B) = tr(F_A @ F_B^{-1}) - log det(F_A @ F_B^{-1})

        This is related to KL divergence on model manifold.
        """
        # Regularize for invertibility
        reg = self.config.fisher_regularization
        fisher_target_inv = torch.linalg.pinv(fisher_target + reg * torch.eye(fisher_target.shape[0], device=fisher_target.device))

        # F_A @ F_B^{-1}
        ratio_matrix = fisher_source @ fisher_target_inv

        # Relative information
        trace_term = torch.trace(ratio_matrix).item()
        logdet_term = torch.slogdet(ratio_matrix)[1].item()

        relative_info = trace_term - logdet_term

        return relative_info

    def _aicc_with_jacobian(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        fisher: torch.Tensor
    ) -> float:
        """
        Compute AICc (corrected Akaike Information Criterion) with Jacobian correction.

        Standard AICc: AICc = 2k - 2log L + 2k(k+1)/(n-k-1)

        Jacobian correction accounts for change of basis (reparameterization invariance):
            AICc_corrected = AICc + log|det(J^T J)|

        where J is Jacobian of model output w.r.t. parameters.

        The correction ensures AICc is coordinate-free (doesn't depend on parameterization).
        """
        model.eval()

        # Number of parameters
        k = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Compute log likelihood
        total_log_likelihood = 0.0
        n_samples = 0

        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
                log_likelihood = log_probs[range(len(targets)), targets].sum().item()

                total_log_likelihood += log_likelihood
                n_samples += len(inputs)

        # Standard AICc
        aicc = 2 * k - 2 * total_log_likelihood

        if n_samples > k + 1:
            aicc += 2 * k * (k + 1) / (n_samples - k - 1)

        # Jacobian correction
        # log|det(J^T J)| ≈ log|det(Fisher)| (Fisher is empirical J^T J)
        try:
            reg = self.config.fisher_regularization
            log_det_fisher = torch.logdet(fisher + reg * torch.eye(fisher.shape[0], device=fisher.device)).item()
            jacobian_correction = self.config.jacobian_coefficient * log_det_fisher
        except:
            jacobian_correction = 0.0

        aicc_corrected = aicc + jacobian_correction

        return aicc_corrected

    def _interpret_comparison(
        self,
        fr_distance: float,
        rel_info_a_to_b: float,
        rel_info_b_to_a: float,
        aicc_delta: float
    ) -> str:
        """
        Generate semantic interpretation of geometric comparison.

        Returns human-readable explanation of what the numbers mean.
        """
        interpretations = []

        # Fisher-Rao distance interpretation
        if fr_distance < self.config.fr_very_close:
            interpretations.append("Models are geometrically VERY CLOSE on statistical manifold (nearly equivalent).")
        elif fr_distance < self.config.fr_similar:
            interpretations.append("Models are geometrically SIMILAR but distinct.")
        elif fr_distance < self.config.fr_different:
            interpretations.append("Models are geometrically DIFFERENT (occupy separate regions of manifold).")
        else:
            interpretations.append("Models are geometrically FAR APART (fundamentally different statistical behaviors).")

        # Relative information interpretation
        if rel_info_a_to_b > rel_info_b_to_a + self.config.rel_info_threshold:
            interpretations.append(f"Model A captures MORE information than B (Δ={rel_info_a_to_b - rel_info_b_to_a:.2f}).")
        elif rel_info_b_to_a > rel_info_a_to_b + self.config.rel_info_threshold:
            interpretations.append(f"Model B captures MORE information than A (Δ={rel_info_b_to_a - rel_info_a_to_b:.2f}).")
        else:
            interpretations.append("Models capture SIMILAR amounts of information.")

        # AICc interpretation
        if aicc_delta < -2:
            interpretations.append(f"Model A is BETTER (ΔAICc={aicc_delta:.1f} < -2, strong evidence).")
        elif aicc_delta > 2:
            interpretations.append(f"Model B is BETTER (ΔAICc={aicc_delta:.1f} > 2, strong evidence).")
        else:
            interpretations.append(f"Models are EQUIVALENT in quality (ΔAICc={aicc_delta:.1f} ∈ [-2, 2]).")

        return " ".join(interpretations)


class SemanticDistiller:
    """
    Distill single training run into actionable semantic JSON.

    Extracts:
    - Geometric position on model manifold
    - Intrinsic dimensionality (effective degrees of freedom)
    - Information flow patterns
    - Phase transitions during training
    - Actionable recommendations
    """

    def __init__(self, device: torch.device, config: StatisticalGeometryConfig, numerical_config: NumericalConfig):
        self.device = device
        self.config = config
        self.numerical_config = numerical_config

    def distill_run(
        self,
        model: nn.Module,
        training_history: List[Dict[str, float]],
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """
        Distill training run into semantic JSON.

        Returns:
            Dictionary with geometric properties, phase transitions, and recommendations
        """
        geometry = StatisticalGeometry(device)

        # Compute Fisher information at end of training
        fisher = geometry._compute_fisher_information(model, data_loader)

        # Intrinsic dimensionality (effective degrees of freedom)
        intrinsic_dim = self._compute_intrinsic_dimensionality(fisher)

        # Information flow during training
        info_flow = self._analyze_information_flow(training_history)

        # Phase transitions (qualitative changes in dynamics)
        phase_transitions = self._detect_phase_transitions(training_history)

        # Recommendations
        nominal_params = sum(p.numel() for p in model.parameters())
        recommendations = self._generate_recommendations(
            intrinsic_dim, info_flow, phase_transitions, training_history, nominal_params
        )

        return {
            "geometric_properties": {
                "intrinsic_dimensionality": intrinsic_dim,
                "nominal_parameters": sum(p.numel() for p in model.parameters()),
                "overparameterization_ratio": sum(p.numel() for p in model.parameters()) / intrinsic_dim if intrinsic_dim > 0 else float('inf'),
                "fisher_condition_number": self._condition_number(fisher),
            },
            "information_flow": info_flow,
            "phase_transitions": phase_transitions,
            "recommendations": recommendations,
            "training_summary": {
                "final_loss": training_history[-1]['loss'],
                "final_accuracy": training_history[-1]['accuracy'],
                "convergence_speed": self._convergence_speed(training_history),
                "stability": self._training_stability(training_history)
            }
        }

    def _compute_intrinsic_dimensionality(self, fisher: torch.Tensor) -> float:
        """
        Compute intrinsic dimensionality of model via Fisher information spectrum.

        Intrinsic dim = (tr(F))^2 / tr(F^2)

        This is effective number of degrees of freedom (may be << nominal parameters).
        """
        trace_fisher = torch.trace(fisher).item()
        trace_fisher_squared = torch.trace(fisher @ fisher).item()

        intrinsic_dim = (trace_fisher ** 2) / (trace_fisher_squared + 1e-10)

        return intrinsic_dim

    def _condition_number(self, fisher: torch.Tensor) -> float:
        """Compute condition number (ratio of largest to smallest eigenvalue)."""
        eigvals = torch.linalg.eigvalsh(fisher + 1e-6 * torch.eye(fisher.shape[0], device=fisher.device))
        return (eigvals.max() / eigvals.min()).item()

    def _analyze_information_flow(self, training_history: List[Dict]) -> Dict:
        """Analyze information flow patterns during training."""
        losses = [h['loss'] for h in training_history]

        # Information gain per epoch (negative loss change)
        info_gains = [-losses[i+1] + losses[i] for i in range(len(losses)-1)]

        return {
            "total_information_gain": sum(info_gains),
            "average_gain_per_epoch": np.mean(info_gains),
            "information_flow_pattern": "increasing" if info_gains[-1] > info_gains[0] else "decreasing"
        }

    def _detect_phase_transitions(self, training_history: List[Dict]) -> List[Dict]:
        """Detect phase transitions (sudden changes in training dynamics)."""
        losses = np.array([h['loss'] for h in training_history])

        # Detect sudden gradient changes
        gradients = np.diff(losses)
        gradient_changes = np.abs(np.diff(gradients))

        # Threshold: 2 std deviations above mean
        threshold = np.mean(gradient_changes) + 2 * np.std(gradient_changes)

        transitions = []
        for i, change in enumerate(gradient_changes):
            if change > threshold:
                transitions.append({
                    "epoch": i + 1,
                    "gradient_change": float(change),
                    "loss_before": float(losses[i]),
                    "loss_after": float(losses[i+2])
                })

        return transitions

    def _generate_recommendations(
        self,
        intrinsic_dim: float,
        info_flow: Dict,
        phase_transitions: List,
        training_history: List[Dict],
        nominal_params: int
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Overparameterization
        if intrinsic_dim < nominal_params / 10:
            recommendations.append(
                f"Model is SEVERELY OVERPARAMETERIZED (intrinsic dim = {intrinsic_dim:.0f}, "
                f"nominal = {nominal_params}). Consider pruning or using smaller architecture."
            )

        # Information flow
        if info_flow['average_gain_per_epoch'] < 0.01:
            recommendations.append(
                "Information gain per epoch is LOW. Consider increasing learning rate or "
                "adding curriculum learning."
            )

        # Phase transitions
        if len(phase_transitions) > 5:
            recommendations.append(
                f"Training had {len(phase_transitions)} phase transitions (unstable). "
                "Consider adding gradient clipping or warmup schedule."
            )

        # Convergence
        final_loss = training_history[-1]['loss']
        if final_loss > 1.0:
            recommendations.append(
                f"High final loss ({final_loss:.2f}). Model may need more capacity or training time."
            )

        if not recommendations:
            recommendations.append("Training looks HEALTHY. No major issues detected.")

        return recommendations

    def _convergence_speed(self, training_history: List[Dict]) -> str:
        """Classify convergence speed."""
        losses = [h['loss'] for h in training_history]
        initial_loss = losses[0]
        final_loss = losses[-1]

        reduction_ratio = (initial_loss - final_loss) / initial_loss

        if reduction_ratio > 0.9:
            return "fast"
        elif reduction_ratio > 0.5:
            return "moderate"
        else:
            return "slow"

    def _training_stability(self, training_history: List[Dict]) -> str:
        """Assess training stability."""
        losses = np.array([h['loss'] for h in training_history])
        loss_std = np.std(np.diff(losses))

        if loss_std < 0.01:
            return "very_stable"
        elif loss_std < 0.1:
            return "stable"
        elif loss_std < 1.0:
            return "moderate"
        else:
            return "unstable"


def save_semantic_json(distilled: Dict[str, Any], output_path: Path):
    """Save semantic distillation to JSON."""
    with open(output_path, 'w') as f:
        json.dump(distilled, f, indent=2)

    logger.info(f"Semantic distillation saved to {output_path}")
