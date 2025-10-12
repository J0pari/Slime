import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)

@dataclass
class LossWeights:
    reconstruction: float = 1.0
    rank_regularization: float = 0.1
    coherence_regularization: float = 0.1
    diversity: float = 0.05
    archive_coverage: float = 0.05
    fitness_variance: float = 0.02

class MultiObjectiveLoss(nn.Module):

    def __init__(self, weights: Optional[LossWeights]=None):
        super().__init__()
        self.weights = weights or LossWeights()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def reconstruction_loss(self, output: torch.Tensor, target: torch.Tensor, task_type: str='regression') -> torch.Tensor:
        if task_type == 'regression':
            return self.mse(output, target)
        elif task_type == 'classification':
            return self.cross_entropy(output, target)
        else:
            raise ValueError(f'Unknown task type: {task_type}')

    def rank_regularization_loss(self, correlation_matrices: list[torch.Tensor]) -> torch.Tensor:
        if not correlation_matrices:
            return torch.tensor(0.0)
        rank_losses = []
        for corr in correlation_matrices:
            if corr.numel() == 0:
                continue
            eigenvalues = torch.linalg.eigvalsh(corr)
            eigenvalues = torch.clamp(eigenvalues, min=1e-10)
            normalized_eigs = eigenvalues / (eigenvalues.sum() + 1e-10)
            entropy = -(normalized_eigs * torch.log(normalized_eigs + 1e-10)).sum()
            max_entropy = torch.log(torch.tensor(float(len(eigenvalues))))
            rank_loss = 1.0 - entropy / (max_entropy + 1e-10)
            rank_losses.append(rank_loss)
        if not rank_losses:
            return torch.tensor(0.0)
        return torch.stack(rank_losses).mean()

    def coherence_regularization_loss(self, coherence_scores: list[torch.Tensor]) -> torch.Tensor:
        if not coherence_scores:
            return torch.tensor(0.0)
        coherence_tensor = torch.stack([c for c in coherence_scores if c.numel() > 0])
        if coherence_tensor.numel() == 0:
            return torch.tensor(0.0)
        target_coherence = 0.5
        return ((coherence_tensor - target_coherence) ** 2).mean()

    def diversity_loss(self, pseudopod_outputs: list[torch.Tensor]) -> torch.Tensor:
        if len(pseudopod_outputs) < 2:
            return torch.tensor(0.0)
        outputs = torch.stack(pseudopod_outputs)
        pairwise_similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                similarity = torch.cosine_similarity(outputs[i].flatten(), outputs[j].flatten(), dim=0)
                pairwise_similarities.append(similarity)
        if not pairwise_similarities:
            return torch.tensor(0.0)
        avg_similarity = torch.stack(pairwise_similarities).mean()
        return avg_similarity

    def archive_coverage_loss(self, archive_coverage: float, target_coverage: float=0.2) -> torch.Tensor:
        coverage_tensor = torch.tensor(archive_coverage)
        if coverage_tensor < target_coverage:
            return (target_coverage - coverage_tensor) ** 2
        else:
            return torch.tensor(0.0)

    def fitness_variance_loss(self, fitness_scores: list[float]) -> torch.Tensor:
        if len(fitness_scores) < 2:
            return torch.tensor(0.0)
        fitness_tensor = torch.tensor(fitness_scores)
        variance = torch.var(fitness_tensor)
        target_variance = 0.1
        return (target_variance - variance) ** 2 if variance < target_variance else torch.tensor(0.0)

    def forward(self, output: torch.Tensor, target: torch.Tensor, correlation_matrices: Optional[list[torch.Tensor]]=None, coherence_scores: Optional[list[torch.Tensor]]=None, pseudopod_outputs: Optional[list[torch.Tensor]]=None, archive_coverage: Optional[float]=None, fitness_scores: Optional[list[float]]=None, task_type: str='regression') -> Dict[str, torch.Tensor]:
        losses = {}
        losses['reconstruction'] = self.reconstruction_loss(output, target, task_type)
        if correlation_matrices:
            losses['rank_regularization'] = self.rank_regularization_loss(correlation_matrices)
        else:
            losses['rank_regularization'] = torch.tensor(0.0)
        if coherence_scores:
            losses['coherence_regularization'] = self.coherence_regularization_loss(coherence_scores)
        else:
            losses['coherence_regularization'] = torch.tensor(0.0)
        if pseudopod_outputs:
            losses['diversity'] = self.diversity_loss(pseudopod_outputs)
        else:
            losses['diversity'] = torch.tensor(0.0)
        if archive_coverage is not None:
            losses['archive_coverage'] = self.archive_coverage_loss(archive_coverage)
        else:
            losses['archive_coverage'] = torch.tensor(0.0)
        if fitness_scores:
            losses['fitness_variance'] = self.fitness_variance_loss(fitness_scores)
        else:
            losses['fitness_variance'] = torch.tensor(0.0)
        total_loss = self.weights.reconstruction * losses['reconstruction'] + self.weights.rank_regularization * losses['rank_regularization'] + self.weights.coherence_regularization * losses['coherence_regularization'] + self.weights.diversity * losses['diversity'] + self.weights.archive_coverage * losses['archive_coverage'] + self.weights.fitness_variance * losses['fitness_variance']
        losses['total'] = total_loss
        return losses

    def get_loss_weights(self) -> LossWeights:
        return self.weights

    def set_loss_weights(self, weights: LossWeights) -> None:
        self.weights = weights
        logger.info(f'Updated loss weights: {weights}')