import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
from dataclasses import dataclass
import logging
from slime.training.stability import StabilityManager, TrainingPhase
from slime.training.lifecycle import LifecycleManager, LifecycleConfig
from slime.training.losses import MultiObjectiveLoss, LossWeights
from slime.training.fitness import FitnessComputer
from slime.observability.metrics import MetricsCollector
from slime.tests.checkpoint import TestResultCheckpointSystem
from pathlib import Path
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    # Training loop params
    num_epochs: int = 100
    batch_size: int = 32
    device: str = 'cuda'
    gradient_clip_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 1000
    warmup_steps: int = 1000
    gentle_steps: int = 5000
    # Optimizer params
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    # Loss weights
    reconstruction_weight: float = 1.0
    rank_regularization_weight: float = 0.01
    coherence_regularization_weight: float = 0.01
    diversity_weight: float = 0.1
    archive_coverage_weight: float = 0.05
    fitness_variance_weight: float = 0.05
    # Lifecycle config
    max_pool_size: int = 64
    max_archive_size: int = 1000
    max_loss_ratio: float = 10.0
    initial_temp: float = 1.0
    min_temp: float = 0.01

class Trainer:

    def __init__(self, model: nn.Module, optimizer: Optimizer, config: Optional[TrainingConfig]=None, loss_weights: Optional[LossWeights]=None, lifecycle_config: Optional[LifecycleConfig]=None, checkpoint_results: bool=True):
        self.model = model
        self.optimizer = optimizer
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)
        self.stability_manager = StabilityManager(warmup_steps=self.config.warmup_steps, gentle_steps=self.config.gentle_steps)
        self.lifecycle_manager = LifecycleManager(lifecycle_config)
        self.loss_fn = MultiObjectiveLoss(loss_weights)
        self.fitness_computer = FitnessComputer()
        self.metrics = MetricsCollector()
        self._step = 0
        self._epoch = 0
        self.checkpoint_results = checkpoint_results
        if checkpoint_results:
            self.results_checkpoint = TestResultCheckpointSystem(checkpoint_type='results')

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.optimizer.zero_grad()
        outputs, state = self.model(inputs)
        organism = self.model.organism if hasattr(self.model, 'organism') else None
        correlation_matrices = []
        coherence_scores = []
        pseudopod_outputs = []
        fitness_scores = []
        if organism:
            for pod in organism.pseudopod_pool.get_all():
                # Skip pseudopods that haven't been used yet (no correlation computed)
                if not (hasattr(pod, '_correlation') and pod._correlation is not None):
                    continue
                correlation_matrices.append(pod._correlation)
                if hasattr(pod, 'coherence'):
                    coherence_scores.append(pod.coherence())
                if hasattr(pod, 'fitness'):
                    fitness_scores.append(pod.fitness)
            archive_coverage = organism.archive.coverage()
        else:
            archive_coverage = None
        losses = self.loss_fn(output=outputs, target=targets, correlation_matrices=correlation_matrices if correlation_matrices else None, coherence_scores=coherence_scores if coherence_scores else None, pseudopod_outputs=pseudopod_outputs if pseudopod_outputs else None, archive_coverage=archive_coverage, fitness_scores=fitness_scores if fitness_scores else None, task_type='classification')
        total_loss = losses['total']
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        if organism and self._step % 100 == 0:
            for i, pod in enumerate(organism.pseudopod_pool.get_all()):
                fitness = self.fitness_computer.compute_gradient_based_fitness(pod, i)
                if hasattr(pod, '_fitness'):
                    pod._fitness = fitness
        phase = self.stability_manager.get_phase(self._step)
        lifecycle_result = self.lifecycle_manager.step_lifecycle(current_loss=total_loss.item())
        if organism:
            if not self.stability_manager.should_allow_birth(self._step):
                pass
            elif self._step % 1000 == 0:
                if self.stability_manager.should_allow_culling(self._step):
                    organism.pseudopod_pool._cull_low_fitness()
            elif self._step % 100 == 0:
                if self.stability_manager.should_allow_birth(self._step):
                    organism.pseudopod_pool._spawn_batch(state.behavior if state else None)
            if lifecycle_result and 'actions' in lifecycle_result:
                if 'force_cull_pool' in lifecycle_result['actions']:
                    self.lifecycle_manager.force_cull_pool(organism.pseudopod_pool)
                if 'cull_archive' in lifecycle_result['actions']:
                    self.lifecycle_manager.cull_archive(organism.archive)
        self._step += 1
        return {'loss': total_loss.item(), 'reconstruction_loss': losses['reconstruction'].item(), 'rank_loss': losses['rank_regularization'].item(), 'coherence_loss': losses['coherence_regularization'].item(), 'diversity_loss': losses['diversity'].item(), 'phase': phase.name.value, 'lifecycle_frozen': self.lifecycle_manager._lifecycle_frozen}

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        epoch_losses = []
        for batch_idx, batch in enumerate(train_loader):
            step_result = self.train_step(batch)
            epoch_losses.append(step_result['loss'])
            if batch_idx % self.config.log_interval == 0:
                logger.info(f"Epoch {epoch} Step {batch_idx}: loss={step_result['loss']:.4f} phase={step_result['phase']}")
        return {'avg_loss': sum(epoch_losses) / len(epoch_losses), 'min_loss': min(epoch_losses), 'max_loss': max(epoch_losses)}

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader]=None) -> Dict:
        logger.info(f'Starting training for {self.config.num_epochs} epochs')
        training_history = []
        for epoch in range(self.config.num_epochs):
            self._epoch = epoch
            epoch_result = self.train_epoch(train_loader, epoch)
            training_history.append(epoch_result)
            logger.info(f"Epoch {epoch} complete: avg_loss={epoch_result['avg_loss']:.4f}")

            if self.checkpoint_results and epoch % 10 == 0:
                epoch_checkpoint = {
                    'epoch': epoch,
                    'step': self._step,
                    'avg_loss': epoch_result['avg_loss'],
                    'min_loss': epoch_result['min_loss'],
                    'max_loss': epoch_result['max_loss'],
                    'stats': self.get_stats()
                }
                self.results_checkpoint.checkpoint_test_result(
                    f"epoch_{epoch:04d}",
                    epoch_checkpoint,
                    message=f"Training epoch {epoch} complete"
                )

            if val_loader and epoch % (self.config.eval_interval // len(train_loader)) == 0:
                val_result = self.evaluate(val_loader)
                logger.info(f"Validation: loss={val_result['avg_loss']:.4f}")

                if self.checkpoint_results:
                    val_checkpoint = {
                        'epoch': epoch,
                        'validation': val_result
                    }
                    self.results_checkpoint.checkpoint_test_result(
                        f"validation_epoch_{epoch:04d}",
                        val_checkpoint,
                        message=f"Validation at epoch {epoch}"
                    )

        final_results = {'training_history': training_history, 'final_stats': self.get_stats()}

        if self.checkpoint_results:
            self.results_checkpoint.checkpoint_test_result(
                'training_final',
                final_results,
                message=f"Training complete after {self.config.num_epochs} epochs"
            )

        return final_results

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        eval_losses = []
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs, _ = self.model(inputs)
            loss = self.loss_fn.reconstruction_loss(outputs, targets)
            eval_losses.append(loss.item())
        return {'avg_loss': sum(eval_losses) / len(eval_losses), 'min_loss': min(eval_losses), 'max_loss': max(eval_losses)}

    def get_stats(self) -> Dict:
        stats = {'step': self._step, 'epoch': self._epoch, 'stability': self.stability_manager.stats(), 'lifecycle': self.lifecycle_manager.stats(), 'fitness': self.fitness_computer.stats()}
        if hasattr(self.model, 'organism'):
            stats['organism'] = self.model.organism.stats()
        return stats