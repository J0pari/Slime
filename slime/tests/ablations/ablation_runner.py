"""
Ablation study runner with statistical testing.

Compares Slime Mold Transformer against baselines and feature-ablated versions.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
import time
import logging
from pathlib import Path
import json
from .statistical_geometry import StatisticalGeometry, SemanticDistiller, save_semantic_json

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation study."""

    # Model configurations to compare
    models: List[str] = field(default_factory=lambda: [
        'slime_full',           # Full Slime system
        'slime_no_lifecycle',   # Disable lifecycle (static pool)
        'slime_no_archive',     # Disable archive (no bootstrapping)
        'slime_no_fitness',     # Random selection instead of fitness
        'transformer_baseline', # Standard transformer
        'flash_attention'       # Flash attention baseline
    ])

    # Training parameters
    num_runs: int = 3          # Statistical significance requires ≥3 runs
    num_epochs: int = 10       # Training epochs per run
    batch_size: int = 32
    learning_rate: float = 1e-4

    # Dataset
    dataset: str = 'mnist'     # mnist, cifar10, tinystories

    # Metrics to track
    metrics: List[str] = field(default_factory=lambda: [
        'accuracy',            # Task accuracy
        'throughput',          # Samples/sec
        'memory_peak',         # Peak GPU memory (MB)
        'loss',                # Final loss value
        'convergence_step',    # Step where loss < threshold
        'pool_diversity',      # Behavioral diversity (Slime only)
        'archive_coverage'     # Archive fill rate (Slime only)
    ])

    # Statistical testing
    significance_level: float = 0.05  # p < 0.05

    # Output
    results_dir: str = 'ablation_results'


@dataclass
class RunResult:
    """Results from single training run."""
    model_name: str
    run_id: int
    metrics: Dict[str, float]
    training_history: List[Dict[str, float]]
    duration_seconds: float


class AblationRunner:
    """
    Runs ablation studies with statistical testing.

    Compares Slime Mold Transformer against:
    1. Baseline transformers (same parameters, standard architecture)
    2. Feature-ablated Slime (disable lifecycle, archive, fitness)

    Reports:
    - Mean ± std for each metric
    - Statistical significance (t-tests)
    - Effect sizes (Cohen's d)
    """

    def __init__(self, config: AblationConfig):
        self.config = config
        self.results: Dict[str, List[RunResult]] = {model: [] for model in config.models}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_all_ablations(self) -> Dict[str, Any]:
        """
        Run complete ablation study.

        Returns:
            Dictionary with statistical comparisons
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ABLATION STUDY: {len(self.config.models)} models × {self.config.num_runs} runs")
        logger.info(f"Dataset: {self.config.dataset}, Epochs: {self.config.num_epochs}")
        logger.info(f"Statistical geometry: Fisher-Rao + AICc with Jacobian")
        logger.info(f"{'='*80}\n")

        # Run all models
        for model_name in self.config.models:

            logger.info(f"\n{'='*80}")
            logger.info(f"Model: {model_name}")
            logger.info(f"{'='*80}")

            for run_id in range(self.config.num_runs):
                logger.info(f"  Run {run_id + 1}/{self.config.num_runs}")
                result = self._run_single(model_name, run_id)
                self.results[model_name].append(result)

                # Save intermediate results
                self._save_results()

        # Compute statistical comparisons
        comparisons = self._compute_comparisons()

        # Generate report
        self._generate_report(comparisons)

        return comparisons

    def _run_single(self, model_name: str, run_id: int) -> RunResult:
        """Run single training instance."""
        start_time = time.time()

        # Create model
        model = self._create_model(model_name)
        model = model.to(self.device)

        # Create data loaders
        train_loader, val_loader = self._create_dataloaders()

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        # Create loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop with metric collection
        training_history = []
        for epoch in range(self.config.num_epochs):
            epoch_metrics = self._train_epoch(
                model, train_loader, val_loader, optimizer, criterion, epoch
            )
            training_history.append(epoch_metrics)

            logger.info(
                f"    Epoch {epoch}: "
                f"acc={epoch_metrics['accuracy']:.3f}, "
                f"loss={epoch_metrics['loss']:.4f}, "
                f"throughput={epoch_metrics['throughput']:.1f} samples/s"
            )

        # Collect final metrics
        final_metrics = self._collect_metrics(model, val_loader, training_history)

        duration = time.time() - start_time

        return RunResult(
            model_name=model_name,
            run_id=run_id,
            metrics=final_metrics,
            training_history=training_history,
            duration_seconds=duration
        )

    def _create_model(self, model_name: str) -> nn.Module:
        """Create model - imports from run.py (no DRY violation)."""
        # Import create_full_system from run.py
        import sys
        sys.path.insert(0, '.')
        from run import create_full_system
        from slime.config.dimensions import TINY
        from slime.bench.transformer import TransformerBaseline
        
        if model_name == 'slime_full':
            model = create_full_system(TINY, self.device)
            return model
            
        elif model_name == 'transformer_baseline':
            # Wrap transformer with embedding + classifier
            class TransformerClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = nn.Linear(784, 64, device=self.device)
                    self.transformer = TransformerBaseline(d_model=64, nhead=4, num_layers=2, dim_feedforward=256, device=self.device)
                    self.classifier = nn.Linear(64, 10, device=self.device)
                
                def forward(self, x):
                    x = self.embedding(x).unsqueeze(1)
                    x = self.transformer(x)
                    x = x.mean(dim=1)
                    return self.classifier(x)
            
            return TransformerClassifier()
        
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train/val dataloaders."""
        if self.config.dataset == 'mnist':
            from slime.bench.datasets import MNISTDataset
            train_dataset = MNISTDataset(train=True)
            val_dataset = MNISTDataset(train=False)
        elif self.config.dataset == 'cifar10':
            from slime.bench.datasets import CIFAR10Dataset
            train_dataset = CIFAR10Dataset(train=True)
            val_dataset = CIFAR10Dataset(train=False)
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        return train_loader, val_loader

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Handle different output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # (output, state) tuple

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        duration = time.time() - start_time

        # Validation accuracy
        val_acc = self._evaluate(model, val_loader)

        return {
            'epoch': epoch,
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total,
            'val_accuracy': val_acc,
            'throughput': total / duration,
            'memory_peak': torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        }

    def _evaluate(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Evaluate model on validation set."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    def _collect_metrics(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        training_history: List[Dict]
    ) -> Dict[str, float]:
        """Collect all metrics for this run."""
        metrics = {}

        # Final accuracy
        metrics['accuracy'] = self._evaluate(model, val_loader)

        # Final loss
        metrics['loss'] = training_history[-1]['loss']

        # Average throughput
        metrics['throughput'] = np.mean([h['throughput'] for h in training_history])

        # Peak memory
        metrics['memory_peak'] = max([h['memory_peak'] for h in training_history])

        # Convergence step (first epoch below 0.5 loss)
        convergence_step = None
        for i, h in enumerate(training_history):
            if h['loss'] < 0.5:
                convergence_step = i
                break
        metrics['convergence_step'] = convergence_step if convergence_step is not None else len(training_history)

        # Slime-specific metrics
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'organism'):
            organism = model.encoder.organism
            metrics['pool_size'] = len(organism.pool._components)
            metrics['archive_size'] = len(organism.archive._elites)
            metrics['archive_coverage'] = len(organism.archive._elites) / organism.archive.num_centroids
        else:
            metrics['pool_size'] = 0
            metrics['archive_size'] = 0
            metrics['archive_coverage'] = 0

        return metrics

    def _compute_comparisons(self) -> Dict[str, Any]:
        """
        Compute statistical + geometric comparisons between models.
        
        Uses ALL functions from statistical_geometry.py:
        - StatisticalGeometry.compare_models() for Fisher-Rao, AICc with Jacobian
        - SemanticDistiller.distill_run() for each model
        - save_semantic_json() to export results
        """
        comparisons = {}
        baseline = 'transformer_baseline'
        
        # Create geometry and distiller instances
        geometry = StatisticalGeometry(self.device)
        distiller = SemanticDistiller()
        
        # Get data loader for Fisher information computation
        _, val_loader = self._create_dataloaders()
        
        # Distill semantic JSON for ALL models
        semantic_distillations = {}
        for model_name in self.config.models:
            if not self.results[model_name]:
                continue
                
            # Use last run for distillation
            last_run = self.results[model_name][-1]
            
            # Recreate model for geometric analysis
            model = self._create_model(model_name).to(self.device)
            
            # Distill using ALL SemanticDistiller methods
            semantic = distiller.distill_run(
                model=model,
                training_history=last_run.training_history,
                data_loader=val_loader,
                device=self.device
            )
            semantic_distillations[model_name] = semantic
            
            # Save individual distillation
            save_semantic_json(semantic, self.results_dir / f'{model_name}_semantic.json')

        # Geometric comparisons using Fisher-Rao + AICc
        for model_name in self.config.models:
            if model_name == baseline or not self.results[model_name]:
                continue

            # Traditional statistics (t-tests, Cohen's d)
            model_stats = self._compute_stats(model_name)
            baseline_stats = self._compute_stats(baseline)
            
            traditional_comparison = {}
            for metric in self.config.metrics:
                if metric not in model_stats or metric not in baseline_stats:
                    continue

                model_values = model_stats[metric]['values']
                baseline_values = baseline_stats[metric]['values']

                t_stat, p_value = stats.ttest_ind(model_values, baseline_values)
                pooled_std = np.sqrt(
                    (np.std(model_values)**2 + np.std(baseline_values)**2) / 2
                )
                cohens_d = (np.mean(model_values) - np.mean(baseline_values)) / pooled_std

                traditional_comparison[metric] = {
                    'model_mean': np.mean(model_values),
                    'model_std': np.std(model_values),
                    'baseline_mean': np.mean(baseline_values),
                    'baseline_std': np.std(baseline_values),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config.significance_level,
                    'cohens_d': cohens_d,
                    'effect_size': self._interpret_effect_size(cohens_d)
                }
            
            # Geometric comparison using StatisticalGeometry.compare_models()
            # This uses ALL methods: _compute_fisher_information, _fisher_rao_distance,
            # _relative_information, _aicc_with_jacobian, _interpret_comparison
            model_a = self._create_model(model_name).to(self.device)
            model_b = self._create_model(baseline).to(self.device)
            
            geometric = geometry.compare_models(
                model_a=model_a,
                model_b=model_b,
                data_loader=val_loader,
                model_a_name=model_name,
                model_b_name=baseline
            )

            comparisons[model_name] = {
                'model': model_name,
                'baseline': baseline,
                'traditional_metrics': traditional_comparison,
                'geometric_comparison': {
                    'fisher_rao_distance': geometric.fisher_rao_distance,
                    'relative_info_a_to_b': geometric.relative_info_a_to_b,
                    'relative_info_b_to_a': geometric.relative_info_b_to_a,
                    'aicc_a': geometric.aicc_a,
                    'aicc_b': geometric.aicc_b,
                    'aicc_delta': geometric.aicc_delta,
                    'interpretation': geometric.interpretation
                },
                'semantic_distillation': semantic_distillations[model_name]
            }

        return comparisons

    def _compute_stats(self, model_name: str) -> Dict[str, Dict]:
        """Compute statistics for model across runs."""
        runs = self.results[model_name]
        stats = {}

        for metric in self.config.metrics:
            values = [r.metrics.get(metric, 0) for r in runs]
            stats[metric] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return stats

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'

    def _save_results(self):
        """Save intermediate results to JSON."""
        results_dict = {
            model_name: [
                {
                    'run_id': r.run_id,
                    'metrics': r.metrics,
                    'duration_seconds': r.duration_seconds
                }
                for r in runs
            ]
            for model_name, runs in self.results.items()
        }

        output_path = self.results_dir / 'ablation_results.json'
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def _generate_report(self, comparisons: Dict[str, Any]):
        """Generate human-readable ablation report."""
        report_path = self.results_dir / 'ablation_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ABLATION STUDY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Configuration:\n")
            f.write(f"  Models: {', '.join(self.config.models)}\n")
            f.write(f"  Runs per model: {self.config.num_runs}\n")
            f.write(f"  Epochs: {self.config.num_epochs}\n")
            f.write(f"  Dataset: {self.config.dataset}\n")
            f.write(f"  Significance level: {self.config.significance_level}\n\n")

            for model_name, comparison in comparisons.items():
                f.write(f"\n{'-' * 80}\n")
                f.write(f"{model_name} vs {comparison['baseline']}\n")
                f.write(f"{'-' * 80}\n\n")

                for metric, stats in comparison['metrics'].items():
                    f.write(f"{metric}:\n")
                    f.write(f"  {model_name}: {stats['model_mean']:.3f} ± {stats['model_std']:.3f}\n")
                    f.write(f"  {comparison['baseline']}: {stats['baseline_mean']:.3f} ± {stats['baseline_std']:.3f}\n")
                    f.write(f"  t-statistic: {stats['t_statistic']:.3f}\n")
                    f.write(f"  p-value: {stats['p_value']:.4f}\n")
                    f.write(f"  Significant: {stats['significant']}\n")
                    f.write(f"  Cohen's d: {stats['cohens_d']:.3f} ({stats['effect_size']})\n\n")

        logger.info(f"Report saved to {report_path}")

        # Also print summary to console
        print(f"\n{'=' * 80}")
        print("ABLATION STUDY SUMMARY")
        print(f"{'=' * 80}\n")

        for model_name, comparison in comparisons.items():
            print(f"{model_name}:")
            for metric, stats in comparison['metrics'].items():
                if stats['significant']:
                    delta = stats['model_mean'] - stats['baseline_mean']
                    direction = "better" if delta > 0 else "worse"
                    print(f"  {metric}: {direction} (Δ={delta:+.3f}, p={stats['p_value']:.4f}, d={stats['cohens_d']:.2f})")
            print()
