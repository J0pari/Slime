"""Visualization tools for behavioral space and training dynamics"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy imports for visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - visualizations disabled")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class BehavioralSpaceVisualizer:
    """Visualize MAP-Elites behavioral space (rank, coherence)"""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (50, 50),
        figsize: Tuple[int, int] = (12, 10),
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for visualization")

        self.grid_size = grid_size
        self.figsize = figsize

        if HAS_SEABORN:
            sns.set_style("whitegrid")

    def plot_archive_heatmap(
        self,
        archive_grid: np.ndarray,
        fitness_grid: np.ndarray,
        save_path: Optional[Path] = None,
        title: str = "Behavioral Space Coverage",
    ) -> None:
        """Plot heatmap of archive coverage and fitness.

        Args:
            archive_grid: Binary grid (1 = occupied, 0 = empty)
            fitness_grid: Fitness values at each grid cell
            save_path: Path to save figure
            title: Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Coverage heatmap
        im1 = axes[0].imshow(
            archive_grid.T,
            origin='lower',
            cmap='Blues',
            aspect='auto',
            interpolation='nearest',
        )
        axes[0].set_xlabel('Rank')
        axes[0].set_ylabel('Coherence')
        axes[0].set_title('Archive Coverage')
        plt.colorbar(im1, ax=axes[0], label='Occupied')

        # Fitness heatmap
        # Mask empty cells
        masked_fitness = np.ma.masked_where(archive_grid == 0, fitness_grid)
        im2 = axes[1].imshow(
            masked_fitness.T,
            origin='lower',
            cmap='viridis',
            aspect='auto',
            interpolation='nearest',
        )
        axes[1].set_xlabel('Rank')
        axes[1].set_ylabel('Coherence')
        axes[1].set_title('Elite Fitness')
        plt.colorbar(im2, ax=axes[1], label='Fitness')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved behavioral space plot to {save_path}")
        else:
            plt.show()

        plt.close(fig)

    def plot_pseudopod_trajectories(
        self,
        trajectories: List[List[Tuple[float, float]]],
        fitness_history: Optional[List[List[float]]] = None,
        save_path: Optional[Path] = None,
        title: str = "Pseudopod Trajectories in Behavioral Space",
    ) -> None:
        """Plot trajectories of pseudopods through behavioral space.

        Args:
            trajectories: List of trajectories, each trajectory is list of (rank, coherence)
            fitness_history: Optional fitness values for each trajectory point
            save_path: Path to save figure
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot each trajectory
        for i, trajectory in enumerate(trajectories):
            if not trajectory:
                continue

            ranks, coherences = zip(*trajectory)

            if fitness_history and i < len(fitness_history):
                # Color by fitness
                points = np.array([ranks, coherences]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                from matplotlib.collections import LineCollection
                lc = LineCollection(
                    segments,
                    cmap='viridis',
                    norm=plt.Normalize(
                        vmin=min(fitness_history[i]),
                        vmax=max(fitness_history[i])
                    ),
                    alpha=0.7,
                    linewidth=2,
                )
                lc.set_array(np.array(fitness_history[i][:-1]))
                ax.add_collection(lc)
            else:
                # Simple line
                ax.plot(ranks, coherences, alpha=0.5, linewidth=1)

            # Mark start and end
            ax.scatter(ranks[0], coherences[0], c='green', s=100, marker='o', zorder=5)
            ax.scatter(ranks[-1], coherences[-1], c='red', s=100, marker='x', zorder=5)

        ax.set_xlabel('Rank')
        ax.set_ylabel('Coherence')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if fitness_history:
            plt.colorbar(lc, ax=ax, label='Fitness')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved trajectory plot to {save_path}")
        else:
            plt.show()

        plt.close(fig)


class TrainingVisualizer:
    """Visualize training metrics and dynamics"""

    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for visualization")

        self.figsize = figsize

    def plot_training_curves(
        self,
        metrics_history: Dict[str, List[float]],
        save_path: Optional[Path] = None,
        title: str = "Training Metrics",
    ) -> None:
        """Plot training metrics over time.

        Args:
            metrics_history: Dict mapping metric name to list of values
            save_path: Path to save figure
            title: Plot title
        """
        num_metrics = len(metrics_history)
        nrows = (num_metrics + 1) // 2
        ncols = 2

        fig, axes = plt.subplots(nrows, ncols, figsize=self.figsize)
        axes = axes.flatten() if num_metrics > 1 else [axes]

        for i, (metric_name, values) in enumerate(metrics_history.items()):
            if i >= len(axes):
                break

            axes[i].plot(values, linewidth=2)
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel(metric_name.replace('_', ' ').title())
            axes[i].set_title(metric_name.replace('_', ' ').title())
            axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(num_metrics, len(axes)):
            axes[i].axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training curves to {save_path}")
        else:
            plt.show()

        plt.close(fig)

    def plot_lifecycle_dynamics(
        self,
        pool_size_history: List[int],
        archive_size_history: List[int],
        birth_events: List[int],
        death_events: List[int],
        save_path: Optional[Path] = None,
        title: str = "Lifecycle Dynamics",
    ) -> None:
        """Plot pool and archive dynamics over time.

        Args:
            pool_size_history: Pool size at each step
            archive_size_history: Archive size at each step
            birth_events: Steps where births occurred
            death_events: Steps where deaths occurred
            save_path: Path to save figure
            title: Plot title
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)

        # Pool size
        axes[0].plot(pool_size_history, linewidth=2, label='Pool Size')
        for birth_step in birth_events:
            axes[0].axvline(birth_step, color='green', alpha=0.3, linewidth=0.5)
        for death_step in death_events:
            axes[0].axvline(death_step, color='red', alpha=0.3, linewidth=0.5)
        axes[0].set_ylabel('Pool Size')
        axes[0].set_title('Pseudopod Pool Dynamics')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Archive size
        axes[1].plot(archive_size_history, linewidth=2, color='purple', label='Archive Size')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Archive Size')
        axes[1].set_title('MAP-Elites Archive Growth')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved lifecycle dynamics to {save_path}")
        else:
            plt.show()

        plt.close(fig)

    def plot_loss_components(
        self,
        loss_history: Dict[str, List[float]],
        save_path: Optional[Path] = None,
        title: str = "Multi-Objective Loss Components",
    ) -> None:
        """Plot individual loss components over time.

        Args:
            loss_history: Dict mapping loss name to list of values
            save_path: Path to save figure
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for loss_name, values in loss_history.items():
            ax.plot(values, linewidth=2, label=loss_name.replace('_', ' ').title(), alpha=0.8)

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved loss components to {save_path}")
        else:
            plt.show()

        plt.close(fig)


class ArchitectureVisualizer:
    """Visualize model architecture and attention patterns"""

    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for visualization")

        self.figsize = figsize

    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        save_path: Optional[Path] = None,
        title: str = "Attention Weights",
    ) -> None:
        """Plot attention weight heatmap.

        Args:
            attention_weights: [num_heads, seq_len, seq_len] attention weights
            save_path: Path to save figure
            title: Plot title
        """
        num_heads = attention_weights.shape[0]
        nrows = int(np.ceil(np.sqrt(num_heads)))
        ncols = int(np.ceil(num_heads / nrows))

        fig, axes = plt.subplots(nrows, ncols, figsize=self.figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]

        for head_idx in range(num_heads):
            im = axes[head_idx].imshow(
                attention_weights[head_idx],
                cmap='viridis',
                aspect='auto',
                interpolation='nearest',
            )
            axes[head_idx].set_title(f'Head {head_idx}')
            axes[head_idx].set_xlabel('Key Position')
            axes[head_idx].set_ylabel('Query Position')
            plt.colorbar(im, ax=axes[head_idx])

        # Hide unused subplots
        for i in range(num_heads, len(axes)):
            axes[i].axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention weights to {save_path}")
        else:
            plt.show()

        plt.close(fig)

    def plot_correlation_matrix(
        self,
        correlation_matrix: np.ndarray,
        save_path: Optional[Path] = None,
        title: str = "Pseudopod Correlation Matrix",
    ) -> None:
        """Plot correlation matrix between pseudopods.

        Args:
            correlation_matrix: [num_pseudopods, num_pseudopods] correlation
            save_path: Path to save figure
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(
            correlation_matrix,
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            aspect='auto',
            interpolation='nearest',
        )
        ax.set_xlabel('Pseudopod Index')
        ax.set_ylabel('Pseudopod Index')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Correlation')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved correlation matrix to {save_path}")
        else:
            plt.show()

        plt.close(fig)


def create_visualization_suite(
    output_dir: Path,
    archive_data: Optional[Dict] = None,
    training_data: Optional[Dict] = None,
    architecture_data: Optional[Dict] = None,
) -> None:
    """Generate complete visualization suite.

    Args:
        output_dir: Directory to save visualizations
        archive_data: Behavioral space data
        training_data: Training metrics data
        architecture_data: Architecture visualization data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if archive_data:
        viz = BehavioralSpaceVisualizer()
        viz.plot_archive_heatmap(
            archive_data['grid'],
            archive_data['fitness'],
            save_path=output_dir / 'behavioral_space.png',
        )
        if 'trajectories' in archive_data:
            viz.plot_pseudopod_trajectories(
                archive_data['trajectories'],
                archive_data.get('fitness_history'),
                save_path=output_dir / 'trajectories.png',
            )

    if training_data:
        viz = TrainingVisualizer()
        viz.plot_training_curves(
            training_data['metrics'],
            save_path=output_dir / 'training_curves.png',
        )
        if 'lifecycle' in training_data:
            viz.plot_lifecycle_dynamics(
                training_data['lifecycle']['pool_size'],
                training_data['lifecycle']['archive_size'],
                training_data['lifecycle']['births'],
                training_data['lifecycle']['deaths'],
                save_path=output_dir / 'lifecycle_dynamics.png',
            )
        if 'losses' in training_data:
            viz.plot_loss_components(
                training_data['losses'],
                save_path=output_dir / 'loss_components.png',
            )

    if architecture_data:
        viz = ArchitectureVisualizer()
        if 'attention_weights' in architecture_data:
            viz.plot_attention_weights(
                architecture_data['attention_weights'],
                save_path=output_dir / 'attention_weights.png',
            )
        if 'correlation' in architecture_data:
            viz.plot_correlation_matrix(
                architecture_data['correlation'],
                save_path=output_dir / 'correlation_matrix.png',
            )

    logger.info(f"Visualization suite saved to {output_dir}")
