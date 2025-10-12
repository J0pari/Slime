import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
logger = logging.getLogger(__name__)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning('matplotlib not available - visualizations disabled')
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

class BehavioralSpaceVisualizer:

    def __init__(self, grid_size: Tuple[int, int]=(50, 50), figsize: Tuple[int, int]=(12, 10)):
        if not HAS_MATPLOTLIB:
            raise ImportError('matplotlib required for visualization')
        self.grid_size = grid_size
        self.figsize = figsize
        if HAS_SEABORN:
            sns.set_style('whitegrid')

    def plot_archive_heatmap(self, archive_grid: np.ndarray, fitness_grid: np.ndarray, save_path: Optional[Path]=None, title: str='Behavioral Space Coverage') -> None:
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        im1 = axes[0].imshow(archive_grid.T, origin='lower', cmap='Blues', aspect='auto', interpolation='nearest')
        axes[0].set_xlabel('Rank')
        axes[0].set_ylabel('Coherence')
        axes[0].set_title('Archive Coverage')
        plt.colorbar(im1, ax=axes[0], label='Occupied')
        masked_fitness = np.ma.masked_where(archive_grid == 0, fitness_grid)
        im2 = axes[1].imshow(masked_fitness.T, origin='lower', cmap='viridis', aspect='auto', interpolation='nearest')
        axes[1].set_xlabel('Rank')
        axes[1].set_ylabel('Coherence')
        axes[1].set_title('Elite Fitness')
        plt.colorbar(im2, ax=axes[1], label='Fitness')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f'Saved behavioral space plot to {save_path}')
        else:
            plt.show()
        plt.close(fig)

    def plot_pseudopod_trajectories(self, trajectories: List[List[Tuple[float, float]]], fitness_history: Optional[List[List[float]]]=None, save_path: Optional[Path]=None, title: str='Pseudopod Trajectories in Behavioral Space') -> None:
        fig, ax = plt.subplots(figsize=self.figsize)
        for i, trajectory in enumerate(trajectories):
            if not trajectory:
                continue
            ranks, coherences = zip(*trajectory)
            if fitness_history and i < len(fitness_history):
                points = np.array([ranks, coherences]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                from matplotlib.collections import LineCollection
                lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(vmin=min(fitness_history[i]), vmax=max(fitness_history[i])), alpha=0.7, linewidth=2)
                lc.set_array(np.array(fitness_history[i][:-1]))
                ax.add_collection(lc)
            else:
                ax.plot(ranks, coherences, alpha=0.5, linewidth=1)
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
            logger.info(f'Saved trajectory plot to {save_path}')
        else:
            plt.show()
        plt.close(fig)

class TrainingVisualizer:

    def __init__(self, figsize: Tuple[int, int]=(14, 10)):
        if not HAS_MATPLOTLIB:
            raise ImportError('matplotlib required for visualization')
        self.figsize = figsize

    def plot_training_curves(self, metrics_history: Dict[str, List[float]], save_path: Optional[Path]=None, title: str='Training Metrics') -> None:
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
        for i in range(num_metrics, len(axes)):
            axes[i].axis('off')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f'Saved training curves to {save_path}')
        else:
            plt.show()
        plt.close(fig)

    def plot_lifecycle_dynamics(self, pool_size_history: List[int], archive_size_history: List[int], birth_events: List[int], death_events: List[int], save_path: Optional[Path]=None, title: str='Lifecycle Dynamics') -> None:
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        axes[0].plot(pool_size_history, linewidth=2, label='Pool Size')
        for birth_step in birth_events:
            axes[0].axvline(birth_step, color='green', alpha=0.3, linewidth=0.5)
        for death_step in death_events:
            axes[0].axvline(death_step, color='red', alpha=0.3, linewidth=0.5)
        axes[0].set_ylabel('Pool Size')
        axes[0].set_title('Pseudopod Pool Dynamics')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
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
            logger.info(f'Saved lifecycle dynamics to {save_path}')
        else:
            plt.show()
        plt.close(fig)

    def plot_loss_components(self, loss_history: Dict[str, List[float]], save_path: Optional[Path]=None, title: str='Multi-Objective Loss Components') -> None:
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
            logger.info(f'Saved loss components to {save_path}')
        else:
            plt.show()
        plt.close(fig)

class ArchitectureVisualizer:

    def __init__(self, figsize: Tuple[int, int]=(12, 10)):
        if not HAS_MATPLOTLIB:
            raise ImportError('matplotlib required for visualization')
        self.figsize = figsize

    def plot_attention_weights(self, attention_weights: np.ndarray, save_path: Optional[Path]=None, title: str='Attention Weights') -> None:
        num_heads = attention_weights.shape[0]
        nrows = int(np.ceil(np.sqrt(num_heads)))
        ncols = int(np.ceil(num_heads / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=self.figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]
        for head_idx in range(num_heads):
            im = axes[head_idx].imshow(attention_weights[head_idx], cmap='viridis', aspect='auto', interpolation='nearest')
            axes[head_idx].set_title(f'Head {head_idx}')
            axes[head_idx].set_xlabel('Key Position')
            axes[head_idx].set_ylabel('Query Position')
            plt.colorbar(im, ax=axes[head_idx])
        for i in range(num_heads, len(axes)):
            axes[i].axis('off')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f'Saved attention weights to {save_path}')
        else:
            plt.show()
        plt.close(fig)

    def plot_correlation_matrix(self, correlation_matrix: np.ndarray, save_path: Optional[Path]=None, title: str='Pseudopod Correlation Matrix') -> None:
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
        ax.set_xlabel('Pseudopod Index')
        ax.set_ylabel('Pseudopod Index')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Correlation')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f'Saved correlation matrix to {save_path}')
        else:
            plt.show()
        plt.close(fig)

def create_visualization_suite(output_dir: Path, archive_data: Optional[Dict]=None, training_data: Optional[Dict]=None, architecture_data: Optional[Dict]=None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if archive_data:
        viz = BehavioralSpaceVisualizer()
        viz.plot_archive_heatmap(archive_data['grid'], archive_data['fitness'], save_path=output_dir / 'behavioral_space.png')
        if 'trajectories' in archive_data:
            viz.plot_pseudopod_trajectories(archive_data['trajectories'], archive_data.get('fitness_history'), save_path=output_dir / 'trajectories.png')
    if training_data:
        viz = TrainingVisualizer()
        viz.plot_training_curves(training_data['metrics'], save_path=output_dir / 'training_curves.png')
        if 'lifecycle' in training_data:
            viz.plot_lifecycle_dynamics(training_data['lifecycle']['pool_size'], training_data['lifecycle']['archive_size'], training_data['lifecycle']['births'], training_data['lifecycle']['deaths'], save_path=output_dir / 'lifecycle_dynamics.png')
        if 'losses' in training_data:
            viz.plot_loss_components(training_data['losses'], save_path=output_dir / 'loss_components.png')
    if architecture_data:
        viz = ArchitectureVisualizer()
        if 'attention_weights' in architecture_data:
            viz.plot_attention_weights(architecture_data['attention_weights'], save_path=output_dir / 'attention_weights.png')
        if 'correlation' in architecture_data:
            viz.plot_correlation_matrix(architecture_data['correlation'], save_path=output_dir / 'correlation_matrix.png')
    logger.info(f'Visualization suite saved to {output_dir}')

def visualize_behavioral_space(archive, save_path: Optional[Path]=None):
    """Convenience function to visualize archive behavioral space."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available - skipping visualization")
        return

    # Create simple 2D projection of behavioral space
    viz = BehavioralSpaceVisualizer()

    # Get archive data - if CVT archive, extract elite positions and fitness
    if hasattr(archive, 'get_all_elites'):
        elites = archive.get_all_elites()
        if not elites:
            logger.warning("Archive is empty - no visualization")
            return

        # Create simple grid visualization
        grid_size = (50, 50)
        archive_grid = np.zeros(grid_size)
        fitness_grid = np.zeros(grid_size)

        for elite_data in elites:
            if 'behavior' in elite_data and 'fitness' in elite_data:
                behavior = elite_data['behavior']
                # Map to grid (simple 2D projection of first 2 dims)
                if len(behavior) >= 2:
                    x = int(np.clip(behavior[0] * grid_size[0], 0, grid_size[0]-1))
                    y = int(np.clip(behavior[1] * grid_size[1], 0, grid_size[1]-1))
                    archive_grid[x, y] = 1
                    fitness_grid[x, y] = elite_data['fitness']

        viz.plot_archive_heatmap(archive_grid, fitness_grid, save_path=save_path)
    else:
        logger.warning(f"Archive type {type(archive)} not supported for visualization")
