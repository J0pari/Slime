import torch
from typing import Optional
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)

@dataclass
class LifecycleConfig:
    max_pool_size: int = 64
    max_archive_size: int = 1000
    max_loss_ratio: float = 10.0
    loss_ema_alpha: float = 0.1
    check_interval: int = 100
    freeze_on_divergence: bool = True

class LifecycleManager:

    def __init__(self, config: Optional[LifecycleConfig]=None):
        self.config = config or LifecycleConfig()
        self._loss_ema: Optional[float] = None
        self._step_count = 0
        self._lifecycle_frozen = False
        self._freeze_count = 0
        self._violation_history: list = []

    def update_loss_ema(self, current_loss: float) -> None:
        if self._loss_ema is None:
            self._loss_ema = current_loss
        else:
            self._loss_ema = self.config.loss_ema_alpha * current_loss + (1 - self.config.loss_ema_alpha) * self._loss_ema

    def check_loss_divergence(self, current_loss: float) -> bool:
        if self._loss_ema is None:
            return False
        ratio = current_loss / (self._loss_ema + 1e-10)
        if ratio > self.config.max_loss_ratio:
            logger.warning(f'Loss divergence detected: current={current_loss:.4f}, ema={self._loss_ema:.4f}, ratio={ratio:.2f}')
            return True
        return False

    def check_pool_size_limit(self, pool_size: int) -> bool:
        if pool_size > self.config.max_pool_size:
            logger.warning(f'Pool size limit exceeded: current={pool_size}, max={self.config.max_pool_size}')
            return True
        return False

    def check_archive_size_limit(self, archive_size: int) -> bool:
        if archive_size > self.config.max_archive_size:
            logger.warning(f'Archive size limit exceeded: current={archive_size}, max={self.config.max_archive_size}')
            return True
        return False

    def step(self, current_loss: float, pool_size: int, archive_size: int) -> dict:
        self._step_count += 1
        self.update_loss_ema(current_loss)
        results = {'step': self._step_count, 'violations': [], 'actions': [], 'lifecycle_frozen': self._lifecycle_frozen}
        if self._step_count % self.config.check_interval == 0:
            loss_diverging = self.check_loss_divergence(current_loss)
            pool_exceeds = self.check_pool_size_limit(pool_size)
            archive_exceeds = self.check_archive_size_limit(archive_size)
            if loss_diverging:
                results['violations'].append('loss_divergence')
                if self.config.freeze_on_divergence and (not self._lifecycle_frozen):
                    self._lifecycle_frozen = True
                    self._freeze_count += 1
                    results['actions'].append('freeze_lifecycle')
                    logger.warning(f'Freezing lifecycle due to loss divergence (freeze #{self._freeze_count})')
            if pool_exceeds:
                results['violations'].append('pool_size_limit')
                results['actions'].append('force_cull_pool')
            if archive_exceeds:
                results['violations'].append('archive_size_limit')
                results['actions'].append('cull_archive')
            if results['violations']:
                self._violation_history.append({'step': self._step_count, 'violations': results['violations'], 'current_loss': current_loss, 'loss_ema': self._loss_ema, 'pool_size': pool_size, 'archive_size': archive_size})
        return results

    def should_allow_lifecycle_changes(self) -> bool:
        return not self._lifecycle_frozen

    def unfreeze_lifecycle(self) -> None:
        if self._lifecycle_frozen:
            logger.info('Unfreezing lifecycle')
            self._lifecycle_frozen = False

    def force_cull_pool(self, pool, target_size: Optional[int]=None) -> int:
        if target_size is None:
            target_size = self.config.max_pool_size
        current_size = pool.size()
        if current_size <= target_size:
            return 0
        to_cull = current_size - target_size
        components = pool.get_all()
        components_sorted = sorted(components, key=lambda c: c.fitness, reverse=True)
        pool._components = components_sorted[:target_size]
        logger.warning(f'Force culled {to_cull} components from pool')
        return to_cull

    def cull_archive(self, archive, target_size: Optional[int]=None) -> int:
        if target_size is None:
            target_size = self.config.max_archive_size
        current_size = archive.size()
        if current_size <= target_size:
            return 0
        to_remove = current_size - target_size
        elites = list(archive._archive.items())
        elites_sorted = sorted(elites, key=lambda x: x[1].fitness, reverse=True)
        archive._archive = dict(elites_sorted[:target_size])
        logger.warning(f'Culled {to_remove} elites from archive')
        return to_remove

    def stats(self) -> dict:
        return {'step_count': self._step_count, 'loss_ema': self._loss_ema, 'lifecycle_frozen': self._lifecycle_frozen, 'freeze_count': self._freeze_count, 'total_violations': len(self._violation_history), 'config': {'max_pool_size': self.config.max_pool_size, 'max_archive_size': self.config.max_archive_size, 'max_loss_ratio': self.config.max_loss_ratio}}

    def get_violation_history(self) -> list:
        return self._violation_history

    def reset(self) -> None:
        self._loss_ema = None
        self._step_count = 0
        self._lifecycle_frozen = False
        self._freeze_count = 0
        self._violation_history.clear()