import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional
logger = logging.getLogger(__name__)

class TrainingPhase(Enum):
    WARMUP = 'warmup'
    GENTLE = 'gentle'
    FULL = 'full'

@dataclass
class PhaseConfig:
    name: TrainingPhase
    min_step: int
    max_step: Optional[int]
    allow_births: bool
    allow_deaths: bool
    allow_culling: bool
    pool_size_limit: Optional[int]
    description: str

class StabilityManager:

    def __init__(self, warmup_steps: int=1000, gentle_steps: int=5000):
        self.warmup_steps = warmup_steps
        self.gentle_steps = gentle_steps
        self.phases = [PhaseConfig(name=TrainingPhase.WARMUP, min_step=0, max_step=warmup_steps, allow_births=False, allow_deaths=False, allow_culling=False, pool_size_limit=None, description='Static pool, no lifecycle changes, establish baseline'), PhaseConfig(name=TrainingPhase.GENTLE, min_step=warmup_steps, max_step=gentle_steps, allow_births=True, allow_deaths=False, allow_culling=False, pool_size_limit=64, description='Allow births but no deaths, controlled growth'), PhaseConfig(name=TrainingPhase.FULL, min_step=gentle_steps, max_step=None, allow_births=True, allow_deaths=True, allow_culling=True, pool_size_limit=None, description='Full dynamics enabled')]
        self._current_phase: Optional[PhaseConfig] = None
        self._phase_change_steps: list = []

    def get_phase(self, step: int) -> PhaseConfig:
        for phase in reversed(self.phases):
            if step >= phase.min_step:
                if phase.max_step is None or step < phase.max_step:
                    if self._current_phase != phase:
                        self._on_phase_change(phase, step)
                    return phase
        return self.phases[0]

    def _on_phase_change(self, new_phase: PhaseConfig, step: int) -> None:
        if self._current_phase:
            logger.info(f'Phase transition at step {step}: {self._current_phase.name.value} -> {new_phase.name.value}')
            logger.info(f'New phase: {new_phase.description}')
        else:
            logger.info(f'Starting phase: {new_phase.name.value} - {new_phase.description}')
        self._current_phase = new_phase
        self._phase_change_steps.append(step)

    def should_allow_birth(self, step: int) -> bool:
        phase = self.get_phase(step)
        return phase.allow_births

    def should_allow_death(self, step: int) -> bool:
        phase = self.get_phase(step)
        return phase.allow_deaths

    def should_allow_culling(self, step: int) -> bool:
        phase = self.get_phase(step)
        return phase.allow_culling

    def get_pool_size_limit(self, step: int) -> Optional[int]:
        phase = self.get_phase(step)
        return phase.pool_size_limit

    def get_recommended_learning_rate_multiplier(self, step: int) -> float:
        phase = self.get_phase(step)
        if phase.name == TrainingPhase.WARMUP:
            progress = step / max(1, self.warmup_steps)
            return 0.1 + 0.9 * progress
        elif phase.name == TrainingPhase.GENTLE:
            return 1.0
        else:
            return 1.0

    def stats(self) -> dict:
        return {'current_phase': self._current_phase.name.value if self._current_phase else None, 'phase_changes': len(self._phase_change_steps), 'warmup_steps': self.warmup_steps, 'gentle_steps': self.gentle_steps}

def create_default_stability_manager() -> StabilityManager:
    return StabilityManager(warmup_steps=1000, gentle_steps=5000)