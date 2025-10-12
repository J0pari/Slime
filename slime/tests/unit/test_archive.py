import pytest
import torch
from slime.memory.archive import BehavioralArchive
from slime.core.pseudopod import Pseudopod
from slime.kernels.torch_fallback import TorchKernel

class TestBehavioralArchive:

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def archive(self):
        return BehavioralArchive(
            dimensions=['rank', 'coherence'],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            resolution=50
        )

    @pytest.fixture
    def pseudopod(self, device):
        kernel = TorchKernel(device)
        return Pseudopod(head_dim=64, kernel=kernel, device=device)

    def test_initialization(self, archive):
        assert archive.size() == 0
        assert len(archive.dimensions) == 2
        assert archive.resolution == 50

    def test_add_real_pseudopod(self, archive, pseudopod):
        behavior = (0.5, 0.6)
        added = archive.add(pseudopod, behavior, fitness=0.75)
        assert added is True
        assert archive.size() == 1

    def test_fitness_replacement(self, archive, device):
        kernel = TorchKernel(device)
        pod1 = Pseudopod(head_dim=64, kernel=kernel, device=device)
        pod2 = Pseudopod(head_dim=64, kernel=kernel, device=device)
        behavior = (0.5, 0.5)
        archive.add(pod1, behavior, fitness=0.5)
        archive.add(pod2, behavior, fitness=0.9)
        assert archive.size() == 1
        cell = archive.get_cell(behavior)
        assert cell.fitness == 0.9

    def test_fitness_rejection(self, archive, device):
        kernel = TorchKernel(device)
        pod1 = Pseudopod(head_dim=64, kernel=kernel, device=device)
        pod2 = Pseudopod(head_dim=64, kernel=kernel, device=device)
        behavior = (0.5, 0.5)
        archive.add(pod1, behavior, fitness=0.9)
        added = archive.add(pod2, behavior, fitness=0.5)
        assert added is False
        cell = archive.get_cell(behavior)
        assert cell.fitness == 0.9

    def test_grid_quantization(self, archive, pseudopod):
        behavior1 = (0.501, 0.502)
        behavior2 = (0.503, 0.504)
        archive.add(pseudopod, behavior1, fitness=0.8)
        cell = archive.get_cell(behavior2)
        assert cell is not None

    def test_boundary_behaviors(self, archive, device):
        kernel = TorchKernel(device)
        behaviors = [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]
        for i, behavior in enumerate(behaviors):
            pod = Pseudopod(head_dim=64, kernel=kernel, device=device)
            added = archive.add(pod, behavior, fitness=float(i))
            assert added is True
        assert archive.size() == 4

    def test_coverage_growth(self, archive, device):
        kernel = TorchKernel(device)
        assert archive.coverage() == 0.0
        for i in range(100):
            pod = Pseudopod(head_dim=64, kernel=kernel, device=device)
            behavior = (i / 100.0, (100 - i) / 100.0)
            archive.add(pod, behavior, fitness=0.5)
        coverage = archive.coverage()
        assert coverage > 0.01

    def test_elite_retrieval(self, archive, device):
        kernel = TorchKernel(device)
        fitnesses = [0.9, 0.5, 0.7, 0.3, 0.8]
        for i, fitness in enumerate(fitnesses):
            pod = Pseudopod(head_dim=64, kernel=kernel, device=device)
            behavior = (i * 0.2, i * 0.2)
            archive.add(pod, behavior, fitness=fitness)
        elites = archive.get_elites()
        assert len(elites) == 5
        assert max(e.fitness for e in elites) == 0.9
        assert min(e.fitness for e in elites) == 0.3

    def test_clear(self, archive, pseudopod):
        for i in range(10):
            behavior = (i * 0.1, i * 0.1)
            archive.add(pseudopod, behavior, fitness=0.5)
        assert archive.size() == 10
        archive.clear()
        assert archive.size() == 0

    def test_out_of_bounds_behavior(self, archive, pseudopod):
        behaviors = [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)]
        for behavior in behaviors:
            added = archive.add(pseudopod, behavior, fitness=0.5)
            assert added is True

    def test_dimension_mismatch(self, archive, pseudopod):
        with pytest.raises(ValueError):
            archive.add(pseudopod, (0.5,), fitness=0.5)
        with pytest.raises(ValueError):
            archive.add(pseudopod, (0.5, 0.5, 0.5), fitness=0.5)

    def test_empty_operations(self, archive):
        assert archive.size() == 0
        assert archive.coverage() == 0.0
        assert len(archive.get_elites()) == 0
        assert archive.get_cell((0.5, 0.5)) is None
