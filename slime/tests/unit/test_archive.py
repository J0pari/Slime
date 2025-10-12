import pytest
import torch
from slime.memory.archive import BehavioralArchive, ArchiveConfig

class TestBehavioralArchive:

    @pytest.fixture
    def archive(self):
        config = ArchiveConfig(grid_size=[10, 10], dimensions=2, max_size=100)
        return BehavioralArchive(config)

    def test_initialization(self, archive):
        assert archive.size() == 0
        assert archive.config.dimensions == 2
        assert archive.config.grid_size == [10, 10]

    def test_add_elite(self, archive):
        component = torch.randn(64, 128)
        behavior = torch.tensor([0.5, 0.5])
        fitness = 1.0
        added = archive.add(component, behavior, fitness)
        assert added is True
        assert archive.size() == 1

    def test_replace_lower_fitness(self, archive):
        component1 = torch.randn(64, 128)
        component2 = torch.randn(64, 128)
        behavior = torch.tensor([0.5, 0.5])
        archive.add(component1, behavior, fitness=1.0)
        assert archive.size() == 1
        archive.add(component2, behavior, fitness=2.0)
        assert archive.size() == 1
        cell = archive.get_cell(behavior)
        assert cell is not None
        assert cell.fitness == 2.0

    def test_keep_higher_fitness(self, archive):
        component1 = torch.randn(64, 128)
        component2 = torch.randn(64, 128)
        behavior = torch.tensor([0.5, 0.5])
        archive.add(component1, behavior, fitness=2.0)
        added = archive.add(component2, behavior, fitness=1.0)
        assert added is False
        assert archive.size() == 1
        cell = archive.get_cell(behavior)
        assert cell.fitness == 2.0

    def test_different_cells(self, archive):
        component1 = torch.randn(64, 128)
        component2 = torch.randn(64, 128)
        behavior1 = torch.tensor([0.2, 0.3])
        behavior2 = torch.tensor([0.7, 0.8])
        archive.add(component1, behavior1, fitness=1.0)
        archive.add(component2, behavior2, fitness=1.0)
        assert archive.size() == 2

    def test_get_neighbors(self, archive):
        for i in range(5):
            component = torch.randn(64, 128)
            behavior = torch.tensor([0.5 + i * 0.05, 0.5])
            archive.add(component, behavior, fitness=1.0)
        query_behavior = torch.tensor([0.5, 0.5])
        neighbors = archive.get_neighbors(query_behavior, k=3)
        assert len(neighbors) <= 3
        assert all((neighbor.fitness > 0 for neighbor in neighbors))

    def test_coverage(self, archive):
        assert archive.coverage() == 0.0
        for i in range(10):
            component = torch.randn(64, 128)
            behavior = torch.tensor([i * 0.1, i * 0.1])
            archive.add(component, behavior, fitness=1.0)
        coverage = archive.coverage()
        assert 0.0 < coverage <= 1.0

    def test_max_size_enforcement(self):
        config = ArchiveConfig(grid_size=[5, 5], dimensions=2, max_size=10)
        archive = BehavioralArchive(config)
        for i in range(25):
            component = torch.randn(64, 128)
            row = i // 5
            col = i % 5
            behavior = torch.tensor([row * 0.2, col * 0.2])
            archive.add(component, behavior, fitness=float(i))
        assert archive.size() <= config.max_size

    def test_get_elites(self, archive):
        for i in range(5):
            component = torch.randn(64, 128)
            behavior = torch.tensor([i * 0.2, i * 0.2])
            archive.add(component, behavior, fitness=float(i))
        elites = archive.get_elites()
        assert len(elites) == 5
        assert all((elite.fitness >= 0 for elite in elites))

    def test_clear(self, archive):
        for i in range(5):
            component = torch.randn(64, 128)
            behavior = torch.tensor([i * 0.2, i * 0.2])
            archive.add(component, behavior, fitness=1.0)
        assert archive.size() == 5
        archive.clear()
        assert archive.size() == 0

    def test_behavior_discretization(self, archive):
        component = torch.randn(64, 128)
        behavior_continuous = torch.tensor([0.23, 0.67])
        archive.add(component, behavior_continuous, fitness=1.0)
        behavior_similar = torch.tensor([0.24, 0.68])
        cell = archive.get_cell(behavior_similar)
        assert cell is not None
        assert cell.fitness == 1.0

    def test_empty_archive_operations(self, archive):
        assert archive.size() == 0
        assert archive.coverage() == 0.0
        assert len(archive.get_elites()) == 0
        behavior = torch.tensor([0.5, 0.5])
        assert archive.get_cell(behavior) is None
        assert len(archive.get_neighbors(behavior, k=5)) == 0