"""Unit tests for BehavioralArchive (MAP-Elites)"""

import pytest
import torch
from slime.memory.archive import BehavioralArchive, ArchiveConfig


class TestBehavioralArchive:
    """Test MAP-Elites behavioral archive"""

    @pytest.fixture
    def archive(self):
        """Create test archive"""
        config = ArchiveConfig(
            grid_size=[10, 10],
            dimensions=2,
            max_size=100,
        )
        return BehavioralArchive(config)

    def test_initialization(self, archive):
        """Test archive initializes correctly"""
        assert archive.size() == 0
        assert archive.config.dimensions == 2
        assert archive.config.grid_size == [10, 10]

    def test_add_elite(self, archive):
        """Test adding elite to archive"""
        component = torch.randn(64, 128)
        behavior = torch.tensor([0.5, 0.5])
        fitness = 1.0

        added = archive.add(component, behavior, fitness)
        assert added is True
        assert archive.size() == 1

    def test_replace_lower_fitness(self, archive):
        """Test replacing elite with higher fitness"""
        component1 = torch.randn(64, 128)
        component2 = torch.randn(64, 128)
        behavior = torch.tensor([0.5, 0.5])

        # Add first elite
        archive.add(component1, behavior, fitness=1.0)
        assert archive.size() == 1

        # Add second elite with higher fitness - should replace
        archive.add(component2, behavior, fitness=2.0)
        assert archive.size() == 1

        # Verify higher fitness elite is stored
        cell = archive.get_cell(behavior)
        assert cell is not None
        assert cell.fitness == 2.0

    def test_keep_higher_fitness(self, archive):
        """Test keeping elite with higher fitness"""
        component1 = torch.randn(64, 128)
        component2 = torch.randn(64, 128)
        behavior = torch.tensor([0.5, 0.5])

        # Add first elite with high fitness
        archive.add(component1, behavior, fitness=2.0)

        # Try to add second elite with lower fitness - should reject
        added = archive.add(component2, behavior, fitness=1.0)
        assert added is False
        assert archive.size() == 1

        # Verify high fitness elite is still stored
        cell = archive.get_cell(behavior)
        assert cell.fitness == 2.0

    def test_different_cells(self, archive):
        """Test adding elites to different cells"""
        component1 = torch.randn(64, 128)
        component2 = torch.randn(64, 128)
        behavior1 = torch.tensor([0.2, 0.3])
        behavior2 = torch.tensor([0.7, 0.8])

        archive.add(component1, behavior1, fitness=1.0)
        archive.add(component2, behavior2, fitness=1.0)

        assert archive.size() == 2

    def test_get_neighbors(self, archive):
        """Test finding neighboring elites"""
        # Add elites in a pattern
        for i in range(5):
            component = torch.randn(64, 128)
            behavior = torch.tensor([0.5 + i * 0.05, 0.5])
            archive.add(component, behavior, fitness=1.0)

        # Get neighbors of center cell
        query_behavior = torch.tensor([0.5, 0.5])
        neighbors = archive.get_neighbors(query_behavior, k=3)

        assert len(neighbors) <= 3
        assert all(neighbor.fitness > 0 for neighbor in neighbors)

    def test_coverage(self, archive):
        """Test behavioral space coverage metric"""
        # Empty archive
        assert archive.coverage() == 0.0

        # Add some elites
        for i in range(10):
            component = torch.randn(64, 128)
            behavior = torch.tensor([i * 0.1, i * 0.1])
            archive.add(component, behavior, fitness=1.0)

        coverage = archive.coverage()
        assert 0.0 < coverage <= 1.0

    def test_max_size_enforcement(self):
        """Test archive respects max size"""
        config = ArchiveConfig(
            grid_size=[5, 5],
            dimensions=2,
            max_size=10,
        )
        archive = BehavioralArchive(config)

        # Try to add more than max_size
        for i in range(25):  # 5x5 grid
            component = torch.randn(64, 128)
            row = i // 5
            col = i % 5
            behavior = torch.tensor([row * 0.2, col * 0.2])
            archive.add(component, behavior, fitness=float(i))

        # Should be capped at max_size
        assert archive.size() <= config.max_size

    def test_get_elites(self, archive):
        """Test retrieving all elites"""
        # Add some elites
        for i in range(5):
            component = torch.randn(64, 128)
            behavior = torch.tensor([i * 0.2, i * 0.2])
            archive.add(component, behavior, fitness=float(i))

        elites = archive.get_elites()
        assert len(elites) == 5
        assert all(elite.fitness >= 0 for elite in elites)

    def test_clear(self, archive):
        """Test clearing archive"""
        # Add elites
        for i in range(5):
            component = torch.randn(64, 128)
            behavior = torch.tensor([i * 0.2, i * 0.2])
            archive.add(component, behavior, fitness=1.0)

        assert archive.size() == 5

        archive.clear()
        assert archive.size() == 0

    def test_behavior_discretization(self, archive):
        """Test behavioral coordinates are discretized correctly"""
        component = torch.randn(64, 128)

        # Add elite with continuous behavior
        behavior_continuous = torch.tensor([0.23, 0.67])
        archive.add(component, behavior_continuous, fitness=1.0)

        # Try to get with slightly different behavior in same cell
        behavior_similar = torch.tensor([0.24, 0.68])
        cell = archive.get_cell(behavior_similar)

        # Should retrieve the same cell
        assert cell is not None
        assert cell.fitness == 1.0

    def test_empty_archive_operations(self, archive):
        """Test operations on empty archive"""
        assert archive.size() == 0
        assert archive.coverage() == 0.0
        assert len(archive.get_elites()) == 0

        behavior = torch.tensor([0.5, 0.5])
        assert archive.get_cell(behavior) is None
        assert len(archive.get_neighbors(behavior, k=5)) == 0
