"""Unit tests for Chemotaxis (behavioral space navigation)"""

import pytest
import torch
from slime.core.chemotaxis import Chemotaxis
from slime.memory.archive import BehavioralArchive, ArchiveConfig


class TestChemotaxis:
    """Test chemotaxis behavioral space navigation"""

    @pytest.fixture
    def archive(self):
        """Create test archive"""
        config = ArchiveConfig(
            grid_size=[10, 10],
            dimensions=2,
            max_size=100,
        )
        return BehavioralArchive(config)

    @pytest.fixture
    def chemotaxis(self, archive):
        """Create test chemotaxis navigator"""
        return Chemotaxis(
            archive=archive,
            device=torch.device('cpu'),
        )

    def test_initialization(self, chemotaxis):
        """Test chemotaxis initializes correctly"""
        assert chemotaxis.archive is not None
        assert chemotaxis.device is not None

    def test_add_nutrient_source(self, chemotaxis):
        """Test adding nutrient source to behavioral space"""
        nutrient = torch.randn(64)
        location = (0.5, 0.5)
        concentration = 1.0

        chemotaxis.add_source(nutrient, location, concentration)

        # Verify source was added
        assert len(chemotaxis._sources) > 0

    def test_compute_gradient(self, chemotaxis):
        """Test computing chemotactic gradient"""
        # Add nutrient sources
        nutrient1 = torch.randn(64)
        nutrient2 = torch.randn(64)

        chemotaxis.add_source(nutrient1, (0.3, 0.3), concentration=1.0)
        chemotaxis.add_source(nutrient2, (0.7, 0.7), concentration=1.0)

        # Compute gradient at query location
        query_location = (0.5, 0.5)
        gradient = chemotaxis.compute_gradient(query_location)

        assert gradient is not None
        assert gradient.shape == (2,)  # 2D gradient (rank, coherence)

    def test_gradient_toward_high_concentration(self, chemotaxis):
        """Test gradient points toward high concentration"""
        # Add strong source at (0.8, 0.8)
        strong_nutrient = torch.randn(64)
        chemotaxis.add_source(strong_nutrient, (0.8, 0.8), concentration=10.0)

        # Add weak source at (0.2, 0.2)
        weak_nutrient = torch.randn(64)
        chemotaxis.add_source(weak_nutrient, (0.2, 0.2), concentration=1.0)

        # Query from center - gradient should point toward strong source
        query_location = (0.5, 0.5)
        gradient = chemotaxis.compute_gradient(query_location)

        # Gradient should have positive components (toward 0.8, 0.8 from 0.5, 0.5)
        assert gradient[0] > 0  # rank dimension
        assert gradient[1] > 0  # coherence dimension

    def test_find_nearest_elite(self, chemotaxis, archive):
        """Test finding nearest elite in archive"""
        # Add elites to archive
        for i in range(5):
            component = torch.randn(64, 128)
            behavior = torch.tensor([i * 0.2, i * 0.2])
            archive.add(component, behavior, fitness=1.0)

        # Find nearest to query
        query = (0.45, 0.45)
        nearest = chemotaxis.find_nearest_elite(query)

        assert nearest is not None
        # Should find elite near (0.4, 0.4) or (0.6, 0.6)

    def test_navigate_toward_nutrients(self, chemotaxis):
        """Test navigation step toward nutrients"""
        # Add nutrient source
        nutrient = torch.randn(64)
        target_location = (0.8, 0.8)
        chemotaxis.add_source(nutrient, target_location, concentration=5.0)

        # Start from origin
        current_location = (0.2, 0.2)
        step_size = 0.1

        # Take navigation step
        new_location = chemotaxis.navigate(current_location, step_size)

        assert new_location is not None
        assert len(new_location) == 2

        # Should move closer to target
        import math
        original_dist = math.sqrt(
            (target_location[0] - current_location[0])**2 +
            (target_location[1] - current_location[1])**2
        )
        new_dist = math.sqrt(
            (target_location[0] - new_location[0])**2 +
            (target_location[1] - new_location[1])**2
        )

        assert new_dist < original_dist

    def test_decay_concentrations(self, chemotaxis):
        """Test nutrient concentration decay"""
        nutrient = torch.randn(64)
        location = (0.5, 0.5)
        initial_concentration = 1.0

        chemotaxis.add_source(nutrient, location, initial_concentration)

        # Apply decay
        decay_rate = 0.5
        chemotaxis.decay_sources(decay_rate)

        # Concentration should decrease
        # (Implementation detail: verify in actual chemotaxis.py)

    def test_clear_sources(self, chemotaxis):
        """Test clearing all nutrient sources"""
        # Add sources
        for i in range(5):
            nutrient = torch.randn(64)
            location = (i * 0.2, i * 0.2)
            chemotaxis.add_source(nutrient, location, concentration=1.0)

        assert len(chemotaxis._sources) > 0

        chemotaxis.clear_sources()
        assert len(chemotaxis._sources) == 0

    def test_empty_archive_navigation(self, chemotaxis):
        """Test navigation with empty archive"""
        # No elites in archive
        query = (0.5, 0.5)
        nearest = chemotaxis.find_nearest_elite(query)

        assert nearest is None

    def test_no_sources_gradient(self, chemotaxis):
        """Test gradient computation with no sources"""
        query = (0.5, 0.5)
        gradient = chemotaxis.compute_gradient(query)

        # Should return zero gradient or None
        if gradient is not None:
            assert torch.allclose(gradient, torch.zeros_like(gradient))

    def test_boundary_navigation(self, chemotaxis):
        """Test navigation respects behavioral space boundaries [0, 1]"""
        # Add source outside typical range
        nutrient = torch.randn(64)
        chemotaxis.add_source(nutrient, (0.95, 0.95), concentration=5.0)

        # Start near boundary
        current_location = (0.9, 0.9)
        step_size = 0.2  # Large step

        new_location = chemotaxis.navigate(current_location, step_size)

        # Should not exceed [0, 1] bounds
        assert 0.0 <= new_location[0] <= 1.0
        assert 0.0 <= new_location[1] <= 1.0

    def test_device_consistency(self):
        """Test chemotaxis respects device placement"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            config = ArchiveConfig(grid_size=[10, 10], dimensions=2, max_size=100)
            archive = BehavioralArchive(config)
            chemotaxis = Chemotaxis(archive=archive, device=device)

            # Add source with CUDA tensor
            nutrient = torch.randn(64, device=device)
            chemotaxis.add_source(nutrient, (0.5, 0.5), concentration=1.0)

            # Gradient should be on correct device
            gradient = chemotaxis.compute_gradient((0.5, 0.5))
            if gradient is not None:
                assert gradient.device.type == 'cuda'
