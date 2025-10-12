import pytest
import torch
from slime.core.chemotaxis import Chemotaxis
from slime.memory.archive import BehavioralArchive, ArchiveConfig

class TestChemotaxis:

    @pytest.fixture
    def archive(self):
        config = ArchiveConfig(grid_size=[10, 10], dimensions=2, max_size=100)
        return BehavioralArchive(config)

    @pytest.fixture
    def chemotaxis(self, archive):
        return Chemotaxis(archive=archive, device=torch.device('cpu'))

    def test_initialization(self, chemotaxis):
        assert chemotaxis.archive is not None
        assert chemotaxis.device is not None

    def test_add_nutrient_source(self, chemotaxis):
        nutrient = torch.randn(64)
        location = (0.5, 0.5)
        concentration = 1.0
        chemotaxis.add_source(nutrient, location, concentration)
        assert len(chemotaxis._sources) > 0

    def test_compute_gradient(self, chemotaxis):
        nutrient1 = torch.randn(64)
        nutrient2 = torch.randn(64)
        chemotaxis.add_source(nutrient1, (0.3, 0.3), concentration=1.0)
        chemotaxis.add_source(nutrient2, (0.7, 0.7), concentration=1.0)
        query_location = (0.5, 0.5)
        gradient = chemotaxis.compute_gradient(query_location)
        assert gradient is not None
        assert gradient.shape == (2,)

    def test_gradient_toward_high_concentration(self, chemotaxis):
        strong_nutrient = torch.randn(64)
        chemotaxis.add_source(strong_nutrient, (0.8, 0.8), concentration=10.0)
        weak_nutrient = torch.randn(64)
        chemotaxis.add_source(weak_nutrient, (0.2, 0.2), concentration=1.0)
        query_location = (0.5, 0.5)
        gradient = chemotaxis.compute_gradient(query_location)
        assert gradient[0] > 0
        assert gradient[1] > 0

    def test_find_nearest_elite(self, chemotaxis, archive):
        for i in range(5):
            component = torch.randn(64, 128)
            behavior = torch.tensor([i * 0.2, i * 0.2])
            archive.add(component, behavior, fitness=1.0)
        query = (0.45, 0.45)
        nearest = chemotaxis.find_nearest_elite(query)
        assert nearest is not None

    def test_navigate_toward_nutrients(self, chemotaxis):
        nutrient = torch.randn(64)
        target_location = (0.8, 0.8)
        chemotaxis.add_source(nutrient, target_location, concentration=5.0)
        current_location = (0.2, 0.2)
        step_size = 0.1
        new_location = chemotaxis.navigate(current_location, step_size)
        assert new_location is not None
        assert len(new_location) == 2
        import math
        original_dist = math.sqrt((target_location[0] - current_location[0]) ** 2 + (target_location[1] - current_location[1]) ** 2)
        new_dist = math.sqrt((target_location[0] - new_location[0]) ** 2 + (target_location[1] - new_location[1]) ** 2)
        assert new_dist < original_dist

    def test_decay_concentrations(self, chemotaxis):
        nutrient = torch.randn(64)
        location = (0.5, 0.5)
        initial_concentration = 1.0
        chemotaxis.add_source(nutrient, location, initial_concentration)
        decay_rate = 0.5
        chemotaxis.decay_sources(decay_rate)

    def test_clear_sources(self, chemotaxis):
        for i in range(5):
            nutrient = torch.randn(64)
            location = (i * 0.2, i * 0.2)
            chemotaxis.add_source(nutrient, location, concentration=1.0)
        assert len(chemotaxis._sources) > 0
        chemotaxis.clear_sources()
        assert len(chemotaxis._sources) == 0

    def test_empty_archive_navigation(self, chemotaxis):
        query = (0.5, 0.5)
        nearest = chemotaxis.find_nearest_elite(query)
        assert nearest is None

    def test_no_sources_gradient(self, chemotaxis):
        query = (0.5, 0.5)
        gradient = chemotaxis.compute_gradient(query)
        if gradient is not None:
            assert torch.allclose(gradient, torch.zeros_like(gradient))

    def test_boundary_navigation(self, chemotaxis):
        nutrient = torch.randn(64)
        chemotaxis.add_source(nutrient, (0.95, 0.95), concentration=5.0)
        current_location = (0.9, 0.9)
        step_size = 0.2
        new_location = chemotaxis.navigate(current_location, step_size)
        assert 0.0 <= new_location[0] <= 1.0
        assert 0.0 <= new_location[1] <= 1.0

    def test_device_consistency(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            config = ArchiveConfig(grid_size=[10, 10], dimensions=2, max_size=100)
            archive = BehavioralArchive(config)
            chemotaxis = Chemotaxis(archive=archive, device=device)
            nutrient = torch.randn(64, device=device)
            chemotaxis.add_source(nutrient, (0.5, 0.5), concentration=1.0)
            gradient = chemotaxis.compute_gradient((0.5, 0.5))
            if gradient is not None:
                assert gradient.device.type == 'cuda'