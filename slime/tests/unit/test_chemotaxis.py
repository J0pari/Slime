import pytest
import torch
from slime.core.chemotaxis import Chemotaxis
from slime.memory.archive import BehavioralArchive
from slime.core.pseudopod import Pseudopod
from slime.kernels.torch_fallback import TorchKernel

class TestChemotaxis:

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
    def chemotaxis(self, archive, device):
        return Chemotaxis(archive=archive, device=device)

    def test_initialization(self, chemotaxis):
        assert chemotaxis.archive is not None
        assert chemotaxis.device is not None

    def test_add_nutrient_source(self, chemotaxis):
        nutrient = torch.randn(64)
        location = (0.5, 0.5)
        chemotaxis.add_source(nutrient, location, concentration=1.0)
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

    def test_navigate_toward_source(self, chemotaxis):
        nutrient = torch.randn(64)
        target_location = (0.8, 0.8)
        chemotaxis.add_source(nutrient, target_location, concentration=5.0)
        current_location = (0.2, 0.2)
        step_size = 0.1
        new_location = chemotaxis.navigate(current_location, step_size)
        assert new_location is not None
        assert len(new_location) == 2

    def test_clear_sources(self, chemotaxis):
        for i in range(5):
            nutrient = torch.randn(64)
            location = (i * 0.2, i * 0.2)
            chemotaxis.add_source(nutrient, location, concentration=1.0)
        assert len(chemotaxis._sources) > 0
        chemotaxis.clear_sources()
        assert len(chemotaxis._sources) == 0

    def test_boundary_navigation(self, chemotaxis):
        nutrient = torch.randn(64)
        chemotaxis.add_source(nutrient, (0.95, 0.95), concentration=5.0)
        current_location = (0.9, 0.9)
        step_size = 0.2
        new_location = chemotaxis.navigate(current_location, step_size)
        assert 0.0 <= new_location[0] <= 1.0
        assert 0.0 <= new_location[1] <= 1.0
