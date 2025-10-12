import pytest
import torch
from slime.memory.tubes import InformationTubes, TubeConfig
from slime.core.state import FlowState

class TestInformationTubes:

    @pytest.fixture
    def tube_config(self):
        return TubeConfig(max_capacity=100, decay_rate=0.1)

    @pytest.fixture
    def tubes(self, tube_config):
        return InformationTubes(config=tube_config)

    def test_initialization(self, tubes):
        assert tubes.size() == 0

    def test_add_flow(self, tubes):
        state = FlowState(batch_size=2, sequence_length=10, latent_dim=64, device=torch.device('cpu'))
        state.information = torch.randn(2, 10, 64)
        tubes.add(state)
        assert tubes.size() == 1

    def test_get_flow(self, tubes):
        state = FlowState(batch_size=2, sequence_length=10, latent_dim=64, device=torch.device('cpu'))
        state.information = torch.randn(2, 10, 64)
        tubes.add(state)
        retrieved = tubes.get_latest()
        assert retrieved is not None
        assert torch.equal(retrieved.information, state.information)

    def test_capacity_enforcement(self):
        config = TubeConfig(max_capacity=5)
        tubes = InformationTubes(config=config)
        for i in range(10):
            state = FlowState(batch_size=2, sequence_length=10, latent_dim=64, device=torch.device('cpu'))
            state.information = torch.randn(2, 10, 64)
            tubes.add(state)
        assert tubes.size() <= config.max_capacity

    def test_clear(self, tubes):
        for _ in range(5):
            state = FlowState(batch_size=2, sequence_length=10, latent_dim=64, device=torch.device('cpu'))
            state.information = torch.randn(2, 10, 64)
            tubes.add(state)
        assert tubes.size() == 5
        tubes.clear()
        assert tubes.size() == 0

    def test_flow_history(self, tubes):
        for i in range(3):
            state = FlowState(batch_size=2, sequence_length=10, latent_dim=64, device=torch.device('cpu'))
            state.information = torch.randn(2, 10, 64) * i
            tubes.add(state)
        history = tubes.get_history(k=3)
        assert len(history) == 3
        assert history[0] is tubes.get_latest()

    def test_empty_tubes_operations(self, tubes):
        assert tubes.size() == 0
        assert tubes.get_latest() is None
        assert len(tubes.get_history(k=5)) == 0