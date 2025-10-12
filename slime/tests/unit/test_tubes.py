import pytest
import torch
from slime.memory.tubes import TubeNetwork
from slime.proto.memory import Memory

class TestTubeNetwork:

    @pytest.fixture
    def tubes(self):
        return TubeNetwork(capacity=100, decay_rate=0.1)

    def test_initialization(self, tubes):
        assert len(tubes._storage) == 0

    def test_store_retrieve(self, tubes):
        data = torch.randn(64, 128)
        tubes.store('test_key', data)
        retrieved = tubes.recall('test_key')
        assert retrieved is not None
        assert torch.allclose(retrieved, data, atol=1e-4)

    def test_capacity_enforcement(self):
        tubes = TubeNetwork(capacity=5, decay_rate=0.0)
        for i in range(10):
            tubes.store(f'key_{i}', torch.randn(10))
        assert len(tubes._storage) <= 5

    def test_decay(self):
        tubes = TubeNetwork(capacity=100, decay_rate=0.5)
        data = torch.ones(10)
        tubes.store('key', data)
        tubes._apply_decay()
        retrieved = tubes.recall('key')
        assert retrieved is not None
        assert (retrieved < data).all()

    def test_nonexistent_key(self, tubes):
        retrieved = tubes.recall('nonexistent')
        assert retrieved is None

    def test_clear(self, tubes):
        for i in range(5):
            tubes.store(f'key_{i}', torch.randn(10))
        assert len(tubes._storage) > 0
        tubes.clear()
        assert len(tubes._storage) == 0
