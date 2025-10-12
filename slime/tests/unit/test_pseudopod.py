import pytest
import torch
from slime.core.pseudopod import Pseudopod
from slime.core.state import FlowState

class TestPseudopod:

    @pytest.fixture
    def pseudopod(self):
        return Pseudopod(latent_dim=64, head_dim=32, component_id=0, device=torch.device('cpu'))

    def test_initialization(self, pseudopod):
        assert pseudopod.latent_dim == 64
        assert pseudopod.head_dim == 32
        assert pseudopod.component_id == 0

    def test_forward_pass(self, pseudopod):
        batch_size = 2
        seq_len = 10
        latent_dim = 64
        state = FlowState(batch_size=batch_size, sequence_length=seq_len, latent_dim=latent_dim, device=torch.device('cpu'))
        state.information = torch.randn(batch_size, seq_len, latent_dim)
        output = pseudopod(state)
        assert output.shape == (batch_size, seq_len, latent_dim)

    def test_query_key_value_shapes(self, pseudopod):
        batch_size = 2
        seq_len = 10
        latent_dim = 64
        state = FlowState(batch_size=batch_size, sequence_length=seq_len, latent_dim=latent_dim, device=torch.device('cpu'))
        state.information = torch.randn(batch_size, seq_len, latent_dim)
        Q, K, V = pseudopod._compute_qkv(state.information)
        assert Q.shape == (batch_size, seq_len, pseudopod.head_dim)
        assert K.shape == (batch_size, seq_len, pseudopod.head_dim)
        assert V.shape == (batch_size, seq_len, pseudopod.head_dim)

    def test_correlation_computation(self, pseudopod):
        batch_size = 2
        seq_len = 10
        latent_dim = 64
        state = FlowState(batch_size=batch_size, sequence_length=seq_len, latent_dim=latent_dim, device=torch.device('cpu'))
        state.information = torch.randn(batch_size, seq_len, latent_dim)
        pseudopod(state)
        assert pseudopod.correlation_matrix is not None
        assert pseudopod.correlation_matrix.shape == (seq_len, seq_len)
        assert (pseudopod.correlation_matrix >= -1.0).all()
        assert (pseudopod.correlation_matrix <= 1.0).all()

    def test_coherence_computation(self, pseudopod):
        batch_size = 2
        seq_len = 10
        latent_dim = 64
        state = FlowState(batch_size=batch_size, sequence_length=seq_len, latent_dim=latent_dim, device=torch.device('cpu'))
        state.information = torch.randn(batch_size, seq_len, latent_dim)
        pseudopod(state)
        assert pseudopod.coherence_score is not None
        assert isinstance(pseudopod.coherence_score, float)
        assert 0.0 <= pseudopod.coherence_score <= 1.0

    def test_behavioral_coordinates(self, pseudopod):
        batch_size = 2
        seq_len = 10
        latent_dim = 64
        state = FlowState(batch_size=batch_size, sequence_length=seq_len, latent_dim=latent_dim, device=torch.device('cpu'))
        state.information = torch.randn(batch_size, seq_len, latent_dim)
        pseudopod(state)
        coords = pseudopod.get_behavioral_coordinates()
        assert coords.shape == (2,)
        assert (coords >= 0.0).all()
        assert (coords <= 1.0).all()

    def test_parameter_count(self, pseudopod):
        num_params = sum((p.numel() for p in pseudopod.parameters()))
        assert num_params > 0
        num_trainable = sum((p.numel() for p in pseudopod.parameters() if p.requires_grad))
        assert num_trainable == num_params

    def test_gradient_flow(self, pseudopod):
        batch_size = 2
        seq_len = 10
        latent_dim = 64
        state = FlowState(batch_size=batch_size, sequence_length=seq_len, latent_dim=latent_dim, device=torch.device('cpu'))
        state.information = torch.randn(batch_size, seq_len, latent_dim, requires_grad=True)
        output = pseudopod(state)
        loss = output.sum()
        loss.backward()
        assert state.information.grad is not None
        for param in pseudopod.parameters():
            assert param.grad is not None

    def test_device_consistency(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            pseudopod = Pseudopod(latent_dim=64, head_dim=32, component_id=0, device=device)
            for param in pseudopod.parameters():
                assert param.device.type == 'cuda'

    def test_eval_mode(self, pseudopod):
        pseudopod.eval()
        batch_size = 2
        seq_len = 10
        latent_dim = 64
        state = FlowState(batch_size=batch_size, sequence_length=seq_len, latent_dim=latent_dim, device=torch.device('cpu'))
        state.information = torch.randn(batch_size, seq_len, latent_dim)
        with torch.no_grad():
            output = pseudopod(state)
        assert output.shape == (batch_size, seq_len, latent_dim)

    def test_multiple_forward_passes(self, pseudopod):
        batch_size = 2
        seq_len = 10
        latent_dim = 64
        state = FlowState(batch_size=batch_size, sequence_length=seq_len, latent_dim=latent_dim, device=torch.device('cpu'))
        state.information = torch.randn(batch_size, seq_len, latent_dim)
        pseudopod(state)
        first_coherence = pseudopod.coherence_score
        state.information = torch.randn(batch_size, seq_len, latent_dim)
        pseudopod(state)
        second_coherence = pseudopod.coherence_score
        assert first_coherence is not None
        assert second_coherence is not None
        assert first_coherence >= 0.0
        assert second_coherence >= 0.0