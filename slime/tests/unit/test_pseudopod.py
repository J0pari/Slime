"""Unit tests for Pseudopod (individual component)"""

import pytest
import torch
from slime.core.pseudopod import Pseudopod
from slime.core.state import FlowState


class TestPseudopod:
    """Test individual pseudopod component"""

    @pytest.fixture
    def pseudopod(self):
        """Create test pseudopod"""
        return Pseudopod(
            latent_dim=64,
            head_dim=32,
            component_id=0,
            device=torch.device('cpu'),
        )

    def test_initialization(self, pseudopod):
        """Test pseudopod initializes correctly"""
        assert pseudopod.latent_dim == 64
        assert pseudopod.head_dim == 32
        assert pseudopod.component_id == 0

    def test_forward_pass(self, pseudopod):
        """Test forward pass produces correct output shape"""
        batch_size = 2
        seq_len = 10
        latent_dim = 64

        state = FlowState(
            batch_size=batch_size,
            sequence_length=seq_len,
            latent_dim=latent_dim,
            device=torch.device('cpu'),
        )
        state.information = torch.randn(batch_size, seq_len, latent_dim)

        output = pseudopod(state)

        assert output.shape == (batch_size, seq_len, latent_dim)

    def test_query_key_value_shapes(self, pseudopod):
        """Test Q, K, V projections have correct shapes"""
        batch_size = 2
        seq_len = 10
        latent_dim = 64

        state = FlowState(
            batch_size=batch_size,
            sequence_length=seq_len,
            latent_dim=latent_dim,
            device=torch.device('cpu'),
        )
        state.information = torch.randn(batch_size, seq_len, latent_dim)

        Q, K, V = pseudopod._compute_qkv(state.information)

        # Should have shape [batch, seq, head_dim]
        assert Q.shape == (batch_size, seq_len, pseudopod.head_dim)
        assert K.shape == (batch_size, seq_len, pseudopod.head_dim)
        assert V.shape == (batch_size, seq_len, pseudopod.head_dim)

    def test_correlation_computation(self, pseudopod):
        """Test correlation matrix computation"""
        batch_size = 2
        seq_len = 10
        latent_dim = 64

        state = FlowState(
            batch_size=batch_size,
            sequence_length=seq_len,
            latent_dim=latent_dim,
            device=torch.device('cpu'),
        )
        state.information = torch.randn(batch_size, seq_len, latent_dim)

        # Run forward to compute correlation
        pseudopod(state)

        assert pseudopod.correlation_matrix is not None
        # Correlation should be [seq, seq]
        assert pseudopod.correlation_matrix.shape == (seq_len, seq_len)

        # Correlation values should be in [-1, 1]
        assert (pseudopod.correlation_matrix >= -1.0).all()
        assert (pseudopod.correlation_matrix <= 1.0).all()

    def test_coherence_computation(self, pseudopod):
        """Test coherence score computation"""
        batch_size = 2
        seq_len = 10
        latent_dim = 64

        state = FlowState(
            batch_size=batch_size,
            sequence_length=seq_len,
            latent_dim=latent_dim,
            device=torch.device('cpu'),
        )
        state.information = torch.randn(batch_size, seq_len, latent_dim)

        # Run forward to compute coherence
        pseudopod(state)

        assert pseudopod.coherence_score is not None
        assert isinstance(pseudopod.coherence_score, float)
        assert 0.0 <= pseudopod.coherence_score <= 1.0

    def test_behavioral_coordinates(self, pseudopod):
        """Test behavioral space coordinates (rank, coherence)"""
        batch_size = 2
        seq_len = 10
        latent_dim = 64

        state = FlowState(
            batch_size=batch_size,
            sequence_length=seq_len,
            latent_dim=latent_dim,
            device=torch.device('cpu'),
        )
        state.information = torch.randn(batch_size, seq_len, latent_dim)

        # Run forward
        pseudopod(state)

        coords = pseudopod.get_behavioral_coordinates()

        assert coords.shape == (2,)  # [rank, coherence]
        assert (coords >= 0.0).all()
        assert (coords <= 1.0).all()

    def test_parameter_count(self, pseudopod):
        """Test pseudopod has trainable parameters"""
        num_params = sum(p.numel() for p in pseudopod.parameters())
        assert num_params > 0

        num_trainable = sum(p.numel() for p in pseudopod.parameters() if p.requires_grad)
        assert num_trainable == num_params

    def test_gradient_flow(self, pseudopod):
        """Test gradients flow through pseudopod"""
        batch_size = 2
        seq_len = 10
        latent_dim = 64

        state = FlowState(
            batch_size=batch_size,
            sequence_length=seq_len,
            latent_dim=latent_dim,
            device=torch.device('cpu'),
        )
        state.information = torch.randn(batch_size, seq_len, latent_dim, requires_grad=True)

        output = pseudopod(state)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert state.information.grad is not None
        for param in pseudopod.parameters():
            assert param.grad is not None

    def test_device_consistency(self):
        """Test pseudopod respects device placement"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            pseudopod = Pseudopod(
                latent_dim=64,
                head_dim=32,
                component_id=0,
                device=device,
            )

            # Check parameters are on correct device
            for param in pseudopod.parameters():
                assert param.device.type == 'cuda'

    def test_eval_mode(self, pseudopod):
        """Test pseudopod in eval mode"""
        pseudopod.eval()

        batch_size = 2
        seq_len = 10
        latent_dim = 64

        state = FlowState(
            batch_size=batch_size,
            sequence_length=seq_len,
            latent_dim=latent_dim,
            device=torch.device('cpu'),
        )
        state.information = torch.randn(batch_size, seq_len, latent_dim)

        with torch.no_grad():
            output = pseudopod(state)

        assert output.shape == (batch_size, seq_len, latent_dim)

    def test_multiple_forward_passes(self, pseudopod):
        """Test multiple forward passes update metrics"""
        batch_size = 2
        seq_len = 10
        latent_dim = 64

        state = FlowState(
            batch_size=batch_size,
            sequence_length=seq_len,
            latent_dim=latent_dim,
            device=torch.device('cpu'),
        )

        # First pass
        state.information = torch.randn(batch_size, seq_len, latent_dim)
        pseudopod(state)
        first_coherence = pseudopod.coherence_score

        # Second pass
        state.information = torch.randn(batch_size, seq_len, latent_dim)
        pseudopod(state)
        second_coherence = pseudopod.coherence_score

        # Metrics should be updated
        assert first_coherence is not None
        assert second_coherence is not None
        # Values may differ
        assert first_coherence >= 0.0
        assert second_coherence >= 0.0
