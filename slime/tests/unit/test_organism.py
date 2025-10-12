import pytest
import torch
import numpy as np
from slime.core.organism import Organism
from slime.config.dimensions import TINY, SMALL, MEDIUM
from slime.memory.pool import PoolConfig
from slime.core.state import FlowState

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def config():
    return TINY

def test_organism_initialization(config, device, constraint):
    organism = Organism(
        sensory_dim=128,
        latent_dim=256,
        head_dim=config.dimensions.head_dim,
        arch_config=config,
        device=device
    )
    constraint('Organism initialized successfully', lambda: (organism is not None, type(organism).__name__, 'Organism', {}))
    constraint('Archive uses config dimensions', lambda: (organism.archive.num_raw_metrics == config.behavioral_space.num_raw_metrics, organism.archive.num_raw_metrics, config.behavioral_space.num_raw_metrics, {}))
    constraint('Archive min_dims from config', lambda: (organism.archive.min_dims == config.behavioral_space.min_dims, organism.archive.min_dims, config.behavioral_space.min_dims, {}))
    constraint('Archive max_dims from config', lambda: (organism.archive.max_dims == config.behavioral_space.max_dims, organism.archive.max_dims, config.behavioral_space.max_dims, {}))
    constraint('Archive num_centroids from config', lambda: (organism.archive.num_centroids == config.behavioral_space.num_centroids, organism.archive.num_centroids, config.behavioral_space.num_centroids, {}))
    constraint('Archive low_rank_k from config', lambda: (organism.archive.low_rank_k == config.compression.low_rank_k, organism.archive.low_rank_k, config.compression.low_rank_k, {}))
    constraint('Pool stencil k_neighbors from config', lambda: (organism.pseudopod_pool.stencil.k_neighbors == config.k_neighbors, organism.pseudopod_pool.stencil.k_neighbors, config.k_neighbors, {}))

def test_organism_forward_pass(config, device, constraint):
    organism = Organism(
        sensory_dim=128,
        latent_dim=256,
        head_dim=config.dimensions.head_dim,
        arch_config=config,
        device=device
    )
    batch_size = 2
    seq_len = 32
    stimulus = torch.randn(batch_size, 128, device=device)
    output, state = organism(stimulus)
    constraint('Output has correct batch size', lambda: (output.shape[0] == batch_size, output.shape[0], batch_size, {}))
    constraint('Output has sequence dimension', lambda: (output.dim() == 3, output.dim(), 3, {}))
    constraint('Output has correct feature dim', lambda: (output.shape[2] == 128, output.shape[2], 128, {}))
    constraint('State returned', lambda: (state is not None, type(state).__name__, 'FlowState', {}))
    constraint('Output is on correct device', lambda: (output.device.type == device.type, output.device.type, device.type, {}))

def test_organism_with_pool_config(config, device, constraint):
    pool_config = PoolConfig(min_size=2, max_size=16, birth_threshold=0.8, death_threshold=0.1)
    organism = Organism(
        sensory_dim=64,
        latent_dim=128,
        head_dim=config.dimensions.head_dim,
        arch_config=config,
        pool_config=pool_config,
        device=device
    )
    constraint('Pool respects min_size', lambda: (len(organism.pseudopod_pool._components) == 2, len(organism.pseudopod_pool._components), 2, {}))
    constraint('Pool config max_size set', lambda: (organism.pseudopod_pool.config.max_size == 16, organism.pseudopod_pool.config.max_size, 16, {}))

def test_organism_deterministic_initialization(config, device, constraint):
    torch.manual_seed(42)
    organism1 = Organism(
        sensory_dim=64,
        latent_dim=128,
        head_dim=config.dimensions.head_dim,
        arch_config=config,
        device=device
    )
    torch.manual_seed(42)
    organism2 = Organism(
        sensory_dim=64,
        latent_dim=128,
        head_dim=config.dimensions.head_dim,
        arch_config=config,
        device=device
    )
    for (name1, param1), (name2, param2) in zip(organism1.named_parameters(), organism2.named_parameters()):
        constraint(f'Parameter {name1} deterministic', lambda p1=param1, p2=param2: (torch.allclose(p1, p2), 'match', 'match', {}))

def test_organism_multiple_forward_passes(config, device, constraint):
    organism = Organism(
        sensory_dim=128,
        latent_dim=256,
        head_dim=config.dimensions.head_dim,
        arch_config=config,
        device=device
    )
    stimulus = torch.randn(2, 128, device=device)
    output1, state1 = organism(stimulus)
    output2, state2 = organism(stimulus, state=state1)
    constraint('Second forward pass succeeds with state', lambda: (output2 is not None, type(output2).__name__, 'Tensor', {}))
    constraint('State evolves between passes', lambda: (state2 is not None and state2 is not state1, True, True, {}))

def test_organism_config_variants(device, constraint):
    for config_name, config in [('TINY', TINY), ('SMALL', SMALL), ('MEDIUM', MEDIUM)]:
        organism = Organism(
            sensory_dim=128,
            latent_dim=256,
            head_dim=config.dimensions.head_dim,
            arch_config=config,
            device=device
        )
        constraint(f'{config_name} config initializes', lambda o=organism: (o is not None, type(o).__name__, 'Organism', {}))
        constraint(f'{config_name} archive has correct centroids', lambda o=organism, c=config: (o.archive.num_centroids == c.behavioral_space.num_centroids, o.archive.num_centroids, c.behavioral_space.num_centroids, {}))

def test_organism_gradient_flow(config, device, constraint):
    organism = Organism(
        sensory_dim=64,
        latent_dim=128,
        head_dim=config.dimensions.head_dim,
        arch_config=config,
        device=device
    )
    stimulus = torch.randn(2, 64, device=device, requires_grad=True)
    output, _ = organism(stimulus)
    loss = output.sum()
    loss.backward()
    constraint('Input gradients computed', lambda: (stimulus.grad is not None, stimulus.grad is not None, True, {}))
    has_grad = False
    for param in organism.parameters():
        if param.grad is not None:
            has_grad = True
            break
    constraint('Model parameters have gradients', lambda: (has_grad, has_grad, True, {}))
