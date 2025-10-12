import torch
import torch.nn as nn
from typing import Protocol, TypeVar, Generic, Callable, List
from dataclasses import dataclass
T = TypeVar('T')
U = TypeVar('U')

class Comonad(Protocol[T]):

    def extract(self) -> T:
        pass

    def extend(self, f: Callable[['Comonad[T]'], U]) -> 'Comonad[U]':
        pass

    def duplicate(self) -> 'Comonad[Comonad[T]]':
        pass

@dataclass
class SpatialContext(Generic[T]):
    focus: T
    neighborhood: List[T]
    distance_fn: Callable[[T, T], float]

    def extract(self) -> T:
        return self.focus

    def extend(self, f: Callable[['SpatialContext[T]'], U]) -> 'SpatialContext[U]':
        new_focus = f(self)
        new_neighborhood = []
        for neighbor in self.neighborhood:
            neighbor_ctx = SpatialContext(focus=neighbor, neighborhood=self.neighborhood, distance_fn=self.distance_fn)
            new_neighborhood.append(f(neighbor_ctx))
        return SpatialContext(focus=new_focus, neighborhood=new_neighborhood, distance_fn=self.distance_fn)

    def duplicate(self) -> 'SpatialContext[SpatialContext[T]]':
        return self.extend(lambda ctx: ctx)

    def get_neighbors_within(self, radius: float) -> List[T]:
        neighbors = []
        for neighbor in self.neighborhood:
            if self.distance_fn(self.focus, neighbor) <= radius:
                neighbors.append(neighbor)
        return neighbors

    def get_k_nearest(self, k: int) -> List[T]:
        if not self.neighborhood:
            return []
        distances = [(self.distance_fn(self.focus, n), n) for n in self.neighborhood]
        distances.sort(key=lambda x: x[0])
        return [n for d, n in distances[:k]]

def behavioral_distance(c1, c2) -> float:
    if not hasattr(c1, 'last_behavior') or not hasattr(c2, 'last_behavior'):
        return float('inf')
    b1 = c1.last_behavior if hasattr(c1, 'last_behavior') else torch.zeros(5)
    b2 = c2.last_behavior if hasattr(c2, 'last_behavior') else torch.zeros(5)
    if isinstance(b1, torch.Tensor) and isinstance(b2, torch.Tensor):
        return torch.dist(b1, b2, p=2).item()
    return float('inf')

def extract_relative_fitness(ctx: SpatialContext) -> float:
    component = ctx.focus
    if not hasattr(component, 'fitness'):
        return 0.0
    my_fitness = component.fitness
    neighbors = ctx.get_k_nearest(k=5)
    if not neighbors:
        return my_fitness
    neighbor_fitnesses = []
    for n in neighbors:
        if hasattr(n, 'fitness'):
            neighbor_fitnesses.append(n.fitness)
    if not neighbor_fitnesses:
        return my_fitness
    mean_neighbor_fitness = sum(neighbor_fitnesses) / len(neighbor_fitnesses)
    std_neighbor_fitness = (sum(((f - mean_neighbor_fitness) ** 2 for f in neighbor_fitnesses)) / len(neighbor_fitnesses)) ** 0.5
    if std_neighbor_fitness < 1e-06:
        return 0.0
    relative_fitness = (my_fitness - mean_neighbor_fitness) / (std_neighbor_fitness + 1e-06)
    return relative_fitness

def extract_behavioral_divergence(ctx: SpatialContext) -> torch.Tensor:
    component = ctx.focus
    if not hasattr(component, 'last_behavior'):
        return torch.zeros(5)
    my_behavior = component.last_behavior
    neighbors = ctx.get_k_nearest(k=5)
    if not neighbors:
        return torch.zeros_like(my_behavior)
    neighbor_behaviors = []
    for n in neighbors:
        if hasattr(n, 'last_behavior'):
            neighbor_behaviors.append(n.last_behavior)
    if not neighbor_behaviors:
        return torch.zeros_like(my_behavior)
    neighbor_tensor = torch.stack(neighbor_behaviors)
    mean_neighbor_behavior = neighbor_tensor.mean(dim=0)
    divergence = my_behavior - mean_neighbor_behavior
    return divergence

def extract_gradient_magnitude_rank(ctx: SpatialContext) -> float:
    component = ctx.focus
    grad_norms = []
    for param in component.parameters():
        if param.grad is not None:
            grad_norms.append(torch.norm(param.grad).item())
    if not grad_norms:
        my_grad = 0.0
    else:
        my_grad = sum(grad_norms) / len(grad_norms)
    neighbors = ctx.get_k_nearest(k=10)
    if not neighbors:
        return 1.0
    neighbor_grads = []
    for n in neighbors:
        n_grad_norms = []
        for param in n.parameters():
            if param.grad is not None:
                n_grad_norms.append(torch.norm(param.grad).item())
        if n_grad_norms:
            neighbor_grads.append(sum(n_grad_norms) / len(n_grad_norms))
    if not neighbor_grads:
        return 1.0
    neighbor_grads.append(my_grad)
    neighbor_grads.sort()
    rank = neighbor_grads.index(my_grad)
    percentile = rank / len(neighbor_grads)
    return percentile

def extract_attention_coherence(ctx: SpatialContext) -> float:
    component = ctx.focus
    if not hasattr(component, '_last_attention_pattern'):
        return 0.0
    my_attn = component._last_attention_pattern
    neighbors = ctx.get_k_nearest(k=5)
    if not neighbors:
        return 0.0
    neighbor_attns = []
    for n in neighbors:
        if hasattr(n, '_last_attention_pattern'):
            neighbor_attns.append(n._last_attention_pattern)
    if not neighbor_attns:
        return 0.0
    coherence_scores = []
    for n_attn in neighbor_attns:
        if my_attn.shape == n_attn.shape:
            similarity = torch.cosine_similarity(my_attn.flatten(), n_attn.flatten(), dim=0).item()
            coherence_scores.append(similarity)
    if not coherence_scores:
        return 0.0
    return sum(coherence_scores) / len(coherence_scores)