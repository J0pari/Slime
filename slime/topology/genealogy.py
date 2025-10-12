"""
Pseudopod phylogenetic tree tracking.

Maintains parent-child relationships for pseudopod spawning and
computes ultrametric distances based on most recent common ancestor (MRCA).

Key Concepts:
- Lineage: Path from root to node (tuple of ancestor IDs)
- MRCA: Most recent common ancestor (deepest shared node)
- Ultrametric: Distance via MRCA depth (tree metric)
- Phylogenetic diversity: Average pairwise distance across lineages

Applications:
- Diversity regularization (preserve multiple lineages)
- Novelty search (explore distant regions of genealogy tree)
- Quality-diversity (balance fitness and lineage distance)

References:
- Faith, "Conservation evaluation and phylogenetic diversity" (1992)
- Lehman & Stanley, "Abandoning objectives" (2011) - Novelty search
"""

from typing import Dict, Tuple, List, Set
import logging

logger = logging.getLogger(__name__)


class Genealogy:
    """
    Tracks pseudopod spawning lineages with ultrametric distances.

    Maintains a phylogenetic tree where:
    - Nodes = pseudopods (identified by ID)
    - Edges = spawning relationships (parent → child)
    - Distance = depth to MRCA (ultrametric)

    Thread-safe: No shared mutable state (all operations return new values).
    """

    def __init__(self) -> None:
        """Initialize empty genealogy with no pseudopods."""
        self._lineages: Dict[int, Tuple[int, ...]] = {}  # pod_id → (ancestor_path)
        self._children: Dict[int, List[int]] = {}  # parent_id → [child_ids]
        self._generation: Dict[int, int] = {}  # pod_id → generation_number

    def register_genesis(self, pod_id: int) -> None:
        """
        Register a genesis pseudopod (no parent).

        Genesis pseudopods are roots of phylogenetic trees.
        Multiple genesis pods create a forest.

        Args:
            pod_id: Unique identifier for pseudopod

        Example:
            >>> gen = Genealogy()
            >>> gen.register_genesis(0)
            >>> gen.get_lineage(0)
            (0,)  # Single-node lineage
        """
        self._lineages[pod_id] = (pod_id,)
        self._children[pod_id] = []
        self._generation[pod_id] = 0
        logger.debug(f"Registered genesis pseudopod {pod_id}")

    def register_spawn(self, parent_id: int, child_id: int) -> None:
        """
        Register child spawned from parent.

        Child inherits parent's lineage plus its own ID.

        Args:
            parent_id: ID of parent pseudopod
            child_id: ID of newly spawned child

        Raises:
            ValueError: Parent not registered in genealogy

        Example:
            >>> gen = Genealogy()
            >>> gen.register_genesis(0)
            >>> gen.register_spawn(0, 1)
            >>> gen.get_lineage(1)
            (0, 1)  # Child inherits parent's lineage
        """
        if parent_id not in self._lineages:
            raise ValueError(
                f"Parent {parent_id} not in genealogy. "
                f"Call register_genesis() or register_spawn() first."
            )

        parent_lineage = self._lineages[parent_id]
        self._lineages[child_id] = parent_lineage + (child_id,)

        if child_id not in self._children:
            self._children[child_id] = []

        if parent_id in self._children:
            self._children[parent_id].append(child_id)
        else:
            self._children[parent_id] = [child_id]

        self._generation[child_id] = self._generation[parent_id] + 1

        logger.debug(
            f"Spawned pseudopod {child_id} from {parent_id} "
            f"(generation {self._generation[child_id]})"
        )

    def get_lineage(self, pod_id: int) -> Tuple[int, ...]:
        """
        Get full lineage path from root to pseudopod.

        Args:
            pod_id: Pseudopod identifier

        Returns:
            Tuple of ancestor IDs from root to pod (inclusive)

        Example:
            >>> gen = Genealogy()
            >>> gen.register_genesis(0)
            >>> gen.register_spawn(0, 3)
            >>> gen.register_spawn(3, 7)
            >>> gen.get_lineage(7)
            (0, 3, 7)
        """
        return self._lineages.get(pod_id, ())

    def get_generation(self, pod_id: int) -> int:
        """
        Get generation number (depth in tree).

        Args:
            pod_id: Pseudopod identifier

        Returns:
            Generation number (0 for genesis, 1 for children, ...)
        """
        return self._generation.get(pod_id, -1)

    def get_children(self, pod_id: int) -> List[int]:
        """
        Get all direct children of pseudopod.

        Args:
            pod_id: Parent identifier

        Returns:
            List of child IDs (empty if no children)
        """
        return self._children.get(pod_id, []).copy()

    def ultrametric_distance(self, pod_a: int, pod_b: int) -> float:
        """
        Compute ultrametric distance via MRCA depth.

        Distance = len(lineage_a) + len(lineage_b) - 2*depth(MRCA)

        Args:
            pod_a: First pseudopod ID
            pod_b: Second pseudopod ID

        Returns:
            Ultrametric distance (infinity if one or both not registered)

        Example:
            >>> gen = Genealogy()
            >>> gen.register_genesis(0)
            >>> gen.register_spawn(0, 3)
            >>> gen.register_spawn(0, 5)
            >>> gen.register_spawn(3, 7)
            >>> gen.ultrametric_distance(7, 5)
            3  # MRCA at root (depth 1)

        Mathematical Properties:
            - Ultrametric: d(x,z) ≤ max(d(x,y), d(y,z))
            - Symmetric: d(x,y) = d(y,x)
            - Non-negative: d(x,y) ≥ 0
            - Identity: d(x,y) = 0 ⟺ x = y
        """
        from slime.topology.p_adic import ultrametric_from_tree

        lineage_a = self.get_lineage(pod_a)
        lineage_b = self.get_lineage(pod_b)

        if not lineage_a or not lineage_b:
            return float('inf')

        return float(ultrametric_from_tree(lineage_a, lineage_b))

    def diversity_score(self) -> float:
        """
        Compute phylogenetic diversity (average pairwise distance).

        Higher diversity = more distinct lineages preserved.

        Returns:
            Average ultrametric distance over all pairs

        Example:
            >>> gen = Genealogy()
            >>> # Create diverse lineages
            >>> gen.register_genesis(0)
            >>> gen.register_spawn(0, 1)
            >>> gen.register_spawn(0, 2)
            >>> gen.diversity_score()
            2.0  # Siblings at same depth

        Use Case:
            Novelty search objective - maximize phylogenetic diversity
            to explore behaviorally distant regions of search space.
        """
        if len(self._lineages) < 2:
            return 0.0

        total_dist = 0.0
        count = 0
        ids = list(self._lineages.keys())

        for i, id_a in enumerate(ids):
            for id_b in ids[i+1:]:
                total_dist += self.ultrametric_distance(id_a, id_b)
                count += 1

        return total_dist / count if count > 0 else 0.0

    def get_roots(self) -> List[int]:
        """
        Get all genesis pseudopods (forest roots).

        Returns:
            List of root IDs (generation 0)

        Example:
            >>> gen = Genealogy()
            >>> gen.register_genesis(0)
            >>> gen.register_genesis(10)
            >>> gen.get_roots()
            [0, 10]
        """
        return [
            pod_id for pod_id, gen in self._generation.items()
            if gen == 0
        ]

    def get_leaves(self) -> List[int]:
        """
        Get all leaf pseudopods (no children).

        Returns:
            List of leaf IDs (terminal nodes)

        Example:
            >>> gen = Genealogy()
            >>> gen.register_genesis(0)
            >>> gen.register_spawn(0, 1)
            >>> gen.get_leaves()
            [1]  # Only child has no children
        """
        return [
            pod_id for pod_id in self._lineages.keys()
            if not self._children.get(pod_id, [])
        ]

    def subtree(self, root_id: int) -> Set[int]:
        """
        Get all descendants of pseudopod (BFS traversal).

        Args:
            root_id: Root of subtree

        Returns:
            Set of descendant IDs (including root)

        Example:
            >>> gen = Genealogy()
            >>> gen.register_genesis(0)
            >>> gen.register_spawn(0, 1)
            >>> gen.register_spawn(1, 2)
            >>> gen.subtree(0)
            {0, 1, 2}
        """
        if root_id not in self._lineages:
            return set()

        descendants = {root_id}
        queue = [root_id]

        while queue:
            current = queue.pop(0)
            children = self._children.get(current, [])
            descendants.update(children)
            queue.extend(children)

        return descendants

    def prune(self, pod_id: int) -> None:
        """
        Remove pseudopod and all descendants from genealogy.

        Args:
            pod_id: Root of subtree to prune

        Example:
            >>> gen = Genealogy()
            >>> gen.register_genesis(0)
            >>> gen.register_spawn(0, 1)
            >>> gen.prune(1)
            >>> gen.get_lineage(1)
            ()  # Removed
        """
        to_remove = self.subtree(pod_id)

        for removed_id in to_remove:
            self._lineages.pop(removed_id, None)
            self._children.pop(removed_id, None)
            self._generation.pop(removed_id, None)

        logger.debug(f"Pruned {len(to_remove)} pseudopods from genealogy")

    def size(self) -> int:
        """
        Get total number of pseudopods in genealogy.

        Returns:
            Number of registered pseudopods
        """
        return len(self._lineages)

    def stats(self) -> Dict[str, float]:
        """
        Get genealogy statistics.

        Returns:
            Dict with keys:
            - size: Number of pseudopods
            - num_roots: Number of genesis pseudopods
            - num_leaves: Number of leaf pseudopods
            - avg_generation: Average generation number
            - max_generation: Deepest generation
            - diversity: Phylogenetic diversity score
        """
        generations = list(self._generation.values())

        return {
            'size': self.size(),
            'num_roots': len(self.get_roots()),
            'num_leaves': len(self.get_leaves()),
            'avg_generation': sum(generations) / len(generations) if generations else 0.0,
            'max_generation': max(generations) if generations else 0,
            'diversity': self.diversity_score(),
        }


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'Genealogy',
]
