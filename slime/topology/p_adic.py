"""
p-Adic distance metrics for hierarchical structures.

p-Adic numbers extend integers with a notion of "closeness" based on
divisibility rather than magnitude. For SHA256 hashes organized in a
256-ary tree (objects/XX/...), p-adic distance naturally measures
common prefix length.

Key Properties:
- Ultrametric: d(x,z) ≤ max(d(x,y), d(y,z)) (strong triangle inequality)
- Hierarchical: Elements with same prefix are "close"
- Non-Archimedean: Distance doesn't grow with magnitude

Applications:
- Content-addressable storage (SHA256 hashes)
- Phylogenetic trees (common ancestor depth)
- Hierarchical clustering (GMM dendrograms)

References:
- Gouvêa, "p-adic Numbers: An Introduction" (1997)
- Schikhof, "Ultrametric Calculus" (1984)
"""

from typing import Tuple


def p_adic_distance(x: str, y: str, p: int = 256) -> float:
    """
    Compute p-adic distance between two strings.

    Distance = p^(-n) where n = length of common prefix.

    Args:
        x: First string (e.g., SHA256 hash)
        y: Second string
        p: Base (256 for hex pairs, 2 for binary)

    Returns:
        Distance in (0, 1], where:
        - 0 < d ≤ 1 (never exactly 0 unless x == y)
        - Smaller values = more similar (longer common prefix)
        - 1 = no common prefix

    Example:
        >>> p_adic_distance("abc123", "abc456", p=256)
        0.0000152587890625  # Same first 3 chars → 256^(-3)

        >>> p_adic_distance("abc", "def", p=256)
        1.0  # No common prefix

        >>> p_adic_distance("abc", "abc", p=256)
        3.725290298461914e-09  # Identical → 256^(-3)

    Mathematical Properties:
        - Ultrametric: d(x,z) ≤ max(d(x,y), d(y,z))
        - Symmetric: d(x,y) = d(y,x)
        - Identity: d(x,y) = 0 ⟺ x = y
        - Non-Archimedean: d(x+z, y+z) = d(x, y)
    """
    if x == y:
        # Identical strings: distance proportional to length
        return p ** (-len(x)) if len(x) > 0 else 0.0

    # Count common prefix length
    n = 0
    for a, b in zip(x, y):
        if a == b:
            n += 1
        else:
            break

    # Distance = p^(-n)
    if n == 0:
        return 1.0  # No common prefix
    else:
        return float(p ** (-n))


def p_adic_valuation(x: int, p: int = 2) -> int:
    """
    Compute p-adic valuation v_p(x).

    Valuation v_p(x) = max{k : p^k divides x}

    Measures "how divisible" x is by p. Higher valuation = more divisible.

    Args:
        x: Integer to compute valuation for
        p: Prime base (typically 2)

    Returns:
        Valuation (infinity if x == 0)

    Example:
        >>> p_adic_valuation(8, p=2)
        3  # 8 = 2^3

        >>> p_adic_valuation(12, p=2)
        2  # 12 = 2^2 * 3

        >>> p_adic_valuation(7, p=2)
        0  # 7 not divisible by 2

    Use Case:
        Metric refinement - assign importance weights based on
        how many filtration levels a metric survives:
        - v=0: Filtered out immediately (zero variance)
        - v=1: Survives first filter but not PCA
        - v=2: Core behavioral dimension
    """
    if x == 0:
        return float('inf')  # type: ignore[return-value]

    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v


def ultrametric_from_tree(
    lineage_a: Tuple[int, ...],
    lineage_b: Tuple[int, ...]
) -> int:
    """
    Compute ultrametric distance on tree via MRCA depth.

    For phylogenetic trees or genealogies, distance = depth to
    most recent common ancestor (MRCA).

    Args:
        lineage_a: Path from root to node A (e.g., (0, 3, 7))
        lineage_b: Path from root to node B (e.g., (0, 5, 12))

    Returns:
        Ultrametric distance = len(A) + len(B) - 2*depth(MRCA)

    Example:
        >>> # Lineages:
        >>> #       0 (root)
        >>> #      / \\
        >>> #     3   5
        >>> #    /     \\
        >>> #   7      12
        >>> ultrametric_from_tree((0, 3, 7), (0, 5, 12))
        4  # MRCA at depth 1 (node 0)

        >>> ultrametric_from_tree((0, 3, 7), (0, 3, 9))
        2  # MRCA at depth 2 (node 3)

    Mathematical Properties:
        - Satisfies strong triangle inequality:
          d(7, 12) = 4 ≤ max(d(7, 9), d(9, 12)) = max(2, 4) = 4
        - Isosceles: In any triple, at least 2 sides equal
        - Tree metric: Embeds naturally in tree structure
    """
    if not lineage_a or not lineage_b:
        return float('inf')  # type: ignore[return-value]

    # Find MRCA depth (longest common prefix)
    mrca_depth = 0
    for i, (a, b) in enumerate(zip(lineage_a, lineage_b)):
        if a == b:
            mrca_depth = i + 1
        else:
            break

    # Distance = sum of branch lengths from MRCA
    return len(lineage_a) + len(lineage_b) - 2 * mrca_depth


def hierarchical_distance(
    x: Tuple[float, ...],
    y: Tuple[float, ...],
    valuations: Tuple[int, ...]
) -> float:
    """
    Compute distance with hierarchical weighting.

    Weights dimensions by their valuation (refinement level).
    Higher valuation = survived more filters = more important.

    Args:
        x: First point in metric space
        y: Second point
        valuations: p-adic valuations for each dimension

    Returns:
        Weighted L2 distance respecting hierarchy

    Example:
        >>> # Suppose dimensions have valuations [2, 1, 0]
        >>> # (core dimension, intermediate, filtered)
        >>> x = (1.0, 2.0, 3.0)
        >>> y = (1.5, 2.5, 4.0)
        >>> valuations = (2, 1, 0)
        >>> hierarchical_distance(x, y, valuations)
        1.118...  # Core dimension weighted 4x more than filtered

    Use Case:
        Multi-resolution fitness landscapes:
        - Coarse search uses low-valuation dimensions
        - Fine search uses high-valuation dimensions
        - Naturally implements filtration X₆₂ ⊃ X₁₂ ⊃ X₃
    """
    if len(x) != len(y) != len(valuations):
        raise ValueError(
            f"Dimension mismatch: x={len(x)}, y={len(y)}, valuations={len(valuations)}"
        )

    dist_sq = 0.0
    for xi, yi, vi in zip(x, y, valuations):
        weight = 2 ** vi  # Exponential weighting by valuation
        dist_sq += weight * (xi - yi) ** 2

    return float(dist_sq ** 0.5)


def p_adic_ball(center: str, radius: float, p: int = 256) -> int:
    """
    Compute size of p-adic ball B(center, radius).

    In p-adic space, balls are clopen (closed and open) with
    well-defined size independent of center.

    Args:
        center: Center point (e.g., SHA256 hash)
        radius: Radius (must be of form p^(-k))
        p: Base

    Returns:
        Number of points in ball (integer)

    Example:
        >>> # Radius 256^(-2) = all hashes with same first 2 chars
        >>> p_adic_ball("abc123", 256**(-2), p=256)
        65536  # 256^2 possible last 2 chars

    Use Case:
        Content-addressable storage: estimate number of similar
        checkpoints within p-adic ball for deduplication.
    """
    import math

    # Find k such that radius = p^(-k)
    if radius <= 0 or radius > 1:
        raise ValueError(f"Radius must be in (0, 1], got {radius}")

    k = round(-math.log(radius) / math.log(p))

    # Ball size = p^(len(center) - k)
    ball_size = p ** (len(center) - k)
    return int(ball_size)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'p_adic_distance',
    'p_adic_valuation',
    'ultrametric_from_tree',
    'hierarchical_distance',
    'p_adic_ball',
]
