"""Rigidity matrices of a planar framework, and the rank test on them.

Three measurements are covered. ``distance`` is the length of a link and
leaves the three rigid motions free. ``bearing`` is the direction of a
link read against a frame every node shares, and leaves translation and
scaling free; its matrix is Zhao and Zelazo's. ``se2`` is that direction
read against each node's own unknown heading: the headings become unknowns
too, and a coordinated rotation joins the trivial motions.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

#: Distance conditioning below this is called ill-conditioned; the
#: generator redraws such a network.
ILL = 1e-10

__all__ = [
    "ILL",
    "Method",
    "expected_rank",
    "is_infinitesimally_rigid",
    "is_well_conditioned",
    "numerical_rank",
    "rigidity_matrix",
    "se2_scale",
    "singular_values",
]

Method = Literal["distance", "bearing", "se2"]


def expected_rank(n: int, method: Method = "distance") -> int:
    """The rank of the rigidity matrix of a rigid framework of ``n`` nodes."""
    if n < 2:
        return 0
    return 3 * n - 4 if method == "se2" else 2 * n - 3


def se2_scale(positions: NDArray[np.float64], edges: NDArray[np.int32]) -> float:
    """Mean link length, which ``se2`` divides positions by."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return 1.0
    mean = float(np.hypot(*(positions[edges[:, 0]] - positions[edges[:, 1]]).T).mean())
    return mean if mean > 0 else 1.0


def rigidity_matrix(
    positions: NDArray[np.float64],
    edges: NDArray[np.int32],
    method: Method = "distance",
    scale: float | None = None,
) -> NDArray[np.float64]:
    """The rigidity matrix at the given positions.

    Args:
        positions: ``(n, 2)`` coordinates.
        edges: ``(m, 2)`` links as node indices.
        method: ``distance`` writes ``p_i - p_j`` into node ``i``'s columns
            and its negative into ``j``'s. ``bearing`` writes that
            difference turned a quarter turn and divided by its squared
            length. ``se2`` writes two rows per link, one for each end,
            each with that end's heading column.
        scale: For ``se2``, the length positions are divided by; the mean
            link length of ``edges`` when None. A caller building the
            matrix in blocks passes the whole network's.

    Returns:
        ``(m, 2n)`` for ``distance`` and ``bearing``, ``(2m, 3n)`` for ``se2``.
    """
    n = len(positions)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    i, j = edges[:, 0], edges[:, 1]
    rows = np.arange(len(edges))
    if method in ("distance", "bearing"):
        diff = positions[i] - positions[j]
        if method == "bearing":
            # Zhao and Zelazo's matrix has the 2 x 2 block P_g / |e| per
            # link, P_g the projection orthogonal to the bearing g. In the
            # plane P_g = g' g'^T with g' the bearing turned a quarter turn,
            # so the one row g'^T / |e| has the same Gram matrix, and hence
            # the same singular values, at half the rows. Turning alone,
            # without the division, would give the distance matrix's values.
            squared = np.einsum("ij,ij->i", diff, diff)
            squared[squared == 0.0] = np.inf
            diff = np.column_stack((diff[:, 1], -diff[:, 0])) / squared[:, None]
        matrix = np.zeros((len(edges), 2 * n))
        for d in range(2):
            matrix[rows, 2 * i + d] = diff[:, d]
            matrix[rows, 2 * j + d] = -diff[:, d]
        return matrix
    if method != "se2":
        raise ValueError(f"Unknown method {method!r}.")
    # Positions are divided by the mean link length first. Otherwise the
    # position columns carry one over a length and the heading columns a
    # bare one, and on a field hundreds of units wide the rank cut throws
    # real stiffness away as numerical dust.
    if scale is None:
        scale = se2_scale(positions, edges)
    matrix = np.zeros((2 * len(edges), 3 * n))
    for side, (near, far) in enumerate(((i, j), (j, i))):
        diff = (positions[far] - positions[near]) / scale
        squared = np.einsum("ij,ij->i", diff, diff)
        squared[squared == 0.0] = np.inf
        turn = np.column_stack((diff[:, 1], -diff[:, 0])) / squared[:, None]
        r = 2 * rows + side
        for d in range(2):
            matrix[r, 3 * near + d] = -turn[:, d]
            matrix[r, 3 * far + d] = turn[:, d]
        matrix[r, 3 * near + 2] = -1.0
    return matrix


def singular_values(
    positions: NDArray[np.float64], edges: NDArray[np.int32], method: Method = "distance"
) -> NDArray[np.float64]:
    """Singular values of the rigidity matrix, largest first."""
    edges = np.asarray(edges).reshape(-1, 2)
    if len(edges) == 0:
        return np.zeros(0)
    # The Gram matrix would be cheaper but squares the condition number,
    # which is exactly what the small singular values cannot afford. A tall
    # matrix is instead folded into its square triangular factor block by
    # block (Householder QR keeps the singular values), so a dense network
    # never holds its whole matrix: complete at n=512 is 131k x 1024 for
    # distances and twice as tall and half as wide again for se2.
    width = len(positions) * (3 if method == "se2" else 2)
    rows_per_edge = 2 if method == "se2" else 1
    block = max(1, (4 * width) // rows_per_edge)
    if len(edges) <= block:
        return np.linalg.svd(rigidity_matrix(positions, edges, method), compute_uv=False)
    scale = se2_scale(positions, edges) if method == "se2" else None
    factor = np.zeros((0, width))
    for start in range(0, len(edges), block):
        part = rigidity_matrix(positions, edges[start : start + block], method, scale)
        factor = np.linalg.qr(np.vstack((factor, part)), mode="r")
    return np.linalg.svd(factor, compute_uv=False)


def is_infinitesimally_rigid(
    positions: NDArray[np.float64], edges: NDArray[np.int32], method: Method = "distance"
) -> bool:
    """Whether the matrix has full rank, with a cut relative to the largest value.

    The cut is numpy's default for :func:`numpy.linalg.matrix_rank`:
    ``max(shape) * eps * s_max``. A fixed floor would call large sparse
    rigid networks flexible for no reason but their size.
    """
    n = len(positions)
    if n < 2:
        return True
    values = singular_values(positions, edges, method)
    if len(values) == 0:
        return False
    return numerical_rank(values, len(edges), n, method) == expected_rank(n, method)


def is_well_conditioned(
    positions: NDArray[np.float64],
    edges: NDArray[np.int32],
    method: Method = "distance",
    floor: float = ILL,
) -> bool:
    """Whether the framework is rigid and its condition is at least ``floor``.

    The condition is the smallest non-trivial singular value over the
    largest, as in the labels.
    """
    n = len(positions)
    wanted = expected_rank(n, method)
    if wanted == 0:
        return True
    values = singular_values(positions, edges, method)
    if len(values) < wanted or values[0] <= 0:
        return False
    rigid = numerical_rank(values, len(edges), n, method) == wanted
    return bool(rigid and values[wanted - 1] >= floor * values[0])


def numerical_rank(
    values: NDArray[np.float64], m: int, n: int, method: Method = "distance"
) -> int:
    """Rank from singular values, with numpy's cut ``max(shape) * eps * s_max``.

    The shape is the full matrix's, not that of any factor it was folded to.
    """
    if len(values) == 0:
        return 0
    shape = (m * (2 if method == "se2" else 1), n * (3 if method == "se2" else 2))
    cut = max(shape) * np.finfo(float).eps * values[0]
    return int((values > cut).sum())
