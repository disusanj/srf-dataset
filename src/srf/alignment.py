"""Collinear and parallel links: exactly, and by how close they come.

Two links are *parallel* when their directions are the same line direction,
and *collinear* when they also share a node, so that the three nodes lie
on one line. Both are exact statements about the stored floats: every
float is a rational, so a candidate found by angle is settled by comparing
slopes in rational arithmetic. Near misses are measured by angle instead.

Random positions are aligned with probability zero, so the counts are
nonzero essentially only for ``lattice``; the angles show how close a
random network comes.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import pi
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["TOLERANCE", "alignment"]

#: Angle (radians) within which two directions are checked exactly. Float
#: angles of exactly parallel links differ by rounding only, far below this.
TOLERANCE = 1e-7
#: The angle reported when there are not two directions to compare.
NONE = pi / 2


def _angles(diff: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each vector's line direction, in ``[0, pi)``."""
    return np.mod(np.arctan2(diff[:, 1], diff[:, 0]), pi)


def _closest(angles: NDArray[np.float64]) -> tuple[float, NDArray[np.int64]]:
    """The smallest angle between two directions, and the candidates within tolerance.

    Returns:
        The smallest angle between two line directions, in ``[0, pi/2]``,
        and the indices of every direction within :data:`TOLERANCE` of
        its neighbour, going round the half-turn.
    """
    if len(angles) < 2:
        return NONE, np.zeros(0, dtype=np.int64)
    order = np.argsort(angles, kind="stable")
    ordered = angles[order]
    gaps = np.append(np.diff(ordered), ordered[0] + pi - ordered[-1])
    close = np.flatnonzero(gaps < TOLERANCE)
    near = np.unique(np.concatenate((close, (close + 1) % len(order))))
    return float(min(gaps.min(), NONE)), order[near]


def _key(dx: Fraction, dy: Fraction) -> Fraction | None:
    """The exact line direction of ``(dx, dy)``: its slope, None if vertical."""
    return None if dx == 0 else dy / dx


def _exact(positions: NDArray[np.float64], a: int, b: int) -> Fraction | None:
    xa, ya = positions[a]
    xb, yb = positions[b]
    return _key(Fraction(xb) - Fraction(xa), Fraction(yb) - Fraction(ya))


def alignment(positions: NDArray[np.float64], edges: NDArray[np.int32]) -> dict[str, Any]:
    """Collinear and parallel link labels of one network.

    Returns:
        ``collinear_triples``: how many triples ``(u, v, w)`` have links
        ``uv`` and ``vw`` exactly on one line. ``collinear_angle``: the
        smallest angle between two links at a common node. ``parallel_pairs``:
        how many pairs of links are exactly parallel, collinear ones
        included. ``parallel_links``: how many links are parallel to at
        least one other. ``parallel_angle``: the smallest angle between
        any two links. ``collinear``: the triples, ``(k, 3)`` with ``u < w``.
        Angles are in radians, in ``[0, pi/2]``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n = len(positions)

    parallel_angle, near = _closest(_angles(positions[edges[:, 1]] - positions[edges[:, 0]]))
    groups = Counter(_exact(positions, *edges[k]) for k in near.tolist())
    sizes = [c for c in groups.values() if c > 1]

    incident: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges.tolist():
        incident[u].append(v)
        incident[v].append(u)
    collinear_angle = NONE
    triples: list[tuple[int, int, int]] = []
    for v, others in enumerate(incident):
        if len(others) < 2:
            continue
        ends = np.asarray(others)
        angle, near = _closest(_angles(positions[ends] - positions[v]))
        collinear_angle = min(collinear_angle, angle)
        lines: dict[Fraction | None, list[int]] = {}
        for k in near.tolist():
            lines.setdefault(_exact(positions, v, int(ends[k])), []).append(int(ends[k]))
        for line in lines.values():
            line.sort()
            triples += [(u, v, w) for i, u in enumerate(line) for w in line[i + 1 :]]

    return {
        "collinear_triples": len(triples),
        "collinear_angle": collinear_angle,
        "parallel_pairs": sum(c * (c - 1) // 2 for c in sizes),
        "parallel_links": sum(sizes),
        "parallel_angle": parallel_angle,
        "collinear": np.asarray(sorted(triples), dtype=np.int64).reshape(-1, 3),
    }
