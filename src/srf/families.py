"""The network families, and how one network of each is drawn.

Every family places ``n`` nodes in the plane and links them. Most link
every pair within the communication range ``R`` (a unit disc graph); a
few add or choose links by construction (``bridged``, ``wheel``,
``laman``). A draw is accepted only if its distance framework is
infinitesimally rigid and well conditioned, so no stored network is an
impossible input or a numerical edge case.

``uniform``   nodes uniform in a square field.
``complete``  a disc of diameter just under ``R``: every pair is linked.
``ring``      an annulus whose hole is wider than ``R``.
``u_shape``   a U whose arms are out of range of each other.
``strip``     a long band half a range wide: the diameter grows with ``n``.
``clusters``  four Gaussian clusters: density varies across the field.
``bridged``   two dense blobs held together by exactly three links.
``wheel``     a hub linked to every node of a jittered rim.
``laman``     minimally rigid by construction (Henneberg type 1).
``lattice``   a triangular lattice at exact positions: not generic.

The field is ``FIELD_SMALL`` square below ``SMALL_BELOW`` nodes and
``FIELD`` square from there on; ``R`` is the same for both. At the large
field a small uniform network is almost never rigid (no rigid draw in 200
at n=8 or n=16, 1 in 200 at n=32), so the small sizes use the small one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import pi, sqrt

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FAMILIES",
    "FIELD",
    "FIELD_SMALL",
    "RANGE",
    "SMALL_BELOW",
    "Drawn",
    "Family",
    "constants",
    "field_side",
    "ring_radii",
    "udg",
]

#: Communication range, the same at every size.
RANGE = 250.0
#: Side of the square field from ``SMALL_BELOW`` nodes on.
FIELD = 1000.0
#: Side of the square field below ``SMALL_BELOW`` nodes.
FIELD_SMALL = 500.0
SMALL_BELOW = 64


def field_side(n: int) -> float:
    """The side of the square field a network of ``n`` nodes is drawn in."""
    return FIELD_SMALL if n < SMALL_BELOW else FIELD


@dataclass(frozen=True, slots=True)
class Drawn:
    """One candidate network, not yet known to be rigid.

    Args:
        positions: ``(n, 2)`` coordinates.
        edges: ``(m, 2)`` links as node indices, any order.
        field: ``(width, height)`` of the area it was drawn in.
        comm_range: The range its disc links were made at.
    """

    positions: NDArray[np.float64]
    edges: NDArray[np.int32]
    field: tuple[float, float]
    comm_range: float


@dataclass(frozen=True, slots=True)
class Family:
    """A kind of network.

    Args:
        name: What the dataset calls it.
        draw: One candidate of ``n`` nodes from a random stream.
        min_n: The smallest size the shape makes sense at.
        random: False when every draw at a size is the same network.
    """

    name: str
    draw: Callable[[int, np.random.Generator], Drawn]
    min_n: int = 3
    random: bool = True


# -- links --------------------------------------------------------------------


def udg(positions: NDArray[np.float64], reach: float) -> NDArray[np.int32]:
    """Every pair of nodes within ``reach`` of each other, ``i < j``, sorted."""
    diff = positions[:, None, :] - positions[None, :, :]
    close = np.einsum("ijk,ijk->ij", diff, diff) <= reach * reach
    i, j = np.nonzero(np.triu(close, k=1))
    return np.column_stack((i, j)).astype(np.int32)


def _inside(
    rng: np.random.Generator,
    n: int,
    box: tuple[float, float],
    keep: Callable[[NDArray[np.float64]], NDArray[np.bool_]],
) -> NDArray[np.float64]:
    """``n`` points uniform over the part of ``box`` that ``keep`` admits."""
    found: list[NDArray[np.float64]] = []
    total = 0
    while total < n:
        darts = rng.uniform((0.0, 0.0), box, size=(max(2 * n, 64), 2))
        darts = darts[keep(darts)]
        found.append(darts)
        total += len(darts)
    return np.concatenate(found)[:n]


def _disc(
    rng: np.random.Generator, n: int, centre: tuple[float, float], radius: float
) -> NDArray[np.float64]:
    """``n`` points uniform over a disc."""
    angle = rng.uniform(0.0, 2 * pi, n)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    return np.asarray(centre) + np.column_stack((r * np.cos(angle), r * np.sin(angle)))


# -- the families -------------------------------------------------------------


def _uniform(n: int, rng: np.random.Generator) -> Drawn:
    side = field_side(n)
    positions = rng.uniform(0.0, side, size=(n, 2))
    return Drawn(positions, udg(positions, RANGE), (side, side), RANGE)


#: Radius of a disc every pair of whose points is in range.
BLOB = 0.49 * RANGE


def _complete(n: int, rng: np.random.Generator) -> Drawn:
    side = field_side(n)
    positions = _disc(rng, n, (side / 2, side / 2), BLOB)
    return Drawn(positions, udg(positions, RANGE), (side, side), RANGE)


#: The annulus has the field's area, so the uniform family's density, and
#: a hole wider than the range, so no link cuts across it.
RING_HOLE = 0.6


def ring_radii(n: int) -> tuple[float, float]:
    """Inner and outer radius of the annulus at ``n`` nodes."""
    side = field_side(n)
    outer = sqrt(side * side / (pi * (1 - RING_HOLE**2)))
    return RING_HOLE * outer, outer


def _ring(n: int, rng: np.random.Generator) -> Drawn:
    inner, outer = ring_radii(n)
    side = 2 * outer + 2.0
    centre = side / 2

    def keep(points: NDArray[np.float64]) -> NDArray[np.bool_]:
        r = np.hypot(points[:, 0] - centre, points[:, 1] - centre)
        return np.asarray((r >= inner) & (r <= outer))

    positions = _inside(rng, n, (side, side), keep)
    return Drawn(positions, udg(positions, RANGE), (side, side), RANGE)


#: An 800 x 700 box with a 300-wide notch cut down from the top, leaving
#: arms and a base 250 thick. The notch is wider than the range, so the
#: arms reach each other only the long way round.
U_BOX = (800.0, 700.0)
U_NOTCH = (250.0, 250.0, 550.0, 700.0)


def _u_shape(n: int, rng: np.random.Generator) -> Drawn:
    x0, y0, x1, _ = U_NOTCH

    def keep(points: NDArray[np.float64]) -> NDArray[np.bool_]:
        x, y = points[:, 0], points[:, 1]
        return np.asarray(~((x > x0) & (x < x1) & (y > y0)))

    positions = _inside(rng, n, U_BOX, keep)
    return Drawn(positions, udg(positions, RANGE), U_BOX, RANGE)


#: Half a range wide, 25 units of length per node: mean degree near 19
#: while the diameter grows in proportion to ``n``.
STRIP_WIDTH = RANGE / 2
STRIP_PER_NODE = 25.0


def _strip(n: int, rng: np.random.Generator) -> Drawn:
    box = (STRIP_PER_NODE * n, STRIP_WIDTH)
    positions = rng.uniform((0.0, 0.0), box, size=(n, 2))
    return Drawn(positions, udg(positions, RANGE), box, RANGE)


CLUSTERS = 4
CLUSTER_SPREAD = 50.0
CLUSTER_INSET = 100.0


def _clusters(n: int, rng: np.random.Generator) -> Drawn:
    side = field_side(n)
    centres = rng.uniform(CLUSTER_INSET, side - CLUSTER_INSET, size=(CLUSTERS, 2))
    which = rng.integers(0, CLUSTERS, n)
    positions = centres[which] + rng.normal(0.0, CLUSTER_SPREAD, size=(n, 2))
    # A node outside the field is drawn again from its cluster. Clipping it
    # to the edge instead would line such nodes up exactly on the border.
    out = ((positions <= 0.0) | (positions >= side)).any(axis=1)
    while out.any():
        positions[out] = centres[which[out]] + rng.normal(0.0, CLUSTER_SPREAD, size=(int(out.sum()), 2))
        out = ((positions <= 0.0) | (positions >= side)).any(axis=1)
    return Drawn(positions, udg(positions, RANGE), (side, side), RANGE)


#: Two blobs 500 apart: 255 of empty field between their rims.
BRIDGED_FIELD = (800.0, 300.0)
BRIDGED_CENTRES = ((150.0, 150.0), (650.0, 150.0))
BRIDGES = 3


def _bridged(n: int, rng: np.random.Generator) -> Drawn:
    halves = [
        _disc(rng, count, centre, BLOB)
        for count, centre in zip((n // 2, n - n // 2), BRIDGED_CENTRES, strict=True)
    ]
    positions = np.concatenate(halves)
    left = rng.choice(n // 2, BRIDGES, replace=False)
    right = n // 2 + rng.choice(n - n // 2, BRIDGES, replace=False)
    bridges = np.column_stack((left, right)).astype(np.int32)
    edges = np.concatenate((udg(positions, RANGE), bridges))
    return Drawn(positions, edges, BRIDGED_FIELD, RANGE)


WHEEL_RADIUS = 200.0


def _wheel(n: int, rng: np.random.Generator) -> Drawn:
    # Node 0 is the hub. The rim is jittered off the circle, so no three
    # rim nodes are collinear and no spoke lies along a rim link.
    side = field_side(n)
    rim = n - 1
    step = 2 * pi / rim
    angle = step * np.arange(rim) + rng.uniform(-0.2, 0.2, rim) * step
    r = WHEEL_RADIUS * rng.uniform(0.95, 1.05, rim)
    positions = np.vstack(
        ([side / 2, side / 2], side / 2 + np.column_stack((r * np.cos(angle), r * np.sin(angle))))
    )
    spokes = [(0, i) for i in range(1, n)]
    tyre = [(1 + k, 1 + (k + 1) % rim) for k in range(rim)]
    edges = np.asarray(spokes + tyre, dtype=np.int32)
    return Drawn(positions, edges, (side, side), RANGE)


LAMAN_NEAREST = 4


def _laman(n: int, rng: np.random.Generator) -> Drawn:
    # Henneberg type 1 from a bar: each new node is tied to two of the
    # four placed nodes nearest it, so the graph is minimally rigid and
    # still local, like something a network could have left behind.
    side = field_side(n)
    positions = rng.uniform(0.0, side, size=(n, 2))
    edges = [(0, 1)]
    for new in range(2, n):
        gaps = np.hypot(*(positions[:new] - positions[new]).T)
        near = np.argsort(gaps)[: min(LAMAN_NEAREST, new)]
        a, b = rng.choice(near, 2, replace=False)
        edges += [(int(a), new), (int(b), new)]
    return Drawn(positions, np.asarray(edges, dtype=np.int32), (side, side), RANGE)


LATTICE_SPACING = 100.0


def _lattice(n: int, rng: np.random.Generator) -> Drawn:
    # Row by row, each row offset by half a spacing, reaching just past
    # one spacing: every node linked to its six neighbours, and whole rows
    # exactly collinear. The stream is not drawn from on purpose.
    del rng
    columns = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / columns))
    step = LATTICE_SPACING
    height = step * sqrt(3) / 2
    positions = np.asarray(
        [
            (step * (c + 0.5 * (r % 2)) + step / 2, height * r + step / 2)
            for r in range(rows)
            for c in range(columns)
        ][:n],
        dtype=np.float64,
    )
    reach = 1.01 * step
    field = (step * (columns + 1.5), height * rows + step)
    return Drawn(positions, udg(positions, reach), field, reach)


FAMILIES: dict[str, Family] = {
    family.name: family
    for family in (
        Family("uniform", _uniform),
        Family("complete", _complete),
        Family("ring", _ring, min_n=24),
        Family("u_shape", _u_shape, min_n=24),
        Family("strip", _strip, min_n=8),
        Family("clusters", _clusters, min_n=16),
        Family("bridged", _bridged, min_n=6),
        Family("wheel", _wheel, min_n=5),
        Family("laman", _laman),
        Family("lattice", _lattice, min_n=8, random=False),
    )
}


def constants() -> dict[str, object]:
    """Every fixed number the families are drawn with, for the dataset index."""
    return {
        "range": RANGE,
        "field": FIELD,
        "field_small": FIELD_SMALL,
        "small_below": SMALL_BELOW,
        "blob_radius": BLOB,
        "ring_hole": RING_HOLE,
        "u_box": list(U_BOX),
        "u_notch": list(U_NOTCH),
        "strip": [STRIP_WIDTH, STRIP_PER_NODE],
        "clusters": [CLUSTERS, CLUSTER_SPREAD, CLUSTER_INSET],
        "bridged": [list(BRIDGED_FIELD), [list(c) for c in BRIDGED_CENTRES], BRIDGES],
        "wheel_radius": WHEEL_RADIUS,
        "laman_nearest": LAMAN_NEAREST,
        "lattice_spacing": LATTICE_SPACING,
    }
