"""Rigidity labels of a network, computed centrally from positions and links.

Nothing here simulates a network: every label is a property of the graph
or of the framework (the graph at its positions), computed on one machine
with numpy and the standard library.

Generic (graph only)
    ``generic_rank`` and ``generic_rigid`` from the (2, 3) pebble game,
    ``minimally_rigid`` (Laman), ``redundant_rigid`` (rigid after losing
    any one link), ``three_connected``, and ``globally_rigid`` (Jackson and
    Jordan: 3-connected and redundantly rigid). ``basis`` is one Laman
    spanning subgraph, as indices into the network's links.

Framework (graph at these positions)
    For each of ``distance``, ``bearing`` and ``se2``: ``*_rigid`` (the
    numerical rank test), ``*_margin`` (the singular value just above the
    trivial motions, in the field's units) and ``*_condition`` (that value
    over the largest). ``distance_rigid_exact`` repeats the distance rank
    test in exact arithmetic on the stored floats, modulo a prime: True is
    a proof, False a proof that the basis *and* every link together fall
    short, None means it could not be settled within the size limit.
    ``uniquely_localizable`` is ``globally_rigid`` and ``distance_rigid``.

Alignment (see :mod:`srf.alignment`)
    ``collinear_triples`` and ``parallel_pairs``, ``parallel_links``:
    links exactly on one line through a shared node, and links exactly
    parallel, on the stored floats; ``collinear_angle`` and
    ``parallel_angle``: the smallest angles, which show near misses.
    ``collinear`` lists the triples.

``degeneracy`` sums up the distance framework: ``flexible`` (not
generically rigid), ``exact`` (generically rigid, but the exact test does
not certify the stored positions), ``ill`` (rigid, condition below
:data:`ILL`), or ``well``.
"""

from __future__ import annotations

from collections import deque
from itertools import pairwise
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from srf.alignment import alignment
from srf.matrix import ILL, Method, expected_rank, numerical_rank, singular_values

__all__ = [
    "ILL",
    "PRIME",
    "PebbleGame",
    "exact_rank",
    "is_three_connected",
    "label",
    "label_fields",
    "pebble",
]

#: A 31-bit prime: products of two residues fit in int64.
PRIME = (1 << 31) - 1
#: Most links the exact test will take the whole matrix for, when the
#: basis alone does not settle it.
EXACT_FULL_LIMIT = 5000


# -- the pebble game ------------------------------------------------------------


class PebbleGame:
    """The (2, 3) pebble game of Jacobs and Hendrickson.

    Links are offered one at a time; each is accepted if independent of
    the ones already accepted. The accepted links are a basis of the
    graph's generic rigidity matroid. When a link is rejected, the nodes
    the failed pebble search reached are the smallest rigid block
    holding both its ends, and the accepted links inside that block with
    the rejected one form its circuit.
    """

    def __init__(self, n: int) -> None:
        self.pebbles = [2] * n
        self.out: list[set[int]] = [set() for _ in range(n)]
        self.rank = 0

    def _search(self, start: int, blocked: tuple[int, int]) -> list[int] | None:
        parent = {start: start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node not in blocked and self.pebbles[node] > 0:
                path = [node]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return path[::-1]
            for other in self.out[node]:
                if other not in parent:
                    parent[other] = node
                    queue.append(other)
        return None

    def _drag(self, path: list[int]) -> None:
        for tail, head in pairwise(path):
            self.out[tail].discard(head)
            self.out[head].add(tail)
        self.pebbles[path[-1]] -= 1
        self.pebbles[path[0]] += 1

    def reach(self, u: int, v: int) -> set[int]:
        """Every node reachable from ``u`` or ``v`` along accepted links."""
        seen = {u, v}
        stack = [u, v]
        while stack:
            for other in self.out[stack.pop()]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return seen

    def insert(self, u: int, v: int) -> bool:
        """Offer link ``uv``; True if it was independent and is now accepted."""
        while self.pebbles[u] + self.pebbles[v] < 4:
            path = self._search(u, (u, v)) or self._search(v, (u, v))
            if path is None:
                return False
            self._drag(path)
        if self.pebbles[u] > 0:
            self.out[u].add(v)
            self.pebbles[u] -= 1
        else:
            self.out[v].add(u)
            self.pebbles[v] -= 1
        self.rank += 1
        return True


def pebble(n: int, edges: NDArray[np.int32]) -> tuple[int, NDArray[np.int64], bool]:
    """Generic rank, a basis, and whether every basis link lies in a circuit.

    Returns:
        ``(rank, basis, redundant)``: the rank of the links in the rigidity
        matroid, the indices of the accepted links, and True when every
        accepted link is in some circuit -- that is, when no single link
        is essential to the rank.
    """
    game = PebbleGame(n)
    basis: list[int] = []
    incident: list[list[int]] = [[] for _ in range(n)]
    covered: set[int] = set()
    full = expected_rank(n)
    for k, (u, v) in enumerate(edges.tolist()):
        if game.insert(u, v):
            basis.append(k)
            incident[u].append(k)
            incident[v].append(k)
        else:
            block = game.reach(u, v)
            for node in block:
                for b in incident[node]:
                    a, c = edges[b]
                    if a in block and c in block:
                        covered.add(b)
        if game.rank == full and len(covered) == full:
            break
    return game.rank, np.asarray(basis, dtype=np.int64), len(covered) == len(basis)


# -- 3-connectivity ----------------------------------------------------------------


def _scan_first_forest(n: int, adjacency: list[set[int]]) -> list[tuple[int, int]]:
    """A breadth-first spanning forest, which is a scan-first search forest."""
    seen = [False] * n
    forest = []
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for other in adjacency[node]:
                if not seen[other]:
                    seen[other] = True
                    forest.append((node, other))
                    queue.append(other)
    return forest


def _biconnected_without(n: int, adjacency: list[set[int]], gone: int) -> bool:
    """Whether the graph minus node ``gone`` is connected with no cut vertex."""
    nodes = [x for x in range(n) if x != gone]
    if len(nodes) <= 2:
        return len(nodes) < 2 or nodes[1] in adjacency[nodes[0]]
    order = [-1] * n
    low = [0] * n
    root = nodes[0]
    order[root] = low[root] = 0
    counter = 1
    root_children = 0
    stack = [(root, -1, iter(adjacency[root]))]
    while stack:
        node, parent, children = stack[-1]
        advanced = False
        for other in children:
            if other in (gone, parent):
                continue
            if order[other] == -1:
                order[other] = low[other] = counter
                counter += 1
                if node == root:
                    root_children += 1
                stack.append((other, node, iter(adjacency[other])))
                advanced = True
                break
            low[node] = min(low[node], order[other])
        if advanced:
            continue
        stack.pop()
        if stack:
            up = stack[-1][0]
            low[up] = min(low[up], low[node])
            if up != root and low[node] >= order[up]:
                return False
    return counter == len(nodes) and root_children <= 1


def is_three_connected(n: int, edges: NDArray[np.int32]) -> bool:
    """Whether the graph stays connected after removing any two nodes.

    Checked on a sparse certificate: the union of three successive
    scan-first search forests keeps 3-connectivity (Cheriyan, Kao and
    Thurimella, 1993) with at most ``3 (n - 1)`` links, so a dense
    network is checked at the cost of a sparse one.
    """
    if n <= 3:
        return False
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for u, v in edges.tolist():
        adjacency[u].add(v)
        adjacency[v].add(u)
    if min(len(a) for a in adjacency) < 3:
        return False
    certificate: list[set[int]] = [set() for _ in range(n)]
    for _ in range(3):
        for u, v in _scan_first_forest(n, adjacency):
            adjacency[u].discard(v)
            adjacency[v].discard(u)
            certificate[u].add(v)
            certificate[v].add(u)
    return all(_biconnected_without(n, certificate, gone) for gone in range(n))


# -- exact rank -------------------------------------------------------------------------


def _residues(values: NDArray[np.float64], p: int) -> NDArray[np.int64]:
    """Each float, as the rational it exactly is, modulo ``p``."""
    mantissa, exponent = np.frexp(values)
    whole = np.round(mantissa * (1 << 53)).astype(np.int64)  # exact: 53 bits
    shift = exponent.astype(np.int64) - 53
    powers = {int(s): pow(2, int(s), p) for s in np.unique(shift)}
    scale = np.vectorize(powers.__getitem__, otypes=[np.int64])(shift) if len(shift) else shift
    return (whole % p) * scale % p


def exact_rank(
    positions: NDArray[np.float64], edges: NDArray[np.int32], p: int = PRIME
) -> int:
    """Rank of the distance rigidity matrix at the stored floats, modulo ``p``.

    Every float is a rational, so the matrix has an exact rank; modulo a
    prime the rank can only come out lower. A full rank here is therefore
    a proof of infinitesimal rigidity at exactly these positions.
    """
    n = len(positions)
    coords = _residues(positions.ravel(), p).reshape(n, 2)
    i, j = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    diff = (coords[i] - coords[j]) % p
    matrix = np.zeros((len(edges), 2 * n), dtype=np.int64)
    rows = np.arange(len(edges))
    for d in range(2):
        matrix[rows, 2 * i + d] = diff[:, d]
        matrix[rows, 2 * j + d] = (-diff[:, d]) % p
    rank = 0
    for column in range(2 * n):
        if rank == len(matrix):
            break
        nonzero = np.flatnonzero(matrix[rank:, column])
        if len(nonzero) == 0:
            continue
        pivot = rank + int(nonzero[0])
        if pivot != rank:
            matrix[[rank, pivot]] = matrix[[pivot, rank]]
        inverse = pow(int(matrix[rank, column]), -1, p)
        matrix[rank] = matrix[rank] * inverse % p
        below = rank + 1 + np.flatnonzero(matrix[rank + 1 :, column])
        if len(below):
            factors = matrix[below, column][:, None]
            matrix[below] = (matrix[below] - factors * matrix[rank]) % p
        rank += 1
    return rank


def _exact_rigid(
    positions: NDArray[np.float64], edges: NDArray[np.int32], basis: NDArray[np.int64]
) -> bool | None:
    n = len(positions)
    full = expected_rank(n)
    if n < 2:
        return True
    if len(basis) == full and exact_rank(positions, edges[basis]) == full:
        return True
    if len(edges) <= EXACT_FULL_LIMIT:
        # A deficient rank mod p is not by itself a proof over the
        # rationals, but a second prime agreeing makes a spurious drop
        # vanishingly unlikely; both are reported through this one answer.
        if exact_rank(positions, edges) == full:
            return True
        return exact_rank(positions, edges, p=(1 << 31) - 19) == full
    return None


# -- the labels ---------------------------------------------------------------------------


def _framework(positions: NDArray[np.float64], edges: NDArray[np.int32], method: Method) -> dict[str, Any]:
    n = len(positions)
    values = singular_values(positions, edges, method)
    wanted = expected_rank(n, method)
    rigid = numerical_rank(values, len(edges), n, method) == wanted
    enough = len(values) >= wanted > 0 and values[0] > 0
    margin = float(values[wanted - 1]) if enough else 0.0
    return {
        f"{method}_rigid": bool(rigid),
        f"{method}_margin": margin,
        f"{method}_condition": margin / float(values[0]) if enough else 0.0,
    }


def label(positions: NDArray[np.float64], edges: NDArray[np.int32]) -> dict[str, Any]:
    """Every label of one network. See the module docstring for each field.

    Returns:
        Scalars keyed by name, plus ``basis`` (link indices),
        ``collinear`` (node triples) and ``seconds`` spent.
    """
    started = perf_counter()
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    n, m = len(positions), len(edges)
    full = expected_rank(n)
    rank, basis, covered = pebble(n, edges)
    generic = rank == full
    three = is_three_connected(n, edges)
    redundant = generic and covered
    if n <= 3:
        globally = m == n * (n - 1) // 2
    else:
        globally = three and redundant
    out: dict[str, Any] = {
        "generic_rank": rank,
        "generic_rigid": generic,
        "minimally_rigid": generic and m == full,
        "redundant_rigid": redundant,
        "redundant_links": m - rank,
        "three_connected": three,
        "globally_rigid": globally,
    }
    for method in ("distance", "bearing", "se2"):
        out |= _framework(positions, edges, method)
    exact = _exact_rigid(positions, edges, basis) if generic else False
    out["distance_rigid_exact"] = exact
    out["uniquely_localizable"] = bool(globally and out["distance_rigid"])
    out |= alignment(positions, edges)
    if not generic:
        out["degeneracy"] = "flexible"
    elif exact is False:
        out["degeneracy"] = "exact"
    elif not out["distance_rigid"] or out["distance_condition"] < ILL:
        out["degeneracy"] = "ill"
    else:
        out["degeneracy"] = "well"
    out["basis"] = basis
    out["seconds"] = perf_counter() - started
    return out


def label_fields() -> list[str]:
    """The scalar label names, in table order."""
    base = [
        "generic_rank",
        "generic_rigid",
        "minimally_rigid",
        "redundant_rigid",
        "redundant_links",
        "three_connected",
        "globally_rigid",
    ]
    for method in ("distance", "bearing", "se2"):
        base += [f"{method}_rigid", f"{method}_margin", f"{method}_condition"]
    return [
        *base,
        "distance_rigid_exact",
        "uniquely_localizable",
        "degeneracy",
        "collinear_triples",
        "collinear_angle",
        "parallel_pairs",
        "parallel_links",
        "parallel_angle",
    ]
