"""The rigidity labels, checked against brute force on small graphs.

Everything here is central: numpy and the standard library, no simulator.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from srf import SRF, build, build_dataset, read_json, read_npz
from srf.alignment import alignment
from srf.io import write_json, write_npz
from srf.labels import exact_rank, is_three_connected, label, label_fields, pebble
from srf.matrix import expected_rank, rigidity_matrix, singular_values

RNG = np.random.default_rng(20260926)


def _generic_rank(n: int, edges: np.ndarray) -> int:
    """Rank at random positions: the generic rank with probability one."""
    if len(edges) == 0:
        return 0
    positions = RNG.uniform(0, 1, (n, 2))
    return int(np.linalg.matrix_rank(rigidity_matrix(positions, edges)))


def _connected(n: int, edges: list[tuple[int, int]], gone: set[int]) -> bool:
    left = [x for x in range(n) if x not in gone]
    seen, stack = {left[0]}, [left[0]]
    adjacency: dict[int, set[int]] = {x: set() for x in left}
    for u, v in edges:
        if u not in gone and v not in gone:
            adjacency[u].add(v)
            adjacency[v].add(u)
    while stack:
        for other in adjacency[stack.pop()]:
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return len(seen) == len(left)


def _random_graph(n: int, p: float) -> np.ndarray:
    pairs = [(i, j) for i, j in combinations(range(n), 2) if RNG.random() < p]
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


GRAPHS = [(n, _random_graph(n, p)) for n in (5, 6, 7, 8, 9) for p in (0.4, 0.6, 0.8) for _ in range(6)]


@pytest.mark.parametrize(("n", "edges"), GRAPHS)
def test_pebble_rank_and_redundancy_match_brute_force(n: int, edges: np.ndarray) -> None:
    rank, basis, covered = pebble(n, edges)
    assert rank == _generic_rank(n, edges)
    assert len(basis) == rank
    assert _generic_rank(n, edges[basis]) == rank
    if rank == expected_rank(n):
        each = all(
            _generic_rank(n, np.delete(edges, k, axis=0)) == rank for k in range(len(edges))
        )
        assert covered == each


@pytest.mark.parametrize(("n", "edges"), GRAPHS)
def test_three_connectivity_matches_brute_force(n: int, edges: np.ndarray) -> None:
    pairs = [tuple(e) for e in edges.tolist()]
    brute = n > 3 and all(
        _connected(n, pairs, set(gone)) for k in (0, 1, 2) for gone in combinations(range(n), k)
    )
    assert is_three_connected(n, edges) == brute


def test_known_graphs() -> None:
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.3, 0.9]])
    k3 = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)
    got = label(tri, k3)
    assert got["globally_rigid"] and got["minimally_rigid"] and got["distance_rigid_exact"]

    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    k4 = np.array(list(combinations(range(4), 2)), dtype=np.int32)
    got = label(square, k4)
    assert got["globally_rigid"] and got["redundant_rigid"] and got["three_connected"]
    assert got["uniquely_localizable"] and got["degeneracy"] == "well"

    got = label(square, k4[:-1])  # K4 minus a link: rigid, not redundantly
    assert got["minimally_rigid"] and not got["redundant_rigid"] and not got["globally_rigid"]

    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    got = label(line, k4)  # generically rigid, but every point on one line
    assert got["generic_rigid"] and got["distance_rigid_exact"] is False
    assert not got["distance_rigid"] and got["degeneracy"] == "exact"

    path = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    assert label(square, path)["degeneracy"] == "flexible"


def test_exact_rank_agrees_with_float_rank_on_generic_positions() -> None:
    for n in (6, 10, 20):
        positions = RNG.uniform(0, 500, (n, 2))
        edges = _random_graph(n, 0.5)
        assert exact_rank(positions, edges) == np.linalg.matrix_rank(rigidity_matrix(positions, edges))


def test_blocked_singular_values_match_a_full_svd() -> None:
    net = build("complete", 64, 0)  # 2016 links, folded in blocks for every method
    for method in ("distance", "bearing", "se2"):
        folded = singular_values(net.positions, net.edges, method)
        full = np.linalg.svd(rigidity_matrix(net.positions, net.edges, method), compute_uv=False)
        assert np.allclose(folded, full, rtol=0, atol=1e-12 * full[0])


def _zhao_zelazo(positions: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The bearing rigidity matrix as Zhao and Zelazo write it: P_g / |e| blocks."""
    n = len(positions)
    matrix = np.zeros((2 * len(edges), 2 * n))
    for k, (i, j) in enumerate(edges):
        e = positions[j] - positions[i]
        length = np.linalg.norm(e)
        g = e / length
        block = (np.eye(2) - np.outer(g, g)) / length
        matrix[2 * k : 2 * k + 2, 2 * i : 2 * i + 2] = -block
        matrix[2 * k : 2 * k + 2, 2 * j : 2 * j + 2] = block
    return matrix


@pytest.mark.parametrize(("family", "n"), [("uniform", 32), ("strip", 64), ("laman", 24), ("complete", 64)])
def test_bearing_matrix_is_zhao_zelazo(family: str, n: int) -> None:
    net = build(family, n, 0)
    full = np.linalg.svd(_zhao_zelazo(net.positions, net.edges), compute_uv=False)
    ours = singular_values(net.positions, net.edges, "bearing")
    rank = expected_rank(n, "bearing")
    assert np.allclose(ours[:rank], full[:rank], rtol=0, atol=1e-12 * full[0])
    assert np.all(full[rank:] < 1e-12 * full[0])
    distance = singular_values(net.positions, net.edges, "distance")
    assert not np.allclose(ours / ours[0], distance / distance[0])  # not a rotated copy


@pytest.mark.parametrize("family", ["uniform", "complete", "wheel", "laman", "lattice", "bridged"])
def test_labels_of_dataset_families(family: str) -> None:
    net = build(family, 32, 0)
    got = label(net.positions, net.edges)
    assert got["generic_rigid"] and got["distance_rigid"] and got["distance_rigid_exact"]
    assert got["minimally_rigid"] == (family == "laman")
    if family == "laman":
        assert not got["redundant_rigid"] and not got["globally_rigid"]
    if family in ("complete", "wheel"):
        assert got["globally_rigid"]
    if family == "bridged":  # three vertex-disjoint bridges between two cliques
        assert got["three_connected"]


def test_labels_survive_both_formats(tmp_path: Path) -> None:
    nets = [build("uniform", 16, c) for c in range(3)]
    from dataclasses import replace

    nets = [replace(net, extra=label(net.positions, net.edges)) for net in nets]
    archive = write_npz(tmp_path, nets)
    for net, from_npz in zip(nets, read_npz(archive), strict=True):
        from_json = read_json(write_json(tmp_path, net))
        for key in label_fields():
            assert from_npz.extra[key] == net.extra[key] == from_json.extra[key], key
        assert np.array_equal(from_npz.extra["basis"], net.extra["basis"])
        assert np.array_equal(from_json.extra["basis"], net.extra["basis"])
        assert np.array_equal(from_npz.extra["collinear"], net.extra["collinear"])
        assert np.array_equal(from_json.extra["collinear"], net.extra["collinear"])


def test_dataset_carries_labels(tmp_path: Path) -> None:
    summary = build_dataset(
        tmp_path, ["uniform", "laman"], sizes={"uniform": [8], "laman": [8]},
        cases={"uniform": 10, "laman": 10}, progress=None,
    )
    assert summary["degeneracy"]["well"] + summary["degeneracy"]["ill"] == 20
    srf = SRF(tmp_path)
    assert all(row["generic_rigid"] == "True" for row in srf.table)
    assert {row["minimally_rigid"] for row in srf.rows(family="laman")} == {"True"}
    net = srf.network("laman", 8, 4)
    assert net.extra["minimally_rigid"] and len(net.extra["basis"]) == 13


def _cross(a: np.ndarray, b: np.ndarray) -> int:
    return int(a[0]) * int(b[1]) - int(a[1]) * int(b[0])


@pytest.mark.parametrize("seed", range(12))
def test_alignment_matches_brute_force_on_a_grid(seed: int) -> None:
    # Integer points on a small grid: many exact alignments, all decidable
    # with integer cross products.
    rng = np.random.default_rng(seed)
    n = 9
    grid = rng.choice(25, n, replace=False)
    positions = np.column_stack((grid % 5, grid // 5)).astype(np.float64) * 37.0
    edges = _random_graph(n, 0.5)
    got = alignment(positions, edges)
    diff = [positions[j] - positions[i] for i, j in edges]
    pairs = sum(1 for a, b in combinations(range(len(edges)), 2) if _cross(diff[a], diff[b]) == 0)
    adjacency = {v: {int(w) for e in edges for w in e if v in e and w != v} for v in range(n)}
    triples = sorted(
        (u, v, w)
        for v in range(n)
        for u, w in combinations(sorted(adjacency[v]), 2)
        if _cross(positions[u] - positions[v], positions[w] - positions[v]) == 0
    )
    assert got["parallel_pairs"] == pairs
    assert got["collinear_triples"] == len(triples)
    assert [tuple(t) for t in got["collinear"].tolist()] == triples
    if pairs:
        assert got["parallel_angle"] < 1e-12


def test_lattice_rows_are_collinear_and_random_networks_are_not() -> None:
    lattice = build("lattice", 16, 0)
    got = alignment(lattice.positions, lattice.edges)
    assert got["collinear_triples"] > 0 and got["parallel_pairs"] > got["collinear_triples"]
    assert got["collinear_angle"] < 1e-12
    for family in ("uniform", "complete", "laman"):
        net = build(family, 32, 0)
        got = alignment(net.positions, net.edges)
        assert got["collinear_triples"] == 0 and got["parallel_pairs"] == 0
        assert 0.0 < got["parallel_angle"] <= got["collinear_angle"] <= np.pi / 2
