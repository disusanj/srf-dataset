"""Fast checks of the generator, the two formats and the loaders."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from srf import SRF, build, build_dataset, read_json, read_npz
from srf.families import FAMILIES, FIELD, FIELD_SMALL, SMALL_BELOW, field_side, ring_radii
from srf.generate import CASES, SIZES, split_of, stream
from srf.io import write_json, write_npz
from srf.matrix import ILL, expected_rank, is_infinitesimally_rigid, is_well_conditioned, singular_values


def _same(a, b) -> None:
    assert (a.family, a.n, a.case, a.split, a.seed) == (b.family, b.n, b.case, b.split, b.seed)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.edges, b.edges)
    assert a.field == b.field and a.comm_range == b.comm_range and a.attempts == b.attempts


# -- names, seeds, splits -------------------------------------------------------


def test_stream_is_a_function_of_the_name_and_salt() -> None:
    assert stream("uniform", 64, 3) == stream("uniform", 64, 3)
    assert stream("uniform", 64, 3) != stream("uniform", 64, 4)
    assert stream("uniform", 64, 3) != stream("uniform", 64, 3, salt="other")


def test_splits_are_70_10_20_by_case_index() -> None:
    for total in (1000, 200):
        got = [split_of("uniform", c, total) for c in range(total)]
        assert got.count("train") == 0.7 * total
        assert got.count("val") == 0.1 * total
        assert got.count("test") == 0.2 * total
        assert got == sorted(got, key=("train", "val", "test").index)
    assert split_of("lattice", 0) == "test"
    with pytest.raises(ValueError):
        split_of("uniform", 1000)


def test_plan_matches_the_decided_sizes_and_counts() -> None:
    assert SIZES["uniform"][0] == 8 and SIZES["uniform"][-1] == 512
    assert all(max(s) == 512 for s in SIZES.values())
    assert CASES["uniform"] == 1000
    assert all(CASES[f] == 200 for f in FAMILIES if f not in ("uniform", "lattice"))


def test_field_switches_at_small_below() -> None:
    assert field_side(SMALL_BELOW - 1) == FIELD_SMALL
    assert field_side(SMALL_BELOW) == FIELD
    inner, _ = ring_radii(8)
    assert 2 * inner > 250.0  # no link across the hole


# -- the matrix -----------------------------------------------------------------


def test_triangle_is_rigid_and_a_path_is_not() -> None:
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.3, 0.9]])
    assert is_infinitesimally_rigid(tri, np.array([[0, 1], [0, 2], [1, 2]]))
    assert not is_infinitesimally_rigid(tri, np.array([[0, 1], [1, 2]]))
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert not is_infinitesimally_rigid(line, np.array([[0, 1], [0, 2], [1, 2]]))
    for method in ("bearing", "se2"):
        values = singular_values(tri, np.array([[0, 1], [0, 2], [1, 2]]), method)
        assert (values > 1e-9).sum() == expected_rank(3, method)


# -- every family ---------------------------------------------------------------


@pytest.mark.parametrize("family", list(FAMILIES))
def test_each_family_builds_a_rigid_canonical_network(family: str) -> None:
    n = max(FAMILIES[family].min_n, 32)
    net = build(family, n, 0)
    again = build(family, n, 0)
    _same(net, again)
    assert net.positions.shape == (n, 2)
    assert np.all(net.edges[:, 0] < net.edges[:, 1])
    assert len(np.unique(net.edges, axis=0)) == net.m
    assert is_infinitesimally_rigid(net.positions, net.edges)
    assert net.degrees().min() >= 2


def test_clusters_nodes_stay_strictly_inside_the_field() -> None:
    # Clipping put nodes exactly on the border, and lined them up there.
    rng = np.random.default_rng(0)
    for n in (128, 512):
        side = field_side(n)
        for _ in range(20):
            positions = FAMILIES["clusters"].draw(n, rng).positions
            assert ((positions > 0) & (positions < side)).all()


def test_an_ill_conditioned_draw_is_redrawn() -> None:
    # A development build kept laman n=512 case 127 at distance condition 2.6e-13; its
    # bearing and se2 frameworks failed the numerical test.
    net = build("laman", 512, 127)
    assert net.attempts > 1
    assert is_well_conditioned(net.positions, net.edges)
    flat = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1e-11]])
    triangle = np.array([[0, 1], [1, 2], [0, 2]])
    assert is_infinitesimally_rigid(flat, triangle)
    assert not is_well_conditioned(flat, triangle, floor=ILL)


def test_small_uniform_is_buildable() -> None:
    for case in range(5):
        assert build("uniform", 8, case).attempts <= 200


# -- the formats ------------------------------------------------------------------


def test_json_and_npz_round_trip_exactly(tmp_path: Path) -> None:
    nets = [build("uniform", 16, c) for c in range(3)]
    for net in nets:
        _same(read_json(write_json(tmp_path, net)), net)
    archive = write_npz(tmp_path, nets)
    back = list(read_npz(archive))
    for a, b in zip(back, nets, strict=True):
        _same(a, b)
    _same(next(read_npz(archive, [2])), nets[2])


def test_npz_refuses_mixed_cells(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_npz(tmp_path, [build("uniform", 16, 0), build("uniform", 24, 0)])


def test_json_loads_in_dsim(tmp_path: Path) -> None:
    netio = pytest.importorskip("dsim.core.netio")
    net = build("bridged", 16, 0)
    loaded = netio.load_network(write_json(tmp_path, net))
    assert len(loaded) == net.n
    assert loaded.graph.number_of_edges() == net.m
    ids = sorted(loaded, key=lambda node: node.id)
    assert np.allclose([loaded.pos[node] for node in ids], net.positions)


# -- a whole (tiny) dataset --------------------------------------------------------


def test_build_and_open_a_dataset(tmp_path: Path) -> None:
    families = ["uniform", "wheel", "lattice"]
    summary = build_dataset(
        tmp_path,
        families,
        sizes={f: [8, 16] for f in families},
        cases={"uniform": 10, "wheel": 10, "lattice": 1},
        workers=1,
        progress=None,
    )
    assert summary["networks"] == 10 * 2 + 10 * 2 + 2 and summary["failed"] == 0
    srf = SRF(tmp_path)
    assert srf.families == families
    assert len(srf.rows(split="train", family="uniform", n=8)) == 7
    assert len(srf.rows(split="val", family="uniform", n=8)) == 1
    assert len(srf.rows(split="test", family="uniform", n=8)) == 2
    by_npz = list(srf.networks(family="uniform"))
    by_json = list(srf.networks(family="uniform", fmt="json"))
    assert len(by_npz) == len(by_json) == 20
    for a, b in zip(
        sorted(by_npz, key=lambda x: (x.n, x.case)),
        sorted(by_json, key=lambda x: (x.n, x.case)),
        strict=True,
    ):
        _same(a, b)
    _same(srf.network("wheel", 16, 3), build("wheel", 16, 3))
    for line in (tmp_path / "checksums.sha256").read_text().splitlines():
        digest, rel = line.split("  ")
        assert hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest() == digest
