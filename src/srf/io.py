"""Writing and reading the dataset, as JSON and as NPZ.

The two formats hold the same networks.

**JSON** is one file per network, ``json/<split>/<family>/n<NNN>/case<NNNN>.json``,
in DSim's ``dsim-network`` format (version 1): the file loads as it is with
:func:`dsim.core.netio.load_network`, and also with :func:`read_json` here,
which needs nothing but the standard library and numpy. Node ids run
``1..n`` in node order.

**NPZ** is one archive per split, family and size,
``npz/<split>/<family>/n<NNN>.npz``, holding every case of that cell:

========== ============ ==================================================
key        shape        what
========== ============ ==================================================
case       (k,)         case index
seed       (k,)         seed of the network's stream
attempts   (k,)         draws tried before it came out rigid
seconds    (k,)         wall time spent drawing it
field      (k, 2)       width and height of the area it was drawn in
comm_range (k,)         range its disc links were made at
positions  (k, n, 2)    coordinates, node order
edge_ptr   (k + 1,)     network ``c``'s links are ``edges[edge_ptr[c]:edge_ptr[c+1]]``
edges      (M, 2)       links as 0-based node indices, ``i < j``
family     ()           the cell's family
split      ()           the cell's split
========== ============ ==================================================

A labelled archive adds ``label_<name>`` per scalar label (see
:func:`srf.labels.label_fields`; ``distance_rigid_exact`` is ``-1`` for
undecided), ``label_basis`` and ``label_collinear`` with their ``_ptr``
offsets like ``edge_ptr``, and ``label_seconds``.

Node indices in NPZ are 0-based; ids in JSON are 1-based. Both loaders
return :class:`~srf.generate.Network` with 0-based indices.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from srf.generate import Network
from srf.labels import label_fields

__all__ = [
    "DSIM_FORMAT",
    "DSIM_VERSION",
    "json_path",
    "npz_path",
    "read_json",
    "read_npz",
    "sha256",
    "to_dsim_dict",
    "write_json",
    "write_npz",
]

DSIM_FORMAT = "dsim-network"
DSIM_VERSION = 1
GENERATOR = "srf"


def json_path(split: str, family: str, n: int, case: int) -> str:
    """Where a network's JSON file sits inside the dataset root."""
    return f"json/{split}/{family}/n{n:03d}/case{case:04d}.json"


def npz_path(split: str, family: str, n: int) -> str:
    """Where a cell's NPZ archive sits inside the dataset root."""
    return f"npz/{split}/{family}/n{n:03d}.npz"


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# -- JSON ---------------------------------------------------------------------


def to_dsim_dict(network: Network, version: str | None = None) -> dict[str, Any]:
    """The network as a ``dsim-network`` document."""
    r = network.comm_range
    return {
        "format": DSIM_FORMAT,
        "version": DSIM_VERSION,
        "meta": {
            "dsim_version": None,
            "saved_at": None,
            "generator": {
                "generator": GENERATOR,
                "generator_version": version,
                **network.meta(),
                **({"labels": _json_labels(network.extra)} if network.extra else {}),
            },
            "note": None,
            "skipped": [],
        },
        "network": {
            "environment": {
                "kind": "Environment2D",
                "shape": [network.field[0], network.field[1]],
                "obstacles": [],
            },
            "channel": {"kind": "StaticChannel", "from_edges": True},
            "comm_range": r,
            "network_routing": True,
            "directed": False,
            "labelling": None,
        },
        "nodes": [
            {
                "id": i + 1,
                "label": str(i + 1),
                "pos": [float(x), float(y)],
                "ori": 0.0,
                "comm_range": r,
                "status": None,
                "memory": {},
                "sensors": [],
            }
            for i, (x, y) in enumerate(network.positions.tolist())
        ],
        "edges": (network.edges + 1).tolist(),
    }


def _json_labels(extra: Any) -> dict[str, Any]:
    out = {key: extra[key] for key in label_fields()}
    out["basis"] = [int(k) for k in extra["basis"]]
    out["collinear"] = np.asarray(extra["collinear"], dtype=np.int64).reshape(-1, 3).tolist()
    out["seconds"] = float(extra["seconds"])
    return {key: (value.item() if isinstance(value, np.generic) else value) for key, value in out.items()}


def _labels_from_json(stored: Any) -> dict[str, Any]:
    if not stored:
        return {}
    out = dict(stored)
    out["basis"] = np.asarray(stored["basis"], dtype=np.int64)
    out["collinear"] = np.asarray(stored["collinear"], dtype=np.int64).reshape(-1, 3)
    return out


def write_json(root: Path, network: Network, version: str | None = None) -> Path:
    """Write one network's JSON file under ``root``; return its path."""
    path = root / json_path(network.split, network.family, network.n, network.case)
    path.parent.mkdir(parents=True, exist_ok=True)
    # repr() of a float round-trips exactly, and so does json's encoder.
    path.write_text(json.dumps(to_dsim_dict(network, version), separators=(",", ":")))
    return path


def read_json(path: str | Path) -> Network:
    """Read one network from its JSON file."""
    data = json.loads(Path(path).read_text())
    if data.get("format") != DSIM_FORMAT:
        raise ValueError(f"{path} is not a {DSIM_FORMAT} file.")
    made = data["meta"]["generator"]
    ids = [int(node["id"]) for node in data["nodes"]]
    where = {node_id: i for i, node_id in enumerate(ids)}
    edges = np.asarray(
        sorted(
            (min(where[a], where[b]), max(where[a], where[b])) for a, b in data["edges"]
        ),
        dtype=np.int32,
    ).reshape(-1, 2)
    width, height = data["network"]["environment"]["shape"]
    return Network(
        family=str(made["family"]),
        n=len(ids),
        case=int(made["case"]),
        split=str(made["split"]),
        seed=int(made["seed"]),
        positions=np.asarray([node["pos"] for node in data["nodes"]], dtype=np.float64),
        edges=edges,
        field=(float(width), float(height)),
        comm_range=float(data["network"]["comm_range"]),
        attempts=int(made["attempts"]),
        seconds=float(made["seconds"]),
        extra=_labels_from_json(made.get("labels")),
    )


# -- NPZ ----------------------------------------------------------------------


def write_npz(root: Path, networks: Sequence[Network]) -> Path:
    """Write one cell -- networks sharing split, family and size -- as NPZ."""
    if not networks:
        raise ValueError("An NPZ cell needs at least one network.")
    first = networks[0]
    cell = {(net.split, net.family, net.n) for net in networks}
    if len(cell) != 1:
        raise ValueError(f"One cell per archive, got {sorted(cell)}.")
    ordered = sorted(networks, key=lambda net: net.case)
    ptr = np.zeros(len(ordered) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([net.m for net in ordered])
    path = root / npz_path(first.split, first.family, first.n)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels: dict[str, Any] = {}
    if all(net.extra for net in ordered):
        for key in label_fields():
            values = [net.extra[key] for net in ordered]
            if key == "distance_rigid_exact":
                labels[f"label_{key}"] = np.asarray([-1 if v is None else int(v) for v in values], dtype=np.int8)
            elif key == "degeneracy":
                labels[f"label_{key}"] = np.asarray(values, dtype=str)
            else:
                labels[f"label_{key}"] = np.asarray(values)
        basis_ptr = np.zeros(len(ordered) + 1, dtype=np.int64)
        basis_ptr[1:] = np.cumsum([len(net.extra["basis"]) for net in ordered])
        labels["label_basis_ptr"] = basis_ptr
        labels["label_basis"] = np.concatenate([np.asarray(net.extra["basis"], dtype=np.int32) for net in ordered])
        collinear = [np.asarray(net.extra["collinear"], dtype=np.int32).reshape(-1, 3) for net in ordered]
        collinear_ptr = np.zeros(len(ordered) + 1, dtype=np.int64)
        collinear_ptr[1:] = np.cumsum([len(c) for c in collinear])
        labels["label_collinear_ptr"] = collinear_ptr
        labels["label_collinear"] = np.concatenate(collinear)
        labels["label_seconds"] = np.asarray([net.extra["seconds"] for net in ordered], dtype=np.float64)
    np.savez_compressed(
        path,
        **labels,
        case=np.asarray([net.case for net in ordered], dtype=np.int32),
        seed=np.asarray([net.seed for net in ordered], dtype=np.uint32),
        attempts=np.asarray([net.attempts for net in ordered], dtype=np.int32),
        seconds=np.asarray([net.seconds for net in ordered], dtype=np.float64),
        field=np.asarray([net.field for net in ordered], dtype=np.float64),
        comm_range=np.asarray([net.comm_range for net in ordered], dtype=np.float64),
        positions=np.stack([net.positions for net in ordered]),
        edge_ptr=ptr,
        edges=np.concatenate([net.edges for net in ordered]).astype(np.int32),
        family=np.asarray(first.family),
        split=np.asarray(first.split),
    )
    return path


def read_npz(path: str | Path, cases: Iterable[int] | None = None) -> Iterator[Network]:
    """Every network in one NPZ archive, or only the listed cases."""
    with np.load(Path(path)) as data:
        family, split = str(data["family"]), str(data["split"])
        index = {int(c): k for k, c in enumerate(data["case"])}
        wanted = sorted(index) if cases is None else list(cases)
        positions, ptr, edges = data["positions"], data["edge_ptr"], data["edges"]
        labelled = "label_basis" in data.files
        columns = {key: data[f"label_{key}"] for key in label_fields()} if labelled else {}
        if labelled:
            basis_ptr, basis = data["label_basis_ptr"], data["label_basis"]
            collinear_ptr, collinear = data["label_collinear_ptr"], data["label_collinear"]
            label_seconds = data["label_seconds"]
        for case in wanted:
            k = index[case]
            extra: dict[str, Any] = {}
            if labelled:
                for key, column in columns.items():
                    value = column[k].item()
                    if key == "distance_rigid_exact":
                        value = None if value == -1 else bool(value)
                    extra[key] = value
                extra["basis"] = basis[basis_ptr[k] : basis_ptr[k + 1]].astype(np.int64)
                extra["collinear"] = collinear[collinear_ptr[k] : collinear_ptr[k + 1]].astype(np.int64)
                extra["seconds"] = float(label_seconds[k])
            width, height = data["field"][k]
            yield Network(
                family=family,
                n=positions.shape[1],
                case=case,
                split=split,
                seed=int(data["seed"][k]),
                positions=positions[k].copy(),
                edges=edges[ptr[k] : ptr[k + 1]].copy(),
                field=(float(width), float(height)),
                comm_range=float(data["comm_range"][k]),
                attempts=int(data["attempts"][k]),
                seconds=float(data["seconds"][k]),
                extra=extra,
            )
