"""The SRF dataset on disk: building it, and opening it again.

A dataset root holds::

    index.json          what was built: salt, sizes, cases, constants, versions
    networks.csv        one row per network: name, split, size, degree, files
    checksums.sha256    every data file; ``sha256sum -c checksums.sha256``
    json/<split>/<family>/n<NNN>/case<NNNN>.json
    npz/<split>/<family>/n<NNN>.npz

Open one with :class:`SRF` and read either format::

    srf = SRF("path/to/root")
    for net in srf.networks(split="train", family="uniform", n=64):
        ...
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

from srf import __version__
from srf.families import constants
from srf.generate import (
    ATTEMPTS,
    CASES,
    FRACTIONS,
    SALT,
    SIZES,
    Network,
    NetworkError,
    build,
    plan,
)
from srf.io import json_path, npz_path, read_json, read_npz, sha256, write_json, write_npz
from srf.labels import label, label_fields

__all__ = ["INDEX", "MANIFEST", "SRF", "TABLE", "build_dataset"]

INDEX = "index.json"
TABLE = "networks.csv"
MANIFEST = "checksums.sha256"
#: Cases one worker task draws, so a slow size spreads over every worker.
TASK = 25
_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

Format = Literal["npz", "json"]


def _chunk(
    task: tuple[str, str, int, int, int, int, bool],
) -> tuple[list[Network], list[dict[str, Any]]]:
    """Draw (and label) cases ``start..stop`` of one cell, writing their JSON."""
    root, family, n, count, start, stop, labelled = task
    built, failed = [], []
    for case in range(start, stop):
        try:
            net = build(family, n, case, count)
        except NetworkError as exc:
            failed.append({"family": family, "n": n, "case": case, "error": str(exc)})
            continue
        if labelled:
            net = replace(net, extra=label(net.positions, net.edges))
        write_json(Path(root), net, __version__)
        built.append(net)
    return built, failed


@contextmanager
def _single_threaded() -> Iterator[None]:
    """One BLAS thread in every worker started inside this block.

    The rank test is an SVD and numpy runs it on every core it can see; a
    pool of workers each doing that fights itself for the machine.
    Spawned workers read these when they import numpy.
    """
    saved = {name: os.environ.get(name) for name in _THREADS}
    os.environ.update(dict.fromkeys(_THREADS, "1"))
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run(tasks: Sequence[Any], workers: int) -> Iterator[Any]:
    if workers <= 1:
        yield from map(_chunk, tasks)
        return
    with (
        _single_threaded(),
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool,
    ):
        yield from pool.map(_chunk, tasks)


def build_dataset(
    root: Path,
    families: Iterable[str] | None = None,
    sizes: Mapping[str, Iterable[int]] | None = None,
    cases: Mapping[str, int] | None = None,
    workers: int = 1,
    progress: Callable[[str], None] | None = print,
    labels: bool = True,
) -> dict[str, Any]:
    """Build a dataset under ``root``, both formats, with index and checksums.

    Args:
        root: Directory to build in; created if missing.
        families: Families to build; all of them by default.
        sizes: Family to sizes, overriding :data:`~srf.generate.SIZES`.
        cases: Family to networks per size, overriding
            :data:`~srf.generate.CASES`.
        workers: Processes to draw on at once.
        progress: Called with a line per finished cell; None for silence.
        labels: Compute every rigidity label (:mod:`srf.labels`) and store
            it with the network in both formats and in the table.

    Returns:
        A summary: networks built and failed, bytes per format, core seconds.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cells = list(plan(families, sizes, cases))
    tasks = [
        (str(root), family, n, count, start, min(start + TASK, count), labels)
        for family, n, count in cells
        for start in range(0, count, TASK)
    ]
    remaining = {(f, n): -(-count // TASK) for f, n, count in cells}
    pending: dict[tuple[str, int], list[Network]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    held: dict[str, dict[str, list[int]]] = defaultdict(dict)
    files: list[str] = []

    for built, lost in _run(tasks, workers):
        failed += lost
        if not built and not lost:
            continue
        key = (built or lost)[0]
        key = (key.family, key.n) if isinstance(key, Network) else (key["family"], key["n"])
        pending[key] += built
        remaining[key] -= 1
        if remaining[key]:
            continue
        family, n = key
        nets = sorted(pending.pop(key), key=lambda net: net.case)
        by_split: dict[str, list[Network]] = defaultdict(list)
        for net in nets:
            by_split[net.split].append(net)
        for split, group in by_split.items():
            archive = write_npz(root, group)
            files.append(str(archive.relative_to(root)))
            for net in group:
                path = json_path(split, family, n, net.case)
                files.append(path)
                scalars = {key: net.extra[key] for key in label_fields()} if net.extra else {}
                if net.extra:
                    scalars["label_seconds"] = net.extra["seconds"]
                rows.append(
                    {**net.meta(), **scalars, "json": path, "npz": str(archive.relative_to(root))}
                )
        held[family][str(n)] = [net.case for net in nets]
        if progress:
            lost_here = sum(1 for f in failed if (f["family"], f["n"]) == key)
            progress(f"{family:>9} n={n:<4} {len(nets)} built" + (f", {lost_here} failed" if lost_here else ""))

    order = {name: i for i, name in enumerate(dict.fromkeys(f for f, _, _ in cells))}
    rows.sort(key=lambda row: (order[row["family"]], row["n"], row["case"]))
    with (root / TABLE).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["family"])
        writer.writeheader()
        writer.writerows(rows)

    index = {
        "name": "SRF",
        "title": "Synthetic Rigidity Frameworks",
        "srf_version": __version__,
        "salt": SALT,
        "attempts": ATTEMPTS,
        "splits": {"train": FRACTIONS[0], "val": FRACTIONS[1], "test": FRACTIONS[2]},
        "families": {
            family: {
                "per_size": (cases or CASES).get(family, CASES[family]),
                "sizes": list((sizes or SIZES).get(family, SIZES[family])),
                "cases": held[family],
            }
            for family in order
        },
        "failed": failed,
        "constants": constants(),
    }
    (root / INDEX).write_text(json.dumps(index, indent=2) + "\n")
    files.sort()
    with (root / MANIFEST).open("w") as handle:
        for rel in files:
            handle.write(f"{sha256(root / rel)}  {rel}\n")

    json_bytes = sum((root / row["json"]).stat().st_size for row in rows)
    npz_bytes = sum((root / rel).stat().st_size for rel in {row["npz"] for row in rows})
    return {
        "networks": len(rows),
        "failed": len(failed),
        "json_bytes": json_bytes,
        "npz_bytes": npz_bytes,
        "core_seconds": sum(row["seconds"] + row.get("label_seconds", 0.0) for row in rows),
        "draw_seconds": sum(row["seconds"] for row in rows),
        "label_seconds": sum(row.get("label_seconds", 0.0) for row in rows),
        "degeneracy": {
            kind: sum(1 for row in rows if row.get("degeneracy") == kind)
            for kind in ("flexible", "exact", "ill", "well")
        },
    }


class SRF:
    """An SRF dataset on disk.

    Args:
        root: The dataset's root directory (the one holding ``index.json``).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index: dict[str, Any] = json.loads((self.root / INDEX).read_text())
        with (self.root / TABLE).open(newline="") as handle:
            self.table: list[dict[str, str]] = list(csv.DictReader(handle))

    @property
    def families(self) -> list[str]:
        """Families this dataset holds."""
        return list(self.index["families"])

    def sizes(self, family: str) -> list[int]:
        """Sizes a family is held at."""
        return sorted(int(n) for n in self.index["families"][family]["cases"])

    def rows(
        self,
        split: str | None = None,
        family: str | None = None,
        n: int | None = None,
    ) -> list[dict[str, str]]:
        """Table rows matching the filters."""
        return [
            row
            for row in self.table
            if (split is None or row["split"] == split)
            and (family is None or row["family"] == family)
            and (n is None or int(row["n"]) == n)
        ]

    def networks(
        self,
        split: str | None = None,
        family: str | None = None,
        n: int | None = None,
        fmt: Format = "npz",
    ) -> Iterator[Network]:
        """Every network matching the filters, read from either format.

        NPZ reads a whole cell per archive, so it is the fast path; JSON
        reads one file per network and gives the same networks.
        """
        chosen = self.rows(split, family, n)
        if fmt == "json":
            for row in chosen:
                yield read_json(self.root / row["json"])
            return
        if fmt != "npz":
            raise ValueError(f"fmt must be 'npz' or 'json', not {fmt!r}.")
        by_archive: dict[str, list[int]] = defaultdict(list)
        for row in chosen:
            by_archive[row["npz"]].append(int(row["case"]))
        for archive, wanted in by_archive.items():
            yield from read_npz(self.root / archive, wanted)

    def network(self, family: str, n: int, case: int, fmt: Format = "npz") -> Network:
        """One network by name."""
        for row in self.rows(family=family, n=n):
            if int(row["case"]) == case:
                if fmt == "json":
                    return read_json(self.root / row["json"])
                return next(read_npz(self.root / row["npz"], [case]))
        raise KeyError(f"No {family} network n={n} case={case} in {self.root}.")

    def path(self, family: str, n: int, case: int, fmt: Format = "json") -> Path:
        """The file a network is stored in."""
        for row in self.rows(family=family, n=n):
            if int(row["case"]) == case:
                split = row["split"]
                rel = json_path(split, family, n, case) if fmt == "json" else npz_path(split, family, n)
                return self.root / rel
        raise KeyError(f"No {family} network n={n} case={case} in {self.root}.")
