"""Draw the networks of the SRF dataset.

A network is named by ``(family, n, case)`` and nothing else: its random
stream is a pure function of that name and of :data:`SALT`, so the same
name always gives the same network, and a different salt gives a corpus
that shares no stream with this one.

Splits are fixed by case index, per family and size: the first 70% of
the cases are ``train``, the next 10% ``val``, the last 20% ``test``.
``lattice`` has one network per size, the same at every draw, and it is
put in ``test``.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from srf.families import FAMILIES
from srf.matrix import is_well_conditioned

__all__ = [
    "ATTEMPTS",
    "CASES",
    "FRACTIONS",
    "SALT",
    "SIZES",
    "SIZE_STEPS",
    "SPLITS",
    "Network",
    "NetworkError",
    "build",
    "plan",
    "split_of",
    "stream",
]

#: Mixed into every stream, so this corpus shares no seed with any other.
SALT = "srf-v1"
#: Draws one network may work through before it is given up.
ATTEMPTS = 200
SPLITS = ("train", "val", "test")
FRACTIONS = (0.7, 0.1, 0.2)

#: Powers of two from 8 to 512 and the steps halfway between them.
SIZE_STEPS = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
#: Sizes each family is built at (below a family's ``min_n`` it is skipped).
SIZES: dict[str, tuple[int, ...]] = {name: SIZE_STEPS for name in FAMILIES}
#: Networks per family and size.
CASES: dict[str, int] = {
    name: 1000 if name == "uniform" else (1 if name == "lattice" else 200)
    for name in FAMILIES
}


class NetworkError(RuntimeError):
    """No acceptable network could be drawn."""


@dataclass(frozen=True, slots=True)
class Network:
    """One network of the dataset.

    Args:
        family: Which family it belongs to.
        n: Number of nodes.
        case: Its index within the family at this size.
        split: ``train``, ``val`` or ``test``.
        seed: Seed of the stream it was drawn from.
        positions: ``(n, 2)`` coordinates, in node order.
        edges: ``(m, 2)`` links, ``i < j``, sorted.
        field: ``(width, height)`` of the area it was drawn in.
        comm_range: The range its disc links were made at.
        attempts: Draws tried, the accepted one included.
        seconds: Wall time spent drawing and testing it.
    """

    family: str
    n: int
    case: int
    split: str
    seed: int
    positions: NDArray[np.float64]
    edges: NDArray[np.int32]
    field: tuple[float, float]
    comm_range: float
    attempts: int = 1
    seconds: float = 0.0
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def m(self) -> int:
        """Number of links."""
        return len(self.edges)

    def degrees(self) -> NDArray[np.int64]:
        """Each node's number of links."""
        return np.bincount(self.edges.ravel(), minlength=self.n)

    def meta(self) -> dict[str, Any]:
        """The scalar description of this network, as stored beside it."""
        degree = self.degrees()
        return {
            "family": self.family,
            "n": self.n,
            "case": self.case,
            "split": self.split,
            "seed": self.seed,
            "m": self.m,
            "mean_degree": float(degree.mean()),
            "min_degree": int(degree.min()),
            "max_degree": int(degree.max()),
            "field_width": self.field[0],
            "field_height": self.field[1],
            "comm_range": self.comm_range,
            "attempts": self.attempts,
            "seconds": self.seconds,
        }


def stream(family: str, n: int, case: int, salt: str = SALT) -> int:
    """The seed of a network's random stream: a pure function of its name."""
    words = [zlib.crc32(salt.encode()), zlib.crc32(family.encode()), n, case]
    return int(np.random.SeedSequence(words).generate_state(1)[0])


def split_of(family: str, case: int, cases: int | None = None) -> str:
    """Which split a case belongs to."""
    if family == "lattice":
        return "test"
    total = CASES[family] if cases is None else cases
    if not 0 <= case < total:
        raise ValueError(f"case {case} outside 0..{total - 1} for {family}.")
    train = round(FRACTIONS[0] * total)
    val = train + round(FRACTIONS[1] * total)
    return "train" if case < train else "val" if case < val else "test"


def _canonical(edges: NDArray[np.int32]) -> NDArray[np.int32]:
    """Each link low end first, sorted, each once."""
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.unique(np.sort(edges, axis=1), axis=0).astype(np.int32)


def build(family: str, n: int, case: int, cases: int | None = None) -> Network:
    """Draw the network the dataset stores at ``(family, n, case)``.

    Raises:
        NetworkError: If no draw in :data:`ATTEMPTS` was rigid and well
            conditioned (see :data:`srf.matrix.ILL`), or the
            family does not make sense at ``n``.
    """
    kind = FAMILIES[family]
    if n < kind.min_n:
        raise NetworkError(f"{family} needs at least {kind.min_n} nodes, not {n}.")
    started = perf_counter()
    seed = stream(family, n, case)
    rng = np.random.default_rng(seed)
    for attempt in range(1, ATTEMPTS + 1):
        drawn = kind.draw(n, rng)
        edges = _canonical(drawn.edges)
        if len(edges) and is_well_conditioned(drawn.positions, edges):
            return Network(
                family=family,
                n=n,
                case=case,
                split=split_of(family, case, cases),
                seed=seed,
                positions=np.asarray(drawn.positions, dtype=np.float64),
                edges=edges,
                field=(float(drawn.field[0]), float(drawn.field[1])),
                comm_range=float(drawn.comm_range),
                attempts=attempt,
                seconds=perf_counter() - started,
            )
        if not kind.random:
            break
    raise NetworkError(f"No rigid {family} network of {n} nodes (case {case}).")


def plan(
    families: Iterable[str] | None = None,
    sizes: Mapping[str, Iterable[int]] | None = None,
    cases: Mapping[str, int] | None = None,
) -> Iterator[tuple[str, int, int]]:
    """Every ``(family, size, cases)`` cell the dataset holds."""
    for family in families if families is not None else FAMILIES:
        count = (cases or CASES).get(family, CASES[family])
        for n in (sizes or SIZES).get(family, SIZES[family]):
            if n >= FAMILIES[family].min_n:
                yield family, n, count
