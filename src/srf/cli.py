"""``srf build`` draws a dataset; ``srf info`` summarises one."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from srf.dataset import SRF, build_dataset
from srf.families import FAMILIES
from srf.generate import CASES, SIZES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="srf", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="draw a dataset into a directory")
    build.add_argument("root", type=Path)
    build.add_argument("--families", nargs="+", choices=list(FAMILIES), default=None)
    build.add_argument("--sizes", type=int, nargs="+", default=None, help="the same sizes for every family")
    build.add_argument("--cases", type=int, default=None, help="networks per size for every family")
    build.add_argument("--workers", type=int, default=1, help="0 for every core")
    info = sub.add_parser("info", help="summarise a dataset")
    info.add_argument("root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        families = args.families or list(FAMILIES)
        sizes = {f: args.sizes for f in families} if args.sizes else None
        cases = {f: (1 if f == "lattice" else args.cases) for f in families} if args.cases else None
        workers = args.workers or os.cpu_count() or 1
        summary = build_dataset(args.root, families, sizes, cases, workers)
        print(json.dumps(summary, indent=2))
        return 0
    srf = SRF(args.root)
    counts = Counter((row["family"], row["split"]) for row in srf.table)
    for family in srf.families:
        split = {s: counts[(family, s)] for s in ("train", "val", "test")}
        print(f"{family:>9} sizes={srf.sizes(family)} {split}")
    print(f"{len(srf.table)} networks; defaults: sizes={SIZES['uniform']}, cases={CASES['uniform']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
