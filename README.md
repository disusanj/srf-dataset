# SRF — Synthetic Rigidity Frameworks

[![tests](https://github.com/disusanj/srf-dataset/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/disusanj/srf-dataset/actions/workflows/tests.yml)

A dataset of planar wireless sensor networks with rigidity labels. Each network is a *framework*: node positions plus the links between them. Every stored network is infinitesimally rigid and well conditioned under distances. Each one is labelled with generic, redundant and global rigidity, 3-connectivity, distance, bearing and SE(2) rigidity with margins and conditioning, and its collinear and parallel links. All labels are computed centrally with numpy.

## Get the data

The built corpus, SRF 1.0.0, is on Zenodo: https://doi.org/10.5281/zenodo.22991571 (CC BY 4.0). Every network can also be rebuilt from this code (see *Build*), and `sha256sum -c checksums.sha256` checks a copy.

## Install

```sh
pip install -e .            # numpy is the only dependency
pip install -e ".[test]"    # plus pytest
```

## Build

```sh
srf build <root> --workers 0      # the full dataset, every core
srf info <root>
```

## Read

```python
from srf import SRF

srf = SRF("<root>")
for net in srf.networks(split="train", family="uniform", n=64):   # NPZ, fast
    net.positions, net.edges, net.extra["globally_rigid"]
net = srf.network("wheel", 128, 7, fmt="json")                     # one file
```

The JSON files are DSim `dsim-network` documents and load with `dsim.core.netio.load_network`. They load without DSim too, through `srf.read_json`.

## Composition

- **Range:** R = 250 at every size.
- **Field:** a square 500 × 500 field below 64 nodes and 1000 × 1000 from 64 nodes on.
- **Split:** by case index within each family and size: the first 70% of cases are `train`, the next 10% `val`, the last 20% `test`.

| family | placement | sizes | per size |
|---|---|---|---|
| `uniform` | uniform in the field | 8–512 | 1000 |
| `complete` | disc of radius 0.49 R, every pair linked | 8–512 | 200 |
| `ring` | annulus of the field's area, hole wider than R, so no link joins opposite sides | 24–512 | 200 |
| `u_shape` | 800 × 700 box, 300-wide notch | 24–512 | 200 |
| `strip` | band R/2 wide, 25 units of length per node | 8–512 | 200 |
| `clusters` | 4 Gaussian clusters, σ = 50 | 16–512 | 200 |
| `bridged` | two blobs of radius 0.49 R, 500 apart, 3 bridge links | 8–512 | 200 |
| `wheel` | a hub plus a jittered rim of radius 200, no other links | 8–512 | 200 |
| `laman` | Henneberg type 1: each new node tied to 2 random ones of its 4 nearest | 8–512 | 200 |
| `lattice` | triangular lattice, spacing 100, ⌈√n⌉ nodes per row, not generic | 8–512 | 1 (test) |

Sizes are 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384 and 512, from each family's smallest size on.

**Drawing:** a network's random stream is a pure function of `(salt, family, n, case)`: `SeedSequence([crc32(salt), crc32(family), n, case])` with salt `srf-v1`, so any network can be regenerated alone and a new salt gives a disjoint corpus. Draws are repeated, up to 200 times, until the distance framework is infinitesimally rigid and well conditioned (distance condition at least 1e-10).

## Labels

These are stored per network in `networks.csv`, in each JSON file's `meta.generator.labels`, and in each NPZ archive as `label_*` arrays.

| label | meaning |
|---|---|
| `generic_rank`, `generic_rigid` | rank in the 2D rigidity matroid (pebble game), and whether it is 2n − 3 |
| `minimally_rigid` | rigid with exactly 2n − 3 links (Laman) |
| `redundant_rigid`, `redundant_links` | still rigid after losing any one link; links beyond the rank |
| `three_connected` | connected after removing any two nodes |
| `globally_rigid` | 3-connected and redundantly rigid (Jackson–Jordán) |
| `distance_rigid`, `bearing_rigid`, `se2_rigid` | numerical rank test at the stored positions; bearing uses Zhao and Zelazo's matrix |
| `*_margin`, `*_condition` | singular value just above the trivial motions, absolute and relative to the largest |
| `distance_rigid_exact` | the distance rank test in exact arithmetic modulo a prime: True proves infinitesimal rigidity at the stored positions, False means not certified (rank modulo a prime only bounds the rational rank from below), None (−1 in NPZ) undecided (more than 5000 links and the basis rows fall short) |
| `uniquely_localizable` | globally rigid and distance rigid: unique localizability with enough anchors, at generic positions; equals `globally_rigid` in SRF 1.0.0 |
| `degeneracy` | `flexible` (not generically rigid), `exact` (generically rigid, but the exact test does not certify the stored positions), `ill` (numerical test fails or distance condition < 1e-10) or `well` |
| `collinear_triples`, `collinear` | node triples `(u, v, w)` whose links `uv`, `vw` lie exactly on one line; the count and the list |
| `parallel_pairs`, `parallel_links` | link pairs exactly parallel (collinear ones included); links parallel to another |
| `collinear_angle`, `parallel_angle` | smallest angle (rad) between two links at a node, and between any two links |
| `basis` | one Laman spanning subgraph, as link indices |

## Files

```
index.json          salt, sizes, cases, constants, versions, failures
networks.csv        one row per network: name, split, degree, labels, files
checksums.sha256    every data file: sha256sum -c checksums.sha256
json/<split>/<family>/n<NNN>/case<NNNN>.json
npz/<split>/<family>/n<NNN>.npz
```

JSON node ids are 1-based; NPZ and the loaders use 0-based indices.

Each NPZ archive holds the `k` networks of one split, family and size as flat arrays:

| key | shape | what |
|---|---|---|
| `case`, `seed`, `attempts`, `seconds` | (k,) | case index, stream seed, draws tried, drawing time |
| `field`, `comm_range` | (k, 2), (k,) | area drawn in, link range |
| `positions` | (k, n, 2) | coordinates |
| `edge_ptr`, `edges` | (k + 1,), (M, 2) | network `c`'s links are `edges[edge_ptr[c]:edge_ptr[c+1]]`, `i < j` |
| `label_<name>` | (k,) | one per scalar label; `distance_rigid_exact` is −1 for undecided |
| `label_basis`, `label_collinear` (+ `_ptr`) | flat | variable-length labels, with offsets like `edge_ptr` |

`networks.csv` has one row per network: family, n, case, split, seed, links, mean/minimum/maximum degree, field, range, draws and drawing time, every scalar label, and the paths of its JSON and NPZ files.

## Citation

If you use SRF, please cite the dataset:

- Ž. Emeršič, D. Sušanj, *SRF: Synthetic Rigidity Frameworks*, version 1.0.0, Zenodo, 2026, https://doi.org/10.5281/zenodo.22991571.

`CITATION.cff` has the same in machine-readable form.

## License

The code is under the BSD 3-Clause License (`LICENSE`). The dataset on Zenodo is under Creative Commons Attribution 4.0 International (CC BY 4.0).

Contact: Diego Sušanj, diego.susanj@unipu.hr.
