# Changelog

## 1.0.0

First public release. It builds SRF version 1.0.0: 32 213 planar networks from ten families, 8 to 512 nodes, salt `srf-v1`.

- Families: `uniform`, `complete`, `ring`, `u_shape`, `strip`, `clusters`, `bridged`, `wheel`, `laman`, `lattice`, on the sizes 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384 and 512.
- A draw is kept only if its distance framework is infinitesimally rigid and well conditioned (distance condition at least 1e-10).
- Labels: generic rank and a Laman basis (pebble game); minimal, redundant and global rigidity; 3-connectivity; distance, bearing (Zhao–Zelazo) and SE(2) rank tests with margins and conditioning; an exact rank certificate; a degeneracy class; exactly collinear and parallel links.
- Formats: dsim-network v1 JSON per network and NPZ per split, family and size, with loaders for both. `networks.csv` and `checksums.sha256` are written with them.
