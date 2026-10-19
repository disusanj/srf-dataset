"""SRF -- Synthetic Rigidity Frameworks.

A dataset of planar networks for wireless sensor network rigidity, each
stored as positions and links, in DSim's JSON format and in NPZ.
"""

__version__ = "1.0.0"

from srf.dataset import SRF, build_dataset
from srf.generate import Network, build
from srf.io import read_json, read_npz

__all__ = ["SRF", "Network", "__version__", "build", "build_dataset", "read_json", "read_npz"]
