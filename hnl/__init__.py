"""Hypergraph–Nonlinear–Latency (HNL) voter model — reference implementation.

This package is the reproducible, tested implementation that accompanies the
HNL voter model paper in ``HNL Voter Model.pdf``.  Version 2 fixes several
specification and implementation problems in the original prototype; see
``docs/ERRATA.md`` for the complete list.

Public API
----------
``HNLVoterModel``   simulation engine
``ModelConfig``     validated simulation parameters
``Hypergraph``      static hypergraph container with structural diagnostics
``SimulationResult`` trajectories and scalar summaries of one run
``load_dataset``    dataset → hypergraph pipeline
"""

from .config import ModelConfig, LATENCY_DISTS, UPDATE_RULES
from .hypergraph import Hypergraph
from .model import HNLVoterModel, SimulationResult

__version__ = "2.3.0"

__all__ = [
    "HNLVoterModel",
    "ModelConfig",
    "SimulationResult",
    "Hypergraph",
    "LATENCY_DISTS",
    "UPDATE_RULES",
    "__version__",
]
