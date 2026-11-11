"""Backward-compatible entry points for the v1 prototype API.

The implementation lives in :mod:`hnl`.  The class and functions below keep the
names and call signatures used by the original scripts, so old code such as::

    from model import HNLVoterModel, load_data

continues to work.  Behavioural differences from v1 are documented in
``docs/ERRATA.md``; the original code is preserved at tag ``v1.0-legacy``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from hnl.config import ModelConfig
from hnl.data import build_dataset
from hnl.model import HNLVoterModel, SimulationResult

__all__ = [
    "HNLVoterModel",
    "SimulationResult",
    "ModelConfig",
    "load_data",
    "run_model_from_file",
]


def load_data(
    file_path: str,
    format_type: Optional[str] = None,
    sentiment_col: str = "Sentiment",
    user_col: str = "User",
    hashtags_col: str = "Hashtags",
    retweets_col: str = "Retweets",
    likes_col: str = "Likes",
):
    """Load a post table into ``(nodes, hyperedges, influence_weights, opinions)``.

    Sentiment labels are whitespace-normalised and mapped through the documented
    valence lexicon.  Users whose labels are all ambiguous receive opinion 0;
    see :func:`hnl.data.build_dataset` for the explicit, non-filling API.
    """
    bundle = build_dataset(
        file_path,
        mapping="valence",
        scope="full",
        fmt=format_type,
        user_col=user_col,
        sentiment_col=sentiment_col,
        hashtags_col=hashtags_col,
        retweet_col=retweets_col,
        like_col=likes_col,
    )
    nodes = np.asarray(bundle.labels)
    hyperedges = [edge.tolist() for edge in bundle.hypergraph.edge_list]
    influence_weights = bundle.hypergraph.node_weights
    opinions = bundle.valence.copy()
    opinions[opinions < 0] = 0
    return nodes, hyperedges, influence_weights, opinions


def run_model_from_file(
    file_path: str,
    format_type: Optional[str] = None,
    q: float = 1.5,
    latency_dist: str = "normal",
    mu: float = 5.0,
    sigma: float = 2.0,
    lambda_param: Optional[float] = None,
    consensus_threshold: float = 0.8,
    max_steps: int = 100,
    update_rule: str = "node_mean",
    seed: Optional[int] = None,
    init: str = "data",
    plot: bool = True,
    plot_path: str = "hnl_voter_metrics.png",
):
    """Load a file, run the model and return the per-step history."""
    nodes, hyperedges, influence_weights, opinions = load_data(file_path, format_type)
    model = HNLVoterModel(
        nodes,
        hyperedges,
        influence_weights,
        q=q,
        latency_dist=latency_dist,
        mu=mu,
        sigma=sigma,
        lambda_param=lambda_param,
        consensus_threshold=consensus_threshold,
        update_rule=update_rule,
        seed=seed,
        initial_opinions=opinions if init == "data" else None,
        max_steps=max_steps,
    )
    history = model.run_simulation(max_steps=max_steps)
    if plot:
        model.plot_metrics(plot_path)
    return history
