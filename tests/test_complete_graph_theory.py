"""Finite-n checks for the complete-hypergraph theorem in docs/MODEL.md."""

from __future__ import annotations

import numpy as np
import pytest

from hnl.config import ModelConfig
from hnl.hypergraph import Hypergraph
from hnl.model import HNLVoterModel


def _f(p: float, q: float) -> float:
    return p**q / (p**q + (1.0 - p) ** q)


def test_complete_graph_kernel_is_constant() -> None:
    n = 10
    opinions = [1, 1, 1] + [0] * (n - 3)
    graph = Hypergraph(n_nodes=n, edges=[tuple(range(n))], weights=[1.0] * n)
    model = HNLVoterModel.from_hypergraph(
        graph,
        ModelConfig(q=1.5, latency_dist="fixed", mu=1, max_steps=1),
        seed=0,
        initial_opinions=opinions,
        latencies=[1] * n,
    )
    expected = _f(0.3, 1.5)
    np.testing.assert_allclose(model.node_probabilities(), expected, atol=1e-12)


@pytest.mark.parametrize("q,sign", [(1.5, 1.0), (0.5, -1.0)])
def test_drift_sign_away_from_one_half(q: float, sign: float) -> None:
    assert np.sign(_f(0.7, q) - 0.7) == sign
    assert np.sign(_f(0.3, q) - 0.3) == -sign


def test_q1_consensus_probability_equals_initial_share() -> None:
    n = 20
    ones = 6
    graph = Hypergraph(n_nodes=n, edges=[tuple(range(n))], weights=[1.0] * n)
    cfg = ModelConfig(
        q=1.0,
        latency_dist="fixed",
        mu=1,
        max_steps=400,
        consensus_threshold=1.0,
    )
    hits = 0
    reps = 500
    for seed in range(reps):
        model = HNLVoterModel.from_hypergraph(
            graph,
            cfg,
            seed=seed,
            initial_opinions=[1] * ones + [0] * (n - ones),
            latencies=[1] * n,
        )
        result = model.run()
        assert result.consensus_time is not None
        hits += int(result.final_opinions.sum() == n)
    share = hits / reps
    assert abs(share - ones / n) < 0.06, share
