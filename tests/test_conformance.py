"""Check the paper's equations against the shared conformance fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hnl.config import ModelConfig
from hnl.hypergraph import Hypergraph
from hnl.model import HNLVoterModel

FIXTURE = Path(__file__).parents[1] / "conformance" / "binary_case.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_shared_binary_fixture(case: dict) -> None:
    graph = Hypergraph(
        n_nodes=case["n_nodes"],
        edges=case["edges"],
        weights=case["weights"],
    )
    model = HNLVoterModel.from_hypergraph(
        graph,
        ModelConfig(q=case["q"], latency_dist="fixed", mu=1.0, max_steps=1),
        seed=0,
        initial_opinions=case["opinions"],
        latencies=[1] * case["n_nodes"],
    )
    np.testing.assert_allclose(model.kernels(), case["expected_edge_p1"], atol=1e-12)
    np.testing.assert_allclose(
        model.node_probabilities(), case["expected_node_p1"], atol=1e-12
    )
