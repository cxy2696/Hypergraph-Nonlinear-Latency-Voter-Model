import numpy as np
import pytest

from hnl.config import ModelConfig
from hnl.hypergraph import Hypergraph
from hnl.model import HNLVoterModel


def make_model(edges, weights, opinions, latencies, q=2.0, seed=0, **kwargs):
    return HNLVoterModel(
        nodes=np.arange(len(weights)),
        hyperedges=edges,
        influence_weights=weights,
        q=q,
        seed=seed,
        initial_opinions=opinions,
        latencies=latencies,
        consensus_threshold=0.99,
        max_steps=50,
        **kwargs,
    )


def test_kernel_matches_equation_two():
    hg = Hypergraph(n_nodes=2, edges=[[0, 1]], weights=[1.0, 1.0])
    model = HNLVoterModel.from_hypergraph(
        hg, ModelConfig(q=2.0, max_steps=5), seed=1,
        initial_opinions=[0, 1], latencies=[0, 0],
    )
    assert model.kernels()[0] == pytest.approx(0.5)

    model.opinions = np.array([1, 1])
    assert model.kernels()[0] == pytest.approx(1.0)
    model.opinions = np.array([0, 0])
    assert model.kernels()[0] == pytest.approx(0.0)


def test_kernel_uses_node_weights():
    # S0 = 1, S1 = 3, q = 2 -> 9 / (1 + 9) = 0.9
    hg = Hypergraph(n_nodes=2, edges=[[0, 1]], weights=[1.0, 3.0])
    model = HNLVoterModel.from_hypergraph(
        hg, ModelConfig(q=2.0), seed=1,
        initial_opinions=[0, 1], latencies=[0, 0],
    )
    assert model.kernels()[0] == pytest.approx(0.9)


def test_q_zero_gives_fair_coin():
    hg = Hypergraph(n_nodes=2, edges=[[0, 1]], weights=[1.0, 1.0])
    model = HNLVoterModel.from_hypergraph(
        hg, ModelConfig(q=0.0), seed=1,
        initial_opinions=[0, 1], latencies=[0, 0],
    )
    assert model.kernels()[0] == pytest.approx(0.5)


def test_node_mean_averages_incident_edges():
    # Edge A has all zeros -> kernel 0; edge B has all ones -> kernel 1.
    hg = Hypergraph(n_nodes=3, edges=[[0, 1], [0, 2]], weights=[1.0, 1.0, 1.0])
    model = HNLVoterModel.from_hypergraph(
        hg, ModelConfig(q=2.0, update_rule="node_mean"), seed=1,
        initial_opinions=[1, 0, 1], latencies=[0, 0, 0],
    )
    # node 0: edge [0,1] has scores (1,1) -> 0.5; edge [0,2] has (0,2) -> 1.0
    assert model.node_probabilities()[0] == pytest.approx(0.75)


def test_single_edge_node_sample_matches_node_mean():
    hg = Hypergraph(n_nodes=2, edges=[[0, 1]], weights=[1.0, 1.0])
    a = HNLVoterModel.from_hypergraph(
        hg, ModelConfig(q=2.0, update_rule="node_mean"), seed=3,
        initial_opinions=[0, 1], latencies=[0, 0],
    )
    b = HNLVoterModel.from_hypergraph(
        hg, ModelConfig(q=2.0, update_rule="node_sample"), seed=3,
        initial_opinions=[0, 1], latencies=[0, 0],
    )
    assert np.allclose(a.node_probabilities(), b.node_probabilities())


def test_consensus_is_absorbing():
    for opinion in (0, 1):
        hg = Hypergraph(n_nodes=4, edges=[[0, 1, 2], [1, 2, 3]], weights=np.ones(4))
        model = HNLVoterModel.from_hypergraph(
            hg, ModelConfig(q=1.5, max_steps=20, consensus_threshold=1.0), seed=5,
            initial_opinions=[opinion] * 4, latencies=[0, 0, 0, 0],
        )
        result = model.run()
        assert np.all(result.final_opinions == opinion)
        assert result.consensus_time == 1


def test_latency_blocks_updates_until_eligible():
    hg = Hypergraph(n_nodes=3, edges=[[0, 1, 2]], weights=np.ones(3))
    model = HNLVoterModel.from_hypergraph(
        hg, ModelConfig(q=2.0, max_steps=8, consensus_threshold=1.0), seed=0,
        initial_opinions=[0, 1, 1], latencies=[5, 5, 5],
    )
    result = model.run()
    assert list(result.eligible[:4]) == [0, 0, 0, 0]
    assert result.eligible[4] == 3


def test_isolated_nodes_are_frozen():
    hg = Hypergraph(n_nodes=3, edges=[[0, 1]], weights=np.ones(3))
    model = HNLVoterModel.from_hypergraph(
        hg, ModelConfig(q=2.0, max_steps=10), seed=0,
        initial_opinions=[0, 1, 1], latencies=[0, 0, 0],
    )
    result = model.run()
    assert result.final_opinions[2] == 1
    assert np.all(result.eligible <= 2)


def test_seed_reproducibility_and_stream_independence():
    hg = Hypergraph(n_nodes=30, edges=[[i, (i + 1) % 30] for i in range(30)], weights=np.ones(30))
    cfg = ModelConfig(q=1.5, max_steps=30)
    r1 = HNLVoterModel.from_hypergraph(hg, cfg, seed=11).run()
    r2 = HNLVoterModel.from_hypergraph(hg, cfg, seed=11).run()
    r3 = HNLVoterModel.from_hypergraph(hg, cfg, seed=12).run()
    assert np.array_equal(r1.final_opinions, r2.final_opinions)
    assert not np.array_equal(r1.final_opinions, r3.final_opinions) or r1.n_steps != r3.n_steps


def test_node_mean_is_invariant_to_edge_order():
    edges = [[0, 1, 2], [1, 2, 3], [0, 3, 4], [2, 4, 5]]
    hg1 = Hypergraph(n_nodes=6, edges=edges, weights=np.ones(6))
    hg2 = Hypergraph(n_nodes=6, edges=list(reversed(edges)), weights=np.ones(6))
    cfg = ModelConfig(q=1.5, max_steps=25, update_rule="node_mean")
    a = HNLVoterModel.from_hypergraph(hg1, cfg, seed=4, initial_opinions=[0, 1, 0, 1, 0, 1], latencies=[0] * 6).run()
    b = HNLVoterModel.from_hypergraph(hg2, cfg, seed=4, initial_opinions=[0, 1, 0, 1, 0, 1], latencies=[0] * 6).run()
    assert np.array_equal(a.final_opinions, b.final_opinions)


def test_edge_sweep_is_order_dependent():
    edges = [[0, 1, 2], [1, 2, 3], [0, 3, 4], [2, 4, 5]]
    cfg = ModelConfig(q=1.5, max_steps=30, update_rule="edge_sweep", consensus_threshold=1.0)
    differs = False
    for seed in range(20):
        hg1 = Hypergraph(n_nodes=6, edges=edges, weights=np.ones(6))
        hg2 = Hypergraph(n_nodes=6, edges=list(reversed(edges)), weights=np.ones(6))
        a = HNLVoterModel.from_hypergraph(hg1, cfg, seed=seed, initial_opinions=[0, 1, 0, 1, 0, 1], latencies=[0] * 6).run()
        b = HNLVoterModel.from_hypergraph(hg2, cfg, seed=seed, initial_opinions=[0, 1, 0, 1, 0, 1], latencies=[0] * 6).run()
        if not np.array_equal(a.final_opinions, b.final_opinions):
            differs = True
            break
    assert differs, "edge_sweep should depend on hyperedge order"


def test_validation_errors():
    hg = Hypergraph(n_nodes=2, edges=[[0, 1]], weights=[1.0, 1.0])
    with pytest.raises(ValueError):
        HNLVoterModel.from_hypergraph(hg, ModelConfig(), initial_opinions=[0])
    with pytest.raises(ValueError):
        HNLVoterModel.from_hypergraph(hg, ModelConfig(), initial_opinions=[0, 2])
    with pytest.raises(ValueError):
        HNLVoterModel.from_hypergraph(hg, ModelConfig(), latencies=[-1, 1])


def test_history_keys_are_backward_compatible():
    hg = Hypergraph(n_nodes=4, edges=[[0, 1], [2, 3]], weights=np.ones(4))
    model = HNLVoterModel.from_hypergraph(hg, ModelConfig(max_steps=5), seed=2)
    history = model.run_simulation(max_steps=5)
    assert set(["diversity", "cascade_size", "oscillation"]).issubset(history[0])
    assert 1 <= len(history) <= 5


def test_componentwise_consensus_is_absorbing():
    # Two disconnected edges, one all-0 and one all-1: every hyperedge is
    # monochromatic, so no node can flip even though the population is split.
    hg = Hypergraph(n_nodes=4, edges=[[0, 1], [2, 3]], weights=np.ones(4))
    model = HNLVoterModel.from_hypergraph(
        hg, ModelConfig(q=2.0, max_steps=20, consensus_threshold=1.0), seed=7,
        initial_opinions=[0, 0, 1, 1], latencies=[0, 0, 0, 0],
    )
    result = model.run()
    assert np.array_equal(result.final_opinions, [0, 0, 1, 1])
    assert int(result.changes.sum()) == 0


def test_mixed_connected_state_is_not_absorbing():
    hg = Hypergraph(n_nodes=2, edges=[[0, 1]], weights=np.ones(2))
    model = HNLVoterModel.from_hypergraph(
        hg, ModelConfig(q=2.0), seed=0,
        initial_opinions=[0, 1], latencies=[0, 0],
    )
    assert model.kernels()[0] == pytest.approx(0.5)
    assert 0 < model.node_probabilities()[0] < 1
