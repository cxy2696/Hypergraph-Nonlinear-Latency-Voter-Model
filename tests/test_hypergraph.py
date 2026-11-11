import numpy as np
import pytest

from hnl.hypergraph import Hypergraph, random_hypergraph


def test_canonicalisation_and_deduplication():
    hg = Hypergraph(
        n_nodes=5,
        edges=[[2, 1, 2], [1, 2], [0], [3, 4, 3]],
        weights=np.ones(5),
    )
    assert hg.n_edges == 2
    assert [e.tolist() for e in hg.edge_list] == [[1, 2], [3, 4]]


def test_invalid_node_id():
    with pytest.raises(ValueError):
        Hypergraph(n_nodes=3, edges=[[0, 3]])


def test_invalid_weights():
    with pytest.raises(ValueError):
        Hypergraph(n_nodes=2, edges=[[0, 1]], weights=[1.0, 0.0])
    with pytest.raises(ValueError):
        Hypergraph(n_nodes=2, edges=[[0, 1]], weights=[1.0, np.nan])


def test_requires_at_least_one_edge():
    with pytest.raises(ValueError):
        Hypergraph(n_nodes=3, edges=[[0], [1]])


def test_components_and_giant_component():
    hg = Hypergraph(n_nodes=6, edges=[[0, 1], [1, 2], [3, 4]], weights=np.ones(6))
    labels, sizes = hg.components()
    assert sizes.tolist() == [3, 2, 1]
    assert labels[0] == labels[2] == 0
    assert labels[3] == labels[4] == 1
    assert labels[5] == 2
    assert set(hg.giant_component().tolist()) == {0, 1, 2}


def test_subgraph_relabels():
    hg = Hypergraph(n_nodes=6, edges=[[0, 1, 2], [2, 3], [4, 5]], weights=np.arange(1, 7, dtype=float))
    sub = hg.subgraph([2, 3, 4])
    assert sub.n_nodes == 3
    assert [e.tolist() for e in sub.edge_list] == [[0, 1]]
    assert sub.node_weights.tolist() == [3.0, 4.0, 5.0]


def test_summary_fields():
    hg = Hypergraph(n_nodes=4, edges=[[0, 1], [2, 3]], weights=np.ones(4))
    s = hg.summary()
    assert s["n_nodes"] == 4
    assert s["n_edges"] == 2
    assert s["n_components"] == 2
    assert s["largest_component"] == 2
    assert s["coverage"] == 1.0
    assert s["isolated_nodes"] == 0


def test_random_hypergraph_shapes():
    hg = random_hypergraph(40, 60, (3, 5), np.random.default_rng(0))
    assert hg.n_nodes == 40
    assert 1 <= hg.n_edges <= 60
    assert hg.sizes.min() >= 3
    assert hg.sizes.max() <= 5
    assert np.all(hg.node_weights > 0)
