import numpy as np
import pandas as pd
import pytest

from hnl.data import build_dataset
from hnl.lexicon import classify, normalize_label


@pytest.fixture()
def toy_csv(tmp_path):
    df = pd.DataFrame(
        {
            "User": ["A ", "A", "B", "C", "D", "E"],
            "Sentiment": [" Positive ", " Joy ", " Negative ", " Bittersweet ", " Neutral ", " Anger "],
            "Hashtags": ["#X #Y", "#X", "#X", "#Y", "#Z", "#Z"],
            "Retweets": [1, 2, 3, 4, 5, 6],
            "Likes": [1, 2, 3, 4, 5, 6],
            "Platform": [" Twitter "] * 6,
        }
    )
    path = tmp_path / "toy.csv"
    df.to_csv(path, index=False)
    return path


def test_lexicon_normalisation():
    assert normalize_label(" Positive  ") == "positive"
    assert classify(" Positive ") == 1
    assert classify("Anger") == 0
    assert classify("Bittersweet") is None


def test_build_dataset_dedupes_and_maps(toy_csv):
    bundle = build_dataset(toy_csv, mapping="valence", scope="full")
    hg = bundle.hypergraph
    assert hg.n_nodes == 5  # trailing space in "A " merged with "A"
    assert hg.n_edges == 3
    assert [e.tolist() for e in hg.edge_list] == [[0, 1], [0, 2], [3, 4]]
    # A positive+joy -> 1, B negative -> 0, C bittersweet -> unmapped, D neutral -> unmapped,
    # E anger -> 0
    assert bundle.valence.tolist() == [1, 0, -1, -1, 0]
    assert hg.node_weights.max() == pytest.approx(1.0)
    assert hg.node_weights.min() == pytest.approx(3 / 12)


def test_scope_active_drops_isolated():
    df = pd.DataFrame(
        {
            "User": ["A", "B", "C", "D"],
            "Sentiment": ["Positive", "Negative", "Joy", "Neutral"],
            "Hashtags": ["#X #Y", "#X", "#Y", "#Only"],
            "Retweets": [1, 1, 1, 1],
            "Likes": [1, 1, 1, 1],
        }
    )
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "t.csv"
        df.to_csv(path, index=False)
        full = build_dataset(path, scope="full")
        active = build_dataset(path, scope="active")
    assert full.hypergraph.n_nodes == 4
    assert full.hypergraph.summary()["isolated_nodes"] == 1
    assert active.hypergraph.n_nodes == 3


def test_opinions_for_is_deterministic(toy_csv):
    bundle = build_dataset(toy_csv, scope="active")
    a = bundle.opinions_for("data", np.random.default_rng(3))
    b = bundle.opinions_for("data", np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert set(np.unique(a)).issubset({0, 1})


def test_legacy_load_data(toy_csv, monkeypatch):
    import model

    nodes, edges, weights, opinions = model.load_data(str(toy_csv), "csv")
    assert len(nodes) == 5
    assert len(weights) == 5
    assert len(opinions) == 5
    assert set(np.unique(opinions)).issubset({0, 1})
