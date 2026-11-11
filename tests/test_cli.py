import json

import pandas as pd

from hnl.cli import main


def test_simulate_prints_json(capsys):
    code = main(["simulate", "--n", "30", "--edges", "40", "--steps", "10", "--seed", "1"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["n_nodes"] == 30
    assert 1 <= payload["steps_run"] <= 10


def test_diagnose_writes_markdown(tmp_path):
    df = pd.DataFrame(
        {
            "User": ["A", "B", "C"],
            "Sentiment": ["Positive", "Negative", "Joy"],
            "Hashtags": ["#X", "#X", "#X"],
            "Retweets": [1, 2, 3],
            "Likes": [1, 2, 3],
        }
    )
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "diag.md"
    code = main(["diagnose", "--csv", str(csv), "--scope", "active", "--out", str(out)])
    assert code == 0
    text = out.read_text()
    assert "Dataset structure" in text
    assert "Structural implication" in text
