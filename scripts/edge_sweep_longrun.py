#!/usr/bin/env python3
"""Supplementary long-horizon check of the paper's edge-sweep update rule.

The main experiment uses the paper's 100-sweep horizon.  With heterogeneous
latency the literal edge sweep does not get close to the 80 % threshold in that
window, while the node-centric rules do.  This script checks whether the edge
sweep is *slow* or *non-absorbing* by running it on a smaller hypergraph for up
to 20,000 sweeps.

Usage::

    python scripts/edge_sweep_longrun.py --out results/edge_sweep_longrun.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hnl.config import ModelConfig
from hnl.hypergraph import random_hypergraph
from hnl.model import HNLVoterModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--edges", type=int, default=60)
    parser.add_argument("--q", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default="results/edge_sweep_longrun.csv")
    args = parser.parse_args()

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        hg = random_hypergraph(args.n, args.edges, (3, 5), rng)
        initial = rng.integers(0, 2, args.n)
        uniforms = rng.random(args.n)
        for rule in ("edge_sweep", "node_sample", "node_mean"):
            cfg = ModelConfig(
                q=args.q,
                latency_dist="normal",
                mu=5.0,
                sigma=2.0,
                update_rule=rule,
                max_steps=args.steps,
                consensus_threshold=0.8,
            )
            latencies = HNLVoterModel.latencies_from_uniforms(uniforms, cfg)
            model = HNLVoterModel.from_hypergraph(
                hg, cfg, seed=seed, initial_opinions=initial, latencies=latencies
            )
            result = model.run()
            rows.append(
                {
                    "n": args.n,
                    "m": args.edges,
                    "q": args.q,
                    "rule": rule,
                    "seed": seed,
                    "steps_run": result.n_steps,
                    "consensus_time": result.consensus_time,
                    "censored_at_20000": result.censored,
                    "final_diversity": result.final_diversity,
                }
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
