"""Random-hypergraph demo (v2).

Example
-------
    python random_simulation.py --q 2.0 --latency_dist exponential --mu 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

import numpy as np

from hnl.config import ModelConfig
from hnl.hypergraph import random_hypergraph
from hnl.model import HNLVoterModel


def simulate_hnl_voter_random(
    n: int = 100,
    num_hyperedges: int = 200,
    hyperedge_size_range=(3, 6),
    q: float = 1.5,
    latency_dist: str = "normal",
    mu: float = 5.0,
    sigma: float = 2.0,
    lambda_param: Optional[float] = None,
    consensus_threshold: float = 0.8,
    max_steps: int = 100,
    update_rule: str = "node_mean",
    seed: Optional[int] = None,
    plot: bool = True,
    plot_path: str = "hnl_voter_metrics.png",
):
    """Simulate one random hypergraph and return the per-step history."""
    rng = np.random.default_rng(seed)
    hypergraph = random_hypergraph(n, num_hyperedges, hyperedge_size_range, rng)
    cfg = ModelConfig(
        q=q,
        latency_dist=latency_dist,
        mu=mu,
        sigma=sigma,
        lambda_param=lambda_param,
        consensus_threshold=consensus_threshold,
        max_steps=max_steps,
        update_rule=update_rule,
    )
    model = HNLVoterModel.from_hypergraph(hypergraph, cfg, seed=seed)
    result = model.run()
    if plot:
        model.plot_metrics(plot_path)
    return result.history


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--edges", type=int, default=200)
    parser.add_argument("--q", type=float, default=1.5)
    parser.add_argument("--latency_dist", choices=("normal", "exponential", "fixed"), default="normal")
    parser.add_argument("--mu", type=float, default=5.0)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--lambda_param", type=float, default=None)
    parser.add_argument("--consensus_threshold", type=float, default=0.8)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--update_rule", choices=("node_mean", "node_sample", "edge_sweep"), default="node_mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="hnl_voter_metrics.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    history = simulate_hnl_voter_random(
        n=args.n,
        num_hyperedges=args.edges,
        q=args.q,
        latency_dist=args.latency_dist,
        mu=args.mu,
        sigma=args.sigma,
        lambda_param=args.lambda_param,
        consensus_threshold=args.consensus_threshold,
        max_steps=args.max_steps,
        update_rule=args.update_rule,
        seed=args.seed,
        plot=not args.no_plot,
        plot_path=args.out,
    )
    print(
        json.dumps(
            {
                "steps": len(history),
                "final_diversity": round(history[-1]["diversity"], 4),
                "final_majority": round(history[-1]["majority"], 4),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
