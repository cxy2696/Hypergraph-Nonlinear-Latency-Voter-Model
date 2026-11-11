"""Simulation engine for the Hypergraph–Nonlinear–Latency voter model.

The model is specified formally in ``docs/MODEL.md``.  In one sweep ``t``:

1. For every hyperedge ``e`` compute the influence scores

       S_x(e) = sum_{j in e} alpha_j * 1[X_j = x],      x in {0, 1},

   and the adoption kernel

       pi_e = S_1(e)^q / (S_0(e)^q + S_1(e)^q)          (0.5 if both are 0).

2. Each node ``i`` that has waited at least ``L_i`` steps since its last
   opinion change receives an adoption probability ``P_i`` obtained from the
   kernels of its incident hyperedges (see ``update_rule``).
3. ``X_i(t+1) ~ Bernoulli(P_i)`` for eligible nodes; ineligible nodes keep
   their opinion.  ``T_i`` is reset to ``t`` whenever the opinion changes.

All eligible nodes are updated simultaneously from the state at the start of
the sweep, which makes ``node_mean`` and ``node_sample`` independent of any
iteration order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import ModelConfig
from .hypergraph import Hypergraph


@dataclass
class SimulationResult:
    """Trajectories and scalar summaries produced by one simulation."""

    config: ModelConfig
    time: np.ndarray
    mean_opinion: np.ndarray
    diversity: np.ndarray
    cascade: np.ndarray
    majority: np.ndarray
    n0: np.ndarray
    n1: np.ndarray
    eligible: np.ndarray
    changes: np.ndarray
    initial_opinions: np.ndarray
    final_opinions: np.ndarray
    latencies: np.ndarray
    consensus_time: Optional[int]
    censored: bool
    hypergraph: Dict[str, float] = field(default_factory=dict)
    seed: Optional[int] = None
    update_rule: str = "node_mean"
    node_labels: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------- properties
    @property
    def n_nodes(self) -> int:
        return int(self.initial_opinions.size)

    @property
    def n_steps(self) -> int:
        return int(self.time.size)

    @property
    def final_majority(self) -> float:
        return float(self.majority[-1]) if self.majority.size else 0.5

    @property
    def final_diversity(self) -> float:
        return float(self.diversity[-1]) if self.diversity.size else 0.0

    @property
    def max_cascade(self) -> float:
        return float(self.cascade.max()) if self.cascade.size else 0.0

    @property
    def oscillation_index(self) -> float:
        """Standard deviation of the mean opinion over time (paper Eq. 7)."""
        if self.mean_opinion.size < 2:
            return 0.0
        return float(np.std(self.mean_opinion, ddof=1))

    @property
    def total_change_fraction(self) -> float:
        n = max(1, self.initial_opinions.size)
        return float(self.changes.sum() / n)

    @property
    def history(self) -> List[Dict[str, float]]:
        """Legacy per-step records (compatible with v1 ``model.py``)."""
        rows = []
        for k in range(self.n_steps):
            rows.append(
                {
                    "diversity": float(self.diversity[k]),
                    "cascade_size": float(self.cascade[k]),
                    "oscillation": float(self.mean_opinion[k]),
                    "mean_opinion": float(self.mean_opinion[k]),
                    "majority": float(self.majority[k]),
                    "n0": int(self.n0[k]),
                    "n1": int(self.n1[k]),
                    "eligible": int(self.eligible[k]),
                    "changes": int(self.changes[k]),
                    "t": int(self.time[k]),
                }
            )
        return rows

    def to_row(self, **extra: object) -> Dict[str, object]:
        """Flat scalar record for tabular output."""
        row: Dict[str, object] = {
            "q": self.config.q,
            "latency_dist": self.config.latency_dist,
            "mu": self.config.mu,
            "sigma": self.config.sigma,
            "update_rule": self.update_rule,
            "consensus_threshold": self.config.consensus_threshold,
            "max_steps": self.config.max_steps,
            "n_steps": self.n_steps,
            "consensus_time": self.consensus_time if self.consensus_time is not None else self.n_steps,
            "censored": bool(self.censored),
            "final_majority": self.final_majority,
            "final_diversity": self.final_diversity,
            "mean_diversity": float(self.diversity.mean()),
            "oscillation_index": self.oscillation_index,
            "max_cascade": self.max_cascade,
            "total_change_fraction": self.total_change_fraction,
            "mean_latency": float(self.latencies.mean()),
            "sd_latency": float(self.latencies.std(ddof=1)) if self.latencies.size > 1 else 0.0,
            "seed": self.seed,
        }
        row.update(extra)
        return row


def _bernoulli_kernel(s0: np.ndarray, s1: np.ndarray, q: float) -> np.ndarray:
    """Vectorised adoption kernel with the documented 0/0 convention."""
    s0 = np.maximum(s0, 0.0)
    s1 = np.maximum(s1, 0.0)
    a = np.power(s0, q)
    b = np.power(s1, q)
    denom = a + b
    out = np.full_like(b, 0.5, dtype=float)
    np.divide(b, denom, out=out, where=denom > 0)
    return out


class HNLVoterModel:
    """HNL voter model on a fixed hypergraph.

    The signature is backward compatible with the v1 prototype
    (``nodes, hyperedges, influence_weights, q, latency_dist, mu, sigma,
    lambda_param, consensus_threshold``) and adds explicit RNG control,
    update-rule selection and user-supplied initial states.
    """

    def __init__(
        self,
        nodes: Sequence[object],
        hyperedges: Sequence[Sequence[int]],
        influence_weights: Sequence[float],
        q: float = 1.5,
        latency_dist: str = "normal",
        mu: float = 5.0,
        sigma: float = 2.0,
        lambda_param: Optional[float] = None,
        consensus_threshold: float = 0.8,
        update_rule: str = "node_mean",
        seed: Optional[int] = None,
        initial_opinions: Optional[Sequence[int]] = None,
        latencies: Optional[Sequence[int]] = None,
        latency_floor: int = 1,
        max_steps: int = 100,
    ) -> None:
        self.nodes = np.asarray(nodes)
        n = len(self.nodes)
        if n == 0:
            raise ValueError("at least one node is required")

        self.config = ModelConfig(
            q=q,
            latency_dist=latency_dist,
            mu=mu,
            sigma=sigma,
            lambda_param=lambda_param,
            latency_floor=latency_floor,
            consensus_threshold=consensus_threshold,
            max_steps=max_steps,
            update_rule=update_rule,
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.hg = Hypergraph(
            n_nodes=n, edges=hyperedges, weights=influence_weights, labels=self.nodes
        )
        self.update_rule = self.config.update_rule

        if initial_opinions is None:
            probs = self.rng.random(n)
            self.initial_opinions = (probs < self.config.initial_opinion_balance).astype(np.int64)
        else:
            self.initial_opinions = self._validate_opinions(initial_opinions)
        self.opinions = self.initial_opinions.copy()

        if latencies is None:
            uniforms = self.rng.random(n)
            self.latencies = self.latencies_from_uniforms(uniforms, self.config)
        else:
            self.latencies = self._validate_latencies(latencies)

        self.last_change = np.zeros(n, dtype=np.int64)
        self.time_step = 0
        self.history: List[Dict[str, float]] = []

        # flattened incidence for fast node-level aggregation
        mask = self.hg.incidence_mask
        self._inc_nodes = np.repeat(np.arange(n), mask.sum(axis=1))
        self._inc_edges = self.hg.incidence[mask]
        self._degree = self.hg.degree
        self._has_edges = self._degree > 0

    # ------------------------------------------------------------- validation
    def _validate_opinions(self, values: Sequence[int]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.int64).ravel()
        if arr.size != len(self.nodes):
            raise ValueError("initial_opinions must have one entry per node")
        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError("opinions must be binary (0/1)")
        return arr

    def _validate_latencies(self, values: Sequence[int]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.int64).ravel()
        if arr.size != len(self.nodes):
            raise ValueError("latencies must have one entry per node")
        if np.any(arr < 0):
            raise ValueError("latencies must be non-negative")
        return arr

    # ------------------------------------------------------------- randomness
    @staticmethod
    def latencies_from_uniforms(
        uniforms: np.ndarray,
        config: ModelConfig,
    ) -> np.ndarray:
        """Map common uniforms to latencies (inverse-CDF sampling).

        Using the same uniforms across latency specifications yields a paired
        design: the ``u``-th node is at the same quantile under every
        distribution, which reduces Monte-Carlo variance in comparisons.
        """
        u = np.clip(np.asarray(uniforms, dtype=float), 1e-12, 1 - 1e-12)
        mu, sigma = config.mu, config.sigma
        if config.latency_dist == "fixed":
            raw = np.full(u.shape, mu, dtype=float)
        elif config.latency_dist == "normal":
            nd = NormalDist()
            raw = np.asarray([mu + sigma * nd.inv_cdf(float(x)) for x in u])
        elif config.latency_dist == "exponential":
            rate = config.lambda_param if config.lambda_param is not None else 1.0 / mu
            raw = -np.log1p(-u) / rate
        else:  # pragma: no cover - guarded by ModelConfig
            raise ValueError(f"unknown latency distribution {config.latency_dist!r}")
        return np.maximum(config.latency_floor, np.rint(raw)).astype(np.int64)

    # ------------------------------------------------------------------ engine
    def reset(self, opinions: Optional[Sequence[int]] = None) -> None:
        """Reset opinions, clocks and history (RNG state is not rewound)."""
        if opinions is not None:
            self.opinions = self._validate_opinions(opinions)
            self.initial_opinions = self.opinions.copy()
        else:
            self.opinions = self.initial_opinions.copy()
        self.last_change = np.zeros(len(self.nodes), dtype=np.int64)
        self.time_step = 0
        self.history = []

    def kernels(self) -> np.ndarray:
        """Adoption kernel ``pi_e`` for every hyperedge at the current state."""
        x = self.opinions[self.hg.edge_matrix]
        mask = self.hg.edge_mask
        w = self.hg.edge_weights
        s1 = np.where(mask, x * w, 0.0).sum(axis=1)
        s0 = np.where(mask, (1 - x) * w, 0.0).sum(axis=1)
        return _bernoulli_kernel(s0, s1, self.config.q)

    def node_probabilities(self) -> np.ndarray:
        """Adoption probability of every node under the current state."""
        p1 = self.kernels()
        n = len(self.nodes)
        if self.update_rule == "node_mean":
            totals = np.bincount(
                self._inc_nodes, weights=p1[self._inc_edges], minlength=n
            )
            p = totals / np.maximum(self._degree, 1)
        elif self.update_rule == "node_sample":
            draw = self.rng.random(n) * np.maximum(self._degree, 1)
            choice = np.minimum(draw.astype(np.int64), np.maximum(self._degree - 1, 0))
            edge = np.where(self._has_edges, self.hg.incidence[np.arange(n), choice], -1)
            p = np.where(self._has_edges, p1[np.maximum(edge, 0)], self.opinions.astype(float))
        else:  # edge_sweep is handled by step()
            raise ValueError("node_probabilities is defined for node-level rules")
        # nodes without incident hyperedges are frozen
        p = np.where(self._has_edges, p, self.opinions.astype(float))
        return p

    def _step_node_level(self) -> tuple[int, int]:
        t = self.time_step
        n = len(self.nodes)
        eligible = ((t - self.last_change) >= self.latencies) & self._has_edges
        p = self.node_probabilities()
        draws = (self.rng.random(n) < p).astype(np.int64)
        new = np.where(eligible, draws, self.opinions)
        changed = new != self.opinions
        n_changed = int(changed.sum())
        self.last_change[changed] = t
        self.opinions = new
        return n_changed, int(eligible.sum())

    def _step_edge_sweep(self) -> tuple[int, int]:
        """Literal Algorithm 1: edges in stored order, in-place updates.

        A node belonging to several hyperedges can be updated several times in
        one sweep; the last update wins.  This is kept for comparison with the
        prototype and is *not* order independent.
        """
        t = self.time_step
        q = self.config.q
        n_changed = 0
        n_eligible = 0
        for edge in self.hg.edge_list:
            edge_w = self.hg.node_weights[edge]
            x = self.opinions[edge]
            s1 = float(edge_w[x == 1].sum())
            s0 = float(edge_w[x == 0].sum())
            for pos, node in enumerate(edge):
                if t - self.last_change[node] < self.latencies[node]:
                    continue
                n_eligible += 1
                s0 = max(0.0, s0)
                s1 = max(0.0, s1)
                denom = s0**q + s1**q
                k = 0.5 if denom == 0 else s1**q / denom
                new = 1 if self.rng.random() < k else 0
                if new != self.opinions[node]:
                    self.opinions[node] = new
                    self.last_change[node] = t
                    n_changed += 1
                    if new == 1:
                        s0 -= edge_w[pos]
                        s1 += edge_w[pos]
                    else:
                        s1 -= edge_w[pos]
                        s0 += edge_w[pos]
        return n_changed, n_eligible

    def _record(self) -> None:
        n = len(self.nodes)
        n1 = int(self.opinions.sum())
        n0 = n - n1
        majority = max(n0, n1) / n
        changes = self._last_changes
        self.history.append(
            {
                "t": self.time_step,
                "diversity": 1.0 - majority,
                "cascade_size": changes / n,
                "oscillation": float(self.opinions.mean()),
                "mean_opinion": float(self.opinions.mean()),
                "majority": majority,
                "n0": n0,
                "n1": n1,
                "eligible": self._last_eligible,
                "changes": changes,
            }
        )

    def step(self) -> None:
        """Advance the simulation by one sweep."""
        self.time_step += 1
        if self.update_rule == "edge_sweep":
            self._last_changes, self._last_eligible = self._step_edge_sweep()
        else:
            self._last_changes, self._last_eligible = self._step_node_level()
        self._record()

    def run(self, max_steps: Optional[int] = None) -> SimulationResult:
        """Run until consensus or the step cap; return a :class:`SimulationResult`."""
        steps = int(max_steps if max_steps is not None else self.config.max_steps)
        self.history = []
        self.time_step = 0
        self._last_changes = 0
        self._last_eligible = 0
        consensus_time: Optional[int] = None
        censored = True
        for _ in range(steps):
            self.step()
            if self.history[-1]["majority"] >= self.config.consensus_threshold:
                consensus_time = self.time_step
                censored = False
                break

        hist = self.history
        if not hist:  # pragma: no cover - max_steps >= 1 guaranteed
            raise RuntimeError("simulation recorded no steps")
        time = np.asarray([h["t"] for h in hist], dtype=np.int64)
        return SimulationResult(
            config=self.config,
            time=time,
            mean_opinion=np.asarray([h["mean_opinion"] for h in hist], dtype=float),
            diversity=np.asarray([h["diversity"] for h in hist], dtype=float),
            cascade=np.asarray([h["cascade_size"] for h in hist], dtype=float),
            majority=np.asarray([h["majority"] for h in hist], dtype=float),
            n0=np.asarray([h["n0"] for h in hist], dtype=np.int64),
            n1=np.asarray([h["n1"] for h in hist], dtype=np.int64),
            eligible=np.asarray([h["eligible"] for h in hist], dtype=np.int64),
            changes=np.asarray([h["changes"] for h in hist], dtype=np.int64),
            initial_opinions=self.initial_opinions.copy(),
            final_opinions=self.opinions.copy(),
            latencies=self.latencies.copy(),
            consensus_time=consensus_time,
            censored=censored,
            hypergraph=self.hg.summary(),
            seed=self.seed,
            update_rule=self.update_rule,
            node_labels=self.hg.labels,
        )

    # ------------------------------------------------------- legacy interface
    def run_simulation(self, max_steps: int = 100) -> List[Dict[str, float]]:
        """v1-compatible entry point returning the per-step history list."""
        self.run(max_steps=max_steps)
        return self.history

    def plot_metrics(self, path: str = "hnl_voter_metrics.png", show: bool = False) -> str:
        """Plot the three v1 panels and save them to ``path``."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist = self.history
        if not hist:
            raise RuntimeError("run the simulation before plotting")
        t = [h["t"] for h in hist]
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        axs[0].plot(t, [h["diversity"] for h in hist])
        axs[0].set_title("Opinion diversity D(t)")
        axs[0].set_xlabel("sweep t")
        axs[1].plot(t, [h["cascade_size"] for h in hist])
        axs[1].set_title("Cascade size C(t)")
        axs[1].set_xlabel("sweep t")
        axs[2].plot(t, [h["mean_opinion"] for h in hist])
        axs[2].set_title("Mean opinion X̄(t)")
        axs[2].set_xlabel("sweep t")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        if show:  # pragma: no cover - interactive only
            plt.show()
        plt.close(fig)
        return path

    @classmethod
    def from_hypergraph(
        cls,
        hypergraph: Hypergraph,
        config: Optional[ModelConfig] = None,
        seed: Optional[int] = None,
        **kwargs: object,
    ) -> "HNLVoterModel":
        cfg = config or ModelConfig()
        model = cls(
            nodes=np.arange(hypergraph.n_nodes),
            hyperedges=[e.tolist() for e in hypergraph.edge_list],
            influence_weights=hypergraph.node_weights,
            q=cfg.q,
            latency_dist=cfg.latency_dist,
            mu=cfg.mu,
            sigma=cfg.sigma,
            lambda_param=cfg.lambda_param,
            consensus_threshold=cfg.consensus_threshold,
            update_rule=cfg.update_rule,
            seed=seed,
            latency_floor=cfg.latency_floor,
            max_steps=cfg.max_steps,
            **kwargs,
        )
        model.hg.labels = hypergraph.labels
        return model
