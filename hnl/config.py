"""Validated configuration objects for the HNL voter model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

LATENCY_DISTS = ("normal", "exponential", "fixed")
UPDATE_RULES = ("node_mean", "node_sample", "edge_sweep")


@dataclass(frozen=True)
class ModelConfig:
    """Parameters of one HNL simulation.

    Parameters
    ----------
    q:
        Nonlinearity exponent in the adoption kernel.  ``q = 1`` recovers
        linear (weighted-proportional) adoption, ``q > 1`` amplifies the
        local majority, ``q < 1`` amplifies the local minority.
    latency_dist:
        ``"normal"``, ``"exponential"`` or ``"fixed"``.
    mu, sigma:
        Mean and standard deviation of the latency distribution (steps).
        For ``"fixed"`` the latency is ``round(mu)`` for every node.
    lambda_param:
        Rate of the exponential latency distribution.  ``None`` means
        ``1 / mu`` (so that the mean latency equals ``mu``).
    latency_floor:
        Smallest admissible latency in steps (default 1).
    consensus_threshold:
        Stop when the majority share reaches this value, ``r`` in the paper.
    max_steps:
        Hard cap on the number of sweeps, ``T`` in the paper.
    update_rule:
        ``"node_mean"`` (default, order independent): a node's adoption
        probability is the mean kernel over its incident hyperedges.
        ``"node_sample"``: one incident hyperedge is sampled uniformly.
        ``"edge_sweep"``: literal in-place sweep of Algorithm 1 (order
        dependent; provided for comparison with the prototype).
    initial_opinion_balance:
        Probability that a node starts with opinion 1 when opinions are not
        supplied explicitly.
    """

    q: float = 1.5
    latency_dist: str = "normal"
    mu: float = 5.0
    sigma: float = 2.0
    lambda_param: Optional[float] = None
    latency_floor: int = 1
    consensus_threshold: float = 0.8
    max_steps: int = 100
    update_rule: str = "node_mean"
    initial_opinion_balance: float = 0.5

    def __post_init__(self) -> None:
        if not (self.q >= 0):
            raise ValueError("q must be non-negative")
        if self.latency_dist not in LATENCY_DISTS:
            raise ValueError(f"latency_dist must be one of {LATENCY_DISTS}")
        if self.update_rule not in UPDATE_RULES:
            raise ValueError(f"update_rule must be one of {UPDATE_RULES}")
        if self.latency_dist == "normal" and self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.latency_dist == "fixed" and self.mu < 0:
            raise ValueError("mu must be non-negative for fixed latency")
        if self.latency_dist == "exponential":
            rate = self.lambda_param if self.lambda_param is not None else (
                1.0 / self.mu if self.mu > 0 else None
            )
            if rate is None or rate <= 0:
                raise ValueError(
                    "exponential latency needs lambda_param > 0 or mu > 0"
                )
        if self.latency_floor < 0:
            raise ValueError("latency_floor must be non-negative")
        if not (0.0 < self.consensus_threshold <= 1.0):
            raise ValueError("consensus_threshold must be in (0, 1]")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if not (0.0 <= self.initial_opinion_balance <= 1.0):
            raise ValueError("initial_opinion_balance must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
