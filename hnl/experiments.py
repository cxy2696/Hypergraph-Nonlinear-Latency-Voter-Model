"""Replicated experiments with paired designs and censoring-aware summaries.

Design choices that matter for inference
----------------------------------------
* **Common random numbers.**  Each replicate draws one hypergraph, one initial
  opinion vector and one vector of uniforms.  Latencies are obtained from the
  uniforms by inverse-CDF for every distribution, and every ``(q, latency)``
  cell runs on the same replicate.  Comparisons are therefore paired, which
  removes between-replicate variance from the contrasts.
* **Right censoring.**  A run that does not reach the consensus threshold
  within ``T`` sweeps is censored, not assigned ``T`` as if it had converged.
  Consensus-time distributions are summarised with Kaplan–Meier medians.
* **Fixed observation window.**  Trajectory metrics (diversity, mean opinion,
  oscillation index) are computed on the padded window ``1..T``, carrying the
  terminal state forward after convergence.  This keeps early-stopping runs
  comparable with censored ones.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import ModelConfig
from .hypergraph import Hypergraph, random_hypergraph
from .metrics import (
    absorption_by_component,
    damping_halflife,
    kaplan_meier,
    km_median,
    restricted_mean_survival_time,
    zero_crossings,
)
from .model import HNLVoterModel
from .stats import holm_adjust, paired_contrast, summarize


@dataclass(frozen=True)
class LatencySpec:
    name: str
    dist: str
    mu: float = 5.0
    sigma: float = 0.0
    lambda_param: Optional[float] = None

    def to_config(
        self, q: float, threshold: float, max_steps: int, update_rule: str
    ) -> ModelConfig:
        return ModelConfig(
            q=q,
            latency_dist=self.dist,
            mu=self.mu,
            sigma=self.sigma,
            lambda_param=self.lambda_param,
            consensus_threshold=threshold,
            max_steps=max_steps,
            update_rule=update_rule,
        )


DEFAULT_SPECS: Tuple[LatencySpec, ...] = (
    LatencySpec("fixed", "fixed", mu=5.0, sigma=0.0),
    LatencySpec("normal_s1", "normal", mu=5.0, sigma=1.0),
    LatencySpec("normal_s2", "normal", mu=5.0, sigma=2.0),
    LatencySpec("normal_s4", "normal", mu=5.0, sigma=4.0),
    LatencySpec("exponential", "exponential", mu=5.0, sigma=0.0),
)


@dataclass
class Design:
    name: str = "random"
    n_nodes: int = 100
    n_edges: int = 200
    size_range: Tuple[int, int] = (3, 6)
    q_values: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.5)
    latency_specs: Tuple[LatencySpec, ...] = DEFAULT_SPECS
    repeats: int = 30
    max_steps: int = 100
    consensus_threshold: float = 0.8
    base_seed: int = 20260101
    update_rule: str = "node_mean"

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["latency_specs"] = [asdict(s) for s in self.latency_specs]
        return d


@dataclass
class ExperimentResult:
    name: str
    runs: pd.DataFrame
    trajectories: Dict[str, np.ndarray]
    design: Dict[str, object]
    metadata: Dict[str, object] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        runs_path = out / f"{self.name}_runs.csv"
        self.runs.to_csv(runs_path, index=False)
        np.savez_compressed(out / f"{self.name}_trajectories.npz", **self.trajectories)
        payload = {
            "name": self.name,
            "design": self.design,
            "metadata": self.metadata,
            "n_runs": int(len(self.runs)),
        }
        (out / f"{self.name}_design.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n"
        )
        return runs_path


def _pad_trajectory(values: np.ndarray, length: int, fill_last: bool = True) -> np.ndarray:
    out = np.empty(length, dtype=float)
    n = min(values.size, length)
    out[:n] = values[:n]
    if n < length:
        out[n:] = values[-1] if (fill_last and values.size) else 0.0
    return out


def _model_seed(base_seed: int, *key: int) -> int:
    ss = np.random.SeedSequence(entropy=int(base_seed), spawn_key=tuple(int(k) for k in key))
    return int(ss.generate_state(1)[0])


def _run_row(
    result,
    design: Design,
    replicate: int,
    latency_spec: LatencySpec,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    pad = _pad_trajectory(result.mean_opinion, design.max_steps)
    pad_div = _pad_trajectory(result.diversity, design.max_steps)
    pad_cas = _pad_trajectory(result.cascade, design.max_steps, fill_last=False)
    half = design.max_steps // 2
    row = result.to_row(
        design=design.name,
        converged=0 if result.censored else 1,
        replicate=int(replicate),
        latency=latency_spec.name,
        mean_latency_target=latency_spec.mu if latency_spec.dist != "exponential" else (
            1.0 / (latency_spec.lambda_param or 1.0 / latency_spec.mu)
        ),
        oscillation_first_half=float(np.std(pad[:half], ddof=1)) if half > 1 else 0.0,
        oscillation_second_half=float(np.std(pad[half:], ddof=1)) if design.max_steps - half > 1 else 0.0,
        oscillation_ratio=(
            float(np.std(pad[half:], ddof=1) / np.std(pad[:half], ddof=1))
            if half > 1 and np.std(pad[:half], ddof=1) > 0
            else float("nan")
        ),
        zero_crossings=zero_crossings(pad),
        damping_halflife=damping_halflife(pad),
        final_diversity_padded=float(pad_div[-1]),
        mean_diversity_padded=float(pad_div.mean()),
        total_cascade_padded=float(pad_cas.sum()),
    )
    if extra:
        row.update(extra)
    return row


def _component_stats(result, hypergraph: Hypergraph) -> Dict[str, float]:
    labels, sizes = hypergraph.components()
    majority, absorbed = absorption_by_component(result.final_opinions, labels)
    return {
        "component_absorbed_fraction": float(absorbed.mean()),
        "component_majority_mean": float(majority.mean()),
        "n_components": int(sizes.size),
    }


def bootstrap_rmst(
    durations: Iterable[float],
    events: Iterable[bool],
    tau: float,
    n_boot: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Bootstrap the restricted mean survival time at ``tau``."""
    d = np.asarray(list(durations), dtype=float)
    e = np.asarray(list(events), dtype=bool)
    rng = rng or np.random.default_rng(0)
    if d.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    values = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, d.size, size=d.size)
        times, survival = kaplan_meier(d[idx], e[idx])
        values[b] = restricted_mean_survival_time(times, survival, tau)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return (float(values.mean()), float(lo), float(hi))


def run_random_experiment(design: Design, progress: bool = False) -> ExperimentResult:
    """Full factorial ``q × latency × replicate`` study on random hypergraphs."""
    children = np.random.SeedSequence(design.base_seed).spawn(design.repeats)
    rows: List[Dict[str, object]] = []
    T = design.max_steps
    nq = len(design.q_values)
    ns = len(design.latency_specs)
    traj = {
        key: np.full((design.repeats, T), np.nan)
        for key in (
            [f"mean|{s.name}|q={q}" for s in design.latency_specs for q in design.q_values]
            + [f"div|{s.name}|q={q}" for s in design.latency_specs for q in design.q_values]
        )
    }

    for r, child in enumerate(children):
        rng = np.random.default_rng(child)
        hg = random_hypergraph(
            design.n_nodes, design.n_edges, design.size_range, rng
        )
        initial = rng.integers(0, 2, size=design.n_nodes, dtype=np.int64)
        uniforms = rng.random(design.n_nodes)
        comp = hg.summary()
        for si, spec in enumerate(design.latency_specs):
            cfg0 = spec.to_config(
                design.q_values[0], design.consensus_threshold, T, design.update_rule
            )
            latencies = HNLVoterModel.latencies_from_uniforms(uniforms, cfg0)
            for qi, q in enumerate(design.q_values):
                cfg = spec.to_config(q, design.consensus_threshold, T, design.update_rule)
                model = HNLVoterModel.from_hypergraph(
                    hg,
                    cfg,
                    seed=_model_seed(design.base_seed, r, si, qi),
                    initial_opinions=initial,
                    latencies=latencies,
                )
                result = model.run()
                row = _run_row(result, design, r, spec, comp)
                row.update(_component_stats(result, hg))
                rows.append(row)
                traj[f"mean|{spec.name}|q={q}"][r] = _pad_trajectory(result.mean_opinion, T)
                traj[f"div|{spec.name}|q={q}"][r] = _pad_trajectory(result.diversity, T)
        if progress and (r + 1) % 10 == 0:
            print(f"  replicate {r + 1}/{design.repeats}", flush=True)

    return ExperimentResult(
        name=design.name,
        runs=pd.DataFrame(rows),
        trajectories=traj,
        design=design.to_dict(),
        metadata={"n_nodes": design.n_nodes, "n_edges": design.n_edges},
    )


def run_dataset_experiment(
    bundle,
    design: Design,
    init: str = "random",
    progress: bool = False,
) -> ExperimentResult:
    """Factorial study on a fixed dataset hypergraph."""
    children = np.random.SeedSequence(design.base_seed).spawn(design.repeats)
    rows: List[Dict[str, object]] = []
    T = design.max_steps
    hg = bundle.hypergraph
    comp = hg.summary()
    traj = {
        key: np.full((design.repeats, T), np.nan)
        for key in (
            [f"mean|{s.name}|q={q}" for s in design.latency_specs for q in design.q_values]
            + [f"div|{s.name}|q={q}" for s in design.latency_specs for q in design.q_values]
        )
    }
    for r, child in enumerate(children):
        rng = np.random.default_rng(child)
        initial = bundle.opinions_for(init, rng)
        uniforms = rng.random(hg.n_nodes)
        for si, spec in enumerate(design.latency_specs):
            cfg0 = spec.to_config(
                design.q_values[0], design.consensus_threshold, T, design.update_rule
            )
            latencies = HNLVoterModel.latencies_from_uniforms(uniforms, cfg0)
            for qi, q in enumerate(design.q_values):
                cfg = spec.to_config(q, design.consensus_threshold, T, design.update_rule)
                model = HNLVoterModel.from_hypergraph(
                    hg,
                    cfg,
                    seed=_model_seed(design.base_seed, r, si, qi),
                    initial_opinions=initial,
                    latencies=latencies,
                )
                result = model.run()
                row = _run_row(
                    result,
                    design,
                    r,
                    spec,
                    {**comp, "init": init, "scope": bundle.metadata.get("scope")},
                )
                row.update(_component_stats(result, hg))
                rows.append(row)
                traj[f"mean|{spec.name}|q={q}"][r] = _pad_trajectory(result.mean_opinion, T)
                traj[f"div|{spec.name}|q={q}"][r] = _pad_trajectory(result.diversity, T)
        if progress and (r + 1) % 10 == 0:
            print(f"  replicate {r + 1}/{design.repeats}", flush=True)
    return ExperimentResult(
        name=design.name,
        runs=pd.DataFrame(rows),
        trajectories=traj,
        design=design.to_dict(),
        metadata={
            "dataset": bundle.metadata,
            "init": init,
            "hypergraph": comp,
        },
    )


def run_rule_sensitivity(
    design: Design,
    rules: Sequence[str] = ("node_mean", "node_sample", "edge_sweep"),
    q_values: Sequence[float] = (0.5, 1.5, 2.5),
    repeats: int = 8,
) -> ExperimentResult:
    """Compare update rules on identical replicates (same graph, init, latency)."""
    children = np.random.SeedSequence(design.base_seed + 7).spawn(repeats)
    rows: List[Dict[str, object]] = []
    T = design.max_steps
    for r, child in enumerate(children):
        rng = np.random.default_rng(child)
        hg = random_hypergraph(design.n_nodes, design.n_edges, design.size_range, rng)
        initial = rng.integers(0, 2, size=design.n_nodes, dtype=np.int64)
        uniforms = rng.random(design.n_nodes)
        spec = DEFAULT_SPECS[2]  # normal, sigma = 2
        cfg0 = spec.to_config(q_values[0], design.consensus_threshold, T, rules[0])
        latencies = HNLVoterModel.latencies_from_uniforms(uniforms, cfg0)
        for ri, rule in enumerate(rules):
            for qi, q in enumerate(q_values):
                cfg = spec.to_config(q, design.consensus_threshold, T, rule)
                model = HNLVoterModel.from_hypergraph(
                    hg,
                    cfg,
                    seed=_model_seed(design.base_seed, r, 90 + ri, qi),
                    initial_opinions=initial,
                    latencies=latencies,
                )
                result = model.run()
                row = _run_row(
                    result,
                    design,
                    r,
                    spec,
                    {"update_rule": rule, **hg.summary()},
                )
                row.update(_component_stats(result, hg))
                rows.append(row)
    name = f"{design.name}_rule_sensitivity"
    return ExperimentResult(
        name=name,
        runs=pd.DataFrame(rows),
        trajectories={},
        design={
            **design.to_dict(),
            "rules": list(rules),
            "q_values": list(q_values),
            "repeats": repeats,
            "latency_spec": "normal_s2",
        },
    )


# ------------------------------------------------------------------ summaries
def summarize_cells(
    runs: pd.DataFrame,
    by: Sequence[str] = ("latency", "q"),
    n_boot: int = 2000,
    seed: int = 12345,
) -> pd.DataFrame:
    """Cell-level summary with bootstrap CIs and Kaplan–Meier consensus median."""
    rng = np.random.default_rng(seed)
    out = []
    for key, group in runs.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(by, key)}
        row["n"] = int(len(group))
        row["consensus_rate"] = float(group["converged"].mean())
        times, survival = kaplan_meier(group["consensus_time"], group["converged"].astype(bool))
        row["km_median_tc"] = km_median(times, survival)
        row["km_consensus_by_T"] = float(1.0 - survival[-1]) if survival.size else 0.0
        tau = float(group["max_steps"].iloc[0])
        rmst_mean, rmst_lo, rmst_hi = bootstrap_rmst(
            group["consensus_time"], group["converged"].astype(bool), tau, n_boot=n_boot, rng=rng
        )
        row["rmst_tc"] = rmst_mean
        row["rmst_tc_ci_low"] = rmst_lo
        row["rmst_tc_ci_high"] = rmst_hi
        for metric in (
            "consensus_time",
            "final_diversity",
            "mean_diversity_padded",
            "oscillation_index",
            "max_cascade",
            "component_absorbed_fraction",
            "oscillation_first_half",
            "oscillation_second_half",
            "zero_crossings",
            "damping_halflife",
        ):
            stats = summarize(group[metric], n_boot=n_boot, rng=rng)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_ci_low"] = stats["ci_low"]
            row[f"{metric}_ci_high"] = stats["ci_high"]
            row[f"{metric}_median"] = stats["median"]
        out.append(row)
    return pd.DataFrame(out)


def paired_contrasts(
    runs: pd.DataFrame,
    factor: str,
    reference,
    metric: str = "consensus_time",
    by: Sequence[str] = ("latency",),
    n_perm: int = 10000,
    n_boot: int = 2000,
    seed: int = 54321,
) -> pd.DataFrame:
    """Paired contrasts of every factor level against ``reference``.

    Rows are aligned on ``(replicate, *by)`` so that each difference compares
    the same graph, initial condition and latency quantiles.
    """
    rng = np.random.default_rng(seed)
    out = []
    index = ["replicate", *by]
    wide = runs.pivot_table(index=index, columns=factor, values=metric, aggfunc="mean")
    if reference not in wide.columns:
        raise KeyError(f"reference level {reference!r} not present")
    group_cols = list(by)
    for key, frame in wide.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        for level in wide.columns:
            if level == reference:
                continue
            pair = frame[[reference, level]].dropna()
            if len(pair) < 3:
                continue
            stats = paired_contrast(
                pair[level].to_numpy(),
                pair[reference].to_numpy(),
                n_perm=n_perm,
                n_boot=n_boot,
                rng=rng,
            )
            row = {name: value for name, value in zip(by, key)}
            row.update(
                {
                    "factor": factor,
                    "level": level,
                    "reference": reference,
                    "metric": metric,
                    **stats,
                }
            )
            out.append(row)
    result = pd.DataFrame(out)
    if not result.empty:
        result["p_holm"] = np.nan
        for _, idx in result.groupby(group_cols).groups.items():
            result.loc[idx, "p_holm"] = holm_adjust(result.loc[idx, "p_value"].to_numpy())
    return result


def damping_table(
    runs: pd.DataFrame,
    by: Sequence[str] = ("latency", "q"),
    seed: int = 999,
) -> pd.DataFrame:
    """Paired comparison of oscillation amplitude in the two halves of the run."""
    rng = np.random.default_rng(seed)
    out = []
    for key, group in runs.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        first = group["oscillation_first_half"].to_numpy()
        second = group["oscillation_second_half"].to_numpy()
        stats = paired_contrast(second, first, n_perm=10000, n_boot=2000, rng=rng)
        row = {name: value for name, value in zip(by, key)}
        row.update(
            {
                "n": stats["n"],
                "mean_first_half": float(np.mean(first)),
                "mean_second_half": float(np.mean(second)),
                "mean_ratio": float(np.mean(group["oscillation_ratio"].dropna())),
                "mean_diff": stats["mean_diff"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "p_value": stats["p_value"],
            }
        )
        out.append(row)
    return pd.DataFrame(out)
