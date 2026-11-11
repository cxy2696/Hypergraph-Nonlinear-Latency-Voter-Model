"""Figure generation from experiment outputs.

Every figure is produced from CSV/NPZ artifacts written by
``python -m hnl.cli reproduce``; nothing is hard-coded.  Figures are saved as
PNG (for GitHub) and PDF (vector, for manuscripts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .experiments import summarize_cells  # noqa: E402
from .metrics import kaplan_meier  # noqa: E402
from .stats import bootstrap_ci  # noqa: E402

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
SPEC_COLORS = {
    "fixed": "#0072B2",
    "normal_s1": "#56B4E9",
    "normal_s2": "#009E73",
    "normal_s4": "#D55E00",
    "exponential": "#CC79A7",
}
SPEC_LABELS = {
    "fixed": "fixed (L=5)",
    "normal_s1": "normal σ=1",
    "normal_s2": "normal σ=2",
    "normal_s4": "normal σ=4",
    "exponential": "exponential (mean 5)",
}
Q_COLORS = {0.5: "#0072B2", 1.0: "#009E73", 1.5: "#E69F00", 2.5: "#D55E00"}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.autolayout": False,
        }
    )


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _band(values: np.ndarray, level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and percentile band across replicates (NaN-safe)."""
    mean = np.nanmean(values, axis=0)
    lo = np.nanpercentile(values, 100 * (1 - level) / 2, axis=0)
    hi = np.nanpercentile(values, 100 * (1 + level) / 2, axis=0)
    return mean, lo, hi


# ------------------------------------------------------------------ figure 1
def plot_diversity(result, path, specs: Sequence[str] = ("fixed", "normal_s4")) -> Path:
    _setup_style()
    qs = sorted(result.runs["q"].unique())
    fig, axes = plt.subplots(1, len(specs), figsize=(4.2 * len(specs), 3.4), sharey=True)
    if len(specs) == 1:
        axes = [axes]
    for ax, spec in zip(axes, specs):
        for q in qs:
            key = f"div|{spec}|q={q}"
            if key not in result.trajectories:
                continue
            mean, lo, hi = _band(result.trajectories[key])
            t = np.arange(1, mean.size + 1)
            color = Q_COLORS.get(q, PALETTE[0])
            ax.plot(t, mean, color=color, lw=1.4, label=f"q={q:g}")
            ax.fill_between(t, lo, hi, color=color, alpha=0.16, linewidth=0)
        ax.set_title(f"{SPEC_LABELS.get(spec, spec)}")
        ax.set_xlabel("sweep t")
        ax.set_ylim(0, 0.5)
    axes[0].set_ylabel("minority share  D(t)")
    axes[0].legend(frameon=False, loc="lower left")
    n = int(result.runs["n_nodes"].iloc[0])
    m = int(result.runs["n_edges"].iloc[0])
    R = int(result.runs["replicate"].nunique())
    fig.suptitle(
        f"Opinion diversity on random hypergraphs (n={n}, m={m}, {R} replicates; mean ± 95% band)",
        y=1.02,
    )
    return _save(fig, path)


# ------------------------------------------------------------------ figure 2
def plot_consensus_km(result, path, specs: Sequence[str] = ("fixed", "normal_s4")) -> Path:
    _setup_style()
    qs = sorted(result.runs["q"].unique())
    fig, axes = plt.subplots(1, len(specs), figsize=(4.2 * len(specs), 3.4), sharey=True)
    if len(specs) == 1:
        axes = [axes]
    for ax, spec in zip(axes, specs):
        for q in qs:
            sub = result.runs[(result.runs["latency"] == spec) & (result.runs["q"] == q)]
            if sub.empty:
                continue
            times, survival = kaplan_meier(sub["consensus_time"], sub["converged"].astype(bool))
            t = np.concatenate(([0.0], times))
            conv = np.concatenate(([0.0], 1.0 - survival))
            ax.step(t, conv, where="post", color=Q_COLORS.get(q, PALETTE[0]), lw=1.5, label=f"q={q:g}")
        ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
        ax.set_xlim(0, result.runs["max_steps"].max())
        ax.set_ylim(0, 1.02)
        ax.set_title(SPEC_LABELS.get(spec, spec))
        ax.set_xlabel("sweep t")
    axes[0].set_ylabel("cumulative consensus incidence (KM)")
    axes[0].legend(frameon=False, loc="lower right")
    R = int(result.runs["replicate"].nunique())
    fig.suptitle(f"Time to 80% majority by nonlinearity q ({R} replicates, censored runs included)", y=1.02)
    return _save(fig, path)


# ------------------------------------------------------------------ figure 3
def plot_oscillation(result, path, q_example: float = 1.5) -> Path:
    _setup_style()
    summary = summarize_cells(result.runs, by=("latency", "q"))
    specs = [s for s in result.runs["latency"].unique()]
    qs = sorted(result.runs["q"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.4))

    ax = axes[0]
    for spec in specs:
        sub = summary[summary["latency"] == spec]
        if sub.empty:
            continue
        ax.errorbar(
            sub["q"],
            sub["oscillation_index_mean"],
            yerr=[
                sub["oscillation_index_mean"] - sub["oscillation_index_ci_low"],
                sub["oscillation_index_ci_high"] - sub["oscillation_index_mean"],
            ],
            marker="o",
            ms=3.5,
            lw=1.2,
            capsize=2,
            color=SPEC_COLORS.get(spec, PALETTE[0]),
            label=SPEC_LABELS.get(spec, spec),
        )
    ax.set_xlabel("nonlinearity q")
    ax.set_ylabel("O = std(X̄(t))")
    ax.set_title("(a) Oscillation index")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for spec in [s for s in ("fixed", "normal_s4") if s in specs]:
        key = f"mean|{spec}|q={q_example}"
        if key not in result.trajectories:
            continue
        arr = result.trajectories[key]
        t = np.arange(1, arr.shape[1] + 1)
        for row in arr[: min(4, arr.shape[0])]:
            ax.plot(t, row, color=SPEC_COLORS[spec], lw=0.7, alpha=0.28)
        ax.plot(t, np.nanmean(arr, axis=0), color=SPEC_COLORS[spec], lw=1.7, label=SPEC_LABELS[spec])
    ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("sweep t")
    ax.set_ylabel("mean opinion  X̄(t)")
    ax.set_title(f"(b) Individual runs at q={q_example:g}")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    finite = result.runs[np.isfinite(result.runs["oscillation_ratio"])]
    for spec in specs:
        sub = finite[finite["latency"] == spec]
        if sub.empty:
            continue
        xs, ys, errs = [], [], []
        for q in qs:
            values = sub[np.isclose(sub["q"], q)]["oscillation_ratio"].to_numpy()
            if values.size == 0:
                continue
            lo, hi = bootstrap_ci(values, np.mean, n_boot=1000, rng=np.random.default_rng(0))
            xs.append(q)
            ys.append(float(np.mean(values)))
            errs.append((hi - lo) / 2)
        ax.errorbar(xs, ys, yerr=errs, marker="o", ms=3, lw=1.0, capsize=2,
                    color=SPEC_COLORS.get(spec, PALETTE[0]), label=SPEC_LABELS.get(spec, spec))
    ax.axhline(1.0, color="0.4", lw=0.9, ls="--")
    ax.set_xlabel("nonlinearity q")
    ax.set_ylabel("late / early amplitude")
    ax.set_title("(c) Damping of swings")
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle(
        "Swing amplitude grows with q and decays within the window once consensus is reachable",
        y=1.03,
    )
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------------ figure 4
def plot_cascade(result, path) -> Path:
    _setup_style()
    summary = summarize_cells(result.runs, by=("latency", "q"))
    specs = [s for s in result.runs["latency"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    ax = axes[0]
    for spec in specs:
        sub = summary[summary["latency"] == spec]
        if sub.empty:
            continue
        ax.errorbar(
            sub["q"],
            sub["max_cascade_mean"],
            yerr=[
                sub["max_cascade_mean"] - sub["max_cascade_ci_low"],
                sub["max_cascade_ci_high"] - sub["max_cascade_mean"],
            ],
            marker="o",
            ms=3.5,
            lw=1.2,
            capsize=2,
            color=SPEC_COLORS.get(spec, PALETTE[0]),
            label=SPEC_LABELS.get(spec, spec),
        )
    ax.set_xlabel("nonlinearity q")
    ax.set_ylabel("max C(t) per run")
    ax.set_ylim(0, 0.5)
    ax.set_title("(a) Largest single-sweep cascade")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    spec = "normal_s4"
    rng = np.random.default_rng(0)
    for q in sorted(result.runs["q"].unique()):
        sub = result.runs[(result.runs["latency"] == spec) & (result.runs["q"] == q)]
        ax.bar(str(q), sub["total_cascade_padded"].mean(), color=Q_COLORS.get(q, PALETTE[0]))
        ax.errorbar(
            str(q),
            sub["total_cascade_padded"].mean(),
            yerr=sub["total_cascade_padded"].std(ddof=1) / np.sqrt(len(sub)),
            color="black",
            capsize=3,
            lw=1,
        )
    ax.set_xlabel("nonlinearity q")
    ax.set_ylabel("summed C(t) over window")
    ax.set_title(f"(b) Total cascade activity ({SPEC_LABELS[spec]})")

    fig.suptitle("Cascades shrink as nonlinearity rises", y=1.03)
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------------ figure 5
def plot_dataset_structure(bundle, path, threshold: float = 0.8) -> Path:
    _setup_style()
    hg = bundle.hypergraph
    labels, sizes = hg.components()
    deg = hg.degree
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2))

    ax = axes[0, 0]
    ranked = np.sort(sizes)[::-1]
    ax.plot(np.arange(1, ranked.size + 1), ranked, marker="o", ms=2.5, lw=1.0, color=PALETTE[0])
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("component rank")
    ax.set_ylabel("component size")
    ax.set_title("(a) Connected components (2-section)")
    ax.annotate(
        f"largest = {ranked[0]} / {hg.n_nodes} = {ranked[0]/hg.n_nodes:.1%}\nconsensus threshold = {threshold:.0%}",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    ax = axes[0, 1]
    counts = np.bincount(deg, minlength=deg.max() + 1)
    ax.bar(np.arange(counts.size), counts, color=PALETTE[1])
    ax.set_yscale("log")
    ax.set_xlabel("hyperedge degree")
    ax.set_ylabel("nodes")
    ax.set_title("(b) Degree distribution")
    ax.annotate(
        f"isolated: {int((deg == 0).sum())}\nmean degree (covered): {deg[deg > 0].mean():.2f}",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    ax = axes[1, 0]
    edge_sizes = hg.sizes
    ax.hist(edge_sizes, bins=np.arange(1.5, edge_sizes.max() + 1.5, 1), color=PALETTE[2], edgecolor="white")
    ax.set_xlabel("hyperedge size |e|")
    ax.set_ylabel("hyperedges")
    ax.set_title("(c) Hyperedge sizes")

    ax = axes[1, 1]
    counts = {int(k): v for k, v in bundle.metadata.get("hashtag_group_size_counts", {}).items()}
    sizes_grid = np.array(sorted(counts))
    values = np.array([counts[s] for s in sizes_grid])
    ax.bar(sizes_grid, values, color=PALETTE[3])
    ax.set_yscale("log")
    ax.set_xlabel("users per hashtag")
    ax.set_ylabel("hashtags")
    ax.set_title("(d) Most hashtags form no group")
    singletons = counts.get(1, 0)
    ax.annotate(
        f"{int(values.sum())} hashtags\n{singletons} used by one user\n"
        f"{int(values[sizes_grid >= 2].sum())} multi-user groups\n"
        f"{hg.n_edges} unique hyperedges",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    fig.suptitle(
        f"Dataset hypergraph is fragmented: {hg.n_nodes} users, {hg.n_edges} hashtag groups, "
        f"{sizes.size} components",
        y=1.0,
    )
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------------ figure 6
def plot_dataset_dynamics(
    full_result,
    lcc_result,
    bundle_full,
    bundle_lcc,
    path,
    q_example: float = 1.5,
    spec: str = "normal_s4",
) -> Path:
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4))
    threshold = float(full_result.runs["consensus_threshold"].iloc[0])

    ax = axes[0, 0]
    key = f"mean|{spec}|q={q_example}"
    mean, lo, hi = _band(full_result.trajectories[key])
    t = np.arange(1, mean.size + 1)
    ax.plot(t, mean, color=PALETTE[1], lw=1.4, label="ensemble mean")
    ax.fill_between(t, lo, hi, color=PALETTE[1], alpha=0.18, linewidth=0)
    ax.axhline(threshold, color="0.35", ls="--", lw=0.9, label=f"r = {threshold:.0%}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("sweep t")
    ax.set_ylabel("mean opinion  X̄(t)")
    conv = full_result.runs[(full_result.runs["latency"] == spec) & (full_result.runs["q"] == q_example)]["converged"].mean()
    ax.set_title(f"(a) Full population (n={bundle_full.hypergraph.n_nodes}), converged {conv:.0%}")
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    ax = axes[0, 1]
    for q in sorted(lcc_result.runs["q"].unique()):
        sub = lcc_result.runs[(lcc_result.runs["latency"] == spec) & (lcc_result.runs["q"] == q)]
        if sub.empty:
            continue
        times, survival = kaplan_meier(sub["consensus_time"], sub["converged"].astype(bool))
        ax.step(
            np.concatenate(([0.0], times)),
            np.concatenate(([0.0], 1.0 - survival)),
            where="post",
            color=Q_COLORS.get(q, PALETTE[0]),
            lw=1.4,
            label=f"q={q:g}",
        )
    ax.axhline(threshold, color="0.35", ls="--", lw=0.9)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, lcc_result.runs["max_steps"].max())
    ax.set_xlabel("sweep t")
    ax.set_ylabel("cumulative consensus (KM)")
    conv = lcc_result.runs[(lcc_result.runs["latency"] == spec) & (lcc_result.runs["q"] == q_example)]["converged"].mean()
    ax.set_title(f"(b) Largest component (n={bundle_lcc.hypergraph.n_nodes}), q=1.5: {conv:.0%}")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[1, 0]
    bins = np.linspace(0.4, 1.0, 25)
    ax.hist(full_result.runs["final_majority"], bins=bins, alpha=0.65, color=PALETTE[1], label="full population", density=True)
    ax.hist(lcc_result.runs["final_majority"], bins=bins, alpha=0.65, color=PALETTE[2], label="largest component", density=True)
    ax.axvline(threshold, color="0.35", ls="--", lw=0.9)
    ax.set_xlabel("final majority share")
    ax.set_ylabel("density")
    ax.set_title("(c) Final majority vs threshold")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    rng = np.random.default_rng(0)
    for q in sorted(full_result.runs["q"].unique()):
        sub = full_result.runs[full_result.runs["q"] == q]
        ax.scatter(
            np.full(len(sub), q) + rng.normal(0, 0.03, len(sub)),
            sub["component_absorbed_fraction"],
            s=8,
            color=Q_COLORS.get(q, PALETTE[0]),
            alpha=0.5,
        )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("nonlinearity q")
    ax.set_ylabel("fraction of components absorbed")
    ax.set_title("(d) Components freeze internally")

    fig.suptitle(
        "On the Kaggle hypergraph, global consensus is structurally out of reach; "
        "dynamics are per-component",
        y=1.0,
    )
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------------ figure 7
def plot_update_rules(result, path) -> Path:
    _setup_style()
    summary = summarize_cells(result.runs, by=("update_rule", "q"))
    rules = list(dict.fromkeys(result.runs["update_rule"]))
    qs = sorted(result.runs["q"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4))
    width = 0.8 / max(1, len(qs))
    x = np.arange(len(rules))
    for ax, metric, label, title in (
        (axes[0], "rmst_tc", "RMST of time to 80%", "(a) Consensus is slower"),
        (axes[1], "final_diversity", "final minority share", "(b) More diversity remains"),
        (axes[2], "oscillation_index", "O = std(X̄(t))", "(c) Stronger swings"),
    ):
        for k, q in enumerate(qs):
            vals, errs = [], []
            for rule in rules:
                row = summary[(summary["update_rule"] == rule) & (summary["q"] == q)]
                if row.empty:
                    vals.append(np.nan)
                    errs.append(0)
                    continue
                if metric == "rmst_tc":
                    vals.append(row["rmst_tc"].iloc[0])
                    errs.append((row["rmst_tc_ci_high"].iloc[0] - row["rmst_tc_ci_low"].iloc[0]) / 2)
                else:
                    vals.append(row[f"{metric}_mean"].iloc[0])
                    errs.append(
                        (row[f"{metric}_ci_high"].iloc[0] - row[f"{metric}_ci_low"].iloc[0]) / 2
                    )
            ax.bar(x + (k - (len(qs) - 1) / 2) * width, vals, width, yerr=errs, capsize=2,
                   color=Q_COLORS.get(q, PALETTE[0]), label=f"q={q:g}")
        ax.set_xticks(x)
        ax.set_xticklabels(rules, rotation=15)
        ax.set_title(title)
        ax.set_ylabel(label)
    axes[1].legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.26))
    fig.suptitle("Update-rule sensitivity: node-mean vs node-sample vs paper's edge sweep", y=1.03)
    fig.tight_layout()
    return _save(fig, path)
