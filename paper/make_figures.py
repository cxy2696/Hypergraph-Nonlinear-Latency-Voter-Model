#!/usr/bin/env python3
"""Build the figures used in the revised manuscript.

Everything is drawn from the committed HNL v2 result files; no numbers are
hard-coded. Run from the repository root:

    python paper/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hnl.data import build_dataset  # noqa: E402
from hnl.metrics import kaplan_meier  # noqa: E402
from hnl.stats import bootstrap_ci  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLORS = {
    "fixed": "#4D4D4D",
    "normal_s1": "#56B4E9",
    "normal_s2": "#009E73",
    "normal_s4": "#D55E00",
    "exponential": "#CC79A7",
}
LABELS = {
    "fixed": r"fixed $L=5$",
    "normal_s1": r"$\sigma=1$",
    "normal_s2": r"$\sigma=2$",
    "normal_s4": r"$\sigma=4$",
    "exponential": "exponential",
}
QCOLORS = {0.5: "#0072B2", 1.0: "#009E73", 1.5: "#E69F00", 2.5: "#D55E00"}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "lines.linewidth": 1.1,
            "savefig.dpi": 300,
        }
    )


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)


def panel(ax, letter: str) -> None:
    ax.text(
        0.015,
        0.975,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    )


def band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    lo = np.nanpercentile(values, 2.5, axis=0)
    hi = np.nanpercentile(values, 97.5, axis=0)
    return mean, lo, hi


def fig_diversity(traj) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, spec, letter in zip(axes, ("fixed", "normal_s4"), ("a", "b")):
        for q in (0.5, 1.0, 1.5, 2.5):
            mean, lo, hi = band(traj[f"div|{spec}|q={q}"])
            t = np.arange(1, mean.size + 1)
            ax.plot(t, mean, color=QCOLORS[q], label=rf"$q={q:g}$")
            ax.fill_between(t, lo, hi, color=QCOLORS[q], alpha=0.13, linewidth=0)
        ax.set_xlim(1, 100)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("sweep $t$")
        ax.grid(True)
        panel(ax, letter)
        ax.text(
            0.97,
            0.06,
            LABELS[spec],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
        )
    axes[0].set_ylabel(r"minority share $D(t)$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.055),
               handlelength=1.4, columnspacing=1.4)
    save(fig, "fig_diversity")


def fig_consensus(runs) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, spec, letter in zip(axes, ("fixed", "normal_s4"), ("a", "b")):
        for q in (0.5, 1.0, 1.5, 2.5):
            sub = runs[(runs["latency"] == spec) & (runs["q"] == q)]
            times, survival = kaplan_meier(
                sub["consensus_time"], sub["converged"].astype(bool)
            )
            t = np.concatenate(([0.0], times))
            conv = np.concatenate(([0.0], 1.0 - survival))
            ax.step(t, conv, where="post", color=QCOLORS[q], label=rf"$q={q:g}$")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("sweep $t$")
        ax.grid(True)
        panel(ax, letter)
        ax.text(
            0.97,
            0.06,
            LABELS[spec],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
        )
    axes[0].set_ylabel("cumulative consensus incidence")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.055),
               handlelength=1.4, columnspacing=1.4)
    save(fig, "fig_consensus")


def fig_oscillation(runs, traj) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    summary = pd.read_csv(ROOT / "results" / "tables" / "random_cells.csv")

    ax = axes[0]
    for spec in ("fixed", "normal_s2", "normal_s4"):
        sub = summary[summary["latency"] == spec]
        ax.errorbar(
            sub["q"],
            sub["oscillation_index_mean"],
            yerr=[
                sub["oscillation_index_mean"] - sub["oscillation_index_ci_low"],
                sub["oscillation_index_ci_high"] - sub["oscillation_index_mean"],
            ],
            marker="o",
            ms=2.6,
            lw=0.9,
            capsize=1.8,
            color=COLORS[spec],
            label=LABELS[spec],
        )
    ax.set_xlabel("nonlinearity $q$")
    ax.set_ylabel(r"$O=\mathrm{sd}(\bar X(t))$")
    ax.set_xticks([0.5, 1.0, 1.5, 2.5])
    ax.grid(True)
    panel(ax, "a")

    ax = axes[1]
    for spec in ("fixed", "normal_s4"):
        arr = traj[f"mean|{spec}|q=1.5"]
        t = np.arange(1, arr.shape[1] + 1)
        for row in arr[:4]:
            ax.plot(t, row, color=COLORS[spec], lw=0.55, alpha=0.28)
        ax.plot(t, np.nanmean(arr, axis=0), color=COLORS[spec], lw=1.2)
    ax.axhline(0.5, color="0.5", lw=0.6, ls=":")
    ax.set_xlim(1, 100)
    ax.set_ylim(0.15, 0.85)
    ax.set_xlabel("sweep $t$")
    ax.set_ylabel(r"mean opinion $\bar X(t)$")
    ax.grid(True)
    panel(ax, "b")
    ax.text(0.03, 0.06, "thick: ensemble mean", transform=ax.transAxes, fontsize=6.5)
    ax.text(0.03, 0.13, "thin: four individual runs", transform=ax.transAxes, fontsize=6.5)

    ax = axes[2]
    finite = runs[np.isfinite(runs["oscillation_ratio"])]
    for spec in ("fixed", "normal_s2", "normal_s4"):
        xs, ys, errs = [], [], []
        for q in (0.5, 1.0, 1.5, 2.5):
            vals = finite[(finite["latency"] == spec) & np.isclose(finite["q"], q)][
                "oscillation_ratio"
            ].to_numpy()
            if vals.size == 0:
                continue
            lo, hi = bootstrap_ci(vals, np.mean, n_boot=1000, rng=np.random.default_rng(0))
            xs.append(q)
            ys.append(float(np.mean(vals)))
            errs.append((hi - lo) / 2)
        ax.errorbar(
            xs,
            ys,
            yerr=errs,
            marker="o",
            ms=2.6,
            lw=0.9,
            capsize=1.8,
            color=COLORS[spec],
            label=LABELS[spec],
        )
    ax.axhline(1.0, color="0.4", lw=0.7, ls="--")
    ax.set_xlabel("nonlinearity $q$")
    ax.set_ylabel("late / early amplitude")
    ax.set_xticks([0.5, 1.0, 1.5, 2.5])
    ax.grid(True)
    panel(ax, "c")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06),
               handlelength=1.4, columnspacing=1.4)
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig_oscillation")


def fig_cascade(runs) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    summary = pd.read_csv(ROOT / "results" / "tables" / "random_cells.csv")

    ax = axes[0]
    for spec in ("fixed", "normal_s2", "normal_s4"):
        sub = summary[summary["latency"] == spec]
        ax.errorbar(
            sub["q"],
            sub["max_cascade_mean"],
            yerr=[
                sub["max_cascade_mean"] - sub["max_cascade_ci_low"],
                sub["max_cascade_ci_high"] - sub["max_cascade_mean"],
            ],
            marker="o",
            ms=2.6,
            lw=0.9,
            capsize=1.8,
            color=COLORS[spec],
            label=LABELS[spec],
        )
    ax.set_xlabel("nonlinearity $q$")
    ax.set_ylabel(r"max $C(t)$")
    ax.set_xticks([0.5, 1.0, 1.5, 2.5])
    ax.set_ylim(0, 0.5)
    ax.grid(True)
    panel(ax, "a")

    ax = axes[1]
    sub = runs[runs["latency"] == "normal_s4"]
    means = sub.groupby("q")["total_cascade_padded"].mean()
    errs = sub.groupby("q")["total_cascade_padded"].sem()
    ax.bar(
        [str(q) for q in means.index],
        means.to_numpy(),
        yerr=errs.to_numpy(),
        capsize=2.0,
        color=[QCOLORS[q] for q in means.index],
        width=0.62,
        error_kw={"lw": 0.7},
    )
    ax.set_xlabel("nonlinearity $q$")
    ax.set_ylabel(r"$\sum_t C(t)$")
    ax.set_ylim(0, 23)
    ax.grid(True, axis="y")
    panel(ax, "b")
    ax.text(0.97, 0.95, LABELS["normal_s4"], transform=ax.transAxes, ha="right", va="top", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06),
               handlelength=1.4, columnspacing=1.4)
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig_cascade")


def fig_dataset(full_bundle, active_bundle, full_runs, lcc_runs, full_traj, lcc_traj) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.15))
    hg_full = full_bundle.hypergraph
    hg_active = active_bundle.hypergraph

    ax = axes[0, 0]
    for hg, color, label in (
        (hg_full, "#0072B2", "full"),
        (hg_active, "#D55E00", "active"),
    ):
        _, sizes = hg.components()
        ranked = np.sort(sizes)[::-1]
        ax.plot(
            np.arange(1, ranked.size + 1),
            ranked,
            color=color,
            marker="o",
            ms=1.8,
            lw=0.8,
            label=f"{label} ($C={sizes.size}$)",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("component rank")
    ax.set_ylabel("component size")
    ax.grid(True, which="both")
    panel(ax, "a")
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    degree = hg_full.degree
    counts = np.bincount(degree)
    ax.bar(np.arange(counts.size), counts, color="#009E73", width=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("hyperedge degree")
    ax.set_ylabel("nodes")
    ax.grid(True, axis="y")
    panel(ax, "b")
    ax.text(
        0.97,
        0.95,
        f"isolated: {int((degree == 0).sum())}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )

    ax = axes[0, 2]
    counts = {int(k): v for k, v in full_bundle.metadata["hashtag_group_size_counts"].items()}
    xs = np.array(sorted(counts))
    ys = np.array([counts[x] for x in xs])
    ax.bar(xs, ys, color="#CC79A7", width=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("users per hashtag")
    ax.set_ylabel("hashtags")
    ax.grid(True, axis="y")
    panel(ax, "c")
    ax.text(
        0.97,
        0.95,
        f"{counts.get(1, 0)} singletons\n{int(ys[xs >= 2].sum())} multi-user",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )

    ax = axes[1, 0]
    arr = full_traj["mean|normal_s4|q=1.5"]
    mean, lo, hi = band(arr)
    t = np.arange(1, mean.size + 1)
    ax.plot(t, mean, color="#D55E00", lw=1.0)
    ax.fill_between(t, lo, hi, color="#D55E00", alpha=0.15, linewidth=0)
    ax.axhline(0.8, color="0.35", ls="--", lw=0.7)
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel("sweep $t$")
    ax.set_ylabel(r"$\bar X(t)$ (full)")
    ax.grid(True)
    panel(ax, "d")
    ax.text(0.97, 0.9, "0/180 converged", transform=ax.transAxes, ha="right", fontsize=7)

    ax = axes[1, 1]
    for q in (0.5, 1.0, 1.5, 2.5):
        sub = lcc_runs[(lcc_runs["latency"] == "normal_s4") & (lcc_runs["q"] == q)]
        times, survival = kaplan_meier(sub["consensus_time"], sub["converged"].astype(bool))
        ax.step(
            np.concatenate(([0.0], times)),
            np.concatenate(([0.0], 1.0 - survival)),
            where="post",
            color=QCOLORS[q],
            lw=0.9,
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("sweep $t$")
    ax.set_ylabel("LCC consensus incidence")
    ax.grid(True)
    panel(ax, "e")

    ax = axes[1, 2]
    bins = np.linspace(0.45, 0.85, 18)
    ax.hist(
        full_runs["final_majority"],
        bins=bins,
        density=True,
        alpha=0.65,
        color="#D55E00",
        label="full",
    )
    ax.hist(
        lcc_runs["final_majority"],
        bins=bins,
        density=True,
        alpha=0.55,
        color="#0072B2",
        label="LCC",
    )
    ax.axvline(0.8, color="0.35", ls="--", lw=0.7)
    ax.set_xlabel("final majority share")
    ax.set_ylabel("density")
    ax.grid(True, axis="y")
    panel(ax, "f")
    ax.legend(loc="upper right", fontsize=6.5, frameon=True, facecolor="white",
              framealpha=0.85, edgecolor="none")

    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    save(fig, "fig_dataset")


def main() -> None:
    style()
    runs = pd.read_csv(ROOT / "results" / "random_runs.csv")
    traj = np.load(ROOT / "results" / "random_trajectories.npz")
    fig_diversity(traj)
    fig_consensus(runs)
    fig_oscillation(runs, traj)
    fig_cascade(runs)

    full_bundle = build_dataset(ROOT / "sentimentdataset.csv", scope="full")
    active_bundle = build_dataset(ROOT / "sentimentdataset.csv", scope="active")
    full_runs = pd.read_csv(ROOT / "results" / "dataset_full_runs.csv")
    lcc_runs = pd.read_csv(ROOT / "results" / "dataset_lcc_runs.csv")
    full_traj = np.load(ROOT / "results" / "dataset_full_trajectories.npz")
    lcc_traj = np.load(ROOT / "results" / "dataset_lcc_trajectories.npz")
    fig_dataset(full_bundle, active_bundle, full_runs, lcc_runs, full_traj, lcc_traj)
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
