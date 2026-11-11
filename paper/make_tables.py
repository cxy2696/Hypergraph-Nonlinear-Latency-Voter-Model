#!/usr/bin/env python3
"""Generate the LaTeX tables in the revised manuscript from result CSVs."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(parents=True, exist_ok=True)

LATEX_LATENCY = {
    "fixed": r"fixed",
    "normal_s1": r"$\mathcal{N}(5,1^2)$",
    "normal_s2": r"$\mathcal{N}(5,2^2)$",
    "normal_s4": r"$\mathcal{N}(5,4^2)$",
    "exponential": r"Exp$(5)$",
}


def f(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def tc(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if math.isinf(x):
        return r"$\infty$"
    return f"{x:.0f}"


def pv(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p <= 0.00035:
        return r"$3\times10^{-4}$"
    if p < 0.001:
        return r"$<0.001$"
    return f"{p:.3f}"


def delta(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f}"


def ci(lo: float, hi: float, nd: int = 3) -> str:
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def write(name: str, body: str) -> None:
    (OUT / f"{name}.tex").write_text(body.strip() + "\n", encoding="utf-8")


def table_random() -> None:
    df = pd.read_csv(ROOT / "results" / "tables" / "random_cells.csv")
    order = ["fixed", "normal_s1", "normal_s2", "normal_s4", "exponential"]
    df["latency"] = pd.Categorical(df["latency"], categories=order, ordered=True)
    df = df.sort_values(["latency", "q"])
    lines = [
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"Latency & $q$ & $P(\mathrm{cons.})$ & KM $T_c$ & RMST & final $D$ & $O$ & $\max C$ \\",
        r"\midrule",
    ]
    current = None
    for _, row in df.iterrows():
        lat = str(row["latency"])
        if current is not None and lat != current:
            lines.append(r"\midrule")
        current = lat
        label = LATEX_LATENCY[lat] if row["q"] == df[df["latency"] == lat]["q"].min() else ""
        lines.append(
            " & ".join(
                [
                    label,
                    f"{row['q']:.1f}",
                    f(row["consensus_rate"]),
                    tc(row["km_median_tc"]),
                    f(row["rmst_tc"], 1),
                    f(row["final_diversity_mean"]),
                    f(row["oscillation_index_mean"]),
                    f(row["max_cascade_mean"]),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_random", "\n".join(lines))


def table_effects() -> None:
    df = pd.read_csv(ROOT / "results" / "tables" / "random_q_contrasts.csv")
    order = ["fixed", "normal_s1", "normal_s2", "normal_s4", "exponential"]
    div = df[(df["metric"] == "final_diversity") & np.isclose(df["level"], 2.5)].set_index("latency")
    conv = df[(df["metric"] == "converged") & np.isclose(df["level"], 2.5)].set_index("latency")
    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{$\Delta$ final diversity} & \multicolumn{3}{c}{$\Delta P(\mathrm{consensus})$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Latency & $\Delta$ & 95\% CI & $p_{\mathrm{Holm}}$ & $\Delta$ & 95\% CI & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for lat in order:
        d = div.loc[lat]
        c = conv.loc[lat]
        lines.append(
            " & ".join(
                [
                    LATEX_LATENCY[lat],
                    delta(d["mean_diff"]),
                    ci(d["ci_low"], d["ci_high"]),
                    pv(d["p_holm"]),
                    delta(c["mean_diff"]),
                    ci(c["ci_low"], c["ci_high"]),
                    pv(c["p_holm"]),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_effects", "\n".join(lines))


def table_dataset() -> None:
    struct = pd.read_csv(ROOT / "results" / "tables" / "dataset_structure.csv").set_index("scope")
    cells = pd.read_csv(ROOT / "results" / "tables" / "dataset_cells.csv")
    runs = {
        "active": pd.read_csv(ROOT / "results" / "dataset_active_runs.csv"),
        "full": pd.read_csv(ROOT / "results" / "dataset_full_runs.csv"),
        "active_data_init": pd.read_csv(ROOT / "results" / "dataset_datainit_runs.csv"),
        "lcc": pd.read_csv(ROOT / "results" / "dataset_lcc_runs.csv"),
    }
    labels = {
        "full": "full",
        "active": "active",
        "active_data_init": "active, data init",
        "lcc": "largest comp.",
    }
    lines = [
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"Scope & $n$ & $m$ & $C$ & largest & $P(\mathrm{cons.})$ & final $D$ & max majority \\",
        r"\midrule",
    ]
    for scope in ("full", "active", "active_data_init", "lcc"):
        key = "active" if scope == "active_data_init" else scope
        s = struct.loc[key]
        c = cells[cells["scope"] == scope]
        p = float(c["consensus_rate"].mean())
        d = float(c["final_diversity_mean"].mean())
        mm = float(runs[scope]["final_majority"].max())
        lines.append(
            " & ".join(
                [
                    labels[scope],
                    f"{int(s['n_nodes'])}",
                    f"{int(s['n_edges'])}",
                    f"{int(s['components'])}",
                    f"{int(s['largest_component'])} ({100*s['largest_component_share']:.1f}\\%)",
                    f(p),
                    f(d),
                    f(mm),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_dataset", "\n".join(lines))


def table_rules() -> None:
    df = pd.read_csv(ROOT / "results" / "tables" / "rule_sensitivity.csv")
    order = ["node_mean", "node_sample", "edge_sweep"]
    pretty = {
        "node_mean": r"node-mean",
        "node_sample": r"node-sample",
        "edge_sweep": r"edge sweep",
    }
    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"Rule & $q$ & $P(\mathrm{cons.})$ & RMST & final $D$ & $O$ \\",
        r"\midrule",
    ]
    for rule in order:
        sub = df[df["update_rule"] == rule].sort_values("q")
        for i, (_, row) in enumerate(sub.iterrows()):
            lines.append(
                " & ".join(
                    [
                        pretty[rule] if i == 0 else "",
                        f"{row['q']:.1f}",
                        f(row["consensus_rate"]),
                        f(row["rmst_tc"], 1),
                        f(row["final_diversity_mean"]),
                        f(row["oscillation_index_mean"]),
                    ]
                )
                + r" \\"
            )
        if rule != order[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_rules", "\n".join(lines))


if __name__ == "__main__":
    table_random()
    table_effects()
    table_dataset()
    table_rules()
    print("wrote tables to", OUT)
