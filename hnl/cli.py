"""Command line interface.

Examples
--------
Single run::

    python -m hnl simulate --q 1.5 --latency normal --sigma 2 --seed 42

Dataset diagnostics::

    python -m hnl diagnose --csv sentimentdataset.csv --out results/diagnostics.md

Full reproduction (experiments + tables + figures + manifest)::

    python -m hnl reproduce --root .
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import __version__
from .data import build_dataset, dataset_report
from .experiments import (
    DEFAULT_SPECS,
    Design,
    LatencySpec,
    damping_table,
    paired_contrasts,
    run_dataset_experiment,
    run_random_experiment,
    run_rule_sensitivity,
    summarize_cells,
)
from .hypergraph import random_hypergraph
from .model import HNLVoterModel
from .provenance import build_manifest, write_json
from .report import md_table, write_markdown


def _floats(text: str) -> Tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def _spec(name: str) -> LatencySpec:
    for spec in DEFAULT_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def _bundle(bundle, csv: str) -> Dict[str, object]:
    return {"csv": csv, "metadata": bundle.metadata}


# --------------------------------------------------------------------- simulate
def cmd_simulate(args: argparse.Namespace) -> int:
    if args.preset == "random":
        rng = np.random.default_rng(args.seed)
        hg = random_hypergraph(
            args.n, args.edges, (args.size_min, args.size_max), rng
        )
        init = rng.integers(0, 2, size=hg.n_nodes, dtype=np.int64)
        labels = None
    else:
        bundle = build_dataset(
            args.csv, mapping=args.mapping, scope=args.scope
        )
        hg = bundle.hypergraph
        init = bundle.opinions_for(args.init, np.random.default_rng(args.seed))
        labels = bundle.labels

    from .config import ModelConfig

    cfg = ModelConfig(
        q=args.q,
        latency_dist=args.latency,
        mu=args.mu,
        sigma=args.sigma,
        consensus_threshold=args.threshold,
        max_steps=args.steps,
        update_rule=args.rule,
    )
    model = HNLVoterModel.from_hypergraph(
        hg, cfg, seed=args.seed, initial_opinions=init
    )
    if labels is not None:
        model.hg.labels = labels
    result = model.run()
    summary = {
        "version": __version__,
        "q": args.q,
        "update_rule": args.rule,
        "latency": args.latency,
        "n_nodes": int(hg.n_nodes),
        "n_edges": int(hg.n_edges),
        "steps_run": result.n_steps,
        "consensus_time": result.consensus_time,
        "censored": result.censored,
        "final_majority": round(result.final_majority, 4),
        "final_diversity": round(result.final_diversity, 4),
        "oscillation_index": round(result.oscillation_index, 4),
        "max_cascade": round(result.max_cascade, 4),
        "mean_latency": round(float(result.latencies.mean()), 3),
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        write_json(args.json, summary)
    if args.plot:
        model.plot_metrics(args.plot)
    return 0


# -------------------------------------------------------------------- diagnose
def cmd_diagnose(args: argparse.Namespace) -> int:
    bundle = build_dataset(
        args.csv, mapping=args.mapping, scope=args.scope
    )
    text = dataset_report(bundle)
    s = bundle.metadata["hypergraph"]
    text += "\n## Structural implication\n\n"
    text += (
        f"The largest connected component contains {s['largest_component']} of "
        f"{s['n_nodes']} nodes ({s['largest_component_share']:.1%}). A population-wide "
        f"majority of r = {args.threshold:.0%} therefore requires agreement across "
        f"components that can only interact through shared hashtags; the model's "
        f"global consensus threshold is not a per-group quantity.\n"
    )
    if args.out:
        write_markdown(args.out, text)
        print(f"wrote {args.out}")
    else:
        print(text)
    if args.json:
        write_json(args.json, bundle.metadata)
        print(f"wrote {args.json}")
    return 0


# -------------------------------------------------------------------- reproduce
def _design(
    name: str,
    seed: int,
    repeats: int,
    q_values: Sequence[float],
    specs: Sequence[LatencySpec],
    max_steps: int,
    threshold: float,
    n_nodes: int = 100,
    n_edges: int = 200,
    size_range: Tuple[int, int] = (3, 6),
    update_rule: str = "node_mean",
) -> Design:
    return Design(
        name=name,
        n_nodes=n_nodes,
        n_edges=n_edges,
        size_range=size_range,
        q_values=tuple(float(q) for q in q_values),
        latency_specs=tuple(specs),
        repeats=int(repeats),
        max_steps=int(max_steps),
        consensus_threshold=float(threshold),
        base_seed=int(seed),
        update_rule=update_rule,
    )


def _contrast_family(
    runs: pd.DataFrame,
    factor: str,
    reference,
    by: Sequence[str],
    metrics: Iterable[str],
) -> pd.DataFrame:
    frames = []
    for metric in metrics:
        frames.append(
            paired_contrasts(runs, factor=factor, reference=reference, metric=metric, by=by)
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_tables(out: Path, payload: Dict[str, pd.DataFrame]) -> None:
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, frame in payload.items():
        frame.to_csv(tables / f"{name}.csv", index=False)


def _results_markdown(tables: Dict[str, pd.DataFrame], structures: Dict[str, Dict]) -> str:
    lines = ["# Generated result tables", ""]
    lines.append(
        "All tables are written by `python -m hnl.cli reproduce`; numbers are "
        "reproducible from the committed seeds and manifest."
    )
    lines.append("")

    if "random_cells" in tables:
        df = tables["random_cells"]
        lines += ["## 1. Random hypergraphs: cell summaries", ""]
        lines.append(
            md_table(
                df,
                [
                    "latency", "q", "n", "consensus_rate", "km_median_tc", "rmst_tc",
                    "final_diversity_mean", "oscillation_index_mean", "max_cascade_mean",
                ],
                ["latency", "q", "n", "P(consensus)", "KM median T_c", "RMST(T_c)",
                 "final D", "O", "max C"],
                formats={"n": "d", "q": ".2f"},
            )
        )
        lines.append("")

    if "random_q_contrasts" in tables:
        df = tables["random_q_contrasts"]
        lines += ["## 2. Paired contrasts across q", ""]
        lines.append(
            md_table(
                df,
                ["latency", "metric", "level", "n", "mean_diff", "ci_low", "ci_high", "p_holm", "cliffs_delta"],
                ["latency", "metric", "level vs q=0.5", "n", "mean diff", "95% CI low",
                 "95% CI high", "p (Holm)", "Cliff δ"],
                formats={"n": "d", "level": ".2f"},
            )
        )
        lines.append("")

    if "random_latency_contrasts" in tables:
        df = tables["random_latency_contrasts"]
        lines += ["## 3. Paired contrasts across latency distributions (reference: fixed)", ""]
        lines.append(
            md_table(
                df,
                ["q", "metric", "level", "n", "mean_diff", "ci_low", "ci_high", "p_holm", "cliffs_delta"],
                ["q", "metric", "level", "n", "mean diff", "95% CI low", "95% CI high",
                 "p (Holm)", "Cliff δ"],
                formats={"n": "d", "q": ".2f"},
            )
        )
        lines.append("")

    if "damping" in tables:
        df = tables["damping"]
        lines += ["## 4. Are oscillations transient?", ""]
        lines.append(
            md_table(
                df,
                ["latency", "q", "n", "mean_first_half", "mean_second_half", "mean_ratio",
                 "mean_diff", "ci_low", "ci_high", "p_value"],
                ["latency", "q", "n", "O first half", "O second half", "ratio",
                 "diff (second−first)", "95% CI low", "95% CI high", "p"],
                formats={"n": "d", "q": ".2f"},
            )
        )
        lines.append("")

    if "dataset_cells" in tables:
        df = tables["dataset_cells"]
        lines += ["## 5. Dataset hypergraph: cell summaries (random initial opinions)", ""]
        lines.append(
            md_table(
                df,
                ["scope", "latency", "q", "n", "consensus_rate", "km_median_tc",
                 "final_diversity_mean", "component_absorbed_fraction_mean"],
                ["scope", "latency", "q", "n", "P(consensus)", "KM median T_c",
                 "final D", "components absorbed"],
                formats={"n": "d", "q": ".2f"},
            )
        )
        lines.append("")

    if "dataset_q_contrasts" in tables:
        df = tables["dataset_q_contrasts"]
        lines += ["## 6. Dataset hypergraph: paired q contrasts", ""]
        lines.append(
            md_table(
                df,
                ["scope", "latency", "metric", "level", "n", "mean_diff", "ci_low",
                 "ci_high", "p_holm", "cliffs_delta"],
                ["scope", "latency", "metric", "level", "n", "mean diff", "95% CI low",
                 "95% CI high", "p (Holm)", "Cliff δ"],
                formats={"n": "d", "level": ".2f"},
            )
        )
        lines.append("")

    if "rule_sensitivity" in tables:
        df = tables["rule_sensitivity"]
        lines += ["## 7. Update-rule sensitivity", ""]
        lines.append(
            md_table(
                df,
                ["update_rule", "q", "n", "consensus_rate", "rmst_tc",
                 "final_diversity_mean", "oscillation_index_mean"],
                ["update rule", "q", "n", "P(consensus)", "RMST(T_c)", "final D", "O"],
                formats={"n": "d", "q": ".2f"},
            )
        )
        lines.append("")
    return "\n".join(lines)


def cmd_reproduce(args: argparse.Namespace) -> int:
    t0 = time.time()
    root = Path(args.root).resolve()
    figures = root / "figures"
    results = root / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    csv = Path(args.csv)
    if not csv.is_absolute():
        csv = root / csv

    quick = args.quick
    repeats = 5 if quick else args.repeats
    steps = 60 if quick else args.steps
    q_values = (0.5, 1.5, 2.5) if quick else (0.5, 1.0, 1.5, 2.5)
    full_specs = (DEFAULT_SPECS[0], DEFAULT_SPECS[1], DEFAULT_SPECS[2], DEFAULT_SPECS[3], DEFAULT_SPECS[4])
    if quick:
        full_specs = (DEFAULT_SPECS[0], DEFAULT_SPECS[3])

    print(f"[1/6] loading dataset {csv.name}", flush=True)
    bundle_full = build_dataset(str(csv), mapping="valence", scope="full")
    bundle_active = build_dataset(str(csv), mapping="valence", scope="active")
    bundle_lcc = bundle_active.subgraph(bundle_active.hypergraph.giant_component())
    structures = {
        "full": bundle_full.metadata,
        "active": bundle_active.metadata,
        "lcc": bundle_lcc.metadata,
    }
    write_markdown(results / "diagnostics.md", dataset_report(bundle_full))
    write_json(results / "diagnostics.json", bundle_full.metadata)

    print("[2/6] random-hypergraph factorial design", flush=True)
    random_design = _design(
        "random", seed=args.seed, repeats=repeats, q_values=q_values,
        specs=full_specs, max_steps=steps, threshold=args.threshold,
    )
    random_res = run_random_experiment(random_design, progress=True)
    random_res.save(results)

    print("[3/6] dataset designs (full / active / largest component / data init)", flush=True)
    dataset_active = run_dataset_experiment(
        bundle_active,
        _design("dataset_active", seed=args.seed + 101, repeats=repeats,
                q_values=q_values, specs=full_specs, max_steps=steps,
                threshold=args.threshold),
        init="random",
        progress=True,
    )
    dataset_active.save(results)

    dataset_full = run_dataset_experiment(
        bundle_full,
        _design("dataset_full", seed=args.seed + 202, repeats=max(5, repeats // 2) if not quick else repeats,
                q_values=q_values, specs=(DEFAULT_SPECS[0], DEFAULT_SPECS[2], DEFAULT_SPECS[3]),
                max_steps=steps, threshold=args.threshold),
        init="random",
        progress=True,
    )
    dataset_full.save(results)

    dataset_lcc = run_dataset_experiment(
        bundle_lcc,
        _design("dataset_lcc", seed=args.seed + 303, repeats=max(5, repeats // 2) if not quick else repeats,
                q_values=q_values, specs=(DEFAULT_SPECS[0], DEFAULT_SPECS[2], DEFAULT_SPECS[3]),
                max_steps=steps, threshold=args.threshold),
        init="random",
        progress=True,
    )
    dataset_lcc.save(results)

    dataset_data = run_dataset_experiment(
        bundle_active,
        _design("dataset_datainit", seed=args.seed + 404,
                repeats=max(5, repeats // 2) if not quick else repeats,
                q_values=q_values, specs=(DEFAULT_SPECS[0], DEFAULT_SPECS[3]),
                max_steps=steps, threshold=args.threshold),
        init="data",
        progress=True,
    )
    dataset_data.save(results)

    print("[4/6] update-rule sensitivity", flush=True)
    rules_res = run_rule_sensitivity(
        random_design,
        rules=("node_mean", "node_sample", "edge_sweep"),
        q_values=(0.5, 1.5, 2.5) if quick else (0.5, 1.0, 1.5, 2.5),
        repeats=4 if quick else 8,
    )
    rules_res.save(results)

    print("[5/6] summaries and contrasts", flush=True)
    tables: Dict[str, pd.DataFrame] = {}
    tables["random_cells"] = summarize_cells(random_res.runs)
    tables["random_q_contrasts"] = _contrast_family(
        random_res.runs, "q", 0.5, ("latency",),
        ("final_diversity", "converged", "oscillation_index", "max_cascade"),
    )
    tables["random_latency_contrasts"] = _contrast_family(
        random_res.runs, "latency", "fixed", ("q",),
        ("final_diversity", "converged", "oscillation_index"),
    )
    tables["damping"] = damping_table(random_res.runs)

    dataset_frames = []
    dataset_contrasts = []
    for scope_name, result in (
        ("active", dataset_active),
        ("full", dataset_full),
        ("lcc", dataset_lcc),
        ("active_data_init", dataset_data),
    ):
        frame = summarize_cells(result.runs)
        frame.insert(0, "scope", scope_name)
        dataset_frames.append(frame)
        contrasts = _contrast_family(
            result.runs, "q", 0.5, ("latency",), ("final_diversity", "oscillation_index")
        )
        if not contrasts.empty:
            contrasts.insert(0, "scope", scope_name)
            dataset_contrasts.append(contrasts)
    tables["dataset_cells"] = pd.concat(dataset_frames, ignore_index=True)
    tables["dataset_q_contrasts"] = (
        pd.concat(dataset_contrasts, ignore_index=True) if dataset_contrasts else pd.DataFrame()
    )
    rules_cells = summarize_cells(rules_res.runs, by=("update_rule", "q"))
    tables["rule_sensitivity"] = rules_cells

    structure_rows = []
    for scope_name, meta in structures.items():
        s = meta["hypergraph"]
        structure_rows.append(
            {
                "scope": scope_name,
                "n_nodes": s["n_nodes"],
                "n_edges": s["n_edges"],
                "coverage": s["coverage"],
                "isolated": s["isolated_nodes"],
                "components": s["n_components"],
                "largest_component": s["largest_component"],
                "largest_component_share": s["largest_component_share"],
                "mean_edge_size": s["mean_edge_size"],
                "mean_degree_covered": s["mean_degree_covered"],
                "users_with_valence": meta["n_users_with_valence"],
                "users_unmapped": meta["n_users_unmapped"],
            }
        )
    tables["dataset_structure"] = pd.DataFrame(structure_rows)
    _write_tables(results, tables)
    write_markdown(results / "TABLES.md", _results_markdown(tables, structures))

    print("[6/6] figures and manifest", flush=True)
    if not args.no_figures:
        from . import plotting

        plotted = []
        plotted.append(plotting.plot_diversity(random_res, figures / "fig1_diversity.png"))
        plotted.append(plotting.plot_consensus_km(random_res, figures / "fig2_consensus_km.png"))
        plotted.append(plotting.plot_oscillation(random_res, figures / "fig3_oscillation.png"))
        plotted.append(plotting.plot_cascade(random_res, figures / "fig4_cascade.png"))
        plotted.append(plotting.plot_dataset_structure(bundle_full, figures / "fig5_dataset_structure.png"))
        plotted.append(
            plotting.plot_dataset_dynamics(
                dataset_full, dataset_lcc, bundle_full, bundle_lcc,
                figures / "fig6_dataset_dynamics.png",
            )
        )
        plotted.append(plotting.plot_update_rules(rules_res, figures / "fig7_update_rules.png"))
        for path in plotted:
            print(f"  figure: {path.relative_to(root)}")

    artifacts = sorted(
        [
            path
            for path in list(results.glob("*"))
            + list((results / "tables").glob("*"))
            + list(figures.glob("*"))
            if path.is_file() and path.name != "manifest.json"
        ]
    )
    manifest = build_manifest(
        root=root,
        argv=sys.argv,
        designs={
            "random": random_res.design,
            "dataset_active": dataset_active.design,
            "dataset_full": dataset_full.design,
            "dataset_lcc": dataset_lcc.design,
            "dataset_datainit": dataset_data.design,
            "rule_sensitivity": rules_res.design,
        },
        datasets={
            "full": bundle_full.metadata,
            "active": bundle_active.metadata,
            "lcc": bundle_lcc.metadata,
        },
        artifacts=artifacts,
    )
    manifest_path = write_json(results / "manifest.json", manifest)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s; manifest: {manifest_path.relative_to(root)}")
    return 0


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hnl",
        description="Hypergraph–Nonlinear–Latency voter model: reference implementation.",
    )
    parser.add_argument("--version", action="version", version=f"hnl {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="run a single simulation")
    p.add_argument("--preset", choices=("random", "dataset"), default="random")
    p.add_argument("--q", type=float, default=1.5)
    p.add_argument("--latency", choices=("normal", "exponential", "fixed"), default="normal")
    p.add_argument("--mu", type=float, default=5.0)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--rule", choices=("node_mean", "node_sample", "edge_sweep"), default="node_mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--edges", type=int, default=200)
    p.add_argument("--size-min", type=int, default=3)
    p.add_argument("--size-max", type=int, default=6)
    p.add_argument("--csv", default="sentimentdataset.csv")
    p.add_argument("--mapping", choices=("valence", "paper", "strict"), default="valence")
    p.add_argument("--scope", choices=("active", "full"), default="active")
    p.add_argument("--init", choices=("random", "data"), default="random")
    p.add_argument("--plot", default=None)
    p.add_argument("--json", default=None)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("diagnose", help="structural diagnostics of a post table")
    p.add_argument("--csv", default="sentimentdataset.csv")
    p.add_argument("--mapping", choices=("valence", "paper", "strict"), default="valence")
    p.add_argument("--scope", choices=("active", "full"), default="full")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--out", default=None)
    p.add_argument("--json", default=None)
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("reproduce", help="run the full experiment suite")
    p.add_argument("--root", default=".")
    p.add_argument("--csv", default="sentimentdataset.csv")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=20260101)
    p.add_argument("--quick", action="store_true", help="small smoke run (5 replicates)")
    p.add_argument("--no-figures", action="store_true")
    p.set_defaults(func=cmd_reproduce)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
