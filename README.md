# Hypergraph–Nonlinear–Latency Voter Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![tests](https://github.com/cxy2696/Hypergraph-Nonlinear-Latency-Voter-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/cxy2696/Hypergraph-Nonlinear-Latency-Voter-Model/actions/workflows/ci.yml)

This is the research repository for the hypergraph–nonlinear–latency (HNL)
voter model. It contains the revised paper, the binary reference
implementation, the experiment scripts, and the result files behind every
figure and table.

The repository has one job: make the paper's numbers checkable. It is not a
service, a forecasting product, or a general-purpose simulation platform. The
engineering implementation in the same model family lives in
[OpinionLab](https://github.com/cxy2696/opinionlab); the division of labour is
described in [`docs/RELATIONSHIP.md`](docs/RELATIONSHIP.md).

## What the paper reports

The original project combined three ideas: group interactions on a hypergraph,
a nonlinear adoption kernel controlled by an exponent `q`, and heterogeneous
reaction delays. It concluded that low `q` and high latency variance preserve
diversity, that moderate `q` produces sustained oscillations, that high `q`
produces larger cascades, and that the model reproduces social-media dynamics.

We rebuilt the model and reran the experiments with paired replicates. Most of
the nonlinearity result holds. The other claims do not survive the reanalysis.
On connected random hypergraphs, an 80 % majority is reached in none of the runs
at `q=0.5` and in 87–100 % of the runs at `q=2.5`. The effect of latency
variance is small and appears only near balance. Oscillations are damped
transients, not cycles. Cascades shrink as `q` rises, rather than growing. The
Kaggle hashtag graph turns out to be too fragmented for a population-level
consensus experiment: none of 900 runs reaches 80 %, and the process freezes
into a mosaic in which each connected component is internally unanimous.

The absorbing-state result is the part of the revision that changes how the
dataset should be read. For binary opinions with positive weights, the
absorbing states are exactly the component-wise consensus states, not only
global consensus. A graph with `C` components therefore has `2^C` absorbing
states. The stable diversity in the original figures is an absorbing mosaic,
not a dynamical coexistence of opinions.

The theory has one exact finite-`n` case. On a complete hypergraph the process reduces to a binomial chain, `N_{t+1} ~ Binomial(n, f(N_t/n))`. At `q=1` the opinion share is a martingale, so the probability of consensus on 1 equals the initial share; for `q≠1` the drift points away from or toward one half according to the sign of `q-1`. The paper states this as Theorem 1, with the proof and the neutral Wright–Fisher corollary.

The revised paper is [`paper/HNL_Voter_Model_v2.pdf`](paper/HNL_Voter_Model_v2.pdf).
The original report is kept unchanged at `HNL Voter Model.pdf`. The full
evidence, including null results and confidence intervals, is in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Quick start

```bash
git clone https://github.com/cxy2696/Hypergraph-Nonlinear-Latency-Voter-Model.git
cd Hypergraph-Nonlinear-Latency-Voter-Model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest                        # 54 checks
python -m hnl.cli reproduce --root .    # about 30 seconds
```

`reproduce` regenerates the result tables, the figures, and the run manifest.
A single simulation:

```bash
python -m hnl.cli simulate --q 1.5 --latency normal --sigma 2 --seed 42
```

Structural diagnostics for the shipped dataset:

```bash
python -m hnl.cli diagnose --csv sentimentdataset.csv --scope full
```

## What is in the repository

```
hnl/                 engine, hypergraph, dataset pipeline, metrics, statistics, CLI
tests/               unit, property and conformance tests for the paper's equations
conformance/         shared numerical fixture used by both repositories
docs/                model specification, errata, data note, results, conformance
results/             raw runs, trajectories, tables, manifest
figures/             generated figures (PNG and PDF)
paper/               revised manuscript source, figures, tables, compiled PDF
model.py             v1-compatible API
random_simulation.py, real_world_simulation.py   v1-compatible entry points
sentimentdataset.csv CC0 dataset (732 posts)
```

The package is small on purpose. The simulation engine is in `hnl/model.py`;
the hypergraph structure is in `hnl/hypergraph.py`; the dataset pipeline is in
`hnl/data.py`; the statistics used in the paper are in `hnl/stats.py`. The
experiment design and the plotting code are separate from the model, so that
the paper's claims can be audited without running the CLI.

## The model

For a hypergraph `(V, E)`, node weights `α_i > 0`, and binary opinions
`X_i ∈ {0,1}`, the influence score of opinion `x` in group `e` is

```
S_x(e) = Σ_{j∈e} α_j · 1[X_j = x]
```

and the group adoption kernel is

```
π_e = S_1(e)^q / (S_0(e)^q + S_1(e)^q)
```

A node averages the kernels of its incident edges (`node_mean`), waits until at
least `L_i` sweeps have passed since its last change, and then samples its new
opinion from a Bernoulli law. Two alternative update rules are implemented for
comparison: `node_sample` draws one incident edge, and `edge_sweep` is the
literal in-place sweep of the original pseudocode. The formal definition,
the propositions, and their proofs are in [`docs/MODEL.md`](docs/MODEL.md).

## Reproducibility

Every run is seeded through `numpy.random.SeedSequence`. The experiment suite
writes a manifest with the git commit, package versions, dataset hash, and the
hash of each generated file. The v1 prototype is preserved at tag
`v1.0-legacy`. The environment used for the committed results is recorded in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

The figure and table pipelines are in `paper/make_figures.py` and
`paper/make_tables.py`. They read `results/` and write `paper/figures/` and
`paper/tables/`; no number in the paper is typed by hand.

## Relation to OpinionLab

[OpinionLab](https://github.com/cxy2696/opinionlab) is the engineering
implementation in the same model family. It generalises the binary model to a
small number of stances, adds ingestion, calibration, Monte Carlo intervals,
reports and a service/UI layer, and supports both a synchronous CSR kernel and
an event-driven kernel.

It is not a reproduction of this paper. Its defaults, RNG streams, feature set,
and scope differ, so its outputs should not be compared numerically with the
numbers reported here. The shared part is the binary conformity kernel and the
node-mean aggregation; the differences are listed in
[`docs/RELATIONSHIP.md`](docs/RELATIONSHIP.md). Both repositories check the same
small numerical fixture in [`conformance/binary_case.json`](conformance/binary_case.json)
to catch a drift in that shared part.

## Documentation

| document | content |
|---|---|
| [`docs/MODEL.md`](docs/MODEL.md) | formal model, update rules, absorbing-state proof, mean-field analysis |
| [`docs/CONFORMANCE.md`](docs/CONFORMANCE.md) | equation → code → test map, shared fixture, known differences |
| [`docs/RELATIONSHIP.md`](docs/RELATIONSHIP.md) | how this repository differs from OpinionLab |
| [`docs/ERRATA.md`](docs/ERRATA.md) | audit of the original prototype and manuscript |
| [`docs/DATA.md`](docs/DATA.md) | dataset provenance, cleaning, valence mapping, limitations |
| [`docs/RESULTS.md`](docs/RESULTS.md) | reanalysis and claim ledger |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | environment, seeds, commands, manifest |
| [`results/TABLES.md`](results/TABLES.md) | generated tables |

## Citation

```bibtex
@software{hnl_voter_model_2026,
  author  = {Cui, Xinyi and Yan, Leyi and Cai, Ruiying and Xiao, Chaowei},
  title   = {Hypergraph--Nonlinear--Latency Voter Model},
  year    = {2026},
  version = {2.3.0},
  url     = {https://github.com/cxy2696/Hypergraph-Nonlinear-Latency-Voter-Model},
  note    = {Research code and reproducibility package}
}
```

## Licence and scope

The code is MIT-licensed; the dataset is CC0. The model is a research object
and does not predict real individuals or events. It does not model
recommendation, media agenda, or offline behaviour. The revised paper is a
reanalysis of the original model, not a claim that the model has been validated
on social media.
