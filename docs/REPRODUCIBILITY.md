# Reproducibility protocol

This repository is built so that every reported number can be regenerated from
a clean checkout, a fixed seed and the shipped dataset. The protocol below is
the one used to produce [`RESULTS.md`](RESULTS.md) and the committed `results/`
and `figures/` directories.

## 1. One-command reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                       # 47 tests
python -m hnl.cli reproduce --root .   # ≈30 s on an Apple-silicon laptop
```

`reproduce` runs the full factorial design, the dataset designs, the update-rule
sensitivity, the bootstrap/permutation analyses and all figures, then writes a
manifest. A quick smoke run (5 replicates, 60 sweeps) is available:

```bash
python -m hnl.cli reproduce --root . --quick
```

## 2. Environment actually used

| component | version |
|---|---|
| OS | macOS 26 (arm64) |
| Python | 3.13 |
| NumPy | 2.5.4 |
| pandas | 3.1.1 |
| matplotlib | 3.11.2 |
| pytest | 9.2.0 |

Minimum supported versions are Python ≥ 3.10, NumPy ≥ 1.26, pandas ≥ 2.2 and
matplotlib ≥ 3.8 (`pyproject.toml`). The environment is recorded again in
`results/manifest.json` at run time.

## 3. Data

The dataset is committed as `sentimentdataset.csv` (CC0, 170 776 bytes,
SHA-256 `725c6f7bc928c2bf741094f37078024b305982523fe6d0b8cb20a6cf5275c63b`).
No download step is required. See [`DATA.md`](DATA.md) for provenance.

## 4. Designs and seeds

| experiment | file | design | seed |
|---|---|---|---|
| random hypergraphs | `results/random_*` | $n=100$, $m=200$, sizes 3–6, $q\in\{0.5,1,1.5,2.5\}$, 5 latency specs, 30 replicates, $T=100$, $r=0.8$ | 20260101 |
| dataset, active | `results/dataset_active_*` | 491 users, 81 components, random init, 30 replicates | 20260101 + 101 |
| dataset, full | `results/dataset_full_*` | 670 users, random init, 15 replicates, 3 latency specs | 20260101 + 202 |
| dataset, largest component | `results/dataset_lcc_*` | 140 users, random init, 15 replicates | 20260101 + 303 |
| dataset, data init | `results/dataset_datainit_*` | active graph, valence init, 15 replicates, 2 specs | 20260101 + 404 |
| update-rule sensitivity | `results/random_rule_sensitivity_*` | 8 replicates, $\sigma=2$, 3 rules × 4 $q$ | 20260101 + 7 |
| long-horizon check | `results/edge_sweep_longrun.csv` | $n=30$, $m=60$, 5 seeds, up to 20 000 sweeps | 0–4 |

Latency specifications: fixed $L=5$; normal $\mu=5$, $\sigma\in\{1,2,4\}$;
exponential with mean 5. Latencies are sampled by inverse CDF from common
uniforms, so comparisons across distributions are paired.

**Random-number discipline.** Every replicate receives a child
`numpy.random.SeedSequence`; the hypergraph, initial opinions and uniforms are
drawn from that child once and shared by every $(q,\text{latency})$ cell. The
per-run Bernoulli stream is derived from
`SeedSequence(entropy=base_seed, spawn_key=(replicate, spec, q))`. No global
`random` or `np.random` state is used anywhere in `hnl/`.

## 5. Outputs

| path | content |
|---|---|
| `results/random_runs.csv` | one row per random-graph run (scalars) |
| `results/random_trajectories.npz` | padded trajectories, key `div|spec|q` / `mean|spec|q` |
| `results/dataset_*_runs.csv` | dataset runs |
| `results/tables/*.csv` | cell summaries and paired contrasts |
| `results/TABLES.md` | all tables rendered as Markdown |
| `results/manifest.json` | commit, environment, designs, dataset hashes, artifact hashes |
| `results/edge_sweep_longrun.csv` | supplementary long-horizon rule check |
| `figures/fig1…fig7.{png,pdf}` | figures generated from the tables |

`reproduce` overwrites these files deterministically given the seeds and the
same package versions.

## 6. Manifest

`results/manifest.json` contains:

* UTC timestamp and the exact command line;
* git commit, branch and a `dirty` flag that ignores `results/` and `figures/`
  (generated artifacts are expected to change);
* Python, NumPy, pandas and matplotlib versions and the platform;
* every design as a JSON object (grid, seeds, thresholds, horizons);
* the dataset path, SHA-256, row and user counts, mapping and scope;
* SHA-256 and byte size of every generated artifact.

A reported number can therefore be traced to a commit, a parameter set and a
data hash.

## 7. Determinism caveats

* Results are bit-reproducible for a fixed Python/NumPy/pandas/matplotlib stack
  on the same machine. Different BLAS implementations or NumPy versions can
  change the last bits of floating-point reductions; aggregate conclusions at
  the reported precision are stable.
* `--quick` uses different seeds only in the sense of fewer replicates; the
  base seeds are unchanged.
* Runtime depends on CPU; the full suite took 24–33 s during development.
* Figures use the `Agg` backend and never call `plt.show()`.

## 8. Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

1. install Python 3.12 and `requirements.txt` + `pytest`;
2. run `python -m pytest`;
3. run `python -m hnl.cli reproduce --root . --quick --no-figures`;
4. upload the quick-run tables as a build artifact.

The full 30-replicate study is not run in CI to keep the workflow fast; the
committed `results/` directory is the reference output.

## 9. Versioning

* `v1.0-legacy` — the original prototype, unchanged.
* `v2.0.0` — this implementation, tests, docs, results and figures.

To inspect the original behaviour:

```bash
git checkout v1.0-legacy
python -c "from model import load_data; print(load_data('sentimentdataset.csv')[3].mean())"
```

## 10. Checklist

- [x] Code, tests and data in one repository.
- [x] Fixed seeds and explicit RNG streams.
- [x] One command regenerates every table and figure.
- [x] Environment and commit recorded in a manifest.
- [x] Dataset licence and SHA-256 recorded.
- [x] Original prototype preserved under a tag.
- [x] Every claim in `RESULTS.md` is backed by a committed table or CSV.
- [x] Null results and limitations reported, not hidden.
