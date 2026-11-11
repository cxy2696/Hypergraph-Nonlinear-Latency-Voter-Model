# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [2.3.1] — 2026-11-12

### Changed

- Removed unused vector copies `figures/*.pdf` and PNG copies under
  `paper/figures/`. The docs use the root PNGs; the paper uses the vector PDFs.
  Both generated variants remain local and are ignored by git.
- Fixed the stale test count in the `Makefile` help (54 checks).
- `paper/README.md` now mentions Theorem 1 and the 12-page length.
- Added a short source-versus-generated note to the README.

### Unchanged

- No model, result, or paper-content change.

## [2.3.0] — 2026-11-12

### Added

- Theorem 1 in `docs/MODEL.md` and Section IV-B of the revised paper: on the
  complete hypergraph the process reduces to the exact finite-`n` chain
  `N_{t+1} ~ Binomial(n, f(N_t/n))`.
- The theorem includes the q=1 martingale result (consensus probability equals
  the initial share), the sign of the drift away from one half for q≠1, almost
  sure absorption for every q>0, and the neutral Wright–Fisher corollary.
- `tests/test_complete_graph_theory.py` checks the constant kernel, the drift
  sign, and the q=1 absorption probability by simulation.
- The paper now has 12 pages and a theorem/proof block; the test count is 54.

### Changed

- `docs/MODEL.md` renumbered the mean-field section and distinguishes the exact
  complete-graph result from the sparse-graph approximation.
- Version metadata to 2.3.0. Experiments, results, and the run manifest are
  unchanged.

## [2.2.0] — 2026-11-12

### Changed

- Repository documentation is English-only. The bilingual README index and the
  Chinese README were removed so that there is one canonical description of the
  model.
- Added `docs/RELATIONSHIP.md`, which states the division between this research
  repository and the OpinionLab engineering repository.
- Added `docs/CONFORMANCE.md` and `conformance/binary_case.json`, a shared
  numerical fixture with three binary cases. The same fixture is checked by
  OpinionLab, so a drift in the shared kernel fails in both repositories.
- Added `tests/test_conformance.py`; the test suite is now 50 checks.
- Rewrote the README in a plainer voice: what the paper reports, what is in the
  repository, and what the code does not claim.

### Unchanged

- The engine, the result files, and the run manifest are unchanged from 2.1.0.
  The paper PDF was rebuilt only to keep the test count accurate.

## [2.1.0] — 2026-11-12

### Added

- Revised manuscript `paper/HNL_Voter_Model_v2.pdf` with LaTeX source,
  vector figures, generated tables, and build instructions.
  It fixes the specification, proves the component-wise absorbing-state
  result, and reports the paired 30-replicate experiments.
- Cross-links to the revised edition from the README files and `docs/RESULTS.md`.

### Changed

- Version metadata to 2.1.0. The engine, tests, results, and manifest are
  unchanged from 2.0.0; the manuscript is a documentation artifact built from
  the same committed result files.

## [2.0.0] — 2026-11-12

### Added

- `hnl` package: validated `ModelConfig`, `Hypergraph`, `HNLVoterModel`,
  `SimulationResult`, metrics, statistics, experiments, plotting and CLI.
- Three update rules: `node_mean` (default, order independent), `node_sample`,
  and the manuscript's literal `edge_sweep`.
- Formal specification in `docs/MODEL.md`, including the absorbing-state
  proposition (component-wise consensus), the mean-field transition at `q=1`,
  and the latency/oscillation argument.
- Audit in `docs/ERRATA.md` with the evidence for each v1 defect and each
  unsupported claim.
- Data documentation, valence lexicon, dataset diagnostics and a CC0
  provenance record in `docs/DATA.md`.
- Reproducible experiments with common random numbers, bootstrap confidence
  intervals, paired permutation tests, Holm correction, Kaplan–Meier medians
  and RMST (`docs/RESULTS.md`, `results/`, `figures/`).
- 47 tests, GitHub Actions CI, MIT licence, `CITATION.cff`, `Makefile`,
  `pyproject.toml`.
- Run manifest with git commit, environment versions, design parameters,
  dataset SHA-256 and artifact hashes.
- Supplementary long-horizon edge-sweep check
  (`scripts/edge_sweep_longrun.py`).

### Fixed

- Sentiment labels are whitespace-normalised; the v1 loader mapped every post
  to 1.
- `real_world_simulation.py` no longer discards data-derived opinions; both
  data and random initialisation are available and reported.
- Hyperedges are duplicate-free and deduplicated; user identifiers are
  stripped (670 users instead of 685 phantom identities).
- Isolated nodes are handled explicitly (`scope="active"` default for the
  dataset analysis).
- The oscillation metric is `sd(X̄(t))`, not `mean(X̄(t))`.
- Consensus time is recorded with a censoring flag.
- RNG is a single seeded `numpy.random.Generator` per run; no global state.
- CLI flags documented in the README now exist and are tested.
- Latency sampling uses rounding and an explicit floor.
- Weight normalisation handles zero engagement.

### Changed

- **Breaking:** the default update rule is `node_mean`, not the order-dependent
  last-edge-wins of v1. Use `--rule edge_sweep` for the literal algorithm.
- **Breaking:** results and figures are regenerated by `python -m hnl.cli
  reproduce`; the v1 scripts remain as compatibility shims.
- The README no longer repeats the manuscript's unqualified claims; it links to
  the re-analysis and states the limitations.

## [1.0.0] — 2024

- Original prototype: `model.py`, `random_simulation.py`,
  `real_world_simulation.py`, the dataset and the manuscript PDF. Preserved at
  tag `v1.0-legacy`.
