# How this repository relates to OpinionLab

There are two repositories in the same model family. They serve different
purposes, and it is easy to mistake one for a newer version of the other. This
note states the division once, so the READMEs can stay short.

## The short version

- **This repository** is the research artifact. It contains the revised paper,
  the binary model, the proofs, the experiment suite, and the result tables.
  Its job is to make the paper's claims checkable.
- **[OpinionLab](https://github.com/cxy2696/opinionlab)** is the engineering
  implementation. It contains a multi-stance engine, ingestion and
  calibration, Monte Carlo intervals, reports, a service API, and a browser
  demo. Its job is to run scenario studies and produce artifacts.

OpinionLab is not a reproduction of the paper and not a validation of it. It
implements a broader model with different defaults, a different random-number
layout, and additional features. Numbers from the two repositories should not
be compared as if they came from the same experiment.

## What is shared

The binary conformity kernel is the same:

```
S_x(e) = Σ_{j∈e} α_j 1[X_j = x]
π_e    = S_1(e)^q / (S_0(e)^q + S_1(e)^q)
P_i    = mean_{e∋i} π_e
```

Both repositories check a small fixture,
[`conformance/binary_case.json`](../conformance/binary_case.json), against
these equations. OpinionLab stores the same file and runs the same cases
through its own functions. That is the only numerical contract between the two
codebases.

## What differs

| dimension | this repository | OpinionLab |
|---|---|---|
| purpose | paper and reanalysis | scenario engine and service |
| opinions | binary | 2–6 stances |
| update rules | `node_mean`, `node_sample`, `edge_sweep` | `mean`, `max`, `size_weighted` aggregation; stochastic, argmax, threshold decisions |
| latency | change-triggered clock | same clock in CSR and integer event kernels; continuous latency in CSR only |
| weights | `α_i > 0`, fixed | uniform, engagement, centrality, custom; zero weights allowed with uniform fallback |
| shocks/interventions | none | one-shot shocks, interventions, time-varying topology |
| uncertainty | paired bootstrap, permutation, Kaplan–Meier | Welford Monte Carlo intervals |
| outputs | paper tables and figures | run artifacts, reports, API responses, browser demo |
| tests | model equations and the paper's experiments | engine, API, service, UI, and deployment |

The shared mathematical subset is the binary kernel with positive weights and
node-mean aggregation. The differences are features, not bugs: OpinionLab is
designed for applied scenario work, while this repository is designed to make a
specific set of scientific claims auditable.

## Which one to use

- If you want to read, cite, or extend the paper, use this repository.
- If you want to run a scenario study, use OpinionLab.
- If you are changing the shared kernel, run the conformance test in both
  repositories.
- If you are changing a feature that only exists in one repository, the other
  repository should not need to change.

## Versioning

The two repositories are versioned independently. This repository is at 2.3.1
(paper edition); OpinionLab is at 1.3.1 (engine edition). A version number in
one repository says nothing about the other.

## Citation

Cite the paper repository for the model and the reanalysis. Cite OpinionLab
only for the engineering implementation or for scenario results produced with
it. The citation files in each repository are separate for that reason.
