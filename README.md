# Hypergraph-Nonlinear-Latency Voter Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository is the research prototype for the **Hybrid Hypergraph-Nonlinear-Latency Voter Model (HNL Voter Model)** — a way to simulate how opinions spread through groups.

Most opinion models treat people as talking in pairs and reacting instantly. This one adds three things that matter in real life:
- people influence each other in **groups** (a chat, a forum, a meeting) rather than only one-to-one;
- adoption is **non-linear** — conformity strength `q` controls how strongly a local majority pulls people in;
- people react at **different speeds**, so a message can keep spreading long after it was posted.

The prototype accompanies the paper below. For new work, use the production successor, [OpinionLab](https://github.com/cxy2696/opinionlab), which is reproducible, tested and ships a browser sandbox.
 
> The Voter Model has long served as a fundamental framework for analyzing opinion dynamics, however, it often oversimplifies interactions as pairwise and assumes immediate responses. This study introduces the Hybrid Hypergraph-Nonlinear-Latency Voter Model (HNL Voter Model), which integrates group-based hypergraph structures, nonlinear adoption probabilities, and latency effects to better reflect the complexity of real-world opinion propagation. By simulating sentiment data from social media platforms, the model explores how network structures, nonlinear influences, and temporal delays collectively shape opinion diversity, cascading effects, and oscillatory behaviors.

Key features:
- Simulates opinion diversity, cascading effects, and oscillations.
- Supports random hypergraphs or real-world data (from Kaggle sentiment datasets).
- Latency distributions: normal (default) or exponential.

## What you can use it for
The same model covers many group-opinion questions:
- **Crisis communication:** when should we debunk a rumor, and how much does timing change the outcome?
- **Community discussion:** is a group converging on a decision, or being pushed into a silent majority?
- **Organizational change:** how do announcement cadence and reaction delays affect sentiment?
- **Product launch / brand:** is early buzz the same as durable word of mouth?
- **Teaching and research:** demonstrate how conformity and latency change consensus, or reproduce the classical voter-model limit.

The successor project makes these concrete with 15 bilingual scenario templates, runnable examples and a browser demo.

## Production successor
This repository is the research prototype accompanying the paper. For a reproducible,
service-ready implementation of the same model family, see
**[OpinionLab](https://github.com/cxy2696/opinionlab)**. OpinionLab adds:

- node-centric, order-independent updates (the prototype's hyperedge loop is order-dependent),
- deterministic RNG streams, run manifests and golden baselines,
- CSR-vectorised and event-driven engines with Monte-Carlo confidence intervals,
- a CLI/REST service, compliance guardrails and offline HTML reports.

The original scripts here remain unchanged so the paper's results can still be reproduced.

## Try the demo
Open the live sandbox: **https://cxy2696.github.io/opinionlab/**

The successor project also ships a local browser sandbox:

[![OpinionLab demo](https://raw.githubusercontent.com/cxy2696/opinionlab/main/docs/demo.png)](https://github.com/cxy2696/opinionlab/blob/main/docs/DEMO.md)

- Choose one of the 15 scenario templates (public communication, community, organizational change, product launch, research) and tune conformity `q`, latency and replicas.
- Tune `q`, latency and replicas with inline explanations, hover tips and one-click presets.
- Read three linked charts: diversity with a run-to-run band, opinion shares and cascade spikes.
- Get a one-sentence verdict, a plain-language “how to read” note and a **decision brief**: what the run can inform, the simulated action window (take-off, peak, majority, intervention) and concrete follow-up checks.
- Replay the evolution step by step.
- Download a single-file offline report with assumptions, limitations and disclaimer.

Run it locally:

```bash
git clone https://github.com/cxy2696/opinionlab.git
cd opinionlab
make demo          # http://127.0.0.1:8000
```

See [docs/DEMO.md](https://github.com/cxy2696/opinionlab/blob/main/docs/DEMO.md) for the API and limits.

## Concrete examples
The successor repository has runnable examples and a case guide:

- [Debunk timing](https://github.com/cxy2696/opinionlab/blob/main/examples/rumor_timing.py):
  `uv run python examples/rumor_timing.py --repeats 8`
- [Community conformity](https://github.com/cxy2696/opinionlab/blob/main/examples/community_consensus.py):
  `uv run python examples/community_consensus.py --repeats 6`
- [Use cases and reading guide](https://github.com/cxy2696/opinionlab/blob/main/docs/USE_CASES.md)

## Scope and limitations
This repository is a paper prototype: results depend on hyperedge order and use global random state.
It is kept for reference and reproduction. For anything beyond that — reproducible runs,
confidence intervals, API/reporting, compliance guardrails — use OpinionLab.
The model does not predict real individuals or events, and it does not model recommender
systems, media agenda or offline behaviour.

## Paper
Full paper: [Hypergraph-Nonlinear-Latency Voter Model.pdf](https://cxy2696.github.io/Hypergraph-Nonlinear-Latency-Voter-Model/HNL%20Voter%20Model.pdf) (From the paper: Initializes random opinions, selects pairs, copies opinions, and checks for consensus.)

## Installation
### Mac
- Install Python: `brew install python@3.12`
- Alias: `echo 'alias python=python3' >> ~/.zshrc && source ~/.zshrc`
- Venv: `python -m venv venv && source venv/bin/activate`

### Windows
- Install Python from python.org (add to PATH).
- Venv: `python -m venv venv && venv\Scripts\activate.bat`

### Linux
- Install: `sudo apt install python3.12 python3.12-venv` (Ubuntu).
- Venv: `python -m venv venv && source venv/bin/activate`


```bash
git clone https://github.com/cxy2696/Hypergraph-Nonlinear-Latency-Voter-Model.git
cd Hypergraph-Nonlinear-Latency-Voter-Model
python -m venv venv
source venv/bin/activate  # Mac/Linux: source venv/bin/activate | Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
### Random Simulation
Generate a random hypergraph and run the model:
```bash
python random_simulation.py
```
- Customize: e.g., adjusting `latency_dist`
```bash
python random_simulation.py --q=2.0 --latency_dist=exponential --mu=5
```

### Real-World Simulation
Requires `sentimentdataset.csv` (included, from Kaggle sentiment analysis dataset):
```bash
python real_world_simulation.py
```
- Supports CSV, Parquet, JSON formats.
- Customize: e.g., having a dataset named `data.json`
```bash
python real_world_simulation.py --file_path=data.json --format_type=json --q=1.0
```

Outputs include plots for diversity, cascade size, and oscillatory behavior.

### Model Parameters for adjusting
- `q`: Nonlinearity parameter (>1 accelerates consensus, <1 prolongs diversity).
- `latency_dist`: 'normal' (mu, sigma) or 'exponential' (lambda_param).
- `consensus_threshold`: Fraction for consensus (default 0.8).
- `max_steps`: Maximum simulation steps (default 100).



