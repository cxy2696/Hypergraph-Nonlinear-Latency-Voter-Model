# Hypergraph-Nonlinear-Latency Voter Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
Open-source implementation of the **Hybrid Hypergraph-Nonlinear-Latency Voter Model (HNL Voter Model)** This model extends the traditional Voter Model to incorporate group-based hypergraph interactions, nonlinear adoption probabilities (controlled by parameter `q`), and latency effects (normal or exponential distributions) for more realistic simulation of opinion dynamics on social networks.
 
> The Voter Model has long served as a fundamental framework for analyzing opinion dynamics, however, it often oversimplifies interactions as pairwise and assumes immediate responses. This study introduces the Hybrid Hypergraph-Nonlinear-Latency Voter Model (HNL Voter Model), which integrates group-based hypergraph structures, nonlinear adoption probabilities, and latency effects to better reflect the complexity of real-world opinion propagation. By simulating sentiment data from social media platforms, the model explores how network structures, nonlinear influences, and temporal delays collectively shape opinion diversity, cascading effects, and oscillatory behaviors.

Key features:
- Simulates opinion diversity, cascading effects, and oscillations.
- Supports random hypergraphs or real-world data (from Kaggle sentiment datasets).
- Latency distributions: normal (default) or exponential.

## Paper
Full paper: [HNL Voter Model.pdf](HNL%20Voter%20Model.pdf) (From the paper: Initializes random opinions, selects pairs, copies opinions, and checks for consensus.)

## Installation
```bash
git clone https://github.com/cxy2696/Hypergraph-Nonlinear-Latency-Voter-Model.git
cd Hypergraph-Nonlinear-Latency-Voter-Model
pip install -r requirements.txt
```

## Usage
### Random Simulation
Generate a random hypergraph and run the model:
```bash
python random_simulation.py
```
- Customize: e.g., `python random_simulation.py --q=2.0 --latency_dist=exponential --mu=5`

### Real-World Simulation
Requires `sentimentdataset.csv` (included, from Kaggle sentiment analysis dataset):
```bash
python real_world_simulation.py
```
- Supports CSV, Parquet, JSON formats.
- Customize: e.g., `python real_world_simulation.py --file_path=mydata.json --format_type=json --q=1.0`

Outputs include plots for diversity, cascade size, and oscillatory behavior.

## Model Parameters
- `q`: Nonlinearity parameter (>1 accelerates consensus, <1 prolongs diversity).
- `latency_dist`: 'normal' (mu, sigma) or 'exponential' (lambda_param).
- `consensus_threshold`: Fraction for consensus (default 0.8).
- `max_steps`: Maximum simulation steps (default 100).

## Citation
If you use this code, please cite the original paper:
@article{hnl_voter_model,
title = {Hypergraph-Nonlinear-Latency Voter Model},
author = {Xinyi Cui and Leyi Yan and Ruiying Cai and Chaowei Xiao},
year = {2024}
}

## License
MIT



