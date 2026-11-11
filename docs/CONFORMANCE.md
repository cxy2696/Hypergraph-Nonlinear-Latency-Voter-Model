# Conformance: from equations to code to tests

A model description is only useful if the code does what the description says.
This note records where each equation lives and which test checks it. It also
defines a small numerical fixture that the companion engineering repository,
OpinionLab, checks against the same expected values.

The fixture is [`conformance/binary_case.json`](../conformance/binary_case.json).
It is not a snapshot of a simulation. The expected values were computed from
the equations in [`MODEL.md`](MODEL.md) for a fixed graph, weight vector and
opinion vector, so they do not depend on a random seed.

## Equation-to-code map

| equation | code | test |
|---|---|---|
| `S_x(e) = Σ α_j 1[X_j=x]` | `hnl/model.py::HNLVoterModel.kernels` | `tests/test_model.py::test_kernel_uses_node_weights` |
| `π_e = S_1^q/(S_0^q+S_1^q)` | `hnl/model.py::_bernoulli_kernel` | `tests/test_model.py::test_kernel_matches_equation_two` |
| `P_i = mean_{e∋i} π_e` | `hnl/model.py::node_probabilities` | `tests/test_model.py::test_node_mean_averages_incident_edges` |
| `t - T_i ≥ L_i` | `hnl/model.py::_step_node_level` | `tests/test_model.py::test_latency_blocks_updates_until_eligible` |
| absorbing states | `hnl/model.py::_step_node_level` | `tests/test_model.py::test_componentwise_consensus_is_absorbing` |
| `D(t) = 1 - max(n0,n1)/n` | `hnl/metrics.py::diversity` | `tests/test_metrics.py::test_diversity_is_minority_share` |
| `O = sd(X̄(t))` | `SimulationResult.oscillation_index` | `tests/test_metrics.py::test_oscillation_index_uses_sample_sd` |
| Kaplan–Meier / RMST | `hnl/metrics.py` | `tests/test_metrics.py::test_kaplan_meier_and_rmst` |
| shared fixture | `hnl/model.py` | `tests/test_conformance.py` |

The shared fixture covers three cases:

1. `binary_q1p5_mean`: five nodes, three overlapping edges, unequal weights,
   `q=1.5`. This is the main case; it exercises the weighted score, the
   nonlinear kernel, and the mean over incident edges.
2. `binary_q1_mean`: one edge, unequal weights, `q=1`. It checks that the
   kernel reduces to the weighted share.
3. `binary_q2_mean`: uniform weights, `q=2`, overlapping edges. It checks the
   majority amplification and the node aggregation.

OpinionLab contains the same JSON file and a test that feeds the same graph,
weights and opinions through its own `edge_distributions` and
`aggregate_node_probabilities` functions. If either implementation changes the
shared part of the model, both tests fail.

## What the fixture does not cover

The shared subset is deliberately narrow: binary opinions, positive weights,
node-mean aggregation, and a single synchronous step. The following are covered
by repository-specific tests, not by the shared fixture:

- `node_sample` and `edge_sweep` updates;
- heterogeneous latency and the event engine's integer clock;
- multi-stance opinions (`K > 2`);
- shocks, interventions, and time-varying topology;
- Monte Carlo intervals and reporting.

The reason is practical. Those features differ between the two repositories, and
a shared fixture would have to encode the differences rather than the shared
mathematics. The differences are listed in [`RELATIONSHIP.md`](RELATIONSHIP.md).

## Reproducing the check

```bash
python -m pytest tests/test_conformance.py
python -m pytest
```

If a deliberate model change invalidates the fixture, update the equations in
`MODEL.md` and the fixture together, and explain the change in the commit
message. Do not regenerate the fixture from the code: that would only test that
the code agrees with itself.
