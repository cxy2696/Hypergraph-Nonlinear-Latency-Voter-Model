# Errata and audit of the v1 prototype

This document records what was wrong, ambiguous or unsupported in the v1
prototype and in the manuscript, and how v2 resolves each item. The original
code and PDF are **not** modified: they are preserved at tag
[`v1.0-legacy`](https://github.com/cxy2696/Hypergraph-Nonlinear-Latency-Voter-Model/tree/v1.0-legacy)
so that the v1 behaviour can still be inspected and reproduced.

The audit was carried out by (i) reading the manuscript, (ii) running the v1
code on the shipped dataset and on random hypergraphs, and (iii) comparing the
results with the claims. Findings are separated into code defects, specification
ambiguities, unsupported claims, and data/bibliography issues.

## A. Code defects

### A1. Every sentiment was mapped to 1

`model.py:110–120` (v1) compared `s.lower()` with a list of 36 lowercase
labels, but the CSV stores labels with trailing spaces (`' Negative  '`,
`' Positive '`). No label matched, so `map_sentiment` returned 1 for every
post.

```text
$ git checkout v1.0-legacy
$ python -c "from model import load_data; print(load_data('sentimentdataset.csv')[3].mean())"
1.0
```

**Impact.** The data-derived initial condition was degenerate. This was masked
by A2. **Fix.** Labels are whitespace-normalised and case-folded before a
documented valence mapping ([`hnl/lexicon.py`](../hnl/lexicon.py)); the mapped
counts are reported in [`DATA.md`](DATA.md).

### A2. The "real-world" run discarded the data-derived opinions

`real_world_simulation.py:7` (v1) replaced the loaded opinions with a fresh
random draw:

```python
model.opinions = np.random.choice([0, 1], size=len(nodes))
```

**Impact.** The dataset entered only through the graph and the weights; the
sentiment labels never influenced the simulation. The manuscript's claim that
the model was applied to sentiment data is therefore not supported by the
released code. **Fix.** `init="data"` and `init="random"` are both available;
v2 reports both and labels the dataset analysis as a *structural application*,
not a sentiment validation ([`RESULTS.md`](RESULTS.md)).

### A3. Duplicate node ids inside hyperedges

`model.py:97–104` (v1) appended a user to a hashtag group once per post. A user
who used the same hashtag in two posts appears twice in that hyperedge. In the
shipped data, 53 of the 202 v1 hyperedges contained duplicate ids.

**Impact.** The duplicated weight is counted twice in $S_0$ and $S_1$, so the
effective influence of those users is inflated. **Fix.** Edges are stored as
duplicate-free sorted sets ([`hnl/hypergraph.py`](../hnl/hypergraph.py)).

### A4. Duplicate hyperedges and phantom users

Two distinct hashtags induce the *same* user set, and 15 users differ only by
trailing whitespace (`"User123 "` vs `"User123"`). v1 kept both duplicates and
685 "users"; v2 has 670 users and 171 unique hyperedges.

**Impact.** Duplicate edges double the influence of a group; phantom users
create spurious isolated nodes. **Fix.** Users are stripped; identical edge
sets are merged; the counts are reported in the run manifest.

### A5. Isolated nodes were frozen but included in the consensus denominator

v1 built the hypergraph on all 685 users. 185 of them belonged to no hyperedge
of size $\ge 2$ and therefore never updated. The stopping rule nevertheless
used the full population.

**Impact.** A population-level 80 % majority may be unreachable by
construction, and the reported "diversity floor" mixes dynamics with frozen
nodes. **Fix.** `scope="active"` restricts the analysis to nodes with at least
one hyperedge; the full graph is still reported for transparency; a structural
feasibility check is part of the diagnostics.

### A6. The "oscillation" series was the mean opinion

`model.py:56` stored `oscillation = np.mean(self.opinions)`, and
`plot_metrics` plotted it under the title "Oscillatory Behavior (Average
Opinion)". The manuscript defines $O=\mathrm{std}(\bar X(t))$ (Eq. 7), which is
a different quantity.

**Impact.** The paper's oscillation claim was assessed visually from a
non-stationary mean, not from the stated metric. **Fix.** $O$ is computed as
the sample standard deviation of $\bar X(t)$; the mean is reported separately;
damping is measured by the late/early amplitude ratio.

### A7. No consensus time

v1 stopped when the majority reached $r$ but did not record $T_c$; the number
of history rows was used as a proxy. **Fix.** $T_c$ is recorded with an
explicit censoring flag; summaries use Kaplan–Meier medians and RMST.

### A8. The update is order dependent and differs from Algorithm 1

`model.py:28–44` (v1) computes $p_1$ once per hyperedge from the
start-of-sweep opinions and writes into a separate `new_opinions` buffer, so a
node in several hyperedges is overwritten by the last edge in the iteration
order. Algorithm 1 in the manuscript updates **in place**, so later edges see
earlier updates. The two are not the same process; the manuscript's pseudocode
is a third variant.

**Impact.** Results depend on the order in which `hyperedges` happens to be
stored; the v1 README already acknowledged this. **Fix.** The default
`node_mean` aggregates the kernels of all incident edges and is provably
invariant to edge order (tested). `node_sample` and the literal `edge_sweep`
are provided; the sensitivity is quantified in
[`RESULTS.md`](RESULTS.md) §6.

### A9. No seed control; documented CLI flags did not exist

v1 mixed Python's `random` with NumPy's global state. `random_simulation.py`
seeded both at import time; `real_world_simulation.py` did not. The README
advertised `python random_simulation.py --q=2.0 ...` and
`--file_path=data.json`, but the scripts contained no argument parser.

**Impact.** Runs were not reproducible and the documented commands failed.
**Fix.** A single `numpy.random.Generator` per run, explicit seeds, an argparse
CLI, and a run manifest with git commit, versions, parameters and file hashes.

### A10. Latency truncation and undocumented parameterisation

v1 used `np.maximum(1, np.random.normal(mu, sigma, n)).astype(int)`, which
truncates toward zero (a mean bias of about $-0.5$ sweep) and clips the lower
tail to 1. The exponential branch used `lambda_param` as a rate but documented
it only implicitly.

**Fix.** v2 rounds to the nearest integer, clips at a configurable floor, and
documents both parameterisations. All distributions are sampled by inverse CDF
from common uniforms for paired comparisons.

### A11. Weight normalisation edge cases

`model.py:106–108` (v1) filled missing engagement with 0.1 *before* dividing by
the maximum, so the intended floor became $0.1/\max$; if every engagement were
zero, the division would fail.

**Fix.** Weights are normalised first and floored afterwards; zero or missing
engagement is handled explicitly. Because $\pi_e$ is invariant to a global
rescaling of $\alpha$, only relative weights matter (stated in
[`MODEL.md`](MODEL.md) §4.2).

## B. Specification ambiguities

| id | ambiguity | v1 behaviour | v2 resolution |
|---|---|---|---|
| B1 | Eq. (3) writes `argmax`; Algorithm 1 samples | samples | sampling is the definition; `argmax` is not implemented |
| B2 | a node in several edges | last edge wins | `node_mean` (default), `node_sample`, `edge_sweep` |
| B3 | when is the latency clock reset? | only on a change | same; stated in Eq. (6) |
| B4 | does $r$ apply to the whole population or the active graph? | whole population | both reported; active graph is the default, with structural diagnostics |
| B5 | $0^0$ and $S_0=S_1=0$ | `p1 = 0.5` if denominator 0 | $\pi=1/2$; tested |
| B6 | self-influence | included | included; documented as inertia |

## C. Claims not supported by the v2 experiments

These are not code bugs; they are statements in the manuscript that the
controlled experiments do not reproduce. The numbers are in
[`RESULTS.md`](RESULTS.md).

### C1. "High latency variance prolongs diversity"

Supported **only at low nonlinearity**. At $q=0.5$, final diversity is higher
under every non-degenerate latency distribution than under fixed latency
($\Delta D=+0.024$ to $+0.054$, Holm-adjusted $p\le0.048$). At $q=1.5$ all
contrasts are null after Holm ($p=1.0$); at $q=2.5$ the exponential
distribution *reduces* diversity ($\Delta D=-0.033$, $p=0.036$). The
generalisation in the abstract is therefore too strong.

### C2. "Moderate $q$ and high latency variance produce sustained oscillations"

At $q=1.5$, the high-variance ($\sigma=4$) condition does not increase the
oscillation index relative to fixed latency ($\Delta O=+0.007$, 95 % CI
$-0.005$ to $+0.020$, $p=0.885$). $O$ is highest at $q=2.5$, not at $q=1.5$.
The late/early amplitude ratio at $q=2.5$ is 0.11–0.43, i.e. strong damping.
Moreover, the process is a finite absorbing chain (Proposition 1): sustained
oscillation is impossible asymptotically. What the data show is a transient
swing of larger amplitude at high $q$, which damps as consensus is reached.

### C3. "Higher $q$ produces larger cascades"

The opposite holds for the paper's own cascade metric: the largest
single-sweep change is smaller at high $q$ (e.g. fixed latency: 0.428 at
$q=0.5$ vs 0.348 at $q=2.5$; paired $\Delta=-0.080$, $p<0.001$). Low $q$
keeps the system near balance, where many nodes can flip in one sweep; high
$q$ reaches consensus early and then has nothing left to flip.

### C4. "Validation against real-world social-media data"

The shipped file has 732 posts and 670 user identifiers, 617 of which occur
once; it contains no per-user opinion trajectory. It therefore cannot validate
a dynamical model. On its hashtag hypergraph the model absorbs into a
component-wise mosaic; the 80 % threshold was reached in **0 of 900** dataset
runs, and the largest final majority was 0.754. The honest description is a
*structural application*: the dataset supplies a real group structure, not a
ground truth.

### C5. "Robustness" from repeated runs without uncertainty

v1 ran 10 replicates and interpreted run-to-run variation qualitatively. v2
uses 30 replicates per cell, common random numbers for paired contrasts,
bootstrap confidence intervals, paired permutation tests with Holm correction,
and Kaplan–Meier/RMST summaries for censored consensus times.

### C6. "Persistent diversity" is an absorbing mosaic

$D(t)$ stabilising does not mean that opinions coexist dynamically. Components
freeze internally (97.2 % of active-graph components are unanimous at the end
of a run), and the global minority is the mass of components that happened to
choose the other colour. The stable diversity is therefore an **absorbing
mosaic**, not a metastable coexistence (Proposition 1 and Corollary 2 in
[`MODEL.md`](MODEL.md)).

## D. Data and bibliography

### D1. Dataset provenance

The Kaggle page for *Social Media Sentiments Analysis Dataset* (Kashish Parmar)
declares **CC0: Public Domain**; the shipped file is 170 776 bytes, matching
the Kaggle archive, so the repository contains the full dataset. Its 732 rows
and pseudonymous handles (`User123`, `CommuterX`) do not support
platform-scale or representative claims. See [`DATA.md`](DATA.md).

### D2. Holley–Liggett reference

The manuscript cites *Annals of Probability* **3**(6), 1975. The article is in
**3**(4), pp. 643–663, August 1975 (DOI
[10.1214/aop/1176996306](https://doi.org/10.1214/aop/1176996306)), confirmed
via Crossref.

### D3. Derivative claim for $q<1$

The manuscript says the adoption-probability gradient is "flatter" for $q<1$.
Away from the boundary this is right, but as $S_1\to0$,

$$
\frac{\partial \pi_e}{\partial S_1}
=\frac{q\,S_0^q S_1^{q-1}}{(S_0^q+S_1^q)^2}
\sim \frac{q}{S_0^{q}}\,S_1^{q-1}\to\infty
\quad (q<1),
$$

so the gradient *diverges* at the minority boundary. The correct statement is
that the kernel is concave and heavily favours a tiny minority, not that its
slope is uniformly smaller.

### D4. Typesetting of references [5]–[7]

In the PDF, references [5], [6] and [7] are merged into a single paragraph
([5] [6] [7] ...), so the bibliography cannot be parsed. This is a PDF
production issue, not a content issue; v2 does not modify the PDF.

## E. What v2 does **not** change

* The model family is unchanged: hypergraph groups, the $q$-exponent kernel,
  node weights and heterogeneous latency.
* The manuscript PDF is unchanged and remains the citable artifact.
* The v1 scripts are preserved at tag `v1.0-legacy`.
* No claim is made that the v2 experiments are a preregistered or exhaustive
  study; they are a controlled re-analysis of the same model family.

## F. How to reproduce the audit

```bash
# v1 behaviour
git checkout v1.0-legacy
python -c "from model import load_data; print(load_data('sentimentdataset.csv')[3].mean())"

# v2 behaviour, tests and full experiments
git checkout main
python -m pytest
python -m hnl.cli reproduce --root .
```

The full experiment takes about 30 s on an Apple-silicon laptop; the manifest
in `results/manifest.json` records the commit, versions, seeds and hashes.
