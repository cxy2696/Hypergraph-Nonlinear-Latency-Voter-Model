# Model specification (v2)

This document is the normative specification of the implementation in
[`hnl/model.py`](../hnl/model.py). It fixes the ambiguities in the original
manuscript and is the reference for the tests in [`tests/`](../tests). For the
list of changes relative to the v1 prototype see
[`ERRATA.md`](ERRATA.md).

## 1. Scope

The model describes binary opinion change in a population whose interactions
are group interactions (hyperedges), with heterogeneous susceptibility to the
local majority (`q`) and heterogeneous reaction delays (`L_i`). It is a
stochastic, finite-state, discrete-time process. It is **not** a model of
belief formation, media effects, recommendation, or network evolution.

## 2. Notation

| symbol | meaning |
|---|---|
| $V=\{1,\dots,n\}$ | nodes (users) |
| $E=\{e_1,\dots,e_m\}$ | hyperedges, $e_k\subseteq V$, $\lvert e_k\rvert\ge 2$ |
| $E_i=\{e\in E: i\in e\}$ | hyperedges incident to $i$; $d_i=\lvert E_i\rvert$ |
| $\alpha_i>0$ | influence weight of node $i$ |
| $X_i(t)\in\{0,1\}$ | opinion of node $i$ at sweep $t$ |
| $q\ge 0$ | nonlinearity exponent |
| $L_i\in\mathbb{N}$ | latency (refractory period) of node $i$ |
| $T_i(t)$ | last sweep at which $i$ changed opinion; $T_i(0)=0$ |
| $r\in(0,1]$ | consensus threshold |
| $T$ | maximum number of sweeps |

## 3. Static structure

### 3.1 Hyperedges

Edges are stored as sorted, duplicate-free node sets. Duplicate edges are
merged; edges of size $<2$ are dropped. Node weights are attached to nodes,
not to node–edge incidences.

### 3.2 The 2-section graph and components

The **2-section** $G_2=(V,F)$ has $\{i,j\}\in F$ iff some hyperedge contains
both $i$ and $j$. Its connected components are the components of the
hypergraph. Hyperedges never cross components, so components cannot influence
one another. Nodes with $d_i=0$ are isolated components of size one.

## 4. Dynamics

### 4.1 Influence scores

For $x\in\{0,1\}$ and $e\in E$,

$$
S_x(e)=\sum_{j\in e}\alpha_j\,\mathbf{1}[X_j(t)=x].
\tag{1}
$$

### 4.2 Adoption kernel

The probability that a node in $e$ adopts opinion 1, conditional on the group
being the only source of influence, is

$$
\pi_e(t)=\frac{S_1(e)^q}{S_0(e)^q+S_1(e)^q},
\tag{2}
$$

with the convention $\pi_e=\tfrac12$ when $S_0(e)=S_1(e)=0$. Because
$\alpha_j>0$, the denominator is zero only for an empty edge, which is
excluded; the convention is nevertheless implemented and tested.

For $q=1$, $\pi_e=S_1/(S_0+S_1)$ is the weight-fraction of 1s. For $q>1$ the
kernel is convex and amplifies the local majority; for $q<1$ it is concave and
amplifies the local minority. $\pi_e$ is invariant to a global rescaling of
$\alpha$.

### 4.3 From edge kernels to a node probability

A node may belong to several hyperedges with conflicting local majorities. The
original Algorithm 1 resolved this by iterating edges and overwriting the
node's opinion, which makes the dynamics depend on edge order. v2 defines
order-independent alternatives and keeps the literal sweep for comparison.

**`node_mean` (default).**

$$
P_i(t)=\frac{1}{d_i}\sum_{e\in E_i}\pi_e(t),\qquad d_i>0.
\tag{3}
$$

**`node_sample`.** Draw $e_i^{\star}\sim\mathrm{Uniform}(E_i)$ independently for
each node and set $P_i(t)=\pi_{e_i^{\star}}(t)$.

**`edge_sweep` (paper's Algorithm 1).** For $e\in E$ in stored order, and for
$i\in e$ in stored order, if $i$ is eligible, draw
$X_i\leftarrow\mathrm{Bernoulli}(\pi_e)$ and update the scores in place
immediately. A node may be updated once per incident edge in a sweep; the last
update wins. This rule is order dependent by construction.

Nodes with $d_i=0$ are frozen: $P_i$ is set to their current opinion.

### 4.4 Latency

Node $i$ is **eligible** at sweep $t$ iff

$$
t-T_i(t)\ \ge\ L_i .
\tag{4}
$$

$L_i$ is drawn once, at initialisation, from

* $\mathrm{Normal}(\mu,\sigma^2)$ rounded to the nearest integer,
* $\mathrm{Exponential}(\lambda)$ with rate $\lambda$ (mean $1/\lambda$),
  rounded, or
* a point mass at $\mu$ (`fixed`),

and clipped below at `latency_floor` (default 1). All distributions are sampled
by inverse-CDF from common uniforms, which yields a paired design across
distributions.

### 4.5 One sweep

For the node-level rules, all eligible nodes are updated **simultaneously**
from the state at the beginning of the sweep:

$$
X_i(t+1)=
\begin{cases}
\mathrm{Bernoulli}\big(P_i(t)\big), & \text{if } i \text{ is eligible and } d_i>0,\\[2pt]
X_i(t), & \text{otherwise,}
\end{cases}
\tag{5}
$$

and

$$
T_i(t+1)=
\begin{cases}
t, & X_i(t+1)\ne X_i(t),\\
T_i(t), & \text{otherwise.}
\end{cases}
\tag{6}
$$

The simulation stops at the first $t\le T$ with
$\max(n_0(t),n_1(t))/n\ge r$, or at $T$ (right-censored).

### 4.6 Pseudocode

```
input: hypergraph (V,E,α), q, latencies L, initial opinions X(0), r, T
T_i ← 0 for all i
for t = 1..T:
    compute S_0(e), S_1(e) for all e from X(t-1)
    compute π_e for all e
    for each i with d_i > 0:
        P_i ← aggregate({π_e : e ∋ i})
        eligible_i ← (t - T_i ≥ L_i)
        X_i(t) ← Bernoulli(P_i) if eligible_i else X_i(t-1)
        T_i ← t if X_i(t) ≠ X_i(t-1)
    record D(t), C(t), X̄(t)
    stop if max(n_0,n_1)/n ≥ r
```

## 5. Metrics

With $n_x(t)=\lvert\{i:X_i(t)=x\}\rvert$:

$$
D(t)=1-\frac{\max(n_0(t),n_1(t))}{n},
\qquad
\bar X(t)=\frac1n\sum_i X_i(t),
\qquad
C(t)=\frac1n\sum_i\mathbf{1}[X_i(t)\ne X_i(t-1)].
\tag{7}
$$

The paper's oscillation index is the sample standard deviation of the mean
opinion over the observation window,

$$
O=\mathrm{sd}\big(\bar X(1),\dots,\bar X(T)\big),
\tag{8}
$$

computed with `ddof=1`. Note that $O$ mixes genuine swinging with any
monotone trend; v2 therefore also reports the **late/early amplitude ratio**
$\mathrm{sd}(\bar X_{T/2+1..T})/\mathrm{sd}(\bar X_{1..T/2})$, which is close
to one for a stationary oscillation and below one for a damped transient.

**Consensus time** $T_c$ is the first $t$ with majority $\ge r$; runs that do
not reach $r$ by $T$ are right-censored. Summaries use the Kaplan–Meier
estimator and the restricted mean survival time

$$
\mathrm{RMST}(T)=\int_0^T S(u)\,du,
\tag{9}
$$

where $S$ is the Kaplan–Meier survival function of "not yet converged". RMST
is defined even when fewer than half of the runs converge, unlike the median.

## 6. Analytical properties

### 6.1 Absorbing states are component-wise, not global

**Proposition 1.** Assume $q>0$, $\alpha_i>0$, and $L_i<\infty$. A state $x$
is absorbing if and only if $x$ is constant on every connected component of
the 2-section graph.

*Proof.* (⇐) If $x$ is constant on a component, every hyperedge inside it is
monochromatic, so $\pi_e\in\{0,1\}$ equals the common opinion; every eligible
node draws its own opinion and nothing changes. Isolated nodes never update.

(⇒) Suppose a component is not constant. Choose $i,j$ in it with
$x_i\ne x_j$ and a path $i=k_0,\dots,k_\ell=j$ in $G_2$. Some path edge
$\{k_{s-1},k_s\}$ has $x_{k_{s-1}}\ne x_{k_s}$; the hyperedge $e$ witnessing
it contains both opinions, so $S_0(e),S_1(e)>0$ and $0<\pi_e<1$. Under
`node_mean`, every node of $e$ has $0<P_i<1$; under `node_sample` each has a
positive probability of sampling $e$; under `edge_sweep` the sweep visits $e$
and each eligible node has a positive flip probability. Since latencies are
finite, every node becomes eligible infinitely often, so the state is not
absorbing. ∎

**Corollary 1.** A hypergraph with $C$ connected components has exactly $2^C$
absorbing states. Global consensus (all nodes equal) is just two of them.

**Corollary 2.** If the hypergraph is connected, the chain is eventually
absorbed in global consensus. If it is fragmented, the chain is eventually
absorbed in a **mosaic** in which each component is internally unanimous but
different components may disagree. On a fragmented graph, global consensus
requires every component to choose the same colour; for $C$ components with
per-component probabilities $p_c$ of choosing 1,

$$
\Pr(\text{global consensus})=\prod_{c=1}^{C}p_c+\prod_{c=1}^{C}(1-p_c)
\;\le\;\max\!\Big(\prod_c p_c,\prod_c(1-p_c)\Big).
\tag{10}
$$

With $C=81$ (the active hashtag graph) and even a generous
$p_c\in[0.1,0.9]$ for every component, the right-hand side is at most
$0.9^{81}\approx 2\times10^{-4}$; with $C=260$ it is at most
$0.9^{260}\approx 1\times10^{-12}$.

Corollary 2 is the formal reason why a population-level 80 % consensus
threshold is not a meaningful target on the Kaggle hashtag hypergraph: the
process converges to a mosaic whose global majority is set by how the
components happen to break, not by a population-wide consensus tendency.

### 6.2 The complete hypergraph: an exact reduction

The mean-field map is usually an approximation. On a complete hypergraph it is
exact for every finite $n$, which gives a clean place to check the intuition
and a reference case for the code.

**Theorem 1 (complete hypergraph).** Let $E=\{V\}$, $\alpha_i=\alpha>0$ for all
$i$, $L_i=1$, and use the `node_mean` rule. Write $N_t=\sum_i X_i(t)$ and
$p_t=N_t/n$. Conditional on $N_t=m$,
$$
N_{t+1}\sim\mathrm{Binomial}\!\left(n,\,f(m/n)\right),
\qquad
f(p)=\frac{p^q}{p^q+(1-p)^q}.
\tag{11}
$$
Moreover:

1. if $q=1$, then $(p_t)$ is a martingale, $\Pr(\lim_t p_t=1)=p_0$, and the
   expected absorption time is finite;
2. if $q>1$, then $f(p)>p$ for $p\in(\tfrac12,1)$ and $f(p)<p$ for
   $p\in(0,\tfrac12)$;
3. if $q<1$, the two inequalities are reversed;
4. for every $q>0$, absorption at $0$ or $n$ occurs almost surely.

*Proof.* Every node belongs to the single edge, so
$S_1=\alpha N_t$, $S_0=\alpha(n-N_t)$, and
$\pi_e=f(N_t/n)$ for every node. The node-mean rule gives $P_i=f(N_t/n)$ for
all $i$, and the $n$ Bernoulli draws are conditionally independent, which is
(11). This proves the transition law.

For the drift, write $p=p_t$ and use $p^q-p(p^q+(1-p)^q)$:
$$
f(p)-p
=\frac{p(1-p)\left(p^{q-1}-(1-p)^{q-1}\right)}
{p^q+(1-p)^q}.
\tag{12}
$$
The denominator is positive for $p\in(0,1)$. If $q>1$, then $p>1/2$ implies
$p^{q-1}>(1-p)^{q-1}$, so the drift is positive; the same comparison reverses
below $1/2$. For $q<1$ the exponent $q-1$ is negative and the comparison
reverses. This proves items 2 and 3.

At $q=1$, $f(p)=p$, so
$\mathbb{E}[N_{t+1}\mid N_t]=n f(N_t/n)=N_t$: $(N_t)$ is a bounded martingale
and converges almost surely. From any interior state the chain can jump to
$0$ or $n$ in one step with probability $(1-f)^n$ or $f^n$, both positive, so
the only recurrent states are $0$ and $n$; absorption is almost sure and the
expected absorption time is finite because the state space is finite. Optional
stopping at the absorption time $T$ gives
$\mathbb{E}[N_T]=N_0$, and $N_T=n$ with probability $\Pr(p_T=1)$; hence
$\Pr(p_T=1)=p_0$. This proves item 1. Item 4 follows from the same positive
one-step probability of hitting a boundary for every interior state. ∎

**Corollary 3.** At $q=1$ the complete-hypergraph model is the neutral
Wright–Fisher chain: the expected opinion share is conserved, the consensus
probability equals the initial share, and
$\mathrm{Var}(N_{t+1}\mid N_t)=n\,p_t(1-p_t)$. The diffusion limit is
$\dd p=\sqrt{p(1-p)}\,\dd W$.

The theorem is useful for two reasons. It turns the mean-field picture into a
finite-$n$ statement on a specific graph, and it gives the code a case where the
conditional distribution is known exactly. The repository checks both the
kernel and the q=1 absorption probability against this result
(`tests/test_complete_graph_theory.py`).

### 6.3 Limits of the nonlinearity


For a non-empty edge with $S_0,S_1>0$:

* $q\to0$: $\pi_e\to\tfrac12$ (the group carries no information);
* $q=1$: $\pi_e=S_1/(S_0+S_1)$ (weighted proportional adoption);
* $q\to\infty$: $\pi_e\to\mathbf{1}[S_1>S_0]+\tfrac12\mathbf{1}[S_1=S_0]$
  (deterministic local majority);
* the map $q\mapsto\pi_e$ is increasing in $S_1/S_0$, and
  $\pi_e(S_0,S_1)=1-\pi_e(S_1,S_0)$ (colour symmetry).

### 6.4 Mean-field approximation on sparse graphs

Consider a well-mixed population with homogeneous weights and all nodes
eligible, and let $\phi_t$ be the fraction of 1s. Ignoring finite-size
fluctuations and correlations, a random edge of size $s$ has
$S_1\approx s\phi_t$ and $S_0\approx s(1-\phi_t)$, hence

$$
\mathbb{E}[\phi_{t+1}\mid\phi_t]\approx f(\phi_t)
=\frac{\phi_t^q}{\phi_t^q+(1-\phi_t)^q}.
\tag{13}
$$

$f$ has fixed points $0$, $1$, and $\tfrac12$ (for $q>0$), and

$$
f'\!\left(\tfrac12\right)=q .
\tag{14}
$$

Thus the symmetric state is unstable for $q>1$, stable for $q<1$, and neutral
for $q=1$. This predicts a continuous transition at $q=1$ in infinite
well-mixed systems: $q>1$ drives the population to one of the two consensus
states, $q<1$ admits a stable coexistence fixed point, and finite systems
eventually absorb because of demographic noise. The simulation results in
[`RESULTS.md`](RESULTS.md) are consistent with this picture on connected
random hypergraphs ($q=0.5$: no consensus within 100 sweeps; $q=2.5$:
87–100 % consensus).

The mean-field argument does **not** apply to a fragmented hypergraph: each
component has its own $\phi_t^{(c)}$ and its own absorbing colour, and the
global fraction is a weighted sum of component states.

### 6.5 Latency does not create sustained oscillations

Latency changes *when* a node may update, not the absorbing set: the
eligibility condition (4) only delays updates, and every finite $L_i$ is
eventually satisfied. The chain therefore remains a finite absorbing chain.
Any swinging of $\bar X(t)$ is transient and must decay once components
absorb. Latency can create **damped waves** (a refractory node is temporarily
removed from the pool, which can push the active pool the other way), but not
a sustained oscillation in a finite population. This is the formal counterpart
of the empirical damping reported in [`RESULTS.md`](RESULTS.md).

## 7. Relation to the manuscript

| manuscript | v2 | note |
|---|---|---|
| Eq. (1), $S_X(e)$ | Eq. (1) | identical |
| Eq. (2), $\pi_e$ | Eq. (2) | identical; 0/0 convention stated |
| Eq. (3), update | Eq. (5) | the manuscript writes `argmax` in Eq. (3) but Algorithm 1 lines 17–21 sample; v2 follows the algorithm (Bernoulli sampling) |
| Eq. (4), $\bar X$ | Eq. (7) | identical |
| Eq. (5), $C(t)$ | Eq. (7) | identical up to the time index convention |
| Eq. (6), $D(t)$ | Eq. (7) | identical |
| Eq. (7), $O$ | Eq. (8) | identical definition; the v1 code plotted $\bar X$, not $O$ |
| Eq. (8), $C(t)$ | Eq. (7) | repeated in the manuscript |
| Algorithm 1 | §4.6 | v2 default replaces the edge loop by `node_mean`; `edge_sweep` implements the literal algorithm |

## 8. Assumptions and limitations

1. **Binary, fixed opinions.** No neutrality, abstention, multi-party opinions,
   or continuous stances.
2. **Static hypergraph.** No edge rewiring, group formation, or dissolution.
   The adaptive-hypergraph variant of Golovin et al. is out of scope.
3. **Static weights.** $\alpha_i$ does not react to the opinion or to
   engagement dynamics.
4. **Self-influence.** A node's own weight is included in every $S_x(e)$, as in
   Eq. (1). This gives inertia; it is part of the model, not a bug.
5. **Sweep time.** Latency is measured in sweeps, not wall-clock time; one
   sweep is a modelling choice, not an empirical unit.
6. **No external shocks, zealots, or media.** All change is endogenous.
7. **Sentiment is not stance.** The dataset labels are emotions; the binary
   opinion is a coarse valence projection (see [`DATA.md`](DATA.md)).
8. **No causal identification.** The experiments are a controlled numerical
   study of the model, not an estimate of effects in a population.
