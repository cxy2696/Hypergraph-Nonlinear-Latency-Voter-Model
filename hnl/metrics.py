"""Metric definitions used in the paper, in one place.

The paper defines (Eq. 6–8):

* diversity        ``D(t) = 1 - max(n0(t), n1(t)) / n``
* oscillation      ``O = std_t( X̄(t) )`` with ``X̄(t) = mean_i X_i(t)``
* cascade size     ``C(t) = (1/n) sum_i 1[X_i(t) != X_i(t-1)]``

``D`` and ``C`` are computed per sweep by the engine.  ``O`` is computed here
from the mean-opinion trajectory (sample standard deviation, ``ddof=1``).
The remaining helpers are diagnostics introduced in v2 and are documented in
``docs/MODEL.md``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


def diversity(n0: int, n1: int) -> float:
    """Minority share, the paper's ``D(t)``."""
    n = n0 + n1
    if n <= 0:
        raise ValueError("empty population")
    return 1.0 - max(n0, n1) / n


def oscillation_index(mean_opinion: Iterable[float]) -> float:
    """Paper Eq. (7): sample standard deviation of the mean opinion."""
    x = np.asarray(list(mean_opinion), dtype=float)
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=1))


def consensus_time(
    majority: Iterable[float], threshold: float
) -> Tuple[Optional[int], bool]:
    """First 1-based sweep at which the majority share reaches ``threshold``.

    Returns ``(time, censored)``.  ``time`` is ``None`` when the threshold is
    never reached within the observed window, in which case the observation is
    right-censored at ``len(majority)``.
    """
    m = np.asarray(list(majority), dtype=float)
    hit = np.flatnonzero(m >= threshold)
    if hit.size == 0:
        return None, True
    return int(hit[0] + 1), False


def zero_crossings(x: Iterable[float], level: float = 0.5) -> int:
    """Number of sign changes of ``x - level`` (a coarse swing counter)."""
    d = np.asarray(list(x), dtype=float) - level
    s = np.sign(d)
    s = s[s != 0]
    if s.size < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))


def damping_halflife(
    x: Iterable[float], tail: int = 5, min_points: int = 5
) -> float:
    """Half-life (sweeps) of the deviation from the terminal opinion share.

    A straight line is fitted to ``log|x(t) - x_terminal|`` for points after
    the first 5 % of the window.  Returns ``inf`` when the envelope does not
    decay, and ``nan`` when fewer than ``min_points`` usable points exist.
    This is a descriptive diagnostic, not a fitted dynamical model.
    """
    x = np.asarray(list(x), dtype=float)
    if x.size < 2 * min_points:
        return float("nan")
    terminal = float(np.mean(x[-max(1, tail):]))
    dev = np.abs(x - terminal)
    t = np.arange(x.size)
    start = max(1, x.size // 20)
    if start >= x.size:
        return float("nan")
    # Follow the main decay only: once the deviation is within 5 % of its
    # initial value it is dominated by the terminal-offset replacement error.
    threshold = max(1e-6, 0.05 * float(dev[start]))
    mask = (t >= start) & (dev > threshold)
    if int(mask.sum()) < min_points:
        return float("nan")
    slope, _ = np.polyfit(t[mask], np.log(dev[mask]), 1)
    if slope >= 0:
        return float("inf")
    return float(np.log(2.0) / (-slope))


def lag1_autocorrelation(x: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    if x.size < 3:
        return float("nan")
    a, b = x[:-1], x[1:]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def absorption_by_component(
    final_opinions: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-component final majorities and absorption flags.

    A component is *absorbed* when all its nodes share one opinion.  Returns
    ``(majority_by_component, absorbed)`` aligned with sorted component ids.
    """
    comps = np.unique(labels)
    majority = np.empty(comps.size, dtype=float)
    absorbed = np.empty(comps.size, dtype=bool)
    for k, c in enumerate(comps):
        vals = final_opinions[labels == c]
        ones = int(vals.sum())
        majority[k] = max(ones, vals.size - ones) / vals.size
        absorbed[k] = bool(vals.min() == vals.max())
    return majority, absorbed


def kaplan_meier(
    durations: Iterable[float], events: Iterable[bool]
) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan–Meier survival estimate for right-censored consensus times.

    Parameters
    ----------
    durations:
        Observed time, i.e. consensus time when ``event`` is true and the
        censoring time (``max_steps``) otherwise.
    events:
        ``True`` when consensus was observed, ``False`` for censoring.

    Returns
    -------
    ``(times, survival)`` evaluated at the event times.
    """
    d = np.asarray(list(durations), dtype=float)
    e = np.asarray(list(events), dtype=bool)
    if d.size != e.size:
        raise ValueError("durations and events must have equal length")
    if d.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(d, kind="stable")
    d, e = d[order], e[order]
    at_risk = d.size
    survival = 1.0
    times, values = [], []
    for value in np.unique(d):
        mask = d == value
        n_events = int(e[mask].sum())
        if n_events:
            survival *= 1.0 - n_events / at_risk
            times.append(float(value))
            values.append(survival)
        at_risk -= int(mask.sum())
    return np.asarray(times), np.asarray(values)


def km_median(times: np.ndarray, survival: np.ndarray) -> float:
    """Median survival time from a Kaplan–Meier curve (``inf`` if unresolved)."""
    if times.size == 0:
        return float("nan")
    idx = np.flatnonzero(survival <= 0.5)
    if idx.size == 0:
        return float("inf")
    return float(times[idx[0]])


def restricted_mean_survival_time(
    times: np.ndarray, survival: np.ndarray, tau: float
) -> float:
    """Area under the Kaplan–Meier curve up to ``tau`` (RMST).

    RMST is a censoring-aware summary of *how long* runs stay unconverged: a
    larger value means slower consensus.  It is defined for every replicate,
    including censored ones, unlike the median when fewer than half of the
    runs converge.
    """
    if tau <= 0:
        return 0.0
    t = np.asarray(times, dtype=float)
    s = np.asarray(survival, dtype=float)
    if t.size == 0:
        return float(tau)
    grid = np.unique(np.concatenate(([0.0], t[t < tau], [float(tau)])))
    area = 0.0
    for k in range(grid.size - 1):
        idx = int(np.searchsorted(t, grid[k], side="right")) - 1
        value = float(s[idx]) if idx >= 0 else 1.0
        area += (grid[k + 1] - grid[k]) * value
    return float(area)
