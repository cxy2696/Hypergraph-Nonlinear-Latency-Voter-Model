"""Small, dependency-free statistical helpers.

The package deliberately avoids SciPy: every interval and test below is
implemented directly with NumPy so that reported numbers can be audited
without a hidden statistical library.  Methods are stated in
``docs/RESULTS.md``.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence

import numpy as np


def bootstrap_ci(
    values: Iterable[float],
    statistic: Callable[..., np.ndarray] = np.mean,
    n_boot: int = 4000,
    level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval.

    ``statistic`` must accept an ``axis`` argument (``np.mean``, ``np.median``).
    """
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    if x.size == 1:
        return (float(x[0]), float(x[0]))
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, x.size, size=(int(n_boot), x.size))
    draws = statistic(x[idx], axis=1)
    lo, hi = np.percentile(draws, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return (float(lo), float(hi))


def summarize(
    values: Iterable[float],
    n_boot: int = 4000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Mean/SD/median/IQR and bootstrap CI of the mean."""
    raw = np.asarray(list(values), dtype=float)
    x = raw[np.isfinite(raw)]
    if x.size == 0:
        nan = float("nan")
        return {
            "n": int(raw.size), "n_finite": 0,
            "mean": nan, "sd": nan, "se": nan, "ci_low": nan, "ci_high": nan,
            "median": nan, "median_ci_low": nan, "median_ci_high": nan,
            "q1": nan, "q3": nan, "min": nan, "max": nan,
        }
    q1, q3 = np.percentile(x, [25, 75]) if x.size > 1 else (x[0], x[0])
    lo, hi = bootstrap_ci(x, np.mean, n_boot=n_boot, rng=rng)
    mlo, mhi = bootstrap_ci(x, np.median, n_boot=n_boot, rng=rng)
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return {
        "n": int(raw.size),
        "n_finite": int(x.size),
        "mean": float(np.mean(x)),
        "sd": sd,
        "se": float(sd / np.sqrt(x.size)) if x.size else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "median": float(np.median(x)),
        "median_ci_low": mlo,
        "median_ci_high": mhi,
        "q1": float(q1),
        "q3": float(q3),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def paired_permutation_test(
    a: Iterable[float],
    b: Iterable[float],
    n_perm: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Two-sided paired sign-flip permutation test of ``mean(a - b) = 0``.

    Under the null the sign of each paired difference is exchangeable.  A
    Monte-Carlo p-value with the standard ``(1 + count) / (1 + n_perm)``
    correction is returned.
    """
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    d = a - b
    if d.size == 0:
        return float("nan")
    observed = abs(float(np.mean(d)))
    if observed == 0.0:
        return 1.0
    rng = rng or np.random.default_rng(0)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (1 + n_perm))


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm–Bonferroni step-down adjustment, returned in input order."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    """Cliff's delta effect size in ``[-1, 1]`` (positive: ``a`` tends larger)."""
    a = np.asarray(list(a), dtype=float)[:, None]
    b = np.asarray(list(b), dtype=float)[None, :]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float((a > b).mean() - (a < b).mean())


def paired_contrast(
    a: Iterable[float],
    b: Iterable[float],
    n_perm: int = 10000,
    n_boot: int = 4000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Difference ``a - b`` with bootstrap CI and paired permutation p-value."""
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    diff = a - b
    rng = rng or np.random.default_rng(0)
    lo, hi = bootstrap_ci(diff, np.mean, n_boot=n_boot, rng=rng)
    return {
        "n": int(diff.size),
        "mean_diff": float(np.mean(diff)) if diff.size else float("nan"),
        "median_diff": float(np.median(diff)) if diff.size else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": paired_permutation_test(a, b, n_perm=n_perm, rng=rng),
        "cliffs_delta": cliffs_delta(a, b),
    }
