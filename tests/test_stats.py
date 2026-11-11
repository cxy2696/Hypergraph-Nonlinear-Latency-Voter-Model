import numpy as np
import pytest

from hnl.stats import (
    bootstrap_ci,
    cliffs_delta,
    holm_adjust,
    paired_contrast,
    paired_permutation_test,
    summarize,
)


def test_bootstrap_ci_constant():
    lo, hi = bootstrap_ci([2.0] * 20, n_boot=200, rng=np.random.default_rng(0))
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)


def test_bootstrap_ci_covers_mean():
    x = np.random.default_rng(0).normal(10, 1, size=200)
    lo, hi = bootstrap_ci(x, n_boot=2000, rng=np.random.default_rng(1))
    assert lo < x.mean() < hi
    assert hi - lo < 1.0


def test_paired_permutation_detects_shift():
    a = np.arange(10, dtype=float)
    b = a - 1.0
    p = paired_permutation_test(a, b, n_perm=5000, rng=np.random.default_rng(0))
    assert p < 0.01


def test_paired_permutation_null():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20)
    b = a + rng.normal(size=20)
    p = paired_permutation_test(a, b, n_perm=5000, rng=np.random.default_rng(1))
    assert p > 0.01


def test_holm_adjust():
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.6]).tolist() == pytest.approx([1.0, 1.0])


def test_cliffs_delta():
    assert cliffs_delta([2, 3], [1]) == 1.0
    assert cliffs_delta([1], [2, 3]) == -1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0


def test_summarize_filters_non_finite():
    stats = summarize([1.0, 2.0, np.inf, np.nan])
    assert stats["n"] == 4
    assert stats["n_finite"] == 2
    assert stats["mean"] == pytest.approx(1.5)
    empty = summarize([np.inf])
    assert empty["n_finite"] == 0
    assert np.isnan(empty["mean"])


def test_paired_contrast_structure():
    out = paired_contrast([2, 3, 4], [1, 2, 3], n_perm=500, n_boot=200, rng=np.random.default_rng(0))
    assert out["mean_diff"] == pytest.approx(1.0)
    assert out["n"] == 3
