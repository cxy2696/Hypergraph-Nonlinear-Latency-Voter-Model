import numpy as np
import pytest

from hnl.metrics import (
    absorption_by_component,
    consensus_time,
    damping_halflife,
    diversity,
    kaplan_meier,
    km_median,
    lag1_autocorrelation,
    oscillation_index,
    restricted_mean_survival_time,
    zero_crossings,
)


def test_diversity_is_minority_share():
    assert diversity(3, 1) == pytest.approx(0.25)
    assert diversity(0, 5) == pytest.approx(0.0)


def test_oscillation_index_uses_sample_sd():
    assert oscillation_index([0.2, 0.4, 0.6]) == pytest.approx(0.2)
    assert oscillation_index([0.5]) == 0.0


def test_consensus_time():
    assert consensus_time([0.5, 0.7, 0.85], 0.8) == (3, False)
    assert consensus_time([0.5, 0.7, 0.75], 0.8) == (None, True)


def test_zero_crossings():
    assert zero_crossings([0.1, 0.6, 0.4, 0.7, 0.2]) == 4
    assert zero_crossings([0.6, 0.7]) == 0


def test_lag1_autocorrelation():
    assert lag1_autocorrelation([1, 2, 3, 4]) == pytest.approx(1.0)
    assert np.isnan(lag1_autocorrelation([1, 1, 1]))


def test_kaplan_meier_and_rmst():
    times, survival = kaplan_meier([1, 2, 3, 4, 5, 6], [1, 1, 1, 0, 0, 0])
    assert times.tolist() == [1.0, 2.0, 3.0]
    assert survival.tolist() == pytest.approx([5 / 6, 2 / 3, 0.5])
    assert km_median(times, survival) == 3.0
    assert restricted_mean_survival_time(times, survival, 6) == pytest.approx(4.0)
    assert restricted_mean_survival_time(np.array([]), np.array([]), 5) == 5.0


def test_km_median_unresolved():
    times, survival = kaplan_meier([1, 2, 3, 4], [1, 0, 0, 0])
    assert km_median(times, survival) == float("inf")


def test_absorption_by_component():
    labels = np.array([0, 0, 1, 1, 2])
    opinions = np.array([1, 1, 0, 1, 0])
    majority, absorbed = absorption_by_component(opinions, labels)
    assert absorbed.tolist() == [True, False, True]
    assert majority.tolist() == pytest.approx([1.0, 0.5, 1.0])


def test_damping_halflife_on_exponential_decay():
    # |x - 0.5| halves every 10 sweeps -> half-life 10
    t = np.arange(80)
    x = 0.5 + 0.4 * 0.5 ** (t / 10)
    assert damping_halflife(x) == pytest.approx(10.0, rel=0.05)
    assert np.isnan(damping_halflife(np.full(80, 0.5)))
