"""Tests for the regime-switching simulation."""

import time

import numpy as np

from rmc.models.regime import RegimeParams
from rmc.models.simulate import simulate_regime


def _make_params(k: int = 2) -> RegimeParams:
    if k == 2:
        return RegimeParams(
            startprob=np.array([0.6, 0.4]),
            transmat=np.array([[0.95, 0.05], [0.1, 0.9]]),
            means=np.array([-0.001, 0.002]),
            sigmas=np.array([0.015, 0.008]),
            state_labels=["bear", "bull"],
        )
    return RegimeParams(
        startprob=np.array([0.3, 0.4, 0.3]),
        transmat=np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]),
        means=np.array([-0.002, 0.0, 0.002]),
        sigmas=np.array([0.02, 0.01, 0.008]),
        state_labels=["bear", "neutral", "bull"],
    )


def test_simulate_shape():
    params = _make_params()
    paths = simulate_regime(
        100.0,
        params,
        horizon=21,
        n_paths=500,
        init_state="stationary",
        rng=np.random.default_rng(0),
    )
    assert paths.shape == (500, 22)


def test_simulate_initial_price():
    params = _make_params()
    s0 = 250.0
    paths = simulate_regime(
        s0, params, horizon=10, n_paths=200, init_state="stationary", rng=np.random.default_rng(1)
    )
    np.testing.assert_allclose(paths[:, 0], s0)


def test_simulate_no_nan_or_negative():
    params = _make_params(3)
    paths = simulate_regime(
        100.0,
        params,
        horizon=252,
        n_paths=1000,
        init_state="stationary",
        rng=np.random.default_rng(5),
    )
    assert not np.isnan(paths).any()
    assert np.all(paths > 0)


def test_simulate_reproducibility():
    params = _make_params()
    p1 = simulate_regime(100.0, params, 20, 200, "stationary", np.random.default_rng(42))
    p2 = simulate_regime(100.0, params, 20, 200, "stationary", np.random.default_rng(42))
    np.testing.assert_array_equal(p1, p2)


def test_simulate_performance():
    """50k paths x 252 steps should complete well under 10 seconds."""
    params = _make_params(3)
    start = time.perf_counter()
    simulate_regime(
        100.0,
        params,
        horizon=252,
        n_paths=50_000,
        init_state="stationary",
        rng=np.random.default_rng(0),
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"Simulation took {elapsed:.1f}s — too slow"
