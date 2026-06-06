"""Tests for the regime HMM model."""

import numpy as np

from rmc.models.regime import fit_regime, stationary_distribution


def test_stationary_distribution_valid():
    transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = stationary_distribution(transmat)
    assert pi.shape == (2,)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-8)
    assert np.all(pi >= 0)


def test_stationary_distribution_invariance():
    transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = stationary_distribution(transmat)
    np.testing.assert_allclose(pi @ transmat, pi, atol=1e-6)


def test_transmat_rows_sum_to_one():
    rng = np.random.default_rng(42)
    # 2-regime synthetic data
    n = 2000
    returns = np.concatenate(
        [rng.normal(-0.002, 0.015, n // 2), rng.normal(0.001, 0.008, n // 2)]
    )
    rng.shuffle(returns)
    params, model = fit_regime(returns, n_states=2, n_iter=50, rng=rng)
    np.testing.assert_allclose(params.transmat.sum(axis=1), np.ones(2), atol=1e-6)


def test_fit_regime_recovers_two_states():
    rng = np.random.default_rng(7)
    n = 3000
    low_vol = rng.normal(-0.003, 0.005, n // 2)
    high_vol = rng.normal(0.003, 0.025, n // 2)
    returns = np.concatenate([low_vol, high_vol])
    params, _ = fit_regime(returns, n_states=2, n_iter=100, rng=rng)
    # Sorted by mean: state 0 < state 1
    assert params.means[0] < params.means[1]
    assert len(params.state_labels) == 2
