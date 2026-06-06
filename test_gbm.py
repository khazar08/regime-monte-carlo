"""Tests for the GBM model."""

import numpy as np

from rmc.models.gbm import GBMParams, fit_gbm, simulate_gbm


def test_fit_gbm_recovers_params():
    rng = np.random.default_rng(42)
    true_mu = 0.0003
    true_sigma = 0.012
    n = 100_000
    returns = rng.normal(true_mu, true_sigma, n)
    params = fit_gbm(returns)
    assert abs(params.mu_daily - true_mu) < 0.001
    assert abs(params.sigma_daily - true_sigma) < 0.001


def test_simulate_gbm_shape():
    rng = np.random.default_rng(0)
    params = GBMParams(mu_daily=0.0003, sigma_daily=0.01)
    paths = simulate_gbm(100.0, params, horizon=21, n_paths=500, dt=1.0, rng=rng)
    assert paths.shape == (500, 22)


def test_simulate_gbm_initial_price():
    rng = np.random.default_rng(0)
    params = GBMParams(mu_daily=0.0001, sigma_daily=0.01)
    s0 = 123.45
    paths = simulate_gbm(s0, params, horizon=10, n_paths=100, dt=1.0, rng=rng)
    np.testing.assert_allclose(paths[:, 0], s0)


def test_simulate_gbm_no_negatives():
    rng = np.random.default_rng(7)
    params = GBMParams(mu_daily=-0.001, sigma_daily=0.02)
    paths = simulate_gbm(50.0, params, horizon=252, n_paths=1000, dt=1.0, rng=rng)
    assert np.all(paths > 0)


def test_simulate_gbm_reproducibility():
    params = GBMParams(mu_daily=0.0002, sigma_daily=0.015)
    paths1 = simulate_gbm(100.0, params, 10, 50, 1.0, np.random.default_rng(99))
    paths2 = simulate_gbm(100.0, params, 10, 50, 1.0, np.random.default_rng(99))
    np.testing.assert_array_equal(paths1, paths2)
