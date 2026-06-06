"""Tests for risk metrics."""

import numpy as np

from rmc.evaluation.risk import (
    conditional_var,
    max_drawdown,
    prob_drawdown_exceeds,
    terminal_returns,
    value_at_risk,
)


def test_terminal_returns_shape():
    paths = np.ones((100, 11)) * np.linspace(1, 1.1, 11)
    rets = terminal_returns(paths)
    assert rets.shape == (100,)


def test_var_cvar_relationship():
    rng = np.random.default_rng(0)
    returns = rng.normal(-0.05, 0.1, 50_000)
    var = value_at_risk(returns, 0.95)
    cvar = conditional_var(returns, 0.95)
    # CVaR >= VaR in terms of loss magnitude
    assert cvar >= var - 1e-6


def test_var_analytical():
    # N(-0.10, 0.05^2): 5th percentile = -0.10 - 1.645*0.05 ≈ -0.18225
    rng = np.random.default_rng(1)
    returns = rng.normal(-0.10, 0.05, 500_000)
    var = value_at_risk(returns, 0.95)
    expected = 0.10 + 1.645 * 0.05
    assert abs(var - expected) < 0.005


def test_max_drawdown_bounds():
    rng = np.random.default_rng(2)
    paths = np.abs(rng.lognormal(0, 0.1, (500, 253)))
    dd = max_drawdown(paths)
    assert np.all(dd >= 0)
    assert np.all(dd <= 1)


def test_max_drawdown_zero_for_monotone():
    # Strictly increasing path -> drawdown = 0
    path = np.linspace(100, 200, 50).reshape(1, -1)
    dd = max_drawdown(path)
    np.testing.assert_allclose(dd, 0.0, atol=1e-10)


def test_prob_drawdown_exceeds_range():
    rng = np.random.default_rng(3)
    paths = np.abs(rng.lognormal(0, 0.1, (1000, 63)))
    p = prob_drawdown_exceeds(paths, 0.10)
    assert 0.0 <= p <= 1.0
