"""Geometric Brownian Motion baseline model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GBMParams:
    """MLE parameters for a GBM in daily units."""

    mu_daily: float
    sigma_daily: float

    @property
    def mu_annual(self) -> float:
        return self.mu_daily * 252

    @property
    def sigma_annual(self) -> float:
        return self.sigma_daily * np.sqrt(252)


def fit_gbm(log_returns: np.ndarray) -> GBMParams:
    """Fit GBM parameters via MLE from daily log returns.

    Parameters
    ----------
    log_returns : np.ndarray
        1-D array of daily log returns.

    Returns
    -------
    GBMParams
        Estimated daily drift and volatility.
    """
    mu = float(np.mean(log_returns))
    sigma = float(np.std(log_returns, ddof=1))
    return GBMParams(mu_daily=mu, sigma_daily=sigma)


def simulate_gbm(
    s0: float,
    params: GBMParams,
    horizon: int,
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate GBM paths. Fully vectorized.

    Parameters
    ----------
    s0 : float
        Initial price.
    params : GBMParams
        Calibrated GBM parameters (daily units).
    horizon : int
        Number of daily steps.
    n_paths : int
        Number of Monte Carlo paths.
    dt : float
        Time step size (1.0 for daily units).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n_paths, horizon+1); column 0 == s0.
    """
    mu = params.mu_daily
    sigma = params.sigma_daily

    # Daily log-return increments: (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
    Z = rng.standard_normal((n_paths, horizon))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative sum → cumulative log return
    cum_log_returns = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )
    paths = s0 * np.exp(cum_log_returns)
    return paths
