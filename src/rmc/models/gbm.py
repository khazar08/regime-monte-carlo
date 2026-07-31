from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class GBMParams:
    
    mu_daily: float
    sigma_daily: float

    @property
    def mu_annual(self) -> float:
        return self.mu_daily * 252

    @property
    def sigma_annual(self) -> float:
        return self.sigma_daily * np.sqrt(252)


def fit_gbm(log_returns: np.ndarray) -> GBMParams:
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
    mu = params.mu_daily
    sigma = params.sigma_daily

    Z = rng.standard_normal((n_paths, horizon))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    cum_log_returns = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )
    paths = s0 * np.exp(cum_log_returns)
    return paths
