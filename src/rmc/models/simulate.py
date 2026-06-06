"""Vectorized regime-switching Monte Carlo simulation."""

from __future__ import annotations

import numpy as np

from rmc.models.regime import RegimeParams, stationary_distribution


def simulate_regime(
    s0: float,
    params: RegimeParams,
    horizon: int,
    n_paths: int,
    init_state: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized regime-switching simulation.

    At each step, all paths simultaneously transition states (categorical
    draw via CDF + uniform), then draw log returns from the active state's
    Normal distribution. No Python loops over paths.

    Parameters
    ----------
    s0 : float
        Initial price.
    params : RegimeParams
        Calibrated HMM parameters.
    horizon : int
        Number of daily steps.
    n_paths : int
        Number of Monte Carlo paths.
    init_state : str
        How to initialize state per path: 'stationary', 'last', or 'sampled'.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Shape (n_paths, horizon+1); column 0 == s0.
    """
    K = len(params.means)
    transmat = params.transmat        # (K, K)
    means = params.means              # (K,)
    sigmas = params.sigmas            # (K,)

    # Cumulative transition matrix for vectorized state sampling
    cum_transmat = np.cumsum(transmat, axis=1)  # (K, K)

    # Initialize states for all paths
    if init_state == "stationary":
        pi = stationary_distribution(params.transmat)
    elif init_state == "sampled":
        pi = params.startprob
    else:  # "last" falls back to startprob; caller should set startprob appropriately
        pi = params.startprob

    states = rng.choice(K, size=n_paths, p=pi)  # (n_paths,)

    log_price_increments = np.empty((n_paths, horizon))

    for t in range(horizon):
        # --- Transition: vectorized categorical draw ---
        u = rng.uniform(size=n_paths)                    # (n_paths,)
        # cum_transmat[states] shape: (n_paths, K)
        cdf = cum_transmat[states]                        # (n_paths, K)
        # New state = first K where CDF exceeds u
        new_states = (cdf < u[:, np.newaxis]).sum(axis=1)  # (n_paths,)
        new_states = np.clip(new_states, 0, K - 1)
        states = new_states

        # --- Draw returns from active state's Normal ---
        z = rng.standard_normal(n_paths)                 # (n_paths,)
        log_returns = means[states] + sigmas[states] * z  # (n_paths,)
        log_price_increments[:, t] = log_returns

    cum_log = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_price_increments, axis=1)],
        axis=1,
    )
    paths = s0 * np.exp(cum_log)
    return paths
