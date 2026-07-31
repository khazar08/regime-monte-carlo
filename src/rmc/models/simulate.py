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

    K = len(params.means)
    transmat = params.transmat        # (K, K)
    means = params.means              # (K,)
    sigmas = params.sigmas            # (K,)

    cum_transmat = np.cumsum(transmat, axis=1)  # (K, K)

    if init_state == "stationary":
        pi = stationary_distribution(params.transmat)
    elif init_state == "sampled":
        pi = params.startprob
    else:  #
        pi = params.startprob

    states = rng.choice(K, size=n_paths, p=pi)  # (n_paths,)

    log_price_increments = np.empty((n_paths, horizon))

    for t in range(horizon):
    
        u = rng.uniform(size=n_paths)                  
        cdf = cum_transmat[states]                        
        new_states = (cdf < u[:, np.newaxis]).sum(axis=1) 
        new_states = np.clip(new_states, 0, K - 1)
        states = new_states

        z = rng.standard_normal(n_paths)                
        log_returns = means[states] + sigmas[states] * z 
        log_price_increments[:, t] = log_returns

    cum_log = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_price_increments, axis=1)],
        axis=1,
    )
    paths = s0 * np.exp(cum_log)
    return paths
