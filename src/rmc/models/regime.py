"""Hidden Markov Model regime detection and parameter extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM


@dataclass
class RegimeParams:
    """Parameters extracted from a fitted GaussianHMM, sorted by mean return."""

    startprob: np.ndarray    # (K,) initial state distribution
    transmat: np.ndarray     # (K, K) transition matrix, rows sum to 1
    means: np.ndarray        # (K,) daily mean log return per state
    sigmas: np.ndarray       # (K,) daily std per state
    state_labels: list[str]  # sorted by mean, e.g. ["bear","neutral","bull"]


def fit_regime(
    log_returns: np.ndarray,
    n_states: int,
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[RegimeParams, GaussianHMM]:
    """Fit a GaussianHMM to log returns, sorting states by mean.

    Parameters
    ----------
    log_returns : np.ndarray
        1-D array of daily log returns.
    n_states : int
        Number of hidden states.
    n_iter : int
        Maximum EM iterations.
    rng : np.random.Generator
        Used to derive an integer seed for GaussianHMM.

    Returns
    -------
    tuple[RegimeParams, GaussianHMM]
        Sorted regime parameters and the fitted model.
    """
    seed = int(rng.integers(0, 2**31))
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=seed,
    )
    obs = log_returns.reshape(-1, 1)
    model.fit(obs)

    raw_means = model.means_[:, 0]          # (K,)
    raw_sigmas = np.sqrt(model.covars_[:, 0, 0])  # (K,)

    # Sort states by mean return: bear < neutral < ... < bull
    order = np.argsort(raw_means)
    sorted_means = raw_means[order]
    sorted_sigmas = raw_sigmas[order]
    sorted_startprob = model.startprob_[order]
    sorted_transmat = model.transmat_[np.ix_(order, order)]

    # Re-label model internals to match sorted order
    model.means_ = sorted_means.reshape(-1, 1)
    model.covars_ = sorted_sigmas.reshape(-1, 1, 1) ** 2
    model.startprob_ = sorted_startprob
    model.transmat_ = sorted_transmat

    labels = _make_labels(n_states)

    params = RegimeParams(
        startprob=sorted_startprob,
        transmat=sorted_transmat,
        means=sorted_means,
        sigmas=sorted_sigmas,
        state_labels=labels,
    )
    return params, model


def _make_labels(n_states: int) -> list[str]:
    """Generate state labels from bear -> bull based on count."""
    presets = {2: ["bear", "bull"], 3: ["bear", "neutral", "bull"]}
    if n_states in presets:
        return presets[n_states]
    return [f"state_{i}" for i in range(n_states)]


def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a transition matrix.

    Parameters
    ----------
    transmat : np.ndarray
        (K, K) row-stochastic transition matrix.

    Returns
    -------
    np.ndarray
        (K,) stationary probability vector satisfying pi @ A == pi.
    """
    K = transmat.shape[0]
    # Left eigenvector for eigenvalue 1: solve (A^T - I) pi = 0
    A = transmat.T - np.eye(K)
    # Replace last equation with normalization constraint
    A[-1] = 1.0
    b = np.zeros(K)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0, None)
    pi /= pi.sum()
    return pi


def decode_states(model: GaussianHMM, log_returns: np.ndarray) -> np.ndarray:
    """Viterbi decode the most-likely state sequence.

    Parameters
    ----------
    model : GaussianHMM
        Fitted HMM.
    log_returns : np.ndarray
        1-D array of daily log returns.

    Returns
    -------
    np.ndarray
        Integer state indices, shape (T,).
    """
    obs = log_returns.reshape(-1, 1)
    _, states = model.decode(obs, algorithm="viterbi")
    return states
