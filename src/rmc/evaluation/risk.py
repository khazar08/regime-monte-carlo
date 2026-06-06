"""Risk metrics derived from simulated price paths."""

from __future__ import annotations

import numpy as np


def terminal_returns(paths: np.ndarray) -> np.ndarray:
    """Compute terminal returns R = S_T / S_0 - 1.

    Parameters
    ----------
    paths : np.ndarray
        Shape (n_paths, horizon+1).

    Returns
    -------
    np.ndarray
        1-D array of terminal returns.
    """
    return paths[:, -1] / paths[:, 0] - 1.0


def value_at_risk(returns: np.ndarray, alpha: float) -> float:
    """Compute Value at Risk at confidence level alpha.

    VaR_alpha = -quantile(R, 1 - alpha)

    Parameters
    ----------
    returns : np.ndarray
        1-D array of returns.
    alpha : float
        Confidence level, e.g. 0.95.

    Returns
    -------
    float
        VaR (positive = loss).
    """
    return float(-np.quantile(returns, 1.0 - alpha))


def conditional_var(returns: np.ndarray, alpha: float) -> float:
    """Compute Conditional VaR (Expected Shortfall) at confidence level alpha.

    CVaR_alpha = -mean( R[ R <= quantile(R, 1 - alpha) ] )

    Parameters
    ----------
    returns : np.ndarray
        1-D array of returns.
    alpha : float
        Confidence level, e.g. 0.95.

    Returns
    -------
    float
        CVaR (positive = expected loss in tail).
    """
    threshold = np.quantile(returns, 1.0 - alpha)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return float(-threshold)
    return float(-np.mean(tail))


def max_drawdown(paths: np.ndarray) -> np.ndarray:
    """Compute maximum drawdown for each path.

    Parameters
    ----------
    paths : np.ndarray
        Shape (n_paths, horizon+1).

    Returns
    -------
    np.ndarray
        1-D array of max drawdowns in [0, 1] per path.
    """
    # Running maximum along time axis
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (running_max - paths) / running_max
    return drawdowns.max(axis=1)


def prob_drawdown_exceeds(paths: np.ndarray, threshold: float) -> float:
    """Probability that max drawdown exceeds a given threshold.

    Parameters
    ----------
    paths : np.ndarray
        Shape (n_paths, horizon+1).
    threshold : float
        Drawdown level, e.g. 0.20 for 20%.

    Returns
    -------
    float
        Fraction of paths where max drawdown > threshold.
    """
    dd = max_drawdown(paths)
    return float(np.mean(dd > threshold))


def risk_report(paths: np.ndarray, alpha: float, dd_threshold: float) -> dict[str, float]:
    """Generate a summary risk report.

    Parameters
    ----------
    paths : np.ndarray
        Shape (n_paths, horizon+1).
    alpha : float
        VaR/CVaR confidence level.
    dd_threshold : float
        Drawdown threshold for exceedance probability.

    Returns
    -------
    dict[str, float]
        Risk metrics dictionary.
    """
    returns = terminal_returns(paths)
    var = value_at_risk(returns, alpha)
    cvar = conditional_var(returns, alpha)
    dd_prob = prob_drawdown_exceeds(paths, dd_threshold)
    median_return = float(np.median(returns))
    mean_return = float(np.mean(returns))

    return {
        f"VaR_{int(alpha*100)}": var,
        f"CVaR_{int(alpha*100)}": cvar,
        f"P(drawdown>{int(dd_threshold*100)}%)": dd_prob,
        "median_return": median_return,
        "mean_return": mean_return,
        "n_paths": float(len(paths)),
    }
