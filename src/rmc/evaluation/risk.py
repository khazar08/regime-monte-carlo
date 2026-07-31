from __future__ import annotations
import numpy as np


def terminal_returns(paths: np.ndarray) -> np.ndarray:
    return paths[:, -1] / paths[:, 0] - 1.0


def value_at_risk(returns: np.ndarray, alpha: float) -> float:
    return float(-np.quantile(returns, 1.0 - alpha))


def conditional_var(returns: np.ndarray, alpha: float) -> float:

    threshold = np.quantile(returns, 1.0 - alpha)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return float(-threshold)
    return float(-np.mean(tail))


def max_drawdown(paths: np.ndarray) -> np.ndarray:
    
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (running_max - paths) / running_max
    return drawdowns.max(axis=1)


def prob_drawdown_exceeds(paths: np.ndarray, threshold: float) -> float:
    
    dd = max_drawdown(paths)
    return float(np.mean(dd > threshold))


def risk_report(paths: np.ndarray, alpha: float, dd_threshold: float) -> dict[str, float]:


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
