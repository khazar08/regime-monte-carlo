"""Publication-quality plot helpers. All functions return Axes; never call plt.show()."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

REGIME_COLORS = ["#d62728", "#aec7e8", "#2ca02c", "#ff7f0e", "#9467bd"]


def plot_fan_chart(
    paths: np.ndarray,
    dates: pd.DatetimeIndex | None = None,
    percentiles: Sequence[float] = (5, 25, 50, 75, 95),
    ax: Axes | None = None,
) -> Axes:
    """Plot a fan chart of simulated price paths.

    Parameters
    ----------
    paths : np.ndarray
        Shape (n_paths, horizon+1).
    dates : pd.DatetimeIndex | None
        Optional x-axis dates.
    percentiles : Sequence[float]
        Percentiles to shade.
    ax : Axes | None
        Axes to draw on; creates one if None.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    x = dates if dates is not None else np.arange(paths.shape[1])
    pcts = np.percentile(paths, percentiles, axis=0)  # (len(percentiles), T+1)

    # Shade bands symmetrically around median
    n = len(percentiles)
    mid = n // 2
    colors = plt.cm.Blues(np.linspace(0.2, 0.6, mid))

    for i in range(mid):
        ax.fill_between(
            x,
            pcts[i],
            pcts[n - 1 - i],
            alpha=0.35,
            color=colors[i],
            label=f"P{percentiles[i]:.0f}-P{percentiles[n-1-i]:.0f}",
        )

    # Median line
    ax.plot(x, pcts[mid], color="steelblue", linewidth=2, label=f"Median (P{percentiles[mid]:.0f})")

    ax.set_xlabel("Date" if dates is not None else "Day")
    ax.set_ylabel("Price")
    ax.set_title("Simulated Price Fan Chart")
    ax.legend(fontsize=8)
    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    return ax


def plot_regimes(
    prices: pd.Series,
    states: np.ndarray,
    state_labels: list[str],
    ax: Axes | None = None,
) -> Axes:
    """Shade price history by detected regime.

    Parameters
    ----------
    prices : pd.Series
        Price series with DatetimeIndex.
    states : np.ndarray
        Integer state array, length == len(prices) - 1 (aligned to returns).
    state_labels : list[str]
        Label per state index.
    ax : Axes | None

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    ax.plot(prices.index, prices.values, color="black", linewidth=0.8, zorder=3)

    # States align to prices[1:] (since returns drop first row)
    dates = prices.index[1:]
    K = len(state_labels)

    for k in range(K):
        mask = states == k
        if not mask.any():
            continue
        color = REGIME_COLORS[k % len(REGIME_COLORS)]
        # Shade each contiguous block
        in_regime = False
        start_date = None
        first_idx = int(np.where(mask)[0][0]) if mask.any() else -1
        for i, (d, m) in enumerate(zip(dates, mask)):
            if m and not in_regime:
                start_date = d
                in_regime = True
            elif not m and in_regime:
                ax.axvspan(
                    start_date,
                    d,
                    alpha=0.25,
                    color=color,
                    label=state_labels[k] if i == first_idx + 1 else "",
                )
                in_regime = False
        if in_regime:
            ax.axvspan(start_date, dates[-1], alpha=0.25, color=color)

    # Build legend manually
    from matplotlib.patches import Patch

    handles = [
        Patch(color=REGIME_COLORS[k % len(REGIME_COLORS)], alpha=0.5, label=state_labels[k])
        for k in range(K)
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_title("Price History with Regime States")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    return ax


def plot_calibration(
    coverage_summary_df: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot empirical vs nominal coverage with 45-degree ideal line.

    Parameters
    ----------
    coverage_summary_df : pd.DataFrame
        Output from coverage_summary(); must have columns:
        nominal_level, empirical_coverage, and optionally 'model'.
    ax : Axes | None

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Ideal (y=x)")

    if "model" in coverage_summary_df.columns:
        for model, grp in coverage_summary_df.groupby("model"):
            ax.scatter(grp["nominal_level"], grp["empirical_coverage"], s=80, label=model, zorder=5)
            ax.plot(grp["nominal_level"], grp["empirical_coverage"], linewidth=1.5)
    else:
        ax.scatter(
            coverage_summary_df["nominal_level"],
            coverage_summary_df["empirical_coverage"],
            s=80,
            zorder=5,
        )
        ax.plot(
            coverage_summary_df["nominal_level"],
            coverage_summary_df["empirical_coverage"],
            linewidth=1.5,
        )

    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Calibration: Empirical vs Nominal Coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return ax


def plot_terminal_distribution(
    returns: np.ndarray,
    var: float,
    cvar: float,
    ax: Axes | None = None,
) -> Axes:
    """Histogram of terminal returns with VaR and CVaR markers.

    Parameters
    ----------
    returns : np.ndarray
        1-D terminal return array.
    var : float
        Value at Risk (positive = loss).
    cvar : float
        Conditional VaR (positive = expected loss).
    ax : Axes | None

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.hist(returns, bins=100, density=True, color="steelblue", alpha=0.7, label="Terminal returns")
    ax.axvline(-var, color="orange", linewidth=2, linestyle="--", label=f"VaR = {var:.2%}")
    ax.axvline(-cvar, color="red", linewidth=2, linestyle="--", label=f"CVaR = {cvar:.2%}")
    ax.set_xlabel("Return (S_T/S_0 - 1)")
    ax.set_ylabel("Density")
    ax.set_title("Terminal Return Distribution")
    ax.legend()
    return ax


def plot_model_comparison(
    gbm_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Side-by-side calibration comparison between GBM and regime model.

    Parameters
    ----------
    gbm_summary : pd.DataFrame
        Output from coverage_summary() for GBM.
    regime_summary : pd.DataFrame
        Output from coverage_summary() for regime model.
    ax : Axes | None

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    combined = pd.concat(
        [gbm_summary.assign(model="GBM"), regime_summary.assign(model="Regime")],
        ignore_index=True,
    )
    return plot_calibration(combined, ax=ax)


def save_figure(fig: Figure, path: str | Path) -> None:
    """Save a figure to disk, creating parent directories as needed.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    path : str | Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
