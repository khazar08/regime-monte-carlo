from __future__ import annotations
import logging
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def predicted_intervals(
    terminal_prices: np.ndarray,
    nominal_levels: list[float],
) -> dict[float, tuple[float, float]]:
    intervals: dict[float, tuple[float, float]] = {}
    for level in nominal_levels:
        tail = (1.0 - level) / 2.0
        lo = float(np.quantile(terminal_prices, tail))
        hi = float(np.quantile(terminal_prices, 1.0 - tail))
        intervals[level] = (lo, hi)
    return intervals


def walk_forward_coverage(
    prices: pd.Series,
    model_kind: str,
    config: Any,
) -> pd.DataFrame:

    
    from rmc.models.gbm import fit_gbm, simulate_gbm
    from rmc.models.regime import fit_regime
    from rmc.models.simulate import simulate_regime

    bt = config.backtest
    sim = config.simulation
    reg = config.regime

    trailing = bt.trailing_window
    horizon = bt.horizon_days
    step = bt.anchor_step
    levels = bt.nominal_levels
    n_paths = sim.n_paths
    dt = sim.dt
    seed = sim.seed

    price_arr = prices.values
    dates = prices.index
    N = len(price_arr)

    records = []
    anchor_indices = range(trailing, N - horizon, step)

    for idx in anchor_indices:
        anchor_date = dates[idx]
        window_returns = np.log(
            price_arr[idx - trailing + 1 : idx + 1] / price_arr[idx - trailing : idx]
        )
        s0 = float(price_arr[idx])
        realized = float(price_arr[idx + horizon])

        rng = np.random.default_rng(seed + idx)

        try:
            if model_kind == "gbm":
                params = fit_gbm(window_returns)
                paths = simulate_gbm(s0, params, horizon, n_paths, dt, rng)
            elif model_kind == "regime":
                params, _ = fit_regime(window_returns, reg.n_states, reg.n_iter, rng)
                paths = simulate_regime(s0, params, horizon, n_paths, reg.init_state, rng)
            else:
                raise ValueError(f"Unknown model_kind: {model_kind}")
        except Exception as exc:
            logger.warning("Skipping anchor %s: %s", anchor_date, exc)
            continue

        terminal = paths[:, -1]
        intervals = predicted_intervals(terminal, levels)

        for level, (lo, hi) in intervals.items():
            records.append(
                {
                    "anchor_date": anchor_date,
                    "nominal_level": level,
                    "covered": lo <= realized <= hi,
                    "realized": realized,
                    "lo": lo,
                    "hi": hi,
                }
            )

    return pd.DataFrame(records)


def coverage_summary(coverage_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        coverage_df.groupby("nominal_level")["covered"]
        .agg(empirical_coverage="mean", n_anchors="count")
        .reset_index()
    )
    grouped["ideal"] = grouped["nominal_level"]
    return grouped
