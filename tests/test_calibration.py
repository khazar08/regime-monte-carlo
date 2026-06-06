"""Tests for calibration backtest."""

import numpy as np
import pandas as pd

from rmc.evaluation.calibration import coverage_summary, predicted_intervals


def test_predicted_intervals_bounds():
    rng = np.random.default_rng(0)
    prices = rng.lognormal(0, 0.1, 10_000) * 100
    intervals = predicted_intervals(prices, [0.50, 0.80, 0.95])
    for level, (lo, hi) in intervals.items():
        assert lo < hi
        assert lo > 0


def test_predicted_intervals_containment():
    prices = np.linspace(50, 150, 1000)
    intervals = predicted_intervals(prices, [0.80])
    lo, hi = intervals[0.80]
    inside = np.sum((prices >= lo) & (prices <= hi)) / len(prices)
    assert abs(inside - 0.80) < 0.05


def test_coverage_summary_structure():
    df = pd.DataFrame(
        {
            "nominal_level": [0.95, 0.95, 0.95, 0.80, 0.80],
            "covered": [True, True, False, True, True],
            "anchor_date": pd.date_range("2020-01-01", periods=5, freq="ME"),
            "realized": [100.0] * 5,
            "lo": [90.0] * 5,
            "hi": [110.0] * 5,
        }
    )
    summary = coverage_summary(df)
    assert set(summary.columns) >= {"nominal_level", "empirical_coverage", "n_anchors"}
    row_95 = summary[summary["nominal_level"] == 0.95].iloc[0]
    np.testing.assert_allclose(row_95["empirical_coverage"], 2 / 3, atol=1e-6)
