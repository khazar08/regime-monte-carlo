"""Tests for data ingestion and log return computation."""

import numpy as np
import pandas as pd

from rmc.data.ingest import compute_log_returns


def test_log_returns_math():
    prices = pd.DataFrame({"A": [100.0, 110.0, 105.0]})
    returns = compute_log_returns(prices)
    assert len(returns) == 2
    expected_r0 = np.log(110.0 / 100.0)
    expected_r1 = np.log(105.0 / 110.0)
    np.testing.assert_allclose(returns["A"].iloc[0], expected_r0, rtol=1e-10)
    np.testing.assert_allclose(returns["A"].iloc[1], expected_r1, rtol=1e-10)


def test_log_returns_no_nan():
    prices = pd.DataFrame({"X": [10.0, 20.0, 15.0, 25.0]})
    returns = compute_log_returns(prices)
    assert not returns.isnull().any().any()
    assert len(returns) == 3


def test_log_returns_shape():
    n = 100
    prices = pd.DataFrame(
        {
            "A": np.random.default_rng(0).standard_normal(n).cumsum() + 50,
            "B": np.random.default_rng(1).standard_normal(n).cumsum() + 50,
        }
    )
    returns = compute_log_returns(prices)
    assert returns.shape == (n - 1, 2)
