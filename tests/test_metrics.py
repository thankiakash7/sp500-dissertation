"""Unit tests for src.evaluation.metrics."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.evaluation.metrics import mae, rmse


def test_rmse_zero_when_forecast_is_perfect():
    actual = pd.Series([1.0, 2.0, 3.0])
    forecast = pd.Series([1.0, 2.0, 3.0])
    assert rmse(actual, forecast) == 0.0


def test_mae_zero_when_forecast_is_perfect():
    actual = pd.Series([1.0, 2.0, 3.0])
    forecast = pd.Series([1.0, 2.0, 3.0])
    assert mae(actual, forecast) == 0.0


def test_rmse_known_value():
    # errors: [1, -1, 1] -> squared: [1, 1, 1] -> mean 1 -> sqrt 1
    actual = pd.Series([1.0, 1.0, 1.0])
    forecast = pd.Series([0.0, 2.0, 0.0])
    assert math.isclose(rmse(actual, forecast), 1.0, rel_tol=1e-9)


def test_mae_known_value():
    # errors: [2, -2, 2] -> abs: [2, 2, 2] -> mean 2
    actual = pd.Series([2.0, 2.0, 2.0])
    forecast = pd.Series([0.0, 4.0, 0.0])
    assert math.isclose(mae(actual, forecast), 2.0, rel_tol=1e-9)


def test_rmse_penalizes_large_errors_more_than_mae():
    """RMSE should be more sensitive to outliers than MAE (squares vs. abs)."""
    actual = pd.Series([0.0, 0.0, 0.0, 0.0])
    forecast = pd.Series([1.0, 1.0, 1.0, 10.0])  # one big outlier
    r = rmse(actual, forecast)
    m = mae(actual, forecast)
    assert r > m


def test_metrics_align_on_index_ignoring_mismatched_dates():
    """A forecast and actual series with only partially overlapping dates
    should be compared only on the overlap, not raise or silently misalign."""
    actual = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3))
    forecast = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2024-01-02", periods=4))
    # overlap is 2024-01-02 and 2024-01-03: actual=[2,3], forecast=[1,2]
    # errors = [1, 1] -> rmse = 1, mae = 1
    assert math.isclose(rmse(actual, forecast), 1.0, rel_tol=1e-9)
    assert math.isclose(mae(actual, forecast), 1.0, rel_tol=1e-9)


def test_rmse_raises_on_no_overlap():
    actual = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
    forecast = pd.Series([1.0, 2.0], index=pd.date_range("2030-01-01", periods=2))
    with pytest.raises(ValueError, match="No overlapping"):
        rmse(actual, forecast)


def test_mae_raises_on_no_overlap():
    actual = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
    forecast = pd.Series([1.0, 2.0], index=pd.date_range("2030-01-01", periods=2))
    with pytest.raises(ValueError, match="No overlapping"):
        mae(actual, forecast)
