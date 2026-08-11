"""
Forecast accuracy metrics.

Small, single-purpose functions shared by every model in the comparison
(classical and deep learning alike), so RMSE/MAE are computed identically
everywhere rather than re-implemented per model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    """
    Root mean squared error between actual and forecast values.

    Both series are aligned on their index before computing the error,
    so mismatched dates don't silently produce wrong numbers.
    """
    actual, forecast = actual.align(forecast, join="inner")
    if len(actual) == 0:
        raise ValueError("No overlapping observations between actual and forecast.")
    errors = actual - forecast
    return float(np.sqrt(np.mean(errors**2)))


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    """Mean absolute error between actual and forecast values, index-aligned."""
    actual, forecast = actual.align(forecast, join="inner")
    if len(actual) == 0:
        raise ValueError("No overlapping observations between actual and forecast.")
    errors = actual - forecast
    return float(np.mean(np.abs(errors)))
