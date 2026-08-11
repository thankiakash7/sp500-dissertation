"""
Unit tests for src.data.loader.

These tests use synthetic price data and do NOT hit the network. That's
deliberate: tests should pass reliably regardless of whether Yahoo Finance
is reachable from wherever they're run (e.g. CI, a sandboxed environment,
or just a flaky connection). Network-dependent behaviour (the actual
yfinance call) is exercised separately via mocking in
test_load_daily_prices_network_mocked.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoadError, compute_log_returns, load_daily_prices


# ---------------------------------------------------------------------------
# compute_log_returns
# ---------------------------------------------------------------------------

def test_compute_log_returns_known_values():
    """A simple, hand-checkable case: price doubling should give a log-return of ln(2)."""
    prices = pd.DataFrame(
        {"AAPL": [100.0, 200.0, 200.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    result = compute_log_returns(prices)

    assert len(result) == 2  # first row dropped, no prior price
    assert math.isclose(result["AAPL"].iloc[0], math.log(2), rel_tol=1e-9)
    assert math.isclose(result["AAPL"].iloc[1], 0.0, abs_tol=1e-9)  # no change


def test_compute_log_returns_multiple_tickers_independent():
    """Each column's log-returns should depend only on that column's own prices."""
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0],
            "MSFT": [50.0, 50.0, 25.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    result = compute_log_returns(prices)

    assert math.isclose(result["AAPL"].iloc[0], math.log(110 / 100), rel_tol=1e-9)
    assert math.isclose(result["MSFT"].iloc[0], 0.0, abs_tol=1e-9)
    assert math.isclose(result["MSFT"].iloc[1], math.log(25 / 50), rel_tol=1e-9)


def test_compute_log_returns_drops_first_row():
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 99.0, 103.0]},
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    result = compute_log_returns(prices)
    assert len(result) == len(prices) - 1
    assert prices.index[0] not in result.index


def test_compute_log_returns_rejects_empty_dataframe():
    with pytest.raises(DataLoadError):
        compute_log_returns(pd.DataFrame())


def test_compute_log_returns_handles_nan_in_one_ticker_only():
    """
    A missing price in one ticker (e.g. a trading halt) should propagate as NaN
    for that ticker, without dropping other tickers' valid data on the same date.
    `dropna(how="all")` only removes rows where every ticker is NaN.
    """
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, np.nan, 102.0],
            "MSFT": [50.0, 51.0, 52.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    result = compute_log_returns(prices)

    assert len(result) == 2  # first row dropped (no prior price for either ticker)
    assert pd.isna(result["AAPL"].iloc[0])  # AAPL gap propagates as NaN
    assert pd.isna(result["AAPL"].iloc[1])  # AAPL has no recovery price to compare from
    assert not result["MSFT"].isna().any()  # MSFT's valid data is preserved throughout


def test_compute_log_returns_single_ticker_nan_drops_row():
    """
    With only one ticker, a NaN price makes that row entirely NaN, so
    dropna(how="all") removes it. This differs from the multi-ticker case
    above only because "all columns NaN" and "one column NaN" coincide
    when there's just one column.
    """
    prices = pd.DataFrame(
        {"AAPL": [100.0, np.nan, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    result = compute_log_returns(prices)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# load_daily_prices — input validation (no network needed)
# ---------------------------------------------------------------------------

def test_load_daily_prices_rejects_empty_ticker_list():
    with pytest.raises(DataLoadError, match="non-empty list"):
        load_daily_prices([], start="2020-01-01", end="2020-12-31")


def test_load_daily_prices_rejects_unsupported_source():
    with pytest.raises(DataLoadError, match="Unsupported source"):
        load_daily_prices(["AAPL"], start="2020-01-01", end="2020-12-31", source="crsp")


# ---------------------------------------------------------------------------
# load_daily_prices — network behaviour, mocked
# ---------------------------------------------------------------------------

def _fake_multi_ticker_download(*args, **kwargs):
    """Mimic yfinance's MultiIndex-column return shape for >1 ticker."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    columns = pd.MultiIndex.from_product(
        [["Close", "Open"], ["AAPL", "MSFT"]], names=["Price", "Ticker"]
    )
    data = np.array(
        [
            [100, 50, 99, 49],
            [101, 51, 100, 50],
            [99, 50, 101, 51],
            [102, 52, 99, 50],
            [103, 53, 102, 52],
        ],
        dtype=float,
    )
    return pd.DataFrame(data, index=dates, columns=columns)


def _fake_single_ticker_download(*args, **kwargs):
    """Mimic yfinance's flat-column return shape for exactly 1 ticker."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "Open": [99.0, 100.0, 101.0, 99.0],
            "Close": [100.0, 101.0, 99.0, 102.0],
        },
        index=dates,
    )


def _fake_empty_download(*args, **kwargs):
    return pd.DataFrame()


def test_load_daily_prices_multi_ticker_shapes_correctly():
    with patch("yfinance.download", side_effect=_fake_multi_ticker_download):
        result = load_daily_prices(["AAPL", "MSFT"], start="2024-01-01", end="2024-01-10")

    assert set(result.keys()) == {"prices", "log_returns"}
    assert list(result["prices"].columns) == ["AAPL", "MSFT"]
    assert len(result["prices"]) == 5
    assert len(result["log_returns"]) == 4  # one fewer, first row dropped


def test_load_daily_prices_single_ticker_shapes_correctly():
    with patch("yfinance.download", side_effect=_fake_single_ticker_download):
        result = load_daily_prices(["AAPL"], start="2024-01-01", end="2024-01-10")

    assert list(result["prices"].columns) == ["AAPL"]
    assert len(result["prices"]) == 4
    assert math.isclose(result["log_returns"]["AAPL"].iloc[0], math.log(101 / 100), rel_tol=1e-9)


def test_load_daily_prices_raises_on_empty_response():
    """Covers the real failure seen when the data source is unreachable
    (e.g. network egress blocked) or tickers/date range are invalid."""
    with patch("yfinance.download", side_effect=_fake_empty_download):
        with pytest.raises(DataLoadError, match="No data returned"):
            load_daily_prices(["AAPL"], start="2024-01-01", end="2024-01-10")
