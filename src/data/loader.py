"""
Data loading for the S&P 500 forecasting dissertation.

This module formalises the ad-hoc pulls done in notebooks 01-03 into a single,
tested, reusable function. It currently supports Yahoo Finance (via yfinance)
as a stand-in data source while WRDS/CRSP access is being set up — CRSP is the
proposal's intended primary source (see Section 4.1).

Design note: the public function signature (`load_daily_prices`) is deliberately
source-agnostic. When CRSP access is available, a `source="crsp"` branch can be
added without changing any code that calls this function — the rest of the
pipeline (EDA, ARMA, walk-forward harness, etc.) should never need to know
which source it's getting data from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataLoadError(ValueError):
    """Raised when requested data cannot be loaded or is unusable."""


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log-returns from a wide price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format prices, one column per ticker, indexed by date.
        Expected to already be adjusted for splits/dividends if that
        matters for the use case (yfinance's "Close" with auto_adjust=True
        handles this; see `load_daily_prices`).

    Returns
    -------
    pd.DataFrame
        Same shape, log-returns, first row dropped (no prior price to
        compare against).
    """
    if prices.empty:
        raise DataLoadError("Cannot compute log-returns from an empty price DataFrame.")

    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how="all")


def load_daily_prices(
    tickers: list[str],
    start: str,
    end: str,
    source: str = "yfinance",
) -> dict[str, pd.DataFrame]:
    """
    Load daily adjusted close prices and log-returns for a list of tickers.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols, e.g. ["AAPL", "MSFT"].
    start, end : str
        Date strings in "YYYY-MM-DD" format, passed through to the source.
    source : str
        Data source to use. Only "yfinance" is implemented currently.
        CRSP/WRDS is the proposal's intended primary source (Section 4.1)
        but is not yet wired up — see module docstring.

    Returns
    -------
    dict with keys:
        "prices":      wide DataFrame of adjusted close, one column per ticker
        "log_returns": wide DataFrame of daily log-returns, one column per ticker

    Raises
    ------
    DataLoadError
        If tickers is empty, the source is unsupported, or no data is
        returned for the requested window (e.g. network unavailable,
        invalid tickers, or an empty date range).
    """
    if not tickers:
        raise DataLoadError("`tickers` must be a non-empty list.")

    if source != "yfinance":
        raise DataLoadError(
            f"Unsupported source '{source}'. Only 'yfinance' is currently implemented. "
            "CRSP/WRDS support is planned once WRDS access is set up."
        )

    import yfinance as yf  # imported lazily so tests don't require it / network access

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # adjusts for splits/dividends, matches notebook 03
        progress=False,
    )

    if raw.empty:
        raise DataLoadError(
            f"No data returned for tickers={tickers}, start={start}, end={end}. "
            "Check ticker symbols, the date range, and network access to Yahoo Finance."
        )

    # yfinance returns a MultiIndex column DataFrame for multiple tickers,
    # but a flat single-level DataFrame for a single ticker. Normalise both
    # to the same wide shape: one "Close" column per ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")

    if prices.empty:
        raise DataLoadError(
            f"Price data for tickers={tickers} was returned but is entirely empty "
            "after dropping missing rows."
        )

    log_returns = compute_log_returns(prices)

    return {"prices": prices, "log_returns": log_returns}
