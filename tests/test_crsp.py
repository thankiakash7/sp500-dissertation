"""
Unit tests for src.data.crsp.

All tests mock the WRDS connection — no live database calls.
The mock returns DataFrames shaped exactly like real CRSP query results,
so the tests exercise all the real logic (delisting merge, market cap
computation, ticker labelling, NaN handling) without needing WRDS access.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np
import pytest

from src.data.crsp import (
    CRSPLoadError,
    _merge_delisting,
    load_crsp_daily,
)


# ---------------------------------------------------------------------------
# Fixtures: realistic fake CRSP data
# ---------------------------------------------------------------------------

def _fake_permnos():
    return pd.DataFrame({"permno": [10001, 10002, 10003]})


def _fake_dsf():
    """Three permnos, five trading days each."""
    dates = pd.bdate_range("2020-01-02", periods=5)
    rows = []
    for permno in [10001, 10002, 10003]:
        for d in dates:
            rows.append({
                "permno": permno,
                "date": d,
                "ret": 0.01,
                "prc": 100.0,
                "shrout": 1000.0,
            })
    return pd.DataFrame(rows)


def _fake_delist_empty():
    return pd.DataFrame(columns=["permno", "dlstdt", "dlret"])


def _fake_delist_with_return():
    """permno 10002 delisted on 2020-01-08 (within DSF range) with a -30% return."""
    return pd.DataFrame([{
        "permno": 10002,
        "dlstdt": pd.Timestamp("2020-01-08"),
        "dlret": -0.30,
    }])


def _fake_delist_late():
    """permno 10003 delisted after its last DSF date — synthetic row needed."""
    return pd.DataFrame([{
        "permno": 10003,
        "dlstdt": pd.Timestamp("2020-01-15"),
        "dlret": -0.50,
    }])


def _fake_tickers():
    return pd.DataFrame([
        {"permno": 10001, "ticker": "AAAA"},
        {"permno": 10002, "ticker": "BBBB"},
        {"permno": 10003, "ticker": "CCCC"},
    ])


def _build_mock_db(dsf=None, delist=None):
    """Build a mock wrds.Connection whose raw_sql returns sensible fake data."""
    db = MagicMock()

    def raw_sql_side_effect(sql, **kwargs):
        sql_lower = sql.lower()
        if "dsp500list" in sql_lower:
            return _fake_permnos()
        elif "dsedelist" in sql_lower:
            return delist if delist is not None else _fake_delist_empty()
        elif "dsenames" in sql_lower and "max(nameendt)" in sql_lower:
            # _pull_tickers uses a subquery with MAX(nameendt)
            return _fake_tickers()
        elif "dsf" in sql_lower:
            # _pull_dsf joins dsenames for shrcd filter — return full dsf shape
            return dsf if dsf is not None else _fake_dsf()
        else:
            return pd.DataFrame()

    db.raw_sql.side_effect = raw_sql_side_effect
    return db


# ---------------------------------------------------------------------------
# _merge_delisting (internal, but critical logic)
# ---------------------------------------------------------------------------

def test_merge_delisting_no_delistings_unchanged():
    dsf = _fake_dsf()
    result = _merge_delisting(dsf, _fake_delist_empty())
    # No rows added or removed
    assert len(result) == len(dsf)


def test_merge_delisting_overrides_ret_on_matching_date():
    """When dlstdt matches an existing DSF row, that row's ret is replaced."""
    dsf = _fake_dsf()
    delist = _fake_delist_with_return()
    result = _merge_delisting(dsf, delist)

    row = result[(result["permno"] == 10002) & (result["date"] == pd.Timestamp("2020-01-08"))]
    assert len(row) == 1
    assert row["ret"].iloc[0] == pytest.approx(-0.30)


def test_merge_delisting_does_not_affect_other_permnos():
    """Only permno 10002 should have a changed return; 10001 and 10003 untouched."""
    dsf = _fake_dsf()
    delist = _fake_delist_with_return()
    result = _merge_delisting(dsf, delist)

    other = result[result["permno"] == 10001]
    assert (other["ret"] == 0.01).all()


def test_merge_delisting_appends_synthetic_row_for_late_delist():
    """When dlstdt is after the last DSF date, a synthetic row is appended."""
    dsf = _fake_dsf()
    delist = _fake_delist_late()
    result = _merge_delisting(dsf, delist)

    # Should have one extra row for permno 10003 on 2020-01-15
    synthetic = result[
        (result["permno"] == 10003) & (result["date"] == pd.Timestamp("2020-01-15"))
    ]
    assert len(synthetic) == 1
    assert synthetic["ret"].iloc[0] == pytest.approx(-0.50)


def test_merge_delisting_null_dlret_is_ignored():
    """A delisting row with dlret=NaN should not override the DSF return."""
    dsf = _fake_dsf()
    delist = pd.DataFrame([{
        "permno": 10001,
        "dlstdt": pd.Timestamp("2020-01-06"),
        "dlret": np.nan,
    }])
    result = _merge_delisting(dsf, delist)
    row = result[(result["permno"] == 10001) & (result["date"] == pd.Timestamp("2020-01-06"))]
    assert row["ret"].iloc[0] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# load_crsp_daily (full pipeline, mocked)
# ---------------------------------------------------------------------------

def test_load_crsp_daily_output_columns():
    db = _build_mock_db()
    result = load_crsp_daily(start="2020-01-01", end="2020-01-10", db=db)
    assert set(result.columns) == {"permno", "date", "ticker", "ret", "mktcap"}


def test_load_crsp_daily_mktcap_computed_correctly():
    """mktcap = |prc| * shrout. With prc=100, shrout=1000: mktcap=100000."""
    db = _build_mock_db()
    result = load_crsp_daily(start="2020-01-01", end="2020-01-10", db=db)
    assert result["mktcap"].dropna().tolist() == pytest.approx([100_000.0] * len(result["mktcap"].dropna()))


def test_load_crsp_daily_mktcap_uses_abs_price():
    """Negative prc (CRSP bid-ask midpoint convention) should still give positive mktcap."""
    dsf = _fake_dsf()
    dsf["prc"] = -100.0  # CRSP uses negative to flag bid-ask midpoints
    db = _build_mock_db(dsf=dsf)
    result = load_crsp_daily(start="2020-01-01", end="2020-01-10", db=db)
    assert (result["mktcap"] > 0).all()


def test_load_crsp_daily_tickers_labelled():
    db = _build_mock_db()
    result = load_crsp_daily(start="2020-01-01", end="2020-01-10", db=db)
    assert set(result["ticker"].dropna().unique()) == {"AAAA", "BBBB", "CCCC"}


def test_load_crsp_daily_raises_on_empty_dsf():
    db = _build_mock_db(dsf=pd.DataFrame(columns=["permno", "date", "ret", "prc", "shrout"]))
    with pytest.raises(CRSPLoadError, match="no rows"):
        load_crsp_daily(start="2020-01-01", end="2020-01-10", db=db)


def test_load_crsp_daily_closes_connection_it_opened():
    """If load_crsp_daily opens its own connection, it must close it."""
    mock_conn = _build_mock_db()
    with patch("wrds.Connection", return_value=mock_conn):
        load_crsp_daily(start="2020-01-01", end="2020-01-10")
    mock_conn.close.assert_called_once()


def test_load_crsp_daily_does_not_close_caller_supplied_connection():
    """If the caller supplies a connection, load_crsp_daily must not close it."""
    db = _build_mock_db()
    load_crsp_daily(start="2020-01-01", end="2020-01-10", db=db)
    db.close.assert_not_called()
