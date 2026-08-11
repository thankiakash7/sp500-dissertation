"""
CRSP daily stock data loader via WRDS.

This is the proposal's primary data source (Section 4.1): a survivorship-
bias-free panel of S&P 500 constituent daily returns, 2010-2024, including
delisting returns for failed/acquired/delisted companies.

Key design decisions
--------------------
1. Returns come from `crsp.dsf.ret`, NOT computed from price.
   CRSP's `ret` column is a fully adjusted total holding period return
   (splits, dividends, corporate actions already handled). `prc` is raw
   and sometimes negative (negative = bid-ask midpoint, not a real
   trade), so it's only pulled here for market-cap calculations, not
   return computation.

2. Delisting returns are merged in from `crsp.dsedelist`.
   When a company is delisted (bankruptcy, acquisition, etc.), its final
   return sits in `dsedelist.dlret`, not in `dsf.ret`. Without this
   merge, failed companies just silently disappear — that's survivorship
   bias. The proposal explicitly commits to handling this (Section 4.1).

3. Shares outstanding (`shrout`) x |price| gives market cap at each date.
   Used downstream for constructing the top-100 universe for the RMT
   experiment (Section 5.2, RQ3).

4. `shrcd in (10, 11)` filters to domestic common equity only.
   Standard filter in the literature (Fischer and Krauss, 2018; Gu et al.,
   2020) — excludes preferred stock, ADRs, closed-end funds, REITs.

5. Queries are always filtered by date range AND permno list.
   The full `crsp.dsf` table exceeds 20GB; never pull it all.

Output format
-------------
The public function `load_crsp_daily` returns a long-format DataFrame with
columns [permno, date, ticker, ret, mktcap]. The `ret` column is the raw
CRSP total return (a decimal, e.g. 0.01 means +1%), NOT a log-return.
Conversion to log-returns is left to the caller (same as `loader.py`'s
`compute_log_returns`), keeping this module responsible only for pulling
and cleaning the raw data.

WRDS credentials
----------------
On first run, `wrds.Connection()` will prompt for username and password
and offer to save them to a local .pgpass file. After that, the connection
is silent. Do NOT hardcode credentials in this file or commit them to git.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CRSPLoadError(ValueError):
    """Raised when the CRSP pull fails or returns unusable data."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_sp500_permnos(db, start: str, end: str) -> list[int]:
    """
    Pull the list of PERMNOs that were ever S&P 500 constituents between
    start and end, via the CRSP/Compustat index constituent table.

    Uses `crsp.dsp500list` — the standard CRSP table for historical S&P 500
    membership, following Fischer and Krauss (2018).
    """
    sql = f"""
        SELECT DISTINCT permno
        FROM crsp.dsp500list
        WHERE ending >= '{start}'
          AND start  <= '{end}'
    """
    result = db.raw_sql(sql, date_cols=[])
    if result.empty:
        raise CRSPLoadError(
            f"No S&P 500 constituent permnos found for {start}–{end}. "
            "Check that crsp.dsp500list is accessible under your WRDS subscription."
        )
    return result["permno"].tolist()


def _pull_dsf(db, permnos: list[int], start: str, end: str) -> pd.DataFrame:
    """
    Pull daily returns, price, and shares from crsp.dsf for the given
    permno list and date range.

    Returns a long DataFrame: [permno, date, ret, prc, shrout].
    `ret` is CRSP's total holding period return — already adjusted for
    splits and dividends. `prc` may be negative (bid-ask midpoint).
    """
    permno_str = ", ".join(str(p) for p in permnos)
    sql = f"""
        SELECT a.permno, a.date, a.ret, a.prc, a.shrout
        FROM crsp.dsf AS a
        INNER JOIN crsp.dsenames AS b
            ON  a.permno = b.permno
            AND a.date  BETWEEN b.namedt AND b.nameendt
        WHERE a.permno IN ({permno_str})
          AND a.date   BETWEEN '{start}' AND '{end}'
          AND b.shrcd  IN (10, 11)
    """
    return db.raw_sql(sql, date_cols=["date"])


def _pull_delisting_returns(db, permnos: list[int], start: str, end: str) -> pd.DataFrame:
    """
    Pull delisting returns from crsp.dsedelist.

    Returns a long DataFrame: [permno, dlstdt, dlret].
    `dlret` is the return on the delisting date. This is merged back into
    the main panel so that failed/acquired/delisted companies contribute
    their final (typically negative) return rather than disappearing silently.
    """
    permno_str = ", ".join(str(p) for p in permnos)
    sql = f"""
        SELECT permno, dlstdt, dlret
        FROM crsp.dsedelist
        WHERE permno IN ({permno_str})
          AND dlstdt BETWEEN '{start}' AND '{end}'
    """
    return db.raw_sql(sql, date_cols=["dlstdt"])


def _pull_tickers(db, permnos: list[int]) -> pd.DataFrame:
    """
    Pull the most recent ticker for each permno from crsp.dsenames.
    Used only for labelling the output — the pipeline always uses permno
    as the canonical identifier.
    """
    permno_str = ", ".join(str(p) for p in permnos)
    sql = f"""
        SELECT permno, ticker
        FROM crsp.dsenames
        WHERE permno IN ({permno_str})
          AND nameendt = (
              SELECT MAX(nameendt) FROM crsp.dsenames n2
              WHERE n2.permno = crsp.dsenames.permno
          )
    """
    return db.raw_sql(sql)


def _merge_delisting(dsf: pd.DataFrame, delist: pd.DataFrame) -> pd.DataFrame:
    """
    Merge delisting returns into the main DSF panel.

    Strategy (following standard practice in the literature):
    - For permnos that appear in dsedelist AND have a row in dsf on the
      delisting date, replace the dsf `ret` with `dlret` where `dlret` is
      not null.
    - For permnos whose final dsf row pre-dates the delisting date, append
      a synthetic row on the delisting date carrying `dlret`.

    Any stock with `dlret` still null after the merge has its return left
    as NaN — the caller's NaN-handling step will deal with this.
    """
    if delist.empty:
        return dsf

    delist = delist.rename(columns={"dlstdt": "date"}).dropna(subset=["dlret"])

    # Merge on (permno, date) — some permnos will match, some won't
    merged = dsf.merge(
        delist[["permno", "date", "dlret"]],
        on=["permno", "date"],
        how="left",
    )
    # Where we have a delisting return, override the DSF return
    mask = merged["dlret"].notna()
    merged.loc[mask, "ret"] = merged.loc[mask, "dlret"]
    merged = merged.drop(columns=["dlret"])

    # For permnos not in dsf on their delisting date, append synthetic rows
    dsf_last = dsf.groupby("permno")["date"].max().reset_index().rename(columns={"date": "last_dsf_date"})
    late_delistings = delist.merge(dsf_last, on="permno", how="left")
    late_delistings = late_delistings[late_delistings["date"] > late_delistings["last_dsf_date"]].copy()

    if not late_delistings.empty:
        # Build synthetic rows with the same shape as dsf
        synthetic = late_delistings[["permno", "date", "dlret"]].copy()
        synthetic = synthetic.rename(columns={"dlret": "ret"})
        # prc and shrout are unavailable for the delisting row — leave as NaN
        for col in ["prc", "shrout"]:
            synthetic[col] = np.nan
        merged = pd.concat([merged, synthetic], ignore_index=True)

    return merged.sort_values(["permno", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_crsp_daily(
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    db=None,
) -> pd.DataFrame:
    """
    Load a survivorship-bias-free daily return panel for S&P 500 constituents.

    Parameters
    ----------
    start, end : str
        Date range in "YYYY-MM-DD" format. Proposal scope: 2010-01-01 to
        2024-12-31.
    db : wrds.Connection, optional
        An open WRDS connection. If None, a new connection is created
        (which will prompt for credentials on first use, then use .pgpass).
        Passing an existing connection is recommended for scripts that make
        multiple calls, to avoid repeatedly opening/closing the connection.

    Returns
    -------
    pd.DataFrame with columns:
        permno   int       CRSP permanent security identifier
        date     date      trading date
        ticker   str       most recent ticker symbol (label only)
        ret      float     total holding period return, decimal
                           (NaN where CRSP has no return)
        mktcap   float     market cap in USD thousands
                           (|prc| * shrout; NaN where price unavailable)

    Notes
    -----
    - `ret` is NOT a log-return. Call `compute_log_returns` from
      `src.data.loader` on the pivoted price/return series as needed.
    - The returned DataFrame is in long format (one row per permno-date).
      Pivot to wide format for model input:
          panel = result.pivot(index='date', columns='permno', values='ret')
    - Delisting returns are already merged in. The raw CRSP `ret` has been
      replaced with `dlret` where a delisting return exists.
    """
    _owns_connection = db is None
    if _owns_connection:
        import wrds
        db = wrds.Connection()

    try:
        # Step 1: get the universe of S&P 500 permnos
        permnos = _get_sp500_permnos(db, start, end)

        # Step 2: pull daily returns from dsf
        dsf = _pull_dsf(db, permnos, start, end)
        if dsf.empty:
            raise CRSPLoadError(
                f"crsp.dsf returned no rows for the {len(permnos)} permnos "
                f"in {start}–{end}. Check the date range and your WRDS access."
            )

        # Step 3: pull and merge delisting returns
        delist = _pull_delisting_returns(db, permnos, start, end)
        dsf = _merge_delisting(dsf, delist)

        # Step 4: compute market cap BEFORE merging tickers
        # prc can be negative (bid-ask midpoint) — take abs for market cap
        dsf["mktcap"] = dsf["shrout"] * dsf["prc"].abs()
        dsf = dsf.drop(columns=["prc", "shrout"])

        # Step 5: add tickers for labelling
        tickers = _pull_tickers(db, permnos)
        dsf = dsf.merge(tickers, on="permno", how="left")
        dsf["date"] = pd.to_datetime(dsf["date"])
        dsf = dsf.sort_values(["permno", "date"]).reset_index(drop=True)

        return dsf[["permno", "date", "ticker", "ret", "mktcap"]]

    finally:
        if _owns_connection:
            db.close()
