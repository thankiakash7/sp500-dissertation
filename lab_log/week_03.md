# Week 3 — Status note

**Date:** [fill in actual date of supervisor meeting]
**For:** Fortnightly meeting with Dr Dahlqvist

## Progress

- WRDS/CRSP access confirmed and live.
- Built `src/data/crsp.py` — the proposal's primary data source, replacing
  Yahoo Finance as a stand-in:
  - Pulls S&P 500 constituent universe from `crsp.dsp500list` (historical
    membership, not just current constituents — avoids survivorship bias).
  - Queries `crsp.dsf` filtered to `shrcd in (10,11)` (domestic common
    equity only) joined with `crsp.dsenames` for share-code filtering.
  - Uses `crsp.dsf.ret` directly (CRSP's total holding period return,
    already adjusted for splits/dividends) — never computes returns from
    raw price.
  - Merges delisting returns from `crsp.dsedelist` — firms that fail,
    delist, or get acquired contribute their final return rather than
    silently disappearing (the survivorship-bias fix the proposal commits
    to in Section 4.1).
  - Computes market cap as `|prc| * shrout` (absolute value because CRSP
    uses negative prc to flag bid-ask midpoints rather than real trades).
  - Returns a long-format DataFrame: permno, date, ticker, ret, mktcap.
  - Caller supplies an open connection or the function opens/closes one
    itself — safe for both one-off and batch usage.
- Built `tests/test_crsp.py` — 12 tests, all passing, fully mocked (no
  live DB calls in the test suite). Caught two real bugs during development:
  a column collision from merging tickers before dropping prc/shrout, and a
  pytest.approx Series assertion style issue.
- Added `notebooks/04_crsp_data_pull.ipynb` — step-by-step verification
  notebook: confirms CRSP schema access, runs a 1-year test pull with
  sanity checks, checks delisting rows are present, then provides the
  commented-out full 2010-2024 pull to run once test pull looks correct.
- Updated `requirements.txt` with `wrds` and `pyarrow` (parquet saving).

## Blockers

None — WRDS access live, pipeline works end-to-end.

## Next steps

- Run `notebooks/04_crsp_data_pull.ipynb`: confirm test pull, then
  execute the full 2010-2024 pull and save to
  `data/processed/crsp_daily_2010_2024.parquet`.
- Wire the saved parquet into the walk-forward pipeline (replaces the
  Yahoo Finance calls in existing notebooks).
- Build LSTM module.
