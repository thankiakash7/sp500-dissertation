# Week 1 — Status note

**Date:** [fill in actual date of supervisor meeting]
**For:** Fortnightly meeting with Dr Dahlqvist

## Progress

- Research proposal finalised and submitted (806G1).
- Exploratory work done in notebooks:
  - `01_first_financial_data.ipynb` — first single-stock (AAPL) price pull
    and log-return calculation, via yfinance.
  - `02_eda_stylised_facts.ipynb` — 10-stock summary statistics table
    (mean, std. dev., autocorrelation, kurtosis, Jarque-Bera) matching
    Table 1 in the proposal. Confirms heavy tails and volatility clustering
    in the real data, as expected from Cont (2001).
  - `03_time_series_basics_and_arma.ipynb` — stationarity testing (ADF) on
    prices vs. returns, ACF/PACF plots, first ARMA(1,1) fit on AAPL with a
    train/test split, evaluated against a naive mean-forecast baseline
    (RMSE/MAE).
- Formalised the data-loading step into a tested module: `src/data/loader.py`,
  with unit tests in `tests/test_loader.py` (11 tests, all passing). This
  replaces copy-pasted notebook cells with a single reusable function and
  removes the risk of inconsistent log-return handling across modules.
- Filled in `README.md` and `requirements.txt` (previously empty placeholders).

## Blockers

- **WRDS/CRSP access is still pending.** All work so far uses Yahoo Finance
  as a stand-in. This needs to be resolved before the proposal's primary
  dataset (survivorship-bias-free CRSP panel, 2010-2024) can be built.

## Decisions needed

- Confirm WRDS access timeline / whether IT support is needed to expedite.
- Confirm whether the Week 1-2 deliverable (cleaned panel + EDA notebook)
  should proceed on yfinance data as a placeholder while WRDS access is
  sorted, or wait.

## Next steps

- Chase WRDS access.
- Build classical baseline models (random walk, ARMA, GARCH) as a tested
  `src/` module, following the same pattern as the data loader.
- Begin survivorship-bias-free index reconstruction once CRSP access is live.
