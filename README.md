# Forecasting Daily S&P 500 Stock Returns

MSc Data Science dissertation (806G1) — University of Sussex.
Candidate 307385, supervised by Dr Antoine Dahlqvist.

A walk-forward comparison of classical time-series models (mean baseline,
ARMA, AR(1)-GARCH(1,1)-t) against deep learning models (LSTM, GRU, Temporal
Fusion Transformer) for daily S&P 500 constituent return forecasting, plus a
Marchenko-Pastur random-matrix-theory denoising experiment on cross-asset
covariance features and a long-short portfolio backtest.

---

## Methodology correction (read this first)

An earlier version of this pipeline had a real methodological inconsistency:
**the classical models (ARMA, GARCH) were evaluated walk-forward (retrained
every 21 trading days), but LSTM/GRU/TFT were fit once on data through 2018
and evaluated statically through 2024** — a materially easier task, since
the deep learning models never had to prove they could adapt to unseen
regimes the way the classical models did. `README_NEXT_NOTEBOOKS.md`
explicitly warned against treating those static results as final, but an
earlier generated results table nonetheless presented all six models
together with no methodological distinction.

**This is fixed.** `step2_deep.py` and `step3_tft.py` now call the same
walk-forward engine (`src/evaluation/walk_forward_dl.py`) that
`step1_classical.py` uses: all six models retrain every 21 trading days.
See `CHANGELOG.md` for the full list of what changed and why, and
`lab_log/` for the week-by-week history of how this was found and fixed.

**What this means for the results below:** the tables and figures shipped
in this repo were generated under the *old* (static-split) methodology,
because reproducing them requires either licensed CRSP data this repo
can't ship, or tens of hours of compute this repo can't run on your
behalf (see "Runtime" below). They're kept as a reference point and are
clearly the right shape to sanity-check a rerun against, but they are
**not** the corrected walk-forward numbers. Re-run the pipeline
(`python run.py`) to get those.

---

## Status

Code is complete and tested (54 unit tests, `pytest tests/ -v`) for all
four pipeline steps: classical walk-forward, deep-learning walk-forward
(LSTM/GRU/TFT), and downstream analysis (Diebold-Mariano, Model Confidence
Set, portfolio backtests, Fama-French-Carhart alpha). What's outstanding is
a genuine full-scale run on real CRSP data — see "Running the pipeline".

Prior (static-split) findings, kept for context — **verify these against a
walk-forward rerun before citing them**:
- LSTM had the best RMSE of the six models, by a small margin.
- No model beat the mean baseline by a statistically significant margin.
- The deep learning models' backtested portfolio performance collapsed
  once realistic transaction costs were applied, driven by high daily
  turnover — see `reports/figures/portfolio_sharpe_by_cost.png`.
- Model Confidence Set testing could not statistically eliminate any of
  the six models.

## Methodology

### Classical models
- **Mean baseline** — in-sample mean return.
- **ARMA** — AR(1,0), selected via BIC search (`MAX_ARMA_ORDER=2`, a
  documented runtime trade-off — see `config.py`).
- **AR(1)-GARCH(1,1)-t** — two-step: ARMA mean equation, then GARCH(1,1)
  with Student-t innovations fit on the residuals (`src/models/baselines.py`
  explains why: the `arch` package has no native ARMA-mean GARCH support).

### Deep learning models
- **LSTM / GRU** — single-layer recurrent regressor, 60-256 day lookback
  window depending on scale, next-day return target.
- **TFT** — a lightweight, pure-PyTorch Temporal Fusion Transformer
  (positional encoding + gated residual network + multi-head attention),
  intentionally without the full quantile/covariate machinery, which adds
  no value for a single-feature point forecast and is far too slow on CPU.

### Walk-forward evaluation (all six models)
Every model retrains every 21 trading days (`RETRAIN_EVERY` in
`config.py`) using only information available up to that point — no model
sees the future. For the deep learning models specifically:
- The **first** block trains from scratch (`INITIAL_EPOCHS`).
- **Every subsequent block warm-starts** from the previous block's weights
  and fine-tunes for a small number of epochs (`FINETUNE_EPOCHS`), rather
  than fully refitting from scratch ~60 times over. This is a standard,
  citable compromise (used in online/continual-learning forecasting
  setups) — but it is *not* the same as a full refit each block, and the
  dissertation methodology section should describe it as such. See
  `src/evaluation/walk_forward_dl.py`'s module docstring for the full
  reasoning.
- Standardisation statistics (mean/std used to scale returns) are
  recomputed at each block using only data up to that block's cutoff — a
  strict improvement over the old static split, which computed them once
  and reused them for the whole 2020-2024 test period.

### Universe
All models are evaluated on the identical top-`N_STOCKS` universe by
market capitalisation as of `TRAIN_END` (`src/data/panel.py`), so
classical and deep-learning models are never compared on different stocks.

### Downstream analysis (`step4_downstream.py`)
- **Diebold-Mariano** tests (Newey-West HAC) vs. the mean baseline.
- **Model Confidence Set** (10% level, bootstrap).
- **Equal-weight long-short portfolio** (top-10/bottom-10 by predicted
  return, daily rebalance, 0/5/10/25bps transaction costs).
- **Volatility-scaled long-short portfolio** — position sizes scaled
  inversely to volatility. This now genuinely uses the AR(1)-GARCH-t
  model's own one-step-ahead conditional volatility *forecast* where
  available (previously fitted every walk-forward block and then
  discarded), falling back to realised 21-day rolling volatility
  elsewhere.
- **Fama-French-Carhart four-factor alpha** regression (downloads daily
  factors from Ken French's data library; cached to
  `data/raw/ff_carhart_daily.csv` after the first successful download).

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # the only requirements file
```

Or run the guided Mac setup script, which checks your Python version and
PyTorch/MPS availability:

```bash
bash setup.sh
```

CRSP data is not included (WRDS licensing — see "Data" below). Once you
have `data/processed/crsp_sp500_daily_corrected_2010_2024.parquet` in
place (via `src/data/crsp.py`, which needs a WRDS subscription), you're
ready to run the pipeline.

## Running the pipeline

**Smoke test first** — verifies the whole pipeline runs correctly on a
tiny slice of data in well under a minute, before committing to a
multi-hour real run:

```bash
python run.py --quick
```

This writes `*_quick`-suffixed output files everywhere, so it can never be
mistaken for or overwrite real results.

**Full run:**

```bash
python run.py                # all four steps
python run.py --skip-tft     # skip TFT (faster, 5-model results)
python run.py --step 1       # just one step, e.g. classical only
python run.py --step 4       # downstream analysis only (after 1-3 are done)
```

### Runtime

Step 1 (classical) takes minutes. **Steps 2 and 3 (LSTM/GRU/TFT) are
genuinely walk-forward now, which is a multi-hour job per model** at full
scale (100 stocks, 252-day lookback, ~60 retrain blocks) on a laptop CPU —
see the methodology note above for why. Both steps print a pre-flight
time estimate (measured on your actual machine) before committing to the
full run:

```
LSTM: 60 retrain blocks, ~0.18s/batch, projected 4.2h total on this machine.
```

Both steps checkpoint after every retrain block, so an interrupted run can
simply be re-run — it picks up exactly where it left off rather than
starting over. An Apple Silicon Mac will use MPS automatically if
available (`config.DEVICE`).

## Testing

```bash
pytest tests/ -v
```

54 tests, covering data loading (CRSP + yfinance), classical models,
metrics, and the walk-forward deep-learning engine (including TFT support,
checkpoint/resume behaviour, and the runtime estimator).

## Project structure

```
config.py               Single source of truth: dates, universe size, hyperparameters
src/
  data/                 CRSP loading (crsp.py), yfinance fallback (loader.py),
                         shared universe selection (panel.py)
  models/                Model architectures shared by step2/step3/walk_forward_dl (deep.py),
                         classical models (baselines.py)
  evaluation/            Walk-forward DL engine (walk_forward_dl.py), forecast metrics
notebooks/               Step-by-step analysis notebooks (EDA through portfolio backtest)
tests/                   pytest unit tests for src/
data/                    raw / interim / processed data (not committed, see below)
reports/                 Generated tables, figures, and model checkpoints
dissertation_materials/  Copies of tables/figures for the write-up
lab_log/                 Weekly status notes (one per supervisor meeting)
scripts/                 Figure generation (generate_dissertation_figures.py)
step1-4_*.py             Pipeline entry-point scripts (classical, deep, TFT, downstream)
run.py                   Top-level pipeline orchestrator
CHANGELOG.md             What changed in the methodology fix, and why
```

## Data

No data is committed to this repository. **CRSP data must never be
redistributed** per WRDS licensing terms — only code and setup
instructions live here. `data/raw`, `data/interim`, and `data/processed`
are local working directories, populated by running the pipeline locally.
If you're setting this up fresh, see `src/data/crsp.py`'s module docstring
for the WRDS pull, and `.gitignore` for what's (correctly) excluded.

This was actually violated once: a data-audit notebook was writing real
CRSP rows into a "report table" CSV. It's fixed — see `CHANGELOG.md`'s
first entry — but it's a reminder to double-check any new `reports/`
output before committing it: "report table" doesn't automatically mean
"safe to commit" if the underlying values are the licensed data itself
rather than a derived/aggregate statistic.

## Notebooks vs. `src/`

`notebooks/` holds the step-by-step analysis work; `README_NEXT_NOTEBOOKS.md`
has the run order and setup notes for those. `src/` holds the code that's
been pulled out, cleaned up, and tested — data loading, models, evaluation.
Anything reused across steps, or that needs to be correct, lives in `src/`
with matching tests in `tests/`. Note: notebooks 09 and 10 (LSTM/GRU, TFT)
still reflect the pre-fix static-split exploration; the authoritative,
corrected implementation is `step2_deep.py` / `step3_tft.py` plus
`src/evaluation/walk_forward_dl.py`, not those notebooks.

## Research questions

- **RQ1 — Do deep learning models outperform classical approaches?**
  Prior (static-split) result: differences in RMSE were small and not
  statistically significant (Diebold-Mariano). Needs re-verification
  under the walk-forward DL methodology.
- **RQ2 — Does GARCH improve on ARMA?** ARMA and AR(1)-GARCH-t produced
  near-identical forecasts — volatility modelling added no directional
  forecasting value, though it *is* now used for volatility-scaled
  position sizing (see Methodology above).
- **RQ3 — Does covariance denoising help?** RMT analysis
  (`reports/figures/rmt_eigenvalue_spectrum.png`) found only 3-7% of
  eigenvalues carry signal above the Marchenko-Pastur noise edge across
  100/200/300-stock universes — strong support for denoising before using
  correlations downstream.
- **RQ4 — Do forecasting improvements create trading value?** Prior
  result: no — small RMSE improvements did not survive realistic
  transaction costs once turnover was accounted for. Needs
  re-verification under the walk-forward DL methodology, since the old
  static-split models never had to pay the cost of trading a model that
  was staying current — genuine walk-forward retraining will very likely
  change the daily turnover profile as well as the forecasts.
