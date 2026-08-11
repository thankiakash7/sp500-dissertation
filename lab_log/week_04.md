# Week 4 Lab Log

## What happened this week

Ran a full audit of the pipeline before the supervisor meeting. Found that
three notebooks (08, 09, 10) were all still set to `QUICK_MODE = True` from
early development — a pilot configuration (5-10 stocks, short test windows,
low epoch counts) that was never switched to the full-scale design described
in the proposal.

## Fixed this week

- **Notebook 08 (classical walk-forward):** was QUICK_MODE (5 stocks, Q1 2020
  only). Re-ran at full scale: the same 10 stocks used by the LSTM/GRU/TFT
  notebooks, full 2020-2024 test period (1,258 trading days, 12,580
  stock-day observations per model). MAX_ARMA_ORDER kept at 2 (not 5) to keep
  runtime reasonable; this is a deliberate, documented trade-off worth
  flagging to Dahlqvist.
- **Notebook 11 (RMT):** executed for the first time. 100/200/300-stock
  universes at the 2019-12-31 anchor, 756-day window. Results: q = 0.13 /
  0.26 / 0.40, MP upper edge = 1.86 / 2.29 / 2.66. Only 7-10 eigenvalues out
  of 100-300 carry signal above the noise edge — strong empirical support
  for denoising before using correlations downstream.
- **Notebook 12 (DM/MCS):** re-run on the corrected, full-coverage data.
  GRU and LSTM beat the mean-return baseline at the 10% significance level
  (p = 0.084 and 0.100 respectively, HAC-robust). The 90% Model Confidence
  Set still includes all six models — no model is statistically dominant
  once multiple testing is accounted for. This is a defensible, honest
  result, not a failure.

## Found, not yet fixed: portfolio backtest is structurally empty

Notebook 13's long-short strategy requires 20 stocks (top 10 long, bottom 10
short) per day. Notebooks 09 and 10 (LSTM/GRU/TFT) are *also* still in
QUICK_MODE — trained on only 10 stocks total, with a 60-day lookback and 5
epochs instead of the proposal's 252-day/50-epoch design. With only 10
stocks, the top-10/bottom-10 groups overlap completely, so every weight is
zero and the backtest returns nothing.

This is the single most important open item. It requires retraining LSTM,
GRU, and TFT at full scale (100 stocks, full lookback, full epochs) — a
genuine multi-hour job that needs to run locally, not in a constrained
sandbox.

## Decisions needed from Dahlqvist

1. Is N=100 stocks (matching the RMT universe size) the right scale for the
   final LSTM/GRU/TFT runs, or should this go straight to N=300/500 to match
   the dissertation's "S&P 500" framing?
2. Is MAX_ARMA_ORDER=2 acceptable for the full-scale classical baseline, or
   should order 5 be re-tested now that the universe/period are fixed (this
   will roughly double runtime)?
3. Given the TFT training time, should TFT be limited to a smaller universe
   than LSTM/GRU as a pragmatic scope decision, with this explicitly stated
   as a limitation in the dissertation?

## Next steps

1. Switch QUICK_MODE to False in notebooks 09 and 10, retrain on full
   compute (own machine), re-save predictions.
2. Re-run notebook 12 with the expanded universe.
3. Re-run notebook 13 — this is the first point at which the portfolio
   backtest will produce real, non-zero numbers.

## Update: response to "can you solve everything?"

Triaged every open item against what's actually solvable in an automated
sandbox vs. what genuinely needs local hardware, a network resource I can't
reach, or a decision only the supervisor/student should make.

**Solved this session:**
- Figures from notebooks 11 and 13 (RMT eigenvalue spectrum, portfolio
  cumulative returns) are now saved as standalone PNGs in
  `reports/figures/`, not just inline notebook output.
- GARCH-in-mean implemented in `src/models/baselines.py`
  (`fit_garch_in_mean`), using iterative backfitting since the `arch`
  package has no native joint-MLE GARCH-M support (documented in the
  function's docstring, same style as the existing ARMA+GARCH note). Ran on
  all 10 stocks over 2010-2018: no stock shows a significant risk-premium
  coefficient (all p > 0.11) — a clean, citable null result. 3 new unit
  tests added; full suite now 46/46 passing.

**Confirmed infeasible here, not just "not done yet":**
- Fama-French/Carhart data: both Kenneth French's site and FRED are outside
  this sandbox's network allowlist. No reliable substitute exists — using
  an unofficial GitHub mirror risks silently feeding wrong numbers into the
  alpha regression, so this was not attempted. Needs a manual download on
  the student's own machine (2-minute job), or the file emailed in for
  processing.
- Full-scale LSTM/GRU/TFT retrain (100 stocks, 252-day lookback, 50 epochs):
  timed at 4.3s/batch, ~781 batches/epoch on this single-core sandbox, i.e.
  ~46.5 hours for LSTM alone, before GRU or TFT. This must run locally.

**Deliberately not auto-decided:**
- Whether to convert LSTM/GRU/TFT from a static train/test split to genuine
  walk-forward retraining is a scope/workload trade-off for Dahlqvist to
  weigh in on, not something to silently rewrite.
- The dissertation write-up itself (Module 13) is the assessed, individually
  authored part of this work and was left for the student to write, with
  structure/examples available on request rather than full drafted chapters.

## Walk-forward retraining for LSTM/GRU: code finished, execution pending

Added `src/evaluation/walk_forward_dl.py` — retrains every 21 trading days
like notebook 08, warm-starting from the previous block's weights and
fine-tuning for a few epochs per block rather than refitting from scratch
(full refits ~60 times over would turn an already multi-hour single fit
into days; documented as an explicit, citable compromise in the module
docstring). 4 new unit tests pass; smoke-tested end-to-end inside the real
notebook 09 on a small window with real CRSP data — works correctly.

A new gated cell at the end of notebook 09 (`RUN_WALK_FORWARD_DL`) runs this
for real. Not run at full scale here: same compute ceiling as the plain
retrain (this sandbox: ~4.3s/batch on 1 CPU core). This is now finished as
code; only execution remains, and that execution needs to happen locally.

## Status: coding is as complete as it can be in this environment

Everything that can be coded, tested, and run without local GPU/multi-core
compute or external network access has been done. Three things remain,
all requiring resources only available locally:

1. Flip `QUICK_MODE = False` in notebooks 09 and 10, run on local hardware.
2. Flip `RUN_WALK_FORWARD_DL = True` in notebook 09 (and build the TFT
   equivalent if walk-forward TFT is wanted), run on local hardware.
3. Download Fama-French/Carhart data locally, run notebook 13's
   `RUN_FACTOR_ALPHA` section.

Test suite: 50/50 passing.
