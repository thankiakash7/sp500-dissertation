# Week 2 — Status note

**Date:** [fill in actual date of supervisor meeting]
**For:** Fortnightly meeting with Dr Dahlqvist

## Progress

- Registered for WRDS access (CRSP route via the Business School). Still
  pending approval — see Blockers.
- Built the classical baselines module, following the same tested pattern
  as the Week 1 data loader:
  - `src/evaluation/metrics.py` — RMSE and MAE, shared by every model
    (classical and deep learning) so accuracy is measured identically
    everywhere. 8 tests, including an index-misalignment guard.
  - `src/models/baselines.py` — three models from proposal Section 5.2:
    - `random_walk_forecast` — the "must beat" null baseline.
    - `fit_arma` — ARMA(p,q), either a fixed order or BIC-selected over
      p, q in {0..5} as specified in the proposal.
    - `fit_garch` — ARMA-mean, GARCH(1,1)-variance. Implementation note:
      the `arch` library's built-in mean equation only supports AR terms,
      not full ARMA (no MA component), so this uses the standard two-step
      approach — fit ARMA via statsmodels first, then GARCH(1,1) via
      `arch` on the ARMA residuals. Documented in the module docstring.
  - `tests/test_baselines.py` — 12 tests on synthetic series with known
    properties (white noise, a simulated GARCH(1,1) process with real
    volatility clustering). One test specifically checks the GARCH
    variance forecast lands in the right order of magnitude relative to
    realised training variance — this guards against a residual-scaling
    bug (residuals are scaled by 100 before fitting GARCH for numerical
    stability, then variance must be unscaled by 100² on the way back
    out; caught and fixed this exact bug during development).
- All 31 tests passing across the full test suite (loader + metrics +
  baselines).
- Sanity-checked the full pipeline end-to-end (loader → ARMA → metrics) on
  AAPL-shaped synthetic data; RMSE/MAE land in the same range as the
  manual fit in notebook 03.
- Added `arch` to `requirements.txt`.

## Blockers

- **WRDS/CRSP access still pending** (Business School approval step).
  Email sent to Dr Dahlqvist asking him to confirm authorised-user status
  given Data Science isn't routed through Business School by default.
- GARCH-in-mean variant (Engle, Lilien and Robins, 1987 — conditional
  variance feeding into the mean equation) is in the proposal but not yet
  implemented; current `fit_garch` is the standard two-step ARMA-mean +
  GARCH(1,1)-variance form only.

## Decisions needed

- None outstanding — proceeding with deep learning models (LSTM) next
  while WRDS access is pending.

## Next steps

- Continue chasing WRDS access.
- Build LSTM forecasting module, same tested pattern.
- Revisit GARCH-in-mean once core baselines are confirmed working against
  real CRSP data.
