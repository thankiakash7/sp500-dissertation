# Week 5 — Status note

**Date:** [fill in actual date of supervisor meeting]
**For:** Fortnightly meeting with Dr Dahlqvist

## Progress

Went back through the whole pipeline end-to-end (not just notebook 09)
looking specifically at the walk-forward question week_04 flagged as a
"deliberately not auto-decided" open item, plus a general cleanup pass
ahead of submission. Found that the decision had effectively been made by
default rather than by Dahlqvist: `step2_deep.py` and `step3_tft.py` (the
pipeline scripts, not the notebooks) were still doing a static fit-once
split, and `README_pipeline.md` had started claiming *"All models are
evaluated using a strict walk-forward forecasting framework"* — which was
true for step1 but not step2/step3. Given week_04's own note
("Do not claim the static LSTM/GRU/TFT results are final walk-forward
results" is in `README_NEXT_NOTEBOOKS.md`), this needed fixing rather than
flagging again.

- Extended `src/evaluation/walk_forward_dl.py` (built in week 4, smoke-tested
  but never wired in) to support TFT as well as LSTM/GRU, via a shared
  `build_model()` factory in a new `src/models/deep.py`. Added resumable
  block-level checkpointing — a full walk-forward DL run is a multi-hour
  job, so this needed the same interrupt-and-resume safety net
  `step1_classical.py` already had. Verified the resume logic for real:
  let a live TFT training run get cut off mid-block by a sandbox timeout,
  reran the same command, confirmed it picked up from the correct block
  instead of restarting.
- Added `estimate_full_run_seconds()` — times a handful of real batches on
  the actual machine and projects total wall-clock time before committing
  to a full run, rather than discovering the true cost hours in.
- Rewrote `step2_deep.py` and `step3_tft.py` to call the walk-forward
  engine. Both now retrain every 21 trading days, matching step1.
- Added `config.py` as a single source of truth for dates/universe
  size/hyperparameters, and `src/data/panel.py` for universe selection —
  these three constants were independently copy-pasted into step1/2/3
  before, which is exactly how step1 and step2/3 were able to drift onto
  different methodologies without either script "knowing."
- Found the GARCH conditional volatility forecast (fit every walk-forward
  block in step1) was computed and then discarded — the vol-scaled
  portfolio in step4 used only realised rolling volatility. Wired the
  actual forecast in, with realised vol as a fallback where it's missing.
- Found `reports/tables/portfolio_summary_5_10_25_bps.csv` was all zeros —
  looks like exactly the QUICK_MODE-writes-into-full-scale-filenames
  failure mode this file's name suggests, from an earlier run. Added a
  `_quick` suffix to every pipeline output filename when `QUICK_MODE` is
  on, so that can't happen again, and removed the corrupted file (the
  correct data for the same cost tiers was already in
  `portfolio_summary.csv`).
- Consolidated 5 requirements files into one, and in doing so found a real
  dependency conflict neither of the two "final" ones would have surfaced
  on its own: `wrds` hard-requires `pandas<2.3`, but `requirements_final.txt`
  had pandas 3.0.3. That's almost certainly *why* there were two different
  "final" requirements files in the first place — one for running the
  modelling pipeline, one for pulling data — rather than one that actually
  covers both.
- Cleaned up: removed `backup_final_rerun/` and
  `final_submission_pipeline.zip` (91MB), stale model checkpoints from the
  old static-split code path, `__pycache__`/`.DS_Store` clutter.
- CRSP parquet was committed in git history despite the project's own
  README and `.gitignore` saying that should never happen — reset git
  history rather than attempting an in-place rewrite, since the old
  history is still a compliance problem in any existing clone.
- Regenerated `reports/figures/` with clearer styling (model-family colour
  coding, value labels, source captions) and one genuinely new chart
  (portfolio Sharpe broken out across all four cost tiers instead of one).
- Merged `README.md` and `README_pipeline.md` into one accurate README,
  removed the dead `reports/final_run_2026_07_17/` reference.
- Test suite: 46 → 54 (new: TFT walk-forward, checkpoint create/clear/resume,
  runtime estimator). All 54 pass under a single, conflict-free
  `requirements.txt`.

## Blockers

None on the code side — steps 1-4 all run end-to-end (verified with
synthetic data shaped like the real CRSP panel, since real CRSP data isn't
available in this environment). What's still needed, same as week_04, all
requiring local resources:

1. Run `python run.py` for real on the full CRSP panel — walk-forward
   LSTM/GRU/TFT is now correct but still a multi-hour-per-model job (the
   pre-flight estimator will give an honest number for your machine).
2. The results and figures currently in `reports/` are from the *old*
   static-split run and need regenerating — they're kept as a reference
   point, clearly labelled as such in `README.md` and `CHANGELOG.md`, not
   as final numbers.
3. Fama-French/Carhart factor download needs network access this sandbox
   doesn't have to Ken French's data library — confirmed it fails cleanly
   now (step4 completes and reports the other results) rather than
   crashing the whole downstream step.

Test suite: 54/54 passing.
