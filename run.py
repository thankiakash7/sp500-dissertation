#!/usr/bin/env python3
"""
Full pipeline orchestrator.
Run from the project root:

    python run.py               # run everything (full scale — see note below)
    python run.py --quick       # fast end-to-end smoke test (tiny universe/window)
    python run.py --skip-tft    # skip TFT (faster, 5-model results)
    python run.py --step 4      # run only step 4 (downstream analysis)

Steps:
  1  Classical walk-forward (mean_baseline, ARMA, AR1-GARCH-t) — resumable
  2  LSTM + GRU walk-forward (retrain every 21 days) — resumable per block
  3  TFT walk-forward (retrain every 21 days) — resumable per block
  4  DM+MCS, portfolio, vol-scaled, FF alpha

Full-scale note: steps 2 and 3 now genuinely retrain every 21 trading days
(see src/evaluation/walk_forward_dl.py), which is a multi-hour job per
model at full scale (100 stocks, 252-day lookback) on a laptop CPU. Both
steps print a pre-flight time estimate before committing to the full run,
and checkpoint after every retrain block so an interrupted run can simply
be re-run to resume. --quick runs the whole pipeline on a tiny slice of
data in well under a minute, to verify the pipeline itself works before
committing to a full-scale run.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_step(script: str, extra_args: list[str] = [], env: dict = None):
    cmd = [sys.executable, str(HERE / script)] + extra_args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n❌  {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def check_done(parquet: str) -> bool:
    return (HERE / "data" / "processed" / parquet).exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tft", action="store_true",
                        help="Skip TFT training (run with 5 models)")
    parser.add_argument("--step", type=int, default=None,
                        help="Run only a single step (1-4)")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke-test run: tiny universe/window, minimal epochs (sets "
                             "DISSERTATION_QUICK_MODE=1 for every step)")
    args = parser.parse_args()

    env = os.environ.copy()
    dl_extra_args = []
    if args.quick:
        env["DISSERTATION_QUICK_MODE"] = "1"
        dl_extra_args = ["--quick"]
        print("Running in --quick mode: tiny universe/window, minimal epochs. "
              "For real dissertation numbers, run without --quick.")

    if args.step:
        steps = [args.step]
    else:
        steps = [1, 2, 3, 4] if not args.skip_tft else [1, 2, 4]

    for step in steps:
        if step == 1:
            if check_done("walk_forward_classical_predictions_top100.parquet") and not args.quick:
                print("\n[Step 1] Already complete — skipping")
            else:
                run_step("step1_classical.py", env=env)

        elif step == 2:
            if check_done("lstm_gru_predictions_top100.parquet") and not args.quick:
                print("\n[Step 2] Already complete — skipping")
            else:
                run_step("step2_deep.py", dl_extra_args, env=env)

        elif step == 3:
            if check_done("tft_predictions_top100.parquet") and not args.quick:
                print("\n[Step 3] Already complete — skipping")
            else:
                run_step("step3_tft.py", dl_extra_args, env=env)

        elif step == 4:
            run_step("step4_downstream.py", env=env)

    print("\n" + "="*60)
    print("✓  Pipeline complete.")
    print("   Results in reports/tables/")
    print("="*60)


if __name__ == "__main__":
    main()
