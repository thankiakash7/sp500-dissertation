"""
Step 2: LSTM + GRU, walk-forward.

This used to fit each model ONCE on data through 2018-12-31 and predict
straight through 2020-2024 without ever updating on new data — see
src/evaluation/walk_forward_dl.py's module docstring for the full story.
This script now calls that module's walk-forward engine instead, so LSTM
and GRU are retrained every RETRAIN_EVERY (21) trading days, exactly like
step1_classical.py's ARMA/GARCH models.

Run: python3 step2_deep.py [--quick]

Resumable: progress is checkpointed after every retrain block (per model),
so an interrupted run picks back up rather than starting over.
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch

import config
from src.data.panel import load_panel, select_universe, pivot_wide
from src.evaluation.walk_forward_dl import (
    WalkForwardDLConfig,
    estimate_full_run_seconds,
    walk_forward_train_predict,
)

OUT_COMBINED = config.DATA_PROCESSED / f"lstm_gru_predictions_top100{config.OUTPUT_SUFFIX}.parquet"


def make_dl_config(cell_type: str) -> WalkForwardDLConfig:
    return WalkForwardDLConfig(
        cell_type=cell_type,
        lookback=config.LOOKBACK,
        hidden_size=config.HIDDEN_SIZE,
        initial_epochs=config.INITIAL_EPOCHS,
        finetune_epochs=config.FINETUNE_EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LR,
        retrain_every=config.RETRAIN_EVERY,
        device=config.DEVICE,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Confirms you intend a QUICK_MODE smoke-test run (also set DISSERTATION_QUICK_MODE=1)")
    parser.add_argument("--skip-estimate", action="store_true",
                         help="Skip the pre-flight runtime estimate")
    args = parser.parse_args()

    if args.quick and not config.QUICK_MODE:
        print("--quick passed but DISSERTATION_QUICK_MODE=1 was not set in the environment.\n"
              "Set it so config.py picks up the smaller universe/window, e.g.:\n"
              "  DISSERTATION_QUICK_MODE=1 python3 step2_deep.py --quick")
        return

    print(f"Device: {config.DEVICE}")
    print(f"QUICK_MODE={config.QUICK_MODE}  N_STOCKS={config.N_STOCKS}  "
          f"TEST_END={config.TEST_END.date()}  LOOKBACK={config.LOOKBACK}")

    torch.manual_seed(config.SEED)

    panel = load_panel()
    universe = select_universe(panel, config.TRAIN_END, config.N_STOCKS)
    wide = pivot_wide(panel, universe)
    print(f"Universe: {len(universe)} stocks, {wide.shape[0]} trading days")

    all_preds = []
    for cell_type in ["LSTM", "GRU"]:
        out_path = config.DATA_PROCESSED / f"{cell_type.lower()}_predictions_top100{config.OUTPUT_SUFFIX}.parquet"
        ckpt_path = config.MODEL_DIR / f"{cell_type.lower()}_walk_forward{config.OUTPUT_SUFFIX}.ckpt"

        if out_path.exists() and not ckpt_path.exists():
            print(f"\n{cell_type} already complete — skipping")
            all_preds.append(pd.read_parquet(out_path))
            continue

        dl_config = make_dl_config(cell_type)

        if not args.skip_estimate and not ckpt_path.exists():
            est = estimate_full_run_seconds(dl_config, wide, config.TRAIN_END, config.TEST_END)
            print(f"\n{cell_type}: {est['n_blocks']} retrain blocks, "
                  f"~{est['seconds_per_batch']:.2f}s/batch, "
                  f"projected {est['projected_total_hours']:.1f}h total on this machine.")

        print(f"\nTraining {cell_type} (walk-forward, retrain every {config.RETRAIN_EVERY} days)...")
        preds = walk_forward_train_predict(
            wide, dl_config, train_end=config.TRAIN_END, test_end=config.TEST_END,
            checkpoint_path=ckpt_path,
        )
        preds.to_parquet(out_path, index=False)
        print(f"  {cell_type} done: {len(preds)} rows -> {out_path.name}")
        all_preds.append(preds)

    combined = pd.concat(all_preds, ignore_index=True)
    combined.to_parquet(OUT_COMBINED, index=False)
    print(f"\nAll DL done: {len(combined)} rows -> {OUT_COMBINED.name}")
    print(combined.groupby("model").size())


if __name__ == "__main__":
    main()
