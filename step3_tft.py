"""
Step 3: Temporal Fusion Transformer (pure PyTorch, no Lightning), walk-forward.

Same fix as step2_deep.py: this used to fit once on data through
2018-12-31 and predict straight through 2020-2024. It now calls the same
walk-forward engine (src/evaluation/walk_forward_dl.py) as LSTM/GRU, using
the lightweight TFTLite architecture from src/models/deep.py.

Run: python3 step3_tft.py [--quick]

Resumable: progress is checkpointed after every retrain block.
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

OUT_PATH = config.DATA_PROCESSED / f"tft_predictions_top100{config.OUTPUT_SUFFIX}.parquet"
CKPT_PATH = config.MODEL_DIR / f"tft_walk_forward{config.OUTPUT_SUFFIX}.ckpt"


def make_tft_config() -> WalkForwardDLConfig:
    return WalkForwardDLConfig(
        cell_type="TFT",
        lookback=config.LOOKBACK,
        tft_d_model=config.TFT_D_MODEL,
        tft_n_heads=config.TFT_N_HEADS,
        tft_dropout=config.TFT_DROPOUT,
        initial_epochs=config.INITIAL_EPOCHS,
        finetune_epochs=config.FINETUNE_EPOCHS,
        batch_size=config.TFT_BATCH_SIZE,
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
              "  DISSERTATION_QUICK_MODE=1 python3 step3_tft.py --quick")
        return

    if OUT_PATH.exists() and not CKPT_PATH.exists():
        print("TFT already complete.")
        return

    print(f"Device: {config.DEVICE}")
    print(f"QUICK_MODE={config.QUICK_MODE}  N_STOCKS={config.N_STOCKS}  "
          f"TEST_END={config.TEST_END.date()}  LOOKBACK={config.LOOKBACK}")

    torch.manual_seed(config.SEED)

    panel = load_panel()
    universe = select_universe(panel, config.TRAIN_END, config.N_STOCKS)
    wide = pivot_wide(panel, universe)
    print(f"Universe: {len(universe)} stocks, {wide.shape[0]} trading days")

    tft_config = make_tft_config()

    if not args.skip_estimate and not CKPT_PATH.exists():
        est = estimate_full_run_seconds(tft_config, wide, config.TRAIN_END, config.TEST_END)
        print(f"\nTFT: {est['n_blocks']} retrain blocks, ~{est['seconds_per_batch']:.2f}s/batch, "
              f"projected {est['projected_total_hours']:.1f}h total on this machine.")

    print(f"\nTraining TFT (walk-forward, retrain every {config.RETRAIN_EVERY} days)...")
    preds = walk_forward_train_predict(
        wide, tft_config, train_end=config.TRAIN_END, test_end=config.TEST_END,
        checkpoint_path=CKPT_PATH,
    )
    preds.to_parquet(OUT_PATH, index=False)
    print(f"TFT done: {len(preds)} rows -> {OUT_PATH.name}")


if __name__ == "__main__":
    main()
