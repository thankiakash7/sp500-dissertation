"""
Step 1: Classical walk-forward with checkpointing.
Saves progress every block so it can be resumed.
Run: python3 step1_classical.py
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

import config
from src.data.panel import load_panel, select_universe

DATA_PROCESSED = config.DATA_PROCESSED
SUFFIX = config.OUTPUT_SUFFIX
CHECKPOINT_FILE = DATA_PROCESSED / f"_classical_checkpoint{SUFFIX}.json"
OUTPUT_FILE = DATA_PROCESSED / f"walk_forward_classical_predictions_top100{SUFFIX}.parquet"
GARCH_VOL_FILE = DATA_PROCESSED / f"garch_conditional_vol_top100{SUFFIX}.parquet"
ORDERS_FILE = DATA_PROCESSED / f"_arma_orders{SUFFIX}.json"

N_STOCKS = config.N_STOCKS
TRAIN_END = config.TRAIN_END
TEST_START = config.TEST_START
TEST_END = config.TEST_END
RETRAIN_EVERY = config.RETRAIN_EVERY
MAX_ARMA_ORDER = config.MAX_ARMA_ORDER


def get_universe(panel):
    return select_universe(panel, TRAIN_END, N_STOCKS)


def select_orders(panel, universe):
    # AR(1,0) for all stocks — standard in EMH literature.
    # BIC search across orders rarely beats AR(1) on daily returns.
    print("Using AR(1,0) for all stocks.")
    return {permno: (1, 0) for permno in universe}


def run(max_blocks=None):
    panel = load_panel()
    universe = get_universe(panel)
    print(f"Universe: {len(universe)} permnos")

    arma_orders = select_orders(panel, universe)
    panel_by_permno = {p: g.sort_values("date") for p, g in panel.groupby("permno")}

    test_dates = (
        panel.loc[panel["date"].between(TEST_START, TEST_END), "date"]
        .drop_duplicates().sort_values().tolist()
    )
    all_blocks = list(range(0, len(test_dates), RETRAIN_EVERY))
    n_blocks = len(all_blocks)

    # load checkpoint
    start_block_idx = 0
    existing_rows = []
    existing_vol_rows = []
    if CHECKPOINT_FILE.exists():
        cp = json.loads(CHECKPOINT_FILE.read_text())
        start_block_idx = cp.get("next_block_idx", 0)
        if OUTPUT_FILE.exists() and start_block_idx > 0:
            existing_rows = pd.read_parquet(OUTPUT_FILE).to_dict("records")
            print(f"Resuming from block {start_block_idx}/{n_blocks} ({len(existing_rows)} rows already done)")
        if GARCH_VOL_FILE.exists() and start_block_idx > 0:
            existing_vol_rows = pd.read_parquet(GARCH_VOL_FILE).to_dict("records")

    rows = list(existing_rows)
    vol_rows = list(existing_vol_rows)

    end_block_idx = n_blocks if max_blocks is None else min(start_block_idx + max_blocks, n_blocks)

    for bi, block_idx in enumerate(all_blocks[start_block_idx:end_block_idx], start=start_block_idx):
        block_dates = test_dates[block_idx: block_idx + RETRAIN_EVERY]
        first_fc = pd.Timestamp(block_dates[0])
        info_date = panel.loc[panel["date"] < first_fc, "date"].max()
        blk_num = bi + 1
        print(f"  Block {blk_num}/{n_blocks}: {first_fc.date()} | info {info_date.date()}")

        for permno in universe:
            if permno not in panel_by_permno:
                continue
            pdata = panel_by_permno[permno]
            tr = pdata.loc[pdata["date"] <= info_date, "model_return"].dropna()
            if len(tr) < 60:
                continue

            drift = float(tr.mean())
            p, q = arma_orders[permno]

            arma_fc = drift
            try:
                m = ARIMA(tr, order=(p, 0, q)).fit()
                arma_fc = float(m.forecast(steps=1).iloc[0])
            except Exception:
                pass

            garch_fc = drift
            garch_vol = np.nan
            try:
                r_sc = tr * 100.0
                ar1 = ARIMA(r_sc, order=(1, 0, 0)).fit()
                gm = arch_model(ar1.resid, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
                garch_fc = float(ar1.forecast(steps=1).iloc[0]) / 100.0
                # One-step-ahead conditional volatility forecast, converted back from
                # the *100-scaled fitting units to the raw return scale. Previously
                # fitted and then discarded; step4_downstream.py's vol-scaled
                # portfolio now uses this in place of realised (backward-looking)
                # volatility when available — see GARCH_VOL_FILE below.
                fcast_var = gm.forecast(horizon=1).variance.iloc[-1, 0]
                garch_vol = float(np.sqrt(fcast_var)) / 100.0
            except Exception:
                pass

            for fc_date in block_dates:
                fc_date = pd.Timestamp(fc_date)
                act_rows = pdata.loc[pdata["date"] == fc_date, "model_return"]
                if act_rows.empty:
                    continue
                actual_value = act_rows.iloc[0]
                
                if pd.isna(actual_value):
                    continue
                
                actual = float(actual_value)
                
                if np.isnan(actual):
                    continue
                for model_name, fc in [("mean_baseline", drift), ("arma", arma_fc), ("ar1_garch_t", garch_fc)]:
                    rows.append({
                        "date": fc_date.isoformat(),
                        "permno": int(permno),
                        "model": model_name,
                        "forecast": fc,
                        "actual": actual,
                    })
                if not np.isnan(garch_vol):
                    vol_rows.append({
                        "date": fc_date.isoformat(),
                        "permno": int(permno),
                        "garch_cond_vol": garch_vol,
                    })

        # save checkpoint BEFORE moving to next block
        next_idx = bi + 1
        df_out = pd.DataFrame(rows)
        df_out["date"] = pd.to_datetime(df_out["date"])
        df_out.to_parquet(OUTPUT_FILE, index=False)
        if vol_rows:
            df_vol = pd.DataFrame(vol_rows)
            df_vol["date"] = pd.to_datetime(df_vol["date"])
            df_vol.to_parquet(GARCH_VOL_FILE, index=False)
        CHECKPOINT_FILE.write_text(json.dumps({"next_block_idx": next_idx}))
        print(f"    checkpoint: {len(rows)} rows, next_block={next_idx}", flush=True)

    print(f"\nDone. {len(rows)} total rows in {OUTPUT_FILE.name}")
    if start_block_idx + (max_blocks or n_blocks) >= n_blocks:
        CHECKPOINT_FILE.unlink(missing_ok=True)
        print("Checkpoint cleared — classical step complete.")
    else:
        remaining = n_blocks - end_block_idx
        print(f"{remaining} blocks remain. Run again to continue.")

    return pd.read_parquet(OUTPUT_FILE)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=None, help="max blocks to process this run")
    args = parser.parse_args()
    run(max_blocks=args.blocks)
