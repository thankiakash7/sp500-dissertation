"""
Walk-forward retraining for sequence models (LSTM, GRU, TFT).

Background
----------
step2_deep.py and step3_tft.py used to fit each model ONCE on data through
TRAIN_END and predict straight through the 2020-2024 test period without
ever updating on new data — a static train/test split, not the walk-forward
evaluation step1_classical.py uses for the classical models (mean baseline,
ARMA, AR(1)-GARCH-t). README_NEXT_NOTEBOOKS.md explicitly warned against
treating those static results as final walk-forward results, and the
generated README_pipeline.md nonetheless presented all six models in one
table with no methodological distinction. This module is what closes that
gap: step2_deep.py and step3_tft.py both call `walk_forward_train_predict`
below instead of doing a single fit-and-predict.

Two honest design choices, stated up front
-------------------------------------------
1. Retraining a sequence model from scratch every `retrain_every` trading
   days for 5 years (~60 retrains) at full epoch counts is a different
   order of compute than the classical models (a handful of ARMA/GARCH MLE
   fits per block). Doing that with `initial_epochs` full passes per block
   would multiply an already expensive single fit by ~60. Instead, this
   module **warm-starts** each retrain from the previous block's weights
   and fine-tunes for a much smaller number of epochs
   (`finetune_epochs`). This is a standard, citable compromise (e.g. used
   in online/continual-learning forecasting setups) — but it is NOT the
   same as fully refitting from scratch each block, and the dissertation
   methodology section should describe it as such.

2. Standardisation statistics (mean/std used to scale returns before
   feeding the network) are recomputed at each retrain block using only
   data up to that block's cutoff, unlike the old static split, which
   computed them once at TRAIN_END and reused them for the entire
   2020-2024 test period. This is a strict improvement (removes a small
   forward-looking convenience), not a free choice — it changes the
   numbers relative to the old static-split results, mention this in the
   methodology write-up.

Compute note
------------
Full scale (100 stocks, 252-day lookback, ~60 retrain blocks) takes on the
order of tens of hours per model on a single CPU core — this needs to run
on the student's own machine, ideally with an Apple Silicon GPU (MPS) or
CUDA, not in a constrained sandbox. `estimate_full_run_seconds` below times
a handful of real batches on the actual machine before committing to a run,
so this can be checked up front rather than discovered five hours in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.deep import build_model


class ReturnSequenceDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        x = torch.tensor(record["x"], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(record["y_standardised"], dtype=torch.float32)
        permno = torch.tensor(record["permno"], dtype=torch.long)
        y_raw = torch.tensor(record["y_raw"], dtype=torch.float32)
        return x, y, permno, y_raw


def create_sequence_records(standardised_panel: pd.DataFrame, raw_panel: pd.DataFrame, lookback: int) -> list[dict]:
    """Build (lookback-window -> next-day-return) training examples for every
    stock/date with enough history and no NaNs in the window or target."""
    records = []
    dates = standardised_panel.index.to_numpy()
    for permno in standardised_panel.columns:
        x_values = standardised_panel[permno].to_numpy(dtype=float)
        y_values = raw_panel[permno].to_numpy(dtype=float)
        for i in range(lookback, len(dates)):
            x_window = x_values[i - lookback:i]
            target_standardised = x_values[i]
            target_raw = y_values[i]
            if np.isnan(x_window).any() or np.isnan(target_standardised) or np.isnan(target_raw):
                continue
            records.append({
                "x": x_window.astype(np.float32),
                "y_standardised": np.float32(target_standardised),
                "y_raw": np.float32(target_raw),
                "date": pd.Timestamp(dates[i]),
                "permno": int(permno),
            })
    return records


@dataclass
class WalkForwardDLConfig:
    cell_type: str  # "LSTM", "GRU", or "TFT"
    lookback: int = 252
    hidden_size: int = 64          # LSTM/GRU only
    tft_d_model: int = 32          # TFT only
    tft_n_heads: int = 4           # TFT only
    tft_dropout: float = 0.1       # TFT only
    initial_epochs: int = 50
    finetune_epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    retrain_every: int = 21
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


def _new_model(config: WalkForwardDLConfig) -> nn.Module:
    return build_model(
        config.cell_type, seq_len=config.lookback, hidden_size=config.hidden_size,
        tft_d_model=config.tft_d_model, tft_n_heads=config.tft_n_heads, tft_dropout=config.tft_dropout,
    ).to(config.device)


def _fit_epochs(model, loader, criterion, optimiser, n_epochs, device):
    model.train()
    for _ in range(n_epochs):
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
    return model


def estimate_full_run_seconds(config: WalkForwardDLConfig, wide_returns: pd.DataFrame,
                               train_end: pd.Timestamp, test_end: pd.Timestamp,
                               n_timing_batches: int = 5) -> dict:
    """
    Time a handful of real training batches on this machine and project the
    total wall-clock time for a full walk-forward run, so a multi-hour
    commitment can be sized up before it's made rather than discovered
    partway through.

    Returns a dict with the measured seconds/batch and the projected total.
    """
    train_window = wide_returns.loc[:train_end]
    means, stds = train_window.mean(), train_window.std().replace(0, np.nan)
    standardised = (train_window - means) / stds
    records = create_sequence_records(standardised, train_window, config.lookback)
    if len(records) < config.batch_size:
        raise ValueError("Not enough history before train_end to time a single batch.")

    loader = DataLoader(ReturnSequenceDataset(records), batch_size=config.batch_size, shuffle=True)
    model = _new_model(config)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    model.train()
    timed = 0
    start = time.perf_counter()
    for x, y, _, _ in loader:
        x, y = x.to(config.device), y.to(config.device)
        optimiser.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()
        timed += 1
        if timed >= n_timing_batches:
            break
    elapsed = time.perf_counter() - start
    seconds_per_batch = elapsed / max(timed, 1)
    batches_per_epoch = max(len(loader), 1)

    n_blocks = len([d for d in wide_returns.index if train_end < d <= test_end][:: config.retrain_every])
    n_blocks = max(n_blocks, 1)
    total_epochs = config.initial_epochs + config.finetune_epochs * max(n_blocks - 1, 0)
    projected_seconds = seconds_per_batch * batches_per_epoch * total_epochs

    return {
        "seconds_per_batch": seconds_per_batch,
        "batches_per_epoch": batches_per_epoch,
        "n_blocks": n_blocks,
        "total_epochs_across_all_blocks": total_epochs,
        "projected_total_seconds": projected_seconds,
        "projected_total_hours": projected_seconds / 3600,
    }


def walk_forward_train_predict(
    wide_returns: pd.DataFrame,
    config: WalkForwardDLConfig,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """
    Walk-forward retrain + predict for a single sequence model (LSTM, GRU,
    or TFT) over (train_end, test_end].

    Block 0 (the first `retrain_every`-day block after train_end) trains
    from scratch for `config.initial_epochs` epochs. Every subsequent block
    warm-starts from the previous block's weights and fine-tunes for
    `config.finetune_epochs` epochs (see module docstring for why).

    Parameters
    ----------
    wide_returns : pd.DataFrame
        Date-indexed, permno-columned raw returns, covering at least
        [start, test_end].
    config : WalkForwardDLConfig
    train_end : pd.Timestamp
        End of the initial training window (first block fits on data up to
        here using `config.lookback` days of history before it).
    test_end : pd.Timestamp
        Last date to produce predictions for.
    checkpoint_path : Path, optional
        If given, progress is saved after every block (model weights, next
        block index, predictions so far) and reloaded automatically if the
        file already exists — mirrors step1_classical.py's block
        checkpointing so a multi-hour run can be safely interrupted and
        resumed. The checkpoint is deleted once the run completes.

    Returns
    -------
    pd.DataFrame with columns [date, permno, actual, forecast, model],
    one row per (date, permno) prediction in the test period.
    """
    all_dates = wide_returns.index
    block_starts = [d for d in all_dates if train_end < d <= test_end][:: config.retrain_every]

    criterion = nn.MSELoss()
    model = None
    prediction_rows: list[dict] = []
    start_block = 0

    if checkpoint_path is not None and checkpoint_path.exists():
        cp = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        model = _new_model(config)
        model.load_state_dict(cp["model_state"])
        start_block = cp["next_block"]
        prediction_rows = cp["prediction_rows"]
        print(f"  [{config.cell_type}] resuming walk-forward from block "
              f"{start_block + 1}/{len(block_starts)} ({len(prediction_rows)} predictions so far)")

    for block_i, block_start_date in enumerate(block_starts):
        if block_i < start_block:
            continue

        cutoff = all_dates[all_dates < block_start_date][-1]  # last date strictly before this block

        train_window = wide_returns.loc[:cutoff]
        means = train_window.mean()
        stds = train_window.std().replace(0, np.nan)
        standardised = (wide_returns.loc[:cutoff] - means) / stds

        train_records = create_sequence_records(standardised, train_window, config.lookback)
        if len(train_records) < config.batch_size:
            continue  # not enough history yet for a meaningful batch

        train_loader = DataLoader(
            ReturnSequenceDataset(train_records), batch_size=config.batch_size, shuffle=True
        )

        if model is None:
            model = _new_model(config)
            n_epochs = config.initial_epochs
        else:
            n_epochs = config.finetune_epochs

        optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        model = _fit_epochs(model, train_loader, criterion, optimiser, n_epochs, config.device)

        # Predict the block: dates from block_start_date up to (but not including) the next block start
        next_block_start = block_starts[block_i + 1] if block_i + 1 < len(block_starts) else None
        block_dates = [
            d for d in all_dates
            if d >= block_start_date and (next_block_start is None or d < next_block_start) and d <= test_end
        ]
        if block_dates:
            # Build prediction sequences using the *cutoff* standardisation stats
            # (no look-ahead: the network never sees its own target's scale).
            full_standardised_for_predict = (wide_returns.loc[:block_dates[-1]] - means) / stds
            predict_records = create_sequence_records(
                full_standardised_for_predict, wide_returns.loc[:block_dates[-1]], config.lookback
            )
            block_date_set = set(block_dates)
            predict_records = [r for r in predict_records if r["date"] in block_date_set]

            if predict_records:
                predict_loader = DataLoader(ReturnSequenceDataset(predict_records), batch_size=config.batch_size, shuffle=False)
                model.eval()
                idx = 0
                with torch.no_grad():
                    for x, y_std, permnos, y_raw in predict_loader:
                        x = x.to(config.device)
                        pred_std = model(x).cpu().numpy()
                        batch_n = len(pred_std)
                        for j in range(batch_n):
                            record = predict_records[idx + j]
                            permno = record["permno"]
                            forecast_raw = float(pred_std[j]) * float(stds.get(permno, np.nan)) + float(means.get(permno, np.nan))
                            prediction_rows.append({
                                "date": record["date"],
                                "permno": permno,
                                "actual": record["y_raw"],
                                "forecast": forecast_raw,
                                "model": config.cell_type.lower(),
                            })
                        idx += batch_n

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "next_block": block_i + 1,
                "prediction_rows": prediction_rows,
            }, checkpoint_path)
            print(f"  [{config.cell_type}] block {block_i + 1}/{len(block_starts)} done, "
                  f"checkpoint saved ({len(prediction_rows)} predictions so far)")

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()

    return pd.DataFrame(prediction_rows)
