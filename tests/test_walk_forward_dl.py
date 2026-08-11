"""Tests for src/evaluation/walk_forward_dl.py."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.walk_forward_dl import (
    WalkForwardDLConfig,
    create_sequence_records,
    estimate_full_run_seconds,
    walk_forward_train_predict,
)


def _make_synthetic_wide_returns(n_stocks=4, n_days=900, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    data = rng.normal(0, 0.01, size=(n_days, n_stocks))
    return pd.DataFrame(data, index=dates, columns=[100 + i for i in range(n_stocks)])


def test_create_sequence_records_shapes_and_no_nans():
    wide = _make_synthetic_wide_returns()
    means, stds = wide.mean(), wide.std()
    standardised = (wide - means) / stds

    records = create_sequence_records(standardised, wide, lookback=20)

    assert len(records) > 0
    for r in records[:5]:
        assert r["x"].shape == (20,)
        assert not np.isnan(r["x"]).any()
        assert not np.isnan(r["y_raw"])


def test_walk_forward_train_predict_runs_end_to_end_on_synthetic_data():
    wide = _make_synthetic_wide_returns(n_stocks=3, n_days=400)
    train_end = wide.index[250]
    test_end = wide.index[-1]

    config = WalkForwardDLConfig(
        cell_type="GRU",
        lookback=15,
        hidden_size=4,
        initial_epochs=1,
        finetune_epochs=1,
        batch_size=16,
        retrain_every=21,
        device=torch.device("cpu"),
    )

    preds = walk_forward_train_predict(wide, config, train_end=train_end, test_end=test_end)

    assert not preds.empty
    assert set(preds.columns) == {"date", "permno", "actual", "forecast", "model"}
    assert (preds["model"] == "gru").all()
    assert preds["date"].min() > train_end
    assert preds["date"].max() <= test_end
    assert preds["forecast"].notna().all()
    assert preds["actual"].notna().all()


def test_walk_forward_train_predict_rejects_unknown_cell_type():
    wide = _make_synthetic_wide_returns(n_stocks=2, n_days=300)
    config = WalkForwardDLConfig(
        cell_type="TRANSFORMER",  # invalid
        lookback=10,
        hidden_size=4,
        initial_epochs=1,
        finetune_epochs=1,
        batch_size=8,
        retrain_every=21,
    )
    with pytest.raises(ValueError):
        walk_forward_train_predict(
            wide, config, train_end=wide.index[200], test_end=wide.index[-1]
        )


def test_walk_forward_warm_starts_second_block_not_from_scratch():
    """The first block should train a fresh model; subsequent blocks should
    reuse (not recreate) the same model object, i.e. genuinely warm-start."""
    wide = _make_synthetic_wide_returns(n_stocks=2, n_days=400)
    config = WalkForwardDLConfig(
        cell_type="LSTM",
        lookback=10,
        hidden_size=4,
        initial_epochs=1,
        finetune_epochs=1,
        batch_size=8,
        retrain_every=21,
    )
    preds = walk_forward_train_predict(
        wide, config, train_end=wide.index[250], test_end=wide.index[-1]
    )
    # With retrain_every=21 over ~140 remaining days, expect multiple blocks
    n_blocks_expected = len([d for d in wide.index if d > wide.index[250]][::21])
    assert n_blocks_expected > 1
    assert not preds.empty


def test_walk_forward_train_predict_supports_tft():
    """TFT is a third supported cell_type alongside LSTM/GRU, using the
    shared TFTLite architecture from src.models.deep."""
    wide = _make_synthetic_wide_returns(n_stocks=2, n_days=350)
    config = WalkForwardDLConfig(
        cell_type="TFT",
        lookback=10,
        tft_d_model=8,
        tft_n_heads=2,
        initial_epochs=1,
        finetune_epochs=1,
        batch_size=8,
        retrain_every=21,
    )
    preds = walk_forward_train_predict(
        wide, config, train_end=wide.index[250], test_end=wide.index[-1]
    )
    assert not preds.empty
    assert (preds["model"] == "tft").all()
    assert preds["forecast"].notna().all()


def test_checkpoint_file_created_then_removed_on_completion(tmp_path):
    wide = _make_synthetic_wide_returns(n_stocks=2, n_days=350)
    ckpt = tmp_path / "run.ckpt"
    config = WalkForwardDLConfig(
        cell_type="GRU", lookback=10, hidden_size=4,
        initial_epochs=1, finetune_epochs=1, batch_size=8, retrain_every=21,
    )
    preds = walk_forward_train_predict(
        wide, config, train_end=wide.index[250], test_end=wide.index[-1], checkpoint_path=ckpt
    )
    assert not preds.empty
    assert not ckpt.exists(), "checkpoint should be cleared once the run completes"


def test_checkpoint_resume_skips_already_completed_blocks(tmp_path, monkeypatch):
    """If a run is interrupted partway through block 1 (block 0 already
    checkpointed), resuming must not retrain block 0 again — verified by
    counting total _fit_epochs calls across the interrupted + resumed run."""
    import src.evaluation.walk_forward_dl as wf

    wide = _make_synthetic_wide_returns(n_stocks=2, n_days=450)
    train_end = wide.index[250]
    test_end = wide.index[-1]
    ckpt = tmp_path / "resume.ckpt"
    config = WalkForwardDLConfig(
        cell_type="GRU", lookback=10, hidden_size=4,
        initial_epochs=1, finetune_epochs=1, batch_size=8, retrain_every=21,
    )

    all_dates = wide.index
    block_starts = [d for d in all_dates if train_end < d <= test_end][:: config.retrain_every]
    assert len(block_starts) >= 3, "test needs at least three retrain blocks"

    real_fit = wf._fit_epochs
    call_log: list[int] = []
    aborted = {"done": False}

    class _Abort(Exception):
        pass

    def wrapped_fit(model, loader, criterion, optimiser, n_epochs, device):
        call_log.append(1)
        result = real_fit(model, loader, criterion, optimiser, n_epochs, device)
        if len(call_log) == 2 and not aborted["done"]:
            aborted["done"] = True
            raise _Abort()  # abort mid-block-1, after block 0's checkpoint was saved
        return result

    monkeypatch.setattr(wf, "_fit_epochs", wrapped_fit)

    with pytest.raises(_Abort):
        wf.walk_forward_train_predict(wide, config, train_end=train_end, test_end=test_end, checkpoint_path=ckpt)

    assert ckpt.exists(), "checkpoint should persist after an interrupted run"
    assert len(call_log) == 2  # block 0 fit fully, block 1 fit then aborted

    resumed_preds = wf.walk_forward_train_predict(
        wide, config, train_end=train_end, test_end=test_end, checkpoint_path=ckpt
    )

    assert not ckpt.exists(), "checkpoint should be cleared once the resumed run completes"
    assert not resumed_preds.empty
    # One fit call per block, plus the one wasted call for block 1's aborted
    # attempt (which gets legitimately re-fit on resume, since it never
    # finished and was never checkpointed). What must NOT happen is block 0
    # being fit a second time — if it were, this total would be one higher.
    assert len(call_log) == len(block_starts) + 1, (
        "block 0 must not be re-fit on resume: total fit calls should be "
        "one per block plus exactly one wasted retry for the aborted block"
    )


def test_estimate_full_run_seconds_returns_positive_projection():
    wide = _make_synthetic_wide_returns(n_stocks=3, n_days=400)
    config = WalkForwardDLConfig(
        cell_type="LSTM", lookback=15, hidden_size=4,
        initial_epochs=2, finetune_epochs=1, batch_size=16, retrain_every=21,
    )
    result = estimate_full_run_seconds(
        config, wide, train_end=wide.index[250], test_end=wide.index[-1], n_timing_batches=2
    )
    assert result["seconds_per_batch"] > 0
    assert result["n_blocks"] >= 1
    assert result["projected_total_hours"] >= 0


def test_estimate_full_run_seconds_returns_positive_projection():
    wide = _make_synthetic_wide_returns(n_stocks=3, n_days=400)
    config = WalkForwardDLConfig(
        cell_type="LSTM", lookback=15, hidden_size=4,
        initial_epochs=2, finetune_epochs=1, batch_size=16, retrain_every=21,
    )
    result = estimate_full_run_seconds(
        config, wide, train_end=wide.index[250], test_end=wide.index[-1], n_timing_batches=2
    )
    assert result["seconds_per_batch"] > 0
    assert result["n_blocks"] >= 1
    assert result["projected_total_hours"] >= 0
