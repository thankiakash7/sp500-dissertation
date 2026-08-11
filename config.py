"""
Single source of truth for constants shared across step1-4.

Why this file exists
---------------------
Before this, N_STOCKS / TRAIN_END / TEST_START / TEST_END / LOOKBACK were
copy-pasted independently into step1_classical.py, step2_deep.py, and
step3_tft.py. That's exactly the kind of duplication that lets two scripts
silently drift apart — which is what happened: step2/step3 kept a static
train/test split while step1 moved to walk-forward, and nothing forced them
to stay in sync. Importing from here instead means changing the universe
size or test window is a one-line edit, not a four-file grep.

QUICK_MODE
----------
Set QUICK_MODE = True (or env var DISSERTATION_QUICK_MODE=1) to run the
whole pipeline in seconds on a laptop, for smoke-testing changes: a tiny
universe, a short test window, and minimal epochs. Full-scale numbers must
be produced with QUICK_MODE = False, which is the default.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "reports" / "models"
REPORT_TABLES = PROJECT_ROOT / "reports" / "tables"
REPORT_FIGURES = PROJECT_ROOT / "reports" / "figures"
CRSP_PANEL_PATH = DATA_PROCESSED / "crsp_sp500_daily_corrected_2010_2024.parquet"

for _d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODEL_DIR, REPORT_TABLES, REPORT_FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

QUICK_MODE = os.environ.get("DISSERTATION_QUICK_MODE", "0") == "1"

# Every output filename gets this suffix in QUICK_MODE, so a fast smoke-test
# run can never silently overwrite (or be mistaken for) real full-scale
# results — this is exactly the failure mode lab_log/week_04.md describes
# happening for real with the old notebooks (QUICK_MODE writing into the
# same "top100"-named files used for the genuine full-scale run).
OUTPUT_SUFFIX = "_quick" if QUICK_MODE else ""

SEED = 42

# ── Universe / dates ─────────────────────────────────────────────────────────
N_STOCKS = 10 if QUICK_MODE else 100
TRAIN_END = pd.Timestamp("2018-12-31")
VALIDATION_END = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2020-01-02")
TEST_END = pd.Timestamp("2020-03-31") if QUICK_MODE else pd.Timestamp("2024-12-31")

# ── Shared walk-forward cadence (classical AND deep learning) ──────────────
RETRAIN_EVERY = 21  # trading days between retrains, both step1 and step2/3

# ── Classical (step1) ────────────────────────────────────────────────────────
MAX_ARMA_ORDER = 2

# ── Deep learning (step2 LSTM/GRU, step3 TFT) ───────────────────────────────
LOOKBACK = 20 if QUICK_MODE else 252
HIDDEN_SIZE = 64
BATCH_SIZE = 32 if QUICK_MODE else 256
LR = 1e-3
INITIAL_EPOCHS = 2 if QUICK_MODE else 50   # first block: train from scratch
FINETUNE_EPOCHS = 1 if QUICK_MODE else 5   # later blocks: warm-start + fine-tune
PATIENCE = 5

# TFT-specific architecture (kept smaller than LSTM/GRU hidden size — attention
# over a 252-day window is the expensive part, per-layer width is not).
TFT_D_MODEL = 16 if QUICK_MODE else 32
TFT_N_HEADS = 4
TFT_DROPOUT = 0.1
TFT_BATCH_SIZE = 64 if QUICK_MODE else 512

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
