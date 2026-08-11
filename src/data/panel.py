"""
Shared CRSP panel loading and universe selection.

Previously step1_classical.py, step2_deep.py, and step3_tft.py each had
their own copy of "load the parquet, take the top N stocks by market cap as
of the last date on/before TRAIN_END". Three copies is how three scripts
can quietly drift onto different universes without anyone noticing — this
module makes it one function so all pipeline steps are guaranteed to
evaluate on the same top-N_STOCKS universe.
"""

from __future__ import annotations

import pandas as pd

from config import CRSP_PANEL_PATH


def load_panel(path=CRSP_PANEL_PATH) -> pd.DataFrame:
    """Load the processed CRSP panel and add the `model_return` alias used
    throughout the pipeline (kept separate from `ret` in case a different
    return definition, e.g. log-returns, is swapped in later)."""
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["model_return"] = panel["ret"]
    return panel


def select_universe(panel: pd.DataFrame, train_end: pd.Timestamp, n_stocks: int) -> list[int]:
    """Top-`n_stocks` permnos by market cap, as of the latest date on or
    before `train_end`. This is the single definition of "the universe"
    that step1/step2/step3 all use, so classical and deep-learning models
    are compared on identical stocks."""
    cross = panel[panel["date"] <= train_end]
    if cross.empty:
        raise ValueError(f"No panel rows on or before train_end={train_end.date()}.")
    latest = cross["date"].max()
    return (
        cross[cross["date"] == latest]
        .dropna(subset=["mktcap"])
        .nlargest(n_stocks, "mktcap")["permno"]
        .tolist()
    )


def pivot_wide(panel: pd.DataFrame, universe: list[int]) -> pd.DataFrame:
    """Date-indexed, permno-columned wide return matrix for the given universe."""
    subset = panel[panel["permno"].isin(universe)]
    return subset.pivot_table(index="date", columns="permno", values="model_return", aggfunc="first").sort_index()
