"""
Generate dissertation figures from pipeline outputs in reports/tables/.

Six figures, each labelled with its data source so it's traceable back to a
specific CSV:

1. model_rmse_comparison.png       — RMSE by model, classical vs deep learning
2. model_directional_accuracy.png  — directional accuracy vs the 50% coin-flip line
3. portfolio_sharpe_by_cost.png    — Sharpe ratio across all four transaction-cost tiers
4. diebold_mariano_forest.png      — DM test statistics vs the mean baseline, with p-values
5. fama_french_alpha.png           — annualised alpha with significance markers
6. rmt_eigenvalue_spectrum.png     — signal vs noise-like eigenvalues by universe size

Run: python3 scripts/generate_dissertation_figures.py

Note on data provenance: this script only plots whatever is currently in
reports/tables/ — it doesn't know whether those numbers came from the
static-split or walk-forward deep-learning methodology. See CHANGELOG.md:
figures shipped in this repo were generated from the last full run, which
predates the walk-forward fix in step2_deep.py/step3_tft.py. Re-run this
script after re-running the corrected pipeline to get figures that match
the walk-forward results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────
CLASSICAL_MODELS = {"mean_baseline", "arma", "ar1_garch_t"}
DL_MODELS = {"lstm", "gru", "tft"}
COLOR_CLASSICAL = "#3B6E8F"   # slate blue
COLOR_DL = "#E07B39"          # warm orange
COLOR_NEUTRAL = "#6B7280"     # grey
COLOR_GOOD = "#2E8B57"        # sea green
COLOR_BAD = "#C0392B"         # brick red

MODEL_LABELS = {
    "mean_baseline": "Mean\nbaseline", "arma": "ARMA", "ar1_garch_t": "AR(1)-\nGARCH-t",
    "lstm": "LSTM", "gru": "GRU", "tft": "TFT",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})


def _model_color(model: str) -> str:
    return COLOR_CLASSICAL if model in CLASSICAL_MODELS else COLOR_DL if model in DL_MODELS else COLOR_NEUTRAL


def _bar_labels(ax, bars, fmt="{:.4f}", offset_frac=0.01):
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, np.sign(h) * y_range * offset_frac * 100 if h else 3),
                    textcoords="offset points", ha="center",
                    va="bottom" if h >= 0 else "top", fontsize=9, color="#333333")


def _legend_family(ax, loc="upper right"):
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_CLASSICAL, label="Classical"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_DL, label="Deep learning"),
    ]
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=9)


def _source_caption(ax, text):
    ax.annotate(text, xy=(0, -0.18), xycoords="axes fraction", fontsize=8,
                color="#888888", ha="left")


# ── 1. RMSE comparison ───────────────────────────────────────────────────
def fig_rmse():
    df = pd.read_csv(TABLES / "all_model_metrics.csv").sort_values("rmse")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [_model_color(m) for m in df["model"]]
    bars = ax.bar([MODEL_LABELS.get(m, m) for m in df["model"]], df["rmse"], color=colors, width=0.6)
    _bar_labels(ax, bars, fmt="{:.5f}")
    ax.set_ylim(df["rmse"].min() * 0.995, df["rmse"].max() * 1.01)
    ax.set_ylabel("RMSE (daily return)")
    ax.set_title("Out-of-Sample Return Forecasting Accuracy")
    ax.set_xlabel(None)
    _legend_family(ax)
    _source_caption(ax, f"Source: reports/tables/all_model_metrics.csv  |  n={int(df['n_obs'].iloc[0]):,} predictions/model")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_rmse_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── 2. Directional accuracy ──────────────────────────────────────────────
def fig_directional_accuracy():
    df = pd.read_csv(TABLES / "all_model_metrics.csv").sort_values("directional_accuracy")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [_model_color(m) for m in df["model"]]
    vals = df["directional_accuracy"] * 100
    bars = ax.bar([MODEL_LABELS.get(m, m) for m in df["model"]], vals, color=colors, width=0.6)
    _bar_labels(ax, bars, fmt="{:.2f}%")
    ax.axhline(50, color=COLOR_BAD, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.annotate("50% — coin flip", xy=(0.01, 50), xycoords=("axes fraction", "data"),
                fontsize=9, color=COLOR_BAD, va="bottom")
    ax.set_ylim(min(45, vals.min() - 2), max(55, vals.max() + 2))
    ax.set_ylabel("Directional accuracy (%)")
    ax.set_title("Forecast Directional Accuracy")
    _legend_family(ax, loc="upper left")
    _source_caption(ax, "Source: reports/tables/all_model_metrics.csv")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_directional_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── 3. Portfolio Sharpe across cost tiers ────────────────────────────────
def fig_portfolio_sharpe_by_cost():
    df = pd.read_csv(TABLES / "portfolio_summary.csv")
    cost_tiers = sorted(df["cost_bps"].unique())
    models = (
        df[df["cost_bps"] == cost_tiers[0]].sort_values("sharpe_ratio", ascending=False)["model"].tolist()
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    n_models, n_tiers = len(models), len(cost_tiers)
    bar_w = 0.8 / n_tiers
    x = np.arange(n_models)
    cmap_tiers = plt.cm.Blues(np.linspace(0.45, 0.95, n_tiers))
    for i, cost in enumerate(cost_tiers):
        sub = df[df["cost_bps"] == cost].set_index("model").reindex(models)
        ax.bar(x + i * bar_w - 0.8 / 2 + bar_w / 2, sub["sharpe_ratio"], width=bar_w,
               color=cmap_tiers[i], label=f"{cost} bps")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m).replace("\n", " ") for m in models])
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Long-Short Portfolio Sharpe Ratio Across Transaction-Cost Tiers")
    ax.legend(title="Cost", frameon=False, fontsize=9, title_fontsize=9)
    _source_caption(ax, "Source: reports/tables/portfolio_summary.csv  |  Equal-weight top-10/bottom-10 long-short, daily rebalance")
    fig.tight_layout()
    fig.savefig(FIGURES / "portfolio_sharpe_by_cost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── 4. Diebold-Mariano forest plot ───────────────────────────────────────
def fig_diebold_mariano():
    df = pd.read_csv(TABLES / "diebold_mariano_hac_results.csv").sort_values("dm_stat")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(df))
    colors = [COLOR_GOOD if p < 0.05 else (COLOR_DL if m in DL_MODELS else COLOR_CLASSICAL)
              for m, p in zip(df["model"], df["p_value"])]
    ax.hlines(y, 0, df["dm_stat"], color=colors, linewidth=2.5, alpha=0.85)
    ax.scatter(df["dm_stat"], y, color=colors, s=80, zorder=3)
    for yi, (stat, p) in enumerate(zip(df["dm_stat"], df["p_value"])):
        ax.annotate(f"p={p:.3f}", (stat, yi), xytext=(0, 14),
                    textcoords="offset points", va="bottom", ha="center", fontsize=9, color="#333333")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m).replace("\n", " ") for m in df["model"]])
    ax.margins(y=0.18)
    ax.set_xlabel("DM statistic  (positive = beats mean baseline)")
    ax.set_title("Diebold-Mariano Test vs. Mean Baseline (Newey-West HAC)")
    _source_caption(ax, "Source: reports/tables/diebold_mariano_hac_results.csv  |  green = significant at 5%")
    fig.tight_layout()
    fig.savefig(FIGURES / "diebold_mariano_forest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── 5. Fama-French-Carhart alpha ─────────────────────────────────────────
def fig_ff_alpha():
    path = TABLES / "ff_carhart_alpha.csv"
    if not path.exists():
        print("  ff_carhart_alpha.csv not found — skipping fama_french_alpha.png "
              "(run step4_downstream.py with network access to Ken French's data library)")
        return
    df = pd.read_csv(path).sort_values("annualised_alpha")
    if df.empty:
        print("  ff_carhart_alpha.csv is empty — skipping fama_french_alpha.png")
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [_model_color(m) for m in df["model"]]
    bars = ax.bar([MODEL_LABELS.get(m, m) for m in df["model"]], df["annualised_alpha"], color=colors, width=0.6)
    _bar_labels(ax, bars, fmt="{:+.3f}")
    ax.margins(y=0.15)
    for b, p in zip(bars, df["alpha_pvalue"]):
        if p < 0.05:
            ax.annotate("*", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 14 if b.get_height() >= 0 else -18), textcoords="offset points",
                        ha="center", fontsize=16, color=COLOR_BAD, fontweight="bold")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Annualised alpha")
    ax.set_title("Fama-French-Carhart Four-Factor Alpha (5bps cost)")
    _legend_family(ax, loc="upper left")
    _source_caption(ax, "Source: reports/tables/ff_carhart_alpha.csv  |  * significant at 5% (HAC)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fama_french_alpha.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── 6. RMT eigenvalue spectrum ───────────────────────────────────────────
def fig_rmt_spectrum():
    path = TABLES / "rmt_100_200_300_spectrum_summary.csv"
    if not path.exists():
        print("  rmt_100_200_300_spectrum_summary.csv not found — skipping rmt_eigenvalue_spectrum.png")
        return
    df = pd.read_csv(path).sort_values("actual_stocks")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(df))
    width = 0.55
    signal = df["signal_eigenvalues"]
    noise = df["noise_like_eigenvalues"]
    ax.bar(x, signal, width, label="Signal eigenvalues", color=COLOR_GOOD)
    ax.bar(x, noise, width, bottom=signal, label="Noise-like eigenvalues", color=COLOR_NEUTRAL, alpha=0.7)
    for xi, (s, n, tot) in enumerate(zip(signal, noise, signal + noise)):
        ax.annotate(f"{s} signal ({100*s/tot:.1f}%)", (xi, s), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=COLOR_GOOD, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top {int(n)}" for n in df["actual_stocks"]])
    ax.set_ylabel("Eigenvalues")
    ax.set_title("Marchenko-Pastur RMT Denoising: Signal vs. Noise-Like Eigenvalues")
    ax.legend(frameon=False, fontsize=9)
    _source_caption(ax, "Source: reports/tables/rmt_100_200_300_spectrum_summary.csv  |  756-day window, 2019-12-31 anchor")
    fig.tight_layout()
    fig.savefig(FIGURES / "rmt_eigenvalue_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_rmse()
    fig_directional_accuracy()
    fig_portfolio_sharpe_by_cost()
    fig_diebold_mariano()
    fig_ff_alpha()
    fig_rmt_spectrum()
    print(f"✓ Dissertation figures generated -> {FIGURES}")
