import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. MODEL RMSE COMPARISON
# ============================================================

# Load model performance metrics
metrics = pd.read_csv("reports/tables/all_model_metrics.csv")

# Sort models from lowest to highest RMSE
metrics = metrics.sort_values("rmse").reset_index(drop=True)


# Colours for the RMSE chart
# Green = best model / mean baseline
# Red = LSTM and GRU
# Blue-grey = classical alternatives / TFT
rmse_colors = [
    "#2C6E52" if model == "mean_baseline"
    else "#A13E3E" if model in ("lstm", "gru")
    else "#5B6B8C"
    for model in metrics["model"]
]


# Human-readable model names
rmse_labels = {
    "mean_baseline": "Mean baseline",
    "arma": "ARMA",
    "ar1_garch_t": "AR(1)-GARCH-t",
    "tft": "TFT",
    "lstm": "LSTM",
    "gru": "GRU",
}


# Create RMSE figure
fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.bar(
    [rmse_labels[m] for m in metrics["model"]],
    metrics["rmse"],
    color=rmse_colors
)


# IMPORTANT:
# Do not start the y-axis at zero because the RMSE values
# are very close together. Starting at zero would make the
# differences almost invisible.
ax.set_ylim(
    metrics["rmse"].min() * 0.999,
    metrics["rmse"].max() * 1.001
)


# Add numerical RMSE values above each bar
for bar, value in zip(bars, metrics["rmse"]):
    ax.annotate(
        f"{value:.5f}",
        (
            bar.get_x() + bar.get_width() / 2,
            value
        ),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )


# Identify the best model
best_idx = metrics["rmse"].idxmin()
best_position = metrics.index.get_loc(best_idx)
best_rmse = metrics.loc[best_idx, "rmse"]


# Highlight the lowest RMSE
ax.annotate(
    "Lowest RMSE",
    xy=(best_position, best_rmse),
    xytext=(0, 40),
    textcoords="offset points",
    ha="center",
    fontweight="bold",
    color="#2C6E52",
    arrowprops=dict(
        arrowstyle="->",
        color="#2C6E52"
    )
)


ax.set_ylabel("Root Mean Squared Error (RMSE)")
ax.set_title("Out-of-Sample Forecast Accuracy by Model")
ax.set_xlabel("Model")


fig.tight_layout()

fig.savefig(
    "reports/figures/model_rmse_comparison_HIGHLIGHTED.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "Saved to reports/figures/"
    "model_rmse_comparison_HIGHLIGHTED.png"
)


# ============================================================
# 2. CUMULATIVE LONG-SHORT PORTFOLIO WEALTH
# ============================================================

# Model colours
COLORS = {
    "mean_baseline": "#3B6E8F",
    "arma": "#5B6B8C",
    "ar1_garch_t": "#5B6B8C",
    "tft": "#E07B39",
    "lstm": "#A13E3E",
    "gru": "#E07B39",
}


# Human-readable model labels
LABELS = {
    "mean_baseline": "Mean baseline",
    "arma": "ARMA",
    "ar1_garch_t": "AR(1)-GARCH-t",
    "tft": "TFT",
    "lstm": "LSTM",
    "gru": "GRU",
}


# Load daily portfolio results
portfolio = pd.read_parquet(
    "data/processed/portfolio_daily_results.parquet"
)

portfolio["date"] = pd.to_datetime(portfolio["date"])


# Keep only the 5 basis-point transaction-cost scenario
d5 = portfolio[
    portfolio["cost_bps"] == 5
].copy()


# Create portfolio figure
fig, ax = plt.subplots(figsize=(10, 6))


# Plot each model
for model in [
    "mean_baseline",
    "arma",
    "ar1_garch_t",
    "gru",
    "lstm",
    "tft"
]:

    sub = (
        d5[d5["model"] == model]
        .sort_values("date")
    )

    # Start with £1 (or 1 unit) and compound daily net returns
    wealth = (
        1 + sub["net_return"]
    ).cumprod()

    ax.plot(
        sub["date"],
        wealth,
        label=LABELS[model],
        color=COLORS[model],
        linewidth=1.6
    )


# Add legend
ax.legend(
    loc="upper left",
    frameon=False
)


ax.set_ylabel("Growth of 1 unit")
ax.set_xlabel("Date")

ax.set_title(
    "Cumulative Long-Short Wealth "
    "After 5bps One-Way Costs"
)


# Reference line showing the starting wealth
ax.axhline(
    1.0,
    color="#333333",
    linewidth=0.6
)


fig.tight_layout()

fig.savefig(
    "reports/figures/portfolio_cumulative_returns_FIXED.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "Saved to reports/figures/"
    "portfolio_cumulative_returns_FIXED.png"
)


# ============================================================
# 3. FINISHED
# ============================================================

print("\nBoth figures have been generated successfully.")