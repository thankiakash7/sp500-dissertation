"""
Step 4: Downstream analysis.
Merges all predictions → DM+MCS → portfolio → vol-scaled → FF alpha.
Run after step1, step2, step3 are complete.
TFT is optional — if its predictions file is missing, runs on 5 models.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import config

DATA_PROCESSED = config.DATA_PROCESSED
DATA_RAW       = config.DATA_RAW
REPORT_TABLES  = config.REPORT_TABLES
SUFFIX         = config.OUTPUT_SUFFIX

N_LONG  = 10
N_SHORT = 10
TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# 1. MERGE
# ─────────────────────────────────────────────────────────────────────────────

def merge_predictions() -> pd.DataFrame:
    print("\n[1] Merging predictions...")
    parts = []

    classical = DATA_PROCESSED / f"walk_forward_classical_predictions_top100{SUFFIX}.parquet"
    dl        = DATA_PROCESSED / f"lstm_gru_predictions_top100{SUFFIX}.parquet"
    tft_path  = DATA_PROCESSED / f"tft_predictions_top100{SUFFIX}.parquet"

    for p in [classical, dl]:
        assert p.exists(), f"Missing: {p.name} — run step1/step2 first"
        parts.append(pd.read_parquet(p))

    if tft_path.exists():
        parts.append(pd.read_parquet(tft_path))
        print("  TFT predictions found — including 6 models")
    else:
        print("  TFT predictions NOT found — running with 5 models (add TFT later)")

    all_preds = pd.concat(parts, ignore_index=True)
    all_preds["date"] = pd.to_datetime(all_preds["date"])
    all_preds = all_preds.dropna(subset=["actual", "forecast"])

    # align: keep only (date, permno) present in ALL models
    pivot = all_preds.pivot_table(
        index=["date", "permno"], columns="model", values="forecast", aggfunc="first"
    )
    complete = pivot.dropna().index
    aligned = all_preds.set_index(["date", "permno"]).loc[complete].reset_index()

    out = DATA_PROCESSED / f"all_model_predictions{SUFFIX}.parquet"
    aligned.to_parquet(out, index=False)
    models = sorted(aligned["model"].unique())
    print(f"  Models: {models}")
    print(f"  {aligned['permno'].nunique()} permnos | {aligned['date'].nunique()} dates | {len(aligned)} rows → {out.name}")
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# 2. DM + MCS
# ─────────────────────────────────────────────────────────────────────────────

def run_dm_mcs(aligned: pd.DataFrame):
    print("\n[2] DM + MCS...")
    from arch.bootstrap import MCS

    baseline = "mean_baseline"
    models   = sorted(aligned["model"].unique())

    aligned = aligned.copy()
    aligned["sq_error"] = (aligned["actual"] - aligned["forecast"]) ** 2
    aligned["abs_error"] = aligned["sq_error"] ** 0.5

    # ── per-model metrics ────────────────────────────────────────────────────
    metric_rows = []
    for m, grp in aligned.groupby("model"):
        rmse = float(np.sqrt(grp["sq_error"].mean()))
        mae  = float(grp["abs_error"].mean())
        da   = float((np.sign(grp["actual"]) == np.sign(grp["forecast"])).mean())
        metric_rows.append({"model": m, "rmse": rmse, "mae": mae,
                            "directional_accuracy": da, "n_obs": len(grp)})
    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(REPORT_TABLES / f"all_model_metrics{SUFFIX}.csv", index=False)

    # ── DM test (Newey-West HAC, cross-sectionally aggregated) ──────────────
    errors = aligned.pivot_table(
        index=["date", "permno"], columns="model", values="sq_error", aggfunc="first"
    )

    dm_rows = []
    for m in models:
        if m == baseline:
            continue
        common = errors[[baseline, m]].dropna()
        d = common[baseline] - common[m]                  # positive = m is better
        d_daily = d.groupby("date").mean()                # cross-sectional avg

        nobs = len(d_daily)
        # Newey-West HAC variance of the daily loss differential
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
        nlags = int(np.ceil(nobs ** (1/3)))
        d_vals = d_daily.values
        ones   = np.ones((nobs, 1))
        ols    = OLS(d_vals, ones).fit()
        nw_se  = ols.get_robustcov_results(cov_type="HAC", maxlags=nlags).bse[0]
        se     = float(nw_se)
        dm_stat = float(d_daily.mean()) / se if se > 0 else np.nan
        p_val   = float(2 * stats.t.sf(abs(dm_stat), df=nobs - 1))

        dm_rows.append({
            "model": m, "vs_baseline": baseline,
            "dm_stat": round(dm_stat, 4), "p_value": round(p_val, 4),
            "n_days": nobs,
            "interpretation": "beats baseline" if (dm_stat > 0 and p_val < 0.05)
                               else ("worse" if dm_stat < 0 else "no sig diff")
        })
        print(f"  DM {m:20s} vs {baseline}: stat={dm_stat:+.3f}  p={p_val:.4f}")

    dm_df = pd.DataFrame(dm_rows)
    dm_df.to_csv(REPORT_TABLES / f"diebold_mariano_hac_results{SUFFIX}.csv", index=False)

    # ── MCS ─────────────────────────────────────────────────────────────────
    daily_mse = aligned.groupby(["date", "model"])["sq_error"].mean().unstack()
    daily_mse = daily_mse.dropna()
    try:
        mcs = MCS(daily_mse, size=0.10, reps=5000, block_size=10, method="R", seed=42)
        mcs.compute()
        mcs.pvalues.to_csv(REPORT_TABLES / f"model_confidence_set_pvalues{SUFFIX}.csv")
        print(f"  MCS included: {list(mcs.included)}")
    except Exception as e:
        print(f"  MCS failed: {e}")

    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# 3. PORTFOLIO BACKTEST (equal-weight long-short)
# ─────────────────────────────────────────────────────────────────────────────

def _make_weights(group: pd.DataFrame, n_long: int, n_short: int) -> pd.Series:
    group = group.sort_values("forecast").copy()
    top_k = min(n_long, len(group) // 2)
    if top_k == 0:
        return pd.Series(0.0, index=group.index)
    w = pd.Series(0.0, index=group.index)
    w.loc[group.tail(top_k).index] =  1.0 / top_k
    w.loc[group.head(top_k).index] = -1.0 / top_k
    return w


def _build_portfolio(mf: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    rate = cost_bps / 10_000
    all_stocks = sorted(mf["permno"].unique())
    prev_w = pd.Series(0.0, index=all_stocks)
    rows = []
    for date, daily in mf.sort_values(["date", "permno"]).groupby("date"):
        daily = daily.set_index("permno").reindex(all_stocks)
        cur_w = daily["weight"].fillna(0.0)
        gross = float((cur_w * daily["actual"].fillna(0.0)).sum())
        delta = float((cur_w - prev_w).abs().sum())
        rows.append({
            "date": date,
            "gross_return": gross,
            "trading_cost": rate * delta,
            "net_return": gross - rate * delta,
            "turnover": 0.5 * delta,
        })
        prev_w = cur_w
    return pd.DataFrame(rows)


def _portfolio_metrics(grp: pd.DataFrame) -> pd.Series:
    d = grp["net_return"].dropna()
    ann_ret = d.mean() * TRADING_DAYS
    ann_vol = d.std()  * np.sqrt(TRADING_DAYS)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    wealth  = (1 + d).cumprod()
    mdd     = float((wealth / wealth.cummax() - 1).min())
    return pd.Series({
        "n_days": len(d),
        "annualised_return": round(ann_ret, 6),
        "annualised_volatility": round(ann_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "maximum_drawdown": round(mdd, 4),
        "avg_daily_turnover": round(grp["turnover"].mean(), 4),
    })


def run_portfolio(aligned: pd.DataFrame):
    print("\n[3] Equal-weight long-short portfolio...")

    weighted_parts = []
    for (date, model), grp in aligned.groupby(["date", "model"]):
        grp = grp.copy()
        grp["weight"] = _make_weights(grp, N_LONG, N_SHORT)
        weighted_parts.append(grp)
    weighted = pd.concat(weighted_parts, ignore_index=True)

    frames = []
    for model_name, mg in weighted.groupby("model"):
        for cost_bps in [0, 5, 10, 25]:
            r = _build_portfolio(mg, cost_bps)
            r["model"] = model_name
            r["cost_bps"] = cost_bps
            frames.append(r)

    port_daily = pd.concat(frames, ignore_index=True)
    port_daily.to_parquet(DATA_PROCESSED / f"portfolio_daily_results{SUFFIX}.parquet", index=False)

    summary = (
        port_daily.groupby(["model", "cost_bps"])
        .apply(_portfolio_metrics)
        .reset_index()
        .sort_values(["cost_bps", "sharpe_ratio"], ascending=[True, False])
    )
    summary.to_csv(REPORT_TABLES / f"portfolio_summary{SUFFIX}.csv", index=False)
    print(summary[["model","cost_bps","annualised_return","sharpe_ratio","maximum_drawdown"]]
          .to_string(index=False))
    return port_daily


# ─────────────────────────────────────────────────────────────────────────────
# 4. VOLATILITY-SCALED PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

def run_vol_scaled(aligned: pd.DataFrame):
    print("\n[4] Volatility-scaled long-short portfolio...")

    # Volatility used to scale position sizes. Prefer the AR(1)-GARCH-t
    # model's own one-step-ahead conditional volatility forecast — it's a
    # genuine forward-looking prediction, fitted every walk-forward block
    # in step1_classical.py and previously computed there but never used
    # downstream. Fall back to realised (backward-looking, 21-day rolling)
    # volatility of actual returns for any (date, permno) it doesn't cover.
    garch_vol_path = DATA_PROCESSED / f"garch_conditional_vol_top100{SUFFIX}.parquet"
    ret_wide = (
        aligned[aligned["model"] == aligned["model"].iloc[0]]
        [["date", "permno", "actual"]]
        .pivot(index="date", columns="permno", values="actual")
        .sort_index()
    )
    realised_vol = ret_wide.rolling(21, min_periods=5).std()

    if garch_vol_path.exists():
        garch_vol = pd.read_parquet(garch_vol_path)
        garch_vol["date"] = pd.to_datetime(garch_vol["date"])
        garch_vol_wide = garch_vol.pivot(index="date", columns="permno", values="garch_cond_vol")
        roll_vol = garch_vol_wide.reindex(index=realised_vol.index, columns=realised_vol.columns)
        n_cells = realised_vol.shape[0] * realised_vol.shape[1]
        n_from_garch = int(roll_vol.notna().sum().sum())
        roll_vol = roll_vol.fillna(realised_vol)
        n_still_missing = n_cells - int(roll_vol.notna().sum().sum())
        print(f"  Vol source: {n_from_garch}/{n_cells} cells from GARCH forecast, "
              f"{n_cells - n_from_garch - n_still_missing} from realised 21-day rolling vol, "
              f"{n_still_missing} still missing")
    else:
        roll_vol = realised_vol
        print("  garch_conditional_vol_top100.parquet not found — "
              "using realised 21-day rolling volatility only (run step1_classical.py to produce it)")

    def vol_weights(group: pd.DataFrame, n_long: int, n_short: int) -> pd.Series:
        group = group.sort_values("forecast").copy()
        top_k = min(n_long, len(group) // 2)
        if top_k == 0:
            return pd.Series(0.0, index=group.index)

        w = pd.Series(0.0, index=group.index)
        w.loc[group.tail(top_k).index] =  1.0
        w.loc[group.head(top_k).index] = -1.0

        date = group["date"].iloc[0]
        if date in roll_vol.index:
            vol_row = roll_vol.loc[date]
            inv_v   = group["permno"].map(
                lambda p: 1.0 / vol_row.get(p, np.nan) if (not pd.isna(vol_row.get(p, np.nan)) and vol_row.get(p, np.nan) > 0) else 1.0
            )
            w = w * inv_v.values

        # renormalise legs
        lm = w > 0; sm2 = w < 0
        if w[lm].sum() > 0:  w[lm]  /= w[lm].sum()
        if w[sm2].abs().sum() > 0:
            w[sm2] /= w[sm2].abs().sum()
            w[sm2] *= -1
        return w

    weighted_parts = []
    for (date, model), grp in aligned.groupby(["date", "model"]):
        grp = grp.copy()
        grp["weight"] = vol_weights(grp, N_LONG, N_SHORT)
        weighted_parts.append(grp)
    weighted = pd.concat(weighted_parts, ignore_index=True)

    frames = []
    for model_name, mg in weighted.groupby("model"):
        for cost_bps in [0, 5, 10, 25]:
            r = _build_portfolio(mg, cost_bps)
            r["model"] = model_name
            r["cost_bps"] = cost_bps
            frames.append(r)

    vol_daily = pd.concat(frames, ignore_index=True)
    vol_daily.to_parquet(DATA_PROCESSED / f"portfolio_vol_scaled_daily{SUFFIX}.parquet", index=False)

    summary = (
        vol_daily.groupby(["model", "cost_bps"])
        .apply(_portfolio_metrics)
        .reset_index()
        .sort_values(["cost_bps", "sharpe_ratio"], ascending=[True, False])
    )
    summary.to_csv(REPORT_TABLES / f"portfolio_vol_scaled_summary{SUFFIX}.csv", index=False)
    print(summary[["model","cost_bps","annualised_return","sharpe_ratio","maximum_drawdown"]]
          .to_string(index=False))
    return vol_daily


# ─────────────────────────────────────────────────────────────────────────────
# 5. FAMA-FRENCH / CARHART ALPHA
# ─────────────────────────────────────────────────────────────────────────────

def _download_ff_factors() -> pd.DataFrame:
    """
    Download Fama-French 3 factors + Carhart momentum daily factors.
    """

    import io
    import zipfile
    import requests

    urls = {
        "ff3": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
        "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
    }

    dfs = {}

    for name, url in urls.items():

        print(f"  Downloading {name} factors...")

        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            raise Exception(f"Download failed: {response.status_code}")

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:

            csv_file = [
                f for f in z.namelist()
                if f.lower().endswith(".csv")
            ][0]

            raw = z.read(csv_file).decode("latin1")


        lines = raw.splitlines()

        # find first data row
        start = None

        for i, line in enumerate(lines):
            if line[:4].isdigit():
                start = i
                break

        if start is None:
            raise Exception("Could not locate data section")


        data = []

        for line in lines[start:]:

            if not line[:4].isdigit():
                break

            parts = line.split(",")

            data.append(parts)


        if name == "ff3":

            df = pd.DataFrame(
                data,
                columns=[
                    "date",
                    "MKT_RF",
                    "SMB",
                    "HML",
                    "RF"
                ]
            )

        else:

            df = pd.DataFrame(
                data,
                columns=[
                    "date",
                    "MOM",
                    "RF_mom"
                ]
            )


        df["date"] = pd.to_datetime(
            df["date"],
            format="%Y%m%d"
        )

        numeric_cols = [
            c for c in df.columns
            if c != "date"
        ]

        for c in numeric_cols:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

        dfs[name] = df


    factors = dfs["ff3"].merge(
        dfs["mom"][["date","MOM"]],
        on="date",
        how="inner"
    )


    factors[
        [
            "MKT_RF",
            "SMB",
            "HML",
            "MOM",
            "RF"
        ]
    ] /= 100


    return factors


def run_ff_alpha(port_daily: pd.DataFrame):
    print("\n[5] Fama-French / Carhart alpha regressions...")

    ff_path = DATA_RAW / "ff_carhart_daily.csv"
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if ff_path.exists():
        factors = pd.read_csv(ff_path, parse_dates=["date"])
        print(f"  Loaded FF factors from {ff_path.name}")
    else:
        print("  Downloading FF+Carhart daily factors...")
        try:
            factors = _download_ff_factors()
        except Exception as e:
            print(f"  Could not download FF factors ({e}) — skipping alpha regression.")
            print(f"  To run this offline: download the daily FF3 and Momentum CSVs from Ken French's "
                  f"data library by hand and save the merged result to {ff_path}, "
                  f"columns [date, MKT_RF, SMB, HML, MOM, RF].")
            return
        if factors.empty:
            print("  Could not obtain FF factors — skipping alpha regression")
            return
        factors.to_csv(ff_path, index=False)
        print(f"  Saved → {ff_path.name}")

    port_daily = port_daily.copy()
    port_daily["date"] = pd.to_datetime(port_daily["date"])

    alpha_rows = []
    for cost_bps in [5]:          # report 5bp as headline
        subset = port_daily[port_daily["cost_bps"] == cost_bps].copy()
        for model_name, grp in subset.groupby("model"):
            reg = grp.merge(factors, on="date", how="inner").dropna(
                subset=["net_return","MKT_RF","SMB","HML","MOM","RF"]
            )
            if len(reg) < 60:
                print(f"  {model_name}: too few obs ({len(reg)}) — skip")
                continue

            reg["excess"] = reg["net_return"] - reg["RF"]
            X = sm.add_constant(reg[["MKT_RF","SMB","HML","MOM"]])
            nlags = int(np.ceil(len(reg) ** (1/3)))
            fit   = sm.OLS(reg["excess"], X).fit(
                cov_type="HAC", cov_kwds={"maxlags": nlags}
            )
            alpha_ann = float(fit.params["const"]) * TRADING_DAYS
            alpha_p   = float(fit.pvalues["const"])
            alpha_rows.append({
                "model":             model_name,
                "cost_bps":          cost_bps,
                "daily_alpha":       round(float(fit.params["const"]), 6),
                "annualised_alpha":  round(alpha_ann, 4),
                "alpha_tstat":       round(float(fit.tvalues["const"]), 4),
                "alpha_pvalue":      round(alpha_p, 4),
                "beta_mkt":          round(float(fit.params["MKT_RF"]), 4),
                "n_obs":             len(reg),
                "r_squared":         round(fit.rsquared, 4),
            })
            sig = "**" if alpha_p < 0.05 else ("*" if alpha_p < 0.10 else "")
            print(f"  {model_name:20s}: α_ann={alpha_ann:+.4f}  p={alpha_p:.4f} {sig}")

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(REPORT_TABLES / f"ff_carhart_alpha{SUFFIX}.csv", index=False)
    print(f"  Saved → ff_carhart_alpha.csv")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    aligned   = merge_predictions()
    aligned   = run_dm_mcs(aligned)
    port_daily= run_portfolio(aligned)
    run_vol_scaled(aligned)
    run_ff_alpha(port_daily)

    print("\n" + "="*60)
    print("✓  Step 4 complete.")
    print("   Outputs in reports/tables/:")
    for f in sorted(REPORT_TABLES.glob("*.csv")):
        rows = sum(1 for _ in open(f)) - 1
        print(f"   {f.name:45s}  ({rows} rows)")
