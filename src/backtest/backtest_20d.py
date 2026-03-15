"""
Stage 4 Extension — 20-Day Holding Period Backtest

Post-earnings announcement drift (PEAD) is documented at 20-60 day
horizons (Bernard & Thomas 1989, Chan et al. 1996). This backtest
tests whether FinSight's signal is exploitable at a longer horizon
where transaction costs matter less per unit of signal.

Key difference from 5-day backtest:
  - Target: target_20d_up (20-day binary)
  - Return: ret_20d (20-day return)
  - TC: still 10bps but amortized over 4x longer holding period

Output:
  experiments/backtest_20d_results.csv
  experiments/backtest_20d_summary.txt
  experiments/equity_curve_20d.png
  experiments/backtest_comparison.png   <- 5d vs 20d side by side

Usage:
    python src/backtest/backtest_20d.py
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, EXPERIMENTS_DIR, TRANSACTION_COST_BPS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

FINBERT_FEATURES = [
    "mgmt_mean_pos","mgmt_mean_neg","mgmt_mean_neu",
    "mgmt_net_sentiment","mgmt_neg_ratio","mgmt_sent_vol","mgmt_n_sentences",
    "qa_mean_pos","qa_mean_neg","qa_mean_neu",
    "qa_net_sentiment","qa_neg_ratio","qa_sent_vol","qa_n_sentences",
]
RAG_FEATURES = [
    "rag_guidance_specificity_score","rag_guidance_specificity_relevance",
    "rag_new_risks_score","rag_new_risks_relevance",
    "rag_management_confidence_score","rag_management_confidence_relevance",
    "rag_forward_looking_score","rag_forward_looking_relevance",
    "rag_cost_pressure_score","rag_cost_pressure_relevance",
]
ALL_FEATURES = FINBERT_FEATURES + RAG_FEATURES

TARGET_20D   = "target_20d_up"
TARGET_5D    = "target_5d_up"
RETURN_20D   = "ret_20d"
RETURN_5D    = "ret_5d"
TC           = TRANSACTION_COST_BPS / 10_000
TRAIN_WINDOW = 3
TEST_YEARS   = [2021, 2022, 2023, 2024]


# ── Walk-forward predictions ───────────────────────────────────────────────────

def generate_predictions(df: pd.DataFrame, target: str) -> pd.DataFrame:
    avail = [c for c in ALL_FEATURES if c in df.columns]
    all_preds = []

    for test_year in TEST_YEARS:
        train_years = list(range(test_year - TRAIN_WINDOW, test_year))
        train = df[df["year"].isin(train_years)].dropna(subset=avail + [target])
        test  = df[df["year"] == test_year].dropna(subset=avail + [RETURN_20D, RETURN_5D])

        if len(train) < 50 or len(test) < 20:
            continue

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train[avail].values)
        X_test  = scaler.transform(test[avail].values)

        model = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        model.fit(X_train, train[target].values)
        proba = model.predict_proba(X_test)[:, 1]

        pred_df = test[["ticker","year","quarter",
                         RETURN_5D, RETURN_20D]].copy()
        pred_df["prob"] = proba
        all_preds.append(pred_df)
        logger.info(f"  {test_year}: {len(test)} predictions generated")

    return pd.concat(all_preds, ignore_index=True)


# ── Portfolio construction ─────────────────────────────────────────────────────

def build_portfolio(predictions: pd.DataFrame, return_col: str) -> pd.DataFrame:
    results = []
    for (year, quarter), group in predictions.groupby(["year","quarter"]):
        if len(group) < 8:
            continue
        group  = group.sort_values("prob", ascending=False)
        n      = len(group)
        q_size = max(1, n // 4)

        long_leg  = group.iloc[:q_size]
        short_leg = group.iloc[-q_size:]

        long_ret  = long_leg[return_col].mean()
        short_ret = short_leg[return_col].mean()
        net_ret   = (long_ret - short_ret) - (4 * TC)

        results.append({
            "year":      year,
            "quarter":   quarter,
            "long_ret":  round(long_ret,  4),
            "short_ret": round(short_ret, 4),
            "net_ret":   round(net_ret,   4),
            "long_hit":  round((long_leg[return_col] > 0).mean(),  4),
            "short_hit": round((short_leg[return_col] < 0).mean(), 4),
            "n_stocks":  n,
            "q_size":    q_size,
        })
    return pd.DataFrame(results)


# ── Performance metrics ────────────────────────────────────────────────────────

def compute_metrics(rets: pd.Series, label: str) -> dict:
    n_years  = len(rets) / 4
    ann_ret  = float((1 + rets).prod() ** (1/n_years) - 1)
    ann_vol  = float(rets.std() * np.sqrt(4))
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside     = rets[rets < 0]
    sortino_vol  = float(downside.std() * np.sqrt(4)) if len(downside) > 1 else np.nan
    sortino      = ann_ret / sortino_vol if (sortino_vol and sortino_vol > 0) else np.nan

    cum    = (1 + rets).cumprod()
    peak   = cum.cummax()
    dd     = (cum - peak) / peak
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    hit    = float((rets > 0).mean())

    # t-test: is mean return significantly different from zero?
    t_stat, p_value = stats.ttest_1samp(rets, 0)

    return {
        "label":      label,
        "ann_return": round(ann_ret,  4),
        "ann_vol":    round(ann_vol,  4),
        "sharpe":     round(sharpe,   4),
        "sortino":    round(sortino,  4),
        "max_dd":     round(max_dd,   4),
        "calmar":     round(calmar,   4),
        "hit_rate":   round(hit,      4),
        "t_stat":     round(float(t_stat),  4),
        "p_value":    round(float(p_value), 4),
        "n_quarters": len(rets),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_equity_curve(portfolio: pd.DataFrame,
                      label: str, color: str,
                      out_path: Path) -> None:
    pt = portfolio.sort_values(["year","quarter"]).reset_index(drop=True)
    pt["period"] = pt["year"].astype(str) + "-Q" + pt["quarter"].astype(str)
    rets = pt["net_ret"]
    cum  = (1 + rets).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)
    ax1.plot(range(len(cum)), cum.values, color=color,
             linewidth=2.5, label=label)
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.fill_between(range(len(cum)), cum.values, 1.0,
                     where=cum.values >= 1.0, alpha=0.15, color=color)
    ax1.fill_between(range(len(cum)), cum.values, 1.0,
                     where=cum.values < 1.0, alpha=0.15, color="#C62828")
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title(f"FinSight — {label}\n10bps TC | Walk-Forward 2021–2024")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(range(len(dd)), dd.values, 0,
                     color="#C62828", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    tick_step = max(1, len(pt) // 8)
    ax2.set_xticks(range(0, len(pt), tick_step))
    ax2.set_xticklabels(pt["period"].iloc[::tick_step],
                        rotation=45, ha="right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(port_5d: pd.DataFrame,
                    port_20d: pd.DataFrame) -> None:
    """Side-by-side comparison of 5-day vs 20-day strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, port, label, color in [
        (axes[0], port_5d,  "5-Day Holding",  "#ef5350"),
        (axes[1], port_20d, "20-Day Holding", "#1565C0"),
    ]:
        pt   = port.sort_values(["year","quarter"]).reset_index(drop=True)
        rets = pt["net_ret"]
        cum  = (1 + rets).cumprod()
        pt["period"] = pt["year"].astype(str) + "-Q" + pt["quarter"].astype(str)

        bar_colors = ["#66bb6a" if r > 0 else "#ef5350" for r in rets]
        ax.bar(range(len(rets)), rets.values * 100,
               color=bar_colors, alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(range(len(cum)), cum.values,
                 color=color, linewidth=2.5, label="Cumulative")
        ax2.axhline(1.0, color="black", linestyle="--",
                    linewidth=0.8, alpha=0.5)
        ax2.set_ylabel("Cumulative Return", color=color)

        ax.set_xlabel("Quarter")
        ax.set_ylabel("Net Return (%)")
        ax.set_title(f"Long-Short Strategy\n{label} | 10bps TC")
        tick_step = max(1, len(pt) // 6)
        ax.set_xticks(range(0, len(pt), tick_step))
        ax.set_xticklabels(pt["period"].iloc[::tick_step],
                           rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle("FinSight — 5-Day vs 20-Day Holding Period Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_DIR / "backtest_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved backtest_comparison.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    fm_path = PROCESSED_DIR / "feature_matrix.parquet"
    if not fm_path.exists():
        logger.error("feature_matrix.parquet not found.")
        sys.exit(1)

    df = pd.read_parquet(fm_path)
    logger.info(f"Loaded {len(df):,} rows.")

    # ── 20-day predictions & portfolio ────────────────────────────────────────
    logger.info("\nGenerating 20-day walk-forward predictions...")
    preds_20d = generate_predictions(df, TARGET_20D)
    port_20d  = build_portfolio(preds_20d, RETURN_20D)
    port_20d.to_csv(EXPERIMENTS_DIR / "backtest_20d_results.csv", index=False)

    # ── 5-day portfolio for comparison (already exists but rebuild) ───────────
    logger.info("Generating 5-day walk-forward predictions for comparison...")
    preds_5d = generate_predictions(df, TARGET_5D)
    port_5d  = build_portfolio(preds_5d, RETURN_5D)

    # ── Metrics ───────────────────────────────────────────────────────────────
    m5  = compute_metrics(port_5d["net_ret"],  "5-Day Strategy")
    m20 = compute_metrics(port_20d["net_ret"], "20-Day Strategy")

    # ── Summary print ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINSIGHT — HOLDING PERIOD COMPARISON")
    print("Long-Short Quartile | 10bps TC | 2021–2024")
    print("="*60)
    print(f"{'Metric':<20} {'5-Day':>12} {'20-Day':>12}")
    print("-"*60)
    for key in ["ann_return","ann_vol","sharpe","sortino",
                "max_dd","calmar","hit_rate","t_stat","p_value"]:
        v5  = m5[key]
        v20 = m20[key]
        flag = " ✅" if key == "sharpe" and v20 > v5 else ""
        flag = flag or (" ✅" if key == "ann_return" and v20 > v5 else "")
        print(f"  {key:<18} {str(v5):>12} {str(v20):>12}{flag}")
    print("="*60)

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_lines = [
        "FINSIGHT — BACKTEST COMPARISON SUMMARY",
        "="*60,
        f"{'Metric':<20} {'5-Day':>12} {'20-Day':>12}",
        "-"*60,
    ]
    for key in ["ann_return","ann_vol","sharpe","sortino",
                "max_dd","calmar","hit_rate","t_stat","p_value"]:
        summary_lines.append(
            f"  {key:<18} {str(m5[key]):>12} {str(m20[key]):>12}")
    summary_lines.append("="*60)

    # 20-day quarterly breakdown
    summary_lines.append("\n20-Day Quarterly Breakdown:")
    summary_lines.append(
        port_20d[["year","quarter","net_ret",
                  "long_hit","short_hit","n_stocks"]]
        .to_string(index=False))

    with open(EXPERIMENTS_DIR / "backtest_20d_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_equity_curve(port_20d, "20-Day Long-Short Strategy",
                      "#1565C0",
                      EXPERIMENTS_DIR / "equity_curve_20d.png")
    plot_comparison(port_5d, port_20d)

    print(f"\nFiles saved:")
    print(f"  experiments/backtest_20d_results.csv")
    print(f"  experiments/backtest_20d_summary.txt")
    print(f"  experiments/equity_curve_20d.png")
    print(f"  experiments/backtest_comparison.png")

    # ── Key interpretation ────────────────────────────────────────────────────
    sharpe_5d  = m5["sharpe"]
    sharpe_20d = m20["sharpe"]
    ret_20d    = m20["ann_return"]

    print("\n" + "─"*60)
    print("INTERPRETATION")
    print("─"*60)
    if sharpe_20d > sharpe_5d:
        print(f"✅ Sharpe improved: {sharpe_5d} (5d) → {sharpe_20d} (20d)")
        print("   Consistent with PEAD: signal is more exploitable at")
        print("   longer horizons where TC is amortized over more signal.")
    else:
        print(f"   Sharpe: {sharpe_5d} (5d) → {sharpe_20d} (20d)")
    if ret_20d > 0:
        print(f"✅ Positive annualized return: {ret_20d*100:.2f}%")
    if m20["p_value"] < 0.1:
        print(f"✅ Statistically significant at 10% level (p={m20['p_value']})")
    print("─"*60)


if __name__ == "__main__":
    main()