"""
Stage 4 — Backtesting Engine

Strategy: Long-Short quartile portfolio
  - Each quarter, rank all stocks by predicted probability
  - Long top quartile (Q4), Short bottom quartile (Q1)
  - Hold for 5 trading days after earnings release
  - Apply transaction costs of 10bps per trade

Output:
  experiments/backtest_results.csv     — quarterly P&L
  experiments/equity_curve.png         — cumulative returns
  experiments/backtest_summary.txt     — Sharpe, drawdown, hit rate

Usage:
    python src/backtest/backtest_engine.py
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
TARGET      = "target_5d_up"
RETURN_COL  = "ret_5d"
TC          = TRANSACTION_COST_BPS / 10_000   # 0.001
TRAIN_WINDOW = 3
TEST_YEARS   = [2021, 2022, 2023, 2024]
TRADING_DAYS_PER_YEAR = 252


# ── Portfolio construction ─────────────────────────────────────────────────────

def build_quarterly_portfolio(
    predictions: pd.DataFrame,
    tc: float = TC,
) -> pd.DataFrame:
    """
    For each (year, quarter), rank stocks by predicted probability.
    Long top quartile, short bottom quartile.
    Net return = mean(long returns) - mean(short returns) - transaction costs.

    Parameters
    ----------
    predictions : DataFrame with columns [ticker, year, quarter, prob, ret_5d]
    tc : one-way transaction cost

    Returns
    -------
    DataFrame with one row per (year, quarter)
    """
    results = []

    for (year, quarter), group in predictions.groupby(["year", "quarter"]):
        if len(group) < 8:
            continue

        group = group.sort_values("prob", ascending=False)
        n     = len(group)
        q_size = max(1, n // 4)

        long_leg  = group.iloc[:q_size]
        short_leg = group.iloc[-q_size:]

        long_ret  = long_leg["ret_5d"].mean()
        short_ret = short_leg["ret_5d"].mean()

        # Long-short return minus round-trip transaction costs (2 legs × 2 sides)
        net_ret = (long_ret - short_ret) - (4 * tc)

        long_hit  = (long_leg["ret_5d"] > 0).mean()
        short_hit = (short_leg["ret_5d"] < 0).mean()

        results.append({
            "year":       year,
            "quarter":    quarter,
            "long_ret":   round(long_ret,  4),
            "short_ret":  round(short_ret, 4),
            "net_ret":    round(net_ret,   4),
            "long_hit":   round(long_hit,  4),
            "short_hit":  round(short_hit, 4),
            "n_stocks":   n,
            "q_size":     q_size,
        })

    return pd.DataFrame(results)


# ── Performance metrics ────────────────────────────────────────────────────────

def compute_metrics(quarterly_rets: pd.Series) -> dict:
    """
    Annualised Sharpe, Sortino, max drawdown, calmar, hit rate.
    Assumes 4 quarterly observations per year.
    """
    n_years = len(quarterly_rets) / 4
    ann_ret = (1 + quarterly_rets).prod() ** (1 / n_years) - 1
    ann_vol = quarterly_rets.std() * np.sqrt(4)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = quarterly_rets[quarterly_rets < 0]
    sortino_vol = downside.std() * np.sqrt(4)
    sortino = ann_ret / sortino_vol if sortino_vol > 0 else np.nan

    cum   = (1 + quarterly_rets).cumprod()
    peak  = cum.cummax()
    dd    = (cum - peak) / peak
    max_dd = dd.min()

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    hit    = (quarterly_rets > 0).mean()

    return {
        "ann_return":   round(float(ann_ret),  4),
        "ann_vol":      round(float(ann_vol),   4),
        "sharpe":       round(float(sharpe),    4),
        "sortino":      round(float(sortino),   4),
        "max_drawdown": round(float(max_dd),    4),
        "calmar":       round(float(calmar),    4),
        "hit_rate":     round(float(hit),       4),
        "n_quarters":   len(quarterly_rets),
    }


# ── Walk-forward prediction ────────────────────────────────────────────────────

def generate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward: for each test year, train LightGBM on prior 3 years,
    predict probability on test year. Returns full predictions DataFrame.
    """
    avail_feats = [c for c in ALL_FEATURES if c in df.columns]
    all_preds   = []

    for test_year in TEST_YEARS:
        train_years = list(range(test_year - TRAIN_WINDOW, test_year))
        train = df[df["year"].isin(train_years)].dropna(
            subset=avail_feats + [TARGET])
        test  = df[df["year"] == test_year].dropna(
            subset=avail_feats + [RETURN_COL])

        if len(train) < 50 or len(test) < 20:
            continue

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train[avail_feats].values)
        X_test  = scaler.transform(test[avail_feats].values)

        model = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        model.fit(X_train, train[TARGET].values)
        proba = model.predict_proba(X_test)[:, 1]

        pred_df = test[["ticker","year","quarter", RETURN_COL]].copy()
        pred_df["prob"] = proba
        all_preds.append(pred_df)

        logger.info(f"Predictions generated for {test_year}: {len(test)} stocks")

    return pd.concat(all_preds, ignore_index=True)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_equity_curve(
    portfolio: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Plot cumulative equity curve with drawdown panel."""
    portfolio = portfolio.sort_values(["year","quarter"]).reset_index(drop=True)
    rets  = portfolio["net_ret"]
    cum   = (1 + rets).cumprod()
    peak  = cum.cummax()
    dd    = (cum - peak) / peak

    labels = [f"{int(r.year)}-Q{int(r.quarter)}" for _, r in portfolio.iterrows()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)

    ax1.plot(range(len(cum)), cum.values, color="#1565C0", linewidth=2,
             label="Long-Short Strategy")
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.fill_between(range(len(cum)), cum.values, 1.0,
                     where=cum.values >= 1.0, alpha=0.15, color="#1565C0")
    ax1.fill_between(range(len(cum)), cum.values, 1.0,
                     where=cum.values < 1.0, alpha=0.15, color="#C62828")
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("FinSight — Long-Short Earnings Strategy\n"
                  f"10bps transaction cost | Walk-Forward 2021–2024")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(range(len(dd)), dd.values, 0,
                     color="#C62828", alpha=0.5, label="Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Quarter")
    ax2.grid(True, alpha=0.3)

    tick_step = max(1, len(labels) // 8)
    ax2.set_xticks(range(0, len(labels), tick_step))
    ax2.set_xticklabels(labels[::tick_step], rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved equity_curve.png")


def plot_quarterly_bars(portfolio: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of quarterly net returns."""
    portfolio = portfolio.sort_values(["year","quarter"]).reset_index(drop=True)
    labels = [f"{int(r.year)}-Q{int(r.quarter)}" for _, r in portfolio.iterrows()]
    colors = ["#1565C0" if r > 0 else "#C62828" for r in portfolio["net_ret"]]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(portfolio)), portfolio["net_ret"].values * 100, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net Return (%)")
    ax.set_title("Quarterly Net Returns — Long-Short Strategy (after 10bps TC)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "quarterly_returns.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved quarterly_returns.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    fm_path = PROCESSED_DIR / "feature_matrix.parquet"
    if not fm_path.exists():
        logger.error("feature_matrix.parquet missing.")
        sys.exit(1)

    df = pd.read_parquet(fm_path)
    logger.info(f"Loaded {len(df):,} rows.")

    # Generate predictions
    logger.info("Generating walk-forward predictions...")
    predictions = generate_predictions(df)
    logger.info(f"Total predictions: {len(predictions):,}")

    # Build portfolio
    portfolio = build_quarterly_portfolio(predictions)
    portfolio.to_csv(EXPERIMENTS_DIR / "backtest_results.csv", index=False)
    logger.info(f"Portfolio built: {len(portfolio)} quarters")

    # Compute metrics
    metrics = compute_metrics(portfolio["net_ret"])

    # Save summary
    summary_path = EXPERIMENTS_DIR / "backtest_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("FINSIGHT — BACKTEST SUMMARY\n")
        f.write("Long-Short Quartile | 10bps TC | 2021–2024\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:<20} {v}\n")
        f.write("=" * 50 + "\n")
        f.write("\nQuarterly breakdown:\n")
        f.write(portfolio[["year","quarter","net_ret","long_hit",
                            "short_hit","n_stocks"]].to_string(index=False))

    # Print summary
    print("\n" + "=" * 50)
    print("FINSIGHT — BACKTEST SUMMARY")
    print("Long-Short Quartile | 10bps TC | 2021–2024")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")
    print("=" * 50)
    print("\nQuarterly breakdown:")
    print(portfolio[["year","quarter","net_ret","long_hit",
                     "short_hit","n_stocks"]].to_string(index=False))

    # Plots
    plot_equity_curve(portfolio, EXPERIMENTS_DIR)
    plot_quarterly_bars(portfolio, EXPERIMENTS_DIR)

    print(f"\nFiles saved to: {EXPERIMENTS_DIR}")
    print("  backtest_results.csv   — per-quarter P&L")
    print("  backtest_summary.txt   — Sharpe, drawdown, metrics")
    print("  equity_curve.png       — cumulative return + drawdown")
    print("  quarterly_returns.png  — bar chart by quarter")


if __name__ == "__main__":
    main()