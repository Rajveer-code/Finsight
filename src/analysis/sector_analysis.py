"""
Sector-Level Signal Analysis

Research question: Are FinSight's NLP signals stronger in
information-asymmetric sectors (Healthcare, Tech) vs
commodity-like sectors (Energy, Utilities)?

Output:
  experiments/sector_ic.csv
  experiments/sector_ic.png
  experiments/sector_shap.png

Usage:
    python src/analysis/sector_analysis.py
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
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from lightgbm import LGBMClassifier
import shap

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, EXPERIMENTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_FEATURES = [
    "mgmt_mean_pos","mgmt_mean_neg","mgmt_mean_neu",
    "mgmt_net_sentiment","mgmt_neg_ratio","mgmt_sent_vol","mgmt_n_sentences",
    "qa_mean_pos","qa_mean_neg","qa_mean_neu",
    "qa_net_sentiment","qa_neg_ratio","qa_sent_vol","qa_n_sentences",
    "rag_guidance_specificity_score","rag_guidance_specificity_relevance",
    "rag_new_risks_score","rag_new_risks_relevance",
    "rag_management_confidence_score","rag_management_confidence_relevance",
    "rag_forward_looking_score","rag_forward_looking_relevance",
    "rag_cost_pressure_score","rag_cost_pressure_relevance",
]
TARGET     = "target_5d_up"
RETURN_COL = "ret_5d"

# GICS sector mapping — covers most S&P 500 tickers
SECTOR_MAP = {
    # Technology
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology",
    "GOOGL":"Technology","META":"Technology","AVGO":"Technology",
    "ORCL":"Technology","CRM":"Technology","AMD":"Technology",
    "INTC":"Technology","QCOM":"Technology","TXN":"Technology",
    "AMAT":"Technology","MU":"Technology","LRCX":"Technology",
    "NOW":"Technology","ADBE":"Technology","KLAC":"Technology",
    "MRVL":"Technology","CSCO":"Technology","IBM":"Technology",
    "HPQ":"Technology","DELL":"Technology","NXPI":"Technology",
    # Healthcare
    "JNJ":"Healthcare","UNH":"Healthcare","LLY":"Healthcare",
    "ABT":"Healthcare","MRK":"Healthcare","TMO":"Healthcare",
    "DHR":"Healthcare","ABBV":"Healthcare","PFE":"Healthcare",
    "BMY":"Healthcare","AMGN":"Healthcare","GILD":"Healthcare",
    "ISRG":"Healthcare","MDT":"Healthcare","CVS":"Healthcare",
    "CI":"Healthcare","HUM":"Healthcare","BIIB":"Healthcare",
    "REGN":"Healthcare","VRTX":"Healthcare","ZTS":"Healthcare",
    # Financials
    "BRK-B":"Financials","JPM":"Financials","BAC":"Financials",
    "WFC":"Financials","GS":"Financials","MS":"Financials",
    "BLK":"Financials","SCHW":"Financials","AXP":"Financials",
    "USB":"Financials","PNC":"Financials","TFC":"Financials",
    "COF":"Financials","CB":"Financials","MMC":"Financials",
    "AON":"Financials","ICE":"Financials","CME":"Financials",
    # Consumer Discretionary
    "AMZN":"Consumer Disc","TSLA":"Consumer Disc","HD":"Consumer Disc",
    "MCD":"Consumer Disc","NKE":"Consumer Disc","SBUX":"Consumer Disc",
    "TGT":"Consumer Disc","LOW":"Consumer Disc","BKNG":"Consumer Disc",
    "TJX":"Consumer Disc","ROST":"Consumer Disc","GM":"Consumer Disc",
    "F":"Consumer Disc","EBAY":"Consumer Disc","MAR":"Consumer Disc",
    # Consumer Staples
    "PG":"Consumer Staples","KO":"Consumer Staples","PEP":"Consumer Staples",
    "COST":"Consumer Staples","WMT":"Consumer Staples","PM":"Consumer Staples",
    "MO":"Consumer Staples","MDLZ":"Consumer Staples","CL":"Consumer Staples",
    "GIS":"Consumer Staples","K":"Consumer Staples","HSY":"Consumer Staples",
    # Energy
    "XOM":"Energy","CVX":"Energy","COP":"Energy","EOG":"Energy",
    "SLB":"Energy","MPC":"Energy","PSX":"Energy","VLO":"Energy",
    "OXY":"Energy","PXD":"Energy","HAL":"Energy","DVN":"Energy",
    # Industrials
    "GE":"Industrials","HON":"Industrials","UPS":"Industrials",
    "CAT":"Industrials","BA":"Industrials","RTX":"Industrials",
    "LMT":"Industrials","DE":"Industrials","MMM":"Industrials",
    "GD":"Industrials","FDX":"Industrials","UNP":"Industrials",
    # Communications
    "NFLX":"Communications","DIS":"Communications","CMCSA":"Communications",
    "T":"Communications","VZ":"Communications","TMUS":"Communications",
    "CHTR":"Communications","PARA":"Communications","WBD":"Communications",
    # Utilities
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities","D":"Utilities",
    "AEP":"Utilities","EXC":"Utilities","SRE":"Utilities","PEG":"Utilities",
    # Real Estate
    "AMT":"Real Estate","PLD":"Real Estate","CCI":"Real Estate",
    "EQIX":"Real Estate","PSA":"Real Estate","O":"Real Estate",
    # Materials
    "LIN":"Materials","APD":"Materials","ECL":"Materials",
    "DD":"Materials","NEM":"Materials","FCX":"Materials",
}


def assign_sectors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sector"] = df["ticker"].map(SECTOR_MAP).fillna("Other")
    return df


def walk_forward_sector(df: pd.DataFrame, sector: str) -> dict:
    """Run walk-forward for a single sector. Returns mean IC and AUC."""
    avail = [c for c in ALL_FEATURES if c in df.columns]
    ics, aucs, ns = [], [], []

    for test_year in [2021, 2022, 2023, 2024]:
        train_years = list(range(test_year - 3, test_year))
        train = df[df["year"].isin(train_years)].dropna(subset=avail + [TARGET])
        test  = df[df["year"] == test_year].dropna(
            subset=avail + [TARGET, RETURN_COL])

        if len(train) < 30 or len(test) < 10:
            continue

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train[avail].values)
        X_test  = scaler.transform(test[avail].values)

        model = LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42, verbose=-1
        )
        model.fit(X_train, train[TARGET].values)
        proba = model.predict_proba(X_test)[:, 1]

        ic, _ = pearsonr(test[RETURN_COL].values, proba)
        try:
            auc = roc_auc_score(test[TARGET].values, proba)
        except Exception:
            auc = 0.5

        ics.append(ic)
        aucs.append(auc)
        ns.append(len(test))

    if not ics:
        return None

    return {
        "sector":       sector,
        "ic_mean":      round(float(np.mean(ics)),  4),
        "ic_std":       round(float(np.std(ics)),   4),
        "auc_mean":     round(float(np.mean(aucs)), 4),
        "n_test_avg":   round(float(np.mean(ns)),   1),
        "n_folds":      len(ics),
    }


def plot_sector_ic(results: pd.DataFrame) -> None:
    df = results.sort_values("ic_mean", ascending=True)
    colors = ["#66bb6a" if v > 0 else "#ef5350" for v in df["ic_mean"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df["sector"], df["ic_mean"], color=colors, alpha=0.85)

    # Error bars
    ax.errorbar(df["ic_mean"], df["sector"],
                xerr=df["ic_std"], fmt="none",
                color="#546e7a", capsize=4, linewidth=1.5)

    # Value labels
    for bar, val in zip(bars, df["ic_mean"]):
        ax.text(val + (0.0005 if val >= 0 else -0.0005),
                bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9, color="#212121")

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean Information Coefficient (IC)", fontsize=11)
    ax.set_title("FinSight Signal Strength by GICS Sector\n"
                 "Walk-Forward IC (2021–2024) | Error bars = ±1 std",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_DIR / "sector_ic.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sector_ic.png")


def plot_sector_heatmap(df: pd.DataFrame) -> None:
    """IC by sector × year heatmap."""
    avail = [c for c in ALL_FEATURES if c in df.columns]
    records = []

    sectors = [s for s in df["sector"].unique() if s != "Other"]

    for sector in sectors:
        sub = df[df["sector"] == sector]
        for test_year in [2021, 2022, 2023, 2024]:
            train_years = list(range(test_year - 3, test_year))
            train = sub[sub["year"].isin(train_years)].dropna(
                subset=avail + [TARGET])
            test  = sub[sub["year"] == test_year].dropna(
                subset=avail + [TARGET, RETURN_COL])

            if len(train) < 20 or len(test) < 8:
                records.append({"sector": sector, "year": test_year, "ic": np.nan})
                continue

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(train[avail].values)
            X_test  = scaler.transform(test[avail].values)

            model = LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42, verbose=-1)
            model.fit(X_train, train[TARGET].values)
            proba = model.predict_proba(X_test)[:, 1]

            try:
                ic, _ = pearsonr(test[RETURN_COL].values, proba)
            except Exception:
                ic = np.nan
            records.append({"sector": sector, "year": test_year, "ic": ic})

    heat_df = pd.DataFrame(records)
    pivot   = heat_df.pivot(index="sector", columns="year", values="ic")

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-0.05, vmax=0.05)
    plt.colorbar(im, ax=ax, label="IC")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(val) > 0.03 else "black")

    ax.set_title("IC by Sector × Year — FinSight Walk-Forward",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_DIR / "sector_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sector_heatmap.png")


def main():
    fm_path = PROCESSED_DIR / "feature_matrix.parquet"
    df = pd.read_parquet(fm_path)
    logger.info(f"Loaded {len(df):,} rows.")

    df = assign_sectors(df)

    sector_counts = df.groupby("sector").size()
    logger.info(f"\nSector distribution:\n{sector_counts.to_string()}")

    # Per-sector walk-forward
    logger.info("\nRunning per-sector walk-forward...")
    results = []
    for sector in sorted(df["sector"].unique()):
        if sector == "Other":
            continue
        sub = df[df["sector"] == sector]
        result = walk_forward_sector(sub, sector)
        if result:
            results.append(result)
            logger.info(
                f"  {sector:<22} IC={result['ic_mean']:+.4f} "
                f"(std={result['ic_std']:.4f}) "
                f"AUC={result['auc_mean']:.4f} "
                f"n={result['n_test_avg']:.0f}"
            )

    results_df = pd.DataFrame(results).sort_values("ic_mean", ascending=False)
    results_df.to_csv(EXPERIMENTS_DIR / "sector_ic.csv", index=False)

    print("\n" + "="*65)
    print("SECTOR ANALYSIS — IC RANKING")
    print("="*65)
    print(f"{'Sector':<22} {'IC Mean':>10} {'IC Std':>10} {'AUC':>8}")
    print("-"*65)
    for _, row in results_df.iterrows():
        flag = " ★" if row["ic_mean"] > 0.025 else ""
        print(f"  {row['sector']:<20} {row['ic_mean']:>+10.4f} "
              f"{row['ic_std']:>10.4f} {row['auc_mean']:>8.4f}{flag}")
    print("="*65)

    # Plots
    plot_sector_ic(results_df)
    plot_sector_heatmap(df)

    # Key finding
    top = results_df.iloc[0]
    bot = results_df.iloc[-1]
    print(f"\nKey finding:")
    print(f"  Strongest sector: {top['sector']} (IC={top['ic_mean']:+.4f})")
    print(f"  Weakest sector:   {bot['sector']} (IC={bot['ic_mean']:+.4f})")
    print(f"\nFiles saved:")
    print(f"  experiments/sector_ic.csv")
    print(f"  experiments/sector_ic.png")
    print(f"  experiments/sector_heatmap.png")


if __name__ == "__main__":
    main()