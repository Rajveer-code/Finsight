"""
Stage 3 — Prediction Models with Walk-Forward Validation

Models trained (in order of complexity):
  1. Baseline     — earnings surprise proxy (no NLP)
  2. FinBERT only — NLP sentiment features only
  3. RAG only     — structured LLM features only
  4. XGBoost      — all features combined
  5. LightGBM     — all features combined

Evaluation: IC, Hit Rate, AUC — all via walk-forward validation.
Walk-forward: train on years T-3 to T-1, test on year T.

Output:
  experiments/model_results.csv     — per-fold metrics
  experiments/model_summary.csv     — aggregated summary table
  experiments/shap_values.parquet   — SHAP values for best model

Usage:
    python src/models/train_models.py
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, EXPERIMENTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature groups ─────────────────────────────────────────────────────────────

FINBERT_FEATURES = [
    "mgmt_mean_pos", "mgmt_mean_neg", "mgmt_mean_neu",
    "mgmt_net_sentiment", "mgmt_neg_ratio", "mgmt_sent_vol",
    "mgmt_n_sentences",
    "qa_mean_pos", "qa_mean_neg", "qa_mean_neu",
    "qa_net_sentiment", "qa_neg_ratio", "qa_sent_vol",
    "qa_n_sentences",
]

RAG_FEATURES = [
    "rag_guidance_specificity_score", "rag_guidance_specificity_relevance",
    "rag_new_risks_score",            "rag_new_risks_relevance",
    "rag_management_confidence_score","rag_management_confidence_relevance",
    "rag_forward_looking_score",      "rag_forward_looking_relevance",
    "rag_cost_pressure_score",        "rag_cost_pressure_relevance",
]

BASELINE_FEATURES = ["quarter"]   # minimal — no NLP at all

TARGET      = "target_5d_up"
RETURN_COL  = "ret_5d"
TEST_YEARS  = [2021, 2022, 2023, 2024]   # walk-forward test years
TRAIN_WINDOW = 3                          # years of training data


# ── Metrics ────────────────────────────────────────────────────────────────────

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between predictions and actual returns."""
    if len(y_true) < 10:
        return np.nan
    ic, _ = pearsonr(y_true, y_pred)
    return round(float(ic), 4)


def hit_rate(y_true_binary: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Fraction of predictions where direction is correct."""
    predicted_direction = (y_pred_proba >= 0.5).astype(int)
    return round(float(np.mean(predicted_direction == y_true_binary)), 4)


def evaluate(
    y_true_ret: np.ndarray,
    y_true_bin: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    """Compute IC, hit rate, and AUC."""
    return {
        "ic":       information_coefficient(y_true_ret, y_pred_proba),
        "hit_rate": hit_rate(y_true_bin, y_pred_proba),
        "auc":      round(float(roc_auc_score(y_true_bin, y_pred_proba)), 4),
        "n_test":   int(len(y_true_bin)),
    }


# ── Walk-forward engine ────────────────────────────────────────────────────────

def walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    model_fn,
) -> pd.DataFrame:
    """
    Walk-forward validation across TEST_YEARS.

    For each test year T:
      - Train: years [T - TRAIN_WINDOW, T - 1]
      - Test:  year T

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    model_name : str
    model_fn : callable — returns a fresh sklearn-compatible model

    Returns
    -------
    pd.DataFrame
        One row per fold with metrics.
    """
    results = []

    for test_year in TEST_YEARS:
        train_years = list(range(test_year - TRAIN_WINDOW, test_year))

        train = df[df["year"].isin(train_years)].copy()
        test  = df[df["year"] == test_year].copy()

        # Drop rows missing any feature
        avail_cols = [c for c in feature_cols if c in df.columns]
        train = train.dropna(subset=avail_cols + [TARGET])
        test  = test.dropna(subset=avail_cols + [TARGET])

        if len(train) < 50 or len(test) < 20:
            logger.warning(f"{model_name} | {test_year}: insufficient data "
                           f"(train={len(train)}, test={len(test)}). Skipping.")
            continue

        X_train = train[avail_cols].values
        y_train = train[TARGET].values
        X_test  = test[avail_cols].values
        y_test  = test[TARGET].values
        ret_test = test[RETURN_COL].values

        # Scale
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Train
        model = model_fn()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(ret_test, y_test, proba)
        metrics.update({
            "model":      model_name,
            "test_year":  test_year,
            "n_train":    len(train),
        })
        results.append(metrics)

        logger.info(
            f"{model_name:20s} | {test_year} | "
            f"IC={metrics['ic']:+.4f} | "
            f"HitRate={metrics['hit_rate']:.4f} | "
            f"AUC={metrics['auc']:.4f} | "
            f"n_test={metrics['n_test']}"
        )

    return pd.DataFrame(results)


# ── Model definitions ──────────────────────────────────────────────────────────

def get_xgb():
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        device="cuda",
    )


def get_lgbm():
    return LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        device="gpu",
    )


def get_baseline():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=500, random_state=42)


# ── SHAP analysis ──────────────────────────────────────────────────────────────

def run_shap_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
) -> None:
    """
    Train XGBoost on full data and compute SHAP values.
    Saves: shap_values.parquet, shap_summary.png, feature_importance.png
    """
    logger.info("Running SHAP analysis on full dataset...")

    avail_cols = [c for c in feature_cols if c in df.columns]
    df_clean   = df.dropna(subset=avail_cols + [TARGET])

    X = df_clean[avail_cols].values
    y = df_clean[TARGET].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = get_xgb()
    # For SHAP, use CPU XGBoost (more stable with SHAP explainer)
    model.set_params(device="cpu")
    model.fit(X_sc, y)

    # Save and reload model to fix XGBoost 2.x / shap base_score format bug
    import tempfile, os
    tmp = tempfile.mktemp(suffix=".json")
    model.save_model(tmp)
    model.load_model(tmp)
    os.remove(tmp)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sc)

    # Save raw SHAP values
    shap_df = pd.DataFrame(shap_vals, columns=avail_cols)
    shap_df.to_parquet(out_dir / "shap_values.parquet", index=False)

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sc, feature_names=avail_cols,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance bar chart
    mean_shap  = np.abs(shap_vals).mean(axis=0)
    importance = pd.Series(mean_shap, index=avail_cols).sort_values(ascending=True)
    top20      = importance.tail(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2196F3" if "rag" in c else
              "#4CAF50" if "mgmt" in c else
              "#FF9800" if "qa" in c else "#9E9E9E"
              for c in top20.index]
    top20.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Feature Importance — XGBoost (All Features)\n"
                 "Blue=RAG | Green=Management FinBERT | Orange=QA FinBERT")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"SHAP plots saved to {out_dir}")


# ── Ablation table ─────────────────────────────────────────────────────────────

def print_summary_table(all_results: pd.DataFrame) -> None:
    """Print the model comparison table that goes in your technical report."""
    summary = (
        all_results
        .groupby("model")[["ic", "hit_rate", "auc", "n_test"]]
        .agg({"ic": ["mean","std"], "hit_rate": ["mean","std"],
              "auc": ["mean","std"], "n_test": "sum"})
    )
    summary.columns = ["IC_mean","IC_std","HitRate_mean","HitRate_std",
                        "AUC_mean","AUC_std","Total_test_samples"]
    summary = summary.sort_values("IC_mean", ascending=False)

    print("\n" + "="*80)
    print("MODEL COMPARISON — WALK-FORWARD VALIDATION (2021–2024)")
    print("="*80)
    print(summary.round(4).to_string())
    print("="*80)
    print("\nIC  = Information Coefficient (Pearson corr of predictions vs actual returns)")
    print("AUC = Area Under ROC Curve (direction prediction quality)")
    print("All metrics are averages across 4 test years (2021-2024)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    fm_path = PROCESSED_DIR / "feature_matrix.parquet"
    if not fm_path.exists():
        logger.error("feature_matrix.parquet missing — run Stage 2 first.")
        sys.exit(1)

    df = pd.read_parquet(fm_path)
    logger.info(f"Loaded feature matrix: {len(df):,} rows, {len(df.columns)} columns.")

    ALL_FEATURES = FINBERT_FEATURES + RAG_FEATURES

    models = [
        ("Baseline",       BASELINE_FEATURES,  get_baseline),
        ("FinBERT_only",   FINBERT_FEATURES,   get_xgb),
        ("RAG_only",       RAG_FEATURES,        get_xgb),
        ("XGBoost_all",    ALL_FEATURES,        get_xgb),
        ("LightGBM_all",   ALL_FEATURES,        get_lgbm),
    ]

    all_results = []
    for model_name, features, model_fn in models:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Running walk-forward: {model_name}")
        results = walk_forward(df, features, model_name, model_fn)
        all_results.append(results)

    all_results_df = pd.concat(all_results, ignore_index=True)

    # Save per-fold results
    per_fold_path = EXPERIMENTS_DIR / "model_results.csv"
    all_results_df.to_csv(per_fold_path, index=False)
    logger.info(f"Per-fold results saved to {per_fold_path}")

    # Summary table
    summary = (
        all_results_df
        .groupby("model")[["ic","hit_rate","auc","n_test"]]
        .agg({"ic":["mean","std"],"hit_rate":["mean","std"],
              "auc":["mean","std"],"n_test":"sum"})
    )
    summary.columns = ["IC_mean","IC_std","HitRate_mean","HitRate_std",
                        "AUC_mean","AUC_std","Total_samples"]
    summary = summary.sort_values("IC_mean", ascending=False).round(4)
    summary.to_csv(EXPERIMENTS_DIR / "model_summary.csv")

    print_summary_table(all_results_df)

    # SHAP analysis on best model
    run_shap_analysis(df, ALL_FEATURES, EXPERIMENTS_DIR)

    # Year-by-year IC chart
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name in all_results_df["model"].unique():
        sub = all_results_df[all_results_df["model"] == model_name]
        ax.plot(sub["test_year"], sub["ic"], marker="o", label=model_name)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Test Year")
    ax.set_ylabel("Information Coefficient (IC)")
    ax.set_title("Walk-Forward IC by Year — All Models")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_DIR / "ic_by_year.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("All done. Check experiments/ for results and plots.")
    print(f"\nFiles saved to: {EXPERIMENTS_DIR}")
    print("  model_results.csv       — per-fold metrics")
    print("  model_summary.csv       — aggregated comparison table")
    print("  shap_summary.png        — SHAP beeswarm plot")
    print("  feature_importance.png  — top 20 features by SHAP")
    print("  ic_by_year.png          — IC trend across years")


if __name__ == "__main__":
    main()