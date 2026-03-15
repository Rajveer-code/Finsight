"""
Merge FinBERT features + RAG features + price targets into
the final feature matrix used by Stage 3 models.

Output: data/processed/feature_matrix.parquet

Usage:
    python src/nlp/build_feature_matrix.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def align_price_to_earnings(
    transcripts: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.DataFrame:
    """
    For each transcript (ticker, date), find the forward returns
    starting from the next trading day after the earnings call.

    Parameters
    ----------
    transcripts : pd.DataFrame
    prices : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Transcripts with ret_1d, ret_5d, ret_20d columns added.
    """
    prices["date"] = pd.to_datetime(prices["date"])
    transcripts    = transcripts.copy()
    transcripts["earnings_date"] = pd.to_datetime(transcripts["date"])

    ret_rows = []
    price_by_ticker = {t: grp.set_index("date").sort_index()
                       for t, grp in prices.groupby("ticker")}

    for _, row in transcripts.iterrows():
        ticker = row["ticker"]
        edate  = row["earnings_date"]

        if ticker not in price_by_ticker:
            ret_rows.append({"ret_1d": np.nan, "ret_5d": np.nan, "ret_20d": np.nan})
            continue

        p = price_by_ticker[ticker]
        # Find next trading day at or after earnings date
        future = p[p.index >= edate]
        if len(future) < 2:
            ret_rows.append({"ret_1d": np.nan, "ret_5d": np.nan, "ret_20d": np.nan})
            continue

        base_price = future["close"].iloc[0]

        def safe_ret(n: int) -> float:
            if len(future) > n:
                return float((future["close"].iloc[n] - base_price) / base_price)
            return np.nan

        ret_rows.append({
            "ret_1d":  safe_ret(1),
            "ret_5d":  safe_ret(5),
            "ret_20d": safe_ret(20),
        })

    ret_df = pd.DataFrame(ret_rows, index=transcripts.index)
    return pd.concat([transcripts, ret_df], axis=1)


def main() -> None:
    # Load all inputs
    logger.info("Loading data...")
    transcripts = pd.read_parquet(PROCESSED_DIR / "transcripts.parquet")
    prices      = pd.read_parquet(PROCESSED_DIR / "price_data.parquet")

    finbert_path = PROCESSED_DIR / "finbert_features.parquet"
    rag_path     = PROCESSED_DIR / "rag_features.parquet"

    if not finbert_path.exists():
        logger.error("finbert_features.parquet missing — run finbert_sentiment.py first.")
        sys.exit(1)
    if not rag_path.exists():
        logger.error("rag_features.parquet missing — run rag_pipeline.py first.")
        sys.exit(1)

    finbert = pd.read_parquet(finbert_path)
    rag     = pd.read_parquet(rag_path)

    # Align prices to earnings dates
    logger.info("Aligning price returns to earnings dates...")
    transcripts = align_price_to_earnings(transcripts, prices)

    # Merge keys
    keys = ["ticker", "year", "quarter"]

    # Start with transcripts metadata + returns
    base = transcripts[keys + ["earnings_date", "company", "ret_1d", "ret_5d", "ret_20d"]].copy()

    # Merge FinBERT
    base = base.merge(finbert.drop(columns=["date", "company"], errors="ignore"),
                      on=keys, how="left")

    # Merge RAG
    base = base.merge(rag.drop(columns=[], errors="ignore"),
                      on=keys, how="left")

    # Create binary target: did stock beat 0% in 5 days?
    base["target_5d_up"]  = (base["ret_5d"]  > 0.0).astype(int)
    base["target_20d_up"] = (base["ret_20d"] > 0.0).astype(int)

    # Drop rows where we have no return data at all
    before = len(base)
    base   = base.dropna(subset=["ret_5d"])
    logger.info(f"Dropped {before - len(base)} rows with no price data.")

    out_path = PROCESSED_DIR / "feature_matrix.parquet"
    base.to_parquet(out_path, index=False)

    print("\n" + "="*55)
    print("FEATURE MATRIX — SUMMARY")
    print("="*55)
    print(f"Rows (transcript-quarters) : {len(base):,}")
    print(f"Unique tickers             : {base['ticker'].nunique()}")
    print(f"Total features             : {len(base.columns)}")
    print(f"Target (5d up) base rate   : {base['target_5d_up'].mean():.2%}")
    print(f"Null values per column:")
    null_summary = base.isnull().sum()
    null_summary = null_summary[null_summary > 0].sort_values(ascending=False)
    print(null_summary.head(10).to_string())
    print("="*55)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()