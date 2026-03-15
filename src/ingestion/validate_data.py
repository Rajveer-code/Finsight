"""
Validate Stage 1 data quality before moving to NLP.
Run after both download_transcripts.py and price_data.py complete.

Usage:
    python src/ingestion/validate_data.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def validate_transcripts(df: pd.DataFrame) -> bool:
    """Run quality checks on transcript data."""
    print("\n── TRANSCRIPTS ──────────────────────────")
    print(f"  Rows              : {len(df):,}")
    print(f"  Unique tickers    : {df['ticker'].nunique()}")
    print(f"  Year range        : {df['year'].min()} – {df['year'].max()}")
    print(f"  Quarters          : {sorted(df['quarter'].unique())}")
    print(f"  Null transcripts  : {df['transcript_text'].isna().sum()}")
    print(f"  Avg text length   : {df['transcript_text'].str.len().mean():.0f} chars")
    print(f"  Min text length   : {df['transcript_text'].str.len().min()} chars")

    issues = 0
    if df["ticker"].isna().any():
        logger.warning("NULL tickers found.")
        issues += 1
    if df["transcript_text"].isna().any():
        logger.warning("NULL transcript texts found.")
        issues += 1
    if df["year"].min() < 2018:
        logger.warning(f"Transcripts before 2018 found: {(df['year'] < 2018).sum()} rows")

    return issues == 0


def validate_prices(df: pd.DataFrame) -> bool:
    """Run quality checks on price data."""
    print("\n── PRICES ───────────────────────────────")
    print(f"  Rows              : {len(df):,}")
    print(f"  Unique tickers    : {df['ticker'].nunique()}")
    print(f"  Date range        : {df['date'].min()} – {df['date'].max()}")
    print(f"  Null close prices : {df['close'].isna().sum()}")
    print(f"  Null ret_5d       : {df['ret_5d'].isna().sum()}")

    issues = 0
    if df["close"].isna().sum() / len(df) > 0.05:
        logger.warning("More than 5% of close prices are null.")
        issues += 1

    return issues == 0


def main() -> None:
    all_ok = True

    # Transcripts
    t_path = PROCESSED_DIR / "transcripts.parquet"
    if not t_path.exists():
        logger.error(f"Missing: {t_path}")
        all_ok = False
    else:
        df_t = pd.read_parquet(t_path)
        all_ok = validate_transcripts(df_t) and all_ok

    # Prices
    p_path = PROCESSED_DIR / "price_data.parquet"
    if not p_path.exists():
        logger.error(f"Missing: {p_path}")
        all_ok = False
    else:
        df_p = pd.read_parquet(p_path)
        all_ok = validate_prices(df_p) and all_ok

    print("\n" + "="*50)
    if all_ok:
        print("✅ STAGE 1 VALIDATION PASSED. Ready for Stage 2 (NLP).")
    else:
        print("⚠️  VALIDATION ISSUES FOUND. Fix before proceeding.")
    print("="*50)


if __name__ == "__main__":
    main()