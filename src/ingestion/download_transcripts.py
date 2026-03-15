"""
Download S&P 500 earnings call transcripts from HuggingFace.
Dataset: kurry/sp500_earnings_transcripts
- 33,362 transcripts | 685 companies | 2005-2025 | MIT license

Run once. Data saved locally. Never scrape again.

Usage:
    python src/ingestion/download_transcripts.py
    python src/ingestion/download_transcripts.py --start-year 2018 --end-year 2024
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RAW_DIR, PROCESSED_DIR, START_YEAR, END_YEAR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def download_and_save(start_year: int, end_year: int) -> Path:
    """
    Download transcripts from HuggingFace and save to local parquet.

    Parameters
    ----------
    start_year : int
    end_year : int

    Returns
    -------
    Path
        Path to saved parquet file.
    """
    logger.info("Downloading S&P 500 transcripts from HuggingFace...")
    logger.info("Dataset: kurry/sp500_earnings_transcripts (~1.8 GB)")
    logger.info("This runs once. Future runs load from local disk.")

    # Download — HuggingFace caches this automatically after first run
    ds = load_dataset(
        "kurry/sp500_earnings_transcripts",
        split="train",
        trust_remote_code=True
    )

    logger.info(f"Downloaded {len(ds)} total transcripts. Filtering {start_year}–{end_year}...")

    # Convert to pandas and filter to our date range
    df = ds.to_pandas()

    # Ensure year column is integer
    df["year"] = df["year"].astype(int)

    # Filter to project date range
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    df = df.reset_index(drop=True)

    logger.info(f"Filtered to {len(df)} transcripts ({start_year}–{end_year}).")
    logger.info(f"Companies covered: {df['symbol'].nunique()}")

    # Keep only the columns we need — drop structured_content (we'll use raw content)
    columns_to_keep = ["symbol", "company_name", "year", "quarter", "date", "content"]
    df = df[columns_to_keep]

    # Rename for consistency
    df = df.rename(columns={
        "symbol":       "ticker",
        "company_name": "company",
        "content":      "transcript_text",
    })

    # Clean up: remove rows with empty transcripts
    before = len(df)
    df = df[df["transcript_text"].str.len() > 500].copy()
    logger.info(f"Removed {before - len(df)} empty/short transcripts.")

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    out_path = PROCESSED_DIR / "transcripts.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} transcripts to {out_path}")

    # Also save ticker list (extracted from actual data — no Wikipedia scraping)
    tickers_df = (
        df[["ticker", "company"]]
        .drop_duplicates(subset="ticker")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    tickers_path = PROCESSED_DIR / "sp500_tickers.parquet"
    tickers_df.to_parquet(tickers_path, index=False)
    logger.info(f"Saved {len(tickers_df)} unique tickers to {tickers_path}")

    # Print summary
    print("\n" + "="*50)
    print("TRANSCRIPT DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Total transcripts : {len(df):,}")
    print(f"Unique companies  : {df['ticker'].nunique()}")
    print(f"Year range        : {df['year'].min()} – {df['year'].max()}")
    print(f"Quarters covered  : {sorted(df['quarter'].unique())}")
    print(f"\nSample records:")
    print(df[["ticker", "company", "year", "quarter"]].head(10).to_string(index=False))
    print("="*50)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download earnings transcripts from HuggingFace.")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year",   type=int, default=END_YEAR)
    args = parser.parse_args()

    # Check if already downloaded
    out_path = PROCESSED_DIR / "transcripts.parquet"
    if out_path.exists():
        df = pd.read_parquet(out_path)
        logger.info(f"Transcripts already downloaded: {len(df):,} rows at {out_path}")
        logger.info("Delete the file and rerun to re-download.")
        return

    download_and_save(args.start_year, args.end_year)


if __name__ == "__main__":
    main()