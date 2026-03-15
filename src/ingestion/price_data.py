"""
Download post-earnings stock price data for all tickers in our transcript dataset.
Computes 1-day, 5-day, and 20-day forward returns aligned to earnings dates.

Usage:
    python src/ingestion/price_data.py
    python src/ingestion/price_data.py --limit 10
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, START_YEAR, END_YEAR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def download_prices(ticker: str) -> pd.DataFrame:
    """
    Download full price history for one ticker via yfinance.

    Parameters
    ----------
    ticker : str

    Returns
    -------
    pd.DataFrame
        Indexed by date with columns: open, high, low, close, volume, ticker.
    """
    try:
        df = yf.download(
            ticker,
            start=f"{START_YEAR - 1}-01-01",
            end=f"{END_YEAR + 1}-01-01",
            progress=False,
            auto_adjust=True
        )
        if df.empty:
            return pd.DataFrame()

        # yfinance 1.2+ returns MultiIndex columns — flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        # Keep only standard OHLCV columns that actually exist
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        if "close" not in keep:
            return pd.DataFrame()

        df = df[keep].copy()
        df["ticker"] = ticker
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        return df

    except Exception as e:
        logger.debug(f"{ticker}: download failed — {e}")
        return pd.DataFrame()


def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward return columns: 1-day, 5-day, 20-day post-date returns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy().sort_index()
    df["ret_1d"]  = df["close"].pct_change(1).shift(-1)
    df["ret_5d"]  = df["close"].pct_change(5).shift(-5)
    df["ret_20d"] = df["close"].pct_change(20).shift(-20)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download stock prices via yfinance.")
    parser.add_argument("--limit", type=int, default=None, help="Max tickers (for testing)")
    args = parser.parse_args()

    tickers_path = PROCESSED_DIR / "sp500_tickers.parquet"
    if not tickers_path.exists():
        logger.error("Run download_transcripts.py first to generate sp500_tickers.parquet.")
        sys.exit(1)

    tickers = pd.read_parquet(tickers_path)["ticker"].tolist()
    if args.limit:
        tickers = tickers[:args.limit]

    logger.info(f"Downloading prices for {len(tickers)} tickers...")

    all_frames = []
    failed = []

    for ticker in tqdm(tickers, desc="Downloading prices"):
        df = download_prices(ticker)
        if df.empty:
            failed.append(ticker)
            continue
        df = compute_forward_returns(df)
        all_frames.append(df)

    if not all_frames:
        logger.error("No price data downloaded at all. Check internet connection.")
        sys.exit(1)

    if failed:
        logger.warning(f"{len(failed)} tickers failed (likely delisted): {failed}")
        logger.info("This is normal — delisted companies won't have price data.")

    combined = pd.concat(all_frames)
    combined = combined.reset_index()

    out_path = PROCESSED_DIR / "price_data.parquet"
    combined.to_parquet(out_path, index=False)

    print("\n" + "="*50)
    print("PRICE DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Tickers downloaded : {combined['ticker'].nunique()}")
    print(f"Total rows         : {len(combined):,}")
    print(f"Date range         : {combined['date'].min()} – {combined['date'].max()}")
    if failed:
        print(f"Failed tickers     : {len(failed)} — {failed[:10]}")
    print("="*50)


if __name__ == "__main__":
    main()