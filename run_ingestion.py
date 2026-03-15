"""
Stage 1 master runner.

Usage:
    python run_ingestion.py           # full run
    python run_ingestion.py --test    # 10 tickers only (price step)
"""

import argparse
import subprocess
import sys


def run(cmd: list) -> None:
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print("Step failed.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Limit price download to 10 tickers")
    args = parser.parse_args()

    limit = ["--limit", "10"] if args.test else []

    # Step 1: Download transcripts from HuggingFace (runs once, skips if done)
    run([sys.executable, "src/ingestion/download_transcripts.py"])

    # Step 2: Download stock prices via yfinance
    run([sys.executable, "src/ingestion/price_data.py"] + limit)

    # Step 3: Validate
    run([sys.executable, "src/ingestion/validate_data.py"])

    print("\n✅ Stage 1 complete.")


if __name__ == "__main__":
    main()