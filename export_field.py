"""
Export a compact "signal field" dataset for the FinSight web experience.

Each row is one earnings call. Values are quantized to small integers so the
payload stays light enough to ship to the GPU on page load.

Run from the repo root with the venv active:
    python export_field.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
sys.path.insert(0, 'src/analysis')
from config import PROCESSED_DIR
from sector_analysis import SECTOR_MAP

OUT = Path('D:/Projects/finsight-web/public/data')
OUT.mkdir(parents=True, exist_ok=True)

SECTORS = [
    'Technology', 'Healthcare', 'Financials', 'Consumer Disc',
    'Consumer Staples', 'Energy', 'Industrials', 'Communications',
    'Utilities', 'Real Estate', 'Materials', 'Other',
]


def q(series: pd.Series, scale: float, lo: float, hi: float) -> list:
    """Clip, scale, and round a float series to ints."""
    return (series.clip(lo, hi) * scale).round().astype(int).tolist()


def main() -> None:
    fm = pd.read_parquet(PROCESSED_DIR / 'feature_matrix.parquet')
    names = pd.read_parquet(PROCESSED_DIR / 'sp500_tickers.parquet')
    name_map = dict(zip(names['ticker'], names['company']))

    df = fm.dropna(subset=['mgmt_net_sentiment', 'qa_neg_ratio',
                           'mgmt_sent_vol', 'ret_5d']).copy()
    print(f'rows: {len(fm)} -> {len(df)} after dropping incomplete calls')

    df['ret_20d'] = df['ret_20d'].fillna(0.0)
    df['sector'] = df['ticker'].map(SECTOR_MAP).fillna('Other')

    tickers = sorted(df['ticker'].unique())
    t_idx = {t: i for i, t in enumerate(tickers)}

    payload = {
        'tickers': tickers,
        'names': [name_map.get(t) or t for t in tickers],
        'sectors': SECTORS,
        't': df['ticker'].map(t_idx).astype(int).tolist(),
        'y': (df['year'].astype(int) - 2018).tolist(),
        'q': df['quarter'].astype(int).tolist(),
        's': df['sector'].map({s: i for i, s in enumerate(SECTORS)}).astype(int).tolist(),
        # sentiment in [-1, 1] -> x1000
        'ms': q(df['mgmt_net_sentiment'], 1000, -1, 1),
        # ratios/vol in [0, 1] -> x1000
        'qn': q(df['qa_neg_ratio'], 1000, 0, 1),
        'mv': q(df['mgmt_sent_vol'], 1000, 0, 1),
        # RAG retrieval relevance in [0, 1] -> x1000
        'rg': q(df['rag_guidance_specificity_relevance'].fillna(0.0), 1000, 0, 1),
        # returns clipped to +/-40% -> x10000 (basis points)
        'r5': q(df['ret_5d'], 10000, -0.4, 0.4),
        'r20': q(df['ret_20d'], 10000, -0.4, 0.4),
    }

    out_path = OUT / 'field.json'
    with open(out_path, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))

    kb = out_path.stat().st_size / 1024
    print(f'wrote {out_path} ({kb:,.0f} KB, {len(df):,} calls, '
          f'{len(tickers)} tickers)')
    print(df['sector'].value_counts().to_string())


if __name__ == '__main__':
    main()
