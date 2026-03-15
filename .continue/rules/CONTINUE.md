# FinSight — AI Coding Rules

## Project
LLM-powered earnings intelligence system for master's application portfolio.
Extracts signals from S&P 500 earnings call transcripts (FinBERT + RAG),
predicts post-earnings stock movement, backtests with transaction costs.

## Pipeline Stages
1. Data ingestion — SEC EDGAR + yfinance
2. NLP — FinBERT sentiment + ChromaDB RAG
3. Prediction — XGBoost, LightGBM, LSTM
4. Backtesting — long-short, transaction costs, Sharpe ratio
5. Dashboard — Streamlit → Hugging Face Spaces

## Coding Rules
- Python 3.10, type hints on every function
- NumPy-style docstrings on every function
- All paths via config.py using pathlib — zero hardcoded strings
- Intermediate data saved as .parquet in data/processed/
- Every script must have CLI entry point via argparse
- Log progress with tqdm, log errors with Python logging module

## Hardware
- Windows 11, RTX 4060 8GB VRAM
- FinBERT batch_size=16, use float16 for inference
- Project root: D:\finsight