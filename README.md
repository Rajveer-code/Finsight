# FinSight — LLM-Powered Earnings Intelligence

> **Live Demo:** https://huggingface.co/spaces/Rajveer234/finsight

An end-to-end ML pipeline extracting alpha signals from S&P 500 earnings transcripts.

## Stack
FinBERT · ChromaDB · XGBoost · LightGBM · SHAP · Streamlit

## Results
- 14,584 transcripts · 601 companies · 2018–2024
- LightGBM IC = 0.0198 (std = 0.009) — 10× more stable than baseline
- Top feature: qa_neg_ratio (SHAP = 0.054)
- Walk-forward validated — zero data leakage

## Pipeline
1. Data Ingestion — SEC EDGAR + yfinance
2. NLP — FinBERT sentiment + RAG feature extraction
3. Models — XGBoost + LightGBM with walk-forward CV
4. Backtest — Long-short quartile strategy, 10bps TC
5. Dashboard — Streamlit on Hugging Face Spaces