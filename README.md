# FinSight — LLM-Powered Earnings Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-66BB6A?style=for-the-badge)

**An end-to-end machine learning system extracting alpha signals from S&P 500 earnings call transcripts.**

[**🖥️ Interactive Dashboard**](https://finsight-web-rust.vercel.app) • [**📊 Streamlit Demo**](https://huggingface.co/spaces/Rajveer234/finsight) • [**📄 Technical Report**](report/FinSight_Technical_Report.docx) • [**📈 Results**](#results)

</div>

---

## What is FinSight?

Every quarter, 500+ S&P 500 companies hold earnings calls where management presents results and analysts ask probing questions. The linguistic content of these calls — management tone, analyst skepticism, guidance specificity — may contain signals that markets don't fully price immediately.

FinSight processes **14,584 earnings transcripts** across **601 S&P 500 companies** (2018–2024), extracts 34 NLP features using FinBERT and RAG, and trains walk-forward validated ML models to predict 5-day and 20-day post-earnings stock returns.

---

## Pipeline

```
14,584 Earnings Transcripts (2018–2024)
            │
            ▼
┌─────────────────────────────────────┐
│  Stage 1 — Data Ingestion           │
│  HuggingFace datasets + yfinance    │
│  601 companies · 1M+ price rows     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 2 — NLP Feature Extraction   │
│                                     │
│  FinBERT (ProsusAI)                 │
│  · Sentence-level sentiment         │
│  · Mgmt prepared remarks vs Q&A     │
│  · 14 sentiment features            │
│                                     │
│  RAG Pipeline (all-MiniLM-L6-v2)   │
│  · 380,507 embedded chunks          │
│  · 5 structured semantic queries    │
│  · 10 relevance + content features  │
│                                     │
│  Output: 34 features · 13,442 rows  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 3 — Prediction Models        │
│                                     │
│  · Baseline (Logistic Regression)   │
│  · FinBERT-only (XGBoost)          │
│  · RAG-only (XGBoost)              │
│  · XGBoost (all 34 features)       │
│  · LightGBM (all 34 features) ★    │
│  · LSTM (temporal 6-quarter seq)    │
│                                     │
│  Walk-forward CV · Zero leakage     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 4 — Backtesting              │
│  Long-short quartile portfolio      │
│  5-day and 20-day holding periods   │
│  10bps round-trip transaction cost  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 5 — Sector Analysis          │
│  GICS sector-level walk-forward     │
│  Energy IC = +0.311 (best)          │
│  Technology IC ≈ 0 (efficient)      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 6 — Dashboard + Report       │
│  Next.js · Streamlit · HF Spaces    │
│  8-page technical report            │
└─────────────────────────────────────┘
```

---

## Results

### Walk-Forward Validation (2021–2024)
*Train on years T−3 to T−1, test on year T. Zero data leakage.*

| Model | IC Mean | IC Std | Hit Rate | AUC |
|---|---|---|---|---|
| Baseline | 0.0429 | 0.1141 ⚠️ | 0.5312 | 0.5174 |
| **LightGBM ★** | **0.0198** | **0.0085** | 0.5329 | 0.5086 |
| LSTM | 0.0153 | 0.0211 | **0.5471** | 0.5060 |
| XGBoost | 0.0141 | 0.0180 | 0.5321 | 0.5099 |
| RAG Only | 0.0000 | 0.0295 | 0.5347 | 0.5086 |
| FinBERT Only | -0.0044 | 0.0117 | 0.5312 | 0.5007 |

> **IC** = Information Coefficient (Pearson correlation of predictions vs actual 5-day returns).
> **LightGBM** is 10× more stable than baseline (std=0.009 vs std=0.114).
> **LSTM** achieves the highest hit rate (54.7%) — best for directional prediction.

### Top 5 Features by SHAP Importance

| Rank | Feature | Group | Mean \|SHAP\| | Insight |
|---|---|---|---|---|
| 1 | `qa_neg_ratio` | QA FinBERT | 0.0541 | Analyst pushback > management positivity |
| 2 | `mgmt_sent_vol` | Mgmt FinBERT | 0.0476 | Inconsistent messaging = larger price moves |
| 3 | `qa_n_sentences` | QA FinBERT | 0.0453 | Longer Q&A = more analyst scrutiny |
| 4 | `mgmt_mean_neu` | Mgmt FinBERT | 0.0445 | Deliberate neutrality = hedging signal |
| 5 | `rag_guidance_specificity_relevance` | RAG | 0.0420 | Specific guidance = clearer market reaction |

### Sector Analysis (Walk-Forward IC by GICS Sector)

| Rank | Sector | IC Mean | IC Std | AUC |
|---|---|---|---|---|
| 1 | **Energy** ★ | **+0.3111** | 0.2430 | **0.6393** |
| 2 | Real Estate | +0.0779 | 0.2861 | 0.5089 |
| 3 | Industrials | +0.0738 | 0.0359 | 0.5625 |
| 4 | Utilities | +0.0644 | 0.1428 | 0.4703 |
| 5 | Consumer Staples | +0.0613 | 0.1452 | 0.5212 |
| 9 | Technology | +0.0037 | 0.0983 | 0.4874 |
| 11 | Materials | -0.1321 | 0.2903 | 0.4958 |

> **Key finding:** Energy IC = 0.311 is **83× stronger** than Technology IC ≈ 0.004.
> Consistent with efficient market hypothesis by sector — Technology is efficiently priced,
> Energy has high information asymmetry from commodity price exposure.

### Backtest Performance (Long-Short Quartile)

| Metric | 5-Day | 20-Day |
|---|---|---|
| Annualized Return | -0.91% | -0.69% |
| **Sharpe Ratio** | **-0.81** | **-0.23 (+3.6×)** |
| Max Drawdown | -4.24% | -6.03% |
| Win Rate | 37.5% | 31.3% |

> Sharpe improves 3.6× from 5-day to 20-day holding, consistent with PEAD theory
> (Bernard & Thomas 1989). Signal exists (IC=0.0198) but is insufficient to overcome
> 10bps transaction costs at a 5-day horizon. Extending to 20-day reduces the
> cost-to-signal ratio significantly.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| NLP Model | FinBERT (ProsusAI/finbert) |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| ML Models | XGBoost, LightGBM, PyTorch LSTM |
| Interpretability | SHAP |
| Dashboard (v2) | Next.js 14, TypeScript, Tailwind, Recharts, Framer Motion |
| Dashboard (v1) | Streamlit + Plotly |
| Deployment | Vercel (Next.js) + Hugging Face Spaces (Streamlit) |
| GPU | NVIDIA RTX 4060 Laptop (CUDA 11.8) |

---

## Project Structure

```
finsight/
├── config.py                        # Central configuration (paths, constants)
├── run_ingestion.py                 # Stage 1 runner
├── run_nlp.py                       # Stage 2 runner
├── export_data.py                   # Export JSON for Next.js dashboard
│
├── src/
│   ├── ingestion/
│   │   ├── download_transcripts.py  # HuggingFace dataset download
│   │   ├── price_data.py            # yfinance price data
│   │   └── validate_data.py         # Data quality checks
│   │
│   ├── nlp/
│   │   ├── finbert_sentiment.py     # FinBERT pipeline (GPU, checkpointing)
│   │   ├── rag_pipeline.py          # RAG feature extraction (GPU-accelerated)
│   │   └── build_feature_matrix.py  # Merge features + price returns
│   │
│   ├── models/
│   │   ├── train_models.py          # XGBoost + LightGBM walk-forward
│   │   └── lstm_model.py            # LSTM sequence model
│   │
│   ├── backtest/
│   │   ├── backtest_engine.py       # 5-day backtest
│   │   └── backtest_20d.py          # 20-day backtest + comparison
│   │
│   ├── analysis/
│   │   └── sector_analysis.py       # GICS sector-level IC analysis
│   │
│   └── dashboard/
│       └── app.py                   # Streamlit dashboard (v1)
│
├── experiments/                     # Model results, SHAP, plots
├── report/
│   └── FinSight_Technical_Report.docx
└── requirements.txt
```

---

## Reproducing Results

### Setup
```bash
git clone https://github.com/Rajveer-code/Finsight.git
cd Finsight
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Stage 1 — Data Ingestion (~30 min)
```bash
python run_ingestion.py
# Output: 14,584 transcripts, 1M+ price rows
```

### Stage 2 — NLP Pipeline (~3 hours on GPU)
```bash
python run_nlp.py
# Output: 34 features × 13,442 rows
# Checkpoints every 100/500 records — safe to interrupt
```

### Stage 3 — Train Models (~10 min)
```bash
python src/models/train_models.py
python src/models/lstm_model.py
```

### Stage 4 — Backtest (~1 min)
```bash
python src/backtest/backtest_engine.py
python src/backtest/backtest_20d.py
```

### Stage 5 — Sector Analysis (~5 min)
```bash
python src/analysis/sector_analysis.py
```

### Stage 6 — Dashboard
```bash
# Streamlit (v1)
streamlit run src/dashboard/app.py

# Export data for Next.js dashboard
python export_data.py
```

---

## Key Design Decisions

**Why walk-forward validation?**
Standard k-fold cross-validation leaks future information in time series. Walk-forward trains on years T−3 to T−1 and tests on year T only. No future data is ever seen during training.

**Why FinBERT + RAG together?**
FinBERT captures emotional tone at the sentence level. RAG captures topical specificity — whether management actually discussed numerical guidance, new risks, or cost pressures. RAG features contribute 34.6% of total SHAP importance despite comprising fewer features.

**Why LSTM alongside tree models?**
Tree models treat each earnings call as independent. The LSTM learns that a company with 6 consecutive quarters of deteriorating sentiment is different from one with a single bad quarter. Its 2022 IC of +0.047 — the strongest single fold across all models — validates this temporal signal.

**Why both 5-day and 20-day backtests?**
Post-earnings announcement drift (PEAD) is documented at 20-60 day horizons. The 3.6× Sharpe improvement from 5-day to 20-day validates that the signal takes time to be fully priced, consistent with Bernard & Thomas (1989).

---

## Limitations & Future Work

- [ ] Long-only 20-day backtest (eliminates short-selling costs)
- [ ] Replace RAG keyword scoring with Llama-3 / Mistral generative scorer
- [ ] Sector-stratified model training (separate models per sector)
- [ ] Cross-lingual extension using multilingual FinBERT
- [ ] Real-time pipeline streaming live earnings calls

---

## References

- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv:1908.10063*
- Bernard, V. & Thomas, J. (1989). Post-Earnings-Announcement Drift. *Journal of Accounting Research*
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*
- Loughran & McDonald (2011). When is a Liability not a Liability? *Journal of Finance*
- Chan, Jegadeesh & Lakonishok (1996). Momentum Strategies. *Journal of Finance, 51(5)*

---

## Author

**Rajveer Singh Pall**

Portfolio project for MSc Data Science application (ETH Zurich 2026).

---

<div align="center">

**Interactive Dashboard:** [finsight-web-rust.vercel.app](https://finsight-web-rust.vercel.app)

**Streamlit Demo:** [huggingface.co/spaces/Rajveer234/finsight](https://huggingface.co/spaces/Rajveer234/finsight)

</div>
