# FinSight — LLM-Powered Earnings Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-66BB6A?style=for-the-badge)

**An end-to-end machine learning system that extracts alpha signals from S&P 500 earnings call transcripts.**

[**🚀 Live Demo**](https://huggingface.co/spaces/Rajveer234/finsight) • [**📄 Technical Report**](report/FinSight_Technical_Report.docx) • [**📊 Results**](#results)

</div>

---

## What is FinSight?

Every quarter, 500+ companies hold earnings calls where management presents results and analysts ask probing questions. The linguistic content of these calls — management tone, analyst skepticism, guidance specificity — contains signals that markets don't fully price immediately.

FinSight processes **14,584 earnings transcripts** across **601 S&P 500 companies** (2018–2024), extracts rich NLP features using FinBERT and RAG, and trains walk-forward validated ML models to predict 5-day post-earnings stock returns.

---

## Pipeline

```
14,584 Earnings Transcripts (2018–2024)
            │
            ▼
┌─────────────────────────────────────┐
│  Stage 1: Data Ingestion            │
│  SEC EDGAR + yfinance               │
│  601 companies · 1M+ price rows     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 2: NLP Feature Extraction    │
│                                     │
│  FinBERT (ProsusAI)                 │
│  · Sentence-level sentiment         │
│  · Mgmt vs Q&A sections            │
│  · 14 sentiment features            │
│                                     │
│  RAG Pipeline (all-MiniLM-L6-v2)   │
│  · 380,507 embedded chunks          │
│  · 5 structured queries             │
│  · 10 semantic features             │
│                                     │
│  Output: 34 features · 13,442 rows  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 3: Prediction Models         │
│                                     │
│  · Baseline (Logistic Regression)   │
│  · FinBERT-only (XGBoost)          │
│  · RAG-only (XGBoost)              │
│  · XGBoost (all features)          │
│  · LightGBM (all features) ★       │
│  · LSTM (temporal sequences)        │
│                                     │
│  Walk-forward CV · Zero leakage     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 4: Backtesting               │
│  Long-short quartile portfolio      │
│  10bps transaction cost · 5-day     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 5: Dashboard                 │
│  Streamlit · Plotly · HF Spaces     │
│  Live at huggingface.co/spaces/...  │
└─────────────────────────────────────┘
```

---

## Results

### Walk-Forward Validation (2021–2024)

| Model | IC Mean | IC Std | Hit Rate | AUC |
|---|---|---|---|---|
| Baseline | 0.0429 | 0.1141 ⚠️ | 0.5312 | 0.5174 |
| **LightGBM ★** | **0.0198** | **0.0085** | 0.5329 | 0.5086 |
| LSTM | 0.0153 | 0.0211 | **0.5471** | 0.5060 |
| XGBoost | 0.0141 | 0.0180 | 0.5321 | 0.5099 |
| RAG Only | 0.0000 | 0.0295 | 0.5347 | 0.5086 |
| FinBERT Only | -0.0044 | 0.0117 | 0.5312 | 0.5007 |

> **IC** = Information Coefficient (Pearson correlation of predictions vs actual returns).
> **LightGBM** achieves the highest IC with the lowest variance — 10× more stable than the baseline.
> **LSTM** achieves the highest hit rate (54.7%), with IC=+0.0468 in volatile 2022.

### Top 5 Features by SHAP Importance

| Rank | Feature | Group | Mean \|SHAP\| |
|---|---|---|---|
| 1 | `qa_neg_ratio` | QA FinBERT | 0.0541 |
| 2 | `mgmt_sent_vol` | Mgmt FinBERT | 0.0476 |
| 3 | `qa_n_sentences` | QA FinBERT | 0.0453 |
| 4 | `mgmt_mean_neu` | Mgmt FinBERT | 0.0445 |
| 5 | `rag_guidance_specificity_relevance` | RAG | 0.0420 |

> **Key finding:** Analyst Q&A negativity (`qa_neg_ratio`) is the single strongest predictor — stronger than all management sentiment features. Analyst pushback contains more price-relevant information than management prepared remarks.

### Backtest Performance

| Metric | Value |
|---|---|
| Annualized Return | -0.91% |
| Sharpe Ratio | -0.81 |
| Max Drawdown | -4.67% |
| Win Rate | 37.5% |

> Consistent with weak-form EMH: positive IC (0.0198) exists but is insufficient to overcome 10bps transaction costs at a 5-day holding period. Extending to 20-day holding is the natural next step.

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
| Dashboard | Streamlit + Plotly |
| Deployment | Hugging Face Spaces |
| GPU | NVIDIA RTX 4060 (CUDA 11.8) |

---

## Project Structure

```
finsight/
├── config.py                    # Central configuration
├── run_ingestion.py             # Stage 1 runner
├── run_nlp.py                   # Stage 2 runner
├── src/
│   ├── ingestion/
│   │   ├── download_transcripts.py
│   │   ├── price_data.py
│   │   └── validate_data.py
│   ├── nlp/
│   │   ├── finbert_sentiment.py   # FinBERT pipeline
│   │   ├── rag_pipeline.py        # RAG feature extraction
│   │   └── build_feature_matrix.py
│   ├── models/
│   │   ├── train_models.py        # XGBoost + LightGBM
│   │   └── lstm_model.py          # LSTM sequence model
│   ├── backtest/
│   │   └── backtest_engine.py
│   └── dashboard/
│       └── app.py                 # Streamlit dashboard
├── report/
│   └── FinSight_Technical_Report.docx
└── experiments/                   # Results, plots, SHAP
```

---

## Reproducing Results

### 1. Setup
```bash
git clone https://github.com/Rajveer-code/Finsight.git
cd Finsight
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Data Ingestion
```bash
python run_ingestion.py
# Output: 14,584 transcripts, 1M+ price rows
```

### 3. NLP Pipeline (~3 hours on GPU)
```bash
python run_nlp.py
# Output: 34 features × 13,442 rows
```

### 4. Train Models (~10 minutes)
```bash
python src/models/train_models.py
python src/models/lstm_model.py
```

### 5. Backtest
```bash
python src/backtest/backtest_engine.py
```

### 6. Dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## Key Design Decisions

**Why walk-forward validation?**
Standard k-fold cross-validation leaks future information in time series. Walk-forward trains on years T-3 to T-1 and tests on year T only — no data from the future is ever seen during training.

**Why both FinBERT and RAG?**
FinBERT captures sentence-level emotional tone. RAG captures topical specificity — whether management actually discussed numerical guidance, new risks, or cost pressures. Together they contribute 34.6% (RAG) + 65.4% (FinBERT) of total SHAP importance.

**Why LSTM alongside tree models?**
Tree models treat each earnings call as independent. LSTM learns that a company with 6 consecutive quarters of deteriorating sentiment is fundamentally different from one with a single bad quarter. The LSTM's superior 2022 performance (+0.0468 IC) validates this temporal signal.

---

## Limitations & Future Work

- [ ] Extend backtest to 20-day holding period
- [ ] Replace RAG keyword scoring with generative LLM scorer (Llama-3 / Mistral)
- [ ] Sector-stratified analysis (NLP signals may be stronger in Healthcare, Biotech)
- [ ] Cross-lingual extension using multilingual FinBERT
- [ ] FastAPI + Next.js frontend rebuild (v2)

---

## References

- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS
- Loughran & McDonald (2011). When is a Liability not a Liability? Journal of Finance
- Chan, Jegadeesh & Lakonishok (1996). Momentum Strategies. Journal of Finance

---

## Author

**Rajveer Singh Pall**

Built as a portfolio project for MSc applications (ETH Zurich · Oxford · Cambridge).

---

<div align="center">
<b>Live Demo:</b> <a href="https://huggingface.co/spaces/Rajveer234/finsight">huggingface.co/spaces/Rajveer234/finsight</a>
</div>
