"""
FinSight Dashboard — LLM-Powered Earnings Intelligence
Stage 5: Production Streamlit Dashboard

Pages:
  1. Overview      — project summary, pipeline, key stats
  2. Model Results — walk-forward IC/AUC comparison, year-by-year
  3. SHAP Analysis — interactive feature importance
  4. Backtest      — equity curve, drawdown, quarterly P&L
  5. Explorer      — browse transcripts with live sentiment

Run:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, EXPERIMENTS_DIR

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinSight | Earnings Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0e1a; color: #e8eaf6; }
  [data-testid="stSidebar"] {
      background: #0d1117;
      border-right: 1px solid #1e2433;
  }
  [data-testid="stSidebar"] .stRadio label {
      color: #8892b0 !important;
      font-size: 0.9rem;
  }
  .metric-card {
      background: linear-gradient(135deg, #0d1117 0%, #161b27 100%);
      border: 1px solid #1e2d4a;
      border-radius: 12px;
      padding: 20px 24px;
      text-align: center;
      transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #3d5a99; }
  .metric-value { font-size: 2rem; font-weight: 700; color: #64b5f6; line-height: 1.1; }
  .metric-label {
      font-size: 0.78rem; color: #8892b0;
      text-transform: uppercase; letter-spacing: 1px; margin-top: 6px;
  }
  .metric-delta { font-size: 0.82rem; margin-top: 4px; }
  .delta-pos { color: #66bb6a; }
  .delta-neg { color: #ef5350; }
  .delta-neu { color: #8892b0; }
  .section-header {
      font-size: 1.4rem; font-weight: 700; color: #e8eaf6;
      border-left: 4px solid #3d5a99; padding-left: 12px;
      margin: 28px 0 16px 0;
  }
  .subsection { font-size: 1rem; font-weight: 600; color: #8892b0; margin: 16px 0 8px 0; }
  .hero-title {
      font-size: 2.8rem; font-weight: 800;
      background: linear-gradient(90deg, #64b5f6, #7c4dff, #64b5f6);
      background-size: 200%;
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      line-height: 1.2;
  }
  .hero-sub {
      font-size: 1.1rem; color: #8892b0; margin-top: 8px;
      max-width: 680px; line-height: 1.6;
  }
  .pipeline-step {
      background: #0d1117; border: 1px solid #1e2433;
      border-radius: 10px; padding: 14px 16px; text-align: center;
  }
  .pipeline-icon { font-size: 1.6rem; }
  .pipeline-label { font-size: 0.78rem; color: #8892b0; margin-top: 4px; }
  .pipeline-title { font-size: 0.9rem; font-weight: 600; color: #cfd8dc; }
  .insight-box {
      background: #0d1117; border-left: 3px solid #3d5a99;
      border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0;
      font-size: 0.88rem; color: #b0bec5; line-height: 1.6;
  }
  .insight-box strong { color: #64b5f6; }
  .badge {
      display: inline-block; padding: 2px 10px; border-radius: 20px;
      font-size: 0.72rem; font-weight: 600; margin: 2px;
  }
  .badge-blue  { background: #1a237e22; color: #64b5f6; border: 1px solid #1a237e; }
  .badge-green { background: #1b5e2022; color: #66bb6a; border: 1px solid #1b5e20; }
  .badge-red   { background: #b71c1c22; color: #ef9a9a; border: 1px solid #b71c1c; }
  hr { border-color: #1e2433 !important; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0a0e1a; }
  ::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Layout helper — avoids duplicate xaxis/yaxis conflicts ────────────────────

BASE_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0a0e1a",
    font=dict(color="#b0bec5", family="Inter, sans-serif"),
    margin=dict(l=50, r=30, t=50, b=50),
    colorway=["#64b5f6","#66bb6a","#ffa726","#ef5350","#ab47bc","#26c6da"],
)
BASE_XAXIS = dict(gridcolor="#1a2035", linecolor="#1e2433", zerolinecolor="#1e2433")
BASE_YAXIS = dict(gridcolor="#1a2035", linecolor="#1e2433", zerolinecolor="#1e2433")

def L(**kwargs):
    """
    Merge base dark-theme layout with chart-specific overrides.
    Merges xaxis/yaxis dicts instead of replacing them, which avoids
    the 'multiple values for keyword argument xaxis' TypeError.
    """
    out = dict(**BASE_LAYOUT)
    if "xaxis" in kwargs:
        out["xaxis"] = {**BASE_XAXIS, **kwargs.pop("xaxis")}
    else:
        out["xaxis"] = BASE_XAXIS
    if "yaxis" in kwargs:
        out["yaxis"] = {**BASE_YAXIS, **kwargs.pop("yaxis")}
    else:
        out["yaxis"] = BASE_YAXIS
    out.update(kwargs)
    return out


# ── Global helpers ─────────────────────────────────────────────────────────────

def metric_card(col, value, label, delta="", delta_type="neu"):
    """Render a dark-theme KPI card inside a Streamlit column."""
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{value}</div>
        <div class='metric-label'>{label}</div>
        <div class='metric-delta delta-{delta_type}'>{delta}</div>
    </div>""", unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_feature_matrix():
    p = PROCESSED_DIR / "feature_matrix.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_model_results():
    p = EXPERIMENTS_DIR / "model_results.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_backtest():
    p = EXPERIMENTS_DIR / "backtest_results.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_shap():
    p = EXPERIMENTS_DIR / "shap_values.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:12px 0 20px 0;'>
        <div style='font-size:1.5rem;font-weight:800;color:#64b5f6;'>📈 FinSight</div>
        <div style='font-size:0.75rem;color:#8892b0;margin-top:4px;'>
            LLM-Powered Earnings Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Overview",
         "📊  Model Performance",
         "🔍  Feature Importance",
         "💹  Backtest Results",
         "🔎  Transcript Explorer"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem;color:#8892b0;line-height:1.8;'>
        <b style='color:#cfd8dc;'>Stack</b><br>
        FinBERT · ChromaDB · XGBoost<br>
        LightGBM · SHAP · Streamlit<br><br>
        <b style='color:#cfd8dc;'>Data</b><br>
        14,584 earnings transcripts<br>
        601 S&amp;P 500 companies<br>
        2018 – 2024<br><br>
        <b style='color:#cfd8dc;'>Author</b><br>
        Rajveer Singh Pall
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠  Overview":
    fm = load_feature_matrix()
    mr = load_model_results()

    st.markdown("""
    <div style='padding:24px 0 8px 0;'>
        <div class='hero-title'>FinSight</div>
        <div class='hero-title' style='font-size:1.8rem;color:#7c4dff;'>
            Earnings Intelligence System
        </div>
        <div class='hero-sub'>
            An end-to-end machine learning pipeline that extracts alpha signals
            from S&amp;P 500 earnings call transcripts using FinBERT sentiment analysis,
            RAG-based structured feature extraction, and walk-forward validated
            gradient boosting models.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    best_ic  = float(mr["ic"].max())       if not mr.empty else 0.0198
    best_auc = float(mr["auc"].max())      if not mr.empty else 0.5201
    best_hr  = float(mr["hit_rate"].max()) if not mr.empty else 0.5427
    n_rows   = len(fm) if not fm.empty else 13442

    c1,c2,c3,c4,c5 = st.columns(5)
    metric_card(c1, "14,584",         "Transcripts",      "601 companies",    "neu")
    metric_card(c2, f"{n_rows:,}",    "Training Samples", "2018–2024",        "neu")
    metric_card(c3, f"{best_ic:.4f}", "Best IC",          "LightGBM",         "pos")
    metric_card(c4, f"{best_auc:.4f}","Best AUC",         "XGBoost 2024",     "pos")
    metric_card(c5, f"{best_hr:.4f}", "Best Hit Rate",    "Walk-forward",     "pos")

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline
    st.markdown("<div class='section-header'>System Architecture</div>",
                unsafe_allow_html=True)
    steps = [
        ("🗄️","Stage 1","Data Ingestion",  "SEC EDGAR · yfinance\n14,584 transcripts"),
        ("🧠","Stage 2","NLP Pipeline",    "FinBERT · ChromaDB RAG\n34 features"),
        ("🤖","Stage 3","ML Models",        "XGBoost · LightGBM\nWalk-forward CV"),
        ("📉","Stage 4","Backtesting",       "Long-short strategy\n10bps TC"),
        ("🖥️","Stage 5","Dashboard",         "Streamlit · Plotly\nHugging Face Spaces"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, stage, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div class='pipeline-step'>
            <div class='pipeline-icon'>{icon}</div>
            <div class='pipeline-label'>{stage}</div>
            <div class='pipeline-title'>{title}</div>
            <div style='font-size:0.72rem;color:#546e7a;margin-top:4px;line-height:1.5;'>
                {desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("<div class='section-header'>Key Findings</div>",
                    unsafe_allow_html=True)
        for f in [
            "<strong>Analyst negativity &gt; management positivity.</strong> "
            "qa_neg_ratio (SHAP=0.054) is the single strongest feature. "
            "Analyst pushback in Q&amp;A contains more information than prepared remarks.",

            "<strong>NLP reduces prediction variance by 87%.</strong> "
            "Baseline IC std=0.114 vs LightGBM std=0.009 — "
            "far more consistent across years.",

            "<strong>Consistent with weak-form EMH.</strong> "
            "Positive IC (0.0198) exists but cannot overcome 10bps transaction "
            "costs at a 5-day holding period.",

            "<strong>RAG guidance relevance is top-5.</strong> "
            "Semantic relevance of the guidance section — not just its content — "
            "carries significant predictive signal.",
        ]:
            st.markdown(f"<div class='insight-box'>{f}</div>",
                        unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-header'>Dataset Coverage</div>",
                    unsafe_allow_html=True)
        if not fm.empty:
            yr = fm.groupby("year").size().reset_index(name="count")
            fig = go.Figure(go.Bar(
                x=yr["year"].astype(str),
                y=yr["count"],
                marker=dict(color=yr["count"],
                            colorscale=[[0,"#1a237e"],[1,"#64b5f6"]],
                            showscale=False),
                text=yr["count"], textposition="outside",
                textfont=dict(size=11),
            ))
            fig.update_layout(**L(title="Transcript Count by Year", height=300,
                                  showlegend=False,
                                  xaxis=dict(title="Year"),
                                  yaxis=dict(title="Transcripts")))
            st.plotly_chart(fig, use_container_width=True)

    # Sentiment heatmap
    if not fm.empty and "mgmt_net_sentiment" in fm.columns:
        st.markdown("<div class='section-header'>Sentiment Landscape</div>",
                    unsafe_allow_html=True)
        heat = (fm.groupby(["ticker","year"])["mgmt_net_sentiment"]
                  .mean().reset_index())
        top_t = fm["ticker"].value_counts().head(30).index
        heat  = heat[heat["ticker"].isin(top_t)]
        pivot = heat.pivot(index="ticker", columns="year",
                           values="mgmt_net_sentiment")
        fig2 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=pivot.index,
            colorscale=[[0,"#b71c1c"],[0.35,"#e53935"],
                        [0.5,"#263238"],[0.65,"#1565c0"],[1,"#64b5f6"]],
            zmid=0,
            colorbar=dict(title="Net Sentiment", tickfont=dict(size=10)),
            hovertemplate="Ticker: %{y}<br>Year: %{x}<br>Sentiment: %{z:.3f}<extra></extra>",
        ))
        fig2.update_layout(**L(
            title="Management Net Sentiment — Top 30 Tickers × Year",
            height=500,
            xaxis=dict(title="Year"),
            yaxis=dict(title=""),
        ))
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊  Model Performance":
    mr = load_model_results()

    st.markdown("<div class='hero-title' style='font-size:2rem;'>Model Performance</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Walk-forward validation (2021–2024). "
                "Train on 3 prior years, test on held-out year. Zero data leakage.</div>",
                unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if mr.empty:
        st.error("model_results.csv not found. Run Stage 3 first.")
        st.stop()

    summary = (
        mr.groupby("model")[["ic","hit_rate","auc"]]
        .agg({"ic":["mean","std"],"hit_rate":["mean","std"],"auc":["mean","std"]})
        .round(4)
    )
    summary.columns = ["IC Mean","IC Std","Hit Rate Mean","Hit Rate Std",
                        "AUC Mean","AUC Std"]
    summary = summary.sort_values("IC Mean", ascending=False)

    st.markdown("<div class='section-header'>Model Comparison</div>",
                unsafe_allow_html=True)

    def color_ic(val):
        if isinstance(val, float):
            if val > 0.015: return "color: #66bb6a; font-weight:600"
            if val < 0:     return "color: #ef5350"
        return ""

    st.dataframe(
        summary.style.applymap(color_ic, subset=["IC Mean"]).format("{:.4f}"),
        use_container_width=True, height=220,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Information Coefficient by Year</div>",
                unsafe_allow_html=True)

    MODEL_COLORS = {
        "Baseline":     "#ffa726",
        "FinBERT_only": "#26c6da",
        "RAG_only":     "#ab47bc",
        "XGBoost_all":  "#ef5350",
        "LightGBM_all": "#66bb6a",
    }

    fig = go.Figure()
    for m in mr["model"].unique():
        sub = mr[mr["model"]==m].sort_values("test_year")
        fig.add_trace(go.Scatter(
            x=sub["test_year"].astype(int),
            y=sub["ic"],
            mode="lines+markers", name=m,
            line=dict(color=MODEL_COLORS.get(m,"#64b5f6"), width=2.5),
            marker=dict(size=9),
            hovertemplate=f"<b>{m}</b><br>Year: %{{x}}<br>IC: %{{y:.4f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="#546e7a", line_width=1.2)
    fig.update_layout(**L(
        title="Walk-Forward IC — Positive = Predictive",
        height=380,
        xaxis=dict(tickvals=[2021,2022,2023,2024], title="Year"),
        yaxis=dict(title="Information Coefficient"),
        legend=dict(bgcolor="#0d1117", bordercolor="#1e2433", borderwidth=1),
    ))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='subsection'>Hit Rate by Year</div>",
                    unsafe_allow_html=True)
        fig2 = go.Figure()
        for m in mr["model"].unique():
            sub = mr[mr["model"]==m].sort_values("test_year")
            fig2.add_trace(go.Scatter(
                x=sub["test_year"].astype(int), y=sub["hit_rate"],
                mode="lines+markers", name=m,
                line=dict(color=MODEL_COLORS.get(m,"#64b5f6"), width=2),
                marker=dict(size=7), showlegend=False,
            ))
        fig2.add_hline(y=0.5, line_dash="dot", line_color="#546e7a", line_width=1)
        fig2.update_layout(**L(
            height=300, title="Hit Rate (>0.5 = better than coin flip)",
            xaxis=dict(tickvals=[2021,2022,2023,2024]),
            yaxis=dict(title="Hit Rate"),
        ))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("<div class='subsection'>AUC by Year</div>",
                    unsafe_allow_html=True)
        fig3 = go.Figure()
        for m in mr["model"].unique():
            sub = mr[mr["model"]==m].sort_values("test_year")
            fig3.add_trace(go.Scatter(
                x=sub["test_year"].astype(int), y=sub["auc"],
                mode="lines+markers", name=m,
                line=dict(color=MODEL_COLORS.get(m,"#64b5f6"), width=2),
                marker=dict(size=7), showlegend=False,
            ))
        fig3.add_hline(y=0.5, line_dash="dot", line_color="#546e7a", line_width=1)
        fig3.update_layout(**L(
            height=300, title="AUC-ROC (>0.5 = better than random)",
            xaxis=dict(tickvals=[2021,2022,2023,2024]),
            yaxis=dict(title="AUC"),
        ))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Stability Analysis — IC Variance</div>",
                unsafe_allow_html=True)
    ic_std  = mr.groupby("model")["ic"].std().sort_values()
    ic_mean = mr.groupby("model")["ic"].mean()
    bar_colors = ["#66bb6a" if ic_mean[m] > 0 else "#ef5350" for m in ic_std.index]

    fig4 = go.Figure(go.Bar(
        y=ic_std.index, x=ic_std.values, orientation="h",
        marker_color=bar_colors,
        text=[f"σ={v:.4f}" for v in ic_std.values],
        textposition="outside", textfont=dict(size=11),
    ))
    fig4.update_layout(**L(
        title="IC Standard Deviation — Lower = More Consistent",
        height=280,
        xaxis=dict(title="IC Std Dev"),
        yaxis=dict(title=""),
    ))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
        <strong>Interpretation:</strong> The Baseline's high IC mean (0.043) is
        misleading — its std of 0.114 shows extreme instability driven by lucky
        quarters. LightGBM achieves IC=0.020 with std=0.009, making it
        <strong>10× more stable</strong>. In live trading, consistency matters
        far more than occasional lucky peaks.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍  Feature Importance":
    shap_df = load_shap()
    fm      = load_feature_matrix()

    st.markdown("<div class='hero-title' style='font-size:2rem;'>Feature Importance</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>SHAP values computed on LightGBM (best model). "
                "Shows which features actually drive predictions.</div>",
                unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if shap_df.empty:
        st.error("shap_values.parquet not found. Run run_shap.py first.")
        st.stop()

    mean_shap = shap_df.abs().mean().sort_values(ascending=False)

    def feat_color(name):
        if name.startswith("rag_"):  return "#64b5f6"
        if name.startswith("mgmt_"): return "#66bb6a"
        if name.startswith("qa_"):   return "#ffa726"
        return "#ab47bc"

    def feat_group(name):
        if name.startswith("rag_"):  return "RAG Features"
        if name.startswith("mgmt_"): return "Management FinBERT"
        if name.startswith("qa_"):   return "QA FinBERT"
        return "Other"

    st.markdown("<div class='section-header'>Top 20 Features by Mean |SHAP|</div>",
                unsafe_allow_html=True)
    top20 = mean_shap.head(20)[::-1]
    fig = go.Figure(go.Bar(
        y=top20.index, x=top20.values, orientation="h",
        marker_color=[feat_color(n) for n in top20.index],
        text=[f"{v:.4f}" for v in top20.values],
        textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**L(
        height=520,
        title="Feature Importance — 🔵 RAG | 🟢 Mgmt FinBERT | 🟠 QA FinBERT",
        xaxis=dict(title="Mean |SHAP Value|"),
        yaxis=dict(title=""),
    ))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Importance by Feature Group</div>",
                    unsafe_allow_html=True)
        gs = (mean_shap.reset_index()
              .rename(columns={"index":"feature", 0:"shap"}))
        gs.columns = ["feature","shap"]
        gs["group"] = gs["feature"].apply(feat_group)
        gt = gs.groupby("group")["shap"].sum()

        fig2 = go.Figure(go.Pie(
            labels=gt.index, values=gt.values, hole=0.55,
            marker=dict(colors=["#64b5f6","#66bb6a","#ffa726","#ab47bc"]),
            textinfo="label+percent", textfont=dict(size=12),
            hovertemplate="<b>%{label}</b><br>Total SHAP: %{value:.4f}<br>%{percent}<extra></extra>",
        ))
        fig2.update_layout(**L(
            height=320, showlegend=False,
            annotations=[dict(text="SHAP<br>Groups", x=0.5, y=0.5,
                              font_size=13, showarrow=False,
                              font_color="#b0bec5")],
        ))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>SHAP vs Correlation with Target</div>",
                    unsafe_allow_html=True)
        if not fm.empty and "target_5d_up" in fm.columns:
            feat_cols = [c for c in shap_df.columns if c in fm.columns]
            corrs = fm[feat_cols+["target_5d_up"]].corr()["target_5d_up"].drop("target_5d_up")
            cdf = pd.DataFrame({
                "feature": corrs.index,
                "shap":    mean_shap.reindex(corrs.index).fillna(0).values,
                "corr":    corrs.values,
                "group":   [feat_group(f) for f in corrs.index],
            })
            cmap = {
                "RAG Features":       "#64b5f6",
                "Management FinBERT": "#66bb6a",
                "QA FinBERT":         "#ffa726",
                "Other":              "#ab47bc",
            }
            fig3 = px.scatter(
                cdf, x="corr", y="shap", color="group",
                color_discrete_map=cmap, hover_data=["feature"],
                labels={"corr":"Pearson Corr with Target",
                        "shap":"Mean |SHAP Value|"},
                height=320,
            )
            fig3.add_vline(x=0, line_dash="dash", line_color="#546e7a")
            fig3.update_layout(**L(
                title="SHAP Importance vs Linear Correlation",
                showlegend=False,
            ))
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Feature Insights</div>",
                unsafe_allow_html=True)
    insights = [
        ("🏆 #1 — qa_neg_ratio",
         "Proportion of negative sentences in analyst Q&amp;A. When analysts "
         "push back hard, it signals market-moving information that management "
         "tried to downplay."),
        ("📊 #2 — mgmt_sent_vol",
         "Volatility of management's sentence-level sentiment. Inconsistent "
         "messaging — mixing optimism with caution — often precedes larger "
         "price moves."),
        ("📝 #3 — qa_n_sentences",
         "Length of the Q&amp;A section. Longer Q&amp;A sessions indicate "
         "more analyst scrutiny, which correlates with uncertainty about "
         "the quarter's results."),
        ("😶 #4 — mgmt_mean_neu",
         "Neutral sentiment ratio in management remarks. Deliberately neutral "
         "language can mask very good or very bad news — a hedging signal."),
        ("🎯 #5 — rag_guidance_relevance",
         "Semantic similarity of the guidance section to specific numerical "
         "guidance queries. More relevant guidance sections contain concrete "
         "targets that markets react to more strongly."),
    ]
    cols = st.columns(len(insights))
    for col, (title, body) in zip(cols, insights):
        col.markdown(f"""
        <div class='pipeline-step' style='text-align:left;height:190px;'>
            <div style='font-size:0.82rem;font-weight:700;color:#64b5f6;
                        margin-bottom:8px;'>{title}</div>
            <div style='font-size:0.76rem;color:#8892b0;line-height:1.6;'>
                {body}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "💹  Backtest Results":
    bt = load_backtest()

    st.markdown("<div class='hero-title' style='font-size:2rem;'>Backtest Results</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Long-short quartile portfolio. "
                "Long top-25% predicted stocks, short bottom-25%. "
                "5-day holding period. 10bps round-trip transaction cost.</div>",
                unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if bt.empty:
        st.error("backtest_results.csv not found. Run Stage 4 first.")
        st.stop()

    bt = bt.sort_values(["year","quarter"]).reset_index(drop=True)
    bt["period"] = bt["year"].astype(str) + "-Q" + bt["quarter"].astype(str)
    rets  = bt["net_ret"]
    cum   = (1 + rets).cumprod()
    peak  = cum.cummax()
    dd    = (cum - peak) / peak

    n_yrs   = len(bt) / 4
    ann_ret = float((1 + rets).prod() ** (1/n_yrs) - 1)
    ann_vol = float(rets.std() * np.sqrt(4))
    sharpe  = ann_ret / ann_vol if ann_vol != 0 else 0.0
    max_dd  = float(dd.min())
    hit     = float((rets > 0).mean())

    c1,c2,c3,c4,c5 = st.columns(5)
    metric_card(c1, f"{ann_ret*100:.2f}%", "Ann. Return",
                "After TC", "pos" if ann_ret > 0 else "neg")
    metric_card(c2, f"{sharpe:.3f}", "Sharpe Ratio",
                ">1.0 = excellent", "pos" if sharpe > 0 else "neg")
    metric_card(c3, f"{max_dd*100:.2f}%", "Max Drawdown",
                "Peak-to-trough", "neg")
    metric_card(c4, f"{hit*100:.0f}%", "Win Rate",
                "Profitable quarters", "pos" if hit > 0.5 else "neg")
    metric_card(c5, str(len(bt)), "Quarters Tested",
                "2021–2024", "neu")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Equity Curve</div>",
                unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, row_heights=[0.72,0.28],
                        shared_xaxes=True, vertical_spacing=0.04)
    fig.add_trace(go.Scatter(
        x=bt["period"], y=cum.values,
        mode="lines+markers",
        line=dict(color="#64b5f6", width=2.5),
        marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(100,181,246,0.06)",
        name="Cumulative Return",
        hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#546e7a",
                  line_width=1, row=1, col=1)
    fig.add_trace(go.Bar(
        x=bt["period"], y=dd.values*100,
        marker_color="#ef5350", opacity=0.7, name="Drawdown %",
        hovertemplate="<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0a0e1a",
        font=dict(color="#b0bec5"),
        margin=dict(l=50,r=30,t=50,b=80),
        title="FinSight Long-Short Strategy — 2021 to 2024",
        height=500, showlegend=False,
        xaxis2=dict(tickangle=45, tickfont_size=10,
                    gridcolor="#1a2035", linecolor="#1e2433"),
        yaxis=dict(title="Cumulative Return",
                   gridcolor="#1a2035", linecolor="#1e2433"),
        yaxis2=dict(title="DD %",
                    gridcolor="#1a2035", linecolor="#1e2433"),
    )
    fig.update_xaxes(gridcolor="#1a2035", linecolor="#1e2433")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='subsection'>Quarterly Net Returns</div>",
                    unsafe_allow_html=True)
        q_colors = ["#66bb6a" if r > 0 else "#ef5350" for r in rets]
        fig2 = go.Figure(go.Bar(
            x=bt["period"], y=rets.values*100,
            marker_color=q_colors,
            text=[f"{v*100:.2f}%" for v in rets.values],
            textposition="outside", textfont=dict(size=9),
            hovertemplate="<b>%{x}</b><br>Net Return: %{y:.2f}%<extra></extra>",
        ))
        fig2.add_hline(y=0, line_color="#546e7a", line_width=1)
        fig2.update_layout(**L(
            height=320,
            title="Net Return per Quarter (after 10bps TC)",
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(title="Net Return (%)"),
        ))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("<div class='subsection'>Long vs Short Leg Hit Rate</div>",
                    unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=bt["period"], y=bt["long_hit"],
            mode="lines+markers",
            line=dict(color="#66bb6a", width=2),
            marker=dict(size=7), name="Long Leg",
        ))
        fig3.add_trace(go.Scatter(
            x=bt["period"], y=bt["short_hit"],
            mode="lines+markers",
            line=dict(color="#ef5350", width=2),
            marker=dict(size=7), name="Short Leg",
        ))
        fig3.add_hline(y=0.5, line_dash="dot", line_color="#546e7a")
        fig3.update_layout(**L(
            height=320,
            title="Direction Accuracy — Long &amp; Short Legs",
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(title="Hit Rate"),
            legend=dict(bgcolor="#0d1117", bordercolor="#1e2433"),
        ))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Quarterly Breakdown</div>",
                unsafe_allow_html=True)
    disp = bt[["period","net_ret","long_ret","short_ret",
               "long_hit","short_hit","n_stocks","q_size"]].copy()
    disp.columns = ["Quarter","Net Ret","Long Ret","Short Ret",
                    "Long Hit","Short Hit","N Stocks","Leg Size"]

    def color_ret(val):
        if isinstance(val, float):
            if val > 0: return "color: #66bb6a"
            if val < 0: return "color: #ef5350"
        return ""

    st.dataframe(
        disp.style.applymap(color_ret,
                            subset=["Net Ret","Long Ret","Short Ret"])
                  .format({c:"{:.4f}" for c in
                           ["Net Ret","Long Ret","Short Ret",
                            "Long Hit","Short Hit"]}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("""
    <div class='insight-box'>
        <strong>Context:</strong> A Sharpe of -0.81 with a 5-day holding period
        is consistent with academic literature on post-earnings announcement
        drift (Chan et al. 1996, Lerman et al. 2008). The signal exists
        (IC=0.0198) but is too weak to survive round-trip transaction costs at
        this frequency. Extending to 20-day holding periods is the natural
        next step.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — TRANSCRIPT EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔎  Transcript Explorer":
    fm = load_feature_matrix()

    st.markdown("<div class='hero-title' style='font-size:2rem;'>Transcript Explorer</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Browse sentiment profiles for any company "
                "and quarter in the dataset.</div>",
                unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if fm.empty:
        st.error("Feature matrix not found.")
        st.stop()

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        all_tickers = sorted(fm["ticker"].dropna().unique())
        default_idx = all_tickers.index("AAPL") if "AAPL" in all_tickers else 0
        ticker = st.selectbox("Select Ticker", all_tickers, index=default_idx)
    with col2:
        years  = sorted(fm["year"].unique(), reverse=True)
        year   = st.selectbox("Year", years)
    with col3:
        quarters = sorted(fm[fm["year"]==year]["quarter"].unique())
        quarter  = st.selectbox("Quarter", quarters)

    row = fm[(fm["ticker"]==ticker) &
             (fm["year"]==year) &
             (fm["quarter"]==quarter)]

    if row.empty:
        st.warning("No data for this combination.")
        st.stop()

    row = row.iloc[0]

    ret_5d = row.get("ret_5d", 0)
    target = int(row.get("target_5d_up", 0))
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:16px;margin:16px 0;'>
        <div style='font-size:2rem;font-weight:800;color:#64b5f6;'>{ticker}</div>
        <div style='font-size:1rem;color:#8892b0;'>{int(year)} Q{int(quarter)}</div>
        <div class='badge badge-{"green" if target==1 else "red"}'>
            {"▲ UP" if target==1 else "▼ DOWN"} 5d
        </div>
        <div class='badge badge-blue'>
            5d Return: {float(ret_5d)*100:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("<div class='subsection'>Sentiment Breakdown</div>",
                    unsafe_allow_html=True)
        cats = ["Mgmt Positive","Mgmt Neutral","Mgmt Negative",
                "QA Positive","QA Neutral","QA Negative"]
        vals = [
            float(row.get("mgmt_mean_pos", 0) or 0),
            float(row.get("mgmt_mean_neu", 0) or 0),
            float(row.get("mgmt_mean_neg", 0) or 0),
            float(row.get("qa_mean_pos",   0) or 0),
            float(row.get("qa_mean_neu",   0) or 0),
            float(row.get("qa_mean_neg",   0) or 0),
        ]
        vals_c = vals + [vals[0]]
        cats_c = cats + [cats[0]]
        fig = go.Figure(go.Scatterpolar(
            r=vals_c, theta=cats_c, fill="toself",
            fillcolor="rgba(100,181,246,0.15)",
            line=dict(color="#64b5f6", width=2), name=ticker,
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117",
            font=dict(color="#b0bec5"),
            polar=dict(
                bgcolor="#0d1117",
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor="#1a2035", linecolor="#1a2035",
                                tickfont=dict(size=9, color="#546e7a")),
                angularaxis=dict(gridcolor="#1a2035", linecolor="#1a2035",
                                 tickfont=dict(size=10, color="#b0bec5")),
            ),
            height=360, showlegend=False,
            title=f"{ticker} — Sentiment Radar",
            margin=dict(l=40,r=40,t=50,b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("<div class='subsection'>Feature Scores</div>",
                    unsafe_allow_html=True)

        def score_bar(label, val, invert=False):
            if val is None or pd.isna(val):
                return
            v     = float(val)
            pct   = max(0, min(1, v)) * 100
            color = "#ef5350" if invert else "#64b5f6"
            st.markdown(f"""
            <div style='margin:8px 0;'>
                <div style='display:flex;justify-content:space-between;
                            font-size:0.8rem;color:#8892b0;margin-bottom:3px;'>
                    <span>{label}</span><span>{v:.3f}</span>
                </div>
                <div style='background:#1a2035;border-radius:4px;height:6px;'>
                    <div style='background:{color};width:{pct:.0f}%;
                                height:6px;border-radius:4px;'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        score_bar("Mgmt Net Sentiment",    row.get("mgmt_net_sentiment"))
        score_bar("QA Net Sentiment",      row.get("qa_net_sentiment"))
        score_bar("Mgmt Negativity",       row.get("mgmt_neg_ratio"),    invert=True)
        score_bar("QA Negativity",         row.get("qa_neg_ratio"),      invert=True)
        score_bar("Guidance Specificity",  row.get("rag_guidance_specificity_score"))
        score_bar("Mgmt Confidence",       row.get("rag_management_confidence_score"))
        score_bar("Forward Looking",       row.get("rag_forward_looking_score"))
        score_bar("New Risks",             row.get("rag_new_risks_score"),     invert=True)
        score_bar("Cost Pressure",         row.get("rag_cost_pressure_score"), invert=True)

    # Historical trend
    st.markdown(f"<div class='section-header'>{ticker} — Historical Sentiment</div>",
                unsafe_allow_html=True)

    td = fm[fm["ticker"]==ticker].copy().sort_values(["year","quarter"])
    td["period"] = td["year"].astype(str) + "-Q" + td["quarter"].astype(str)

    if len(td) > 1:
        fig2 = go.Figure()
        for col_name, label, color in [
            ("mgmt_net_sentiment", "Mgmt Sentiment", "#66bb6a"),
            ("qa_net_sentiment",   "QA Sentiment",   "#64b5f6"),
            ("mgmt_neg_ratio",     "Mgmt Negativity","#ef5350"),
        ]:
            if col_name in td.columns:
                fig2.add_trace(go.Scatter(
                    x=td["period"], y=td[col_name],
                    mode="lines+markers", name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y:.3f}}<extra></extra>",
                ))

        # Mark selected quarter — use index position to avoid type issues
        cur_period = f"{int(year)}-Q{int(quarter)}"
        if cur_period in td["period"].values:
            cur_idx = td[td["period"]==cur_period].index[0]
            cur_pos = td["period"].tolist().index(cur_period)
            fig2.add_vrect(
                x0=cur_period, x1=cur_period,
                line_dash="dot", line_color="#ffa726", line_width=2,
            )

        fig2.add_hline(y=0, line_dash="dash", line_color="#546e7a", line_width=0.8)
        fig2.update_layout(**L(
            height=320,
            title=f"{ticker} — Sentiment Over Time",
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(title="Score"),
            legend=dict(bgcolor="#0d1117", bordercolor="#1e2433"),
        ))
        st.plotly_chart(fig2, use_container_width=True)

        # Scatter: sentiment vs return
        if "ret_5d" in td.columns and "mgmt_net_sentiment" in td.columns:
            st.markdown("<div class='subsection'>Sentiment vs 5-Day Return</div>",
                        unsafe_allow_html=True)
            tc = td.dropna(subset=["ret_5d","mgmt_net_sentiment"]).copy()
            tc["ret_pct"] = tc["ret_5d"].astype(float) * 100
            sc_colors = ["#66bb6a" if r > 0 else "#ef5350"
                         for r in tc["ret_pct"]]
            fig3 = go.Figure(go.Scatter(
                x=tc["mgmt_net_sentiment"].astype(float),
                y=tc["ret_pct"],
                mode="markers+text",
                text=tc["period"],
                textposition="top center",
                textfont=dict(size=8, color="#546e7a"),
                marker=dict(color=sc_colors, size=9, opacity=0.85),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Mgmt Sentiment: %{x:.3f}<br>"
                    "5d Return: %{y:.2f}%<extra></extra>"
                ),
            ))
            fig3.add_vline(x=0, line_dash="dash", line_color="#546e7a")
            fig3.add_hline(y=0, line_dash="dash", line_color="#546e7a")
            fig3.update_layout(**L(
                height=340,
                title=f"{ticker} — Mgmt Sentiment vs 5-Day Return",
                xaxis=dict(title="Management Net Sentiment"),
                yaxis=dict(title="5-Day Return (%)"),
            ))
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Not enough historical data for this ticker.")