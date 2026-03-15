import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from config import PROCESSED_DIR, EXPERIMENTS_DIR

FINBERT_FEATURES = [
    'mgmt_mean_pos','mgmt_mean_neg','mgmt_mean_neu',
    'mgmt_net_sentiment','mgmt_neg_ratio','mgmt_sent_vol','mgmt_n_sentences',
    'qa_mean_pos','qa_mean_neg','qa_mean_neu',
    'qa_net_sentiment','qa_neg_ratio','qa_sent_vol','qa_n_sentences',
]
RAG_FEATURES = [
    'rag_guidance_specificity_score','rag_guidance_specificity_relevance',
    'rag_new_risks_score','rag_new_risks_relevance',
    'rag_management_confidence_score','rag_management_confidence_relevance',
    'rag_forward_looking_score','rag_forward_looking_relevance',
    'rag_cost_pressure_score','rag_cost_pressure_relevance',
]
ALL_FEATURES = FINBERT_FEATURES + RAG_FEATURES
TARGET = 'target_5d_up'

df = pd.read_parquet(PROCESSED_DIR / 'feature_matrix.parquet')
avail = [c for c in ALL_FEATURES if c in df.columns]
df_clean = df.dropna(subset=avail + [TARGET])

X = df_clean[avail].values
y = df_clean[TARGET].values

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# Use LightGBM — best model AND fully compatible with shap 0.49.1
model = LGBMClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)
model.fit(X_sc, y)
print('LightGBM model trained.')

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_sc)

# LightGBM returns [neg_class, pos_class] — take positive class
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

pd.DataFrame(shap_vals, columns=avail).to_parquet(
    EXPERIMENTS_DIR / 'shap_values.parquet', index=False)
print('Saved shap_values.parquet')

# Beeswarm plot
shap.summary_plot(shap_vals, X_sc, feature_names=avail, show=False, max_display=20)
plt.tight_layout()
plt.savefig(EXPERIMENTS_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved shap_summary.png')

# Feature importance bar chart
mean_shap = np.abs(shap_vals).mean(axis=0)
importance = pd.Series(mean_shap, index=avail).sort_values(ascending=True).tail(20)
colors = ['#2196F3' if 'rag' in c else '#4CAF50' if 'mgmt' in c else '#FF9800'
          for c in importance.index]
fig, ax = plt.subplots(figsize=(10, 7))
importance.plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel('Mean |SHAP Value|')
ax.set_title('Feature Importance — LightGBM (Best Model)\nBlue=RAG | Green=Mgmt FinBERT | Orange=QA FinBERT')
plt.tight_layout()
plt.savefig(EXPERIMENTS_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved feature_importance.png')

top5 = pd.Series(mean_shap, index=avail).sort_values(ascending=False).head(5)
print('\nTop 5 features by SHAP:')
print(top5.round(4).to_string())
print('\nDone.')