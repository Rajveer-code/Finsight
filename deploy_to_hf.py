"""
Deploy FinSight to Hugging Face Spaces.
Run from D:\finsight\ with venv active.
"""
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID   = "Rajveer234/finsight"
LOCAL_DIR = Path(".")
api       = HfApi()

# ── Files to upload ────────────────────────────────────────────────────────────
uploads = [
    # (local path,                          path in Space)
    ("src/dashboard/app.py",                "app.py"),
    ("config.py",                           "config.py"),
    ("data/processed/feature_matrix.parquet",   "data/processed/feature_matrix.parquet"),
    ("data/processed/finbert_features.parquet", "data/processed/finbert_features.parquet"),
    ("data/processed/rag_features.parquet",     "data/processed/rag_features.parquet"),
    ("experiments/model_results.csv",           "experiments/model_results.csv"),
    ("experiments/backtest_results.csv",        "experiments/backtest_results.csv"),
    ("experiments/shap_values.parquet",         "experiments/shap_values.parquet"),
]

# ── requirements.txt for Spaces ────────────────────────────────────────────────
REQUIREMENTS = """streamlit==1.41.1
plotly==5.24.1
pandas==2.2.3
numpy==1.26.4
pyarrow==18.1.0
scikit-learn==1.5.2
lightgbm==4.5.0
"""

# ── config.py for Spaces (uses relative paths) ────────────────────────────────
CONFIG = """from pathlib import Path
ROOT_DIR       = Path(__file__).resolve().parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
START_YEAR     = 2018
END_YEAR       = 2024
FINBERT_MODEL  = "ProsusAI/finbert"
EMBED_MODEL    = "all-MiniLM-L6-v2"
BATCH_SIZE     = 16
DEVICE         = "cpu"
TRANSACTION_COST_BPS = 10
REBALANCE_FREQ = "Q"
"""

print("Writing requirements.txt and config.py for Spaces...")
Path("hf_deploy/requirements.txt").parent.mkdir(exist_ok=True)
Path("hf_deploy/requirements.txt").write_text(REQUIREMENTS)
Path("hf_deploy/config.py").write_text(CONFIG)

print(f"\nUploading to {REPO_ID}...")

# Upload requirements and config
for local, remote in [
    ("hf_deploy/requirements.txt", "requirements.txt"),
    ("hf_deploy/config.py",        "config.py"),
]:
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=REPO_ID,
        repo_type="space",
    )
    print(f"  ✅ {remote}")

# Upload all data and source files
for local, remote in uploads:
    p = Path(local)
    if not p.exists():
        print(f"  ⚠️  MISSING: {local} — skipping")
        continue
    api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=remote,
        repo_id=REPO_ID,
        repo_type="space",
    )
    print(f"  ✅ {remote}")

print(f"""
╔══════════════════════════════════════════════════════╗
║  DEPLOYMENT COMPLETE                                 ║
║  Your app: https://huggingface.co/spaces/Rajveer234/finsight  ║
║  Wait ~2 minutes for Space to rebuild               ║
╚══════════════════════════════════════════════════════╝
""")