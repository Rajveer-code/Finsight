from pathlib import Path

# Root
ROOT_DIR = Path(__file__).resolve().parent

# Data paths
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Source paths
SRC_DIR       = ROOT_DIR / "src"

# Output paths
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
REPORT_DIR      = ROOT_DIR / "report"

# Data settings
START_YEAR = 2018
END_YEAR   = 2024

# Model settings
FINBERT_MODEL = "ProsusAI/finbert"
EMBED_MODEL   = "all-MiniLM-L6-v2"
BATCH_SIZE    = 16
DEVICE        = "cuda"

# Backtest settings
TRANSACTION_COST_BPS = 10
REBALANCE_FREQ       = "Q"