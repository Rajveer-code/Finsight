"""
Stage 3 Extension — LSTM Sequence Model

Key insight XGBoost misses: a company with 6 quarters of
deteriorating sentiment is different from one with one bad quarter.
LSTM captures this temporal pattern.

Architecture:
    Input  : (batch, 6 quarters, 34 features)
    LSTM-1 : 64 hidden units, dropout 0.3
    LSTM-2 : 32 hidden units, dropout 0.3
    Dense  : 16 units + ReLU
    Output : 1 unit + Sigmoid

Training:
    - Walk-forward: train T-3 to T-1, test year T
    - Adam lr=0.001, early stopping patience=10
    - GPU accelerated on RTX 4060

Usage:
    python src/models/lstm_model.py
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, EXPERIMENTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────

SEQ_LEN      = 6        # quarters of history
HIDDEN_1     = 64
HIDDEN_2     = 32
DENSE        = 16
DROPOUT      = 0.3
BATCH_SIZE   = 64
LR           = 0.001
MAX_EPOCHS   = 60
PATIENCE     = 10       # early stopping
TEST_YEARS   = [2021, 2022, 2023, 2024]
TRAIN_WINDOW = 3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "mgmt_mean_pos","mgmt_mean_neg","mgmt_mean_neu",
    "mgmt_net_sentiment","mgmt_neg_ratio","mgmt_sent_vol","mgmt_n_sentences",
    "qa_mean_pos","qa_mean_neg","qa_mean_neu",
    "qa_net_sentiment","qa_neg_ratio","qa_sent_vol","qa_n_sentences",
    "rag_guidance_specificity_score","rag_guidance_specificity_relevance",
    "rag_new_risks_score","rag_new_risks_relevance",
    "rag_management_confidence_score","rag_management_confidence_relevance",
    "rag_forward_looking_score","rag_forward_looking_relevance",
    "rag_cost_pressure_score","rag_cost_pressure_relevance",
]
TARGET     = "target_5d_up"
RETURN_COL = "ret_5d"


# ── Dataset ────────────────────────────────────────────────────────────────────

class EarningsSequenceDataset(Dataset):
    """
    For each earnings call, build a sequence of the prior SEQ_LEN quarters
    for the same ticker. Pads with zeros if history is shorter than SEQ_LEN.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        seq_len: int = SEQ_LEN,
        scaler: StandardScaler = None,
        fit_scaler: bool = False,
    ):
        self.seq_len  = seq_len
        self.feat_dim = len(feature_cols)

        # Sort chronologically
        df = df.sort_values(["ticker","year","quarter"]).reset_index(drop=True)

        # Scale features
        feats = df[feature_cols].values.astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0)

        if fit_scaler:
            self.scaler = StandardScaler()
            feats = self.scaler.fit_transform(feats)
        else:
            self.scaler = scaler
            feats = self.scaler.transform(feats) if scaler else feats

        df = df.copy()
        df[feature_cols] = feats

        # Build sequences
        self.sequences = []
        self.targets   = []
        self.returns   = []
        self.meta      = []   # (ticker, year, quarter)

        for ticker, group in df.groupby("ticker", sort=False):
            group = group.sort_values(["year","quarter"]).reset_index(drop=True)
            feat_arr  = group[feature_cols].values.astype(np.float32)
            tgt_arr   = group[target_col].values
            ret_arr   = group[RETURN_COL].values if RETURN_COL in group else np.zeros(len(group))

            for i in range(len(group)):
                if pd.isna(tgt_arr[i]):
                    continue

                # Sequence: up to SEQ_LEN prior quarters (not including current)
                start = max(0, i - seq_len)
                hist  = feat_arr[start:i]               # shape (k, D), k <= seq_len

                # Pad with zeros at the front if history is short
                pad_len = seq_len - len(hist)
                if pad_len > 0:
                    pad = np.zeros((pad_len, self.feat_dim), dtype=np.float32)
                    hist = np.vstack([pad, hist]) if len(hist) > 0 else pad

                self.sequences.append(hist)             # (seq_len, D)
                self.targets.append(float(tgt_arr[i]))
                self.returns.append(float(ret_arr[i]) if not pd.isna(ret_arr[i]) else 0.0)
                self.meta.append((ticker,
                                  int(group.iloc[i]["year"]),
                                  int(group.iloc[i]["quarter"])))

        self.sequences = np.array(self.sequences, dtype=np.float32)  # (N, seq_len, D)
        self.targets   = np.array(self.targets,   dtype=np.float32)
        self.returns   = np.array(self.returns,   dtype=np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.targets[idx]),
        )


# ── Model ──────────────────────────────────────────────────────────────────────

class FinSightLSTM(nn.Module):
    """
    Two-layer LSTM for earnings sentiment sequences.

        Input  : (batch, seq_len, n_features)
        Output : (batch, 1) — probability of 5-day positive return
    """

    def __init__(self, input_dim: int, hidden1: int = HIDDEN_1,
                 hidden2: int = HIDDEN_2, dense: int = DENSE,
                 dropout: float = DROPOUT):
        super().__init__()

        self.lstm1 = nn.LSTM(input_dim, hidden1,
                             batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2,
                             batch_first=True, dropout=dropout)
        self.head  = nn.Sequential(
            nn.Linear(hidden2, dense),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out    = out[:, -1, :]    # take last timestep
        return self.head(out).squeeze(1)


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            total_loss += criterion(pred, y).item() * len(y)
            all_probs.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    return total_loss / len(loader.dataset), np.array(all_probs), np.array(all_targets)


def train_model(train_ds, val_ds):
    """Train with early stopping. Returns best model and training history."""
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = FinSightLSTM(input_dim=len(FEATURE_COLS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    history        = {"train_loss": [], "val_loss": []}

    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = eval_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            logger.info(f"  Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, history


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(returns, targets, probs):
    ic, _    = pearsonr(returns, probs)
    hit_rate = float(np.mean((probs >= 0.5).astype(int) == targets.astype(int)))
    auc      = roc_auc_score(targets, probs)
    return {
        "ic":       round(float(ic),       4),
        "hit_rate": round(float(hit_rate), 4),
        "auc":      round(float(auc),      4),
        "n_test":   int(len(targets)),
    }


# ── Walk-forward ───────────────────────────────────────────────────────────────

def walk_forward_lstm(df: pd.DataFrame) -> pd.DataFrame:
    avail_feats = [c for c in FEATURE_COLS if c in df.columns]
    all_results = []

    for test_year in TEST_YEARS:
        train_years = list(range(test_year - TRAIN_WINDOW, test_year))

        train_df = df[df["year"].isin(train_years)].copy()
        test_df  = df[df["year"] == test_year].copy()

        # Drop rows missing target or return
        train_df = train_df.dropna(subset=avail_feats + [TARGET])
        test_df  = test_df.dropna(subset=avail_feats + [TARGET, RETURN_COL])

        if len(train_df) < 100 or len(test_df) < 20:
            logger.warning(f"Insufficient data for {test_year}, skipping.")
            continue

        logger.info(f"\nYear {test_year} | train={len(train_df)} | test={len(test_df)}")

        # Build datasets — fit scaler on train only
        train_ds = EarningsSequenceDataset(
            train_df, avail_feats, TARGET, fit_scaler=True)
        test_ds  = EarningsSequenceDataset(
            test_df,  avail_feats, TARGET,
            scaler=train_ds.scaler, fit_scaler=False)

        # Use 15% of train as validation for early stopping
        val_size  = max(1, int(0.15 * len(train_ds)))
        train_sub = torch.utils.data.Subset(
            train_ds, range(len(train_ds) - val_size))
        val_sub   = torch.utils.data.Subset(
            train_ds, range(len(train_ds) - val_size, len(train_ds)))

        # Train
        model, history = train_model(
            torch.utils.data.Subset(train_ds, range(len(train_ds) - val_size)),
            torch.utils.data.Subset(train_ds, range(len(train_ds) - val_size, len(train_ds))),
        )

        # Evaluate on test set
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        _, probs, targets = eval_epoch(
            model, test_loader, nn.BCELoss())

        returns = test_ds.returns
        metrics = compute_metrics(returns, targets, probs)
        metrics.update({
            "model":     "LSTM",
            "test_year": test_year,
            "n_train":   len(train_ds),
            "epochs":    len(history["train_loss"]),
        })
        all_results.append(metrics)

        logger.info(
            f"LSTM | {test_year} | "
            f"IC={metrics['ic']:+.4f} | "
            f"HitRate={metrics['hit_rate']:.4f} | "
            f"AUC={metrics['auc']:.4f} | "
            f"Epochs={metrics['epochs']}"
        )

        # Save training curve
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history["train_loss"], label="Train", color="#64b5f6")
        ax.plot(history["val_loss"],   label="Val",   color="#ef5350")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.set_title(f"LSTM Training — Test Year {test_year}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            EXPERIMENTS_DIR / f"lstm_training_{test_year}.png",
            dpi=120, bbox_inches="tight")
        plt.close()

    return pd.DataFrame(all_results)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Device: {DEVICE}")

    fm_path = PROCESSED_DIR / "feature_matrix.parquet"
    if not fm_path.exists():
        logger.error("feature_matrix.parquet not found.")
        sys.exit(1)

    df = pd.read_parquet(fm_path)
    logger.info(f"Loaded {len(df):,} rows.")

    results = walk_forward_lstm(df)

    if results.empty:
        logger.error("No results generated.")
        sys.exit(1)

    # Save results
    out_path = EXPERIMENTS_DIR / "lstm_results.csv"
    results.to_csv(out_path, index=False)

    # Merge with existing model results for comparison
    existing_path = EXPERIMENTS_DIR / "model_results.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        combined = pd.concat([existing, results], ignore_index=True)
        combined.to_csv(existing_path, index=False)
        logger.info("Merged LSTM results into model_results.csv")

    # Print summary
    print("\n" + "="*55)
    print("LSTM — WALK-FORWARD SUMMARY")
    print("="*55)
    for _, row in results.iterrows():
        print(f"  {int(row['test_year'])} | "
              f"IC={row['ic']:+.4f} | "
              f"HitRate={row['hit_rate']:.4f} | "
              f"AUC={row['auc']:.4f} | "
              f"Epochs={int(row['epochs'])}")

    print("\nMean across years:")
    print(f"  IC       = {results['ic'].mean():+.4f} (std={results['ic'].std():.4f})")
    print(f"  Hit Rate = {results['hit_rate'].mean():.4f}")
    print(f"  AUC      = {results['auc'].mean():.4f}")
    print("="*55)
    print(f"\nSaved to: {out_path}")

    # Comparison chart vs XGBoost and LightGBM
    if existing_path.exists():
        all_res = pd.read_csv(existing_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors  = {
            "LightGBM_all": "#66bb6a",
            "XGBoost_all":  "#ef5350",
            "LSTM":         "#64b5f6",
            "Baseline":     "#ffa726",
        }
        for model in ["Baseline","XGBoost_all","LightGBM_all","LSTM"]:
            sub = all_res[all_res["model"]==model].sort_values("test_year")
            if sub.empty:
                continue
            ax.plot(sub["test_year"], sub["ic"],
                    marker="o", linewidth=2.5, markersize=8,
                    label=model, color=colors.get(model,"#b0bec5"))
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Test Year")
        ax.set_ylabel("Information Coefficient")
        ax.set_title("Walk-Forward IC — All Models Including LSTM")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(EXPERIMENTS_DIR / "ic_all_models.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved ic_all_models.png")


if __name__ == "__main__":
    main()