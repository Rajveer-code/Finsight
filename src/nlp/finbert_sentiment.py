"""
Track A — FinBERT Sentiment Analysis
Processes all earnings call transcripts through FinBERT.
Produces sentence-level sentiment aggregated to document-level features.

Output: data/processed/finbert_features.parquet

Usage:
    python src/nlp/finbert_sentiment.py
    python src/nlp/finbert_sentiment.py --limit 20
"""

import argparse
import logging
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import nltk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, FINBERT_MODEL, BATCH_SIZE, DEVICE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = PROCESSED_DIR / "finbert_checkpoint.json"
OUTPUT_PATH     = PROCESSED_DIR / "finbert_features.parquet"


# ── NLTK setup ────────────────────────────────────────────────────────────────

def ensure_nltk() -> None:
    """Download NLTK sentence tokenizer if not present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt_tab", quiet=True)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_finbert(device: str) -> tuple:
    """
    Load FinBERT tokenizer and model.

    Parameters
    ----------
    device : str
        'cuda' or 'cpu'

    Returns
    -------
    tuple
        (tokenizer, model, device_obj)
    """
    logger.info(f"Loading FinBERT model: {FINBERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

    # Use GPU if available
    actual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(actual_device) == "cpu" and device == "cuda":
        logger.warning("CUDA not available — running on CPU (slower).")

    model = model.to(actual_device)
    model.eval()

    if str(actual_device) == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU.")

    return tokenizer, model, actual_device


# ── Text splitting ─────────────────────────────────────────────────────────────

def split_transcript(text: str) -> tuple[list[str], list[str]]:
    """
    Split transcript into management prepared remarks vs analyst Q&A.

    Parameters
    ----------
    text : str

    Returns
    -------
    tuple[list[str], list[str]]
        (management_sentences, qa_sentences)
    """
    text_lower = text.lower()
    text_len   = len(text)

    # Q&A begins when analysts start asking questions.
    # We only look for these markers AFTER the first 25% of the document
    # to avoid matching the call-opening "Operator:" introduction.
    min_offset = int(text_len * 0.25)

    qa_markers = [
        "question-and-answer",
        "q&a session",
        "question and answer",
        "open the floor to questions",
        "we will now begin the question",
        "we'll now begin the question",
        "first question comes from",
        "first question is from",
        "our first question",
        "take your first question",
    ]

    split_idx = text_len  # default: everything is management
    for marker in qa_markers:
        idx = text_lower.find(marker, min_offset)
        if idx != -1 and idx < split_idx:
            split_idx = idx

    management_text = text[:split_idx].strip()
    qa_text         = text[split_idx:].strip()

    mgmt_sentences = nltk.sent_tokenize(management_text) if management_text else []
    qa_sentences   = nltk.sent_tokenize(qa_text)         if qa_text         else []

    # Filter very short sentences (noise)
    mgmt_sentences = [s for s in mgmt_sentences if len(s.split()) >= 5]
    qa_sentences   = [s for s in qa_sentences   if len(s.split()) >= 5]

    return mgmt_sentences, qa_sentences


# ── Batch inference ────────────────────────────────────────────────────────────

def score_sentences(
    sentences: list[str],
    tokenizer,
    model,
    device,
    batch_size: int = BATCH_SIZE
) -> list[dict]:
    """
    Run FinBERT on a list of sentences in batches.

    Parameters
    ----------
    sentences : list[str]
    tokenizer, model, device : FinBERT components
    batch_size : int

    Returns
    -------
    list[dict]
        Each dict has keys: positive, negative, neutral
    """
    if not sentences:
        return []

    results = []
    label_map = {0: "positive", 1: "negative", 2: "neutral"}

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            if str(device) == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(**encoded)
            else:
                outputs = model(**encoded)

        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        for row in probs:
            results.append({
                "positive": float(row[0]),
                "negative": float(row[1]),
                "neutral":  float(row[2]),
            })

    return results


# ── Feature aggregation ────────────────────────────────────────────────────────

def aggregate_features(scores: list[dict], prefix: str = "") -> dict:
    """
    Aggregate sentence-level FinBERT scores into document-level features.

    Parameters
    ----------
    scores : list[dict]
    prefix : str
        e.g. 'mgmt_' or 'qa_'

    Returns
    -------
    dict
    """
    if not scores:
        return {
            f"{prefix}mean_pos":      np.nan,
            f"{prefix}mean_neg":      np.nan,
            f"{prefix}mean_neu":      np.nan,
            f"{prefix}net_sentiment": np.nan,
            f"{prefix}neg_ratio":     np.nan,
            f"{prefix}sent_vol":      np.nan,
            f"{prefix}n_sentences":   0,
        }

    pos = np.array([s["positive"] for s in scores])
    neg = np.array([s["negative"] for s in scores])
    neu = np.array([s["neutral"]  for s in scores])

    net = pos - neg

    return {
        f"{prefix}mean_pos":      float(np.mean(pos)),
        f"{prefix}mean_neg":      float(np.mean(neg)),
        f"{prefix}mean_neu":      float(np.mean(neu)),
        f"{prefix}net_sentiment": float(np.mean(net)),
        f"{prefix}neg_ratio":     float(np.mean(neg > 0.5)),
        f"{prefix}sent_vol":      float(np.std(net)),
        f"{prefix}n_sentences":   len(scores),
    }


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    """Load set of already-processed (ticker, year, quarter) tuples."""
    if not CHECKPOINT_PATH.exists():
        return set()
    with open(CHECKPOINT_PATH) as f:
        data = json.load(f)
    return set(tuple(x) for x in data)


def save_checkpoint(done: set) -> None:
    """Persist checkpoint to disk."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump([list(x) for x in done], f)


# ── Main processing loop ───────────────────────────────────────────────────────

def process_transcripts(
    df: pd.DataFrame,
    tokenizer,
    model,
    device,
    checkpoint_every: int = 100
) -> pd.DataFrame:
    """
    Run FinBERT on all transcripts, with checkpointing.

    Parameters
    ----------
    df : pd.DataFrame
    tokenizer, model, device : FinBERT components
    checkpoint_every : int

    Returns
    -------
    pd.DataFrame
        Feature rows.
    """
    done      = load_checkpoint()
    all_rows  = []

    # Load existing partial results if any
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        all_rows = existing.to_dict("records")
        logger.info(f"Resuming — {len(all_rows)} records already processed.")

    to_process = [
        row for _, row in df.iterrows()
        if (row["ticker"], row["year"], row["quarter"]) not in done
    ]

    logger.info(f"Transcripts to process: {len(to_process)} "
                f"(skipping {len(done)} already done)")

    for i, row in enumerate(tqdm(to_process, desc="FinBERT sentiment")):
        try:
            mgmt_sents, qa_sents = split_transcript(row["transcript_text"])

            mgmt_scores = score_sentences(mgmt_sents, tokenizer, model, device)
            qa_scores   = score_sentences(qa_sents,   tokenizer, model, device)

            mgmt_feats = aggregate_features(mgmt_scores, prefix="mgmt_")
            qa_feats   = aggregate_features(qa_scores,   prefix="qa_")

            record = {
                "ticker":  row["ticker"],
                "company": row["company"],
                "year":    int(row["year"]),
                "quarter": int(row["quarter"]),
                "date":    row["date"],
                **mgmt_feats,
                **qa_feats,
            }
            all_rows.append(record)
            done.add((row["ticker"], int(row["year"]), int(row["quarter"])))

        except Exception as e:
            logger.warning(f"Failed {row['ticker']} {row['year']} Q{row['quarter']}: {e}")
            continue

        # Checkpoint and save every N records
        if (i + 1) % checkpoint_every == 0:
            save_checkpoint(done)
            pd.DataFrame(all_rows).to_parquet(OUTPUT_PATH, index=False)
            logger.info(f"Checkpoint: {len(all_rows)} records saved.")

    # Final save
    save_checkpoint(done)
    result_df = pd.DataFrame(all_rows)
    result_df.to_parquet(OUTPUT_PATH, index=False)
    return result_df


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinBERT sentiment on transcripts.")
    parser.add_argument("--limit", type=int, default=None, help="Max transcripts (for testing)")
    args = parser.parse_args()

    ensure_nltk()

    transcripts_path = PROCESSED_DIR / "transcripts.parquet"
    if not transcripts_path.exists():
        logger.error("Run Stage 1 first.")
        sys.exit(1)

    df = pd.read_parquet(transcripts_path)
    if args.limit:
        df = df.head(args.limit)

    logger.info(f"Loaded {len(df)} transcripts.")

    tokenizer, model, device = load_finbert(DEVICE)

    result_df = process_transcripts(df, tokenizer, model, device)

    print("\n" + "="*55)
    print("FINBERT SENTIMENT — SUMMARY")
    print("="*55)
    print(f"Transcripts processed : {len(result_df):,}")
    print(f"Unique tickers        : {result_df['ticker'].nunique()}")
    print(f"\nFeature means:")
    numeric_cols = result_df.select_dtypes(include=np.number).columns
    for col in sorted(numeric_cols):
        print(f"  {col:<30} {result_df[col].mean():.4f}")
    print("="*55)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()