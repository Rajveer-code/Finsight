"""
Track B — RAG Pipeline (GPU-accelerated, no ChromaDB bottleneck)

Strategy: embed each transcript directly on GPU, compute cosine similarity
against pre-computed query embeddings in pure numpy. ~50x faster than
ChromaDB per-document filtering.

Usage:
    python src/nlp/rag_pipeline.py           # full run / resume
    python src/nlp/rag_pipeline.py --limit 50
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, EMBED_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_PATH     = PROCESSED_DIR / "rag_features.parquet"
CHECKPOINT_PATH = PROCESSED_DIR / "rag_checkpoint.json"
CHUNK_SIZE      = 400
CHUNK_OVERLAP   = 50
TOP_K           = 3
CHECKPOINT_EVERY = 500
ENCODE_BATCH    = 128   # encode this many chunks at once on GPU

RAG_QUERIES = {
    "guidance_specificity": (
        "What specific numerical guidance did management give about "
        "revenue, earnings, or growth for the next quarter or year?"
    ),
    "new_risks": (
        "What new risks or concerns did management mention that were "
        "not part of normal business operations?"
    ),
    "management_confidence": (
        "How confident and certain was management about their outlook "
        "and future performance?"
    ),
    "forward_looking": (
        "What forward-looking statements about future growth, expansion, "
        "or strategic initiatives did management make?"
    ),
    "cost_pressure": (
        "Did management mention cost pressures, margin compression, "
        "inflation, or supply chain issues?"
    ),
}


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words  = text.split()
    step   = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.split()) >= 20:
            chunks.append(chunk)
    return chunks


# ── Content scoring ────────────────────────────────────────────────────────────

def score_content(text: str, feature_type: str) -> float:
    """Keyword heuristic scoring of retrieved text for each feature."""
    score = 0.0
    if feature_type == "guidance_specificity":
        numbers = len(re.findall(r"\d+\.?\d*\s*%", text))
        dollars = len(re.findall(r"\$\s*\d+", text))
        words   = ["guidance","expect","forecast","project","anticipate",
                   "outlook","target","range","billion","million"]
        score   = min(1.0, numbers*0.15 + dollars*0.1 +
                      sum(0.05 for w in words if w in text))
    elif feature_type == "new_risks":
        words = ["risk","concern","challenge","headwind","uncertainty",
                 "pressure","macro","geopolit","regulatory","tariff",
                 "competition","disruption","impact","adverse"]
        score = min(1.0, sum(0.08 for w in words if w in text))
    elif feature_type == "management_confidence":
        pos = ["confident","strong","excellent","record","exceed","outperform",
               "momentum","robust","pleased","solid","growth","opportunity"]
        neg = ["uncertain","difficult","challenging","cautious","volatile",
               "headwind","concern","pressure"]
        p   = sum(1 for w in pos if w in text)
        n   = sum(1 for w in neg if w in text)
        score = (p / (p + n)) if (p + n) > 0 else 0.5
    elif feature_type == "forward_looking":
        words = ["will","plan","expect","intend","next quarter","next year",
                 "future","long-term","strategy","invest","expand","launch",
                 "develop","initiative"]
        score = min(1.0, sum(0.07 for w in words if w in text))
    elif feature_type == "cost_pressure":
        words = ["cost","inflation","margin","expense","supply chain","tariff",
                 "wage","pricing","input cost","compress","efficiency",
                 "restructur","headcount"]
        score = min(1.0, sum(0.08 for w in words if w in text))
    return round(score, 4)


# ── Cosine similarity (numpy, no DB needed) ────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and N chunk vectors.

    Parameters
    ----------
    a : np.ndarray  shape (D,)
    b : np.ndarray  shape (N, D)

    Returns
    -------
    np.ndarray shape (N,)
    """
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_norm @ a_norm


def extract_features_in_memory(
    text: str,
    ticker: str,
    year: int,
    quarter: int,
    query_embeddings: dict,   # {feat_name: np.ndarray shape (D,)}
    embed_model: SentenceTransformer,
) -> dict:
    """
    Extract RAG features for one transcript entirely in memory.
    No database calls — pure GPU embedding + numpy similarity.

    Parameters
    ----------
    text : str
    ticker, year, quarter : identifiers
    query_embeddings : pre-computed query vectors
    embed_model : SentenceTransformer

    Returns
    -------
    dict
    """
    features = {"ticker": ticker, "year": year, "quarter": quarter}

    # 1. Chunk the transcript
    chunks = chunk_text(text)
    if not chunks:
        for feat_name in RAG_QUERIES:
            features[f"rag_{feat_name}_score"]     = np.nan
            features[f"rag_{feat_name}_relevance"] = np.nan
        return features

    # 2. Embed all chunks on GPU in one shot
    chunk_embs = embed_model.encode(
        chunks,
        batch_size=ENCODE_BATCH,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )  # shape: (num_chunks, D)

    # 3. For each query, find top-K most similar chunks
    for feat_name, q_emb in query_embeddings.items():
        sims      = cosine_sim(q_emb, chunk_embs)          # (num_chunks,)
        top_idx   = np.argsort(sims)[::-1][:TOP_K]
        top_sims  = sims[top_idx]
        top_texts = [chunks[i] for i in top_idx]

        relevance     = float(np.mean(top_sims))
        combined_text = " ".join(top_texts).lower()

        features[f"rag_{feat_name}_relevance"] = round(relevance, 4)
        features[f"rag_{feat_name}_score"]     = score_content(combined_text, feat_name)

    return features


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    if not CHECKPOINT_PATH.exists():
        return set()
    with open(CHECKPOINT_PATH) as f:
        return set(tuple(x) for x in json.load(f)["done"])


def save_checkpoint(done: set, rows: list) -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"done": [list(x) for x in done]}, f)
    if rows:
        pd.DataFrame(rows).to_parquet(OUTPUT_PATH, index=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    transcripts_path = PROCESSED_DIR / "transcripts.parquet"
    if not transcripts_path.exists():
        logger.error("Run Stage 1 first.")
        sys.exit(1)

    df = pd.read_parquet(transcripts_path)
    if args.limit:
        df = df.head(args.limit)

    # Load model onto GPU
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL, device="cuda"
                                      if torch.cuda.is_available() else "cpu")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Pre-compute all query embeddings ONCE
    logger.info("Pre-computing query embeddings...")
    query_embeddings = {}
    for feat_name, query_text in RAG_QUERIES.items():
        emb = embed_model.encode(
            [query_text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        query_embeddings[feat_name] = emb
    logger.info(f"Ready. {len(query_embeddings)} queries pre-computed.")

    # Resume from checkpoint
    done     = load_checkpoint()
    all_rows = []

    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        all_rows = existing.to_dict("records")
        logger.info(f"Resuming from checkpoint: {len(all_rows):,} already done.")

    to_process = [
        row for _, row in df.iterrows()
        if (row["ticker"], int(row["year"]), int(row["quarter"])) not in done
    ]
    logger.info(f"Remaining: {len(to_process):,} transcripts.")

    for i, row in enumerate(tqdm(to_process, desc="RAG features")):
        try:
            feats = extract_features_in_memory(
                text             = row["transcript_text"],
                ticker           = row["ticker"],
                year             = int(row["year"]),
                quarter          = int(row["quarter"]),
                query_embeddings = query_embeddings,
                embed_model      = embed_model,
            )
            all_rows.append(feats)
            done.add((row["ticker"], int(row["year"]), int(row["quarter"])))
        except Exception as e:
            logger.warning(f"Failed {row['ticker']} {row['year']} Q{row['quarter']}: {e}")
            continue

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(done, all_rows)
            logger.info(f"Checkpoint: {len(all_rows):,} records saved.")

    save_checkpoint(done, all_rows)
    result_df = pd.DataFrame(all_rows)

    print("\n" + "="*55)
    print("RAG PIPELINE — SUMMARY")
    print("="*55)
    print(f"Transcripts processed : {len(result_df):,}")
    print(f"Unique tickers        : {result_df['ticker'].nunique()}")
    print(f"Features per row      : {len(result_df.columns) - 3}")
    score_cols = [c for c in result_df.columns if c.startswith("rag_")]
    for col in score_cols:
        print(f"  {col:<42} {result_df[col].mean():.4f}")
    print("="*55)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()