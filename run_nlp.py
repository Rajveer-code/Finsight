"""
Stage 2 master runner — NLP pipeline.

Usage:
    python run_nlp.py --test         # 20 transcripts only
    python run_nlp.py                # full run (2-4 hours)
    python run_nlp.py --rebuild-rag  # force rebuild ChromaDB
"""

import argparse
import subprocess
import sys


def run(cmd: list) -> None:
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",        action="store_true", help="Run on 20 transcripts")
    parser.add_argument("--rebuild-rag", action="store_true", help="Rebuild ChromaDB")
    args = parser.parse_args()

    limit       = ["--limit", "20"] if args.test else []
    rebuild_rag = ["--rebuild"]     if args.rebuild_rag else []

    # Track A — FinBERT
    run([sys.executable, "src/nlp/finbert_sentiment.py"] + limit)

    # Track B — RAG
    run([sys.executable, "src/nlp/rag_pipeline.py"] + limit + rebuild_rag)

    # Merge into feature matrix
    run([sys.executable, "src/nlp/build_feature_matrix.py"])

    print("\n✅ Stage 2 complete. Feature matrix ready for Stage 3.")


if __name__ == "__main__":
    main()