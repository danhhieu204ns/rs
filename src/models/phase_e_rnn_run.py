from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rnn_branch import fit_rnn, score_test_pairs_rnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase E RNN scoring")
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/ratings_train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/ratings_test.csv"))
    parser.add_argument("--out-path", type=Path, default=Path("reports/phase_e_rnn_scores.csv"))
    parser.add_argument("--summary-path", type=Path, default=Path("reports/phase_e_rnn_summary.json"))
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    # --- Disable GPU for RNN on DirectML due to allocation tracking bugs ---
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    # -----------------------------------------------------------------------
    
    args = parse_args()
    
    # Check if files exist
    if not args.train_path.exists() or not args.test_path.exists():
        print("Train or test data not found. Please run Data Pipeline (Phase A) first.")
        sys.exit(1)
        
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    artifacts = fit_rnn(
        train_df,
        embedding_dim=args.embedding_dim,
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    scored = score_test_pairs_rnn(test_df, artifacts)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out_path, index=False)

    summary = {
        "rows_scored": int(len(scored)),
        "rnn_score_finite": bool(np.isfinite(scored["rnn_score"]).all()),
        "n_users_train": int(train_df["user_id"].nunique()),
        "n_items_train": int(train_df["movie_id"].nunique()),
        "max_seq_len": args.max_seq_len,
    }
    with args.summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
