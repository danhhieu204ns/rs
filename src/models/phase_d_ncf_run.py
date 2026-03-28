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

from src.models.ncf_branch import fit_ncf, score_test_pairs_ncf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase D NCF scoring")
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/ratings_train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/ratings_test.csv"))
    parser.add_argument("--out-path", type=Path, default=Path("reports/phase_d_ncf_scores.csv"))
    parser.add_argument("--summary-path", type=Path, default=Path("reports/phase_d_ncf_summary.json"))
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    artifacts = fit_ncf(
        train_df,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    scored = score_test_pairs_ncf(test_df, artifacts)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out_path, index=False)

    summary = {
        "rows_scored": int(len(scored)),
        "ncf_score_finite": bool(np.isfinite(scored["ncf_score"]).all()),
        "n_users_train": int(train_df["user_id"].nunique()),
        "n_items_train": int(train_df["movie_id"].nunique()),
    }
    with args.summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
