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

from src.models.cbf_branch import fit_cbf_artifacts, score_test_pairs_cbf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase C CBF branch scoring")
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/ratings_train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/ratings_test.csv"))
    parser.add_argument("--items-path", type=Path, default=Path("data/processed/items_clean.csv"))
    parser.add_argument("--out-path", type=Path, default=Path("reports/phase_c_cbf_scores.csv"))
    parser.add_argument("--summary-path", type=Path, default=Path("reports/phase_c_cbf_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    items_df = pd.read_csv(args.items_path)

    artifacts = fit_cbf_artifacts(items_df, train_df)
    scored = score_test_pairs_cbf(test_df, train_df, artifacts)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out_path, index=False)

    summary = {
        "rows_scored": int(len(scored)),
        "cbf_score_finite": bool(np.isfinite(scored["cbf_score"]).all()),
        "item_vocab_size": int(artifacts.tfidf_matrix.shape[1]),
        "item_count": int(len(artifacts.movie_ids)),
    }

    with args.summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
