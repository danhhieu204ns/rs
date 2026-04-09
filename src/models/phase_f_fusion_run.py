from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_and_merge_scores(paths: Dict[str, Path]) -> pd.DataFrame:
    df_merged = None
    
    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {key} results at {p}")
        df_part = pd.read_csv(p)
        
        # Merge key
        merge_cols = ["user_id", "movie_id", "rating"]
        
        if df_merged is None:
            df_merged = df_part
        else:
            # Ensure no duplications or missing pairs
            df_merged = pd.merge(df_merged, df_part, on=merge_cols, how="inner")
            
    return df_merged


def compute_weighted_fusion(df: pd.DataFrame, alpha=0.2, beta=0.2, gamma=0.2, delta=0.2, epsilon=0.2) -> pd.DataFrame:
    """
    r_hat = a*SVD + b*ItemBased + c*ContentBased + d*Neural + e*RNN
    """
    df["final_score"] = (
        alpha * df["svd_score"] +
        beta * df["item_cf_score"] +
        gamma * df["cbf_score"] +
        delta * df["ncf_score"] +
        epsilon * df["rnn_score"]
    )
    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    y_true = df["rating"]
    y_pred = df["final_score"]
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    # Simple threshold-based precision/recall (threshold = 3.5 like standard ML100k protocols)
    y_true_bin = (y_true >= 3.5).astype(int)
    y_pred_bin = (y_pred >= 3.5).astype(int)
    
    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
    fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
    fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "Precision": float(precision),
        "Recall": float(recall)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase F Fusion and Metrics")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    
    # Weights for fusion
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.2)
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = args.reports_dir
    
    paths = {
        "CF": reports_dir / "phase_b_cf_scores.csv",
        "CBF": reports_dir / "phase_c_cbf_scores.csv",
        "NCF": reports_dir / "phase_d_ncf_scores.csv",
        "RNN": reports_dir / "phase_e_rnn_scores.csv"
    }
    
    df_merged = load_and_merge_scores(paths)
    df_fusion = compute_weighted_fusion(
        df_merged, 
        alpha=args.alpha, 
        beta=args.beta, 
        gamma=args.gamma, 
        delta=args.delta, 
        epsilon=args.epsilon
    )
    
    metrics = calculate_metrics(df_fusion)
    
    metrics["fusion_weights"] = {
        "alpha (SVD)": args.alpha,
        "beta (ItemBased)": args.beta,
        "gamma (ContentBased)": args.gamma,
        "delta (NCF)": args.delta,
        "epsilon (RNN)": args.epsilon
    }
    
    # Save results
    out_path = reports_dir / "phase_f_fusion_results.csv"
    summary_path = reports_dir / "phase_f_fusion_summary.json"
    
    df_fusion.to_csv(out_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("=== Fusion Metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
