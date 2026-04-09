from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


METHOD_SPECS: List[Tuple[str, str, str, bool]] = [
    ("SVD", "phase_b_cf_scores.csv", "svd_score", False),
    ("ItemCF", "phase_b_cf_scores.csv", "item_cf_score", False),
    ("CBF", "phase_c_cbf_scores.csv", "cbf_score", False),
    ("NCF", "phase_d_ncf_scores.csv", "ncf_score", False),
    ("RNN", "phase_e_rnn_scores.csv", "rnn_score", False),
    ("Fusion(0.2 each)", "phase_f_fusion_results.csv", "final_score", False),
    ("Fusion(tuned)", "phase_g_tuned_predictions.csv", "final_score", True),
]


def calculate_metrics(df: pd.DataFrame, pred_col: str, threshold: float) -> Dict[str, float]:
    y_true = df["rating"].to_numpy()
    y_pred = df[pred_col].to_numpy()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)

    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Precision": precision,
        "Recall": recall,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


def build_comparison_table(reports_dir: Path, threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for method_name, file_name, pred_col, optional in METHOD_SPECS:
        file_path = reports_dir / file_name

        if not file_path.exists():
            if optional:
                continue
            raise FileNotFoundError(f"Missing required file for {method_name}: {file_path}")

        df = pd.read_csv(file_path)
        required_cols = {"rating", pred_col}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(
                f"{file_path} is missing columns {sorted(missing)} required for {method_name}."
            )

        metrics = calculate_metrics(df, pred_col=pred_col, threshold=threshold)

        rows.append(
            {
                "Method": method_name,
                "Rows": int(len(df)),
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
            }
        )

    if not rows:
        raise ValueError("No model result files were found to compare.")

    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame, metric_cols: List[str]) -> str:
    columns = ["Method"] + metric_cols
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    lines = [header, separator]
    for _, row in df.iterrows():
        vals = [str(row["Method"])]
        for col in metric_cols:
            vals.append(f"{row[col]:.6f}")
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and export per-method comparison metrics vs fusion"
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--threshold", type=float, default=3.5)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/phase_h_model_comparison.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/phase_h_model_comparison.md"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/phase_h_model_comparison.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    comparison_df = build_comparison_table(
        reports_dir=args.reports_dir,
        threshold=args.threshold,
    )

    metric_cols = ["RMSE", "MAE", "Precision", "Recall"]

    for path in [args.out_csv, args.out_md, args.out_json]:
        path.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(args.out_csv, index=False)

    md_text = dataframe_to_markdown(comparison_df, metric_cols=metric_cols)
    args.out_md.write_text(md_text, encoding="utf-8")

    payload = {
        "threshold": args.threshold,
        "metrics": metric_cols,
        "results": comparison_df.to_dict(orient="records"),
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Model Comparison Metrics ===")
    print(comparison_df[["Method"] + metric_cols].to_string(index=False))
    print(f"CSV exported to: {args.out_csv}")
    print(f"Markdown exported to: {args.out_md}")
    print(f"JSON exported to: {args.out_json}")


if __name__ == "__main__":
    main()