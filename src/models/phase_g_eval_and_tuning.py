import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FEATURE_COLS = ["svd_score", "item_cf_score", "cbf_score", "ncf_score", "rnn_score"]


def _weights_array_to_dict(w: np.ndarray) -> dict:
    return {
        "alpha": float(w[0]),
        "beta": float(w[1]),
        "gamma": float(w[2]),
        "delta": float(w[3]),
        "epsilon": float(w[4]),
    }


def _predict_weighted(df: pd.DataFrame, w: np.ndarray, clip_scores: bool = True) -> np.ndarray:
    y_pred = df[FEATURE_COLS].to_numpy(dtype=np.float32) @ w.astype(np.float32)
    if clip_scores:
        y_pred = np.clip(y_pred, 1.0, 5.0)
    return y_pred


def _recall_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
    fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def _objective_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    objective: str,
    threshold: float,
    recall_weight: float,
) -> float:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if objective == "rmse":
        return rmse
    if objective == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    if objective == "rmse_recall":
        recall = _recall_from_arrays(y_true, y_pred, threshold)
        return rmse - (recall_weight * recall)
    raise ValueError(f"Unsupported objective: {objective}")


def _generate_grid_weights(grid_step: float):
    if grid_step <= 0 or grid_step > 1:
        raise ValueError("grid_step must be in (0, 1].")

    scale = int(round(1.0 / grid_step))
    if not np.isclose(scale * grid_step, 1.0):
        raise ValueError("grid_step must divide 1.0 evenly (e.g., 0.1, 0.05, 0.02).")

    for a in range(scale + 1):
        for b in range(scale + 1 - a):
            for c in range(scale + 1 - a - b):
                for d in range(scale + 1 - a - b - c):
                    e = scale - (a + b + c + d)
                    yield np.array([a, b, c, d, e], dtype=np.float64) / scale

def calculate_metrics(df_predicted: pd.DataFrame, threshold: float = 3.5) -> dict:
    """Sao chép/Dùng lại metric giống phase F"""
    y_true = df_predicted["rating"].values
    y_pred = df_predicted["final_score"].values
    
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    
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

def tune_fusion_weights(
    df_tune: pd.DataFrame,
    grid_step: float,
    random_trials: int,
    objective: str,
    threshold: float,
    recall_weight: float,
    seed: int,
    clip_scores: bool,
):
    """
    Tuning 2 tầng:
    1) Grid search trên simplex (coarse)
    2) Random search Dirichlet (fine)
    """
    y_true = df_tune["rating"].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)

    best_w = None
    best_obj = float("inf")
    best_source = ""
    grid_count = 0

    logging.info(
        "Bắt đầu tuning fusion: objective=%s, grid_step=%.4f, random_trials=%d",
        objective,
        grid_step,
        random_trials,
    )

    for w in _generate_grid_weights(grid_step):
        y_pred = _predict_weighted(df_tune, w, clip_scores=clip_scores)
        obj = _objective_value(
            y_true=y_true,
            y_pred=y_pred,
            objective=objective,
            threshold=threshold,
            recall_weight=recall_weight,
        )
        grid_count += 1

        if obj < best_obj:
            best_obj = obj
            best_w = w.copy()
            best_source = "grid"

    for _ in range(random_trials):
        w = rng.dirichlet(np.ones(5, dtype=np.float64))
        y_pred = _predict_weighted(df_tune, w, clip_scores=clip_scores)
        obj = _objective_value(
            y_true=y_true,
            y_pred=y_pred,
            objective=objective,
            threshold=threshold,
            recall_weight=recall_weight,
        )

        if obj < best_obj:
            best_obj = obj
            best_w = w.copy()
            best_source = "random"

    if best_w is None:
        raise RuntimeError("Could not find a valid fusion weight configuration.")

    logging.info("Hoàn thành tuning: best objective=%.6f (from %s)", best_obj, best_source)
    return _weights_array_to_dict(best_w), {
        "objective": objective,
        "best_objective_value": float(best_obj),
        "best_source": best_source,
        "grid_step": grid_step,
        "grid_config_count": int(grid_count),
        "random_trials": int(random_trials),
        "seed": int(seed),
        "clip_scores": bool(clip_scores),
    }

def evaluate_cold_start(df_train: pd.DataFrame, df_test_scored: pd.DataFrame) -> dict:
    """
    Đánh giá mô hình trên Cold-Start subset (New Users, New Items).
    - New Users: User xuất hiện trong test nhưng k có trong train. Do ML-100k user ít nhất 20 ratings -> 0 overlap nếu chia 80/20 ngẫu nhiên toàn cục là hiếm. Ta dùng mask n_interactions <= 25 (thay thế giả định cold-start).
    - New Items: Item xuất hiện trong test nhưng ko có trong train.
    """
    train_users = set(df_train['user_id'].unique())
    train_items = set(df_train['movie_id'].unique())
    
    # Strict "New Users" (ko có trong train)
    new_users_strict = df_test_scored[~df_test_scored['user_id'].isin(train_users)]
    
    # Strict "New Items" (ko có trong train)
    new_items_strict = df_test_scored[~df_test_scored['movie_id'].isin(train_items)]
    
    # Nếu bộ new_users strict rỗng, ta lấy top 5% user ít tương tác nhất làm subset
    if new_users_strict.empty:
        logging.warning("Không có True New Users. Lấy các users có <= 25 interaction trong tập train để coi là Cold Users.")
        user_counts = df_train['user_id'].value_counts()
        cold_users = set(user_counts[user_counts <= 25].index)
        df_new_users = df_test_scored[df_test_scored['user_id'].isin(cold_users)]
    else:
        df_new_users = new_users_strict
        
    logging.info(f"Số mẫu quy vào Cold-start Users: {len(df_new_users)}")
    logging.info(f"Số mẫu quy vào Cold-start Items (True New): {len(new_items_strict)}")
    
    cs_results = {}
    
    if not df_new_users.empty:
        metrics_u = calculate_metrics(df_new_users)
        cs_results["New Users"] = metrics_u
    else:
        cs_results["New Users"] = None
        
    if not new_items_strict.empty:
        metrics_i = calculate_metrics(new_items_strict)
        cs_results["New Items"] = metrics_i
    else:
        cs_results["New Items"] = None
        
    return cs_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--data-processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument("--random-trials", type=int, default=3000)
    parser.add_argument("--objective", choices=["rmse", "mae", "rmse_recall"], default="rmse")
    parser.add_argument("--recall-weight", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-clip-scores", action="store_true")
    args = parser.parse_args()
    
    reports_dir = args.reports_dir
    data_dir = args.data_processed_dir
    clip_scores = not args.no_clip_scores

    if not (0.0 < args.holdout_ratio < 1.0):
        raise ValueError("--holdout-ratio must be in (0, 1).")
    if args.random_trials < 0:
        raise ValueError("--random-trials must be >= 0.")
    
    # 1. Load data
    df_cf = pd.read_csv(reports_dir / "phase_b_cf_scores.csv")
    df_cbf = pd.read_csv(reports_dir / "phase_c_cbf_scores.csv")
    df_ncf = pd.read_csv(reports_dir / "phase_d_ncf_scores.csv")
    df_rnn = pd.read_csv(reports_dir / "phase_e_rnn_scores.csv")
    
    # Merge
    df_merged = df_cf.merge(df_cbf, on=["user_id", "movie_id", "rating"])
    df_merged = df_merged.merge(df_ncf, on=["user_id", "movie_id", "rating"])
    df_merged = df_merged.merge(df_rnn, on=["user_id", "movie_id", "rating"])
    
    # Đọc lại train data cho cold-start test
    df_train = pd.read_csv(data_dir / "ratings_train.csv")

    # 2) Holdout split cho fusion tuning (tránh tune và eval trên cùng dữ liệu)
    df_tune, df_eval = train_test_split(
        df_merged,
        test_size=args.holdout_ratio,
        random_state=args.seed,
        shuffle=True,
    )
    df_tune = df_tune.reset_index(drop=True)
    df_eval = df_eval.reset_index(drop=True)

    # 3) Baseline trước tuning (equal weights) trên holdout eval
    equal_w = np.asarray([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    df_eval_baseline = df_eval.copy()
    df_eval_baseline["final_score"] = _predict_weighted(df_eval_baseline, equal_w, clip_scores=clip_scores)
    baseline_metrics = calculate_metrics(df_eval_baseline, threshold=args.threshold)

    # 4) Tuning trọng số trên tune split
    best_weights, tuning_info = tune_fusion_weights(
        df_tune=df_tune,
        grid_step=args.grid_step,
        random_trials=args.random_trials,
        objective=args.objective,
        threshold=args.threshold,
        recall_weight=args.recall_weight,
        seed=args.seed,
        clip_scores=clip_scores,
    )

    best_w_array = np.asarray(
        [
            best_weights["alpha"],
            best_weights["beta"],
            best_weights["gamma"],
            best_weights["delta"],
            best_weights["epsilon"],
        ],
        dtype=np.float32,
    )

    # 5) Đánh giá sau tuning trên holdout eval
    df_eval_tuned = df_eval.copy()
    df_eval_tuned["final_score"] = _predict_weighted(df_eval_tuned, best_w_array, clip_scores=clip_scores)
    best_metrics = calculate_metrics(df_eval_tuned, threshold=args.threshold)

    # 6) Xuất prediction tuned trên full merged để tiện so sánh và cold-start
    df_best_pred = df_merged.copy()
    df_best_pred["final_score"] = _predict_weighted(df_best_pred, best_w_array, clip_scores=clip_scores)

    df_full_baseline = df_merged.copy()
    df_full_baseline["final_score"] = _predict_weighted(df_full_baseline, equal_w, clip_scores=clip_scores)

    full_before = calculate_metrics(df_full_baseline, threshold=args.threshold)
    full_after = calculate_metrics(df_best_pred, threshold=args.threshold)

    # 7) Đánh giá Cold Start (dùng prediction model tốt nhất)
    cs_results = evaluate_cold_start(df_train, df_best_pred)

    # 8) Ghi output
    report_dict = {
        "Protocol": {
            "fusion_holdout_ratio": float(args.holdout_ratio),
            "threshold": float(args.threshold),
            "seed": int(args.seed),
            "objective": args.objective,
            "recall_weight": float(args.recall_weight),
            "clip_scores": bool(clip_scores),
            "tune_rows": int(len(df_tune)),
            "eval_rows": int(len(df_eval)),
        },
        "Before Tuning": baseline_metrics,
        "After Tuning": {
            "Metrics": best_metrics,
            "Best Weights": best_weights
        },
        "Full Data Reference": {
            "Before Tuning": full_before,
            "After Tuning": full_after,
        },
        "Tuning Diagnostics": tuning_info,
        "Cold Start": cs_results,
        "Counterparts": {
            "Target After Tuning": {"RMSE": 0.7723, "MAE": 0.6018, "Precision": 0.8127, "Recall": 0.7312},
            "Target Before Tuning": {"RMSE": 0.930, "MAE": 0.730, "Precision": 0.730, "Recall": 0.645},
            "Cold-start Target New Users": {"Precision": 0.762, "Recall": 0.685},
            "Cold-start Target New Items": {"MAE": 0.612, "Precision": 0.788, "Recall": 0.702}
        }
    }
    
    output_path = reports_dir / "phase_g_eval_and_tuning.json"
    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=4)
        
    df_best_pred.to_csv(reports_dir / "phase_g_tuned_predictions.csv", index=False)
    
    logging.info(f"Đã xuất báo cáo cuối vào: {output_path}")

if __name__ == "__main__":
    main()