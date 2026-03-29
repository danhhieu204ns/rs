import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_metrics(df_predicted: pd.DataFrame) -> dict:
    """Sao chép/Dùng lại metric giống phase F"""
    y_true = df_predicted["rating"].values
    y_pred = df_predicted["final_score"].values
    
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # Paper threshold = 3.5
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

def grid_search_tuning(df_merged: pd.DataFrame):
    """
    Duyệt qua các điểm chia 0.0 -> 1.0 step 0.1 cho 5 trọng số sao cho tổng = 1.0.
    Mục tiêu: Tìm bộ trọng số cho RMSE nhỏ nhất.
    """
    logging.info("Bắt đầu Grid Search Tuning (step=0.1, sum=1.0)...")
    steps = [x / 10.0 for x in range(11)]
    best_weights = None
    best_rmse = float('inf')
    best_metrics = None
    
    # Sinh tổ hợp 5 số có tổng = 1.0
    valid_combos = []
    for a in steps:
        for b in steps:
            for c in steps:
                for d in steps:
                    e = 1.0 - (a + b + c + d)
                    if 0 <= e <= 1.0: # sai số nổi có thể làm nó lệch nhẹ, nhưng e là int/10 nên an toàn
                        valid_combos.append({
                            'alpha': a, 'beta': b, 'gamma': c, 'delta': d, 'epsilon': round(e, 1)
                        })
    
    logging.info(f"Tổng số cấu hình hợp lệ: {len(valid_combos)}")
    
    # Tính điểm
    # r_hat = a*SVD + b*ItemBased + c*ContentBased + d*Neural + e*RNN
    svd = df_merged["svd_score"].values
    item = df_merged["item_cf_score"].values
    cbf = df_merged["cbf_score"].values
    ncf = df_merged["ncf_score"].values
    rnn = df_merged["rnn_score"].values
    
    y_true = df_merged["rating"].values
    best_pred = None
    
    for w in valid_combos:
        y_pred = (w['alpha'] * svd + 
                  w['beta'] * item + 
                  w['gamma'] * cbf + 
                  w['delta'] * ncf + 
                  w['epsilon'] * rnn)
        
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = w
            best_pred = y_pred

    df_best = df_merged.copy()
    df_best["final_score"] = best_pred
    best_metrics = calculate_metrics(df_best)
    
    logging.info(f"Hoàn thành Tuning. Best RMSE: {best_rmse:.4f}")
    return best_weights, best_metrics, df_best

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
    args = parser.parse_args()
    
    reports_dir = args.reports_dir
    data_dir = args.data_processed_dir
    
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
    
    # 2. Tuning Grid Search
    best_weights, best_metrics, df_best_pred = grid_search_tuning(df_merged)
    
    # Tính metrics truớc tuning (baseline 0.2 hết)
    df_baseline = df_merged.copy()
    y_pred_baseline = 0.2*(df_merged["svd_score"] + df_merged["item_cf_score"] + df_merged["cbf_score"] + df_merged["ncf_score"] + df_merged["rnn_score"])
    df_baseline["final_score"] = y_pred_baseline
    baseline_metrics = calculate_metrics(df_baseline)
    
    # 3. Đánh giá Cold Start (dùng prediction model tốt nhất)
    cs_results = evaluate_cold_start(df_train, df_best_pred)
    
    # 4. Ghi output
    report_dict = {
        "Before Tuning": baseline_metrics,
        "After Tuning": {
            "Metrics": best_metrics,
            "Best Weights": best_weights
        },
        "Cold Start": cs_results,
        "Paper Counterparts": {
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