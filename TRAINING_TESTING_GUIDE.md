# HRS-IU-DL Training and Testing Guide

Tai lieu nay huong dan chay code theo tung phase, test tung phase, test tong the, va quy trinh training theo code hien tai trong workspace.

## 1. Pham vi code hien tai

Da co code va test cho:
- Phase A: Data ingestion, cleaning, split 80/20, QA report
- Phase B: CF branch (SVD + Item-based cosine)
- Phase C: CBF branch (TF-IDF)
- Phase D: NCF branch (TensorFlow)
- GPU check utility

Chua co code day du cho:
- Phase E: RNN branch
- Phase F: Hybrid weighted fusion (5 thanh phan)
- Phase G: Main evaluation va cold-start final report

## 2. Cau truc file quan trong

- src/data/phase_a_data_pipeline.py
- src/models/cf_branch.py
- src/models/phase_b_cf_run.py
- src/models/cbf_branch.py
- src/models/phase_c_cbf_run.py
- src/models/ncf_branch.py
- src/models/phase_d_ncf_run.py
- src/models/check_gpu.py
- tests/test_phase_a_data_pipeline.py
- tests/test_cf_branch.py
- tests/test_cbf_branch.py
- tests/test_ncf_branch.py
- requirements.txt

## 3. Setup moi truong

Buoc 1: mo PowerShell tai thu muc project
D:\PTIT\Ki 2 nam 4\RS

Buoc 2: kich hoat virtual environment
.\.venv\Scripts\Activate.ps1

Buoc 3: cai dependency
python -m pip install -r requirements.txt

## 4. Kiem tra GPU

Lenh kiem tra:
python src/models/check_gpu.py

Ky vong neu chay GPU qua DirectML:
- tensorflow_version: 2.10.0
- gpu_count >= 1
- co PhysicalDevice device_type GPU

Neu gpu_count = 0:
- Kiem tra driver GPU
- Kiem tra da cai tensorflow-directml-plugin
- Kiem tra dang dung dung venv cua project

## 5. Chay va test tung phase

### Phase A - Data

Run:
python src/data/phase_a_data_pipeline.py

Output du kien:
- data/raw/ml-100k/...
- data/processed/ratings_clean.csv
- data/processed/items_clean.csv
- data/processed/ratings_train.csv
- data/processed/ratings_test.csv
- data/processed/split_indices.json
- reports/phase_a_qa_report.json

Test phase A:
python -m unittest tests.test_phase_a_data_pipeline

Tieu chi pass nhanh:
- users = 943
- movies = 1682
- ratings = 100000
- split test_ratio = 0.2
- no_leakage = true

### Phase B - CF (SVD + Item cosine)

Run:
python src/models/phase_b_cf_run.py

Output du kien:
- reports/phase_b_cf_scores.csv
- reports/phase_b_cf_summary.json

Test phase B:
python -m unittest tests.test_cf_branch

Tieu chi pass nhanh:
- rows_scored = so dong test
- svd_score_finite = true
- item_cf_score_finite = true
- similarity(i,i) = 1

### Phase C - CBF (TF-IDF)

Run:
python src/models/phase_c_cbf_run.py

Output du kien:
- reports/phase_c_cbf_scores.csv
- reports/phase_c_cbf_summary.json

Test phase C:
python -m unittest tests.test_cbf_branch

Tieu chi pass nhanh:
- rows_scored = so dong test
- cbf_score_finite = true
- item_vocab_size > 0

### Phase D - NCF

Run:
python src/models/phase_d_ncf_run.py --epochs 5 --batch-size 512

Output du kien:
- reports/phase_d_ncf_scores.csv
- reports/phase_d_ncf_summary.json

Test phase D:
python -m unittest tests.test_ncf_branch

Tieu chi pass nhanh:
- rows_scored = so dong test
- ncf_score_finite = true
- train duoc va predict duoc

## 6. Test tong the (hien tai)

Chay full unit test:
python -m unittest discover -s tests -p "test_*.py"

Muc tieu:
- tat ca test pass
- khong co loi import
- khong co NaN/Inf trong cac score branch hien tai

## 7. Quy trinh training day du (hien tai trong code)

Thu tu khuyen nghi:
1. python src/data/phase_a_data_pipeline.py
2. python src/models/phase_b_cf_run.py
3. python src/models/phase_c_cbf_run.py
4. python src/models/phase_d_ncf_run.py --epochs 5 --batch-size 512
5. python -m unittest discover -s tests -p "test_*.py"

Danh sach artifact can co sau khi chay:
- reports/phase_a_qa_report.json
- reports/phase_b_cf_scores.csv
- reports/phase_b_cf_summary.json
- reports/phase_c_cbf_scores.csv
- reports/phase_c_cbf_summary.json
- reports/phase_d_ncf_scores.csv
- reports/phase_d_ncf_summary.json

## 8. Quy trinh training full theo paper (khi hoan tat cac phase con lai)

Sau khi code xong Phase E/F/G, quy trinh full can bo sung:
1. Train va score RNN branch (Phase E)
2. Weighted fusion 5 thanh phan theo paper (Phase F)
3. Evaluation RMSE, MAE, Precision, Recall va cold-start (Phase G)
4. Lap bang doi chieu ket qua voi paper

Moc doi chieu paper:
- Sau tuning: RMSE 0.7723, MAE 0.6018, Precision 0.8127, Recall 0.7312
- Cold-start new users: Precision 0.762, Recall 0.685
- Cold-start new items: MAE 0.612, Precision 0.788, Recall 0.702

## 9. Troubleshooting nhanh

Loi ModuleNotFoundError: No module named src
- Dam bao dang chay lenh tu root project
- Dam bao venv dang active

TensorFlow khong nhan GPU
- Chay lai python src/models/check_gpu.py
- Dam bao tensorflow-cpu 2.10.0 + tensorflow-directml-plugin da cai
- Dam bao numpy < 2

Test fail do data chua co
- Chay lai Phase A truoc

## 10. Lenh nhanh copy-run

Setup:
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

GPU:
python src/models/check_gpu.py

Run theo phase:
python src/data/phase_a_data_pipeline.py
python src/models/phase_b_cf_run.py
python src/models/phase_c_cbf_run.py
python src/models/phase_d_ncf_run.py --epochs 5 --batch-size 512

Test:
python -m unittest tests.test_phase_a_data_pipeline
python -m unittest tests.test_cf_branch
python -m unittest tests.test_cbf_branch
python -m unittest tests.test_ncf_branch
python -m unittest discover -s tests -p "test_*.py"
