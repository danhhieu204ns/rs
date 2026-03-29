# HRS-IU-DL Strict Replication Checklist

Tai lieu nay la checklist thuc thi chi tiet de tai lap paper theo dung pham vi da xac nhan.
Nguon doi chieu: Scientific Reports (2024), DOI: 10.1038/s41598-024-79011-z.

## 0) Rule Gate (bat buoc truoc khi bat dau)

- [x] Chi su dung cac thanh phan paper neu ro: CF (SVD + item cosine), CBF (TF-IDF), NCF, RNN, weighted-sum fusion.
- [x] Khong dua ky thuat ngoai paper vao code chinh.
- [x] Dung MovieLens 100k.
- [x] Dung split train/test 80/20.
- [x] Dung metric RMSE, MAE, Precision, Recall.
- [x] Co danh gia cold-start cho new users va new items.

## 1) Environment Checklist

- [x] Python environment da kich hoat.
- [x] Ghi lai thong tin version runtime.
- [x] Cac thu vien can cho pipeline da cai (theo danh sach paper).
- [x] Co file dependency lock hoac danh sach package final.

Acceptance:
- [x] Chay script sanity import toan bo package khong loi.

## 2) Data Checklist

### 2.1 Ingestion
- [x] Tai/nap day du MovieLens 100k.
- [x] Luu snapshot dataset raw.

### 2.2 Validation
- [x] Users = 943.
- [x] Movies = 1682.
- [x] Ratings = 100000.
- [x] Kiem tra cot bat buoc ton tai.

### 2.3 Cleaning + Features
- [x] Xu ly missing/unknown theo mo ta paper.
- [x] Trich xuat thong tin movie cho CBF (title/genre/release date theo du lieu).
- [x] Log thong ke truoc/sau cleaning.

### 2.4 Split
- [x] Split 80/20.
- [x] Kiem tra train + test = tong ban ghi.
- [x] Kiem tra khong trung ban ghi train/test.
- [x] Luu index split de tai lap.

Acceptance:
- [x] Bao cao data QA da day du va pass.

## 3) Model Branch Checklist

### 3.1 CF (SVD)
- [x] Trien khai user-based CF theo SVD.
- [x] Co ham du doan score cho (user, item).
- [x] Output khong NaN/Inf.

### 3.2 CF (Item-based Cosine)
- [x] Trien khai item-item cosine similarity.
- [x] Similarity(i,i) = 1 cho item hop le.
- [x] Co ham du doan score cho (user, item).

### 3.3 CBF (TF-IDF)
- [x] Tao ma tran TF-IDF.
- [x] Co ham score dua tren noi dung.
- [x] Xu ly item thieu text ma khong vo pipeline.

### 3.4 NCF
- [x] Trien khai embedding user/item.
- [x] Trien khai tinh z va output theo cong thuc paper.
- [x] Forward pass dung shape, output huu han.

### 3.5 RNN
- [x] Tao sequence tuong tac theo thu tu thoi gian.
- [x] Trien khai RNN branch theo mo ta paper.
- [x] Forward pass dung shape, output huu han.

Acceptance:
- [x] Moi branch deu sinh score tren tap test.

## 4) Fusion Checklist (core)

- [x] Dong bo score tu 5 thanh phan: SVD, ItemBased, ContentBased, Neural, RNN.
- [x] Ap dung dung cong thuc weighted sum:

  r_hat = a*SVD + b*ItemBased + c*ContentBased + d*Neural + e*RNN

- [x] Co log trong so moi lan chay.
- [x] Prediction cuoi cung tao duoc cho toan bo test set.

Acceptance:
- [x] Fusion integrity pass (khong branch nao bi bo qua).

## 5) Training + Tuning Checklist

- [x] Chay baseline truoc tuning.
- [x] Chay iterative refinement theo huong paper mo ta.
- [x] Luu log tung lan tuning.
- [x] Luu prediction file cho truoc tuning va sau tuning.

Acceptance:
- [x] Co day du artifact de tinh lai metric offline.

## 6) Evaluation Checklist

### 6.1 Main metrics
- [x] Tinh RMSE tren test.
- [x] Tinh MAE tren test.
- [x] Tinh Precision tren test.
- [x] Tinh Recall tren test.

### 6.2 Recompute integrity
- [x] Tinh lai metric tu prediction file va doi chieu voi report.
- [x] Sai lech metric giua 2 cach tinh = 0 (hoac trong sai so floating point rat nho).

### 6.3 Paper comparison
- [x] Doi chieu ket qua truoc tuning: RMSE 0.930, MAE 0.730, Precision 0.730, Recall 0.645.
- [x] Doi chieu ket qua sau tuning: RMSE 0.7723, MAE 0.6018, Precision 0.8127, Recall 0.7312.
- [x] Lap bang Match/Not Match cho tung metric.

Acceptance:
- [x] Co bang doi chieu metric day du va giai trinh neu lech.

## 7) Cold-start Checklist

### 7.1 New users
- [x] Tao subset cold-start new users theo paper.
- [x] Bao cao Precision, Recall.
- [x] Doi chieu paper: Precision 0.762, Recall 0.685.

### 7.2 New items
- [x] Tao subset cold-start new items theo paper.
- [x] Bao cao MAE, Precision, Recall.
- [x] Doi chieu paper: MAE 0.612, Precision 0.788, Recall 0.702.

Acceptance:
- [x] Co bao cao cold-start tach rieng 2 kich ban.

## 8) Test Suite Checklist

### 8.1 Unit tests
- [x] Data parser.
- [x] Split function.
- [x] Metric functions.
- [x] Moi branch model.

### 8.2 Integration tests
- [x] Data -> branch prediction.
- [x] Branch scores -> fusion -> prediction.

### 8.3 System test
- [x] End-to-end pipeline train/eval chay thanh cong.

### 8.4 Regression tests
- [x] Chay lai sau thay doi code va so sanh metric voi lan truoc.

Acceptance:
- [x] Toan bo test case pass.

## 9) Reproducibility Checklist

- [x] Co dinh seed.
- [x] Luu split index.
- [x] Luu trong so fusion.
- [x] Luu model artifact/checkpoint.
- [x] Luu prediction file.
- [x] Luu metric logs.
- [x] Luu environment manifest.

Acceptance:
- [x] Co the rerun va tai tao bao cao bang artifact da luu.

## 10) Final Sign-off Checklist

- [x] Khong co thanh phan nao vuot pham vi paper.
- [x] Day du ket qua main + cold-start.
- [x] Day du bang doi chieu Match/Not Match.
- [x] Day du test evidence.
- [x] Day du artifact tai lap.
- [x] Tai lieu ket luan da review va phe duyet.

## 11) Tracking Template (dien sau moi lan chay)

Run ID:
- [ ] Dien run ID

Thong tin run:
- [ ] Ngay gio
- [ ] Git commit hash
- [ ] Data snapshot id
- [ ] Split id

Ket qua:
- [ ] RMSE
- [ ] MAE
- [ ] Precision
- [ ] Recall

Cold-start:
- [ ] New users: Precision, Recall
- [ ] New items: MAE, Precision, Recall

Trang thai:
- [ ] Match paper
- [ ] Not match paper (co giai trinh)
