# Ke hoach trien khai va testing tai lap paper HRS-IU-DL (strict)

Nguon doi chieu: A deep learning based hybrid recommendation model for internet users (Scientific Reports, 2024), DOI: 10.1038/s41598-024-79011-z.

Muc tieu tai lieu: lap ke hoach trien khai tung phan, testing day du, kiem tra doi chieu ket qua theo paper, khong them bot ky thuat ngoai paper.

## 1) Nguyen tac thuc hien strict

1. Chi trien khai cac thanh phan paper neu ro: CF (SVD + Item-based cosine), CBF (TF-IDF), NCF, RNN, weighted-sum fusion.
2. Chi dung setup du lieu paper neu ro: MovieLens 100k, split 80/20.
3. Chi danh gia metric paper neu ro: RMSE, MAE, Precision, Recall.
4. Bao cao ket qua cold-start theo 2 kich ban paper neu: new users, new items.
5. Moi muc paper khong cong bo chi tiet phai danh dau la "khong cong bo" thay vi tu dien.

## 2) Baseline thong tin phai khoa truoc khi code

1. Dataset: MovieLens 100k.
2. Quy mo dataset doi chieu:
   - Users: 943
   - Movies: 1682
   - Ratings: 100000
3. Ti le tach tap: train 80%, test 20%.
4. Ket qua doi chieu sau tuning:
   - RMSE = 0.7723
   - MAE = 0.6018
   - Precision = 0.8127
   - Recall = 0.7312
5. Ket qua doi chieu truoc tuning:
   - RMSE = 0.930
   - MAE = 0.730
   - Precision = 0.730
   - Recall = 0.645
6. Ket qua cold-start doi chieu:
   - New users: Precision = 0.762, Recall = 0.685
   - New items: MAE = 0.612, Precision = 0.788, Recall = 0.702

## 3) WBS trien khai tung phan

### Phase A - Data ingestion va validation

Muc tieu: co bo du lieu dung format, dung quy mo, dung split paper.

Cong viec:
1. Tai va nap MovieLens 100k.
2. Kiem tra so luong users/movies/ratings.
3. Lam sach missing/unknown theo mo ta paper.
4. Trich xuat thong tin phim cho CBF (title, genre, release date theo mo ta du lieu).
5. Tach train/test 80/20.

Test va tieu chi dat:
1. Data shape test:
   - Pass neu users=943, movies=1682, ratings=100000.
2. Missing handling test:
   - Pass neu khong con gia tri null o truong bat buoc dung cho train.
3. Split test:
   - Pass neu test chiem 20% tong so ban ghi (sai so toi da do lam tron <= 1 ban ghi).
4. Leakage test:
   - Pass neu tong train + test = tong ban ghi va khong trung dong du lieu giua 2 tap.

Deliverable:
1. File thong ke dataset truoc/sau cleaning.
2. File luu index train/test.

### Phase B - CF branch

Muc tieu: co 2 thanh phan CF theo paper.

Cong viec:
1. Trien khai User-based CF theo SVD.
2. Trien khai Item-based CF theo cosine similarity.
3. Tao API tra score cho cap (user, item).

Test va tieu chi dat:
1. SVD reconstruction sanity test:
   - Pass neu diem du doan tao ra la huu han (khong NaN/Inf) cho mau test.
2. Item-cosine matrix test:
   - Pass neu similarity(i,i)=1 voi item hop le.
3. Inference consistency test:
   - Pass neu cung input thi cung output (deterministic trong cung seed).

Deliverable:
1. Bang score CF tren tap test.

### Phase C - CBF branch (TF-IDF)

Muc tieu: tao score content-based theo TF-IDF nhu paper.

Cong viec:
1. Tao TF-IDF vector tu du lieu movie attributes paper de cap.
2. Xay ham tinh do tuong dong noi dung de sinh score CBF.

Test va tieu chi dat:
1. TF-IDF vocabulary test:
   - Pass neu so chieu vector > 0 va co matrix TF-IDF hop le.
2. Similarity test:
   - Pass neu do tuong dong nam trong mien [-1, 1] neu dung cosine.
3. Missing feature test:
   - Pass neu item thieu text duoc xu ly ma khong lam vo pipeline.

Deliverable:
1. Bang score CBF tren tap test.

### Phase D - NCF branch

Muc tieu: dung neural collaborative filtering paper mo ta.

Cong viec:
1. Trien khai embedding user va item.
2. Tinh z va output \hat r_ui theo cong thuc paper.

Test va tieu chi dat:
1. Forward-shape test:
   - Pass neu output co shape dung so mau input.
2. Numeric stability test:
   - Pass neu output khong NaN/Inf tren mini-batch.
3. Backprop test:
   - Pass neu gradient ton tai va cap nhat duoc qua it nhat 1 buoc train.

Deliverable:
1. Bang score NCF tren tap test.

### Phase E - RNN branch

Muc tieu: trien khai nhanh RNN theo cong thuc trang thai an paper neu.

Cong viec:
1. Xay du lieu chuoi tuong tac user theo thu tu thoi gian.
2. Trien khai RNN de xuat dac trung/score phuc vu du doan.

Test va tieu chi dat:
1. Sequence order test:
   - Pass neu sequence cua tung user duoc sap theo timestamp tang dan.
2. Forward pass test:
   - Pass neu output RNN huu han va dung shape.
3. Short-sequence test:
   - Pass neu user co it tuong tac van duoc xu ly khong loi.

Deliverable:
1. Bang score RNN tren tap test.

### Phase F - Hybrid fusion theo paper

Muc tieu: ghep dung cong thuc weighted-sum paper neu.

Cong viec:
1. Dong bo score tu 5 thanh phan: SVD, ItemBased, ContentBased, Neural, RNN.
2. Ap dung cong thuc:

$$
\hat r_{ui}=\alpha\cdot SVD_{ui}+\beta\cdot ItemBased_{ui}+\gamma\cdot ContentBased_{ui}+\delta\cdot Neural_{ui}+\epsilon\cdot RNN_{ui}
$$

3. Tuning trong so theo iterative refinement paper de cap.

Test va tieu chi dat:
1. Fusion integrity test:
   - Pass neu tat ca thanh phan deu dong gop vao score cuoi (khong bi null branch).
2. Weight validity test:
   - Pass neu tap trong so duoc ghi log day du cho tung lan chay.
3. End-to-end prediction test:
   - Pass neu co du doan hop le cho toan bo tap test.

Deliverable:
1. Bang du doan hybrid tren test.
2. Bang log trong so fusion moi lan tuning.

### Phase G - Evaluation va cold-start

Muc tieu: tinh metric dung dinh nghia paper va doi chieu ket qua.

Cong viec:
1. Tinh RMSE, MAE, Precision, Recall tren tap test tong.
2. Chay danh gia cold-start:
   - Kich ban new users.
   - Kich ban new items.
3. Lap bang ket qua truoc tuning va sau tuning.

Test va tieu chi dat:
1. Metric formula test:
   - Pass neu cong thuc RMSE/MAE/Precision/Recall trung dinh nghia paper.
2. Recompute test:
   - Pass neu metric tinh lai tu file prediction khop voi metric bao cao.
3. Cold-start split test:
   - Pass neu subset cold-start duoc tach rieng va bao cao rieng.

Deliverable:
1. Bao cao metric day du.
2. Bang doi chieu voi ket qua paper.

## 4) Ke hoach testing tong the

### 4.1 Muc tieu testing

1. Dung ve du lieu.
2. Dung ve cong thuc tung nhanh.
3. Dung ve cong thuc fusion.
4. Dung ve metric va bao cao.
5. Dung ve tai lap (reproducibility).

### 4.2 Cac lop test can co

1. Unit tests:
   - Data parser, split function, metric function, moi model branch.
2. Integration tests:
   - Pipeline tu data den prediction cho tung branch.
3. System test:
   - End-to-end train/eval hybrid.
4. Regression test:
   - Chay lai sau moi thay doi de dam bao metric khong sai huong bat thuong.

### 4.3 Test cases toi thieu can thuc hien

1. TC-DATA-001: Kiem tra quy mo dataset.
2. TC-DATA-002: Kiem tra split 80/20.
3. TC-CF-001: Kiem tra output SVD hop le.
4. TC-CF-002: Kiem tra cosine similarity hop le.
5. TC-CBF-001: Kiem tra TF-IDF tao thanh cong.
6. TC-NCF-001: Kiem tra embedding va forward pass.
7. TC-RNN-001: Kiem tra sequence dung thu tu.
8. TC-FUS-001: Kiem tra weighted fusion dung cong thuc.
9. TC-MET-001: Kiem tra RMSE/MAE.
10. TC-MET-002: Kiem tra Precision/Recall.
11. TC-CS-001: Kiem tra cold-start new users.
12. TC-CS-002: Kiem tra cold-start new items.
13. TC-REP-001: Kiem tra chay lai 2 lan cho ket qua cung xu huong.

### 4.4 Tieu chi pass/fail cap du an

Pass khi dong thoi dat:
1. Toan bo test case o muc 4.3 pass.
2. Pipeline chay xuyen suot khong loi.
3. Co bao cao metric day du cho test tong va cold-start.
4. Co bang doi chieu voi paper (truoc tuning, sau tuning).
5. Khong co thanh phan ngoai paper duoc dua vao implementation.

Fail neu:
1. Thieu bat ky metric paper yeu cau.
2. Cong thuc fusion khong dung weighted-sum paper.
3. Khong tach duoc cold-start subset.
4. Khong tai lap duoc ket qua sau khi chay lai quy trinh.

## 5) Quy trinh kiem tra doi chieu voi paper

1. Lap bang doi chieu theo cot:
   - Hang muc
   - Paper de cap
   - Implementation hien tai
   - Ket qua do duoc
   - Trang thai (Match/Not Match)
2. Doi chieu 6 nhom bat buoc:
   - Data va split
   - Thanh phan mo hinh
   - Cong thuc fusion
   - Metric
   - Ket qua chung
   - Cold-start
3. Moi muc Not Match phai co bien ban ly do va hanh dong khac phuc.

## 6) Quan tri rui ro va cach xu ly (khong them ky thuat)

1. Rui ro metric lech paper:
   - Xu ly: kiem tra lai split, cleaning, cong thuc metric, cong thuc fusion.
2. Rui ro loi leakage:
   - Xu ly: kiem tra giao nhau train/test va sequence tao mau.
3. Rui ro khong tai lap duoc:
   - Xu ly: co dinh seed, luu toan bo artifact va cau hinh moi lan chay.

## 7) Dinh nghia hoan thanh

Du an duoc xem la hoan thanh khi:
1. Da trien khai day du cac thanh phan paper neu.
2. Da chay du bo test va dat pass theo muc 4.4.
3. Da co bao cao doi chieu ket qua paper cho ca setting chung va cold-start.
4. Da co ho so tai lap day du (du lieu split, trong so fusion, metric logs, prediction files).

## 8) Luu y ve yeu cau "chinh xac 100%"

Trong nghien cuu thuc nghiem, khong the cam ket 100% trung khop so metric tuyet doi vi phu thuoc moi truong chay va chi tiet khong cong bo het trong paper.

Cach dam bao cao nhat co the:
1. Bam sat dung nhung gi paper cong bo.
2. Test day du theo ke hoach tren.
3. Luu vet day du de audit va doi chieu tung buoc.