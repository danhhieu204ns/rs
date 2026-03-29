# Run all phases to generate full results for the hybrid recommendation model
# Ensure your virtual environment is active before running this script

$ErrorActionPreference = "Stop"
$PYTHON_EXEC = ".\.venv\Scripts\python.exe"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " RUNNING HRS-IU-DL PIPELINE" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# -------------------------------------------------------------------------
# PHASE A: Data Ingestion and Pipeline 
# -------------------------------------------------------------------------
Write-Host "`n[1/6] Phase A: Data Pipeline..." -ForegroundColor Yellow
if (Test-Path "data/processed/ratings_train.csv") {
    Write-Host " -> Skipped: Filtered data already exists in /data/processed/" -ForegroundColor DarkGray
} else {
    & $PYTHON_EXEC src/data_code/phase_a_data_pipeline.py
    if ($LASTEXITCODE -ne 0) { throw "Phase A Failed!" }
    Write-Host " -> Phase A Completed. Filtered data saved to /data/processed/" -ForegroundColor Green
}

# -------------------------------------------------------------------------
# PHASE B: Collaborative Filtering Branch (SVD & ItemBased)
# -------------------------------------------------------------------------
Write-Host "`n[2/6] Phase B: CF Branch..." -ForegroundColor Yellow
if (Test-Path "reports/phase_b_cf_scores.csv") {
    Write-Host " -> Skipped: Reports already exist in /reports/" -ForegroundColor DarkGray
} else {
    & $PYTHON_EXEC src/models/phase_b_cf_run.py
    if ($LASTEXITCODE -ne 0) { throw "Phase B Failed!" }
    Write-Host " -> Phase B Completed. Reports saved to /reports/" -ForegroundColor Green
}

# -------------------------------------------------------------------------
# PHASE C: Content-Based Filtering Branch (TF-IDF)
# -------------------------------------------------------------------------
Write-Host "`n[3/6] Phase C: CBF Branch..." -ForegroundColor Yellow
if (Test-Path "reports/phase_c_cbf_scores.csv") {
    Write-Host " -> Skipped: Reports already exist in /reports/" -ForegroundColor DarkGray
} else {
    & $PYTHON_EXEC src/models/phase_c_cbf_run.py
    if ($LASTEXITCODE -ne 0) { throw "Phase C Failed!" }
    Write-Host " -> Phase C Completed. Reports saved to /reports/" -ForegroundColor Green
}

# -------------------------------------------------------------------------
# PHASE D: Neural Collaborative Filtering Branch 
# -------------------------------------------------------------------------
Write-Host "`n[4/6] Phase D: NCF Branch..." -ForegroundColor Yellow
if (Test-Path "reports/phase_d_ncf_scores.csv") {
    Write-Host " -> Skipped: Reports already exist in /reports/" -ForegroundColor DarkGray
} else {
    & $PYTHON_EXEC src/models/phase_d_ncf_run.py
    if ($LASTEXITCODE -ne 0) { throw "Phase D Failed!" }
    Write-Host " -> Phase D Completed. Reports saved to /reports/" -ForegroundColor Green
}

# -------------------------------------------------------------------------
# PHASE E: Recurrent Neural Network Branch
# -------------------------------------------------------------------------
Write-Host "`n[5/6] Phase E: RNN Branch..." -ForegroundColor Yellow
if (Test-Path "reports/phase_e_rnn_scores.csv") {
    Write-Host " -> Skipped: Reports already exist in /reports/" -ForegroundColor DarkGray
} else {
    & $PYTHON_EXEC src/models/phase_e_rnn_run.py --batch-size 128
    if ($LASTEXITCODE -ne 0) { throw "Phase E Failed!" }
    Write-Host " -> Phase E Completed. Reports saved to /reports/" -ForegroundColor Green
}

# -------------------------------------------------------------------------
# PHASE F: Fusion & Evaluation
# -------------------------------------------------------------------------
Write-Host "`n[6/6] Phase F: Fusion Model (Baseline Params)..." -ForegroundColor Yellow
# Phase F is usually fast and we might want to re-run it with different params, 
# but per your request, we can skip if it exists. Remove the IF condition here if you want F to always run.
if (Test-Path "reports/phase_f_fusion_results.csv") {
    Write-Host " -> Skipped: Final reports already exist in /reports/" -ForegroundColor DarkGray
} else {
    & $PYTHON_EXEC src/models/phase_f_fusion_run.py --alpha 0.2 --beta 0.2 --gamma 0.2 --delta 0.2 --epsilon 0.2
    if ($LASTEXITCODE -ne 0) { throw "Phase F Failed!" }
    Write-Host " -> Phase F Completed. Final reports available in /reports/" -ForegroundColor Green
}

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host " Pipeline Execution Finished Successfully!" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
