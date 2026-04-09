param(
    [switch]$Force,
    [ValidateSet("A", "B", "C", "D", "E", "F", "G", "H")]
    [string]$FromPhase = "A",
    [int]$NcfEpochs = 10,
    [int]$RnnEpochs = 10,
    [int]$BatchSize = 512,
    [double]$FusionHoldoutRatio = 0.2,
    [double]$FusionGridStep = 0.1,
    [int]$FusionRandomTrials = 3000
)

$ErrorActionPreference = "Stop"
$PYTHON_EXEC = ".\.venv\Scripts\python.exe"
$PHASE_ORDER = @{ A = 1; B = 2; C = 3; D = 4; E = 5; F = 6; G = 7; H = 8 }

function Should-RunPhase {
    param([string]$Phase)
    return $PHASE_ORDER[$Phase] -ge $PHASE_ORDER[$FromPhase]
}

function Test-AllOutputsExist {
    param([string[]]$Paths)
    foreach ($p in $Paths) {
        if (-not (Test-Path $p)) {
            return $false
        }
    }
    return $true
}

function Invoke-Phase {
    param(
        [string]$Phase,
        [string]$Title,
        [string[]]$Outputs,
        [scriptblock]$Runner
    )

    Write-Host "`n[$Phase] $Title" -ForegroundColor Yellow

    if (-not (Should-RunPhase -Phase $Phase)) {
        Write-Host " -> Skipped: not selected by -FromPhase $FromPhase" -ForegroundColor DarkGray
        return
    }

    if ((Test-AllOutputsExist -Paths $Outputs) -and (-not $Force)) {
        Write-Host " -> Skipped: outputs already exist (use -Force to rerun)." -ForegroundColor DarkGray
        return
    }

    & $Runner
    if ($LASTEXITCODE -ne 0) {
        throw "Phase $Phase failed: $Title"
    }

    Write-Host " -> Completed" -ForegroundColor Green
}

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " RUNNING HRS-IU-DL PIPELINE" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " FromPhase=$FromPhase | Force=$Force | NcfEpochs=$NcfEpochs | RnnEpochs=$RnnEpochs | BatchSize=$BatchSize" -ForegroundColor Cyan

Invoke-Phase -Phase "A" -Title "Phase A: Data Pipeline" `
    -Outputs @("data/processed/ratings_train.csv", "data/processed/ratings_test.csv", "reports/phase_a_qa_report.json") `
    -Runner { & $PYTHON_EXEC src/data_code/phase_a_data_pipeline.py }

Invoke-Phase -Phase "B" -Title "Phase B: CF Branch" `
    -Outputs @("reports/phase_b_cf_scores.csv", "reports/phase_b_cf_summary.json") `
    -Runner { & $PYTHON_EXEC src/models/phase_b_cf_run.py }

Invoke-Phase -Phase "C" -Title "Phase C: CBF Branch" `
    -Outputs @("reports/phase_c_cbf_scores.csv", "reports/phase_c_cbf_summary.json") `
    -Runner { & $PYTHON_EXEC src/models/phase_c_cbf_run.py }

Invoke-Phase -Phase "D" -Title "Phase D: NCF Branch" `
    -Outputs @("reports/phase_d_ncf_scores.csv", "reports/phase_d_ncf_summary.json") `
    -Runner { & $PYTHON_EXEC src/models/phase_d_ncf_run.py --epochs $NcfEpochs --batch-size $BatchSize }

Invoke-Phase -Phase "E" -Title "Phase E: RNN Branch" `
    -Outputs @("reports/phase_e_rnn_scores.csv", "reports/phase_e_rnn_summary.json") `
    -Runner { & $PYTHON_EXEC src/models/phase_e_rnn_run.py --epochs $RnnEpochs --batch-size $BatchSize }

Invoke-Phase -Phase "F" -Title "Phase F: Fusion Baseline" `
    -Outputs @("reports/phase_f_fusion_results.csv", "reports/phase_f_fusion_summary.json") `
    -Runner { & $PYTHON_EXEC src/models/phase_f_fusion_run.py --alpha 0.2 --beta 0.2 --gamma 0.2 --delta 0.2 --epsilon 0.2 }

Invoke-Phase -Phase "G" -Title "Phase G: Eval and Tuning" `
    -Outputs @("reports/phase_g_eval_and_tuning.json", "reports/phase_g_tuned_predictions.csv") `
    -Runner {
        & $PYTHON_EXEC src/models/phase_g_eval_and_tuning.py `
            --holdout-ratio $FusionHoldoutRatio `
            --grid-step $FusionGridStep `
            --random-trials $FusionRandomTrials `
            --objective rmse
    }

Invoke-Phase -Phase "H" -Title "Phase H: Model Comparison Table" `
    -Outputs @("reports/phase_h_model_comparison.csv", "reports/phase_h_model_comparison.md", "reports/phase_h_model_comparison.json") `
    -Runner { & $PYTHON_EXEC src/models/phase_h_model_comparison.py }

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host " Pipeline execution finished successfully" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
