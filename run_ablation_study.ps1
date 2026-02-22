# Ablation Study: SAE Impact Comparison
# This PowerShell script demonstrates the A/B testing methodology for use_sae parameter

param(
    [string]$Dataset = "msl",
    [string]$Device = "cpu",
    [int]$Epochs = 5
)

Write-Host "==============================================================="
Write-Host "TopoFuSAGNet Ablation Study: SAE Impact (use_sae parameter)" -ForegroundColor Green
Write-Host "==============================================================="
Write-Host ""

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Dataset: $Dataset"
Write-Host "  Device: $Device"
Write-Host "  Epochs: $Epochs (use more for production, 5 is for quick test)"
Write-Host ""

# Create result directories
New-Item -ItemType Directory -Force -Path "./pretrained/topofusagnet_with_sae" | Out-Null
New-Item -ItemType Directory -Force -Path "./pretrained/topofusagnet_no_sae" | Out-Null
New-Item -ItemType Directory -Force -Path "./results/topofusagnet_with_sae" | Out-Null
New-Item -ItemType Directory -Force -Path "./results/topofusagnet_no_sae" | Out-Null

Write-Host "==============================================================="
Write-Host "[A] Training with SAE (use_sae=1) - Full Model" -ForegroundColor Cyan
Write-Host "==============================================================="
Write-Host ""
Write-Host "Command:"
Write-Host "python main.py -dataset $Dataset -device $Device -epoch $Epochs -use_sae 1 -save_path_pattern topofusagnet_with_sae" -ForegroundColor Gray
Write-Host ""
Write-Host "Expected behavior:" -ForegroundColor Yellow
Write-Host "  • SAE module initialized"
Write-Host "  • Loss includes: forecast + reconstruction + sparsity"
Write-Host "  • Slower training (~baseline)"
Write-Host "  • Higher memory usage"
Write-Host ""
Write-Host "Running (this may take a moment)..." -ForegroundColor Yellow
Write-Host ""

$startTime_A = Get-Date
python main.py `
    -dataset $Dataset `
    -device $Device `
    -epoch $Epochs `
    -use_sae 1 `
    -save_path_pattern topofusagnet_with_sae `
    -batch 32 `
    -log_interval 100

$endTime_A = Get-Date
$duration_A = ($endTime_A - $startTime_A).TotalSeconds

Write-Host ""
Write-Host "[A] Training completed in $($duration_A.ToString('F2')) seconds" -ForegroundColor Green
Write-Host ""

Write-Host "==============================================================="
Write-Host "[B] Training without SAE (use_sae=0) - Ablation Model" -ForegroundColor Cyan
Write-Host "==============================================================="
Write-Host ""
Write-Host "Command:"
Write-Host "python main.py -dataset $Dataset -device $Device -epoch $Epochs -use_sae 0 -save_path_pattern topofusagnet_no_sae" -ForegroundColor Gray
Write-Host ""
Write-Host "Expected behavior:" -ForegroundColor Yellow
Write-Host "  • Linear projection layer instead of SAE"
Write-Host "  • Loss only: forecast (recon & sparsity = 0)"
Write-Host "  • Faster training (~20-30% speedup)"
Write-Host "  • Lower memory usage (~15-25%)"
Write-Host ""
Write-Host "Running (this may take a moment)..." -ForegroundColor Yellow
Write-Host ""

$startTime_B = Get-Date
python main.py `
    -dataset $Dataset `
    -device $Device `
    -epoch $Epochs `
    -use_sae 0 `
    -save_path_pattern topofusagnet_no_sae `
    -batch 32 `
    -log_interval 100

$endTime_B = Get-Date
$duration_B = ($endTime_B - $startTime_B).TotalSeconds

Write-Host ""
Write-Host "[B] Training completed in $($duration_B.ToString('F2')) seconds" -ForegroundColor Green
Write-Host ""

# Calculate speedup
if ($duration_A -gt 0) {
    $speedup = $duration_A / $duration_B
    Write-Host "Performance Impact:" -ForegroundColor Yellow
    Write-Host "  Time A (with SAE):    $($duration_A.ToString('F2'))s"
    Write-Host "  Time B (no SAE):      $($duration_B.ToString('F2'))s"
    Write-Host "  Speedup:              $($speedup.ToString('F2'))x" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "==============================================================="
Write-Host "Ablation Study Complete!" -ForegroundColor Green
Write-Host "==============================================================="
Write-Host ""

Write-Host "Checkpoint Locations:" -ForegroundColor Yellow
Write-Host "  [A] with SAE:    ./pretrained/topofusagnet_with_sae/"
Write-Host "  [B] no SAE:      ./pretrained/topofusagnet_no_sae/"
Write-Host ""

Write-Host "Results Locations:" -ForegroundColor Yellow
Write-Host "  [A] with SAE:    ./results/topofusagnet_with_sae/"
Write-Host "  [B] no SAE:      ./results/topofusagnet_no_sae/"
Write-Host ""

Write-Host "Log Files (check for loss components):" -ForegroundColor Yellow
Get-ChildItem -Path "./logs/*.log" -ErrorAction SilentlyContinue | Sort-Object -Property LastWriteTime -Descending | Select-Object -First 2 | Format-Table -Property Name, LastWriteTime
Write-Host ""

Write-Host "Key Observations to Compare:" -ForegroundColor Yellow
Write-Host "  1. Training time (with_sae vs no_sae)"
Write-Host "  2. Loss curves (forecast vs total)"
Write-Host "  3. Anomaly detection F1 scores"
Write-Host "  4. Model size & inference speed"
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Compare logs: use_sae=1 vs use_sae=0"
Write-Host "  2. Analyze results: Compare Best-F1 scores in results/"
Write-Host "  3. Benchmark: Compare inference speed and memory usage"
Write-Host ""

Write-Host "==============================================================="
