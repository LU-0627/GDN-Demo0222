param(
    [string]$GpuId = "0",
    [string]$Dataset = "msl"
)

$ErrorActionPreference = "Stop"

Write-Host "使用 Tesla T4 (GPU $GpuId) 运行数据集: $Dataset" -ForegroundColor Green

# 设置环境变量指定 GPU
$env:CUDA_VISIBLE_DEVICES = $GpuId

# 运行项目
python main.py `
    -device cuda `
    -dataset $Dataset `
    -save_path_pattern "topofusagnet" `
    -slide_win 15 `
    -batch 64 `
    -epoch 30 `
    -val_ratio 0.2 `
    -random_seed 5
