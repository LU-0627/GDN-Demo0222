param(
    [string]$EnvName = "gdn-topofusagnet"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Checking conda command..."
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda not found in PATH. Please install Miniconda/Anaconda first."
}

Write-Host "[2/3] Creating (or updating) conda env: $EnvName"
conda env create -n $EnvName -f environment.topofusagnet.yml --force

Write-Host "[3/3] Running TopoFuSAGNet sanity check in env: $EnvName"
conda run -n $EnvName python models/topofusagnet.py
