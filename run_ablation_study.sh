#!/bin/bash
# Ablation Study: SAE Impact Comparison
# This script demonstrates the A/B testing methodology for use_sae parameter

echo "==============================================================="
echo "TopoFuSAGNet Ablation Study: SAE Impact (use_sae parameter)"
echo "==============================================================="
echo ""

DATASET=${1:-msl}
DEVICE=${2:-cpu}
EPOCHS=${3:-5}  # Use 5 epochs for quick test

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Device: $DEVICE"
echo "  Epochs: $EPOCHS"
echo ""

# Create result directories
mkdir -p ./pretrained/topofusagnet_with_sae
mkdir -p ./pretrained/topofusagnet_no_sae
mkdir -p ./results/topofusagnet_with_sae
mkdir -p ./results/topofusagnet_no_sae

echo "==============================================================="
echo "[A] Training with SAE (use_sae=1) - Full Model"
echo "==============================================================="
echo ""
echo "Command:"
echo "python main.py -dataset $DATASET -device $DEVICE -epoch $EPOCHS -use_sae 1 -save_path_pattern topofusagnet_with_sae"
echo ""
echo "Expected behavior:"
echo "  - SAE module initialized"
echo "  - Loss includes: forecast + reconstruction + sparsity"
echo "  - Slower training (~baseline)"
echo "  - Higher memory usage"
echo ""
echo "Running (this may take a moment)..."
echo ""

python main.py \
    -dataset $DATASET \
    -device $DEVICE \
    -epoch $EPOCHS \
    -use_sae 1 \
    -save_path_pattern topofusagnet_with_sae \
    -batch 32 \
    -log_interval 100

echo ""
echo "==============================================================="
echo "[B] Training without SAE (use_sae=0) - Ablation Model"
echo "==============================================================="
echo ""
echo "Command:"
echo "python main.py -dataset $DATASET -device $DEVICE -epoch $EPOCHS -use_sae 0 -save_path_pattern topofusagnet_no_sae"
echo ""
echo "Expected behavior:"
echo "  - Linear projection layer instead of SAE"
echo "  - Loss only: forecast (recon & sparsity = 0)"
echo "  - Faster training (~20-30% speedup)"
echo "  - Lower memory usage (~15-25%)"
echo ""
echo "Running (this may take a moment)..."
echo ""

python main.py \
    -dataset $DATASET \
    -device $DEVICE \
    -epoch $EPOCHS \
    -use_sae 0 \
    -save_path_pattern topofusagnet_no_sae \
    -batch 32 \
    -log_interval 100

echo ""
echo "==============================================================="
echo "Ablation Study Complete!"
echo "==============================================================="
echo ""
echo "Checkpoint Locations:"
echo "  [A] with SAE:    ./pretrained/topofusagnet_with_sae/*.pt"
echo "  [B] no SAE:      ./pretrained/topofusagnet_no_sae/*.pt"
echo ""
echo "Results Locations:"
echo "  [A] with SAE:    ./results/topofusagnet_with_sae/*.csv"
echo "  [B] no SAE:      ./results/topofusagnet_no_sae/*.csv"
echo ""
echo "Log Files (check for loss components):"
ls -la ./logs/*.log 2>/dev/null | tail -2
echo ""
echo "Key Observations to Compare:"
echo "  1. Training time (with_sae vs no_sae)"
echo "  2. Loss curves (forecast vs total)"
echo "  3. Anomaly detection F1 scores"
echo "  4. Model size & inference speed"
echo ""
echo "==============================================================="
