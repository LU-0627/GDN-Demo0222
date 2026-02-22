# GDN

Code implementation for : [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series(AAAI'21)](https://arxiv.org/pdf/2106.06947.pdf)


# Installation
### Requirements
* Python >= 3.6
* cuda == 10.2
* [Pytorch==1.5.1](https://pytorch.org/)
* [PyG: torch-geometric==1.5.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Install packages
```
    # run after installing correct Pytorch package
    bash install.sh
```

### Conda environment for TopoFuSAGNet (recommended)
```
    conda env create -f environment.topofusagnet.yml
    conda activate gdn-topofusagnet
```

Windows (PowerShell) one-command setup + sanity run:
```
    .\run_topofusagnet.ps1
```

Quick sanity run:
```
    python models/topofusagnet.py
```

### Quick Start
Run to check if the environment is ready
```
    bash run.sh cpu msl
    # or with gpu
    bash run.sh <gpu_id> msl    # e.g. bash run.sh 1 msl
```


# Usage
We use part of msl dataset(refer to [telemanom](https://github.com/khundman/telemanom)) as demo example. 

## Data Preparation
```
# put your dataset under data/ directory with the same structure shown in the data/msl/

data
 |-msl
 | |-list.txt    # the feature names, one feature per line
 | |-train.csv   # training data
 | |-test.csv    # test data
 |-your_dataset
 | |-list.txt
 | |-train.csv
 | |-test.csv
 | ...

```

### Notices:
* The first column in .csv will be regarded as index column. 
* The column sequence in .csv don't need to match the sequence in list.txt, we will rearrange the data columns according to the sequence in list.txt.
* test.csv should have a column named "attack" which contains ground truth label(0/1) of being attacked or not(0: normal, 1: attacked)

## Run
```
    # using gpu
    bash run.sh <gpu_id> <dataset>

    # or using cpu
    bash run.sh cpu <dataset>
```
You can change running parameters in the run.sh.

### Ablation Study: SAE Removal (use_sae parameter)

**Note**: The `-use_sae` parameter enables ablation study by control whether to use Sparse Autoencoder (SAE).

#### Default Configuration (use_sae=1, Full TopoFuSAGNet)
```bash
# Full model with SAE + Forecast + Reconstruction losses
python main.py -dataset msl -device cuda -epoch 30 -use_sae 1
```
- **Expected behavior**: Model trains with SAE enabled (default)
- **Loss components**: forecast + reconstruction + sparsity
- **Output**: All four loss terms (total, forecast, reconstruction, sparsity)

#### Ablation: Disable SAE (use_sae=0)
```bash
# Lightweight model without SAE: only Forecast loss
python main.py -dataset msl -device cuda -epoch 30 -use_sae 0
```
- **Expected behavior**: 
  - MSTCN outputs → Linear projection (z_dim) → Graph Learning + GAT → Forecast head
  - SAE module NOT initialized; reconstruction and sparsity losses NOT computed
  - Only forecast loss is optimized
- **Output**: Reconstruction and sparsity loss set to 0 in logs
- **Log format** (consistent across modes):
  ```
  [Train] total=0.123456 fore=0.123456 recon=0.000000 kl=0.000000
  [Val]   total=0.098765 fore=0.098765 recon=0.000000 kl=0.000000 (# recon & kl marked as 0)
  ```

#### Model Checkpoint Compatibility
- **use_sae=0**: Saves with `proj` layer (Linear projection) instead of `sae`
- **Checkpoint loading**: Uses `strict=False` to allow architecture mismatch
  - Can load use_sae=0 checkpoint for inference in either mode (missing keys ignored)
  - Recommended: Keep separate checkpoint directories for clean experiment tracking
  ```bash
  ./pretrained/topofusagnet_with_sae/     # use_sae=1 models
  ./pretrained/topofusagnet_no_sae/       # use_sae=0 models
  ```

#### A/B Testing Commands
```bash
# Training with SAE (baseline)
python main.py -dataset msl -device cuda -epoch 30 -use_sae 1 -save_path_pattern topofusagnet_with_sae

# Training without SAE (ablation)
python main.py -dataset msl -device cuda -epoch 30 -use_sae 0 -save_path_pattern topofusagnet_no_sae

# Test & evaluate (model loads automatically from trained checkpoint)
# Just re-run the same command to evaluate on test set
```

#### Key Parameter Details
- `-use_sae` (int, default=1): 
  - `0`: Disable SAE, use only linear projection
  - `1`: Enable SAE (standard TopoFuSAGNet)
- `-sae_score_type` (str, default=`recon`, choices: `recon`, `sparsity_dev`):
  - `recon`: use reconstruction error as SAE branch score (backward-compatible)
  - `sparsity_dev`: use latent activation deviation score, defined as mean over latent dim of `|sigmoid(z) - rho|` per sample and per node

#### Why `sparsity_dev` can be more robust
- Reconstruction error can be weak when SAE trends toward identity mapping.
- `sparsity_dev` directly measures whether latent activations violate target sparsity `rho`, so it is less dependent on pixel/value-level reconstruction fidelity.

#### Score-Type Comparison Commands
```bash
# (1) fusion + recon score (default)
python main.py -dataset msl -device cuda -epoch 30 -use_sae 1 -score_lambda 0.5 -sae_score_type recon

# (2) fusion + latent sparsity deviation score
python main.py -dataset msl -device cuda -epoch 30 -use_sae 1 -score_lambda 0.5 -sae_score_type sparsity_dev
```

#### SWaT A/B/C Auto Ablation Script
```bash
# Option 1: bash wrapper
bash ./run_ablation_swat.sh --device cuda --epoch 30 --c-score-lambda 0.8

# Option 2: python directly
python tools/run_ablation.py --dataset swat --device cuda --epoch 30 --c-score-lambda 0.8
```
- It auto-runs A/B/C settings, reuses checkpoints via `-load_model_path` when available, and writes summary to:
  - `results/ablation/swat_ablation_summary.csv`
  - `results/ablation/swat_ablation_summary.md`



# Others
SWaT and WADI datasets can be requested from [iTrust](https://itrust.sutd.edu.sg/)


# Citation
If you find this repo or our work useful for your research, please consider citing the paper
```
@inproceedings{deng2021graph,
  title={Graph neural network-based anomaly detection in multivariate time series},
  author={Deng, Ailin and Hooi, Bryan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4027--4035},
  year={2021}
}
```
