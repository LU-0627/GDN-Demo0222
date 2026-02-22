import numpy as np
import torch
from scipy.stats import iqr
from sklearn.metrics import precision_score, recall_score, f1_score

def _parse_joint_loss_output(loss_out):
    if isinstance(loss_out, dict):
        return loss_out["total"], loss_out["forecast"], loss_out["reconstruction"], loss_out["sparsity"]
    return loss_out[0], loss_out[1], loss_out[2], loss_out[3]

def _get_recon_target(x: torch.Tensor, mode: str = "input") -> torch.Tensor:
    if mode == "input":
        return x
    raise ValueError(f"Unsupported recon_target mode: {mode}")

def weighted_harmonic_mean(fore_err: np.ndarray, recon_err: np.ndarray, score_lambda: float = 0.5):
    """
    加权调和平均。因为传入的 error 是标准化后的，可能有负数（低于中位数的正常波动）。
    我们用 np.maximum 强行截断底线，防止公式出现负数或除零崩溃。
    """
    eps = 1e-4
    fore_err = np.maximum(fore_err, eps)
    recon_err = np.maximum(recon_err, eps)
    denom = score_lambda / fore_err + (1.0 - score_lambda) / recon_err
    return 1.0 / denom

def get_raw_errors(model, dataloader, criterion, device, recon_target_mode="input"):
    """
    【修正核心】：绝不跨节点做 Mean。保留 [Num_samples, Num_nodes] 形状。
    当 use_sae=0 时，reconstructed_vals 为 None，recon_err 应为 0。
    """
    model.eval()
    total_meter, fore_meter, recon_meter, kl_meter, steps = 0.0, 0.0, 0.0, 0.0, 0
    all_fore_err, all_recon_err, all_labels = [], [], []

    with torch.no_grad():
        for x, y, labels, _ in dataloader:
            x, y, labels = x.float().to(device), y.float().to(device), labels.float().to(device)
            predicted_vals, reconstructed_vals, extra = model(x)
            recon_target = _get_recon_target(x, mode=recon_target_mode)

            loss_out = criterion(predicted_vals, y, reconstructed_vals, recon_target, extra["kl_sparsity"])
            total_loss, loss_fore, loss_recon, loss_kl = _parse_joint_loss_output(loss_out)

            total_meter += float(total_loss.item())
            fore_meter += float(loss_fore.item())
            recon_meter += float(loss_recon.item())
            kl_meter += float(loss_kl.item())
            steps += 1

            # 【修复点1】：保留 [Batch, Nodes] 维度
            fore_err_batch = torch.abs(predicted_vals - y)
            
            # 重建误差：当 reconstructed_vals 为 None（use_sae=0）时，置为 0
            if reconstructed_vals is not None:
                recon_err_batch = torch.mean(torch.abs(reconstructed_vals - recon_target), dim=2)
            else:
                recon_err_batch = torch.zeros_like(fore_err_batch)

            all_fore_err.append(fore_err_batch.cpu().numpy())
            all_recon_err.append(recon_err_batch.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return {
        "loss": {
            "total": total_meter / max(1, steps), "forecast": fore_meter / max(1, steps),
            "reconstruction": recon_meter / max(1, steps), "sparsity": kl_meter / max(1, steps),
        },
        "fore_err": np.concatenate(all_fore_err, axis=0),   # Shape: [N_samples, N_nodes]
        "recon_err": np.concatenate(all_recon_err, axis=0), # Shape: [N_samples, N_nodes]
        "labels": np.concatenate(all_labels, axis=0).astype(int)
    }

def get_val_stats(val_err_matrix):
    """【修复点2】：计算每个传感器的 Median 和 IQR (鲁棒量纲统一)"""
    median = np.median(val_err_matrix, axis=0) # [N_nodes]
    iqr_val = iqr(val_err_matrix, axis=0)      # [N_nodes]
    
    # 【致命坑修复】：因为 main.py 已经改用 MinMaxScaler 将数据压到了 [0, 1] 之间
    # 最大的误差也不过是 1.0 左右。所以这里的 IQR 托底值必须降到 0.01。
    # 如果还用 0.1，会把很多真实的微小异常直接抹平；如果用 1e-4，又会导致 0 方差传感器分数爆炸。
    iqr_val = np.maximum(iqr_val, 0.01) 
    
    return median, iqr_val

def normalize_and_score(err_matrix, median, iqr_val):
    return (err_matrix - median) / iqr_val

def evaluate_with_threshold(test_scores, test_labels, threshold):
    pred_labels = (test_scores > threshold).astype(int)
    return {
        "precision": float(precision_score(test_labels, pred_labels, zero_division=0)),
        "recall": float(recall_score(test_labels, pred_labels, zero_division=0)),
        "f1": float(f1_score(test_labels, pred_labels, zero_division=0)),
    }