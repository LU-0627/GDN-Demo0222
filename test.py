from typing import Dict, Union, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


LossOutput = Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


def _parse_joint_loss_output(loss_out: LossOutput):
    if isinstance(loss_out, dict):
        return (
            loss_out["total"],
            loss_out["forecast"],
            loss_out["reconstruction"],
            loss_out["sparsity"],
        )

    if isinstance(loss_out, (tuple, list)) and len(loss_out) == 4:
        return loss_out[0], loss_out[1], loss_out[2], loss_out[3]

    raise TypeError("JointLoss 输出必须是 dict 或长度为 4 的 tuple/list")


def _get_recon_target(x: torch.Tensor, mode: str = "input") -> torch.Tensor:
    if mode == "input":
        return x
    raise ValueError(f"Unsupported recon_target mode: {mode}")


def weighted_harmonic_mean(fore_err: np.ndarray, recon_err: np.ndarray, score_lambda: float = 0.5, eps: float = 1e-8):
    """
    加权调和平均（WHM）融合：
        score = 1 / (lambda/(fore_err+eps) + (1-lambda)/(recon_err+eps))

    注意：
    - fore_err、recon_err 均为每个样本的一维误差向量 [num_samples]
    - eps 用于数值稳定，防止除零
    """
    if not (0.0 < score_lambda < 1.0):
        raise ValueError(f"score_lambda must be in (0,1), got {score_lambda}")

    denom = score_lambda / (fore_err + eps) + (1.0 - score_lambda) / (recon_err + eps)
    return 1.0 / (denom + eps)


def evaluate_loop(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device,
    score_lambda: float = 0.5,
    recon_target_mode: str = "input",
):
    """
    统一的验证/测试循环：
    - 计算四项 loss（total / forecast / recon / kl）
    - 计算每个样本的 forecasting error 与 reconstruction error
    - 使用 WHM 融合得到最终异常分数

    返回:
    {
      "loss": {"total", "forecast", "reconstruction", "sparsity"},
      "forecast_error": np.ndarray [num_samples],
      "recon_error": np.ndarray [num_samples],
      "anomaly_score": np.ndarray [num_samples],
      "labels": np.ndarray [num_samples]
    }
    """
    model.eval()

    total_meter = 0.0
    fore_meter = 0.0
    recon_meter = 0.0
    kl_meter = 0.0
    steps = 0

    all_fore_err = []
    all_recon_err = []
    all_labels = []

    with torch.no_grad():
        for x, y, labels, _ in dataloader:
            x = x.float().to(device)
            y = y.float().to(device)
            labels = labels.float().to(device)

            predicted_vals, reconstructed_vals, extra = model(x)
            recon_target = _get_recon_target(x, mode=recon_target_mode)

            loss_out = criterion(
                predicted_vals=predicted_vals,
                forecast_target=y,
                reconstructed_vals=reconstructed_vals,
                recon_target=recon_target,
                kl_sparsity=extra["kl_sparsity"],
            )
            total_loss, loss_fore, loss_recon, loss_kl = _parse_joint_loss_output(loss_out)

            total_meter += float(total_loss.item())
            fore_meter += float(loss_fore.item())
            recon_meter += float(loss_recon.item())
            kl_meter += float(loss_kl.item())
            steps += 1

            # ===== 关键注释：误差定义（按样本聚合） =====
            # forecasting error: |y - y_hat| 在节点维 N 上取均值 -> [B]
            # reconstruction error: |x - x_recon| 在 [N, W] 上取均值 -> [B]
            fore_err_batch = torch.mean(torch.abs(predicted_vals - y), dim=1)
            recon_err_batch = torch.mean(torch.abs(reconstructed_vals - recon_target), dim=(1, 2))

            all_fore_err.append(fore_err_batch.detach().cpu().numpy())
            all_recon_err.append(recon_err_batch.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    if steps == 0:
        return {
            "loss": {"total": 0.0, "forecast": 0.0, "reconstruction": 0.0, "sparsity": 0.0},
            "forecast_error": np.array([]),
            "recon_error": np.array([]),
            "anomaly_score": np.array([]),
            "labels": np.array([]),
        }

    forecast_error = np.concatenate(all_fore_err, axis=0)
    recon_error = np.concatenate(all_recon_err, axis=0)
    labels_np = np.concatenate(all_labels, axis=0).astype(int)

    # ===== 关键注释：异常分数融合（WHM） =====
    anomaly_score = weighted_harmonic_mean(
        fore_err=forecast_error,
        recon_err=recon_error,
        score_lambda=score_lambda,
    )

    return {
        "loss": {
            "total": total_meter / steps,
            "forecast": fore_meter / steps,
            "reconstruction": recon_meter / steps,
            "sparsity": kl_meter / steps,
        },
        "forecast_error": forecast_error,
        "recon_error": recon_error,
        "anomaly_score": anomaly_score,
        "labels": labels_np,
    }


def get_threshold_from_validation(val_scores: np.ndarray) -> float:
    """
    按你的要求：阈值 = 验证集所有样本异常分数的最大值。
    """
    if val_scores.size == 0:
        raise ValueError("Validation scores are empty, cannot compute threshold.")
    return float(np.max(val_scores))


def evaluate_with_threshold(test_scores: np.ndarray, test_labels: np.ndarray, threshold: float):
    """
    在测试集上进行阈值判定并返回 Precision/Recall/F1。
    """
    pred_labels = (test_scores > threshold).astype(int)

    precision = precision_score(test_labels, pred_labels, zero_division=0)
    recall = recall_score(test_labels, pred_labels, zero_division=0)
    f1 = f1_score(test_labels, pred_labels, zero_division=0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_labels": pred_labels,
    }
