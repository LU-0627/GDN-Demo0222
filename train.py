import time
from typing import Dict, Tuple, Union

import torch


LossOutput = Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


def _parse_joint_loss_output(loss_out: LossOutput):
    """
    兼容两种 JointLoss 返回格式：
    1) dict: {"total", "forecast", "reconstruction", "sparsity"}
    2) tuple/list: (total_loss, loss_fore, loss_recon, loss_kl)
    """
    if isinstance(loss_out, dict):
        return (
            loss_out["total"],
            loss_out["forecast"],
            loss_out["reconstruction"],
            loss_out["sparsity"],
        )

    if isinstance(loss_out, (tuple, list)) and len(loss_out) == 4:
        return loss_out[0], loss_out[1], loss_out[2], loss_out[3]

    raise TypeError(
        "JointLoss 的返回格式不符合预期，必须是 dict 或长度为 4 的 tuple/list。"
    )


def _get_recon_target(x: torch.Tensor, mode: str = "input") -> torch.Tensor:
    """
    重建目标切换入口：
    - mode='input'：使用原始输入窗口 x 作为重建目标（默认）
    - 后续如果你要切换为卷积特征，可在这里统一替换
    """
    if mode == "input":
        return x
    raise ValueError(f"Unsupported recon_target mode: {mode}")


def validate_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device,
    recon_target_mode: str = "input",
):
    model.eval()

    total_meter = 0.0
    fore_meter = 0.0
    recon_meter = 0.0
    kl_meter = 0.0
    steps = 0

    with torch.no_grad():
        for x, y, _, _ in dataloader:
            x = x.float().to(device)
            y = y.float().to(device)

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

    if steps == 0:
        return {"total": 0.0, "forecast": 0.0, "reconstruction": 0.0, "sparsity": 0.0}

    return {
        "total": total_meter / steps,
        "forecast": fore_meter / steps,
        "reconstruction": recon_meter / steps,
        "sparsity": kl_meter / steps,
    }


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    train_dataloader,
    val_dataloader,
    device,
    epochs: int,
    save_path: str,
    grad_clip_norm: float = 5.0,
    recon_target_mode: str = "input",
    log_interval: int = 100,
):
    """
    训练主循环（按你的新要求实现）：
    1) 解析 model(x) -> predicted_vals, reconstructed_vals, extra
    2) JointLoss 同时接收预测/重建/KL 三部分
    3) backward 后 step 前执行梯度裁剪 clip_grad_norm_(..., max_norm=5.0)
    4) total/forecast/reconstruction/kl 四项 loss 独立记录与打印
    """
    history = {
        "train": [],
        "val": [],
    }

    best_val_total = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()

        total_meter = 0.0
        fore_meter = 0.0
        recon_meter = 0.0
        kl_meter = 0.0
        steps = 0

        for step, (x, y, _, _) in enumerate(train_dataloader, start=1):
            x = x.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()

            predicted_vals, reconstructed_vals, extra = model(x)

            # ===== 关键注释：重建目标切换入口 =====
            # 默认用原始输入窗口 x 作为 recon_target。
            # 若你后续想改成“卷积后的特征”作为重建目标，只需要改 _get_recon_target。
            recon_target = _get_recon_target(x, mode=recon_target_mode)

            # ===== 关键注释：JointLoss 三路信号 =====
            # forecast: predicted_vals vs y
            # reconstruction: reconstructed_vals vs recon_target
            # sparsity: extra['kl_sparsity']
            loss_out = criterion(
                predicted_vals=predicted_vals,
                forecast_target=y,
                reconstructed_vals=reconstructed_vals,
                recon_target=recon_target,
                kl_sparsity=extra["kl_sparsity"],
            )

            total_loss, loss_fore, loss_recon, loss_kl = _parse_joint_loss_output(loss_out)

            total_loss.backward()

            # ===== 关键注释：防爆机制（梯度裁剪） =====
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            total_meter += float(total_loss.item())
            fore_meter += float(loss_fore.item())
            recon_meter += float(loss_recon.item())
            kl_meter += float(loss_kl.item())
            steps += 1

            if step % log_interval == 0:
                print(
                    f"[Train][Epoch {epoch}/{epochs}][Step {step}] "
                    f"total={total_meter/steps:.6f} "
                    f"fore={fore_meter/steps:.6f} "
                    f"recon={recon_meter/steps:.6f} "
                    f"kl={kl_meter/steps:.6f}",
                    flush=True,
                )

        train_epoch_stats = {
            "total": total_meter / max(steps, 1),
            "forecast": fore_meter / max(steps, 1),
            "reconstruction": recon_meter / max(steps, 1),
            "sparsity": kl_meter / max(steps, 1),
        }

        if val_dataloader is not None:
            val_epoch_stats = validate_epoch(
                model=model,
                dataloader=val_dataloader,
                criterion=criterion,
                device=device,
                recon_target_mode=recon_target_mode,
            )

            if val_epoch_stats["total"] < best_val_total:
                best_val_total = val_epoch_stats["total"]
                torch.save(model.state_dict(), save_path)

            print(
                f"[Epoch {epoch}/{epochs}] "
                f"Train(total={train_epoch_stats['total']:.6f}, fore={train_epoch_stats['forecast']:.6f}, "
                f"recon={train_epoch_stats['reconstruction']:.6f}, kl={train_epoch_stats['sparsity']:.6f}) | "
                f"Val(total={val_epoch_stats['total']:.6f}, fore={val_epoch_stats['forecast']:.6f}, "
                f"recon={val_epoch_stats['reconstruction']:.6f}, kl={val_epoch_stats['sparsity']:.6f})",
                flush=True,
            )

            history["val"].append(val_epoch_stats)
        else:
            if train_epoch_stats["total"] < best_val_total:
                best_val_total = train_epoch_stats["total"]
                torch.save(model.state_dict(), save_path)

            print(
                f"[Epoch {epoch}/{epochs}] "
                f"Train(total={train_epoch_stats['total']:.6f}, fore={train_epoch_stats['forecast']:.6f}, "
                f"recon={train_epoch_stats['reconstruction']:.6f}, kl={train_epoch_stats['sparsity']:.6f})",
                flush=True,
            )

        history["train"].append(train_epoch_stats)

    print(f"Training finished in {(time.time() - start_time):.2f}s", flush=True)
    return history
