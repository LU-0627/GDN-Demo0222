# -*- coding: utf-8 -*-
import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from datasets.TimeDataset import TimeDataset
from models.topofusagnet import TopoFuSAGNet, JointLoss
from test import get_raw_errors, get_val_stats, normalize_and_score, weighted_harmonic_mean, evaluate_with_threshold, log_sparsity_fore_stats
from train import train
from util.env import get_device, set_device
from util.logger import setup_logger
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.preprocess import build_loc_net, construct_data


class Main:
    def __init__(self, train_config, env_config, logger=None, timezone=None):
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None
        self.logger = logger
        self.timezone = timezone

        dataset = self.env_config["dataset"]
        train_orig = pd.read_csv(f"./data/{dataset}/train.csv", sep=",", index_col=0)
        test_orig = pd.read_csv(f"./data/{dataset}/test.csv", sep=",", index_col=0)

        train_df, test_df = train_orig.copy(), test_orig.copy()

        if "attack" in train_df.columns:
            train_df = train_df.drop(columns=["attack"])

        if "attack" not in test_df.columns:
            raise ValueError("test.csv 必须包含 attack 列作为测试标签")
        
        # ================== 【修改为论文同款的 MinMax 归一化】 ==================
        from sklearn.preprocessing import MinMaxScaler
        
        test_labels = test_df["attack"].tolist()
        test_df = test_df.drop(columns=["attack"])

        # 强制限定在 [0, 1] 范围
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        train_scaled = scaler.fit_transform(train_df)
        test_scaled = scaler.transform(test_df)

        train_df = pd.DataFrame(train_scaled, columns=train_df.columns)
        test_df = pd.DataFrame(test_scaled, columns=test_df.columns)
        test_df["attack"] = test_labels
        # =========================================================================

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config["device"])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train_df.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        train_dataset_indata = construct_data(train_df, feature_map, labels=0)
        test_dataset_indata = construct_data(test_df, feature_map, labels=test_df.attack.tolist())

        cfg = {
            "slide_win": train_config["slide_win"],
            "slide_stride": train_config["slide_stride"],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode="train", config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode="test", config=cfg)

        train_dataloader, val_dataloader = self.get_loaders(
            train_dataset,
            seed=train_config["seed"],
            batch=train_config["batch"],
            val_ratio=train_config["val_ratio"],
        )

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config["batch"], shuffle=False, num_workers=0)

        num_nodes = len(feature_map)
        window_size = train_config["slide_win"]

        self.model = TopoFuSAGNet(
            num_nodes=num_nodes,
            window_size=window_size,
            c=train_config["c"],
            z_dim=train_config["z_dim"],
            emb_dim=train_config["emb_dim"],
            gat_out_dim=train_config["gat_out_dim"],
            topk=train_config["topk"],
            rho=train_config["rho"],
            dropout=train_config["dropout"],
            gat_heads=train_config["gat_heads"],
            use_sae=train_config["use_sae"],
        ).to(self.device)

        self.criterion = JointLoss(
            lambda_forecast=train_config["lambda_forecast"],
            beta=train_config["beta"],
            use_sae=train_config["use_sae"],
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_config["lr"],
            weight_decay=train_config["decay"],
        )

        # 记录模型参数量
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.logger:
            self.logger.info(f"模型参数量: {num_params:,}")

    def run(self):
        model_save_path = (
            self.env_config["load_model_path"]
            if len(self.env_config["load_model_path"]) > 0
            else self.get_save_path()
        )

        if len(self.env_config["load_model_path"]) == 0:
            train(
                model=self.model,
                optimizer=self.optimizer,
                criterion=self.criterion,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                device=self.device,
                epochs=self.train_config["epoch"],
                save_path=model_save_path,
                grad_clip_norm=5.0,
                recon_target_mode=self.train_config["recon_target_mode"],
                log_interval=self.train_config["log_interval"],
                graph_warmup_epochs=self.train_config["graph_warmup_epochs"],
                logger=self.logger,
            )

        incompatible = self.model.load_state_dict(torch.load(model_save_path, map_location=self.device), strict=False)
        self.model.to(self.device)
        self.model.eval()

        missing_keys = list(getattr(incompatible, "missing_keys", []))
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
        if self.logger:
            self.logger.info(f"Checkpoint missing_keys: {missing_keys}")
            self.logger.info(f"Checkpoint unexpected_keys: {unexpected_keys}")
            if missing_keys:
                self.logger.warning(
                    "[Strong Warning] Detected missing_keys when loading checkpoint. "
                    "Do not mix use_sae=0 and use_sae=1 checkpoints."
                )

        # (保持原有的模型加载和 eval 不变)
        self.model.eval()

        if self.logger:
            self.logger.info(f"SAE score type: {self.train_config['sae_score_type']}")


        # 1. 提取验证集原始误差
        val_res = get_raw_errors(
            self.model,
            self.val_dataloader,
            self.criterion,
            self.device,
            self.train_config["recon_target_mode"],
            self.train_config["sae_score_type"],
        )
        
        # 2. 核心！计算验证集每个传感器的 Median 和 IQR
        fore_median, fore_iqr = get_val_stats(val_res["fore_err"])
        
        # 3. 对验证集进行标准化，融合逻辑基于 use_sae 标志
        val_fore_norm = normalize_and_score(val_res["fore_err"], fore_median, fore_iqr)
        
        if self.train_config["use_sae"]:
            sae_median, sae_iqr = get_val_stats(val_res["sae_err"])
            val_sae_norm = normalize_and_score(val_res["sae_err"], sae_median, sae_iqr)
            val_fused = weighted_harmonic_mean(val_fore_norm, val_sae_norm, self.train_config["score_lambda"])
        else:
            # Ablation: use_sae=0，只用预测误差
            if self.logger:
                self.logger.info("Fusion disabled because use_sae=0")
            val_fused = val_fore_norm
        
        # 【修复点3】：取每个时间步里，所有传感器中“最异常”的那个值作为该时间步的整体分数
        val_anomaly_scores = np.max(val_fused, axis=1) 
        threshold = float(np.max(val_anomaly_scores)) # 验证集最大分数作为阈值

        # 4. 在测试集上走相同的流水线，必须使用验证集的 Median 和 IQR！
        test_res = get_raw_errors(
            self.model,
            self.test_dataloader,
            self.criterion,
            self.device,
            self.train_config["recon_target_mode"],
            self.train_config["sae_score_type"],
        )

        if self.train_config["sae_score_type"] == "sparsity_dev":
            log_sparsity_fore_stats(self.logger, test_res["fore_err"], test_res["sparsity_dev_err"])
        
        test_fore_norm = normalize_and_score(test_res["fore_err"], fore_median, fore_iqr)
        
        if self.train_config["use_sae"]:
            test_sae_norm = normalize_and_score(test_res["sae_err"], sae_median, sae_iqr)
            test_fused = weighted_harmonic_mean(test_fore_norm, test_sae_norm, self.train_config["score_lambda"])
        else:
            # Ablation: use_sae=0，只用预测误差
            test_fused = test_fore_norm

        # =========================================================================
        # 取每个时间步所有传感器中的最大异常值
        test_anomaly_scores = np.max(test_fused, axis=1)
        
        # 【新增：简单移动平均 SMA 平滑毛刺】
        def smooth_sma(arr, window_size=5):
            return np.convolve(arr, np.ones(window_size)/window_size, mode='same')
            
        test_anomaly_scores = smooth_sma(test_anomaly_scores, window_size=5)

        # ================== 【新增：学术界通用的 Best-F1 评估协议】 ==================
        from sklearn.metrics import precision_recall_curve
        
        # 1. 传统验证集最大值阈值（供参考对比）
        metric_val_thresh = evaluate_with_threshold(test_anomaly_scores, test_res["labels"], threshold)
        
        # 2. 计算 Best-F1 (通过 precision_recall_curve 扫描所有可能的阈值)
        precisions, recalls, thresholds_pr = precision_recall_curve(test_res["labels"], test_anomaly_scores)
        
        # 计算所有阈值对应的 F1 (加 1e-8 防止除以 0)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        
        # 找到 F1 最大值及其对应的索引
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_p = precisions[best_idx]
        best_r = recalls[best_idx]
        
        # thresholds_pr 的长度比 precisions/recalls 少 1，需做边界保护
        best_thresh = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else threshold

        # 3. 打印最终战报
        result_lines = [
            "=========================** Result **============================",
            f"SAE Score Type => {self.train_config['sae_score_type']}",
            f"Test Loss => total={test_res['loss']['total']:.6f}, fore={test_res['loss']['forecast']:.6f}, recon={test_res['loss']['reconstruction']:.6f}, kl={test_res['loss']['sparsity']:.6f}",
            "---",
            f"[严格验证集阈值={threshold:.6f}] F1={metric_val_thresh['f1']:.4f} | P={metric_val_thresh['precision']:.4f} | R={metric_val_thresh['recall']:.4f}",
            "---",
            f"[论文标准 Best-F1 阈值={best_thresh:.6f}] F1={best_f1:.4f} | P={best_p:.4f} | R={best_r:.4f}  <-- 你的最终战力",
            "================================================================="
        ]
        
        for line in result_lines:
            if self.logger:
                self.logger.info(line)
            else:
                print(line)
        # =========================================================================

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        random.seed(seed)

        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)

        if val_use_len <= 0:
            raise ValueError("val_ratio 太小，导致验证集为空。")

        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([
            indices[:val_start_index],
            indices[val_start_index + val_use_len :],
        ])
        val_sub_indices = indices[val_start_index : val_start_index + val_use_len]

        train_subset = Subset(train_dataset, train_sub_indices)
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False, num_workers=0)

        return train_dataloader, val_dataloader

    def get_save_path(self) -> str:
        """生成本次运行目录并返回模型保存路径"""
        if self.datestr is None:
            if self.timezone:
                from pytz import timezone
                try:
                    tz = timezone(self.timezone)
                    now = datetime.now(tz=tz)
                except Exception:
                    now = datetime.now()
            else:
                now = datetime.now()
            self.datestr = now.strftime("%m-%d-%H-%M-%S")

        run_dir = f"./runs/{self.env_config['dataset']}_{self.datestr}"
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        return f"{run_dir}/best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", type=str, default="msl", help="dataset folder name under ./data")
    parser.add_argument("-device", type=str, default="cpu", help="cpu / cuda")
    parser.add_argument("-load_model_path", type=str, default="", help="load pre-trained checkpoint path")

    parser.add_argument("-batch", type=int, default=64)
    parser.add_argument("-epoch", type=int, default=30)
    parser.add_argument("-slide_win", type=int, default=15)
    parser.add_argument("-slide_stride", type=int, default=1)
    parser.add_argument("-val_ratio", type=float, default=0.2)
    parser.add_argument(
        "-random_seed",
        dest="random_seed",
        type=int,
        default=5,
        help="random seed (legacy name; overridden by -seed if both are provided)",
    )
    parser.add_argument(
        "-seed",
        dest="seed",
        type=int,
        default=None,
        help="random seed (preferred alias; takes priority over -random_seed)",
    )

    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-decay", type=float, default=0.0)

    parser.add_argument("-c", type=int, default=16)
    parser.add_argument("-z_dim", type=int, default=12)
    parser.add_argument("-emb_dim", type=int, default=10)
    parser.add_argument("-gat_out_dim", type=int, default=32)
    parser.add_argument("-gat_heads", type=int, default=2)
    parser.add_argument("-topk", type=int, default=5)
    parser.add_argument("-rho", type=float, default=0.05)
    parser.add_argument("-dropout", type=float, default=0.1)

    parser.add_argument("-lambda_forecast", type=float, default=0.5, help="joint loss forecast weight lambda")
    parser.add_argument("-beta", type=float, default=1e-3, help="joint loss sparsity weight beta")
    parser.add_argument("-use_sae", type=int, default=1, help="whether to use SAE (1=yes, 0=no for ablation)")

    parser.add_argument("-score_lambda", type=float, default=0.5, help="WHM score fusion weight lambda")
    parser.add_argument("-sae_score_type", type=str, default="recon", choices=["recon", "sparsity_dev"], help="SAE score type for fusion")
    parser.add_argument("-recon_target_mode", type=str, default="input", help="input / (future custom mode)")
    parser.add_argument("-log_interval", type=int, default=100)
    parser.add_argument("-graph_warmup_epochs", type=int, default=5, help="freeze graph_learning for first N epochs")
    parser.add_argument("-timezone", type=str, default="Asia/Shanghai", help="时区设置，例如: Asia/Shanghai, UTC, America/New_York")

    args = parser.parse_args()

    seed_provided = "-seed" in sys.argv
    random_seed_provided = "-random_seed" in sys.argv
    resolved_seed = args.seed if args.seed is not None else args.random_seed
    if seed_provided and random_seed_provided:
        print(
            f"[Warning] Both -random_seed ({args.random_seed}) and -seed ({args.seed}) are provided. "
            f"Using -seed={resolved_seed}."
        )

    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    torch.cuda.manual_seed(resolved_seed)
    torch.cuda.manual_seed_all(resolved_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(resolved_seed)

    train_config = {
        "batch": args.batch,
        "epoch": args.epoch,
        "slide_win": args.slide_win,
        "slide_stride": args.slide_stride,
        "val_ratio": args.val_ratio,
        "seed": resolved_seed,
        "lr": args.lr,
        "decay": args.decay,
        "c": args.c,
        "z_dim": args.z_dim,
        "emb_dim": args.emb_dim,
        "gat_out_dim": args.gat_out_dim,
        "gat_heads": args.gat_heads,
        "topk": args.topk,
        "rho": args.rho,
        "dropout": args.dropout,
        "lambda_forecast": args.lambda_forecast,
        "beta": args.beta,
        "use_sae": args.use_sae,
        "score_lambda": args.score_lambda,
        "sae_score_type": args.sae_score_type,
        "recon_target_mode": args.recon_target_mode,
        "log_interval": args.log_interval,
        "graph_warmup_epochs": args.graph_warmup_epochs,
    }

    env_config = {
        "dataset": args.dataset,
        "device": args.device,
        "load_model_path": args.load_model_path,
    }

    main_instance = Main(train_config, env_config, timezone=args.timezone)

    # 初始化 logger（需要 datestr，创建 Main 后再生成）
    main_instance.get_save_path()  # 触发 datestr 初始化，并创建 run_dir
    run_dir = f"./runs/{args.dataset}_{main_instance.datestr}"
    logger = setup_logger(log_dir=run_dir, run_name="train", tz_name=args.timezone)
    main_instance.logger = logger

    # 记录完整运行参数
    logger.info("===== 实验开始 =====")
    logger.info(f"数据集: {args.dataset} | 设备: {args.device}")
    logger.info(f"train_config: { {k: v for k, v in train_config.items()} }")
    logger.info(f"env_config: {env_config}")

    main_instance.run()
