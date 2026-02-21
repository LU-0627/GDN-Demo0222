# -*- coding: utf-8 -*-
import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from datasets.TimeDataset import TimeDataset
from models.topofusagnet import TopoFuSAGNet, JointLoss
from test import evaluate_loop, get_threshold_from_validation, evaluate_with_threshold
from train import train
from util.env import get_device, set_device
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.preprocess import build_loc_net, construct_data


class Main:
    def __init__(self, train_config, env_config):
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config["dataset"]
        train_orig = pd.read_csv(f"./data/{dataset}/train.csv", sep=",", index_col=0)
        test_orig = pd.read_csv(f"./data/{dataset}/test.csv", sep=",", index_col=0)

        train_df, test_df = train_orig.copy(), test_orig.copy()

        if "attack" in train_df.columns:
            train_df = train_df.drop(columns=["attack"])

        if "attack" not in test_df.columns:
            raise ValueError("test.csv 必须包含 attack 列作为测试标签")

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
        ).to(self.device)

        self.criterion = JointLoss(
            lambda_forecast=train_config["lambda_forecast"],
            beta=train_config["beta"],
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_config["lr"],
            weight_decay=train_config["decay"],
        )

    def run(self):
        model_save_path = (
            self.env_config["load_model_path"]
            if len(self.env_config["load_model_path"]) > 0
            else self.get_save_path()[0]
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
            )

        self.model.load_state_dict(torch.load(model_save_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        # ===== 关键注释：验证集阈值 =====
        # 使用改造后的融合分数，在验证集上取最大值作为 threshold
        val_eval = evaluate_loop(
            model=self.model,
            dataloader=self.val_dataloader,
            criterion=self.criterion,
            device=self.device,
            score_lambda=self.train_config["score_lambda"],
            recon_target_mode=self.train_config["recon_target_mode"],
        )
        threshold = get_threshold_from_validation(val_eval["anomaly_score"])

        # ===== 关键注释：测试集判定 =====
        # 最终融合异常分数 > threshold => 异常(1)
        test_eval = evaluate_loop(
            model=self.model,
            dataloader=self.test_dataloader,
            criterion=self.criterion,
            device=self.device,
            score_lambda=self.train_config["score_lambda"],
            recon_target_mode=self.train_config["recon_target_mode"],
        )

        metric = evaluate_with_threshold(
            test_scores=test_eval["anomaly_score"],
            test_labels=test_eval["labels"],
            threshold=threshold,
        )

        print("=========================** Result **============================")
        print(f"Threshold (from val max score): {threshold:.6f}")
        print(
            f"Test Loss => total={test_eval['loss']['total']:.6f}, "
            f"fore={test_eval['loss']['forecast']:.6f}, "
            f"recon={test_eval['loss']['reconstruction']:.6f}, "
            f"kl={test_eval['loss']['sparsity']:.6f}"
        )
        print(f"Precision: {metric['precision']:.6f}")
        print(f"Recall:    {metric['recall']:.6f}")
        print(f"F1-Score:  {metric['f1']:.6f}")

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

    def get_save_path(self):
        dir_path = self.env_config["save_path"]

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime("%m-%d-%H-%M-%S")
        datestr = self.datestr

        paths = [
            f"./pretrained/{dir_path}/best_{datestr}.pt",
            f"./results/{dir_path}/{datestr}.csv",
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", type=str, default="msl", help="dataset folder name under ./data")
    parser.add_argument("-device", type=str, default="cpu", help="cpu / cuda")
    parser.add_argument("-load_model_path", type=str, default="", help="load pre-trained checkpoint path")
    parser.add_argument("-save_path_pattern", type=str, default="topofusagnet", help="save path sub-folder")

    parser.add_argument("-batch", type=int, default=64)
    parser.add_argument("-epoch", type=int, default=30)
    parser.add_argument("-slide_win", type=int, default=15)
    parser.add_argument("-slide_stride", type=int, default=1)
    parser.add_argument("-val_ratio", type=float, default=0.2)
    parser.add_argument("-random_seed", type=int, default=5)

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

    parser.add_argument("-score_lambda", type=float, default=0.5, help="WHM score fusion weight lambda")
    parser.add_argument("-recon_target_mode", type=str, default="input", help="input / (future custom mode)")
    parser.add_argument("-log_interval", type=int, default=100)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)

    train_config = {
        "batch": args.batch,
        "epoch": args.epoch,
        "slide_win": args.slide_win,
        "slide_stride": args.slide_stride,
        "val_ratio": args.val_ratio,
        "seed": args.random_seed,
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
        "score_lambda": args.score_lambda,
        "recon_target_mode": args.recon_target_mode,
        "log_interval": args.log_interval,
    }

    env_config = {
        "save_path": args.save_path_pattern,
        "dataset": args.dataset,
        "device": args.device,
        "load_model_path": args.load_model_path,
    }

    main = Main(train_config, env_config)
    main.run()
