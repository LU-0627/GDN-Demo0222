#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BEST_F1_PATTERN = re.compile(
    r"Best-F1\s*阈值=([0-9eE+\-.]+)\]\s*F1=([0-9eE+\-.]+)\s*\|\s*P=([0-9eE+\-.]+)\s*\|\s*R=([0-9eE+\-.]+)"
)


@dataclass
class EvalSetting:
    setting: str
    use_sae: int
    score_lambda: float
    sae_score_type: str
    ckpt_group: str


def _run_command(cmd: List[str], cwd: str) -> str:
    print(f"\n[RUN] {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = proc.stdout
    print(output)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code={proc.returncode}: {' '.join(cmd)}")
    return output


def _parse_best_f1(output: str) -> Tuple[float, float, float, float]:
    matches = BEST_F1_PATTERN.findall(output)
    if not matches:
        raise RuntimeError("Cannot find Best-F1 line in output.")
    best_thresh, f1, p, r = matches[-1]
    return float(f1), float(p), float(r), float(best_thresh)


def _find_latest_checkpoint(repo_root: str, save_path_pattern: str) -> Optional[str]:
    ckpt_glob = os.path.join(repo_root, "pretrained", save_path_pattern, "best_*.pt")
    paths = glob.glob(ckpt_glob)
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def _ensure_checkpoint(
    repo_root: str,
    dataset: str,
    device: str,
    epoch: int,
    batch: int,
    random_seed: int,
    save_path_pattern: str,
    use_sae: int,
    force_train: bool,
) -> str:
    existing = _find_latest_checkpoint(repo_root, save_path_pattern)
    if existing is not None and not force_train:
        print(f"[INFO] Reuse checkpoint: {existing}")
        return existing

    cmd = [
        sys.executable,
        "main.py",
        "-dataset",
        dataset,
        "-device",
        device,
        "-epoch",
        str(epoch),
        "-batch",
        str(batch),
        "-random_seed",
        str(random_seed),
        "-use_sae",
        str(use_sae),
        "-save_path_pattern",
        save_path_pattern,
        "-score_lambda",
        "1.0",
        "-sae_score_type",
        "recon",
    ]
    _run_command(cmd, cwd=repo_root)

    created = _find_latest_checkpoint(repo_root, save_path_pattern)
    if created is None:
        raise RuntimeError(f"No checkpoint found after training under pretrained/{save_path_pattern}")
    print(f"[INFO] New checkpoint: {created}")
    return created


def _run_eval(
    repo_root: str,
    dataset: str,
    device: str,
    batch: int,
    random_seed: int,
    checkpoint_path: str,
    setting: EvalSetting,
    save_path_pattern: str,
) -> Dict[str, str]:
    cmd = [
        sys.executable,
        "main.py",
        "-dataset",
        dataset,
        "-device",
        device,
        "-batch",
        str(batch),
        "-random_seed",
        str(random_seed),
        "-use_sae",
        str(setting.use_sae),
        "-score_lambda",
        str(setting.score_lambda),
        "-sae_score_type",
        setting.sae_score_type,
        "-save_path_pattern",
        save_path_pattern,
        "-load_model_path",
        checkpoint_path,
    ]
    out = _run_command(cmd, cwd=repo_root)
    f1, precision, recall, best_thresh = _parse_best_f1(out)

    return {
        "setting": setting.setting,
        "F1": f"{f1:.6f}",
        "Precision": f"{precision:.6f}",
        "Recall": f"{recall:.6f}",
        "BestThresh": f"{best_thresh:.6f}",
        "score_lambda": str(setting.score_lambda),
        "sae_score_type": setting.sae_score_type,
        "use_sae": str(setting.use_sae),
        "checkpoint_path": checkpoint_path,
    }


def _write_csv(rows: List[Dict[str, str]], output_csv: str):
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "setting",
        "F1",
        "Precision",
        "Recall",
        "BestThresh",
        "score_lambda",
        "sae_score_type",
        "use_sae",
        "checkpoint_path",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: List[Dict[str, str]], output_md: str):
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "setting",
        "F1",
        "Precision",
        "Recall",
        "BestThresh",
        "score_lambda",
        "sae_score_type",
        "use_sae",
        "checkpoint_path",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run SWaT A/B/C ablation and summarize Best-F1 metrics.")
    parser.add_argument("--dataset", type=str, default="swat")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=5)
    parser.add_argument("--c-score-lambda", type=float, default=0.8)
    parser.add_argument("--output-csv", type=str, default="results/ablation/swat_ablation_summary.csv")
    parser.add_argument("--output-md", type=str, default="results/ablation/swat_ablation_summary.md")
    parser.add_argument("--save-prefix", type=str, default="ablation_swat")
    parser.add_argument("--force-train", action="store_true", help="Force retraining even if checkpoints exist.")
    args = parser.parse_args()

    repo_root = str(Path(__file__).resolve().parents[1])

    ckpt_patterns = {
        "use_sae0": f"{args.save_prefix}_use_sae0",
        "use_sae1": f"{args.save_prefix}_use_sae1",
    }

    checkpoint_map = {
        "use_sae0": _ensure_checkpoint(
            repo_root=repo_root,
            dataset=args.dataset,
            device=args.device,
            epoch=args.epoch,
            batch=args.batch,
            random_seed=args.random_seed,
            save_path_pattern=ckpt_patterns["use_sae0"],
            use_sae=0,
            force_train=args.force_train,
        ),
        "use_sae1": _ensure_checkpoint(
            repo_root=repo_root,
            dataset=args.dataset,
            device=args.device,
            epoch=args.epoch,
            batch=args.batch,
            random_seed=args.random_seed,
            save_path_pattern=ckpt_patterns["use_sae1"],
            use_sae=1,
            force_train=args.force_train,
        ),
    }

    settings = [
        EvalSetting(
            setting="A_use_sae0_fore_only",
            use_sae=0,
            score_lambda=1.0,
            sae_score_type="recon",
            ckpt_group="use_sae0",
        ),
        EvalSetting(
            setting="B_use_sae1_fore_only",
            use_sae=1,
            score_lambda=1.0,
            sae_score_type="recon",
            ckpt_group="use_sae1",
        ),
        EvalSetting(
            setting="C_use_sae1_fusion_recon",
            use_sae=1,
            score_lambda=args.c_score_lambda,
            sae_score_type="recon",
            ckpt_group="use_sae1",
        ),
        EvalSetting(
            setting="C_use_sae1_fusion_sparsity_dev",
            use_sae=1,
            score_lambda=args.c_score_lambda,
            sae_score_type="sparsity_dev",
            ckpt_group="use_sae1",
        ),
    ]

    rows: List[Dict[str, str]] = []
    for setting in settings:
        ckpt_path = checkpoint_map[setting.ckpt_group]
        save_path_pattern = f"{args.save_prefix}_eval_{setting.setting}"
        row = _run_eval(
            repo_root=repo_root,
            dataset=args.dataset,
            device=args.device,
            batch=args.batch,
            random_seed=args.random_seed,
            checkpoint_path=ckpt_path,
            setting=setting,
            save_path_pattern=save_path_pattern,
        )
        rows.append(row)

    _write_csv(rows, args.output_csv)
    _write_markdown(rows, args.output_md)

    print("\n================ Ablation Summary ================")
    print(f"CSV: {os.path.abspath(args.output_csv)}")
    print(f"MD : {os.path.abspath(args.output_md)}")
    for row in rows:
        print(
            f"- {row['setting']}: F1={row['F1']} P={row['Precision']} "
            f"R={row['Recall']} Th={row['BestThresh']}"
        )


if __name__ == "__main__":
    main()
