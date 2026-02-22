# -*- coding: utf-8 -*-
"""
日志工具模块：同时输出到控制台和日志文件，方便云服务器实验追踪。
"""
import logging
import os
from pathlib import Path


def setup_logger(log_dir: str = "logs", run_name: str = "run") -> logging.Logger:
    """
    初始化并返回项目统一 logger。

    Args:
        log_dir:  日志目录（默认 ./logs/）
        run_name: 日志文件名前缀，建议格式 "<dataset>_<timestamp>"

    Returns:
        配置好的 logging.Logger 实例
    """
    # 确保日志目录存在
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(log_dir, f"{run_name}.log")

    # 日志格式：[时间] [级别] 消息
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler（重复调用时）
    if logger.handlers:
        logger.handlers.clear()

    # 文件 handler：记录到日志文件
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 控制台 handler：SSH 会话实时查看
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"日志文件路径: {os.path.abspath(log_file)}")
    return logger
