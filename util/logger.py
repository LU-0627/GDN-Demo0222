# -*- coding: utf-8 -*-
"""
日志工具模块：同时输出到控制台和日志文件，方便云服务器实验追踪。
支持时区设置，确保时间戳准确。
"""
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from pytz import timezone, utc


class TZFormatter(logging.Formatter):
    """带时区支持的日志格式化器"""
    
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = tz or timezone('Asia/Shanghai')  # 默认使用北京时间
    
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=utc)
        return dt.astimezone(self.tz)
    
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec='milliseconds')
            except TypeError:
                s = dt.isoformat()
        return s


def setup_logger(log_dir: str = "logs", run_name: str = "run", tz_name: str = "Asia/Shanghai") -> logging.Logger:
    """
    初始化并返回项目统一 logger。

    Args:
        log_dir:  日志目录（默认 ./logs/）
        run_name: 日志文件名前缀，建议格式 "<dataset>_<timestamp>"
        tz_name:  时区名称，例如 "Asia/Shanghai"（北京时间）、"UTC"、"America/New_York" 等

    Returns:
        配置好的 logging.Logger 实例
    """
    # 确保日志目录存在
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(log_dir, f"{run_name}.log")

    # 日志格式：[时间] [级别] 消息
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    # 创建带时区的格式化器
    try:
        tz = timezone(tz_name)
    except:
        tz = timezone('Asia/Shanghai')  # 如果时区名称无效，默认使用北京时间
    
    formatter = TZFormatter(fmt=fmt, datefmt=datefmt, tz=tz)

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

    # 显示当前时区信息
    current_time = datetime.now(tz=tz)
    logger.info(f"日志文件路径: {os.path.abspath(log_file)}")
    logger.info(f"当前时区: {tz_name}")
    logger.info(f"当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    return logger
