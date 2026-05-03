from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime

# 配置日志系统
def setup_logging():
    """配置日志：同时输出到控制台和日志文件"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"gvs2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # 控制台
            logging.FileHandler(log_file, encoding='utf-8')  # 文件
        ]
    )
    return log_file

log_file = setup_logging()
logger = logging.getLogger(__name__)

from pipeline.runner import run_gvs2

__all__ = ["run_gvs2", "launch"]


def launch() -> int:
    logger.info(f"GVS2启动，日志文件: {log_file}")
    from ui.main_window import launch as launch_ui
    return launch_ui()


if __name__ == "__main__":
    raise SystemExit(launch())
