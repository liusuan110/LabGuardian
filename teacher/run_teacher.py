#!/usr/bin/env python3
"""
LabGuardian Teacher Server — 独立启动入口
=========================================

独立运行教师端服务器 (不需要学生端 GUI)。
适用于: 教师在自己的电脑上运行, 或在单独的服务器上部署。

用法:
    python run_teacher.py                   # 默认 0.0.0.0:8080
    python run_teacher.py --port 9090       # 指定端口
    python run_teacher.py --host 127.0.0.1  # 仅本地访问
"""

import sys
import argparse
import logging
from pathlib import Path

# ---- 路径锚定 ----
TEACHER_DIR = Path(__file__).resolve().parent
LAB_ROOT = TEACHER_DIR.parent

if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("TeacherServer")


def main():
    parser = argparse.ArgumentParser(description="LabGuardian Teacher Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="监听端口 (default: 8080)")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("LabGuardian Teacher Server (Standalone)")
    logger.info("=" * 50)
    logger.info(f"  监听: http://{args.host}:{args.port}")
    logger.info(f"  API 文档: http://localhost:{args.port}/docs")
    logger.info("=" * 50)

    import uvicorn
    from teacher.server import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
