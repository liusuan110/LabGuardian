#!/usr/bin/env python3
"""
LabGuardian 竞赛级启动器
==========================================

功能:
  1. 自动定位项目根目录（不依赖 CWD）
  2. 自动加载 .env 配置文件
  3. 启动前依赖检查 & 诊断报告
  4. 跨平台字体适配（Windows / Ubuntu DK-2500）
  5. 全局异常捕获 + 崩溃日志
  6. 可选看门狗自动重启（--watchdog）

用法:
  python launcher.py              # 正常启动
  python launcher.py --diag       # 仅运行诊断，不启动 GUI
  python launcher.py --watchdog   # 启动看门狗，崩溃后自动重启
  python launcher.py --env .env   # 指定 .env 文件路径

环境:
  Windows:  python launcher.py  /  双击 launch.bat
  DK-2500:  ./start.sh  /  python3 launcher.py
"""

import sys
import os
import time
import logging
import argparse
import platform
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================
# 0. 路径锚定 — 无论从哪里执行都能找到 src_v2
# ============================================================
LAUNCHER_DIR = Path(__file__).resolve().parent              # src_v2/
PROJECT_ROOT = LAUNCHER_DIR.parent                          # LabGuardian/
LOG_DIR = PROJECT_ROOT / "logs"

# 确保 src_v2 在 Python 路径最前面
if str(LAUNCHER_DIR) not in sys.path:
    sys.path.insert(0, str(LAUNCHER_DIR))

# 切换工作目录到 src_v2（兼容相对路径引用）
os.chdir(LAUNCHER_DIR)


# ============================================================
# 1. 日志系统
# ============================================================
def setup_logging(log_dir: Path = LOG_DIR) -> logging.Logger:
    """配置文件 + 控制台双输出日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"labguardian_{timestamp}.log"

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # 文件 handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger = logging.getLogger("LabGuardian")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"日志文件: {log_file}")
    return logger


# ============================================================
# 2. .env 文件加载器（零依赖，不需要 python-dotenv）
# ============================================================
def load_dotenv(env_path: Path) -> int:
    """解析 .env 文件并注入 os.environ，返回加载的变量数量"""
    if not env_path.exists():
        return 0
    count = 0
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:  # 不覆盖已有环境变量
                os.environ[key] = value
                count += 1
    return count


# ============================================================
# 3. 依赖预检（快速失败，不要等到 import 才报错）
# ============================================================
REQUIRED_MODULES = {
    "PySide6": "PySide6",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "ultralytics": "ultralytics",
    "networkx": "networkx",
}

OPTIONAL_MODULES = {
    "openvino_genai": "openvino-genai (本地 LLM)",
    "openvino": "openvino (模型加速)",
    "openai": "openai (云端 LLM API)",
    "schemdraw": "schemdraw (原理图可视化)",
}


def check_dependencies(logger: logging.Logger) -> bool:
    """检查必需 & 可选依赖，返回是否全部满足"""
    all_ok = True

    logger.info("─── 依赖检查 ───")
    for module, pip_name in REQUIRED_MODULES.items():
        try:
            __import__(module)
            logger.info(f"  ✔ {module}")
        except ImportError:
            logger.error(f"  ✘ {module} — 请安装: pip install {pip_name}")
            all_ok = False

    for module, desc in OPTIONAL_MODULES.items():
        try:
            __import__(module)
            logger.info(f"  ○ {module} (可选, 已安装)")
        except ImportError:
            logger.info(f"  △ {module} (可选, 未安装 — {desc})")

    return all_ok


# ============================================================
# 4. 系统诊断
# ============================================================
def run_diagnostics(logger: logging.Logger) -> dict:
    """运行系统诊断，返回状态字典"""
    diag = {
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": str(Path.cwd()),
        "project_root": str(PROJECT_ROOT),
    }

    # Python 版本
    logger.info("─── 系统诊断 ───")
    logger.info(f"  平台:     {diag['platform']}")
    logger.info(f"  Python:   {sys.version.split()[0]}")
    logger.info(f"  项目根:   {PROJECT_ROOT}")

    # 摄像头检测
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"  摄像头:   ✔ 设备 0 ({w}x{h})")
            diag["camera"] = f"OK ({w}x{h})"
            cap.release()
        else:
            logger.warning("  摄像头:   ✘ 设备 0 无法打开")
            diag["camera"] = "FAIL"
            cap.release()
    except Exception as e:
        logger.warning(f"  摄像头:   ✘ {e}")
        diag["camera"] = f"ERROR: {e}"

    # YOLO 模型检测
    try:
        from config import find_best_yolo_model, MODELS_DIR
        model_path = find_best_yolo_model()
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1024 / 1024
            logger.info(f"  YOLO模型: ✔ {model_path.name} ({size_mb:.1f} MB)")
            diag["yolo_model"] = str(model_path)
        else:
            logger.warning(f"  YOLO模型: ✘ 未找到")
            diag["yolo_model"] = "NOT FOUND"
    except Exception as e:
        logger.warning(f"  YOLO模型: ✘ {e}")
        diag["yolo_model"] = f"ERROR: {e}"

    # LLM 模型检测
    try:
        from config import llm as llm_cfg
        if llm_cfg.is_cloud_ready:
            logger.info(f"  LLM:      ✔ 云端 API 已配置 ({llm_cfg.cloud_provider})")
            diag["llm_cloud"] = "READY"
        else:
            logger.info(f"  LLM:      △ 云端 API 未配置 (将使用本地/规则引擎)")
            diag["llm_cloud"] = "NOT CONFIGURED"

        if Path(llm_cfg.local_model_path).exists():
            logger.info(f"  LLM本地:  ✔ {Path(llm_cfg.local_model_path).name}")
            diag["llm_local"] = str(llm_cfg.local_model_path)
        else:
            logger.info(f"  LLM本地:  △ 未找到本地模型")
            diag["llm_local"] = "NOT FOUND"
    except Exception as e:
        logger.warning(f"  LLM:      ✘ {e}")

    # GPU / OpenVINO 检测
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"  CUDA:     ✔ {gpu_name}")
            diag["cuda"] = gpu_name
        else:
            logger.info(f"  CUDA:     △ 不可用")
            diag["cuda"] = "N/A"
    except Exception:
        diag["cuda"] = "N/A"

    try:
        from openvino import Core
        core = Core()
        devices = core.available_devices
        logger.info(f"  OpenVINO: ✔ 设备: {', '.join(devices)}")
        diag["openvino"] = devices
    except Exception:
        logger.info(f"  OpenVINO: △ 未安装或不可用")
        diag["openvino"] = "N/A"

    logger.info("─── 诊断完成 ───")
    return diag


# ============================================================
# 5. 跨平台字体选择
# ============================================================
def get_platform_font() -> tuple:
    """返回 (font_family, font_size)，适配当前系统"""
    if sys.platform == "win32":
        # Windows: Segoe UI 是最佳选择
        return ("Segoe UI", 10)
    elif sys.platform == "linux":
        # Ubuntu DK-2500: 按优先级尝试
        for font in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Ubuntu", "DejaVu Sans"]:
            try:
                # 快速检测字体是否存在
                result = subprocess.run(
                    ["fc-list", f":family={font}"],
                    capture_output=True, text=True, timeout=3
                )
                if result.stdout.strip():
                    return (font, 10)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return ("Sans", 10)
    else:
        return ("Helvetica", 10)


# ============================================================
# 6. 全局异常钩子
# ============================================================
def setup_exception_hook(logger: logging.Logger):
    """捕获所有未处理异常，写入日志而非直接退出"""
    def handler(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            logger.info("用户中断 (Ctrl+C)")
            sys.exit(0)
        import traceback
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical(f"未捕获异常:\n{tb_str}")
        # 尝试弹窗提示（如果 Qt 可用）
        try:
            from PySide6.QtWidgets import QMessageBox, QApplication
            app = QApplication.instance()
            if app:
                QMessageBox.critical(
                    None, "LabGuardian 错误",
                    f"发生未预期的错误:\n\n{exc_type.__name__}: {exc_value}\n\n"
                    f"详细日志已保存到 logs/ 目录"
                )
        except Exception:
            pass

    sys.excepthook = handler


# ============================================================
# 7. 主启动函数
# ============================================================
def launch_gui(logger: logging.Logger):
    """启动 PySide6 GUI"""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont

    from gui_qt.main_window import MainWindow
    from gui_qt.styles import GLOBAL_STYLE

    # HiDPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLE)

    # 跨平台字体
    font_family, font_size = get_platform_font()
    font = QFont(font_family, font_size)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    logger.info(f"字体: {font_family} {font_size}pt")

    # 主窗口
    window = MainWindow()
    window.show()
    logger.info("GUI 已启动")

    exit_code = app.exec()
    logger.info(f"GUI 退出, code={exit_code}")
    return exit_code


# ============================================================
# 8. 看门狗（评审现场防崩溃）
# ============================================================
def watchdog_loop(logger: logging.Logger, max_restarts: int = 5):
    """看门狗：崩溃后自动重启，最多重试 max_restarts 次"""
    restarts = 0
    while restarts < max_restarts:
        logger.info(f"看门狗: 启动 GUI (第 {restarts + 1} 次)")
        try:
            code = launch_gui(logger)
            if code == 0:
                logger.info("看门狗: 正常退出")
                break
        except SystemExit as e:
            if e.code == 0:
                break
            logger.warning(f"看门狗: SystemExit code={e.code}")
        except Exception as e:
            logger.error(f"看门狗: 异常崩溃 — {e}")

        restarts += 1
        if restarts < max_restarts:
            wait = min(3 * restarts, 10)
            logger.info(f"看门狗: {wait}s 后重启...")
            time.sleep(wait)

    if restarts >= max_restarts:
        logger.critical(f"看门狗: 已达最大重启次数 ({max_restarts}), 放弃")


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="LabGuardian 竞赛级启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python launcher.py              正常启动
  python launcher.py --diag       仅诊断
  python launcher.py --watchdog   看门狗模式（评审推荐）
  python launcher.py --env .env   指定环境文件
        """
    )
    parser.add_argument("--diag", action="store_true",
                        help="仅运行系统诊断，不启动 GUI")
    parser.add_argument("--watchdog", action="store_true",
                        help="启用看门狗模式，崩溃后自动重启")
    parser.add_argument("--env", type=str, default=None,
                        help=".env 文件路径 (默认自动搜索)")
    parser.add_argument("--no-diag", action="store_true",
                        help="跳过启动诊断（加速启动）")
    args = parser.parse_args()

    # === 日志 ===
    logger = setup_logging()
    logger.info("=" * 50)
    logger.info("LabGuardian Launcher v2.0")
    logger.info("=" * 50)

    # === .env 加载 ===
    env_paths = [
        Path(args.env) if args.env else None,
        LAUNCHER_DIR / ".env",           # src_v2/.env
        PROJECT_ROOT / ".env",           # LabGuardian/.env
    ]
    for ep in env_paths:
        if ep and ep.exists():
            count = load_dotenv(ep)
            logger.info(f".env 加载: {ep} ({count} 个变量)")
            break
    else:
        logger.info(".env 文件未找到 (使用默认配置)")

    # === 全局异常钩子 ===
    setup_exception_hook(logger)

    # === 依赖检查 ===
    if not check_dependencies(logger):
        logger.critical("必需依赖缺失，请先安装。退出。")
        print("\n❌ 缺少必需的 Python 包，请运行:")
        print("   pip install PySide6 opencv-python numpy ultralytics networkx")
        sys.exit(1)

    # === 系统诊断 ===
    if not args.no_diag or args.diag:
        diag = run_diagnostics(logger)
        if args.diag:
            print("\n诊断完成。详细日志见 logs/ 目录。")
            sys.exit(0)

    # === 启动 GUI ===
    if args.watchdog:
        logger.info("模式: 看门狗 (自动重启)")
        watchdog_loop(logger)
    else:
        logger.info("模式: 普通启动")
        try:
            code = launch_gui(logger)
            sys.exit(code)
        except SystemExit:
            raise
        except Exception as e:
            logger.critical(f"启动失败: {e}")
            import traceback
            logger.critical(traceback.format_exc())
            sys.exit(1)


if __name__ == "__main__":
    main()
