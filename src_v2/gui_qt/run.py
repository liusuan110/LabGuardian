"""
LabGuardian PySide6 入口
=========================================
推荐通过 launcher.py 启动（自动诊断 + .env 加载）。
此文件保留作为 gui_qt 包的直接入口（向后兼容）:
  python -m gui_qt.run   (从 src_v2/ 目录)
  python run_qt.py       (从 src_v2/ 目录)
"""

import sys
import platform
import subprocess
from pathlib import Path

# 确保 src_v2 在路径中
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from app_context import AppContext
from gui_qt.main_window import MainWindow
from gui_qt.styles import GLOBAL_STYLE


def _detect_font() -> tuple:
    """跨平台字体选择: Windows → Segoe UI, Linux → Noto Sans CJK SC / 回退"""
    if sys.platform == "win32":
        return ("Segoe UI", 10)
    elif sys.platform == "linux":
        for font in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Ubuntu", "DejaVu Sans"]:
            try:
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


def main():
    # 高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # 全局样式
    app.setStyleSheet(GLOBAL_STYLE)

    # 跨平台字体
    font_family, font_size = _detect_font()
    font = QFont(font_family, font_size)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    # 创建 AppContext (服务注册中心)
    ctx = AppContext()

    # 主窗口
    window = MainWindow(ctx=ctx)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
