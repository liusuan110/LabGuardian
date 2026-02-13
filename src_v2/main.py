#!/usr/bin/env python3
"""
LabGuardian v2 — 主入口
基于边缘AI的智能理工科实验助教系统

Usage:
    python main.py                     # 默认启动
    LG_CAMERA_ID=1 python main.py      # 指定摄像头
    LG_API_KEY=sk-xxx python main.py   # 指定 API Key
"""

import sys
import tkinter as tk

from config import print_config_summary
from gui.app import LabGuardianApp


def main():
    print_config_summary()

    root = tk.Tk()
    app = LabGuardianApp(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        app._close()
        sys.exit(0)


if __name__ == "__main__":
    main()
