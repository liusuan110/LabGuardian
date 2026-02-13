"""
视频显示面板
职责：管理摄像头/图片输入，显示带标注的视频帧
"""

import cv2
import time
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from typing import Optional, Callable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import camera as cam_cfg, gui as gui_cfg


class VideoPanel:
    """视频面板：管理视频输入源和帧显示"""

    def __init__(self, parent: tk.Widget, on_frame_callback: Optional[Callable] = None):
        """
        Args:
            parent: 父容器
            on_frame_callback: 每帧回调 fn(frame) -> annotated_frame
        """
        self.parent = parent
        self.on_frame = on_frame_callback
        self.is_running = True

        # 输入源
        self.input_source = "camera"  # "camera" | "image"
        self.static_frame: Optional[np.ndarray] = None

        # 创建 UI
        self.label = ttk.Label(parent, text="摄像头初始化中...",
                               background="black", foreground="white")
        self.label.pack(fill=tk.BOTH, expand=True)

    def start_video_thread(self):
        """启动视频采集线程"""
        threading.Thread(target=self._video_loop, daemon=True).start()

    def load_test_image(self):
        """从文件加载测试图片"""
        file_path = filedialog.askopenfilename(
            title="Select Circuit Image",
            filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")]
        )
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                self.static_frame = img
                self.input_source = "image"
                return Path(file_path).name
        return None

    def switch_to_camera(self):
        """切换回摄像头模式"""
        self.input_source = "camera"
        self.static_frame = None

    def stop(self):
        """停止视频线程"""
        self.is_running = False

    def _video_loop(self):
        """视频采集主循环 (在后台线程运行)"""
        cap = cv2.VideoCapture(cam_cfg.device_id, cam_cfg.cv2_backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.height)

        while self.is_running:
            if self.input_source == "image" and self.static_frame is not None:
                frame = self.static_frame.copy()
                time.sleep(0.05)
            else:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

            # 调用帧处理回调 (检测 + 标注)
            if self.on_frame:
                try:
                    annotated = self.on_frame(frame)
                    display_frame = annotated if annotated is not None else frame
                except Exception as e:
                    print(f"[VideoPanel] Frame callback error: {e}")
                    display_frame = frame
            else:
                display_frame = frame

            # 转换并显示
            self._update_display(display_frame)
            time.sleep(0.01)

        cap.release()

    def _update_display(self, frame: np.ndarray):
        """将 OpenCV 帧更新到 Tkinter Label 上"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # 获取容器大小
            lw = self.label.winfo_width()
            lh = self.label.winfo_height()
            if lw < 100:
                lw = gui_cfg.video_max_display[0]
            if lh < 100:
                lh = gui_cfg.video_max_display[1]

            pil_img.thumbnail((lw, lh), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(image=pil_img)

            self.label.after(0, lambda img=tk_img: self._set_image(img))
        except Exception:
            pass

    def _set_image(self, img):
        self.label.configure(image=img)
        self.label.image = img
