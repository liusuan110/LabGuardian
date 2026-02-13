"""
QThread 工作线程
职责: 将耗时操作从主线程解耦, 通过信号驱动 UI 更新

线程:
  - VideoWorker:  摄像头采集 → 帧处理 → 发送 QPixmap
  - ModelLoader:  后台加载 YOLO / LLM 模型
  - LLMWorker:    异步 LLM 问答
"""

import cv2
import time
import traceback
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker, Slot
from PySide6.QtGui import QImage, QPixmap
from pathlib import Path


class VideoWorker(QThread):
    """
    视频采集 + 帧处理线程

    信号:
        frame_ready(QPixmap): 处理后的帧 → UI 显示
        fps_updated(float):   实时帧率
        error(str):           错误信息
    """
    frame_ready = Signal(QPixmap)
    fps_updated = Signal(float)
    error = Signal(str)

    def __init__(self, device_id: int = 0, backend=None):
        super().__init__()
        self.device_id = device_id
        self.backend = backend
        self._running = True
        self._mutex = QMutex()

        # 外部注入的帧处理回调
        self._process_callback = None

        # 输入源模式
        self._source_mode = "camera"   # "camera" | "image"
        self._static_frame = None

        # 性能统计
        self._frame_count = 0
        self._fps_timer = time.time()

    def set_process_callback(self, callback):
        """设置帧处理回调: fn(np.ndarray) -> np.ndarray"""
        self._process_callback = callback

    def load_image(self, path: str) -> bool:
        """加载静态测试图片"""
        img = cv2.imread(path)
        if img is not None:
            with QMutexLocker(self._mutex):
                self._static_frame = img
                self._source_mode = "image"
            return True
        return False

    def switch_to_camera(self):
        """切换回摄像头模式"""
        with QMutexLocker(self._mutex):
            self._source_mode = "camera"
            self._static_frame = None

    @property
    def static_frame(self):
        return self._static_frame

    def stop(self):
        self._running = False
        self.wait(3000)

    def run(self):
        """线程主循环"""
        cap = None
        try:
            backend_flag = self.backend if self.backend else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.device_id, backend_flag)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                self.error.emit("无法打开摄像头，请检查设备连接")

            while self._running:
                with QMutexLocker(self._mutex):
                    mode = self._source_mode
                    static = self._static_frame

                if mode == "image" and static is not None:
                    frame = static.copy()
                    time.sleep(0.05)
                else:
                    if cap is None or not cap.isOpened():
                        time.sleep(0.5)
                        continue
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue

                # 帧处理
                if self._process_callback:
                    try:
                        processed = self._process_callback(frame)
                        display = processed if processed is not None else frame
                    except Exception as e:
                        display = frame
                        self.error.emit(f"帧处理错误: {e}")
                else:
                    display = frame

                # 转 QPixmap
                pixmap = self._cv2_to_qpixmap(display)
                self.frame_ready.emit(pixmap)

                # FPS 计算
                self._frame_count += 1
                elapsed = time.time() - self._fps_timer
                if elapsed >= 1.0:
                    fps = self._frame_count / elapsed
                    self.fps_updated.emit(fps)
                    self._frame_count = 0
                    self._fps_timer = time.time()

                time.sleep(0.01)

        except Exception as e:
            self.error.emit(f"视频线程异常: {traceback.format_exc()}")
        finally:
            if cap:
                cap.release()

    @staticmethod
    def _cv2_to_qpixmap(frame: np.ndarray) -> QPixmap:
        """BGR ndarray → QPixmap (零拷贝优化)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())  # copy() 确保线程安全


class ModelLoaderWorker(QThread):
    """
    后台模型加载线程

    信号:
        progress(str):    进度消息
        finished(bool):   加载完成 (success/fail)
    """
    progress = Signal(str)
    finished = Signal(bool)

    def __init__(self, detector, llm_engine, ocr_engine=None):
        super().__init__()
        self.detector = detector
        self.llm_engine = llm_engine
        self.ocr_engine = ocr_engine

    def run(self):
        success = True
        try:
            # 1. 加载 YOLO
            self.progress.emit("正在加载视觉识别模型...")
            if self.detector.load():
                self.progress.emit("✅ 视觉模型加载成功")
            else:
                self.progress.emit("⚠️ 视觉模型加载失败，使用回退模式")
                success = False

            # 2. 加载 LLM (+ RAG)
            self.progress.emit("正在加载语言模型...")
            status = self.llm_engine.load()
            self.progress.emit(f"✅ {status}")

            # 3. 加载 OCR
            if self.ocr_engine:
                self.progress.emit("正在加载 OCR 丝印识别引擎...")
                if self.ocr_engine.initialize():
                    self.progress.emit(f"✅ OCR 已就绪 ({self.ocr_engine.backend_name})")
                else:
                    self.progress.emit("⚠️ OCR 未加载 (芯片丝印识别不可用)")

        except Exception as e:
            self.progress.emit(f"❌ 模型加载错误: {e}")
            success = False

        self.finished.emit(success)


class LLMWorker(QThread):
    """
    异步 LLM 问答线程

    信号:
        response_ready(str):  AI 回复
        error(str):           错误信息
    """
    response_ready = Signal(str)
    error = Signal(str)

    def __init__(self, llm_engine, question: str, context: str = ""):
        super().__init__()
        self.llm_engine = llm_engine
        self.question = question
        self.context = context

    def run(self):
        try:
            answer = self.llm_engine.ask(self.question, self.context)
            self.response_ready.emit(answer)
        except Exception as e:
            self.error.emit(f"AI 回复错误: {e}")
