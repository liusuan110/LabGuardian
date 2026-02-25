"""
QThread 工作线程
职责: 将耗时操作从主线程解耦, 通过信号驱动 UI 更新

线程:
  - VideoWorker:      摄像头采集 → 帧处理 → 发送 QPixmap
  - ModelLoader:      后台加载 YOLO / LLM 模型
  - LLMWorker:        异步 LLM 问答
  - HeartbeatWorker:  课堂模式心跳上报 + 教师指导接收
"""

import cv2
import time
import traceback
import logging
import base64
import json
import threading
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker, Slot
from PySide6.QtGui import QImage, QPixmap
from pathlib import Path

logger = logging.getLogger(__name__)


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


class HeartbeatWorker(QThread):
    """
    课堂模式心跳工作线程

    职责:
      1. 每 N 秒从 AppContext 采集电路状态, POST 到教师服务器
      2. 维护 WebSocket 连接, 接收教师指导消息
      3. 通过信号通知主线程显示 Toast

    信号:
      guidance_received(str, str, str):  (type, message, sender) 教师指导
      connection_status(bool):           服务器连接状态
    """
    guidance_received = Signal(str, str, str)   # (type, message, sender)
    connection_status = Signal(bool)            # True=connected

    def __init__(self, ctx, station_id: str, server_url: str,
                 interval: float = 2.0, thumbnail_size: tuple = (160, 120),
                 thumbnail_quality: int = 70, student_name: str = ""):
        super().__init__()
        self.ctx = ctx
        self.station_id = station_id
        self.student_name = student_name
        self.server_url = server_url.rstrip("/")
        self.interval = interval
        self.thumb_size = thumbnail_size
        self.thumb_quality = thumbnail_quality
        self._running = True

        # 最新帧 (由主线程写入)
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_fps = 0.0

    def update_frame(self, frame: np.ndarray):
        """由主线程调用, 更新最新帧 (用于生成缩略图)"""
        with self._frame_lock:
            self._latest_frame = frame

    def update_fps(self, fps: float):
        """由主线程调用, 更新 FPS"""
        self._latest_fps = fps

    def stop(self):
        self._running = False
        self.wait(5000)

    def run(self):
        """主循环: 心跳上报 + WebSocket 监听"""
        # 延迟导入 (requests 是轻量依赖)
        try:
            import requests
        except ImportError:
            logger.error("[Heartbeat] requests 未安装, 心跳上报不可用")
            return

        # 启动 WebSocket 监听线程
        ws_thread = threading.Thread(
            target=self._ws_listener,
            daemon=True,
            name="HeartbeatWS",
        )
        ws_thread.start()

        while self._running:
            try:
                payload = self._build_heartbeat()
                resp = requests.post(
                    f"{self.server_url}/api/heartbeat",
                    json=payload,
                    timeout=2,
                )
                if resp.status_code == 200:
                    self.connection_status.emit(True)
                else:
                    self.connection_status.emit(False)
            except Exception as e:
                logger.debug(f"[Heartbeat] 上报失败: {e}")
                self.connection_status.emit(False)

            time.sleep(self.interval)

    def _build_heartbeat(self) -> dict:
        """从 AppContext 采集数据, 构建心跳包"""
        # 延迟导入 shared 模块
        import sys as _sys
        _lab_root = str(Path(__file__).resolve().parent.parent.parent)
        if _lab_root not in _sys.path:
            _sys.path.insert(0, _lab_root)

        from shared.models import StationHeartbeat, ComponentInfo
        from shared.risk import classify_risk

        ctx = self.ctx

        # ---- 从 AppContext 安全读取 ----
        components = []
        component_count = 0
        net_count = 0
        circuit_snapshot = ""
        diagnostics = []
        progress = 0.0
        similarity = 0.0
        missing_components = []
        detector_ok = False
        llm_backend = ""
        ocr_backend = ""

        try:
            with ctx.read_lock():
                component_count = len(ctx.analyzer.components)
                for comp in ctx.analyzer.components:
                    components.append(ComponentInfo(
                        name=comp.name,
                        type=comp.type,
                        polarity=comp.polarity.value if hasattr(comp.polarity, 'value') else str(comp.polarity),
                        pin1=list(comp.pin1_loc) if comp.pin1_loc else [],
                        pin2=list(comp.pin2_loc) if comp.pin2_loc else [],
                        pin3=list(comp.pin3_loc) if comp.pin3_loc else [],
                        confidence=comp.confidence,
                    ).model_dump())

                # 网络数
                import networkx as _nx
                net_count = _nx.number_connected_components(ctx.analyzer.graph)

            # 电路快照 (无需持有 lock, 使用 snapshot)
            circuit_snapshot = ctx.get_circuit_snapshot()[:500]

            # 诊断
            try:
                from logic.validator import CircuitValidator
                with ctx.read_lock():
                    diagnostics = CircuitValidator.diagnose(ctx.analyzer)
            except Exception:
                pass

            # 验证进度
            try:
                if ctx.validator.has_reference:
                    with ctx.read_lock():
                        result = ctx.validator.compare(ctx.analyzer)
                        progress = result.get("progress", 0.0)
                        similarity = result.get("similarity", 0.0)
                        missing_components = result.get("missing_components", [])
            except Exception:
                pass

            # 系统状态
            detector_ok = ctx.detector.model is not None
            llm_backend = getattr(ctx.llm, 'current_backend', '')
            ocr_backend = getattr(ctx.ocr, 'backend_name', '')

        except Exception as e:
            logger.debug(f"[Heartbeat] 数据采集异常: {e}")

        # 风险分级
        risk_level_enum, risk_reasons = classify_risk(diagnostics)

        # 缩略图
        thumbnail_b64 = ""
        with self._frame_lock:
            frame = self._latest_frame
        if frame is not None:
            try:
                thumb = cv2.resize(frame, self.thumb_size)
                _, buf = cv2.imencode(
                    '.jpg', thumb,
                    [cv2.IMWRITE_JPEG_QUALITY, self.thumb_quality]
                )
                thumbnail_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
            except Exception:
                pass

        heartbeat = StationHeartbeat(
            station_id=self.station_id,
            student_name=self.student_name,
            component_count=component_count,
            net_count=net_count,
            components=components,
            progress=progress,
            similarity=similarity,
            missing_components=missing_components,
            diagnostics=diagnostics,
            risk_level=risk_level_enum.value,
            risk_reasons=risk_reasons,
            circuit_snapshot=circuit_snapshot,
            fps=self._latest_fps,
            detector_ok=detector_ok,
            llm_backend=llm_backend,
            ocr_backend=ocr_backend,
            thumbnail_b64=thumbnail_b64,
        )
        return heartbeat.model_dump()

    def _ws_listener(self):
        """WebSocket 监听线程: 接收教师指导消息"""
        try:
            import websocket
        except ImportError:
            # websocket-client 未安装, 回退到轮询模式
            logger.info("[Heartbeat] websocket-client 未安装, 跳过 WS 监听")
            return

        ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/station/{self.station_id}"

        while self._running:
            try:
                ws = websocket.WebSocket()
                ws.settimeout(5)
                ws.connect(ws_url)
                logger.info(f"[Heartbeat] WebSocket 已连接: {ws_url}")

                while self._running:
                    try:
                        msg = ws.recv()
                        if msg:
                            data = json.loads(msg)
                            msg_type = data.get("type", "hint")
                            message = data.get("message", "")
                            sender = data.get("sender", "Teacher")
                            self.guidance_received.emit(msg_type, message, sender)
                    except websocket.WebSocketTimeoutException:
                        # 超时无消息, 发送 ping
                        try:
                            ws.send("ping")
                        except Exception:
                            break
                    except Exception:
                        break

                ws.close()
            except Exception as e:
                logger.debug(f"[Heartbeat] WebSocket 连接失败: {e}")

            # 重连等待
            if self._running:
                time.sleep(5)
