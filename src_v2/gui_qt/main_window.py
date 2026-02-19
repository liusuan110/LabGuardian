"""
LabGuardian 主窗口 (PySide6)
PyDracula 风格: 侧边栏 + 多页面路由 + 自定义标题栏

布局:
  ┌──────────────────────────────────────────────────┐
  │  自定义标题栏 (拖拽移动, 最小化/最大化/关闭)      │
  ├────────┬─────────────────────────────────────────┤
  │        │                                          │
  │ 侧边栏 │        内容区 (QStackedWidget)           │
  │        │   home:    Dashboard + 视频 (双栏)       │
  │  导航   │   video:   视频全屏                     │
  │        │   chat:    AI 聊天全屏                   │
  │        │   circuit: 电路验证工具                   │
  │        │   settings:设置页                        │
  │        │                                          │
  ├────────┴─────────────────────────────────────────┤
  │  状态栏 (系统状态 | 检测结果 | LLM 状态)          │
  └──────────────────────────────────────────────────┘
"""

import sys
import traceback
import numpy as np
import cv2
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QLabel, QFrame, QPushButton,
    QApplication, QFileDialog, QSplitter, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QPoint
from PySide6.QtGui import QFont, QPixmap, QIcon

# 确保 src_v2 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    gui as gui_cfg, vision as vision_cfg,
    camera as cam_cfg, circuit as circuit_cfg,
)
from vision.detector import ComponentDetector
from vision.calibrator import board_calibrator
from vision.stabilizer import DetectionStabilizer
from logic.circuit import CircuitAnalyzer, CircuitComponent
from logic.polarity import polarity_resolver
from logic.validator import validator
from logic.schematic import SchematicGenerator
from ai.llm_engine import LLMEngine
from ai.ocr_engine import OCREngine, OCR_TARGET_CLASSES

from .styles import GLOBAL_STYLE, ACCENT, BG_DARK, BG_MEDIUM, BORDER, TEXT, TEXT_DIM, SUCCESS, WARNING, DANGER
from .resources import Icons
from .sidebar import Sidebar
from .video_panel import VideoPanel
from .chat_panel import ChatPanel
from .dashboard import Dashboard
from .circuit_page import CircuitPage
from .settings_page import SettingsPage
from .workers import VideoWorker, ModelLoaderWorker, LLMWorker


# ============================================================
# 自定义标题栏
# ============================================================

class TitleBar(QFrame):
    """无边框窗口的自定义标题栏"""

    minimize_clicked = Signal()
    maximize_clicked = Signal()
    close_clicked    = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet(
            f"background-color: {BG_MEDIUM}; border-bottom: 1px solid {BORDER};"
        )
        self._drag_pos = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 4, 0)
        layout.setSpacing(0)

        # 应用图标+名称
        app_icon = QLabel(f"{Icons.APP}")
        app_icon.setFont(QFont("Segoe UI Emoji", 12))
        app_icon.setStyleSheet("background: transparent;")
        layout.addWidget(app_icon)

        app_name = QLabel(" LabGuardian")
        app_name.setStyleSheet(
            f"color: {TEXT}; font-size: 13px; font-weight: bold; background: transparent;"
        )
        layout.addWidget(app_name)

        layout.addStretch()

        # 窗口控制按钮
        btn_style = (
            "QPushButton {{ background: transparent; color: {color}; "
            "font-size: 14px; border: none; padding: 4px 12px; }}"
            "QPushButton:hover {{ background: {hover_bg}; }}"
        )

        btn_min = QPushButton(Icons.MINIMIZE)
        btn_min.setStyleSheet(btn_style.format(color=TEXT_DIM, hover_bg="#3a3f47"))
        btn_min.setFixedSize(46, 36)
        btn_min.clicked.connect(self.minimize_clicked.emit)
        layout.addWidget(btn_min)

        btn_max = QPushButton(Icons.MAXIMIZE)
        btn_max.setStyleSheet(btn_style.format(color=TEXT_DIM, hover_bg="#3a3f47"))
        btn_max.setFixedSize(46, 36)
        btn_max.clicked.connect(self.maximize_clicked.emit)
        layout.addWidget(btn_max)

        btn_close = QPushButton(Icons.CLOSE)
        btn_close.setStyleSheet(btn_style.format(color=TEXT_DIM, hover_bg=DANGER))
        btn_close.setFixedSize(46, 36)
        btn_close.clicked.connect(self.close_clicked.emit)
        layout.addWidget(btn_close)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.window().pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event):
        self.maximize_clicked.emit()


# ============================================================
# 主窗口
# ============================================================

class MainWindow(QMainWindow):
    """
    LabGuardian 主窗口

    架构:
      TitleBar
      ├── Sidebar (导航)
      └── QStackedWidget
            ├── page_home    (Dashboard + VideoPanel 双栏)
            ├── page_video   (VideoPanel 全屏)
            ├── page_chat    (ChatPanel 全屏)
            ├── page_circuit (CircuitPage)
            └── page_settings(SettingsPage)
    """

    def __init__(self):
        super().__init__()

        # 无边框窗口
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setMinimumSize(1200, 800)
        self.resize(gui_cfg.window_size[0], gui_cfg.window_size[1])
        self.setWindowTitle(gui_cfg.window_title)

        # 核心模块
        self.detector = ComponentDetector()
        self.stabilizer = DetectionStabilizer(window_size=5, min_hits=3)
        self.analyzer = CircuitAnalyzer()
        self.llm = LLMEngine()
        self.ocr = OCREngine()

        # OCR 结果缓存 (class_name -> chip_model, 避免每帧重复 OCR)
        self._ocr_cache: dict = {}      # {"IC_row15": "NE555"}
        self._ocr_frame_skip = 0        # 跳帧计数 (OCR 比较慢, 每 N 帧执行一次)
        self._ocr_interval = 30         # 每 30 帧执行一次 OCR

        # 状态
        self.current_detection = "None"
        self.ar_missing_links = []

        self._setup_ui()
        self._connect_signals()

        # 后台加载模型
        self._start_model_loading()

    # ================================================================
    # UI 搭建
    # ================================================================

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ---- 自定义标题栏 ----
        self._title_bar = TitleBar()
        root_layout.addWidget(self._title_bar)

        # ---- 主体区 (侧边栏 + 内容) ----
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # 侧边栏
        self._sidebar = Sidebar()
        body.addWidget(self._sidebar)

        # 内容页
        self._stack = QStackedWidget()
        body.addWidget(self._stack, stretch=1)

        root_layout.addLayout(body, stretch=1)

        # ---- 创建各页面 ----
        self._create_pages()

        # ---- 底部状态栏 ----
        self._statusbar = QFrame()
        self._statusbar.setFixedHeight(28)
        self._statusbar.setStyleSheet(
            f"background-color: {BG_MEDIUM}; border-top: 1px solid {BORDER};"
        )
        sb_layout = QHBoxLayout(self._statusbar)
        sb_layout.setContentsMargins(12, 0, 12, 0)
        sb_layout.setSpacing(16)

        self._sb_status = QLabel(f"{Icons.LOADING} 系统启动中...")
        self._sb_status.setStyleSheet(f"color: {WARNING}; font-size: 11px;")
        sb_layout.addWidget(self._sb_status)

        sb_layout.addStretch()

        self._sb_detect = QLabel("检测: --")
        self._sb_detect.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        sb_layout.addWidget(self._sb_detect)

        self._sb_llm = QLabel("LLM: --")
        self._sb_llm.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        sb_layout.addWidget(self._sb_llm)

        self._sb_fps = QLabel("FPS: --")
        self._sb_fps.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        sb_layout.addWidget(self._sb_fps)

        root_layout.addWidget(self._statusbar)

    def _create_pages(self):
        """创建全部页面并添加到 stack"""

        # ---- page_home: 仪表盘 + 视频 双栏 ----
        self._page_home = QWidget()
        home_layout = QHBoxLayout(self._page_home)
        home_layout.setContentsMargins(0, 0, 0, 0)
        home_layout.setSpacing(0)

        home_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧: 视频
        self._video_panel = VideoPanel()
        home_splitter.addWidget(self._video_panel)

        # 右侧: 仪表盘
        self._dashboard = Dashboard()
        home_splitter.addWidget(self._dashboard)

        home_splitter.setSizes([700, 400])
        home_splitter.setStretchFactor(0, 3)
        home_splitter.setStretchFactor(1, 2)
        home_layout.addWidget(home_splitter)

        self._stack.addWidget(self._page_home)

        # ---- page_video: 全屏视频 + 聊天侧栏 ----
        self._page_video = QWidget()
        video_layout = QHBoxLayout(self._page_video)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)

        video_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._video_panel_full = VideoPanel()
        video_splitter.addWidget(self._video_panel_full)

        self._chat_side = ChatPanel()
        video_splitter.addWidget(self._chat_side)
        video_splitter.setSizes([800, 350])
        video_splitter.setStretchFactor(0, 3)
        video_splitter.setStretchFactor(1, 1)
        video_layout.addWidget(video_splitter)

        self._stack.addWidget(self._page_video)

        # ---- page_chat: 聊天全屏 ----
        self._chat_panel = ChatPanel()
        self._stack.addWidget(self._chat_panel)

        # ---- page_circuit: 电路验证 ----
        self._circuit_page = CircuitPage()
        self._stack.addWidget(self._circuit_page)

        # ---- page_settings: 设置 ----
        self._settings_page = SettingsPage()
        self._stack.addWidget(self._settings_page)

        # 页面名→索引映射
        self._page_map = {
            "home":     0,
            "video":    1,
            "chat":     2,
            "circuit":  3,
            "settings": 4,
        }

    # ================================================================
    # 信号连接
    # ================================================================

    def _connect_signals(self):
        # 标题栏
        self._title_bar.minimize_clicked.connect(self.showMinimized)
        self._title_bar.maximize_clicked.connect(self._toggle_maximize)
        self._title_bar.close_clicked.connect(self.close)

        # 侧边栏导航
        self._sidebar.page_changed.connect(self._switch_page)

        # 视频面板信号
        for vp in [self._video_panel, self._video_panel_full]:
            vp.calibrate_requested.connect(self._start_calibration)
            vp.load_image_requested.connect(self._load_image)
            vp.conf_changed.connect(self._on_conf_changed)

        # 聊天信号
        self._chat_panel.message_sent.connect(self._ask_ai)
        self._chat_side.message_sent.connect(self._ask_ai)

        # 电路页面信号
        self._circuit_page.golden_ref_requested.connect(self._set_golden_ref)
        self._circuit_page.save_template_requested.connect(self._save_template)
        self._circuit_page.load_template_requested.connect(self._load_template)
        self._circuit_page.validate_requested.connect(self._validate_circuit)
        self._circuit_page.show_netlist_requested.connect(self._show_netlist)
        self._circuit_page.draw_schematic_requested.connect(self._draw_schematic)
        self._circuit_page.reset_requested.connect(self._reset_analyzer)

    # ================================================================
    # 页面切换
    # ================================================================

    @Slot(str)
    def _switch_page(self, page_name: str):
        idx = self._page_map.get(page_name, 0)
        self._stack.setCurrentIndex(idx)

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # ================================================================
    # 模型加载
    # ================================================================

    def _start_model_loading(self):
        self._model_loader = ModelLoaderWorker(self.detector, self.llm, self.ocr)
        self._model_loader.progress.connect(self._on_model_progress)
        self._model_loader.finished.connect(self._on_model_loaded)
        self._model_loader.start()

    @Slot(str)
    def _on_model_progress(self, msg: str):
        self._dashboard.add_log(msg)
        self._chat_panel.log(msg)
        self._chat_side.log(msg)
        self._sb_status.setText(f"{Icons.LOADING} {msg}")

    @Slot(bool)
    def _on_model_loaded(self, success: bool):
        if success:
            self._sb_status.setText(f"{Icons.OK} 系统就绪")
            self._sb_status.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")
            self._dashboard.update_system_status("就绪", SUCCESS)
            self._dashboard.update_module_status("vision", True, "YOLO-OBB 已加载")
            self._dashboard.update_module_status("polarity", True, "极性引擎就绪")
        else:
            self._sb_status.setText(f"{Icons.WARN} 部分模块加载失败")
            self._sb_status.setStyleSheet(f"color: {WARNING}; font-size: 11px;")
            self._dashboard.update_system_status("部分就绪", WARNING)

        # 检测 LLM 状态
        if self.llm._active:
            backend_name = self.llm.backend_name
            self._dashboard.update_module_status("llm", True, backend_name)
            self._sb_llm.setText(f"LLM: {backend_name}")
            self._sb_llm.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")
            # 更新聊天面板状态
            self._chat_panel.set_ai_status(f"{Icons.OK} {backend_name}", SUCCESS)
            self._chat_side.set_ai_status(f"{Icons.OK} {backend_name}", SUCCESS)

        # OCR 状态
        if self.ocr.is_ready:
            self._dashboard.update_module_status("ocr", True, f"OCR: {self.ocr.backend_name}")
            self._dashboard.add_log(f"✅ OCR 丝印识别引擎: {self.ocr.backend_name}")
        else:
            self._dashboard.update_module_status("ocr", False, "OCR: 未加载")

        # RAG 状态
        if self.llm.rag_ready:
            self._dashboard.update_module_status("rag", True, f"RAG: {self.llm.rag.doc_count} 知识块")

        # 启动视频线程
        self._start_video()

    # ================================================================
    # 视频管线
    # ================================================================

    def _start_video(self):
        self._video_worker = VideoWorker(
            device_id=cam_cfg.device_id,
            backend=cam_cfg.cv2_backend,
        )
        self._video_worker.set_process_callback(self._process_frame)

        # 连接帧信号到两个视频面板
        self._video_worker.frame_ready.connect(self._video_panel.update_frame)
        self._video_worker.frame_ready.connect(self._video_panel_full.update_frame)
        self._video_worker.fps_updated.connect(self._on_fps_updated)
        self._video_worker.error.connect(self._on_video_error)

        self._video_worker.start()
        self._dashboard.add_log("视频流已启动")

    @Slot(float)
    def _on_fps_updated(self, fps: float):
        self._video_panel.update_fps(fps)
        self._video_panel_full.update_fps(fps)
        self._dashboard.update_fps(fps)
        self._sb_fps.setText(f"FPS: {fps:.1f}")
        color = SUCCESS if fps >= 15 else (WARNING if fps >= 8 else DANGER)
        self._sb_fps.setStyleSheet(f"color: {color}; font-size: 11px;")

    @Slot(str)
    def _on_video_error(self, msg: str):
        self._dashboard.add_log(f"⚠️ {msg}")

    def _on_conf_changed(self, value: float):
        """置信度阈值变更"""
        vision_cfg.conf_threshold = value

    # ================================================================
    # 帧处理管线 (核心逻辑 — 从旧 app.py 迁移)
    # ================================================================

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        每帧回调: 检测 → 稳定化 → 坐标映射 → 电路分析 → 标注
        在 VideoWorker 线程中执行
        """
        if not self.detector.model:
            return frame

        conf = vision_cfg.conf_threshold

        # 1. YOLO 检测
        detections = self.detector.detect(frame, conf=conf)

        # 2. 多帧稳定化
        stable_dets = self.stabilizer.update(detections)

        # 3. 绘制检测结果
        annotated = self.detector.annotate_frame(frame, stable_dets)

        # 4. 坐标映射 + 电路分析
        if board_calibrator.is_calibrated and stable_dets:
            self.analyzer.reset()

            for det in stable_dets:
                if det.pin1_pixel and det.pin2_pixel:
                    loc1 = board_calibrator.frame_pixel_to_logic(*det.pin1_pixel)
                    loc2 = board_calibrator.frame_pixel_to_logic(*det.pin2_pixel)

                    if (loc1 and loc2 and
                            loc1[0] != "Groove" and loc2[0] != "Groove"):
                        comp = CircuitComponent(
                            name=det.class_name,
                            type=det.class_name,
                            pin1_loc=loc1,
                            pin2_loc=loc2,
                            confidence=det.confidence,
                        )

                        obb_corners = det.obb_corners if det.is_obb else None
                        polarity_resolver.enrich(
                            comp,
                            obb_corners=obb_corners,
                            orientation_deg=0.0,
                        )

                        self.analyzer.add_component(comp)

                        info = f"{loc1[0]}-{loc2[0]}"
                        x1, y1 = det.bbox[:2]
                        cv2.putText(annotated, info, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 网络数
            try:
                net_count = self.analyzer.get_net_count()
                cv2.putText(annotated, f"Nets: {net_count}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                # 更新 UI (跨线程安全通过信号)
            except Exception:
                pass

            # 幽灵线
            self._draw_ghost_wires(annotated)

        # 5. OCR 芯片丝印识别 (每 N 帧执行一次, 避免性能瓶颈)
        self._ocr_frame_skip += 1
        if self.ocr.is_ready and stable_dets and self._ocr_frame_skip >= self._ocr_interval:
            self._ocr_frame_skip = 0
            self._run_ocr_on_detections(frame, stable_dets, annotated)

        # 6. 在帧上显示已识别的芯片型号
        self._draw_ocr_labels(annotated, stable_dets)

        # 7. 更新检测状态
        comp_count = len(stable_dets) if stable_dets else 0
        if stable_dets:
            top = max(stable_dets, key=lambda d: d.confidence)
            self.current_detection = top.class_name
        else:
            self.current_detection = "None"

        return annotated

    def _run_ocr_on_detections(self, frame: np.ndarray, detections: list,
                                annotated: np.ndarray):
        """对芯片类检测结果执行 OCR 识别丝印，新识别出型号时自动查询 RAG"""
        target_classes = {c.upper() for c in OCR_TARGET_CLASSES}
        for det in detections:
            if det.class_name.upper() not in target_classes:
                continue
            # 使用 bbox 中心作为缓存 key (避免重复识别同一芯片)
            cx = (det.bbox[0] + det.bbox[2]) // 2
            cy = (det.bbox[1] + det.bbox[3]) // 2
            cache_key = f"{cx//50}_{cy//50}"  # 量化坐标做 key
            if cache_key in self._ocr_cache:
                continue
            try:
                result = self.ocr.recognize_chip(frame, det.bbox)
                if result.chip_model:
                    self._ocr_cache[cache_key] = result.chip_model
                    logger.info(f"[OCR] {det.class_name} → {result.chip_model}")
                    # --- OCR+RAG 自动知识检索 ---
                    self._auto_rag_lookup(result.chip_model)
            except Exception as e:
                logger.warning(f"[OCR] 识别出错: {e}")

    def _auto_rag_lookup(self, chip_model: str):
        """当 OCR 识别出新芯片型号时，自动查询 RAG 知识库并显示摘要"""
        # 避免重复查询同一个型号
        if not hasattr(self, '_rag_queried_models'):
            self._rag_queried_models: set = set()
        if chip_model.upper() in self._rag_queried_models:
            return
        self._rag_queried_models.add(chip_model.upper())

        # 需要 LLM 引擎的 RAG 就绪
        if not self.llm.rag_ready:
            self._log_all(f"🔍 识别到芯片: {chip_model} (RAG 未就绪, 跳过知识检索)")
            return

        try:
            # 查询 RAG 获取芯片简要信息
            query = f"{chip_model} 引脚定义 功能 使用方法"
            results = self.llm.rag.query(query, top_k=2, min_score=0.3)
            if results:
                # 取最相关片段并截取摘要
                best = results[0]
                snippet = best["text"][:150].replace("\n", " ").strip()
                if len(best["text"]) > 150:
                    snippet += "..."
                info_msg = f"🔍 识别到 [{chip_model}] — {snippet}"
                self._log_all(info_msg)
                # 在聊天面板以系统消息形式推送
                self._chat_panel.add_message(
                    f"📦 **自动识别**: 检测到芯片 **{chip_model}**\n"
                    f"> {snippet}\n\n"
                    f"💡 输入 `{chip_model} 引脚` 可查看详细引脚定义",
                    "system"
                )
                self._chat_side.add_message(
                    f"📦 识别到芯片 {chip_model}: {snippet}",
                    "system"
                )
                logger.info(f"[RAG] 自动检索 {chip_model}: 相关度 {best['score']:.1%}")
            else:
                self._log_all(f"🔍 识别到芯片: {chip_model} (知识库暂无相关信息)")
        except Exception as e:
            logger.warning(f"[RAG] 自动检索 {chip_model} 出错: {e}")

    def _draw_ocr_labels(self, frame: np.ndarray, detections: list):
        """在帧上绘制已识别的芯片型号标签"""
        if not self._ocr_cache:
            return
        target_classes = {c.upper() for c in OCR_TARGET_CLASSES}
        for det in detections:
            if det.class_name.upper() not in target_classes:
                continue
            cx = (det.bbox[0] + det.bbox[2]) // 2
            cy = (det.bbox[1] + det.bbox[3]) // 2
            cache_key = f"{cx//50}_{cy//50}"
            model_name = self._ocr_cache.get(cache_key)
            if model_name:
                # 在检测框下方显示芯片型号 (青色标签)
                x1, y2 = det.bbox[0], det.bbox[3]
                label = f"[{model_name}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y2 + 2), (x1 + tw + 4, y2 + th + 8),
                              (128, 64, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y2 + th + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _draw_ghost_wires(self, frame: np.ndarray):
        """绘制缺失连接的幽灵线引导"""
        if not self.ar_missing_links:
            return
        for pin1_loc, pin2_loc in self.ar_missing_links:
            try:
                p1 = board_calibrator.logic_to_frame_pixel(pin1_loc[0], pin1_loc[1])
                p2 = board_calibrator.logic_to_frame_pixel(pin2_loc[0], pin2_loc[1])
                if p1 and p2:
                    cv2.arrowedLine(frame, p1, p2, (0, 255, 255), 3, tipLength=0.2)
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    cv2.putText(frame, "MISSING", mid,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            except Exception:
                pass

    # ================================================================
    # 用户操作回调
    # ================================================================

    def _start_calibration(self):
        """校准 (TODO: 迁移到 PySide6 对话框)"""
        self._dashboard.add_log("启动校准...")
        self._log_all("校准: 请在弹出窗口中点击面包板4个角点")

        # 暂时仍用 OpenCV 窗口 (后续迁移到 QDialog)
        import threading
        threading.Thread(target=self._calibration_flow, daemon=True).start()

    def _calibration_flow(self):
        """校准交互 (在 OpenCV 窗口完成)"""
        if self._video_worker._source_mode == "image" and self._video_worker.static_frame is not None:
            frame = self._video_worker.static_frame.copy()
        else:
            cap = cv2.VideoCapture(cam_cfg.device_id)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self._log_all("❌ 无法获取帧用于校准")
                return

        points = []
        win_name = "Calibrate: Click 4 corners (TL->TR->BR->BL)"

        h, w = frame.shape[:2]
        max_w, max_h = 1000, 700
        scale = min(max_w / w, max_h / h, 1.0)
        disp = cv2.resize(frame, (int(w * scale), int(h * scale)))

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                real_x, real_y = int(x / scale), int(y / scale)
                points.append([real_x, real_y])

        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, on_click)

        while len(points) < 4:
            draw = disp.copy()
            for i, p in enumerate(points):
                sx, sy = int(p[0] * scale), int(p[1] * scale)
                cv2.circle(draw, (sx, sy), 5, (0, 0, 255), -1)
                cv2.putText(draw, str(i + 1), (sx + 10, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(win_name, draw)
            if cv2.waitKey(50) == ord('q'):
                cv2.destroyWindow(win_name)
                return

        cv2.destroyWindow(win_name)

        src_pts = np.float32(points)
        board_calibrator.calibrate(src_pts)

        warped = board_calibrator.warp(frame)
        hole_count = board_calibrator.detect_holes(warped)
        msg = f"✅ 校准完成，检测到 {hole_count} 个孔洞"
        self._log_all(msg)
        self._dashboard.update_module_status("calibr", True, f"{hole_count} 孔洞")

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择电路图片", "",
            "图片 (*.jpg *.png *.jpeg *.bmp)"
        )
        if path:
            if self._video_worker.load_image(path):
                name = Path(path).name
                self._log_all(f"已加载: {name}")
                self._dashboard.add_log(f"加载测试图片: {name}")

    def _show_netlist(self):
        netlist = self.analyzer.get_circuit_description()
        self._circuit_page.set_result(netlist)
        self._log_all("已生成网表")

    def _draw_schematic(self):
        try:
            gen = SchematicGenerator(self.analyzer)
            gen.generate_schematic(show=True)
            self._log_all("✅ 原理图绘制完成")
        except Exception as e:
            self._log_all(f"原理图错误: {e}")

    def _reset_analyzer(self):
        self.analyzer.reset()
        self.stabilizer.clear()
        self._log_all("🔄 分析器已重置")

    def _set_golden_ref(self):
        if self.analyzer.components:
            validator.set_reference(self.analyzer)
            count = len(self.analyzer.components)
            self._log_all(f"⭐ 已设为金标准 ({count} 个元件)")
            self._circuit_page.set_template_info(f"当前金标准: {count} 个元件")
        else:
            self._log_all("⚠️ 未检测到元件，无法设为金标准")

    def _save_template(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存电路模板", "",
            "LabGuardian 模板 (*.json)"
        )
        if path:
            validator.save_reference(path)
            self._log_all(f"💾 模板已保存: {Path(path).name}")

    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载电路模板", "",
            "LabGuardian 模板 (*.json)"
        )
        if path:
            validator.load_reference(path)
            name = Path(path).name
            self._log_all(f"📂 模板已加载: {name}")
            self._circuit_page.set_template_info(f"已加载: {name}")

    def _validate_circuit(self):
        self._circuit_page.clear_result()
        self._log_all("正在验证电路...")

        results = validator.compare(self.analyzer)

        output = "--- 验证报告 ---\n"

        # v3: 显示相似度和进度
        similarity = results.get('similarity', 0)
        progress = results.get('progress', 0)
        if similarity > 0:
            output += f"📏 电路相似度: {similarity:.0%}\n"
        if 0 < progress < 1.0:
            output += f"📊 搭建进度: {progress:.0%}\n"

        for msg in results.get('errors', []):
            output += f"{msg}\n"
            self._circuit_page.append_result(msg)

        # v3: 显示极性错误
        for pol_err in results.get('polarity_errors', []):
            output += f"{pol_err}\n"

        self.ar_missing_links = results.get('missing_links', [])
        if self.ar_missing_links:
            output += f"\n⚠️ 缺失连接: {len(self.ar_missing_links)} 处 (已在视频中标注)"

        # v3: 显示缺失/多余元件摘要
        missing_c = results.get('missing_components', [])
        extra_c = results.get('extra_components', [])
        if missing_c:
            output += f"\n📋 待搭建元件: {', '.join(missing_c)}"
        if extra_c:
            output += f"\n📋 多余元件: {', '.join(extra_c)}"

        self._circuit_page.set_result(output)
        self._log_all(f"验证完成: {len(results.get('errors', []))} 个问题")

    # ================================================================
    # AI 问答
    # ================================================================

    @Slot(str)
    def _ask_ai(self, question: str):
        """处理 AI 问答请求"""
        if not question:
            return

        # 更新状态
        self._chat_panel.set_ai_status(f"{Icons.LOADING} 思考中...", WARNING)
        self._chat_side.set_ai_status(f"{Icons.LOADING} 思考中...", WARNING)

        context = self.analyzer.get_circuit_description()

        self._llm_worker = LLMWorker(self.llm, question, context)
        self._llm_worker.response_ready.connect(self._on_ai_response)
        self._llm_worker.error.connect(self._on_ai_error)
        self._llm_worker.start()

    @Slot(str)
    def _on_ai_response(self, answer: str):
        self._chat_panel.add_message(answer, "ai")
        self._chat_side.add_message(answer, "ai")
        self._chat_panel.set_ai_status(f"{Icons.OK} 就绪", SUCCESS)
        self._chat_side.set_ai_status(f"{Icons.OK} 就绪", SUCCESS)

    @Slot(str)
    def _on_ai_error(self, error: str):
        self._chat_panel.add_message(f"错误: {error}", "system")
        self._chat_side.add_message(f"错误: {error}", "system")
        self._chat_panel.set_ai_status(f"{Icons.ERROR} 错误", DANGER)
        self._chat_side.set_ai_status(f"{Icons.ERROR} 错误", DANGER)

    # ================================================================
    # 工具方法
    # ================================================================

    def _log_all(self, text: str):
        """向所有日志面板广播消息"""
        self._dashboard.add_log(text)
        self._chat_panel.log(text)
        self._chat_side.log(text)

    # ================================================================
    # 生命周期
    # ================================================================

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        if hasattr(self, '_video_worker'):
            self._video_worker.stop()
        event.accept()
