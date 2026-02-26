"""
LabGuardian 主窗口 (PySide6)
PyDracula 风格: 侧边栏 + 多页面路由 + 自定义标题栏

架构 (v4 — 模块拆分):
  - 帧处理管线提取至 frame_pipeline.FramePipeline (纯计算, 无 Qt 依赖)
  - 校准交互提取至 calibration.CalibrationHelper
  - MainWindow 仅负责 UI 编排 / 信号连接 / 用户操作回调

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
import logging
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
    camera as cam_cfg,
    classroom as classroom_cfg,
)
from app_context import AppContext

from .styles import GLOBAL_STYLE, ACCENT, BG_DARK, BG_MEDIUM, BORDER, TEXT, TEXT_DIM, SUCCESS, WARNING, DANGER
from .resources import Icons
from .sidebar import Sidebar
from .video_panel import VideoPanel
from .chat_panel import ChatPanel
from .dashboard import Dashboard
from .circuit_page import CircuitPage
from .settings_page import SettingsPage
from .workers import VideoWorker, ModelLoaderWorker, LLMWorker
from .frame_pipeline import FramePipeline
from .calibration import CalibrationHelper

logger = logging.getLogger(__name__)


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

    架构 (v4 — 模块拆分):
      TitleBar
      ├── Sidebar (导航)
      └── QStackedWidget
            ├── page_home    (Dashboard + VideoPanel 双栏)
            ├── page_video   (VideoPanel 全屏)
            ├── page_chat    (ChatPanel 全屏)
            ├── page_circuit (CircuitPage)
            └── page_settings(SettingsPage)

    职责划分:
      - FramePipeline:      帧处理 / OCR / 坐标映射 / 电路分析 (纯计算)
      - CalibrationHelper:  面包板校准交互 (OpenCV 窗口 + 自动检测)
      - MainWindow:         UI 编排 / 信号连接 / 用户操作回调
    """

    def __init__(self, ctx: AppContext = None):
        super().__init__()

        # ---- AppContext (核心服务注册中心) ----
        self.ctx = ctx if ctx is not None else AppContext()

        # 无边框窗口
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setMinimumSize(1200, 800)
        self.resize(gui_cfg.window_size[0], gui_cfg.window_size[1])
        self.setWindowTitle(gui_cfg.window_title)

        # ---- 帧处理管线 + 校准辅助 ----
        self._pipeline = FramePipeline(self.ctx)
        self._calibration = CalibrationHelper(self.ctx)

        self._setup_ui()
        self._connect_signals()
        self._connect_pipeline_callbacks()

        # 后台加载模型
        self._start_model_loading()

    # ================================================================
    # 管线回调连接
    # ================================================================

    def _connect_pipeline_callbacks(self):
        """连接 FramePipeline / CalibrationHelper 的回调到 UI"""
        self._pipeline.on_log = self._log_all
        self._pipeline.on_rag_result = self._on_rag_result
        self._pipeline.on_frame_processed = self._on_pipeline_frame

        self._calibration.on_log = self._log_all
        self._calibration.on_status = self._dashboard.update_module_status

    def _on_rag_result(self, chip_model: str, detail: str, short: str):
        """FramePipeline OCR+RAG 回调 → 推送到聊天面板"""
        self._chat_panel.add_message(detail, "system")
        self._chat_side.add_message(short, "system")

    def _on_pipeline_frame(self, annotated):
        """FramePipeline 处理完帧的回调 → 喂给心跳线程"""
        if hasattr(self, '_heartbeat_worker'):
            self._heartbeat_worker.update_frame(annotated)

    @property
    def current_detection(self):
        return self._pipeline.current_detection

    @current_detection.setter
    def current_detection(self, value):
        self._pipeline.current_detection = value

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

        # 页面名->索引映射
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
        self._circuit_page.reset_requested.connect(self._reset_analyzer)

        # ---- 课堂模式: 心跳上报 + 教师指导接收 ----
        if classroom_cfg.enabled:
            self._init_classroom()

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
        self._model_loader = ModelLoaderWorker(
            self.ctx.detector, self.ctx.llm, self.ctx.ocr
        )
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
        if self.ctx.llm._active:
            backend_name = self.ctx.llm.backend_name
            self._dashboard.update_module_status("llm", True, backend_name)
            self._sb_llm.setText(f"LLM: {backend_name}")
            self._sb_llm.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")
            # 更新聊天面板状态
            self._chat_panel.set_ai_status(f"{Icons.OK} {backend_name}", SUCCESS)
            self._chat_side.set_ai_status(f"{Icons.OK} {backend_name}", SUCCESS)

        # OCR 状态
        if self.ctx.ocr.is_ready:
            self._dashboard.update_module_status("ocr", True, f"OCR: {self.ctx.ocr.backend_name}")
            self._dashboard.add_log(f"OCR 丝印识别引擎: {self.ctx.ocr.backend_name}")
        else:
            self._dashboard.update_module_status("ocr", False, "OCR: 未加载")

        # RAG 状态
        if self.ctx.llm.rag_ready:
            self._dashboard.update_module_status("rag", True, f"RAG: {self.ctx.llm.rag.doc_count} 知识块")

        # 启动视频线程
        self._start_video()

    # ================================================================
    # 课堂模式 (心跳上报 + 教师指导接收)
    # ================================================================

    def _init_classroom(self):
        """初始化课堂模式: 启动心跳线程 + Toast 通知"""
        from .workers import HeartbeatWorker
        from .toast import ToastNotification

        # Toast 通知 (挂在 centralWidget 上, 顶部滑入)
        self._toast = ToastNotification(self.centralWidget())
        self._toast.setFixedWidth(500)

        # 心跳工作线程
        self._heartbeat_worker = HeartbeatWorker(
            ctx=self.ctx,
            station_id=classroom_cfg.station_id,
            server_url=classroom_cfg.server_url,
            interval=classroom_cfg.heartbeat_interval,
            thumbnail_size=classroom_cfg.thumbnail_size,
            thumbnail_quality=classroom_cfg.thumbnail_quality,
            student_name=classroom_cfg.student_name,
        )

        # 教师指导 → Toast 弹窗
        self._heartbeat_worker.guidance_received.connect(self._on_guidance_received)
        # 连接状态 → 状态栏
        self._heartbeat_worker.connection_status.connect(self._on_classroom_connection)

        self._heartbeat_worker.start()
        self._dashboard.add_log(f"课堂模式就绪: 工位 {classroom_cfg.station_id}")
        logger.info(f"[Classroom] HeartbeatWorker 已启动, station={classroom_cfg.station_id}")

    @Slot(str, str, str)
    def _on_guidance_received(self, msg_type: str, message: str, sender: str):
        """教师指导消息 → Toast 弹窗 + 聊天面板"""
        if hasattr(self, '_toast'):
            self._toast.show_message(msg_type, message, sender)

        # 同时显示在聊天面板
        prefix = {"hint": "💡", "warning": "⚠️", "approve": "✅"}.get(msg_type, "📢")
        self._chat_panel.log(f"{prefix} [{sender}] {message}")
        self._chat_side.log(f"{prefix} [{sender}] {message}")
        self._dashboard.add_log(f"教师指导: {message[:50]}")

    @Slot(bool)
    def _on_classroom_connection(self, connected: bool):
        """课堂服务器连接状态更新"""
        if connected:
            self._sb_status.setText(f"☁ 课堂已连接")
            self._sb_status.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")
        # 不在断开时覆盖状态 (避免频繁闪烁)

    # ================================================================
    # 视频管线
    # ================================================================

    def _start_video(self):
        self._video_worker = VideoWorker(
            device_id=cam_cfg.device_id,
            backend=cam_cfg.cv2_backend,
        )
        self._video_worker.set_process_callback(self._pipeline.process_frame)

        # 连接帧信号到两个视频面板
        self._video_worker.frame_ready.connect(self._video_panel.update_frame)
        self._video_worker.frame_ready.connect(self._video_panel_full.update_frame)
        self._video_worker.fps_updated.connect(self._on_fps_updated)
        self._video_worker.error.connect(self._on_video_error)

        self._video_worker.start()
        self._dashboard.add_log("视频流已启动")

        # 如果课堂模式, 把 FPS 喂给心跳线程
        if hasattr(self, '_heartbeat_worker'):
            self._video_worker.fps_updated.connect(self._heartbeat_worker.update_fps)

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
        self._dashboard.add_log(f"[Video] {msg}")

    def _on_conf_changed(self, value: float):
        """置信度阈值变更"""
        vision_cfg.conf_threshold = value

    # ================================================================
    # 用户操作回调 (校准 / 图片加载)
    # ================================================================

    def _start_calibration(self):
        """校准 — 委托给 CalibrationHelper"""
        self._dashboard.add_log("启动校准...")
        self._calibration.start_calibration(self._video_worker)

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
                # 自动检测面包板并校准
                frame = self._video_worker.static_frame
                if frame is not None:
                    self._calibration.auto_detect_board(frame)

    # ================================================================
    # 电路操作 (主线程, 需要 read_lock)
    # ================================================================

    def _show_netlist(self):
        """显示网表 (主线程, read_lock 保护)"""
        with self.ctx.read_lock():
            netlist = self.ctx.analyzer.get_circuit_description()
        self._circuit_page.set_result(netlist)
        self._log_all("已生成网表")

    def _reset_analyzer(self):
        """重置分析器 (主线程, 通过 AppContext 线程安全方法)"""
        self.ctx.reset_analysis()
        self._log_all("分析器已重置")

    def _set_golden_ref(self):
        """设置金标准 (主线程, read_lock 保护)"""
        with self.ctx.read_lock():
            if self.ctx.analyzer.components:
                self.ctx.validator.set_reference(self.ctx.analyzer)
                count = len(self.ctx.analyzer.components)
                self._log_all(f"已设为金标准 ({count} 个元件)")
                self._circuit_page.set_template_info(f"当前金标准: {count} 个元件")
            else:
                self._log_all("未检测到元件, 无法设为金标准")

    def _save_template(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存电路模板", "",
            "LabGuardian 模板 (*.json)"
        )
        if path:
            self.ctx.validator.save_reference(path)
            self._log_all(f"模板已保存: {Path(path).name}")

    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载电路模板", "",
            "LabGuardian 模板 (*.json)"
        )
        if path:
            self.ctx.validator.load_reference(path)
            name = Path(path).name
            self._log_all(f"模板已加载: {name}")
            self._circuit_page.set_template_info(f"已加载: {name}")

    def _validate_circuit(self):
        """验证电路 (主线程, read_lock 保护)"""
        self._circuit_page.clear_result()
        self._log_all("正在验证电路...")

        with self.ctx.read_lock():
            results = self.ctx.validator.compare(self.ctx.analyzer)

        output = "--- 验证报告 ---\n"

        # 显示相似度和进度
        similarity = results.get('similarity', 0)
        progress = results.get('progress', 0)
        if similarity > 0:
            output += f"电路相似度: {similarity:.0%}\n"
        if 0 < progress < 1.0:
            output += f"搭建进度: {progress:.0%}\n"

        for msg in results.get('errors', []):
            output += f"{msg}\n"
            self._circuit_page.append_result(msg)

        # 显示极性错误
        for pol_err in results.get('polarity_errors', []):
            output += f"{pol_err}\n"

        missing_links = results.get('missing_links', [])
        self.ctx.set_missing_links(missing_links)
        if missing_links:
            output += f"\n缺失连接: {len(missing_links)} 处 (已在视频中标注)"

        # 显示缺失/多余元件摘要
        missing_c = results.get('missing_components', [])
        extra_c = results.get('extra_components', [])
        if missing_c:
            output += f"\n待搭建元件: {', '.join(missing_c)}"
        if extra_c:
            output += f"\n多余元件: {', '.join(extra_c)}"

        self._circuit_page.set_result(output)
        self._log_all(f"验证完成: {len(results.get('errors', []))} 个问题")

    # ================================================================
    # AI 问答
    # ================================================================

    @Slot(str)
    def _ask_ai(self, question: str):
        """处理 AI 问答请求 (主线程)

        使用 ctx.get_circuit_snapshot() 获取电路描述, 无需加锁,
        因为快照是在 _process_frame 的写锁内更新的字符串副本。
        """
        if not question:
            return

        # 更新状态
        self._chat_panel.set_ai_status(f"{Icons.LOADING} 思考中...", WARNING)
        self._chat_side.set_ai_status(f"{Icons.LOADING} 思考中...", WARNING)

        # 使用快照而非直接读 analyzer (避免跨线程竞争)
        context = self.ctx.get_circuit_snapshot()

        self._llm_worker = LLMWorker(self.ctx.llm, question, context)
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
        if hasattr(self, '_heartbeat_worker'):
            self._heartbeat_worker.stop()
        event.accept()
