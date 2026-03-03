"""
LabGuardian 主窗口 (PySide6)
PyDracula 风格: 侧边栏 + 多页面路由 + 自定义标题栏

架构 (v5 — Image-only):
  - ImageAnalyzer:        图片分析引擎 (YOLO + Wire + 电路拓扑)
  - CalibrationHelper:    面包板校准交互 (自动检测)
  - MainWindow:           UI 编排 / 信号连接 / 用户操作回调

布局:
  ┌──────────────────────────────────────────────────┐
  │  自定义标题栏 (拖拽移动, 最小化/最大化/关闭)      │
  ├────────┬─────────────────────────────────────────┤
  │        │                                          │
  │ 侧边栏 │        内容区 (QStackedWidget)           │
  │        │   home:    UploadPage + Dashboard        │
  │  导航   │   results: 标注结果图 + AI 助手侧栏     │
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
import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QLabel, QFrame, QPushButton,
    QApplication, QFileDialog, QSplitter, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QPoint
from PySide6.QtGui import QFont, QPixmap, QImage, QIcon

# 确保 src_v2 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    gui as gui_cfg, vision as vision_cfg,
    classroom as classroom_cfg,
)
from app_context import AppContext

from .styles import GLOBAL_STYLE, ACCENT, BG_DARK, BG_MEDIUM, BORDER, TEXT, TEXT_DIM, SUCCESS, WARNING, DANGER
from .resources import Icons
from .sidebar import Sidebar
from .upload_page import UploadPage
from .chat_panel import ChatPanel
from .dashboard import Dashboard
from .circuit_page import CircuitPage
from .settings_page import SettingsPage
from .workers import ModelLoaderWorker, LLMWorker, ImageAnalysisWorker
from .calibration import CalibrationHelper
from vision.image_analyzer import ImageAnalyzer

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

    架构 (v5 — Image-only):
      TitleBar
      ├── Sidebar (导航)
      └── QStackedWidget
            ├── page_home    (UploadPage + Dashboard 双栏)
            ├── page_results (标注结果图 + ChatPanel 侧栏)
            ├── page_chat    (ChatPanel 全屏)
            ├── page_circuit (CircuitPage)
            └── page_settings(SettingsPage)

    职责划分:
      - ImageAnalyzer:      图片分析 / OCR / 坐标映射 / 电路建模 (纯计算)
      - CalibrationHelper:  面包板校准交互 (自动检测)
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

        # ---- 图片分析引擎 + 校准辅助 ----
        self._image_analyzer = ImageAnalyzer(self.ctx)
        self._calibration = CalibrationHelper(self.ctx)

        self._setup_ui()
        self._connect_signals()
        self._connect_calibration_callbacks()

        # 后台加载模型
        self._start_model_loading()

    # ================================================================
    # 校准回调连接
    # ================================================================

    def _connect_calibration_callbacks(self):
        """连接 CalibrationHelper 的回调到 UI"""
        self._calibration.on_log = self._log_all
        self._calibration.on_status = self._dashboard.update_module_status

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

        self._sb_analysis = QLabel("分析: 待命")
        self._sb_analysis.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        sb_layout.addWidget(self._sb_analysis)

        root_layout.addWidget(self._statusbar)

    def _create_pages(self):
        """创建全部页面并添加到 stack"""

        # ---- page_home: 上传页 + 仪表盘 双栏 ----
        self._page_home = QWidget()
        home_layout = QHBoxLayout(self._page_home)
        home_layout.setContentsMargins(0, 0, 0, 0)
        home_layout.setSpacing(0)

        home_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧: 图片上传+分析
        self._upload_page = UploadPage()
        home_splitter.addWidget(self._upload_page)

        # 右侧: 仪表盘
        self._dashboard = Dashboard()
        home_splitter.addWidget(self._dashboard)

        home_splitter.setSizes([700, 400])
        home_splitter.setStretchFactor(0, 3)
        home_splitter.setStretchFactor(1, 2)
        home_layout.addWidget(home_splitter)

        self._stack.addWidget(self._page_home)

        # ---- page_results: 标注结果图 + 聊天侧栏 ----
        self._page_results = QWidget()
        results_layout = QHBoxLayout(self._page_results)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(0)

        results_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._result_image = QLabel()
        self._result_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_image.setStyleSheet(
            f"background-color: {BG_MEDIUM}; "
            f"border: 1px solid {BORDER}; border-radius: 6px;"
        )
        self._result_image.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        results_splitter.addWidget(self._result_image)

        self._chat_side = ChatPanel()
        results_splitter.addWidget(self._chat_side)
        results_splitter.setSizes([800, 350])
        results_splitter.setStretchFactor(0, 3)
        results_splitter.setStretchFactor(1, 1)
        results_layout.addWidget(results_splitter)

        self._stack.addWidget(self._page_results)

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
            "results":  1,
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

        # 上传页信号
        self._upload_page.analyze_requested.connect(self._start_analysis)
        self._upload_page.calibrate_requested.connect(self._start_calibration_image)

        # 聊天信号
        self._chat_panel.message_sent.connect(self._ask_ai)
        self._chat_side.message_sent.connect(self._ask_ai)

        # 电路页面信号
        self._circuit_page.golden_ref_requested.connect(self._set_golden_ref)
        self._circuit_page.save_template_requested.connect(self._save_template)
        self._circuit_page.load_template_requested.connect(self._load_template)
        self._circuit_page.validate_requested.connect(self._validate_circuit)
        self._circuit_page.show_netlist_requested.connect(self._show_netlist)
        self._circuit_page.export_netlist_requested.connect(self._export_netlist)
        self._circuit_page.reset_requested.connect(self._reset_analyzer)
        self._circuit_page.rail_assigned.connect(self._on_rail_assigned)
        self._circuit_page.rail_cleared.connect(self._on_rail_cleared)

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
            self._sb_status.setText(f"{Icons.OK} 系统就绪 - 请上传图片开始分析")
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

        self._dashboard.add_log("系统就绪, 请上传电路图片进行分析")

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
    # 图片分析
    # ================================================================

    def _start_analysis(self):
        """UploadPage '开始分析' 按钮 → 启动 ImageAnalysisWorker"""
        images = self._upload_page.get_images()
        if not images:
            return

        conf = self._upload_page.get_confidence()
        imgsz = self._upload_page.get_resolution()

        self._upload_page.set_analyzing(True)
        self._sb_analysis.setText("分析: 运行中...")
        self._sb_analysis.setStyleSheet(f"color: {WARNING}; font-size: 11px;")
        self._log_all(f"开始分析 {len(images)} 张图片 (conf={conf}, imgsz={imgsz})")

        self._analysis_worker = ImageAnalysisWorker(
            self._image_analyzer, images, conf, imgsz)
        self._analysis_worker.progress.connect(self._on_analysis_progress)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.start()

    @Slot(str)
    def _on_analysis_progress(self, msg: str):
        """分析进度更新"""
        self._upload_page.set_progress(msg)
        self._sb_analysis.setText(f"分析: {msg}")

    @Slot(object)
    def _on_analysis_finished(self, result):
        """分析完成 → 显示结果"""
        self._upload_page.set_analyzing(False)
        self._upload_page.show_result(result)

        # 更新状态栏
        n = result.component_count
        self._sb_detect.setText(f"检测: {n} 个元件")
        self._sb_detect.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")
        self._sb_analysis.setText("分析: 完成")
        self._sb_analysis.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")

        # 更新结果页大图
        pixmap = self._cv2_to_qpixmap(result.annotated_image)
        available = self._result_image.size()
        scaled = pixmap.scaled(
            available, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._result_image.setPixmap(scaled)

        # 喂给心跳线程
        if hasattr(self, '_heartbeat_worker'):
            self._heartbeat_worker.update_frame(result.annotated_image)

        # 检查未标注电源轨
        unassigned = self.ctx.get_unassigned_active_rails()
        if unassigned:
            self._handle_rail_notification_list(unassigned)

        self._log_all(
            f"分析完成: {result.component_count} 个元件, "
            f"{result.net_count} 个网络"
        )

    @Slot(str)
    def _on_analysis_error(self, error_msg: str):
        """分析出错"""
        self._upload_page.set_analyzing(False)
        self._sb_analysis.setText("分析: 错误")
        self._sb_analysis.setStyleSheet(f"color: {DANGER}; font-size: 11px;")
        self._log_all(f"分析错误: {error_msg}")

    def _start_calibration_image(self, image):
        """从上传的图片执行校准: 先尝试自动, 失败则弹出手动校准对话框"""
        self._dashboard.add_log("从图片执行校准...")
        success = self._calibration.auto_detect_board(image)
        if not success:
            self._dashboard.add_log("自动校准失败, 请在弹出的对话框中手动标记面包板角点")
            self._calibration.manual_calibrate(image, parent_widget=self)

    @staticmethod
    def _cv2_to_qpixmap(frame: np.ndarray) -> QPixmap:
        """BGR ndarray → QPixmap"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    # ================================================================
    # 电路操作 (主线程, 需要 read_lock)
    # ================================================================

    def _show_netlist(self):
        """显示网表 (主线程, read_lock 保护)"""
        with self.ctx.read_lock():
            netlist = self.ctx.analyzer.get_circuit_description()
        self._circuit_page.set_result(netlist)
        self._log_all("已生成网表")

    def _export_netlist(self):
        """导出结构化网表到 JSON 文件"""
        import json
        with self.ctx.read_lock():
            if not self.ctx.analyzer.components:
                self._log_all("没有检测到元件, 无法导出网表")
                self._circuit_page.set_result("错误: 没有检测到元件, 请先分析图片")
                return
            netlist_data = self.ctx.analyzer.export_netlist()

        # 序列化处理: tuple → list (JSON 不支持 tuple)
        def _serialize(obj):
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, set):
                return list(obj)
            if hasattr(obj, 'value'):
                return obj.value
            return str(obj)

        path, _ = QFileDialog.getSaveFileName(
            self, "导出网表", "circuit_netlist.json",
            "JSON 文件 (*.json);;所有文件 (*)"
        )
        if not path:
            return

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(netlist_data, f, ensure_ascii=False, indent=2, default=_serialize)
            self._log_all(f"网表已导出: {Path(path).name}")
            self._circuit_page.set_result(
                f"网表已成功导出到:\n{path}\n\n"
                f"包含 {len(netlist_data.get('components', []))} 个元件, "
                f"{len(netlist_data.get('nets', {}))} 个网络"
            )
        except Exception as e:
            self._log_all(f"网表导出失败: {e}")
            self._circuit_page.set_result(f"导出失败: {e}")

    def _reset_analyzer(self):
        """重置分析器 (主线程, 通过 AppContext 线程安全方法)"""
        self.ctx.reset_analysis()
        self._log_all("分析器已重置")

    # ================================================================
    # 电源轨标注
    # ================================================================

    @Slot(str, str)
    def _on_rail_assigned(self, track_id: str, label: str):
        """学生在 circuit_page 标注了某条轨道"""
        self.ctx.set_rail_assignment(track_id, label)
        self._log_all(f"电源轨 {track_id} 已标注为: {label}")

    @Slot()
    def _on_rail_cleared(self):
        """学生清除了所有轨道标注"""
        self.ctx.clear_rail_assignments()
        self._log_all("所有电源轨标注已清除")

    def _handle_rail_notification_list(self, unassigned_rails: list):
        """处理电源轨未标注提示"""
        if not unassigned_rails:
            return
        names = ", ".join(unassigned_rails)
        self._log_all(f"检测到电源轨连接, 请标注用途: {names}")
        self._circuit_page.highlight_unassigned_rails(unassigned_rails)

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
            output += f"\n缺失连接: {len(missing_links)} 处"

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

        使用 ctx.get_circuit_snapshot() 获取电路描述, 无需加锁.
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
        if hasattr(self, '_analysis_worker') and self._analysis_worker.isRunning():
            self._analysis_worker.wait(3000)
        if hasattr(self, '_heartbeat_worker'):
            self._heartbeat_worker.stop()
        event.accept()
