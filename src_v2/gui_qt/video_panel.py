"""
视频面板 (PySide6)
职责: 接收 QPixmap 显示实时视频流, 提供 FPS/状态叠加层
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QWidget, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap, QFont

from .resources import Icons
from . import styles


class VideoOverlay(QWidget):
    """视频叠加信息层 (FPS, 状态, 网络数)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: transparent;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # 左侧: FPS
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet(
            f"color: {styles.SUCCESS}; background: rgba(0,0,0,160); "
            f"border-radius: 4px; padding: 2px 8px; font-size: 12px; font-weight: bold;"
        )
        layout.addWidget(self.fps_label)

        # 中间: 状态
        self.status_label = QLabel("初始化...")
        self.status_label.setStyleSheet(
            f"color: {styles.WARNING}; background: rgba(0,0,0,160); "
            f"border-radius: 4px; padding: 2px 8px; font-size: 12px;"
        )
        layout.addWidget(self.status_label)

        layout.addStretch()

        # 右侧: 网络数
        self.net_label = QLabel("")
        self.net_label.setStyleSheet(
            f"color: {styles.ACCENT}; background: rgba(0,0,0,160); "
            f"border-radius: 4px; padding: 2px 8px; font-size: 12px;"
        )
        layout.addWidget(self.net_label)

    def update_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_status(self, text: str, color: str = None):
        self.status_label.setText(text)
        if color:
            self.status_label.setStyleSheet(
                f"color: {color}; background: rgba(0,0,0,160); "
                f"border-radius: 4px; padding: 2px 8px; font-size: 12px;"
            )

    def update_nets(self, count: int):
        if count > 0:
            self.net_label.setText(f"Nets: {count}")
            self.net_label.show()
        else:
            self.net_label.hide()


class VideoPanel(QFrame):
    """
    视频显示面板

    功能:
      - 接收 QPixmap 并缩放显示
      - FPS / 状态叠加层
      - 底部工具条: 置信度滑块 + 截图 + 切换源
    """

    conf_changed = Signal(float)
    calibrate_requested = Signal()
    load_image_requested = Signal()
    screenshot_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- 视频显示区 ----
        self._video_container = QWidget()
        self._video_container.setStyleSheet(f"background-color: #000;")
        container_layout = QVBoxLayout(self._video_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self._video_label = QLabel("📹 等待视频流...")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet(
            f"color: {styles.TEXT_DIM}; font-size: 16px; background: #000;"
        )
        self._video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._video_label.setMinimumSize(320, 240)
        container_layout.addWidget(self._video_label)

        layout.addWidget(self._video_container, stretch=1)

        # 叠加层
        self._overlay = VideoOverlay(self._video_label)

        # ---- 底部工具条 ----
        toolbar = QFrame()
        toolbar.setFixedHeight(48)
        toolbar.setStyleSheet(f"background-color: {styles.BG_MEDIUM};")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(12, 4, 12, 4)
        tb_layout.setSpacing(8)

        # 置信度滑块
        conf_label = QLabel("置信度:")
        conf_label.setStyleSheet(f"color: {styles.TEXT_DIM}; font-size: 12px;")
        tb_layout.addWidget(conf_label)

        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(1, 99)
        self._conf_slider.setValue(25)
        self._conf_slider.setFixedWidth(140)
        self._conf_slider.valueChanged.connect(
            lambda v: self.conf_changed.emit(v / 100.0)
        )
        tb_layout.addWidget(self._conf_slider)

        self._conf_value = QLabel("0.25")
        self._conf_value.setFixedWidth(36)
        self._conf_value.setStyleSheet(f"color: {styles.ACCENT}; font-size: 12px;")
        self._conf_slider.valueChanged.connect(
            lambda v: self._conf_value.setText(f"{v/100:.2f}")
        )
        tb_layout.addWidget(self._conf_value)

        tb_layout.addStretch()

        # 工具按钮
        btn_calibrate = QPushButton(f"{Icons.CALIBRATE} 校准")
        btn_calibrate.setFixedHeight(32)
        btn_calibrate.clicked.connect(self.calibrate_requested.emit)
        tb_layout.addWidget(btn_calibrate)

        btn_load = QPushButton(f"{Icons.LOAD_IMG} 图片")
        btn_load.setFixedHeight(32)
        btn_load.clicked.connect(self.load_image_requested.emit)
        tb_layout.addWidget(btn_load)

        btn_screenshot = QPushButton(f"{Icons.CAMERA} 截图")
        btn_screenshot.setFixedHeight(32)
        btn_screenshot.clicked.connect(self.screenshot_requested.emit)
        tb_layout.addWidget(btn_screenshot)

        layout.addWidget(toolbar)

    @Slot(QPixmap)
    def update_frame(self, pixmap: QPixmap):
        """接收新的视频帧"""
        if pixmap.isNull():
            return
        # 缩放到 label 尺寸, 保持比例
        scaled = pixmap.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    @Slot(float)
    def update_fps(self, fps: float):
        self._overlay.update_fps(fps)

    def update_overlay_status(self, text: str, color: str = None):
        self._overlay.update_status(text, color)

    def update_overlay_nets(self, count: int):
        self._overlay.update_nets(count)

    def resizeEvent(self, event):
        """确保叠加层跟随视频区大小"""
        super().resizeEvent(event)
        self._overlay.setGeometry(0, 0, self._video_label.width(), 36)

    def get_confidence(self) -> float:
        return self._conf_slider.value() / 100.0
