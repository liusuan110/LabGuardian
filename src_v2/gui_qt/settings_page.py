"""
设置页面
可视化配置: 摄像头 / 模型 / LLM / 校准参数
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QComboBox, QGroupBox,
    QGridLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QScrollArea, QWidget,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from .resources import Icons
from . import styles


class SettingsPage(QFrame):
    """
    设置页面

    信号:
        settings_changed: 设置已修改 (需要重载模块)
    """
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        # 外层滚动
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # 标题
        title = QLabel(f"{Icons.SETTINGS} 设置")
        title.setObjectName("title")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        # ---- 摄像头设置 ----
        cam_group = QGroupBox("摄像头")
        cam_layout = QGridLayout(cam_group)
        cam_layout.setSpacing(8)

        cam_layout.addWidget(QLabel("设备号:"), 0, 0)
        self._cam_device = QSpinBox()
        self._cam_device.setRange(0, 10)
        self._cam_device.setValue(0)
        cam_layout.addWidget(self._cam_device, 0, 1)

        cam_layout.addWidget(QLabel("分辨率:"), 1, 0)
        res_layout = QHBoxLayout()
        self._cam_width = QSpinBox()
        self._cam_width.setRange(320, 1920)
        self._cam_width.setValue(640)
        res_layout.addWidget(self._cam_width)
        res_layout.addWidget(QLabel("×"))
        self._cam_height = QSpinBox()
        self._cam_height.setRange(240, 1080)
        self._cam_height.setValue(480)
        res_layout.addWidget(self._cam_height)
        cam_layout.addLayout(res_layout, 1, 1)

        layout.addWidget(cam_group)

        # ---- YOLO 模型设置 ----
        yolo_group = QGroupBox("视觉检测 (YOLO)")
        yolo_layout = QGridLayout(yolo_group)
        yolo_layout.setSpacing(8)

        yolo_layout.addWidget(QLabel("置信度阈值:"), 0, 0)
        self._conf_threshold = QDoubleSpinBox()
        self._conf_threshold.setRange(0.01, 0.99)
        self._conf_threshold.setSingleStep(0.05)
        self._conf_threshold.setValue(0.25)
        yolo_layout.addWidget(self._conf_threshold, 0, 1)

        yolo_layout.addWidget(QLabel("NMS IoU:"), 1, 0)
        self._iou_threshold = QDoubleSpinBox()
        self._iou_threshold.setRange(0.1, 0.9)
        self._iou_threshold.setSingleStep(0.05)
        self._iou_threshold.setValue(0.5)
        yolo_layout.addWidget(self._iou_threshold, 1, 1)

        yolo_layout.addWidget(QLabel("推理分辨率:"), 2, 0)
        self._imgsz = QComboBox()
        self._imgsz.addItems(["640", "800", "960", "1280"])
        self._imgsz.setCurrentText("960")
        yolo_layout.addWidget(self._imgsz, 2, 1)

        yolo_layout.addWidget(QLabel("推理设备:"), 3, 0)
        self._device = QComboBox()
        self._device.addItems(["0 (GPU)", "cpu", "GPU (OpenVINO)"])
        yolo_layout.addWidget(self._device, 3, 1)

        layout.addWidget(yolo_group)

        # ---- LLM 设置 ----
        llm_group = QGroupBox("语言模型 (LLM)")
        llm_layout = QGridLayout(llm_group)
        llm_layout.setSpacing(8)

        self._use_cloud = QCheckBox("启用云端 API")
        self._use_cloud.setChecked(True)
        llm_layout.addWidget(self._use_cloud, 0, 0, 1, 2)

        llm_layout.addWidget(QLabel("API 提供商:"), 1, 0)
        self._cloud_provider = QComboBox()
        self._cloud_provider.addItems(["deepseek", "openai", "qwen"])
        llm_layout.addWidget(self._cloud_provider, 1, 1)

        llm_layout.addWidget(QLabel("API Key:"), 2, 0)
        self._api_key = QLineEdit()
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setPlaceholderText("从环境变量 LG_API_KEY 读取")
        llm_layout.addWidget(self._api_key, 2, 1)

        llm_layout.addWidget(QLabel("本地设备:"), 3, 0)
        self._llm_device = QComboBox()
        self._llm_device.addItems(["NPU", "GPU", "CPU"])
        llm_layout.addWidget(self._llm_device, 3, 1)

        llm_layout.addWidget(QLabel("最大 Tokens:"), 4, 0)
        self._max_tokens = QSpinBox()
        self._max_tokens.setRange(50, 2048)
        self._max_tokens.setValue(300)
        llm_layout.addWidget(self._max_tokens, 4, 1)

        layout.addWidget(llm_group)

        # ---- 电路分析设置 ----
        circuit_group = QGroupBox("电路分析")
        c_layout = QGridLayout(circuit_group)
        c_layout.setSpacing(8)

        self._enable_polarity = QCheckBox("启用极性推断")
        self._enable_polarity.setChecked(True)
        c_layout.addWidget(self._enable_polarity, 0, 0, 1, 2)

        c_layout.addWidget(QLabel("极性置信度:"), 1, 0)
        self._polarity_conf = QDoubleSpinBox()
        self._polarity_conf.setRange(0.1, 0.9)
        self._polarity_conf.setSingleStep(0.05)
        self._polarity_conf.setValue(0.4)
        c_layout.addWidget(self._polarity_conf, 1, 1)

        self._check_polarity_on_validate = QCheckBox("验证时检查极性")
        self._check_polarity_on_validate.setChecked(True)
        c_layout.addWidget(self._check_polarity_on_validate, 2, 0, 1, 2)

        layout.addWidget(circuit_group)

        # ---- 保存按钮 ----
        save_btn = QPushButton(f"{Icons.SAVE} 保存设置")
        save_btn.setObjectName("accent")
        save_btn.setFixedHeight(44)
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self.settings_changed.emit)
        layout.addWidget(save_btn)

        layout.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)

    # ---- 公开读取方法 (供 MainWindow 调用) ----

    def get_camera_settings(self) -> dict:
        return {
            "device_id": self._cam_device.value(),
            "width": self._cam_width.value(),
            "height": self._cam_height.value(),
        }

    def get_vision_settings(self) -> dict:
        return {
            "conf_threshold": self._conf_threshold.value(),
            "iou_threshold": self._iou_threshold.value(),
            "imgsz": int(self._imgsz.currentText()),
        }

    def get_llm_settings(self) -> dict:
        return {
            "use_cloud": self._use_cloud.isChecked(),
            "cloud_provider": self._cloud_provider.currentText(),
            "api_key": self._api_key.text(),
            "local_device": self._llm_device.currentText(),
            "max_tokens": self._max_tokens.value(),
        }
