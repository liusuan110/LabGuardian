"""
状态仪表盘页面
实时显示系统各模块状态、检测统计、性能指标
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QGridLayout, QProgressBar, QWidget, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont

from .resources import Icons
from . import styles


class StatusCard(QFrame):
    """单个状态卡片"""

    def __init__(self, icon: str, title: str, value: str = "--", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumSize(160, 90)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        # 标题行
        header = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Segoe UI Emoji", 16))
        icon_label.setStyleSheet("background: transparent;")
        header.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color: {styles.TEXT_DIM}; font-size: 11px; background: transparent;"
        )
        header.addWidget(title_label)
        header.addStretch()
        layout.addLayout(header)

        # 数值
        self._value_label = QLabel(value)
        self._value_label.setStyleSheet(
            f"color: {styles.TEXT}; font-size: 22px; font-weight: bold; background: transparent;"
        )
        layout.addWidget(self._value_label)

        # 子标签 (可选)
        self._sub_label = QLabel("")
        self._sub_label.setStyleSheet(
            f"color: {styles.TEXT_DIM}; font-size: 10px; background: transparent;"
        )
        self._sub_label.hide()
        layout.addWidget(self._sub_label)

    def set_value(self, value: str, color: str = None):
        self._value_label.setText(value)
        if color:
            self._value_label.setStyleSheet(
                f"color: {color}; font-size: 22px; font-weight: bold; background: transparent;"
            )

    def set_sub_text(self, text: str):
        self._sub_label.setText(text)
        self._sub_label.show()


class Dashboard(QFrame):
    """
    主面板 / 仪表盘

    布局:
      ┌────────────────────────────────────┐
      │  状态卡片行 (4列)                   │
      │  [系统状态] [FPS] [元件数] [网络数] │
      ├────────────────────────────────────┤
      │  模型状态区                         │
      │  Vision: ✅  |  LLM: ✅  |  Cal: ❌│
      ├────────────────────────────────────┤
      │  检测日志 (最近 N 条)               │
      └────────────────────────────────────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # ---- 欢迎标题 ----
        welcome = QLabel(f"{Icons.APP} LabGuardian 控制面板")
        welcome.setObjectName("title")
        welcome.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        layout.addWidget(welcome)

        subtitle = QLabel("基于边缘AI的智能理工科实验助教系统 · Intel Cup 2026")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        # ---- 状态卡片网格 ----
        cards_layout = QGridLayout()
        cards_layout.setSpacing(12)

        self.card_status = StatusCard(Icons.OK, "系统状态", "初始化中...")
        self.card_fps    = StatusCard(Icons.FPS, "帧率", "--")
        self.card_comps  = StatusCard("🔍", "检测元件", "0")
        self.card_nets   = StatusCard("🌐", "电路网络", "0")

        cards_layout.addWidget(self.card_status, 0, 0)
        cards_layout.addWidget(self.card_fps, 0, 1)
        cards_layout.addWidget(self.card_comps, 0, 2)
        cards_layout.addWidget(self.card_nets, 0, 3)

        layout.addLayout(cards_layout)

        # ---- 模型状态区 ----
        model_frame = QFrame()
        model_frame.setObjectName("card")
        m_layout = QVBoxLayout(model_frame)
        m_layout.setContentsMargins(16, 12, 16, 12)
        m_layout.setSpacing(8)

        m_title = QLabel("模块状态")
        m_title.setStyleSheet(
            f"color: {styles.ACCENT}; font-weight: bold; font-size: 14px; "
            f"background: transparent;"
        )
        m_layout.addWidget(m_title)

        # 各模块状态行
        self._module_labels = {}
        modules = [
            ("vision",  "视觉识别模型 (YOLO-OBB)"),
            ("llm",     "语言模型 (LLM)"),
            ("calibr",  "面包板校准"),
            ("polarity","极性推断引擎"),
        ]
        for key, desc in modules:
            row = QHBoxLayout()
            status_dot = QLabel(Icons.LOADING)
            status_dot.setFixedWidth(24)
            status_dot.setStyleSheet("background: transparent;")
            row.addWidget(status_dot)

            desc_label = QLabel(desc)
            desc_label.setStyleSheet(
                f"color: {styles.TEXT}; font-size: 13px; background: transparent;"
            )
            row.addWidget(desc_label)

            row.addStretch()

            info_label = QLabel("等待加载")
            info_label.setStyleSheet(
                f"color: {styles.TEXT_DIM}; font-size: 12px; background: transparent;"
            )
            row.addWidget(info_label)

            m_layout.addLayout(row)
            self._module_labels[key] = (status_dot, info_label)

        layout.addWidget(model_frame)

        # ---- 最近操作日志 ----
        log_frame = QFrame()
        log_frame.setObjectName("card")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(16, 12, 16, 12)
        log_layout.setSpacing(4)

        log_title = QLabel("最近操作")
        log_title.setStyleSheet(
            f"color: {styles.ACCENT}; font-weight: bold; font-size: 14px; "
            f"background: transparent;"
        )
        log_layout.addWidget(log_title)

        self._log_labels = []
        for _ in range(6):
            log_item = QLabel("")
            log_item.setStyleSheet(
                f"color: {styles.TEXT_DIM}; font-size: 12px; background: transparent;"
            )
            log_item.hide()
            log_layout.addWidget(log_item)
            self._log_labels.append(log_item)

        layout.addWidget(log_frame)

        layout.addStretch()

    # ---- 公开更新方法 ----

    def update_system_status(self, status: str, color: str = styles.SUCCESS):
        self.card_status.set_value(status, color)

    def update_fps(self, fps: float):
        color = styles.SUCCESS if fps >= 15 else (styles.WARNING if fps >= 8 else styles.DANGER)
        self.card_fps.set_value(f"{fps:.1f}", color)

    def update_component_count(self, count: int):
        self.card_comps.set_value(str(count))

    def update_net_count(self, count: int):
        self.card_nets.set_value(str(count))

    def update_module_status(self, key: str, ok: bool, info: str = ""):
        """更新模块状态"""
        if key in self._module_labels:
            dot, label = self._module_labels[key]
            dot.setText(Icons.OK if ok else Icons.ERROR)
            label.setText(info or ("就绪" if ok else "未加载"))
            label.setStyleSheet(
                f"color: {styles.SUCCESS if ok else styles.DANGER}; "
                f"font-size: 12px; background: transparent;"
            )

    def add_log(self, text: str):
        """添加最近操作日志 (FIFO, 最多 6 条)"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        full = f"[{timestamp}] {text}"

        # 后移
        for i in range(len(self._log_labels) - 1, 0, -1):
            prev_text = self._log_labels[i - 1].text()
            self._log_labels[i].setText(prev_text)
            self._log_labels[i].setVisible(bool(prev_text))

        self._log_labels[0].setText(full)
        self._log_labels[0].show()
