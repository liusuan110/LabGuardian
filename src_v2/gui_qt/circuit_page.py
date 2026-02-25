"""
电路验证页面
功能: 设置金标准 / 保存加载模板 / 运行验证 / 显示网表
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QWidget, QGroupBox,
    QGridLayout, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from .resources import Icons
from . import styles


class CircuitPage(QFrame):
    """
    电路验证与调试页面

    信号:
        golden_ref_requested:    设置金标准
        save_template_requested: 保存模板
        load_template_requested: 加载模板
        validate_requested:      运行验证
        show_netlist_requested:  显示网表
        reset_requested:         重置分析器
    """

    golden_ref_requested     = Signal()
    save_template_requested  = Signal()
    load_template_requested  = Signal()
    validate_requested       = Signal()
    show_netlist_requested   = Signal()
    reset_requested          = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # 标题
        title = QLabel(f"{Icons.CIRCUIT} 电路验证 & 调试")
        title.setObjectName("title")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        # ---- 操作按钮组 ----
        btn_group = QGroupBox("操作")
        btn_layout = QGridLayout(btn_group)
        btn_layout.setSpacing(10)

        buttons = [
            (f"{Icons.GOLDEN} 设为金标准",  self.golden_ref_requested,     0, 0, "accent"),
            (f"{Icons.VALIDATE} 验证电路",   self.validate_requested,       0, 1, "accent"),
            (f"{Icons.SAVE} 保存模板",       self.save_template_requested,  1, 0, None),
            (f"{Icons.LOAD_IMG} 加载模板",   self.load_template_requested,  1, 1, None),
            (f"{Icons.NETLIST} 网表",        self.show_netlist_requested,   2, 0, None),
            (f"{Icons.RESET} 重置分析器",    self.reset_requested,          2, 1, "danger"),
        ]
        for text, signal, row, col, style_id in buttons:
            btn = QPushButton(text)
            btn.setFixedHeight(40)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            if style_id:
                btn.setObjectName(style_id)
            btn.clicked.connect(signal.emit)
            btn_layout.addWidget(btn, row, col)

        layout.addWidget(btn_group)

        # ---- 验证结果区 ----
        result_group = QGroupBox("验证结果")
        r_layout = QVBoxLayout(result_group)

        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setPlaceholderText("验证结果将显示在这里...\n\n"
            "步骤:\n"
            "1. 检测到元件后点击 '设为金标准'\n"
            "2. 修改电路后点击 '验证电路'\n"
            "3. 差异将以红/绿标注在视频画面上")
        r_layout.addWidget(self._result_text)

        layout.addWidget(result_group, stretch=1)

        # ---- 模板信息 ----
        self._template_info = QLabel("未加载模板")
        self._template_info.setObjectName("subtitle")
        layout.addWidget(self._template_info)

    def set_result(self, text: str):
        """显示验证结果"""
        self._result_text.setPlainText(text)

    def append_result(self, text: str):
        """追加验证结果"""
        self._result_text.append(text)

    def set_template_info(self, text: str):
        self._template_info.setText(text)

    def clear_result(self):
        self._result_text.clear()
