"""
电路验证页面
功能: 设置金标准 / 保存加载模板 / 运行验证 / 显示网表 / 电源轨标注
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QWidget, QGroupBox,
    QGridLayout, QSizePolicy, QComboBox, QLineEdit,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from .resources import Icons
from . import styles

# 电源轨预设选项
RAIL_PRESETS = [
    "",              # 空 = 未标注
    "VCC +5V",
    "VCC +3.3V",
    "VCC +12V",
    "GND",
]

# 轨道显示名称
RAIL_DISPLAY_NAMES = {
    "RAIL_TOP_1":    "顶部-外侧轨",
    "RAIL_TOP_2":    "顶部-内侧轨",
    "RAIL_BOTTOM_1": "底部-内侧轨",
    "RAIL_BOTTOM_2": "底部-外侧轨",
}


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
        rail_assigned(str, str): 轨道标注变更 (track_id, label)
        rail_cleared:            清除所有轨道标注
    """

    golden_ref_requested     = Signal()
    save_template_requested  = Signal()
    load_template_requested  = Signal()
    validate_requested       = Signal()
    show_netlist_requested   = Signal()
    reset_requested          = Signal()
    rail_assigned            = Signal(str, str)   # (track_id, label)
    rail_cleared             = Signal()

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

        # ---- 电源轨标注组 ----
        rail_group = QGroupBox("电源轨配置 (请在检测到轨道连接后标注)")
        rail_layout = QGridLayout(rail_group)
        rail_layout.setSpacing(8)

        # 表头
        rail_layout.addWidget(QLabel("轨道"), 0, 0)
        rail_layout.addWidget(QLabel("用途"), 0, 1)
        rail_layout.addWidget(QLabel("自定义"), 0, 2)

        self._rail_combos: dict = {}   # track_id → QComboBox
        self._rail_custom: dict = {}   # track_id → QLineEdit
        self._rail_status: dict = {}   # track_id → QLabel

        rail_ids = ["RAIL_TOP_1", "RAIL_TOP_2", "RAIL_BOTTOM_1", "RAIL_BOTTOM_2"]
        for i, track_id in enumerate(rail_ids):
            row = i + 1
            display_name = RAIL_DISPLAY_NAMES.get(track_id, track_id)

            # 状态标签 (含名称 + 状态指示)
            status_label = QLabel(f"{display_name}")
            status_label.setMinimumWidth(100)
            self._rail_status[track_id] = status_label
            rail_layout.addWidget(status_label, row, 0)

            # 预设下拉框
            combo = QComboBox()
            combo.addItems(["-- 未标注 --"] + RAIL_PRESETS[1:])
            combo.setFixedHeight(30)
            combo.currentIndexChanged.connect(
                lambda idx, tid=track_id: self._on_rail_combo_changed(tid, idx)
            )
            self._rail_combos[track_id] = combo
            rail_layout.addWidget(combo, row, 1)

            # 自定义输入框 (如需非预设值)
            custom_edit = QLineEdit()
            custom_edit.setPlaceholderText("自定义 (如 -12V)")
            custom_edit.setFixedHeight(30)
            custom_edit.editingFinished.connect(
                lambda tid=track_id: self._on_rail_custom_edited(tid)
            )
            self._rail_custom[track_id] = custom_edit
            rail_layout.addWidget(custom_edit, row, 2)

        # 清除按钮
        clear_rail_btn = QPushButton("清除所有轨道标注")
        clear_rail_btn.setFixedHeight(32)
        clear_rail_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_rail_btn.clicked.connect(self._on_clear_rails)
        rail_layout.addWidget(clear_rail_btn, len(rail_ids) + 1, 0, 1, 3)

        layout.addWidget(rail_group)

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

    # ---- 电源轨标注交互 ----

    def _on_rail_combo_changed(self, track_id: str, index: int):
        """下拉框选项改变时发送标注信号"""
        if index == 0:
            # "-- 未标注 --" 被选中, 不发送 (除非之前有值, 则清除)
            return
        label = RAIL_PRESETS[index]  # index 0 是空, 与 RAIL_PRESETS 对齐
        # 清空自定义输入框
        self._rail_custom[track_id].clear()
        self.rail_assigned.emit(track_id, label)

    def _on_rail_custom_edited(self, track_id: str):
        """自定义输入框编辑完成时发送标注信号"""
        text = self._rail_custom[track_id].text().strip()
        if text:
            # 将下拉框重置为 "-- 未标注 --" (因为使用自定义值)
            self._rail_combos[track_id].blockSignals(True)
            self._rail_combos[track_id].setCurrentIndex(0)
            self._rail_combos[track_id].blockSignals(False)
            self.rail_assigned.emit(track_id, text)

    def _on_clear_rails(self):
        """清除所有轨道标注"""
        for track_id in self._rail_combos:
            self._rail_combos[track_id].blockSignals(True)
            self._rail_combos[track_id].setCurrentIndex(0)
            self._rail_combos[track_id].blockSignals(False)
            self._rail_custom[track_id].clear()
        self.rail_cleared.emit()

    def highlight_unassigned_rails(self, rail_ids: list):
        """高亮需要标注的轨道 (由系统检测到连接后调用)"""
        for track_id, label in self._rail_status.items():
            display_name = RAIL_DISPLAY_NAMES.get(track_id, track_id)
            if track_id in rail_ids:
                label.setText(f"* {display_name}")
                label.setStyleSheet("color: #ffcc00; font-weight: bold;")
            else:
                label.setText(f"{display_name}")
                label.setStyleSheet("")

    def update_rail_status(self, assignments: dict):
        """根据当前标注状态更新 UI (用于恢复状态)"""
        for track_id, combo in self._rail_combos.items():
            label = assignments.get(track_id, "")
            if not label:
                combo.blockSignals(True)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
                continue
            # 尝试匹配预设
            matched = False
            for i, preset in enumerate(RAIL_PRESETS):
                if preset == label:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(i)
                    combo.blockSignals(False)
                    matched = True
                    break
            if not matched:
                # 非预设值, 放入自定义输入框
                combo.blockSignals(True)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
                self._rail_custom[track_id].setText(label)
