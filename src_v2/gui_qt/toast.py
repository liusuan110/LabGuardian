"""
Toast 通知组件 — 显示教师指导消息
=================================

从窗口顶部滑入, 自动消失。支持三种类型:
  - hint:    蓝色, 提示信息
  - warning: 橙色, 警告
  - approve: 绿色, 认可
"""

from PySide6.QtWidgets import QFrame, QLabel, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property, QPoint
from PySide6.QtGui import QFont


# 类型 → (背景色, 图标, 边框色)
_TOAST_STYLES = {
    "hint": ("#1a3a5c", "#4a9eff", "#2a6ab5"),
    "warning": ("#5c3a1a", "#ff9f4a", "#b56a2a"),
    "approve": ("#1a5c3a", "#4aff9f", "#2ab56a"),
    "broadcast": ("#3a1a5c", "#9f4aff", "#6a2ab5"),
}

_DEFAULT_STYLE = ("#1a3a5c", "#4a9eff", "#2a6ab5")


class ToastNotification(QFrame):
    """
    滑入式通知, 显示教师指导消息

    用法:
        toast = ToastNotification(parent_widget)
        toast.show_message("hint", "LED需要串联限流电阻", "李老师")
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setMinimumWidth(400)
        self.hide()

        # 自动消失定时器
        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self._dismiss)
        self._dismiss_duration = 8000  # 8 秒

        # 布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        # 图标
        self._icon_label = QLabel()
        self._icon_label.setFixedWidth(24)
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setFont(QFont("Segoe UI Emoji", 14))
        self._icon_label.setStyleSheet("background: transparent;")
        layout.addWidget(self._icon_label)

        # 发送者
        self._sender_label = QLabel()
        self._sender_label.setFixedWidth(60)
        self._sender_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self._sender_label.setStyleSheet("background: transparent;")
        layout.addWidget(self._sender_label)

        # 消息文本
        self._msg_label = QLabel()
        self._msg_label.setFont(QFont("Segoe UI", 11))
        self._msg_label.setWordWrap(True)
        self._msg_label.setStyleSheet("background: transparent;")
        layout.addWidget(self._msg_label, 1)

        # 关闭按钮
        close_btn = QPushButton("x")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #888; border: none; font-size: 14px; }"
            "QPushButton:hover { color: #fff; }"
        )
        close_btn.clicked.connect(self._dismiss)
        layout.addWidget(close_btn)

    def show_message(self, msg_type: str, message: str, sender: str = "Teacher"):
        """
        显示一条通知

        Args:
            msg_type: "hint" / "warning" / "approve" / "broadcast"
            message: 消息文本
            sender: 发送者名称
        """
        bg, accent, border = _TOAST_STYLES.get(msg_type, _DEFAULT_STYLE)

        # 图标
        icons = {
            "hint": "💡",
            "warning": "⚠️",
            "approve": "✅",
            "broadcast": "📢",
        }
        self._icon_label.setText(icons.get(msg_type, "💬"))
        self._sender_label.setText(sender)
        self._sender_label.setStyleSheet(f"color: {accent}; background: transparent;")
        self._msg_label.setText(message)
        self._msg_label.setStyleSheet(f"color: #e0e0e0; background: transparent;")

        # 整体样式
        self.setStyleSheet(
            f"ToastNotification {{"
            f"  background-color: {bg};"
            f"  border: 1px solid {border};"
            f"  border-radius: 8px;"
            f"}}"
        )

        # 定位到父组件顶部中央
        if self.parent():
            pw = self.parent().width()
            toast_w = min(pw - 40, 700)
            self.setFixedWidth(toast_w)
            x = (pw - toast_w) // 2
            self.move(x, 8)

        self.show()
        self.raise_()  # 确保在最上层

        # 重启自动消失定时器
        self._dismiss_timer.stop()
        self._dismiss_timer.start(self._dismiss_duration)

    def _dismiss(self):
        """隐藏通知"""
        self._dismiss_timer.stop()
        self.hide()
