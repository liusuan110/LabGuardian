"""
AI 聊天面板 (PySide6)
职责: 消息气泡显示 + 用户输入 + 快捷操作

特性:
  - 消息气泡区分 User / AI / System
  - Markdown 简单渲染 (代码块高亮)
  - 快捷按钮: 问当前元件 / 分析电路 / 安全检查
"""

from datetime import datetime
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QScrollArea, QWidget,
    QSizePolicy, QTextEdit,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QTextCursor

from .resources import Icons
from . import styles


class MessageBubble(QFrame):
    """单条消息气泡"""

    def __init__(self, text: str, role: str = "system", parent=None):
        """
        Args:
            text: 消息内容
            role: "user" | "ai" | "system"
        """
        super().__init__(parent)
        self.setObjectName("card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # 角色标签 + 时间
        header = QHBoxLayout()
        role_map = {
            "user":   (f"{Icons.SEND} 你", styles.ACCENT),
            "ai":     (f"{Icons.AI} AI 助手", styles.ACCENT_ALT),
            "system": (f"{Icons.OK} 系统", styles.TEXT_DIM),
        }
        role_text, role_color = role_map.get(role, (role, styles.TEXT_DIM))

        role_label = QLabel(role_text)
        role_label.setStyleSheet(
            f"color: {role_color}; font-weight: bold; font-size: 12px; background: transparent;"
        )
        header.addWidget(role_label)

        header.addStretch()

        time_label = QLabel(datetime.now().strftime("%H:%M:%S"))
        time_label.setStyleSheet(
            f"color: {styles.TEXT_DIM}; font-size: 10px; background: transparent;"
        )
        header.addWidget(time_label)
        layout.addLayout(header)

        # 消息内容
        content = QLabel(text)
        content.setWordWrap(True)
        content.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        content.setStyleSheet(
            f"color: {styles.TEXT}; font-size: 13px; background: transparent; "
            f"line-height: 1.5;"
        )
        layout.addWidget(content)

        # 根据角色设置背景
        bg_map = {
            "user":   styles.BG_LIGHT,
            "ai":     "#1e2430",
            "system": styles.BG_MEDIUM,
        }
        bg = bg_map.get(role, styles.BG_MEDIUM)
        border = styles.ACCENT if role == "user" else (
            styles.ACCENT_ALT if role == "ai" else styles.BORDER
        )
        self.setStyleSheet(
            f"QFrame#card {{ background-color: {bg}; "
            f"border: 1px solid {border}; border-radius: 8px; }}"
        )


class ChatPanel(QFrame):
    """
    AI 聊天面板

    信号:
        message_sent(str): 用户发送消息
    """
    message_sent = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- 标题栏 ----
        header = QFrame()
        header.setFixedHeight(40)
        header.setStyleSheet(f"background-color: {styles.BG_MEDIUM};")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(12, 0, 12, 0)

        title = QLabel(f"{Icons.CHAT} AI 助手")
        title.setStyleSheet(
            f"color: {styles.ACCENT}; font-weight: bold; font-size: 14px;"
        )
        h_layout.addWidget(title)
        h_layout.addStretch()

        self._status_indicator = QLabel(f"{Icons.LOADING} 初始化中")
        self._status_indicator.setStyleSheet(
            f"color: {styles.WARNING}; font-size: 11px;"
        )
        h_layout.addWidget(self._status_indicator)

        layout.addWidget(header)

        # ---- 消息滚动区 ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._messages_container = QWidget()
        self._messages_layout = QVBoxLayout(self._messages_container)
        self._messages_layout.setContentsMargins(8, 8, 8, 8)
        self._messages_layout.setSpacing(8)
        self._messages_layout.addStretch()  # 消息从底部增长

        scroll.setWidget(self._messages_container)
        self._scroll = scroll
        layout.addWidget(scroll, stretch=1)

        # ---- 快捷按钮区 ----
        quick_frame = QFrame()
        quick_frame.setStyleSheet(f"background-color: {styles.BG_MEDIUM};")
        q_layout = QHBoxLayout(quick_frame)
        q_layout.setContentsMargins(8, 4, 8, 4)
        q_layout.setSpacing(6)

        quick_actions = [
            ("分析当前元件", "分析我当前看到的元件在电路中的连接和功能"),
            ("安全检查", "检查当前电路是否存在安全隐患，例如LED是否有限流电阻"),
            ("解释电路", "用通俗的语言解释当前面包板上的完整电路功能"),
        ]
        for label, prompt in quick_actions:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.setStyleSheet(
                f"QPushButton {{ font-size: 11px; padding: 2px 10px; "
                f"background: {styles.BG_LIGHT}; border-radius: 14px; }}"
                f"QPushButton:hover {{ background: {styles.ACCENT}; color: #000; }}"
            )
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, p=prompt: self._send_quick(p))
            q_layout.addWidget(btn)

        q_layout.addStretch()
        layout.addWidget(quick_frame)

        # ---- 输入区 ----
        input_frame = QFrame()
        input_frame.setFixedHeight(52)
        input_frame.setStyleSheet(
            f"background-color: {styles.BG_MEDIUM}; "
            f"border-top: 1px solid {styles.BORDER};"
        )
        i_layout = QHBoxLayout(input_frame)
        i_layout.setContentsMargins(12, 8, 12, 8)
        i_layout.setSpacing(8)

        self._input = QLineEdit()
        self._input.setPlaceholderText("输入问题，按 Enter 发送...")
        self._input.returnPressed.connect(self._on_send)
        i_layout.addWidget(self._input)

        send_btn = QPushButton(f"{Icons.SEND} 发送")
        send_btn.setObjectName("accent")
        send_btn.setFixedWidth(80)
        send_btn.setFixedHeight(36)
        send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        send_btn.clicked.connect(self._on_send)
        i_layout.addWidget(send_btn)

        layout.addWidget(input_frame)

        # ---- 初始欢迎消息 ----
        self.add_message("欢迎使用 LabGuardian AI 助手！你可以：\n"
                        "• 询问电路相关问题\n"
                        "• 让我分析当前检测到的元件\n"
                        "• 进行电路安全检查", "system")

    def _on_send(self):
        text = self._input.text().strip()
        if text:
            self._input.clear()
            self.add_message(text, "user")
            self.message_sent.emit(text)

    def _send_quick(self, prompt: str):
        self.add_message(prompt, "user")
        self.message_sent.emit(prompt)

    def add_message(self, text: str, role: str = "system"):
        """添加一条消息气泡"""
        bubble = MessageBubble(text, role)
        # 插入到 stretch 之前
        count = self._messages_layout.count()
        self._messages_layout.insertWidget(count - 1, bubble)

        # 自动滚动到底部
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_ai_status(self, text: str, color: str = None):
        """更新 AI 状态指示器"""
        self._status_indicator.setText(text)
        if color:
            self._status_indicator.setStyleSheet(
                f"color: {color}; font-size: 11px;"
            )

    def log(self, text: str):
        """兼容旧 API: 添加系统消息"""
        self.add_message(text, "system")

    def get_user_input(self) -> str:
        """兼容旧 API"""
        text = self._input.text().strip()
        self._input.clear()
        return text
