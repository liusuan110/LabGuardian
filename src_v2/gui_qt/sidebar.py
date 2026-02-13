"""
侧边栏导航组件
PyDracula 风格: 可折叠 + 图标+文字 + 高亮选中项 + Logo
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSizePolicy, QSpacerItem, QWidget,
)
from PySide6.QtCore import Signal, Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont

from .resources import Icons
from . import styles


class SidebarButton(QPushButton):
    """侧边栏导航按钮"""

    def __init__(self, icon: str, text: str, page_name: str, parent=None):
        super().__init__(f"  {icon}  {text}", parent)
        self.page_name = page_name
        self._is_active = False
        self.setObjectName("sidebarBtn")
        self.setFixedHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_active(self, active: bool):
        self._is_active = active
        self.setObjectName("sidebarBtnActive" if active else "sidebarBtn")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_collapsed(self, collapsed: bool):
        """折叠时只显示图标"""
        parts = self.text().split("  ")
        if collapsed and len(parts) >= 2:
            self.setText(f"  {parts[1]}  ")  # 只保留图标
        # 展开时需要外部重设文本


class Sidebar(QFrame):
    """
    可折叠侧边栏

    信号:
        page_changed(str): 页面名称切换
    """
    page_changed = Signal(str)

    EXPANDED_WIDTH = 200
    COLLAPSED_WIDTH = 60

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(self.EXPANDED_WIDTH)
        self._expanded = True
        self._buttons: list[SidebarButton] = []
        self._button_data: list[tuple] = []  # (icon, text, page_name)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- Logo 区域 ----
        logo_frame = QFrame()
        logo_layout = QHBoxLayout(logo_frame)
        logo_layout.setContentsMargins(12, 16, 12, 16)

        self._logo_icon = QLabel(Icons.APP)
        self._logo_icon.setFont(QFont("Segoe UI Emoji", 20))
        logo_layout.addWidget(self._logo_icon)

        self._logo_text = QLabel("LabGuardian")
        self._logo_text.setObjectName("title")
        self._logo_text.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        logo_layout.addWidget(self._logo_text)
        logo_layout.addStretch()

        layout.addWidget(logo_frame)

        # ---- 分隔线 ----
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {styles.BORDER};")
        layout.addWidget(sep)

        # ---- 导航按钮区 ----
        self._nav_layout = QVBoxLayout()
        self._nav_layout.setContentsMargins(4, 8, 4, 8)
        self._nav_layout.setSpacing(2)
        layout.addLayout(self._nav_layout)

        # 页面按钮
        pages = [
            (Icons.HOME,     "主面板",    "home"),
            (Icons.VIDEO,    "视频检测",  "video"),
            (Icons.CHAT,     "AI 助手",   "chat"),
            (Icons.CIRCUIT,  "电路验证",  "circuit"),
            (Icons.SETTINGS, "设置",      "settings"),
        ]
        for icon, text, name in pages:
            self._add_nav_button(icon, text, name)

        # 默认选中第一个
        if self._buttons:
            self._buttons[0].set_active(True)

        # ---- 弹簧 ----
        layout.addStretch()

        # ---- 底部折叠按钮 ----
        self._toggle_btn = QPushButton(f"  {Icons.COLLAPSE}  收起")
        self._toggle_btn.setObjectName("sidebarBtn")
        self._toggle_btn.setFixedHeight(40)
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.clicked.connect(self.toggle_collapse)
        layout.addWidget(self._toggle_btn)

        # ---- 版本标签 ----
        ver_label = QLabel("v2.0 — Intel Cup 2026")
        ver_label.setObjectName("subtitle")
        ver_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver_label.setFixedHeight(24)
        layout.addWidget(ver_label)

    def _add_nav_button(self, icon: str, text: str, page_name: str):
        btn = SidebarButton(icon, text, page_name)
        btn.clicked.connect(lambda checked=False, n=page_name: self._on_click(n))
        self._nav_layout.addWidget(btn)
        self._buttons.append(btn)
        self._button_data.append((icon, text, page_name))

    def _on_click(self, page_name: str):
        for btn in self._buttons:
            btn.set_active(btn.page_name == page_name)
        self.page_changed.emit(page_name)

    def toggle_collapse(self):
        """切换折叠/展开"""
        self._expanded = not self._expanded
        target_width = self.EXPANDED_WIDTH if self._expanded else self.COLLAPSED_WIDTH

        anim = QPropertyAnimation(self, b"minimumWidth")
        anim.setDuration(200)
        anim.setStartValue(self.width())
        anim.setEndValue(target_width)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        anim.start()
        self._anim = anim  # prevent GC

        anim2 = QPropertyAnimation(self, b"maximumWidth")
        anim2.setDuration(200)
        anim2.setStartValue(self.width())
        anim2.setEndValue(target_width)
        anim2.setEasingCurve(QEasingCurve.Type.InOutQuad)
        anim2.start()
        self._anim2 = anim2

        # 更新按钮文字
        if self._expanded:
            self._toggle_btn.setText(f"  {Icons.COLLAPSE}  收起")
            self._logo_text.show()
            for i, btn in enumerate(self._buttons):
                icon, text, _ = self._button_data[i]
                btn.setText(f"  {icon}  {text}")
        else:
            self._toggle_btn.setText(f"  {Icons.EXPAND}")
            self._logo_text.hide()
            for i, btn in enumerate(self._buttons):
                icon, _, _ = self._button_data[i]
                btn.setText(f" {icon}")

    def select_page(self, page_name: str):
        """编程方式选中页面"""
        self._on_click(page_name)
