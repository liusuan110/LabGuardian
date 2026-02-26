"""
QSS 暗色主题样式表
灵感来源: PyDracula + Fluent Design
色板:
  bg_dark:    #1b1e23   主背景
  bg_medium:  #21252b   面板背景
  bg_light:   #2c313a   悬停/卡片
  accent:     #4fc3f7   主色调 (Intel 蓝)
  accent_alt: #7c4dff   辅助色 (紫)
  text:       #e0e0e0   正文
  text_dim:   #8b949e   次要文字
  success:    #66bb6a   成功
  warning:    #ffa726   警告
  danger:     #ef5350   错误
  border:     #30363d   边框
"""

# ============================================================
# 色板常量 (供 Python 代码引用)
# ============================================================

BG_DARK    = "#1b1e23"
BG_MEDIUM  = "#21252b"
BG_LIGHT   = "#2c313a"
ACCENT     = "#4fc3f7"
ACCENT_ALT = "#7c4dff"
TEXT       = "#e0e0e0"
TEXT_DIM   = "#8b949e"
SUCCESS    = "#66bb6a"
WARNING    = "#ffa726"
DANGER     = "#ef5350"
BORDER     = "#30363d"

# ============================================================
# 全局 QSS
# ============================================================

GLOBAL_STYLE = """
/* ---- 全局 ---- */
QWidget {
    background-color: """ + BG_DARK + """;
    color: """ + TEXT + """;
    font-family: "Segoe UI", "Microsoft YaHei UI", "Noto Sans CJK SC", sans-serif;
    font-size: 13px;
    border: none;
}

/* ---- 主窗口 ---- */
QMainWindow {
    background-color: """ + BG_DARK + """;
}

/* ---- QFrame / 卡片容器 ---- */
QFrame {
    background-color: transparent;
}
QFrame#card {
    background-color: """ + BG_MEDIUM + """;
    border: 1px solid """ + BORDER + """;
    border-radius: 8px;
}

/* ---- 标签 ---- */
QLabel {
    background: transparent;
    color: """ + TEXT + """;
    padding: 0px;
}
QLabel#title {
    font-size: 18px;
    font-weight: bold;
    color: """ + ACCENT + """;
}
QLabel#subtitle {
    font-size: 11px;
    color: """ + TEXT_DIM + """;
}
QLabel#statusGood {
    color: """ + SUCCESS + """;
    font-weight: bold;
}
QLabel#statusBad {
    color: """ + DANGER + """;
    font-weight: bold;
}

/* ---- 按钮 ---- */
QPushButton {
    background-color: """ + BG_LIGHT + """;
    color: """ + TEXT + """;
    border: 1px solid """ + BORDER + """;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: """ + ACCENT + """;
    color: #000000;
    border-color: """ + ACCENT + """;
}
QPushButton:pressed {
    background-color: #3da8d4;
}
QPushButton:disabled {
    background-color: #1a1d21;
    color: #555;
    border-color: #252830;
}
QPushButton#accent {
    background-color: """ + ACCENT + """;
    color: #000;
    font-weight: bold;
    border: none;
}
QPushButton#accent:hover {
    background-color: #80d8ff;
}
QPushButton#danger {
    background-color: """ + DANGER + """;
    color: #fff;
    border: none;
}
QPushButton#danger:hover {
    background-color: #ff7043;
}

/* ---- 侧边栏 ---- */
QFrame#sidebar {
    background-color: """ + BG_MEDIUM + """;
    border-right: 1px solid """ + BORDER + """;
}
QPushButton#sidebarBtn {
    background-color: transparent;
    color: """ + TEXT_DIM + """;
    text-align: left;
    padding: 12px 16px;
    border-radius: 0px;
    border: none;
    font-size: 13px;
}
QPushButton#sidebarBtn:hover {
    background-color: """ + BG_LIGHT + """;
    color: """ + TEXT + """;
}
QPushButton#sidebarBtnActive {
    background-color: """ + BG_LIGHT + """;
    color: """ + ACCENT + """;
    text-align: left;
    padding: 12px 16px;
    border-radius: 0px;
    border-left: 3px solid """ + ACCENT + """;
    font-size: 13px;
    font-weight: bold;
}

/* ---- 输入框 ---- */
QLineEdit {
    background-color: """ + BG_LIGHT + """;
    color: """ + TEXT + """;
    border: 1px solid """ + BORDER + """;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: """ + ACCENT + """;
}
QLineEdit:focus {
    border-color: """ + ACCENT + """;
}

/* ---- 文本框 ---- */
QTextEdit, QPlainTextEdit {
    background-color: """ + BG_LIGHT + """;
    color: """ + TEXT + """;
    border: 1px solid """ + BORDER + """;
    border-radius: 6px;
    padding: 8px;
    font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
    font-size: 12px;
    selection-background-color: """ + ACCENT + """;
}
QTextEdit:focus, QPlainTextEdit:focus {
    border-color: """ + ACCENT + """;
}

/* ---- 滚动条 ---- */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: """ + BORDER + """;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: """ + TEXT_DIM + """;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: """ + BORDER + """;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: """ + TEXT_DIM + """;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ---- 滑块 ---- */
QSlider::groove:horizontal {
    background: """ + BORDER + """;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: """ + ACCENT + """;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #80d8ff;
}
QSlider::sub-page:horizontal {
    background: """ + ACCENT + """;
    border-radius: 2px;
}

/* ---- 下拉框 ---- */
QComboBox {
    background-color: """ + BG_LIGHT + """;
    color: """ + TEXT + """;
    border: 1px solid """ + BORDER + """;
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 100px;
}
QComboBox:hover {
    border-color: """ + ACCENT + """;
}
QComboBox::drop-down {
    border: none;
    width: 30px;
}
QComboBox QAbstractItemView {
    background-color: """ + BG_MEDIUM + """;
    color: """ + TEXT + """;
    selection-background-color: """ + BG_LIGHT + """;
    border: 1px solid """ + BORDER + """;
}

/* ---- 进度条 ---- */
QProgressBar {
    background-color: """ + BORDER + """;
    border: none;
    border-radius: 4px;
    text-align: center;
    color: """ + TEXT + """;
    height: 8px;
}
QProgressBar::chunk {
    background-color: """ + ACCENT + """;
    border-radius: 4px;
}

/* ---- 分组框 ---- */
QGroupBox {
    background-color: """ + BG_MEDIUM + """;
    border: 1px solid """ + BORDER + """;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 20px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    color: """ + ACCENT + """;
}

/* ---- 工具提示 ---- */
QToolTip {
    background-color: """ + BG_MEDIUM + """;
    color: """ + TEXT + """;
    border: 1px solid """ + BORDER + """;
    border-radius: 4px;
    padding: 4px 8px;
}

/* ---- 分割器 ---- */
QSplitter::handle {
    background-color: """ + BORDER + """;
}
QSplitter::handle:horizontal {
    width: 2px;
}
QSplitter::handle:vertical {
    height: 2px;
}
QSplitter::handle:hover {
    background-color: """ + ACCENT + """;
}

/* ---- 标签页 ---- */
QTabWidget::pane {
    border: 1px solid """ + BORDER + """;
    border-radius: 4px;
    background-color: """ + BG_MEDIUM + """;
}
QTabBar::tab {
    background-color: """ + BG_DARK + """;
    color: """ + TEXT_DIM + """;
    padding: 8px 16px;
    border: none;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected {
    color: """ + ACCENT + """;
    border-bottom: 2px solid """ + ACCENT + """;
}
QTabBar::tab:hover {
    color: """ + TEXT + """;
    background-color: """ + BG_LIGHT + """;
}
"""
