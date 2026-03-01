"""
图片上传与分析页面
==================
替代 VideoPanel, 提供图片上传 → 分析 → 结果展示的一站式界面.

布局:
  - 顶部: 图片缩略图槽位 (最多 3 张)
  - 中部: 参数控制 (置信度/分辨率) + 操作按钮
  - 底部: 进度条 + 标注结果图 + 分析报告
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QComboBox, QProgressBar, QFileDialog, QFrame,
    QScrollArea, QTextEdit, QSizePolicy, QSplitter,
)
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QPixmap, QImage, QFont

from . import styles
from .resources import Icons


class ImageSlot(QFrame):
    """单个图片缩略图槽位 (点击添加/替换图片)."""

    clicked = Signal(int)          # 槽位索引
    image_loaded = Signal(int)     # 图片已加载

    THUMB_SIZE = QSize(180, 135)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self._image: Optional[np.ndarray] = None
        self.setObjectName("card")
        self.setFixedSize(200, 170)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 4)
        layout.setSpacing(4)

        # 缩略图
        self._thumb = QLabel()
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setFixedSize(self.THUMB_SIZE)
        self._thumb.setStyleSheet(
            f"background-color: {styles.BG_LIGHT}; "
            f"border: 2px dashed {styles.BORDER}; border-radius: 6px;"
        )
        self._set_placeholder()
        layout.addWidget(self._thumb, alignment=Qt.AlignmentFlag.AlignCenter)

        # 标签
        self._label = QLabel(f"图片 {index + 1}")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(f"color: {styles.TEXT_DIM}; font-size: 11px;")
        layout.addWidget(self._label)

    def _set_placeholder(self):
        self._thumb.setText(f"{Icons.UPLOAD}\n点击添加")
        self._thumb.setStyleSheet(
            f"background-color: {styles.BG_LIGHT}; "
            f"border: 2px dashed {styles.BORDER}; border-radius: 6px; "
            f"color: {styles.TEXT_DIM}; font-size: 13px;"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)

    def set_image(self, image: np.ndarray, filename: str = ""):
        """设置图片并显示缩略图."""
        self._image = image
        # BGR → RGB → QPixmap
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        scaled = pixmap.scaled(
            self.THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb.setPixmap(scaled)
        self._thumb.setStyleSheet(
            f"background-color: {styles.BG_LIGHT}; "
            f"border: 2px solid {styles.ACCENT}; border-radius: 6px;"
        )
        name = Path(filename).name if filename else f"图片 {self.index + 1}"
        self._label.setText(name)
        self._label.setStyleSheet(f"color: {styles.ACCENT}; font-size: 11px;")
        self.image_loaded.emit(self.index)

    def clear(self):
        self._image = None
        self._set_placeholder()
        self._label.setText(f"图片 {self.index + 1}")
        self._label.setStyleSheet(f"color: {styles.TEXT_DIM}; font-size: 11px;")

    @property
    def image(self) -> Optional[np.ndarray]:
        return self._image


class UploadPage(QWidget):
    """图片上传 + 分析触发 + 结果展示页面.

    Signals:
        analyze_requested():               开始分析
        calibrate_requested(np.ndarray):   校准请求 (传递首张图片)
    """

    analyze_requested = Signal()
    calibrate_requested = Signal(object)   # np.ndarray

    MAX_SLOTS = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(12)

        # ---- 标题 ----
        header = QLabel(f"{Icons.CAMERA} 电路图片分析")
        header.setObjectName("title")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        main_layout.addWidget(header)

        # ---- 图片槽位区 ----
        slots_frame = QFrame()
        slots_frame.setObjectName("card")
        slots_layout = QVBoxLayout(slots_frame)
        slots_layout.setContentsMargins(12, 12, 12, 8)

        slot_row = QHBoxLayout()
        slot_row.setSpacing(12)
        self._slots: List[ImageSlot] = []
        for i in range(self.MAX_SLOTS):
            slot = ImageSlot(i)
            slot.clicked.connect(self._pick_image)
            self._slots.append(slot)
            slot_row.addWidget(slot)
        slot_row.addStretch()
        slots_layout.addLayout(slot_row)

        hint = QLabel("提示: 第 1 张全局俯拍, 第 2-3 张可补充遮挡区域")
        hint.setStyleSheet(f"color: {styles.TEXT_DIM}; font-size: 11px;")
        slots_layout.addWidget(hint)

        main_layout.addWidget(slots_frame)

        # ---- 参数 + 操作区 ----
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("card")
        ctrl_layout = QHBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(12, 8, 12, 8)
        ctrl_layout.setSpacing(16)

        # 置信度
        conf_label = QLabel("置信度:")
        ctrl_layout.addWidget(conf_label)
        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(10, 90)
        self._conf_slider.setValue(25)
        self._conf_slider.setFixedWidth(120)
        ctrl_layout.addWidget(self._conf_slider)
        self._conf_value = QLabel("0.25")
        self._conf_value.setFixedWidth(36)
        ctrl_layout.addWidget(self._conf_value)
        self._conf_slider.valueChanged.connect(
            lambda v: self._conf_value.setText(f"{v / 100:.2f}"))

        # 分辨率
        res_label = QLabel("分辨率:")
        ctrl_layout.addWidget(res_label)
        self._res_combo = QComboBox()
        self._res_combo.addItems(["640", "960", "1280"])
        self._res_combo.setCurrentText("1280")
        self._res_combo.setFixedWidth(80)
        ctrl_layout.addWidget(self._res_combo)

        ctrl_layout.addStretch()

        # 按钮组
        self._btn_calibrate = QPushButton(f"{Icons.CALIBRATE} 校准")
        self._btn_calibrate.clicked.connect(self._request_calibrate)
        ctrl_layout.addWidget(self._btn_calibrate)

        self._btn_analyze = QPushButton(f"{Icons.ANALYZE} 开始分析")
        self._btn_analyze.setObjectName("accent")
        self._btn_analyze.clicked.connect(self._request_analyze)
        ctrl_layout.addWidget(self._btn_analyze)

        self._btn_clear = QPushButton(f"{Icons.CLEAR} 清空")
        self._btn_clear.clicked.connect(self._clear_all)
        ctrl_layout.addWidget(self._btn_clear)

        main_layout.addWidget(ctrl_frame)

        # ---- 进度条 ----
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setFixedHeight(4)
        self._progress.hide()
        main_layout.addWidget(self._progress)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(f"color: {styles.TEXT_DIM}; font-size: 11px;")
        self._progress_label.hide()
        main_layout.addWidget(self._progress_label)

        # ---- 结果区 (分割器: 标注图 + 报告) ----
        result_splitter = QSplitter(Qt.Orientation.Vertical)

        # 标注结果图
        self._result_image = QLabel()
        self._result_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_image.setStyleSheet(
            f"background-color: {styles.BG_MEDIUM}; "
            f"border: 1px solid {styles.BORDER}; border-radius: 6px;"
        )
        self._result_image.setMinimumHeight(200)
        self._result_image.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        result_splitter.addWidget(self._result_image)

        # 统计摘要
        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet(
            f"color: {styles.ACCENT}; font-size: 12px; padding: 4px;")
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 分析报告
        report_frame = QFrame()
        report_layout = QVBoxLayout(report_frame)
        report_layout.setContentsMargins(0, 4, 0, 0)
        report_layout.setSpacing(2)
        report_layout.addWidget(self._summary_label)

        self._report_text = QTextEdit()
        self._report_text.setReadOnly(True)
        self._report_text.setPlaceholderText("分析报告将在此显示...")
        self._report_text.setMaximumHeight(200)
        report_layout.addWidget(self._report_text)
        result_splitter.addWidget(report_frame)

        result_splitter.setSizes([400, 200])
        main_layout.addWidget(result_splitter, stretch=1)

    # ================================================================
    # 图片操作
    # ================================================================

    def _pick_image(self, index: int):
        """打开文件对话框选择图片."""
        path, _ = QFileDialog.getOpenFileName(
            self, f"选择图片 {index + 1}", "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)"
        )
        if path:
            img = cv2.imread(path)
            if img is not None:
                self._slots[index].set_image(img, path)

    def _clear_all(self):
        """清空所有图片和结果."""
        for slot in self._slots:
            slot.clear()
        self._result_image.clear()
        self._result_image.setText("")
        self._summary_label.setText("")
        self._report_text.clear()

    # ================================================================
    # 外部接口
    # ================================================================

    def get_images(self) -> List[np.ndarray]:
        """返回已加载的图片列表 (按槽位顺序, 跳过空槽位)."""
        return [s.image for s in self._slots if s.image is not None]

    def get_confidence(self) -> float:
        return self._conf_slider.value() / 100.0

    def get_resolution(self) -> int:
        return int(self._res_combo.currentText())

    def set_analyzing(self, analyzing: bool):
        """设置分析状态 (禁用按钮 + 显示进度条)."""
        self._btn_analyze.setEnabled(not analyzing)
        self._btn_clear.setEnabled(not analyzing)
        self._btn_calibrate.setEnabled(not analyzing)
        if analyzing:
            self._progress.show()
            self._progress_label.show()
        else:
            self._progress.hide()
            self._progress_label.hide()

    def set_progress(self, msg: str):
        """更新进度文本."""
        self._progress_label.setText(msg)

    def show_result(self, result):
        """显示分析结果 (AnalysisResult)."""
        # 标注图
        img = result.annotated_image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        # 缩放到可用区域
        available = self._result_image.size()
        scaled = pixmap.scaled(
            available, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._result_image.setPixmap(scaled)

        # 摘要
        n_comp = result.component_count
        n_wire = len(result.detections) - n_comp
        self._summary_label.setText(
            f"元件: {n_comp} | 导线: {n_wire} | 网络: {result.net_count}"
        )

        # 报告
        self._report_text.setPlainText(result.report)

    # ================================================================
    # 信号触发
    # ================================================================

    def _request_analyze(self):
        images = self.get_images()
        if not images:
            self._summary_label.setText("请先添加至少一张图片")
            self._summary_label.setStyleSheet(
                f"color: {styles.WARNING}; font-size: 12px; padding: 4px;")
            return
        self._summary_label.setStyleSheet(
            f"color: {styles.ACCENT}; font-size: 12px; padding: 4px;")
        self.analyze_requested.emit()

    def _request_calibrate(self):
        images = self.get_images()
        if not images:
            self._summary_label.setText("请先添加至少一张图片以进行校准")
            self._summary_label.setStyleSheet(
                f"color: {styles.WARNING}; font-size: 12px; padding: 4px;")
            return
        self._summary_label.setStyleSheet(
            f"color: {styles.ACCENT}; font-size: 12px; padding: 4px;")
        self.calibrate_requested.emit(images[0])
