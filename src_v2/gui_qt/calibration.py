"""
LabGuardian 校准辅助
====================
面包板校准交互逻辑:
  1. 自动检测 (auto_detect_board): 全自动从图片识别面包板
  2. 手动校准 (ManualCalibrationDialog): Qt 内嵌对话框, 点击4角
  3. 全图回退 (fallback_calibrate): 以全图为面包板区域

v5.1: 移除 OpenCV 弹窗, 改用 PySide6 对话框实现手动校准
"""

import logging
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

from config import vision as vision_cfg
from app_context import AppContext

logger = logging.getLogger(__name__)


class ImageCalibrationLabel(QLabel):
    """可点击的图片标签, 用于在图片上标记面包板角点."""

    point_added = Signal(int)  # 当前已标记的点数

    MAX_POINTS = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)

        self._image: np.ndarray = None   # BGR
        self._pixmap: QPixmap = None
        self._points: list = []          # [(x_real, y_real), ...]
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

    def set_image(self, image: np.ndarray):
        """设置用于校准的图片 (BGR)."""
        self._image = image
        self._points.clear()
        self._update_display()

    def get_points(self) -> np.ndarray:
        """返回标记的 4 个角点 (原图坐标), shape (4, 2)."""
        return np.array(self._points, dtype=np.float32)

    @property
    def point_count(self) -> int:
        return len(self._points)

    def undo_last(self):
        """撤销最后一个标记点."""
        if self._points:
            self._points.pop()
            self._update_display()

    def clear_points(self):
        """清除所有标记点."""
        self._points.clear()
        self._update_display()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        if len(self._points) >= self.MAX_POINTS:
            return

        if self._pixmap is None:
            return

        # 计算点击位置 → 原图坐标
        pos = event.position()
        px, py = pos.x(), pos.y()

        # 将 QLabel 坐标转换为原图坐标
        real_x = (px - self._offset_x) / self._scale
        real_y = (py - self._offset_y) / self._scale

        if real_x < 0 or real_y < 0:
            return
        if self._image is not None:
            h, w = self._image.shape[:2]
            if real_x > w or real_y > h:
                return

        self._points.append((real_x, real_y))
        self._update_display()
        self.point_added.emit(len(self._points))

    def _update_display(self):
        """重绘图片 + 标记点."""
        if self._image is None:
            return

        rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        base_pixmap = QPixmap.fromImage(qimg.copy())

        # 缩放至可用区域
        available = self.size()
        scaled = base_pixmap.scaled(
            available, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self._scale = scaled.width() / w
        # 居中偏移
        self._offset_x = (available.width() - scaled.width()) / 2
        self._offset_y = (available.height() - scaled.height()) / 2

        # 在缩放后的图像上绘制标记点
        painter = QPainter(scaled)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        labels = ["TL", "TR", "BR", "BL"]
        colors = [QColor(255, 0, 0), QColor(0, 255, 0),
                  QColor(0, 0, 255), QColor(255, 255, 0)]

        for i, (rx, ry) in enumerate(self._points):
            sx = int(rx * self._scale)
            sy = int(ry * self._scale)

            # 圆点
            pen = QPen(colors[i], 3)
            painter.setPen(pen)
            painter.setBrush(colors[i])
            painter.drawEllipse(QPoint(sx, sy), 6, 6)

            # 标签
            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawText(sx + 10, sy - 5, f"{i+1}:{labels[i]}")

        # 如果有多个点, 连线
        if len(self._points) > 1:
            pen = QPen(QColor(0, 255, 255), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(self._points) - 1):
                p1 = QPoint(
                    int(self._points[i][0] * self._scale),
                    int(self._points[i][1] * self._scale))
                p2 = QPoint(
                    int(self._points[i+1][0] * self._scale),
                    int(self._points[i+1][1] * self._scale))
                painter.drawLine(p1, p2)
            # 如果 4 点齐全, 闭合
            if len(self._points) == 4:
                p_last = QPoint(
                    int(self._points[3][0] * self._scale),
                    int(self._points[3][1] * self._scale))
                p_first = QPoint(
                    int(self._points[0][0] * self._scale),
                    int(self._points[0][1] * self._scale))
                painter.drawLine(p_last, p_first)

        painter.end()
        self.setPixmap(scaled)


class ManualCalibrationDialog(QDialog):
    """手动校准对话框 — 用户在图片中点击面包板 4 个角点.

    使用方法:
        dialog = ManualCalibrationDialog(image, parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            points = dialog.points  # np.ndarray shape (4, 2)
    """

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("手动校准 — 按顺序点击面包板 4 个角点")
        self.setMinimumSize(900, 650)
        self.resize(1100, 750)

        self.points: np.ndarray = None

        layout = QVBoxLayout(self)

        # 提示
        hint = QLabel(
            "请按顺序点击面包板的 4 个角点: "
            "1.左上 → 2.右上 → 3.右下 → 4.左下\n"
            "点击 '撤销' 可以回退, 标记完 4 个点后点击 '确认' 完成校准"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 13px; padding: 8px; color: #e0e0e0;")
        layout.addWidget(hint)

        # 图片区
        self._img_label = ImageCalibrationLabel()
        self._img_label.set_image(image)
        self._img_label.point_added.connect(self._on_point_added)
        layout.addWidget(self._img_label, stretch=1)

        # 状态
        self._status = QLabel("已标记: 0/4")
        self._status.setStyleSheet("font-size: 12px; padding: 4px; color: #aaa;")
        layout.addWidget(self._status)

        # 按钮区
        btn_layout = QHBoxLayout()

        self._btn_undo = QPushButton("撤销")
        self._btn_undo.clicked.connect(self._undo)
        btn_layout.addWidget(self._btn_undo)

        self._btn_clear = QPushButton("清除")
        self._btn_clear.clicked.connect(self._clear)
        btn_layout.addWidget(self._btn_clear)

        btn_layout.addStretch()

        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self._btn_cancel)

        self._btn_ok = QPushButton("确认校准")
        self._btn_ok.setEnabled(False)
        self._btn_ok.setStyleSheet(
            "QPushButton:enabled { background-color: #4CAF50; color: white; }"
        )
        self._btn_ok.clicked.connect(self._confirm)
        btn_layout.addWidget(self._btn_ok)

        layout.addLayout(btn_layout)

    def _on_point_added(self, count: int):
        self._status.setText(f"已标记: {count}/4")
        self._btn_ok.setEnabled(count == 4)

    def _undo(self):
        self._img_label.undo_last()
        count = self._img_label.point_count
        self._status.setText(f"已标记: {count}/4")
        self._btn_ok.setEnabled(count == 4)

    def _clear(self):
        self._img_label.clear_points()
        self._status.setText("已标记: 0/4")
        self._btn_ok.setEnabled(False)

    def _confirm(self):
        if self._img_label.point_count == 4:
            self.points = self._img_label.get_points()
            self.accept()


class CalibrationHelper:
    """面包板校准辅助 — 自动检测 + Qt手动校准 + 全图回退

    日志通过 on_log / on_status 回调通知 UI, 不直接操作 GUI 组件.
    """

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.on_log = None       # Callable[[str], None]
        self.on_status = None    # Callable[[str, bool, str], None]

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(msg)

    def auto_detect_board(self, frame: np.ndarray) -> bool:
        """自动检测面包板区域并执行完整校准.

        Returns:
            True 如果成功
        """
        if frame is None:
            return False

        ctx = self.ctx
        if ctx.calibrator.auto_calibrate(frame):
            hole_count = len(ctx.calibrator.hole_centers)
            roi = ctx.calibrator.get_roi_rect(
                frame.shape, padding=vision_cfg.roi_padding
            )
            ctx._roi_rect = roi
            self._log(
                f"自动检测到面包板, {hole_count} 个孔洞, "
                f"ROI: {roi}"
            )
            if self.on_status:
                self.on_status("calibr", True, f"{hole_count} 孔洞 (自动)")
            return True
        else:
            self._log("自动检测失败, 请使用手动校准 (点击4个角点)")
            return False

    def manual_calibrate(self, frame: np.ndarray, parent_widget=None) -> bool:
        """打开 Qt 手动校准对话框.

        Args:
            frame: 用于校准的图片 (BGR)
            parent_widget: 父窗口

        Returns:
            True 如果校准成功
        """
        dialog = ManualCalibrationDialog(frame, parent_widget)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._log("手动校准已取消")
            return False

        points = dialog.points
        if points is None or len(points) != 4:
            self._log("手动校准: 角点数据无效")
            return False

        ctx = self.ctx
        ctx.calibrator.calibrate(points)
        warped = ctx.calibrator.warp(frame)
        hole_count = ctx.calibrator.detect_holes(warped)

        roi = ctx.calibrator.get_roi_rect(
            frame.shape, padding=vision_cfg.roi_padding
        )
        ctx._roi_rect = roi

        self._log(f"手动校准完成, 检测到 {hole_count} 个孔洞")
        if self.on_status:
            self.on_status("calibr", True, f"{hole_count} 孔洞 (手动)")
        return True

    def fallback_calibrate(self, frame: np.ndarray) -> bool:
        """全图回退校准 — 以整张图片为面包板区域."""
        if frame is None:
            return False

        ctx = self.ctx
        h, w = frame.shape[:2]
        margin = 0.05
        corners = np.array([
            [w * margin, h * margin],
            [w * (1 - margin), h * margin],
            [w * (1 - margin), h * (1 - margin)],
            [w * margin, h * (1 - margin)],
        ], dtype=np.float32)

        ctx.calibrator.calibrate(corners)
        warped = ctx.calibrator.warp(frame)
        hole_count = ctx.calibrator.detect_holes(warped)

        roi = ctx.calibrator.get_roi_rect(
            frame.shape, padding=vision_cfg.roi_padding
        )
        ctx._roi_rect = roi

        self._log(f"全图回退校准完成, {hole_count} 个孔洞")
        if self.on_status:
            self.on_status("calibr", True, f"{hole_count} 孔洞 (回退)")
        return True
