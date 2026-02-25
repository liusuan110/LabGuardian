"""
LabGuardian 校准辅助
====================
从 MainWindow 中提取的面包板校准交互逻辑.
包含 OpenCV 手动校准窗口和自动检测.
"""

import logging
import threading
import numpy as np
import cv2

from config import vision as vision_cfg, camera as cam_cfg
from app_context import AppContext

logger = logging.getLogger(__name__)


class CalibrationHelper:
    """面包板校准辅助 — 手动校准 (OpenCV 窗口) + 自动检测

    日志通过 on_log / on_status 回调通知 UI, 不直接操作 GUI 组件.
    """

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.on_log = None       # Callable[[str], None]
        self.on_status = None    # Callable[[str, bool, str], None]
        # on_status(module_name, ok, detail) → dashboard.update_module_status

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(msg)

    def start_calibration(self, video_worker):
        """启动校准 (在子线程中打开 OpenCV 窗口)"""
        self._log("校准: 请在弹出窗口中点击面包板4个角点")
        threading.Thread(
            target=self._calibration_flow,
            args=(video_worker,),
            daemon=True,
        ).start()

    def _calibration_flow(self, video_worker):
        """校准交互 (在 OpenCV 窗口完成)"""
        if video_worker._source_mode == "image" and video_worker.static_frame is not None:
            frame = video_worker.static_frame.copy()
        else:
            cap = cv2.VideoCapture(cam_cfg.device_id)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self._log("无法获取帧用于校准")
                return

        points = []
        win_name = "Calibrate: Click 4 corners (TL->TR->BR->BL)"

        h, w = frame.shape[:2]
        max_w, max_h = 1000, 700
        scale = min(max_w / w, max_h / h, 1.0)
        disp = cv2.resize(frame, (int(w * scale), int(h * scale)))

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                real_x, real_y = int(x / scale), int(y / scale)
                points.append([real_x, real_y])

        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, on_click)

        while len(points) < 4:
            draw = disp.copy()
            for i, p in enumerate(points):
                sx, sy = int(p[0] * scale), int(p[1] * scale)
                cv2.circle(draw, (sx, sy), 5, (0, 0, 255), -1)
                cv2.putText(draw, str(i + 1), (sx + 10, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(win_name, draw)
            if cv2.waitKey(50) == ord('q'):
                cv2.destroyWindow(win_name)
                return

        cv2.destroyWindow(win_name)

        src_pts = np.float32(points)
        self.ctx.calibrator.calibrate(src_pts)

        warped = self.ctx.calibrator.warp(frame)
        hole_count = self.ctx.calibrator.detect_holes(warped)

        roi = self.ctx.calibrator.get_roi_rect(
            frame.shape, padding=vision_cfg.roi_padding
        )
        self.ctx._roi_rect = roi

        self._log(f"校准完成, 检测到 {hole_count} 个孔洞")
        if self.on_status:
            self.on_status("calibr", True, f"{hole_count} 孔洞")

    def auto_detect_board(self, frame: np.ndarray) -> bool:
        """加载图片后自动检测面包板区域并校准

        Returns:
            True 如果成功检测到面包板
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
            self._log("未检测到面包板, 请手动校准或调整拍摄角度")
            return False
