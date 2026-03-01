"""
YOLO 推理封装模块
职责：加载模型、执行推理、返回结构化检测结果
不涉及任何 GUI 逻辑
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import vision as vision_cfg, find_best_yolo_model, TWO_PIN_COMPONENTS


# ============================================================
# 引脚延伸先验 — 补偿元件遮挡导致的引脚位置偏差
# ============================================================
# OBB / HBB 只捕获元件本体, 真实引脚插入点在本体外侧。
# 比例含义: 沿长轴方向, 每端向外延伸 (长边长度 × 比例) 个像素。
# 参考: 电阻色环体 ≈ 6mm, 引线外露 ≈ 2-4mm → 比例 0.08-0.12

_PIN_EXTENSION = {
    "RESISTOR":       0.10,  # 电阻: 引线细长, 略超出体边
    "Resistor":       0.10,
    "LED":            0.08,  # LED: 引脚从底座伸出, 顶视遮挡严重
    "DIODE":          0.10,  # 二极管: 类似电阻
    "CAPACITOR":      0.08,  # 电容: 引脚较短
    "Wire":           0.02,  # 导线: 端点基本就是连接点
    "Push_Button":    0.06,  # 按钮: 引脚在底部
    "TRANSISTOR":     0.10,  # 三极管: TO-92 引脚从封装体伸出
    "IC_DIP":         0.02,  # IC: 引脚紧贴封装体边缘
    "POTENTIOMETER":  0.08,  # 电位器: 引脚在底部
}
_DEFAULT_PIN_EXTENSION = 0.08


@dataclass
class Detection:
    """单个检测结果的结构化表示"""
    class_name: str                           # 类别名称 (e.g. "RESISTOR")
    class_id: int                             # 类别 ID
    confidence: float                         # 置信度
    bbox: Tuple[int, int, int, int]           # 外接矩形 (x1, y1, x2, y2)
    pin1_pixel: Optional[Tuple[float, float]] = None  # 引脚1像素坐标
    pin2_pixel: Optional[Tuple[float, float]] = None  # 引脚2像素坐标
    is_obb: bool = False                      # 是否为旋转检测结果
    obb_corners: Optional[np.ndarray] = None  # OBB 四角坐标 (4,2)
    wire_color: Optional[str] = None          # 导线颜色 (仅 Wire 类型, 由 WireAnalyzer 填充)


class ComponentDetector:
    """
    电子元件检测器
    封装 YOLO 模型的加载和推理逻辑
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model: Optional[YOLO] = None
        self.model_path = model_path
        self.is_obb_model = False

    def load(self) -> bool:
        """加载模型，返回是否成功"""
        try:
            if self.model_path is None:
                self.model_path = find_best_yolo_model()

            self.model = YOLO(str(self.model_path))
            self.is_obb_model = getattr(self.model, 'task', 'detect') == 'obb'
            print(f"[Detector] 模型加载成功: {self.model_path}")
            print(f"[Detector] 模型类型: {'OBB (旋转检测)' if self.is_obb_model else 'HBB (标准检测)'}")
            return True
        except Exception as e:
            print(f"[Detector] 模型加载失败: {e}")
            # 回退到通用权重
            try:
                self.model = YOLO("yolov8n.pt")
                self.is_obb_model = False
                print("[Detector] 回退到 yolov8n.pt")
                return True
            except Exception as e2:
                print(f"[Detector] 回退也失败: {e2}")
                return False

    def detect(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Detection]:
        """
        对单帧图像执行检测
        
        Args:
            frame: BGR 图像 (numpy array)
            conf: 置信度阈值覆盖, None 则使用配置默认值
            
        Returns:
            List[Detection]: 结构化检测结果列表
        """
        if self.model is None:
            return []

        threshold = conf if conf is not None else vision_cfg.conf_threshold

        results = self.model(
            frame,
            verbose=False,
            conf=threshold,
            imgsz=vision_cfg.imgsz,
            iou=vision_cfg.iou_threshold,
        )

        detections = []
        result = results[0]

        if self.is_obb_model and result.obb is not None:
            detections = self._parse_obb_results(result)
        elif result.boxes is not None:
            detections = self._parse_hbb_results(result)

        return detections

    def _parse_obb_results(self, result) -> List[Detection]:
        """解析 OBB 旋转检测结果"""
        detections = []
        for obb in result.obb:
            cls_id = int(obb.cls[0])
            cls_name = result.names[cls_id]
            conf = float(obb.conf[0])

            corners = obb.xyxyxyxy[0].cpu().numpy().reshape(4, 2)
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]
            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))

            pin1, pin2 = self._extract_pins_obb(cls_name, corners)

            detections.append(Detection(
                class_name=cls_name,
                class_id=cls_id,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                pin1_pixel=pin1,
                pin2_pixel=pin2,
                is_obb=True,
                obb_corners=corners,
            ))
        return detections

    def _parse_hbb_results(self, result) -> List[Detection]:
        """解析 HBB 标准检测结果"""
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            pin1, pin2 = self._extract_pins_hbb(cls_name, x1, y1, x2, y2, w, h, cx, cy)

            detections.append(Detection(
                class_name=cls_name,
                class_id=cls_id,
                confidence=conf,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                pin1_pixel=pin1,
                pin2_pixel=pin2,
                is_obb=False,
            ))
        return detections

    @staticmethod
    def _extract_pins_obb(cls_name: str, corners: np.ndarray):
        """从 OBB 旋转框中提取两端引脚坐标 (含遮挡补偿延伸)。

        元件本体遮挡引脚时，OBB 短边中点 ≠ 真实引脚插入点。
        沿长轴方向向外延伸，使估计位置更接近实际孔洞。
        """
        p0, p1, p2, p3 = corners
        d01 = np.linalg.norm(p0 - p1)
        d12 = np.linalg.norm(p1 - p2)

        if cls_name in TWO_PIN_COMPONENTS:
            if d01 < d12:
                # d01 是短边, d12 是长边
                mid1 = (p0 + p1) / 2   # 短边中点 → pin1 初始估计
                mid2 = (p2 + p3) / 2   # 对侧短边中点 → pin2 初始估计
                long_len = d12
            else:
                mid1 = (p1 + p2) / 2
                mid2 = (p3 + p0) / 2
                long_len = d01

            # 沿长轴向外延伸, 补偿元件遮挡
            ext_ratio = _PIN_EXTENSION.get(cls_name, _DEFAULT_PIN_EXTENSION)
            direction = mid2 - mid1
            dir_len = np.linalg.norm(direction)
            if dir_len > 1e-5:
                unit_dir = direction / dir_len
                extension = ext_ratio * long_len
                pin1 = tuple((mid1 - unit_dir * extension).tolist())
                pin2 = tuple((mid2 + unit_dir * extension).tolist())
            else:
                pin1 = tuple(mid1.tolist())
                pin2 = tuple(mid2.tolist())
        else:
            # 非二端元件 (三极管等): 保持原策略
            cx = np.mean(corners[:, 0])
            cy = np.mean(corners[:, 1])
            h = np.max(corners[:, 1]) - np.min(corners[:, 1])
            pin1 = (cx, np.min(corners[:, 1]) + h * 0.15)
            pin2 = (cx, np.max(corners[:, 1]) - h * 0.15)

        return pin1, pin2

    @staticmethod
    def _extract_pins_hbb(cls_name, x1, y1, x2, y2, w, h, cx, cy):
        """从 HBB 标准框中提取两端引脚坐标 (含遮挡补偿延伸)。

        将引脚估计从 bbox 边缘向外延伸, 更接近真实插入点。
        """
        ext_ratio = _PIN_EXTENSION.get(cls_name, _DEFAULT_PIN_EXTENSION)
        if w > h:
            # 水平元件: 引脚在左右两端外侧
            extension = w * ext_ratio
            pin1 = (x1 - extension, cy)
            pin2 = (x2 + extension, cy)
        else:
            # 垂直元件: 引脚在上下两端外侧
            extension = h * ext_ratio
            pin1 = (cx, y1 - extension)
            pin2 = (cx, y2 + extension)
        return pin1, pin2

    def annotate_frame(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        在帧上绘制检测结果 (仅绘制，不修改检测逻辑)
        
        Args:
            frame: 原始帧 (会被复制)
            detections: 检测结果列表
            
        Returns:
            绘制了检测标注的帧
        """
        from config import CLASS_COLORS

        annotated = frame.copy()

        for det in detections:
            color = CLASS_COLORS.get(det.class_name, (128, 128, 128))

            if det.is_obb and det.obb_corners is not None:
                # 绘制旋转框
                pts = det.obb_corners.astype(int)
                cv2.polylines(annotated, [pts], True, color, 2)
            else:
                # 绘制水平框
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 标签文字
            label = f"{det.class_name} {det.confidence:.2f}"
            x1, y1 = det.bbox[:2]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 绘制引脚
            if det.pin1_pixel:
                cv2.circle(annotated, (int(det.pin1_pixel[0]), int(det.pin1_pixel[1])), 4, (255, 0, 0), -1)
            if det.pin2_pixel:
                cv2.circle(annotated, (int(det.pin2_pixel[0]), int(det.pin2_pixel[1])), 4, (0, 0, 255), -1)

            # 引脚连线
            if det.pin1_pixel and det.pin2_pixel:
                cv2.line(annotated,
                         (int(det.pin1_pixel[0]), int(det.pin1_pixel[1])),
                         (int(det.pin2_pixel[0]), int(det.pin2_pixel[1])),
                         (0, 255, 0), 2)

        return annotated

    @staticmethod
    def offset_detections(detections: List[Detection],
                          roi_x: int, roi_y: int) -> List[Detection]:
        """
        将 ROI 裁剪区域内的检测坐标偏移回原帧坐标系

        当 YOLO 在裁剪后的 ROI 子图上运行时, 所有坐标都相对于 ROI 左上角。
        此方法将 bbox / pin / obb_corners 全部加上 (roi_x, roi_y) 偏移。

        Args:
            detections: ROI 坐标系下的检测结果 (会被原地修改)
            roi_x: ROI 左上角在原帧中的 x 坐标
            roi_y: ROI 左上角在原帧中的 y 坐标

        Returns:
            偏移后的检测结果列表 (同一个列表, 原地修改)
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            det.bbox = (x1 + roi_x, y1 + roi_y, x2 + roi_x, y2 + roi_y)
            if det.pin1_pixel:
                det.pin1_pixel = (det.pin1_pixel[0] + roi_x,
                                  det.pin1_pixel[1] + roi_y)
            if det.pin2_pixel:
                det.pin2_pixel = (det.pin2_pixel[0] + roi_x,
                                  det.pin2_pixel[1] + roi_y)
            if det.obb_corners is not None:
                det.obb_corners = det.obb_corners + np.array([roi_x, roi_y])
        return detections
