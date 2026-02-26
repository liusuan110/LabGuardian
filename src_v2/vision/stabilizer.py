"""
多帧检测稳定化模块
职责：对连续帧的检测结果进行滑窗投票，消除单帧抖动/误检
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .detector import Detection


@dataclass
class StableDetection(Detection):
    """稳定后的检测结果，附加稳定性信息"""
    stability: float = 0.0     # 稳定性得分 (0~1)
    frame_count: int = 0       # 在滑窗内出现的帧数


class DetectionStabilizer:
    """
    多帧检测结果稳定器
    
    原理: 维护一个滑动窗口 (默认5帧)，
    只有在窗口内同一位置同一类别出现 ≥ threshold 次才确认为有效检测
    """

    def __init__(self, window_size: int = 5, min_hits: int = 3, 
                 iou_threshold: float = 0.3):
        """
        Args:
            window_size: 滑窗大小 (帧数)
            min_hits: 最少出现次数才确认
            iou_threshold: 判定"同一位置"的 IoU 阈值
        """
        self.window_size = window_size
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._history: List[List[Detection]] = []

    def update(self, detections: List[Detection]) -> List[StableDetection]:
        """
        输入当前帧的检测结果，输出稳定后的检测结果
        
        Args:
            detections: 当前帧的原始检测列表
            
        Returns:
            稳定后的检测列表 (仅包含通过投票的检测)
        """
        self._history.append(detections)
        if len(self._history) > self.window_size:
            self._history.pop(0)

        # 如果还没攒够帧数，直接返回当前帧结果
        if len(self._history) < self.min_hits:
            return [
                StableDetection(
                    class_name=d.class_name, class_id=d.class_id,
                    confidence=d.confidence, bbox=d.bbox,
                    pin1_pixel=d.pin1_pixel, pin2_pixel=d.pin2_pixel,
                    is_obb=d.is_obb, obb_corners=d.obb_corners,
                    stability=1.0 / self.min_hits, frame_count=1,
                )
                for d in detections
            ]

        # 对当前帧的每个检测，统计历史窗口内的投票
        stable_results = []
        for det in detections:
            hit_count = 0
            conf_sum = 0.0

            for frame_dets in self._history:
                for hist_det in frame_dets:
                    if (hist_det.class_name == det.class_name and
                            self._compute_iou(det.bbox, hist_det.bbox) >= self.iou_threshold):
                        hit_count += 1
                        conf_sum += hist_det.confidence
                        break  # 每帧最多计一次

            if hit_count >= self.min_hits:
                avg_conf = conf_sum / hit_count
                stable_results.append(StableDetection(
                    class_name=det.class_name,
                    class_id=det.class_id,
                    confidence=avg_conf,
                    bbox=det.bbox,
                    pin1_pixel=det.pin1_pixel,
                    pin2_pixel=det.pin2_pixel,
                    is_obb=det.is_obb,
                    obb_corners=det.obb_corners,
                    stability=hit_count / len(self._history),
                    frame_count=hit_count,
                ))

        return stable_results

    def clear(self):
        """清空历史"""
        self._history.clear()

    @staticmethod
    def _compute_iou(box1: Tuple[int, int, int, int],
                     box2: Tuple[int, int, int, int]) -> float:
        """计算两个矩形框的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0
