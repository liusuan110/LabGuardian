"""
视觉引脚-孔洞局部验证器 (Visual Pin-Hole Verifier)
===================================================
当 YOLO-Pose 关键点置信度不足（元件遮挡引脚）时，
对估计位置附近的少量孔洞做局部图像分析，判断哪些孔洞被占用。

设计原则:
  - 仅分析元件附近的 10-30 个孔洞，而非全图 650+ 个
  - 无需全局统计建模，用简单对比度阈值判断占用
  - 电气约束评分复用 pin_utils.score_electrical_constraints()

使用流程:
    verifier = PinHoleVerifier()
    loc1, loc2 = verifier.find_pins_locally(
        frame, calibrator, det, comp_type)
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from vision.pin_utils import (
    get_pin_props, score_electrical_constraints, RAIL_ROWS,
)

logger = logging.getLogger(__name__)


@dataclass
class HoleCandidate:
    """单个候选孔洞"""
    warped_pos: Tuple[float, float]   # 矫正空间坐标
    logic_loc: Tuple[str, str]        # 逻辑坐标 ("行号", "列字母")
    occupancy: float                  # 占用评分 (0-1, 越高越可能被占用)
    distance: float                   # 到估计位置的距离


class PinHoleVerifier:
    """
    引脚孔洞局部视觉验证器

    对单个元件的估计引脚位置附近做局部分析，
    通过图像对比度判断哪些孔洞被引脚占用。

    替代原 PinHoleDetector 的全图扫描方案。
    """

    # 对比度阈值: 空洞 contrast > 此值; 被占用 contrast < 此值
    CONTRAST_EMPTY_THRESHOLD = 10.0

    def __init__(self):
        self._patch_radius = 12

    # ================================================================
    # 公开接口
    # ================================================================

    def find_pins_locally(
        self,
        frame: np.ndarray,
        calibrator,
        det,
        comp_type: str = "",
    ) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
        """
        为单个元件查找引脚插入孔洞 (局部视觉分析).

        流程:
          1. 将估计引脚位置变换到矫正空间
          2. 在附近搜索孔洞 → 提取 patch → 计算占用评分
          3. 结合占用分 + 距离 + 电气约束选最佳引脚对

        Args:
            frame: 原始 BGR 帧 (高分辨率)
            calibrator: 已校准的 BreadboardCalibrator
            det: Detection 对象 (含 pin1_pixel, pin2_pixel)
            comp_type: 元件类型名

        Returns:
            (loc1, loc2) 或 (None, None)
        """
        if (not calibrator.is_calibrated or calibrator.matrix is None or
                det.pin1_pixel is None or det.pin2_pixel is None):
            return None, None

        # 估计 patch 半径
        self._patch_radius = self._estimate_patch_radius(calibrator)

        # 变换到矫正空间
        p1_w = self._to_warped(det.pin1_pixel, calibrator.matrix)
        p2_w = self._to_warped(det.pin2_pixel, calibrator.matrix)

        # 搜索半径
        props = get_pin_props(comp_type)
        spacing = self._get_spacing(calibrator)
        axis_len = float(np.linalg.norm(p2_w - p1_w))
        search_radius = max(spacing * props["search_mult"], axis_len * 0.6)

        # 预计算灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv_matrix = np.linalg.inv(calibrator.matrix)

        # 为每个引脚收集候选并评分
        cands1 = self._score_nearby_holes(
            p1_w, frame, gray, calibrator, inv_matrix, search_radius)
        cands2 = self._score_nearby_holes(
            p2_w, frame, gray, calibrator, inv_matrix, search_radius)

        if not cands1 or not cands2:
            return None, None

        # 选最佳引脚对
        best = self._select_best_pair(cands1, cands2, comp_type, spacing)
        if best is None:
            return None, None

        h1, h2 = best
        return h1.logic_loc, h2.logic_loc

    # ================================================================
    # 局部分析
    # ================================================================

    def _score_nearby_holes(
        self,
        center_warped: np.ndarray,
        frame: np.ndarray,
        gray: np.ndarray,
        calibrator,
        inv_matrix: np.ndarray,
        search_radius: float,
    ) -> List[HoleCandidate]:
        """对指定位置附近的孔洞做视觉评分.

        只分析搜索范围内的孔洞 (通常 10-30 个)。
        用简单的对比度特征判断占用状态。
        """
        cx, cy = float(center_warped[0]), float(center_warped[1])
        r = self._patch_radius
        h_img, w_img = frame.shape[:2]
        candidates = []

        for hx, hy in calibrator.hole_centers:
            dist = np.sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
            if dist > search_radius:
                continue

            loc = calibrator.hole_to_logic(hx, hy)
            if loc is None or loc[0] == "Groove":
                continue

            # 逆变换到原始帧坐标
            pt_w = np.array([[[hx, hy]]], dtype=np.float32)
            pt_f = cv2.perspectiveTransform(pt_w, inv_matrix)[0][0]
            fx, fy = int(round(pt_f[0])), int(round(pt_f[1]))

            # 边界检查
            if fx - r < 0 or fy - r < 0 or fx + r >= w_img or fy + r >= h_img:
                continue

            # 提取 patch 并计算占用评分
            patch = gray[fy - r:fy + r + 1, fx - r:fx + r + 1]
            if patch.shape[0] != 2 * r + 1 or patch.shape[1] != 2 * r + 1:
                continue

            occ_score = self._compute_occupancy(patch, r)

            candidates.append(HoleCandidate(
                warped_pos=(hx, hy),
                logic_loc=loc,
                occupancy=occ_score,
                distance=dist,
            ))

        # 按占用分排序, 取 Top-15
        candidates.sort(key=lambda c: c.occupancy, reverse=True)
        return candidates[:15]

    @staticmethod
    def _compute_occupancy(patch_gray: np.ndarray, r: int) -> float:
        """计算单个孔洞的占用概率.

        核心特征: 中心-周围对比度
          - 空洞: 中心暗, 周围亮 → contrast > 0
          - 被占用: 中心被引脚填充 → contrast ≈ 0 或 < 0

        对比度消失或反转 → 占用概率高
        """
        cr = max(2, r // 3)  # 中心圆半径

        # 中心区域
        Y, X = np.ogrid[:2*r+1, :2*r+1]
        center_mask = ((X - r) ** 2 + (Y - r) ** 2) <= cr ** 2
        surround_mask = (((X - r)**2 + (Y - r)**2) > cr**2) & \
                        (((X - r)**2 + (Y - r)**2) <= r**2)

        ci = float(np.mean(patch_gray[center_mask]))
        si = float(np.mean(patch_gray[surround_mask]))
        contrast = si - ci  # 正 = 空洞, 低/负 = 被占用

        # 简单 Sigmoid 映射: contrast 越低 → 占用概率越高
        # 当 contrast < THRESHOLD 时开始判定为被占用
        x = (10.0 - contrast) / 8.0  # 归一化, contrast=10 → x=0, contrast=-6 → x=2
        score = 1.0 / (1.0 + np.exp(-x * 3.0))
        return float(np.clip(score, 0.0, 1.0))

    # ================================================================
    # 最佳引脚对选择
    # ================================================================

    @staticmethod
    def _select_best_pair(
        cands1: List[HoleCandidate],
        cands2: List[HoleCandidate],
        comp_type: str,
        spacing: float,
    ) -> Optional[Tuple[HoleCandidate, HoleCandidate]]:
        """综合占用分、距离和电气约束选最佳引脚对.

        评分 = 占用分×0.5 + 近距×0.3 - 约束惩罚×0.2
        """
        from logic.circuit import norm_component_type
        ntype = norm_component_type(comp_type)
        is_wire = (ntype == "Wire")

        best_score = -float('inf')
        best_pair = None

        # 只比较 Top-10 × Top-10 (已按占用分排序)
        for c1 in cands1[:10]:
            for c2 in cands2[:10]:
                # 不能是同一个孔洞
                if (abs(c1.warped_pos[0] - c2.warped_pos[0]) < 1 and
                        abs(c1.warped_pos[1] - c2.warped_pos[1]) < 1):
                    continue

                # 最小引脚间距
                pin_sep = np.sqrt(
                    (c1.warped_pos[0] - c2.warped_pos[0]) ** 2 +
                    (c1.warped_pos[1] - c2.warped_pos[1]) ** 2)
                if not is_wire and pin_sep < spacing * 0.8:
                    continue

                # 占用分 (均值, 0-1)
                occ = (c1.occupancy + c2.occupancy) / 2.0

                # 距离分 (反比, 归一化)
                max_dist = max(c1.distance, c2.distance, 1.0)
                dist_score = max(0.0, 1.0 - (c1.distance + c2.distance) / (2.0 * max_dist + spacing))

                # 电气约束惩罚 (归一化到 0-1 范围)
                raw_penalty = score_electrical_constraints(
                    c1.logic_loc, c2.logic_loc, comp_type)
                norm_penalty = min(raw_penalty / 100.0, 1.0)

                total = occ * 0.50 + dist_score * 0.30 - norm_penalty * 0.20

                if total > best_score:
                    best_score = total
                    best_pair = (c1, c2)

        return best_pair

    # ================================================================
    # 工具函数
    # ================================================================

    @staticmethod
    def _get_spacing(calibrator) -> float:
        if calibrator.row_centers is not None and len(calibrator.row_centers) > 1:
            return float(np.mean(np.diff(calibrator.row_centers)))
        return 12.0

    def _estimate_patch_radius(self, calibrator) -> int:
        """动态估计原始帧中的 patch 半径"""
        spacing = self._get_spacing(calibrator)
        if calibrator.matrix is not None:
            try:
                inv_m = np.linalg.inv(calibrator.matrix)
                cw = calibrator.width / 2
                ch = calibrator.height / 2
                p1 = np.array([[[cw, ch]]], dtype=np.float32)
                p2 = np.array([[[cw + spacing, ch]]], dtype=np.float32)
                f1 = cv2.perspectiveTransform(p1, inv_m)[0][0]
                f2 = cv2.perspectiveTransform(p2, inv_m)[0][0]
                frame_spacing = float(np.linalg.norm(f2 - f1))
            except Exception:
                frame_spacing = spacing * 3
        else:
            frame_spacing = spacing * 3

        radius = max(5, int(frame_spacing * 0.45))
        return min(radius, 50)

    @staticmethod
    def _to_warped(pixel: Tuple[float, float], matrix: np.ndarray) -> np.ndarray:
        """单点像素坐标 → 矫正空间"""
        p = np.array([[[pixel[0], pixel[1]]]], dtype=np.float32)
        w = cv2.perspectiveTransform(p, matrix)
        return w[0][0]
