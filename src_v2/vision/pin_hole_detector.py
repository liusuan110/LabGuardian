"""
视觉引脚-孔洞占用检测器 (Visual Pin-in-Hole Detector)
=====================================================
通过分析面包板孔洞局部图像特征，直接判断哪些孔洞被元件引脚占用。

核心思路:
  面包板上大多数孔洞是空的，呈现为深色小圆洞。当元件引脚插入后，
  孔洞的中心被金属引脚遮挡，视觉特征发生显著变化：
    - 中心亮度升高（金属反光取代暗空洞）
    - 中心-周围对比度降低（不再是暗洞+亮面）
    - 梯度能量升高（引脚边缘产生强梯度）
    - 颜色多样性增加（金属色 vs 塑料）

算法:
  1. 将校准后的孔洞坐标逆变换到原始帧坐标 (高分辨率)
  2. 提取每个孔洞中心周围的局部 patch
  3. 对 patch 提取多维视觉特征 (亮度/对比度/梯度/颜色)
  4. 用鲁棒统计 (Median + MAD) 建立空孔洞基线模型
  5. 偏离基线程度 → 占用概率 (Sigmoid)
  6. 结合元件 OBB 位置，关联被占用孔洞与具体元件引脚

参考:
  - Otsu (1979). "A threshold selection method from gray-level histograms."
  - Rousseeuw & Croux (1993). Alternatives to MAD.
  - Fischler & Bolles (1981). "Random Sample Consensus" (鲁棒估计范式).
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HoleFeatures:
    """单个孔洞的视觉特征向量"""
    center_intensity: float     # 中心区域平均灰度 (0-255)
    surround_intensity: float   # 周围区域平均灰度 (0-255)
    contrast: float             # 周围-中心灰度差 (正=空洞, 低/负=被占用)
    gradient_energy: float      # 中心区域平均梯度幅值
    color_std: float            # 中心区域 BGR 标准差 (颜色多样性)


class PinHoleDetector:
    """
    视觉引脚-孔洞占用检测器

    通过图像分析直接判断面包板孔洞是否被元件引脚实际占用，
    替代传统的 OBB 几何延伸 + 最近邻猜测方法。

    使用流程:
        phd = PinHoleDetector()
        occ_map = phd.detect_occupied_holes(frame, calibrator)
        for det in detections:
            loc1, loc2 = phd.find_component_pins(det, calibrator, occ_map)
    """

    OCCUPANCY_THRESHOLD = 0.55  # 占用判定阈值 (从 0.45 提高，减少假阳性)

    def __init__(self):
        self._empty_model = None
        self._patch_radius_frame = 12

    # ================================================================
    # 公开接口
    # ================================================================

    def detect_occupied_holes(
        self,
        frame: np.ndarray,
        calibrator,
    ) -> Dict[Tuple[float, float], float]:
        """
        分析所有校准孔洞的占用状态

        Args:
            frame: 原始 BGR 帧 (高分辨率)
            calibrator: 已完成校准的 BreadboardCalibrator

        Returns:
            占用分数字典 {(warped_x, warped_y): score}
            score ∈ [0, 1], 越接近 1 越可能被占用
        """
        if (not calibrator.is_calibrated or
                calibrator.matrix is None or
                len(calibrator.hole_centers) < 20):
            return {}

        holes_warped = calibrator.hole_centers

        # 1. 估计帧空间 patch 尺寸
        self._patch_radius_frame = self._estimate_patch_radius(calibrator)
        r = self._patch_radius_frame

        # 2. 逆变换所有孔洞到原始帧坐标
        inv_matrix = np.linalg.inv(calibrator.matrix)
        holes_frame = self._batch_transform(holes_warped, inv_matrix)

        # 3. 预计算梯度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        h_img, w_img = frame.shape[:2]

        # 4. 预计算圆形 mask (只算一次，所有 patch 复用)
        patch_size = 2 * r + 1
        Y, X = np.ogrid[:patch_size, :patch_size]
        cr = max(2, r // 3)                               # 中心圆半径
        center_mask = ((X - r) ** 2 + (Y - r) ** 2) <= cr ** 2
        surround_mask = (((X - r) ** 2 + (Y - r) ** 2) > cr ** 2) & \
                        (((X - r) ** 2 + (Y - r) ** 2) <= r ** 2)

        # 5. 遍历所有孔洞提取特征
        features_list: List[HoleFeatures] = []
        valid_indices: List[int] = []

        for i, (fx, fy) in enumerate(holes_frame):
            ix, iy = int(round(fx)), int(round(fy))

            # 边界检查
            if ix - r < 0 or iy - r < 0 or ix + r >= w_img or iy + r >= h_img:
                continue

            patch_gray = gray[iy - r:iy + r + 1, ix - r:ix + r + 1]
            patch_bgr = frame[iy - r:iy + r + 1, ix - r:ix + r + 1]
            patch_grad = grad_mag[iy - r:iy + r + 1, ix - r:ix + r + 1]

            if patch_gray.shape[0] != patch_size or patch_gray.shape[1] != patch_size:
                continue

            feat = self._extract_features(
                patch_bgr, patch_gray, patch_grad, center_mask, surround_mask
            )
            features_list.append(feat)
            valid_indices.append(i)

        if not features_list:
            return {}

        # 6. 建立空孔洞基线模型
        self._empty_model = self._build_empty_model(features_list)

        # 7. 计算占用分数
        occupancy_map: Dict[Tuple[float, float], float] = {}
        for feat, idx in zip(features_list, valid_indices):
            score = self._compute_occupancy_score(feat)
            hx, hy = holes_warped[idx]
            occupancy_map[(hx, hy)] = score

        occ_count = sum(1 for s in occupancy_map.values()
                        if s > self.OCCUPANCY_THRESHOLD)

        # 8. 线状簇滤波: 排除可能由导线阴影产生的假阳性
        #    真实引脚插入仅影响 1-2 个孔洞; 导线经过时会在一条线上
        #    造成多个相邻孔洞特征异常 → 这些是假阳性
        occupancy_map = self._filter_wire_shadow_clusters(
            occupancy_map, calibrator)

        occ_count_after = sum(1 for s in occupancy_map.values()
                              if s > self.OCCUPANCY_THRESHOLD)
        if occ_count_after < occ_count:
            logger.info(
                f"[PinHole] 线状簇滤波: {occ_count} → {occ_count_after} 被占用孔洞")

        logger.info(
            f"[PinHole] 分析 {len(occupancy_map)} 孔洞, "
            f"{occ_count_after} 个疑似被占用 (阈值>{self.OCCUPANCY_THRESHOLD:.2f})"
        )
        return occupancy_map

    def find_component_pins(
        self,
        det,
        calibrator,
        occupancy_map: Dict[Tuple[float, float], float],
        comp_type: str = "",
    ) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
        """
        为检测到的元件确定其两个引脚插入的孔洞逻辑坐标

        v2 改进:
          - 自适应搜索半径: 根据元件类型和 OBB 尺寸动态调整
          - 轴向约束: 候选孔洞必须在元件主轴附近 (减少相邻行误匹配)
          - 分离约束: 两引脚间距必须合理

        Args:
            det: Detection 对象
            calibrator: BreadboardCalibrator
            occupancy_map: 占用分数字典
            comp_type: 元件类型名称

        Returns:
            (loc1, loc2): 两引脚的逻辑坐标, 如 ("41", "f") ("45", "d")
            任一为 None 表示未能确定
        """
        if (not calibrator.is_calibrated or calibrator.matrix is None or
                det.pin1_pixel is None or det.pin2_pixel is None):
            return None, None

        # 将估计引脚位置变换到矫正空间
        p1_w = self._to_warped(det.pin1_pixel, calibrator.matrix)
        p2_w = self._to_warped(det.pin2_pixel, calibrator.matrix)

        # 元件轴向
        axis = p2_w - p1_w
        axis_len = float(np.linalg.norm(axis))

        # 孔洞间距
        if calibrator.row_centers is not None and len(calibrator.row_centers) > 1:
            spacing = float(np.mean(np.diff(calibrator.row_centers)))
        else:
            spacing = 12.0

        # ---- 自适应搜索半径 (根据元件类型 + OBB 尺寸) ----
        from logic.circuit import norm_component_type
        ntype = norm_component_type(comp_type)
        if ntype == "Resistor":
            # 电阻引脚长, 弯折角度大 → 需要更大搜索范围
            search_radius = max(spacing * 5.0, axis_len * 0.6)
        elif ntype in ("LED", "Diode"):
            search_radius = max(spacing * 4.0, axis_len * 0.5)
        else:
            search_radius = max(spacing * 3.5, axis_len * 0.5)

        # ---- 轴向约束 (元件足够长时启用) ----
        # 候选孔洞必须在元件主轴线附近，排除相邻行的干扰
        use_axis = axis_len > spacing * 1.5
        if use_axis:
            axis_unit = axis / axis_len
            perp_unit = np.array([-axis_unit[1], axis_unit[0]])
            # 允许偏离轴线最多 2 个孔间距 (覆盖引脚弯折)
            perp_tol = spacing * 2.0
        else:
            perp_unit = None
            perp_tol = None

        # 收集轴向约束的候选
        cands1 = self._get_candidates_near(
            p1_w, occupancy_map, search_radius, perp_unit, perp_tol)
        cands2 = self._get_candidates_near(
            p2_w, occupancy_map, search_radius, perp_unit, perp_tol)

        # 轴向约束太严找不到候选时，放宽重试
        if (not cands1 or not cands2) and use_axis:
            cands1 = self._get_candidates_near(
                p1_w, occupancy_map, search_radius, None, None)
            cands2 = self._get_candidates_near(
                p2_w, occupancy_map, search_radius, None, None)

        if not cands1 or not cands2:
            return None, None

        # 选择最佳引脚对
        best_pair = self._select_best_pair(
            cands1, cands2, calibrator, comp_type, search_radius
        )

        if best_pair is None:
            return None, None

        h1, h2 = best_pair
        loc1 = calibrator.hole_to_logic(h1[0], h1[1])
        loc2 = calibrator.hole_to_logic(h2[0], h2[1])
        return loc1, loc2

    # ================================================================
    # 特征提取
    # ================================================================

    @staticmethod
    def _extract_features(
        patch_bgr: np.ndarray,
        patch_gray: np.ndarray,
        patch_grad: np.ndarray,
        center_mask: np.ndarray,
        surround_mask: np.ndarray,
    ) -> HoleFeatures:
        """从一个孔洞 patch 提取视觉特征"""
        center_pix = patch_gray[center_mask]
        surr_pix = patch_gray[surround_mask]

        ci = float(np.mean(center_pix)) if len(center_pix) > 0 else 128.0
        si = float(np.mean(surr_pix)) if len(surr_pix) > 0 else 128.0
        contrast = si - ci    # 空洞: 正(周围亮中心暗); 被占用: 趋近0或负

        center_grad = patch_grad[center_mask]
        ge = float(np.mean(center_grad)) if len(center_grad) > 0 else 0.0

        center_bgr = patch_bgr[center_mask]
        cs = float(np.std(center_bgr.astype(np.float32))) if len(center_bgr) > 0 else 0.0

        return HoleFeatures(
            center_intensity=ci,
            surround_intensity=si,
            contrast=contrast,
            gradient_energy=ge,
            color_std=cs,
        )

    # ================================================================
    # 基线模型 + 评分
    # ================================================================

    @staticmethod
    def _build_empty_model(features: List[HoleFeatures]) -> dict:
        """
        用鲁棒统计建立空孔洞基线

        多数孔洞为空 → 中位数 = 典型空孔洞
        MAD (Median Absolute Deviation) = 鲁棒离散度度量
        """
        eps = 1e-6
        contrasts = np.array([f.contrast for f in features])
        centers = np.array([f.center_intensity for f in features])
        grads = np.array([f.gradient_energy for f in features])
        colors = np.array([f.color_std for f in features])

        return {
            'contrast_med': float(np.median(contrasts)),
            'contrast_mad': float(np.median(np.abs(contrasts - np.median(contrasts)))) + eps,
            'center_med': float(np.median(centers)),
            'center_mad': float(np.median(np.abs(centers - np.median(centers)))) + eps,
            'grad_med': float(np.median(grads)),
            'grad_mad': float(np.median(np.abs(grads - np.median(grads)))) + eps,
            'color_med': float(np.median(colors)),
            'color_mad': float(np.median(np.abs(colors - np.median(colors)))) + eps,
        }

    def _compute_occupancy_score(self, feat: HoleFeatures) -> float:
        """
        计算单个孔洞的占用概率

        用 Modified Z-score (基于 MAD) 衡量偏离基线程度:
          - 对比度降低 → z_contrast 升高 → 被占用
          - 中心变亮 → z_center 升高 → 被占用
          - 梯度增大 → z_grad 升高 → 被占用
          - 颜色多样 → z_color 升高 → 被占用

        v2: 增加绝对特征门控，减少假阳性:
          - 中心仍暗 + 对比度仍正 → 空洞特征明显 → 压制 z 分数
          - Sigmoid 中心从 2σ 提高到 3σ，需要更强偏离才判定为被占用
        """
        m = self._empty_model
        if m is None:
            return 0.0

        z_contrast = (m['contrast_med'] - feat.contrast) / m['contrast_mad']
        z_center = (feat.center_intensity - m['center_med']) / m['center_mad']
        z_grad = (feat.gradient_energy - m['grad_med']) / m['grad_mad']
        z_color = (feat.color_std - m['color_med']) / m['color_mad']

        # === 绝对特征门控 (减少因纹理/光照引起的假阳性) ===
        # 空洞特征: 中心暗 (低灰度) + 对比度高 (周围亮中心暗)
        # 如果这些绝对特征仍呈现空洞模式, 压制 z 分数
        if feat.contrast > 15 and feat.center_intensity < m['center_med'] * 0.85:
            z_contrast = min(z_contrast, 0.5)
            z_center = min(z_center, 0.5)

        # 如果对比度为负 (中心比周围更亮), 这是强被占用信号, 给予额外加分
        if feat.contrast < -5:
            z_contrast = max(z_contrast, 2.0)

        raw = (z_contrast * 0.40 +     # 提高对比度权重 (最具区分度)
               z_center * 0.25 +
               z_grad * 0.20 +          # 略降梯度权重 (噪声敏感)
               z_color * 0.15)

        # Sigmoid: 需要 ~3σ 偏离才达到 0.5 (从 2σ 提高, 更严格)
        score = 1.0 / (1.0 + np.exp(-(raw - 3.0)))
        return float(np.clip(score, 0.0, 1.0))

    # ================================================================
    # 候选搜索 + 最佳配对
    # ================================================================

    def _get_candidates_near(
        self,
        center: np.ndarray,
        occupancy_map: Dict,
        search_radius: float,
        perp_unit: Optional[np.ndarray] = None,
        perp_tolerance: Optional[float] = None,
    ) -> List[Tuple[Tuple[float, float], float, float]]:
        """
        在搜索区域内收集候选孔洞

        v2 改进: 增加轴向垂直距离约束，排除偏离元件主轴的孔洞。

        Args:
            center: 搜索中心 (warped 坐标)
            occupancy_map: 占用分数字典
            search_radius: 搜索半径
            perp_unit: 轴向垂直单位向量 (None=不约束)
            perp_tolerance: 最大垂直偏离距离 (None=不约束)

        Returns:
            [(hole_warped_pos, occupancy_score, distance), ...]
        """
        cx, cy = float(center[0]), float(center[1])
        all_cands = []

        for (hx, hy), score in occupancy_map.items():
            dist = np.sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
            if dist > search_radius:
                continue

            # 轴向垂直距离约束: 排除偏离元件主轴的孔洞
            if perp_unit is not None and perp_tolerance is not None:
                v = np.array([hx - cx, hy - cy])
                perp_dist = abs(float(np.dot(v, perp_unit)))
                if perp_dist > perp_tolerance:
                    continue

            all_cands.append(((hx, hy), score, dist))

        if not all_cands:
            return []

        # 混合选择: 按占用分数的前10 + 按距离的前10，去重
        by_occ = sorted(all_cands, key=lambda x: x[1], reverse=True)[:10]
        by_dist = sorted(all_cands, key=lambda x: x[2])[:10]

        seen = set()
        result = []
        for c in by_occ + by_dist:
            key = c[0]
            if key not in seen:
                seen.add(key)
                result.append(c)

        return result[:15]

    @staticmethod
    def _select_best_pair(
        cands1, cands2,
        calibrator,
        comp_type: str,
        search_radius: float,
    ):
        """
        选择最佳引脚孔洞对

        综合评分:
          - 占用分数 (两孔之和, 权重 0.5)
          - 距离分 (离估计位置越近越好, 权重 0.25)
          - 电气约束惩罚 (短路/异常跨度, 权重 0.2)
          - 分离约束 (两引脚间距须合理, 权重 0.05)

        v2 改进:
          - 增加最小引脚间距约束 (排除两引脚选到同一区域)
          - 增加引脚间距合理性评分
        """
        from logic.circuit import norm_component_type
        from config import circuit as circuit_cfg

        ntype = norm_component_type(comp_type)
        is_wire = (ntype == "Wire")
        RAIL_ROWS = {"1", "2", "63", "64", "65"}

        # 估计孔洞间距用于分离约束
        if calibrator.row_centers is not None and len(calibrator.row_centers) > 1:
            spacing = float(np.mean(np.diff(calibrator.row_centers)))
        else:
            spacing = 12.0

        best_score = -float('inf')
        best_pair = None

        for (h1, occ1, dist1) in cands1:
            loc1 = calibrator.hole_to_logic(h1[0], h1[1])
            if loc1 is None or loc1[0] == "Groove":
                continue

            for (h2, occ2, dist2) in cands2:
                # 不能是同一个孔洞
                if abs(h1[0] - h2[0]) < 1 and abs(h1[1] - h2[1]) < 1:
                    continue

                loc2 = calibrator.hole_to_logic(h2[0], h2[1])
                if loc2 is None or loc2[0] == "Groove":
                    continue

                # ---- 最小引脚间距 (两引脚不能太近) ----
                pin_sep = np.sqrt((h1[0] - h2[0]) ** 2 + (h1[1] - h2[1]) ** 2)
                if not is_wire and pin_sep < spacing * 0.8:
                    continue  # 两引脚几乎重叠，跳过

                # --- 占用分数 (0-1 归一化) ---
                occ_combined = (occ1 + occ2) / 2.0

                # --- 距离分 (0-1 归一化, 越近越高) ---
                dist_combined = max(0.0, 1.0 - (dist1 + dist2) / (2.0 * search_radius))

                # --- 电气约束惩罚 ---
                penalty = 0.0
                try:
                    r1, r2 = int(loc1[0]), int(loc2[0])
                    c1, c2 = loc1[1], loc2[1]
                    is_rail1 = loc1[0] in RAIL_ROWS
                    is_rail2 = loc2[0] in RAIL_ROWS
                    group1 = 'L' if c1 in 'abcde' else 'R'
                    group2 = 'L' if c2 in 'abcde' else 'R'

                    if not is_wire:
                        # 同行同侧 = 短路
                        if (r1 == r2 and group1 == group2 and
                                not is_rail1 and not is_rail2):
                            penalty += 0.8

                        # 纯同行无跨缝
                        span = abs(r2 - r1)
                        if span == 0 and not is_rail1 and not is_rail2:
                            penalty += 0.5

                        # 异常大跨度
                        elif span > circuit_cfg.pin_large_span_threshold:
                            penalty += 0.3

                        # span=1 对长体元件 (电阻/二极管/电容) 不合理
                        elif span == 1 and ntype in (
                                "Resistor", "Diode",
                                "Electrolytic_Capacitor", "Ceramic_Capacitor"):
                            penalty += 0.2

                        # 合理跨度奖励 (span≥2)
                        elif 2 <= span <= 5:
                            penalty -= 0.1

                except (ValueError, TypeError):
                    pass

                # --- 引脚间距合理性 (鼓励合理分离) ---
                sep_bonus = 0.0
                if not is_wire and pin_sep > spacing * 1.5:
                    sep_bonus = min(0.1, (pin_sep / spacing - 1.5) * 0.02)

                total = (occ_combined * 0.50 +
                         dist_combined * 0.25 -
                         penalty * 0.20 +
                         sep_bonus * 0.05)

                if total > best_score:
                    best_score = total
                    best_pair = (h1, h2)

        return best_pair

    # ================================================================
    # 辅助工具
    # ================================================================

    def _filter_wire_shadow_clusters(
        self,
        occupancy_map: Dict[Tuple[float, float], float],
        calibrator,
    ) -> Dict[Tuple[float, float], float]:
        """
        线状簇 (wire shadow) 滤波

        问题: 导线经过面包板表面时, 会在一条线上遮挡多个连续孔洞,
        导致这些孔洞的视觉特征显著偏离空洞基线, 被误判为 "引脚占用"。

        真实引脚 vs 导线阴影:
          - 引脚: 通常只占 1-2 个孔洞 (引脚直径 ≈ 1 个孔洞)
          - 导线: 经过时产生线状连续高分区 (≥4 个相邻孔洞)

        算法:
          1. 收集所有高分孔洞 (> threshold)
          2. 将它们映射到网格坐标 (row, col)
          3. 检测同一行或同一列中的连续高分孔洞段
          4. 如果连续段长度 ≥ 4, 判定为导线阴影, 降低分数
             (但保留两端点, 因为导线端点可能是真实连接点)

        Returns:
            过滤后的占用分数字典
        """
        if (calibrator.row_centers is None or calibrator.col_centers is None
                or len(occupancy_map) < 10):
            return occupancy_map

        threshold = self.OCCUPANCY_THRESHOLD

        # 1. 收集高分孔洞的网格位置
        high_score_grid: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for (hx, hy), score in occupancy_map.items():
            if score <= threshold:
                continue
            loc = calibrator.hole_to_logic(hx, hy)
            if loc is None or loc[0] == "Groove":
                continue
            try:
                row = int(loc[0])
                col_idx = ord(loc[1]) - ord('a')
                high_score_grid[(row, col_idx)] = (hx, hy)
            except (ValueError, TypeError):
                continue

        if not high_score_grid:
            return occupancy_map

        # 2. 检测同列连续段 (导线通常纵向穿过多行)
        shadow_holes: set = set()
        cols_used = set(c for _, c in high_score_grid.keys())

        for col in cols_used:
            rows_in_col = sorted(r for r, c in high_score_grid if c == col)
            if len(rows_in_col) < 4:
                continue

            # 找连续段
            segments = []
            seg_start = rows_in_col[0]
            prev = rows_in_col[0]
            for r in rows_in_col[1:]:
                if r - prev <= 2:  # 允许间隔 1 行 (孔洞可能被跳过)
                    prev = r
                else:
                    segments.append((seg_start, prev))
                    seg_start = r
                    prev = r
            segments.append((seg_start, prev))

            for seg_s, seg_e in segments:
                seg_rows = [r for r in rows_in_col if seg_s <= r <= seg_e]
                if len(seg_rows) >= 4:
                    # 线状簇: 保留两端, 压制中间
                    for r in seg_rows[1:-1]:
                        wh = high_score_grid.get((r, col))
                        if wh:
                            shadow_holes.add(wh)

        # 3. 检测同行连续段 (导线水平穿过多列)
        rows_used = set(r for r, _ in high_score_grid.keys())

        for row in rows_used:
            cols_in_row = sorted(c for r, c in high_score_grid if r == row)
            if len(cols_in_row) < 4:
                continue

            segments = []
            seg_start = cols_in_row[0]
            prev = cols_in_row[0]
            for c in cols_in_row[1:]:
                if c - prev <= 2:
                    prev = c
                else:
                    segments.append((seg_start, prev))
                    seg_start = c
                    prev = c
            segments.append((seg_start, prev))

            for seg_s, seg_e in segments:
                seg_cols = [c for c in cols_in_row if seg_s <= c <= seg_e]
                if len(seg_cols) >= 4:
                    for c in seg_cols[1:-1]:
                        wh = high_score_grid.get((row, c))
                        if wh:
                            shadow_holes.add(wh)

        # 4. 压制导线阴影孔洞的分数 (降到阈值以下)
        if shadow_holes:
            result = dict(occupancy_map)
            for (hx, hy) in shadow_holes:
                if (hx, hy) in result:
                    result[(hx, hy)] = min(result[(hx, hy)], threshold * 0.5)
            return result

        return occupancy_map

    def _estimate_patch_radius(self, calibrator) -> int:
        """动态估计原始帧空间中的 patch 半径"""
        if calibrator.row_centers is not None and len(calibrator.row_centers) > 1:
            warped_spacing = float(np.mean(np.diff(calibrator.row_centers)))
        else:
            warped_spacing = 12.0

        if calibrator.matrix is not None:
            try:
                inv_m = np.linalg.inv(calibrator.matrix)
                cw = calibrator.width / 2
                ch = calibrator.height / 2
                p1 = np.array([[[cw, ch]]], dtype=np.float32)
                p2 = np.array([[[cw + warped_spacing, ch]]], dtype=np.float32)
                f1 = cv2.perspectiveTransform(p1, inv_m)[0][0]
                f2 = cv2.perspectiveTransform(p2, inv_m)[0][0]
                frame_spacing = float(np.linalg.norm(f2 - f1))
            except Exception:
                frame_spacing = warped_spacing * 3
        else:
            frame_spacing = warped_spacing * 3

        # Patch 覆盖约 0.45 倍孔洞间距 (够覆盖孔洞+少量周围)
        radius = max(5, int(frame_spacing * 0.45))
        radius = min(radius, 50)
        logger.debug(f"[PinHole] patch_radius={radius}px "
                     f"(frame_spacing≈{frame_spacing:.1f}px)")
        return radius

    @staticmethod
    def _batch_transform(
        points: List[Tuple[float, float]],
        matrix: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """批量透视变换"""
        if not points:
            return []
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, matrix)
        return [(float(p[0][0]), float(p[0][1])) for p in transformed]

    @staticmethod
    def _to_warped(
        pixel: Tuple[float, float],
        matrix: np.ndarray,
    ) -> np.ndarray:
        """单点 → 矫正空间"""
        p = np.array([[[pixel[0], pixel[1]]]], dtype=np.float32)
        w = cv2.perspectiveTransform(p, matrix)
        return w[0][0]
