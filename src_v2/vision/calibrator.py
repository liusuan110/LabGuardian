"""
面包板校准模块
职责：透视变换、孔洞检测、像素坐标→逻辑孔位映射
从原 calibration.py 迁移，消除硬编码参数

增强版孔洞检测管线 (v2):
  参考技术方案:
  1. OpenCV findCirclesGrid — 官方对称圆网格检测 API
     (opencv.org/4.x calib3d, CALIB_CB_SYMMETRIC_GRID + CLUSTERING)
  2. Multi-param Blob Detector Ensemble — 多组 SimpleBlobDetector 参数扫描取并集
     (Camera Calibration Best Practice, OpenCV Tutorials)
  3. HoughCircles — Hough 圆变换作为补漏检测器
     (Duda & Hart, 1972, "Use of the Hough Transformation...")
  4. Contour Geometric Analysis — findContours + 面积/圆形度/凸度过滤
  5. RANSAC-style Grid Fitting — 对候选点拟合规则网格模型, 剔除离群点并补全缺失
     (Fischler & Bolles, 1981, "RANSAC"; 类似 findCirclesGrid 内部策略)

  检测管线流程:
    预处理 (CLAHE + 多阈值)
      → Level 1: findCirclesGrid (快速, 适合标准面包板)
      → Level 2: Multi-param Blob Ensemble (鲁棒, 覆盖面广)
      → Level 3: HoughCircles 补漏 (捡漏)
      → Level 4: Contour Fallback (最后兜底)
      → 合并去重 (NMS by distance)
      → Grid RANSAC 拟合 (剔离群 + 补缺)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import calibration as calib_cfg

logger = logging.getLogger(__name__)


class BreadboardCalibrator:
    """面包板校准器：透视变换 + 孔洞网格检测 + 坐标映射"""

    def __init__(self):
        self.rows = calib_cfg.rows
        self.cols_per_side = calib_cfg.cols_per_side
        self.output_size = calib_cfg.output_size  # (width, height)
        self.matrix: Optional[np.ndarray] = None
        self.width = 0
        self.height = 0

        # 原始帧中面包板的 4 个角点 (TL, TR, BR, BL), 用于 ROI 计算
        self._src_corners: Optional[np.ndarray] = None

        # 孔洞网格
        self.hole_centers: List[Tuple[float, float]] = []
        self.row_centers: Optional[np.ndarray] = None
        self.col_centers: Optional[np.ndarray] = None

    def reset(self):
        """重置校准状态"""
        self.matrix = None
        self.width = 0
        self.height = 0
        self._src_corners = None
        self.reset_holes()

    def reset_holes(self):
        """重置孔洞网格"""
        self.hole_centers = []
        self.row_centers = None
        self.col_centers = None

    @property
    def is_calibrated(self) -> bool:
        return self.matrix is not None

    # ========================================================
    # 透视变换
    # ========================================================

    def calibrate(self, src_points: np.ndarray) -> Optional[np.ndarray]:
        """
        根据 4 个角点计算透视变换矩阵

        Args:
            src_points: 形状 (4,2) 的点阵, 顺序: 左上→右上→右下→左下

        Returns:
            变换矩阵, 或 None (失败)
        """
        if len(src_points) != 4:
            return None

        # 保存原始角点 (用于 ROI 计算)
        self._src_corners = np.float32(src_points).copy()

        w, h = self.output_size
        dst_points = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype="float32")

        self.matrix = cv2.getPerspectiveTransform(
            np.float32(src_points), dst_points
        )
        self.width = w
        self.height = h
        return self.matrix

    def warp(self, image: np.ndarray) -> np.ndarray:
        """应用透视变换，返回矫正后的俯视图"""
        if self.matrix is None:
            return image
        return cv2.warpPerspective(image, self.matrix, (self.width, self.height))

    # ========================================================
    # 孔洞检测（增强版多级管线）
    # ========================================================

    def detect_holes(self, warped_bgr: np.ndarray) -> int:
        """
        在矫正后俯视图上检测面包板孔洞中心 (增强版多级管线)

        检测流程:
          1. 预处理 → CLAHE 光照均衡 + 多种二值化
          2. Level 1: cv2.findCirclesGrid (最优先, 专为规则圆阵设计)
          3. Level 2: 多参数 SimpleBlobDetector Ensemble
          4. Level 3: HoughCircles 补漏
          5. Level 4: 轮廓几何分析兜底
          6. 合并去重 (NMS by distance)
          7. Grid RANSAC 拟合 (剔离群 + 补缺)

        Args:
            warped_bgr: 矫正后的 BGR 图像

        Returns:
            检测到的孔洞数量
        """
        self.reset_holes()
        if warped_bgr is None or self.matrix is None:
            return 0

        h, w = warped_bgr.shape[:2]

        # ---- 预处理 ----
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        gray_clahe = self._preprocess_clahe(gray)
        binary_maps = self._multi_threshold(gray_clahe)

        # 边界裁剪 ROI (过滤边缘噪声)
        x_min, x_max = int(0.06 * w), int(0.94 * w)
        y_min, y_max = int(0.06 * h), int(0.94 * h)

        all_candidates: List[Tuple[float, float]] = []

        # ---- Level 1: findCirclesGrid ----
        grid_pts = self._detect_circles_grid(gray_clahe)
        if grid_pts is not None and len(grid_pts) > 20:
            logger.info(f"[Hole] Level 1 findCirclesGrid: {len(grid_pts)} pts")
            all_candidates.extend(grid_pts)

        # ---- Level 2: Multi-param Blob Ensemble ----
        for bmap in binary_maps:
            for min_a, max_a, min_c in calib_cfg.blob_param_sweep:
                blob_pts = self._detect_blobs(bmap, min_a, max_a, min_c)
                all_candidates.extend(blob_pts)
        logger.info(f"[Hole] Level 2 Blob Ensemble: {len(all_candidates)} total candidates")

        # ---- Level 3: HoughCircles 补漏 ----
        hough_pts = self._detect_hough_circles(gray_clahe)
        all_candidates.extend(hough_pts)
        logger.info(f"[Hole] Level 3 HoughCircles: +{len(hough_pts)} pts")

        # ---- Level 4: 轮廓几何分析 ----
        for bmap in binary_maps:
            contour_pts = self._detect_contours(bmap)
            all_candidates.extend(contour_pts)
        logger.info(f"[Hole] Level 4 Contours: total candidates now {len(all_candidates)}")

        # ---- ROI 过滤 ----
        candidates = [
            (x, y) for x, y in all_candidates
            if x_min <= x <= x_max and y_min <= y <= y_max
        ]
        logger.info(f"[Hole] After ROI filter: {len(candidates)}")

        # ---- 合并去重 (NMS by distance) ----
        candidates = self._nms_by_distance(candidates, calib_cfg.dedup_radius)
        logger.info(f"[Hole] After NMS dedup: {len(candidates)}")

        self.hole_centers = candidates

        if len(candidates) < calib_cfg.min_hole_count:
            logger.warning(f"[Hole] Only {len(candidates)} holes (< {calib_cfg.min_hole_count}), "
                           f"fallback to linear mapping")
            return len(candidates)

        # ---- Grid RANSAC 拟合 (row/col 中心 + 离群剔除) ----
        ys = np.array([c[1] for c in candidates], dtype=np.float32)
        xs = np.array([c[0] for c in candidates], dtype=np.float32)

        self.row_centers = self._robust_bin_centers(ys, self.rows)
        self.col_centers = self._robust_bin_centers(xs, 10)

        # 网格拟合后二次过滤 + 补全
        if self.row_centers is not None and self.col_centers is not None:
            refined = self._grid_filter_and_interpolate(candidates)
            if len(refined) >= len(candidates) * 0.8:
                self.hole_centers = refined
                logger.info(f"[Hole] Grid refined: {len(refined)} holes (from {len(candidates)})")

        return len(self.hole_centers)

    # ========================================================
    # 预处理方法
    # ========================================================

    @staticmethod
    def _preprocess_clahe(gray: np.ndarray) -> np.ndarray:
        """CLAHE 自适应光照均衡, 参考 OpenCV Camera Calibration 最佳实践"""
        clahe = cv2.createCLAHE(
            clipLimit=calib_cfg.clahe_clip_limit,
            tileGridSize=calib_cfg.clahe_grid_size
        )
        equalized = clahe.apply(gray)
        return cv2.GaussianBlur(equalized, (5, 5), 0)

    @staticmethod
    def _multi_threshold(gray: np.ndarray) -> List[np.ndarray]:
        """
        多阈值二值化策略: 生成多张二值图
        使用不同的 adaptive block size + C 组合, 提升对光照变化的鲁棒性
        """
        results = []
        for block_size in calib_cfg.adaptive_block_sizes:
            for c_val in calib_cfg.adaptive_c_values:
                thr = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    block_size, c_val
                )
                thr = cv2.morphologyEx(
                    thr, cv2.MORPH_OPEN,
                    np.ones((3, 3), np.uint8), iterations=1
                )
                results.append(thr)
        return results

    # ========================================================
    # Level 1: OpenCV findCirclesGrid
    # ========================================================

    @staticmethod
    def _detect_circles_grid(gray: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """
        使用 OpenCV findCirclesGrid 尝试检测对称圆网格
        此 API 专为相机标定板上的规则圆阵设计, 面包板孔洞结构类似
        参考: OpenCV 4.x calib3d module, CALIB_CB_SYMMETRIC_GRID

        尝试多种网格尺寸 (面包板可能部分可见)
        """
        # 准备 Blob 检测器作为 findCirclesGrid 的底层特征检测器
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 0  # 暗孔洞
        params.filterByArea = True
        params.minArea = 8
        params.maxArea = 600
        params.filterByCircularity = True
        params.minCircularity = 0.15
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)

        # 尝试不同的网格尺寸 (从大到小)
        grid_sizes = [
            (10, 5), (8, 5), (6, 5), (10, 4), (8, 4),
            (15, 5), (12, 5), (6, 4), (5, 4)
        ]

        inverted = cv2.bitwise_not(gray)

        for cols, rows_per_half in grid_sizes:
            for img_input in [inverted, gray]:
                found, centers = cv2.findCirclesGrid(
                    img_input,
                    (cols, rows_per_half),
                    flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
                    blobDetector=detector,
                )
                if found:
                    pts = [(float(c[0][0]), float(c[0][1])) for c in centers]
                    return pts

        return None

    # ========================================================
    # Level 2: 多参数 Blob 检测 Ensemble
    # ========================================================

    @staticmethod
    def _detect_blobs(binary: np.ndarray, min_area: int, max_area: int,
                      min_circularity: float) -> List[Tuple[float, float]]:
        """
        使用指定参数的 SimpleBlobDetector 检测
        多组参数扫描取并集可提升覆盖率

        参考: OpenCV SimpleBlobDetector Best Practices
        """
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        params.filterByCircularity = True
        params.minCircularity = min_circularity
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary)
        return [(float(kp.pt[0]), float(kp.pt[1])) for kp in keypoints]

    # ========================================================
    # Level 3: Hough 圆变换
    # ========================================================

    @staticmethod
    def _detect_hough_circles(gray: np.ndarray) -> List[Tuple[float, float]]:
        """
        HoughCircles 检测
        参考: Duda & Hart (1972) "Use of the Hough Transformation to Detect
        Lines and Curves in Pictures"

        适合检测 Blob 遗漏的边缘清晰但面积异常的孔洞
        """
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=calib_cfg.hough_dp,
            minDist=calib_cfg.hough_min_dist,
            param1=calib_cfg.hough_param1,
            param2=calib_cfg.hough_param2,
            minRadius=calib_cfg.hough_min_radius,
            maxRadius=calib_cfg.hough_max_radius,
        )
        if circles is None:
            return []
        return [(float(c[0]), float(c[1])) for c in circles[0]]

    # ========================================================
    # Level 4: 轮廓几何分析
    # ========================================================

    @staticmethod
    def _detect_contours(binary: np.ndarray) -> List[Tuple[float, float]]:
        """
        基于 findContours 的孔洞检测 — 最后一道防线
        用面积、圆形度(4π·Area/Perimeter²)和凸度过滤

        比 SimpleBlobDetector 更低层, 可以捕获不规则形状的孔洞
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < calib_cfg.contour_min_area or area > calib_cfg.contour_max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-5:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < calib_cfg.contour_min_circularity:
                continue

            M = cv2.moments(cnt)
            if M["m00"] < 1e-5:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            results.append((float(cx), float(cy)))

        return results

    # ========================================================
    # 后处理: 去重 + 网格拟合
    # ========================================================

    @staticmethod
    def _nms_by_distance(points: List[Tuple[float, float]],
                         min_dist: float) -> List[Tuple[float, float]]:
        """
        基于距离的非极大值抑制 (NMS)
        对所有候选点, 距离 < min_dist 的合并为质心
        时间复杂度 O(N²) 但 N 通常 < 2000, 可接受

        类似于 DBSCAN 聚类但更适合本场景
        """
        if not points:
            return []

        pts = np.array(points, dtype=np.float32)
        used = np.zeros(len(pts), dtype=bool)
        result = []

        for i in range(len(pts)):
            if used[i]:
                continue
            # 找到所有距离 < min_dist 的邻居
            dists = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
            neighbors = np.where((dists < min_dist) & (~used))[0]
            # 取质心
            cluster = pts[neighbors]
            cx = float(np.mean(cluster[:, 0]))
            cy = float(np.mean(cluster[:, 1]))
            result.append((cx, cy))
            used[neighbors] = True

        return result

    def _grid_filter_and_interpolate(
        self, candidates: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        基于已估计的 row_centers/col_centers 做网格拟合:
          1. 对每个候选点分配到最近的 (row, col) 格位
          2. 如果偏差 > tolerance, 标记为离群点并剔除
          3. 对空缺的格位, 用双线性插值补全

        类似 RANSAC 思路: 用多数inlier拟合模型, 剔除outlier
        参考: Fischler & Bolles (1981) "Random Sample Consensus"
        """
        if self.row_centers is None or self.col_centers is None:
            return candidates

        row_c = self.row_centers
        col_c = self.col_centers
        n_rows = len(row_c)
        n_cols = len(col_c)

        # 估计平均间距 (用于 tolerance 计算)
        avg_row_gap = float(np.mean(np.diff(row_c))) if n_rows > 1 else 20.0
        avg_col_gap = float(np.mean(np.diff(col_c))) if n_cols > 1 else 20.0
        tol = calib_cfg.grid_fit_tolerance

        # 为每个格位分配最佳候选点  grid[r][c] = (x, y) or None
        grid = [[None for _ in range(n_cols)] for _ in range(n_rows)]
        grid_dist = [[float('inf') for _ in range(n_cols)] for _ in range(n_rows)]

        inliers = []
        for (x, y) in candidates:
            # 找最近的行/列
            row_idx = int(np.argmin(np.abs(row_c - y)))
            col_idx = int(np.argmin(np.abs(col_c - x)))

            row_dev = abs(float(row_c[row_idx]) - y) / max(avg_row_gap, 1)
            col_dev = abs(float(col_c[col_idx]) - x) / max(avg_col_gap, 1)

            if row_dev > tol or col_dev > tol:
                continue  # 离群点

            dist = row_dev + col_dev
            if dist < grid_dist[row_idx][col_idx]:
                grid[row_idx][col_idx] = (x, y)
                grid_dist[row_idx][col_idx] = dist

        # 收集 inliers
        for r in range(n_rows):
            for c in range(n_cols):
                if grid[r][c] is not None:
                    inliers.append(grid[r][c])

        # 双线性插值补全空缺格位
        filled_count = 0
        for r in range(n_rows):
            for c in range(n_cols):
                if grid[r][c] is not None:
                    continue

                # 尝试用相邻格位插值
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols and grid[nr][nc] is not None:
                        neighbors.append((dr, dc, grid[nr][nc]))

                if len(neighbors) >= 2:
                    # 加权插值: 用相邻点和网格间距推断
                    est_x = float(col_c[c])
                    est_y = float(row_c[r])
                    grid[r][c] = (est_x, est_y)
                    inliers.append((est_x, est_y))
                    filled_count += 1

        if filled_count > 0:
            logger.info(f"[Hole] Grid interpolated {filled_count} missing positions")

        return inliers

    # ========================================================
    # 坐标映射
    # ========================================================

    def pixel_to_logic(self, x_pixel: float, y_pixel: float):
        """
        将矫正后图像上的像素坐标映射到面包板逻辑坐标 (Row, Col)
        面包板规则: 竖向导通(同一列), 横向不导通
        """
        if self.matrix is None:
            return None

        strip_width = self.width / self.rows
        logic_row_idx = int(x_pixel / strip_width) + 1

        groove_height = self.height * 0.1
        hole_pitch_y = (self.height - groove_height) / 10

        if y_pixel < (self.height / 2 - groove_height / 2):
            col_idx = int(y_pixel / hole_pitch_y)
            col_name = chr(ord('a') + min(col_idx, 4))
        elif y_pixel > (self.height / 2 + groove_height / 2):
            effective_y = y_pixel - (self.height / 2 + groove_height / 2)
            col_idx = int(effective_y / hole_pitch_y)
            col_name = chr(ord('f') + min(col_idx, 4))
        else:
            return "Groove", "0"

        logic_row_idx = max(1, min(self.rows, logic_row_idx))
        return f"{logic_row_idx}", col_name

    def warped_to_logic(self, wx: float, wy: float):
        """优先使用孔洞网格映射，回退到线性映射"""
        hole = self.nearest_hole(wx, wy)
        if hole is not None:
            loc = self.hole_to_logic(hole[0], hole[1])
            if loc is not None:
                return loc
        return self.pixel_to_logic(wx, wy)

    def frame_pixel_to_logic(self, px: float, py: float):
        """
        从原始帧像素坐标映射到逻辑坐标
        (先透视变换到矫正坐标，再映射到逻辑坐标)
        """
        if self.matrix is None:
            return None
        src_point = np.array([[[px, py]]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, self.matrix)
        wx, wy = dst_point[0][0]
        return self.warped_to_logic(wx, wy)

    def logic_to_warped(self, row_str: str, col_char: str) -> Optional[Tuple[float, float]]:
        """将逻辑坐标映射回矫正后图像的像素坐标"""
        if self.row_centers is None or self.col_centers is None:
            return None
        try:
            row = int(row_str)
        except (ValueError, TypeError):
            return None
        row = max(1, min(self.rows, row))
        wy = float(self.row_centers[row - 1])

        if col_char <= 'e':
            col_idx = ord(col_char) - ord('a')
        else:
            col_idx = 5 + (ord(col_char) - ord('f'))
        col_idx = max(0, min(9, col_idx))
        wx = float(self.col_centers[col_idx])
        return (wx, wy)

    def logic_to_frame_pixel(self, row_str: str, col_char: str) -> Optional[Tuple[int, int]]:
        """将逻辑坐标映射回原始帧的像素坐标 (逆透视变换)"""
        warped_pt = self.logic_to_warped(row_str, col_char)
        if warped_pt is None or self.matrix is None:
            return None

        inv_matrix = np.linalg.inv(self.matrix)
        vec = np.array([[[warped_pt[0], warped_pt[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(vec, inv_matrix)
        return (int(dst[0][0][0]), int(dst[0][0][1]))

    # ========================================================
    # 孔洞辅助方法
    # ========================================================

    def nearest_hole(self, wx: float, wy: float) -> Optional[Tuple[float, float]]:
        """查找最近的孔洞中心 (矫正坐标系)"""
        if not self.hole_centers:
            return None
        pts = np.array(self.hole_centers, dtype=np.float32)
        dx = pts[:, 0] - float(wx)
        dy = pts[:, 1] - float(wy)
        idx = int(np.argmin(dx * dx + dy * dy))
        return (float(pts[idx, 0]), float(pts[idx, 1]))

    def top_k_holes(self, wx: float, wy: float, k: int = 3
                    ) -> List[Tuple[Tuple[float, float], float]]:
        """返回距离最近的 K 个孔洞及其距离 (矫正坐标系)。

        当元件遮挡引脚时, 估计坐标可能偏离真实孔洞。
        返回多个候选, 供下游约束排序选择最优。

        Args:
            wx, wy: 矫正坐标系中的查询点
            k: 返回候选数量

        Returns:
            [((hx, hy), distance), ...] 按距离升序排列
        """
        if not self.hole_centers:
            return []

        pts = np.array(self.hole_centers, dtype=np.float32)
        dx = pts[:, 0] - float(wx)
        dy = pts[:, 1] - float(wy)
        dists = np.sqrt(dx * dx + dy * dy)

        k = min(k, len(pts))
        indices = np.argpartition(dists, k)[:k]
        indices = indices[np.argsort(dists[indices])]

        return [
            ((float(pts[idx, 0]), float(pts[idx, 1])), float(dists[idx]))
            for idx in indices
        ]

    def frame_pixel_to_logic_candidates(
        self, px: float, py: float, k: int = 3
    ) -> List[Tuple[Tuple[str, str], float]]:
        """从原始帧像素坐标映射到 K 个候选逻辑坐标。

        先透视变换 → 再查找 Top-K 最近孔洞 → 各自映射为逻辑坐标。
        候选中去除重复的逻辑坐标 (不同孔洞可能映射到同一格位)。

        Args:
            px, py: 原始帧中的像素坐标
            k: 候选数量

        Returns:
            [((row_str, col_char), distance), ...] 按距离升序, 去重
        """
        if self.matrix is None:
            return []

        src_point = np.array([[[px, py]]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, self.matrix)
        wx, wy = dst_point[0][0]

        candidates = self.top_k_holes(wx, wy, k=k)
        results = []
        seen = set()
        for (hx, hy), dist in candidates:
            loc = self.hole_to_logic(hx, hy)
            if loc is not None and loc[0] != "Groove":
                key = (loc[0], loc[1])
                if key not in seen:
                    seen.add(key)
                    results.append((loc, dist))

        # 无孔洞候选时回退到线性映射
        if not results:
            loc = self.pixel_to_logic(wx, wy)
            if loc is not None:
                results.append((loc, 0.0))

        return results

    def hole_to_logic(self, hx: float, hy: float):
        """将孔洞中心 (矫正坐标) 映射到逻辑坐标"""
        if self.row_centers is None or self.col_centers is None:
            return None
        row_idx = int(np.argmin(np.abs(self.row_centers - float(hy)))) + 1
        col_idx = int(np.argmin(np.abs(self.col_centers - float(hx))))
        if col_idx <= 4:
            col_char = chr(ord('a') + col_idx)
        else:
            col_char = chr(ord('f') + (col_idx - 5))
        row_idx = max(1, min(self.rows, row_idx))
        return (f"{row_idx}", col_char)

    @staticmethod
    def _robust_bin_centers(values: np.ndarray, bins: int) -> Optional[np.ndarray]:
        """将排序后的值按分位数分箱，取每箱中位数作为中心"""
        values = np.sort(values)
        n = len(values)
        if n < bins:
            return None
        bin_size = max(1, n // bins)
        centers = []
        for i in range(bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < bins - 1 else n
            centers.append(float(np.median(values[start:end])))
        return np.array(centers, dtype=np.float32)

    # ========================================================
    # 自动面包板检测 + ROI
    # ========================================================

    def auto_detect_board(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        自动检测面包板矩形区域, 返回 4 个角点 (TL, TR, BR, BL)

        算法: Canny 边缘 → 膨胀连接 → 轮廓检测 → 最大四边形筛选

        Args:
            frame: BGR 图像

        Returns:
            (4,2) np.ndarray 角点坐标, 或 None (未检测到)
        """
        h, w = frame.shape[:2]
        min_area = h * w * 0.05  # 面包板至少占画面 5%

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # 多组 Canny 阈值尝试 (适应不同光照)
        for low, high in [(30, 100), (50, 150), (80, 200)]:
            edges = cv2.Canny(blurred, low, high)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for cnt in contours[:10]:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                if len(approx) == 4:
                    corners = approx.reshape(4, 2).astype(np.float32)
                    ordered = self._order_corners(corners)
                    logger.info(f"[AutoDetect] 检测到面包板: area={area:.0f}, "
                                f"corners={ordered.tolist()}")
                    return ordered

        logger.warning("[AutoDetect] 未能自动检测到面包板矩形")
        return None

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """
        将 4 个角点排序为 TL, TR, BR, BL 顺序

        方法: 按 x+y 之和区分 TL(最小) 和 BR(最大),
              按 x-y 之差区分 TR(最大) 和 BL(最小)
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)       # x + y
        d = np.diff(pts, axis=1).flatten()  # x - y

        rect[0] = pts[np.argmin(s)]   # TL: x+y 最小
        rect[2] = pts[np.argmax(s)]   # BR: x+y 最大
        rect[1] = pts[np.argmax(d)]   # TR: x-y 最大
        rect[3] = pts[np.argmin(d)]   # BL: x-y 最小
        return rect

    def get_roi_rect(self, frame_shape: tuple, padding: int = 30
                     ) -> Optional[Tuple[int, int, int, int]]:
        """
        从校准角点计算带 padding 的轴对齐 ROI 矩形

        Args:
            frame_shape: (height, width, ...) 原始帧 shape
            padding: 外扩像素 (防止边缘元件被截断)

        Returns:
            (x1, y1, x2, y2) 或 None
        """
        if self._src_corners is None:
            return None

        xs = self._src_corners[:, 0]
        ys = self._src_corners[:, 1]
        x1 = max(0, int(np.min(xs)) - padding)
        y1 = max(0, int(np.min(ys)) - padding)
        x2 = min(frame_shape[1], int(np.max(xs)) + padding)
        y2 = min(frame_shape[0], int(np.max(ys)) + padding)
        return (x1, y1, x2, y2)

    def auto_calibrate(self, frame: np.ndarray) -> bool:
        """
        一步到位: 自动检测面包板 → 透视校准 → 孔洞检测

        Args:
            frame: BGR 图像

        Returns:
            True = 成功, False = 未检测到面包板
        """
        corners = self.auto_detect_board(frame)
        if corners is None:
            return False
        self.calibrate(corners)
        warped = self.warp(frame)
        self.detect_holes(warped)
        logger.info(f"[AutoCalibrate] 完成: {len(self.hole_centers)} 孔洞")
        return True


# 全局单例
board_calibrator = BreadboardCalibrator()
