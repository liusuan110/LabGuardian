"""
导线骨架分析模块
================
职责:
  - 对 YOLO 检测到的 Wire bbox 执行颜色分割 + 骨架化
  - 提取骨架端点作为精确的导线连接点 (比 OBB 短边中点准确)
  - 分类导线颜色 (红/黑/蓝/黄/绿/白/橙)

算法流程:
  1. 裁剪 Wire bbox 区域 (含 padding)
  2. HSV 颜色分割: 排除面包板白色背景, 提取前景 (导线) 像素
  3. 形态学清理 + 保留最大连通域
  4. Zhang-Suen 骨架化 (优先 ximgproc, 回退到形态学迭代)
  5. 端点检测: 8-邻域中邻居数 == 1 的骨架像素
  6. 选取欧氏距离最远的两个端点作为导线连接点
  7. 映射回原帧像素坐标

依赖: 仅 OpenCV + NumPy, 不需要额外训练模型
"""

import logging
import cv2
import numpy as np
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class WireAnalyzer:
    """导线骨架端点提取 + 颜色分类

    用于替代 OBB/HBB 短边中点估计, 提升导线端点定位精度。
    仅对 Wire 类检测结果生效, 分析失败时回退到原始 OBB/HBB 引脚。
    """

    # ---- HSV 颜色范围 (H: 0-180, S: 0-255, V: 0-255) ----
    _COLOR_RANGES = {
        'red':    [((0, 80, 50), (10, 255, 255)),
                   ((170, 80, 50), (180, 255, 255))],
        'blue':   [((95, 70, 50), (130, 255, 255))],
        'green':  [((35, 70, 50), (85, 255, 255))],
        'yellow': [((18, 80, 50), (38, 255, 255))],
        'orange': [((10, 80, 50), (18, 255, 255))],
        'black':  [((0, 0, 0), (180, 100, 60))],
        'white':  [((0, 0, 200), (180, 40, 255))],
    }
    _COLOR_PRIORITY = ['red', 'blue', 'green', 'yellow', 'orange', 'black', 'white']

    # ---- 可调参数 ----
    BBOX_PADDING = 5           # bbox 外扩像素 (防止端点刚好在边缘)
    MIN_WIRE_AREA = 50         # 最小导线面积 (低于此则分析失败)
    MIN_SKELETON_PX = 8        # 最小骨架像素数
    SAT_THRESHOLD = 50         # 饱和度 > 此值 → 彩色前景
    VAL_LOW_THRESHOLD = 60     # 亮度 < 此值 → 黑色前景
    BG_SAT_MAX = 45            # 背景最大饱和度 (面包板白/米色)
    BG_VAL_MIN = 140           # 背景最小亮度
    COLOR_MATCH_MIN_RATIO = 0.15  # 颜色匹配最低占比

    # ================================================================
    # 主入口
    # ================================================================

    def analyze_wire(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
    ) -> Tuple[Optional[Tuple[Tuple[float, float], Tuple[float, float]]], str]:
        """一次性提取导线端点 + 颜色 (共享分割结果, 避免重复计算)。

        Args:
            frame: 原始帧 (BGR)
            bbox: (x1, y1, x2, y2) 原帧坐标

        Returns:
            (endpoints, color)
            - endpoints: ((x1,y1), (x2,y2)) 原帧像素坐标, 或 None (分析失败)
            - color: 'red'/'black'/'blue'/'green'/'yellow'/'orange'/'white'/'unknown'
        """
        crop, offset = self._safe_crop(frame, bbox)
        if crop is None:
            return None, 'unknown'

        ox, oy = offset

        # ---- 颜色分割 ----
        wire_mask = self._segment_wire(crop)
        if wire_mask is None or cv2.countNonZero(wire_mask) < self.MIN_WIRE_AREA:
            return None, 'unknown'

        wire_mask = self._keep_largest_component(wire_mask)
        if cv2.countNonZero(wire_mask) < self.MIN_WIRE_AREA:
            return None, 'unknown'

        # ---- 颜色分类 (在骨架化之前, 使用完整掩码) ----
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        color = self._classify_from_mask(hsv, wire_mask)

        # ---- 骨架化 + 端点检测 ----
        skeleton = self._skeletonize(wire_mask)
        if cv2.countNonZero(skeleton) < self.MIN_SKELETON_PX:
            return None, color

        endpoints = self._find_skeleton_endpoints(skeleton)
        if len(endpoints) < 2:
            pair = self._find_farthest_skeleton_pair(skeleton)
            if pair is None:
                return None, color
            ep1, ep2 = pair
        else:
            ep1, ep2 = self._pick_farthest_pair(endpoints)

        # 映射回原帧坐标 (row,col → x,y)
        p1 = (float(ep1[1] + ox), float(ep1[0] + oy))
        p2 = (float(ep2[1] + ox), float(ep2[0] + oy))

        return (p1, p2), color

    # ================================================================
    # 裁剪
    # ================================================================

    def _safe_crop(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """安全裁剪 bbox 区域 (含 padding, 边界约束)。

        Returns:
            (crop, (offset_x, offset_y)) 或 (None, (0, 0))
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = self.BBOX_PADDING

        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)

        if cx2 <= cx1 or cy2 <= cy1:
            return None, (0, 0)

        return frame[cy1:cy2, cx1:cx2], (cx1, cy1)

    # ================================================================
    # 颜色分割
    # ================================================================

    def _segment_wire(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """HSV 分割: 从 bbox 裁剪区域中提取导线前景掩码。

        策略:
          - 面包板背景 = 低饱和度 + 高亮度 (白色/米色表面)
          - 彩色导线 = 高饱和度
          - 黑色导线 = 低亮度
          - 导线掩码 = 非背景 ∩ (彩色 ∪ 黑色)
          - 形态学闭合 (连接断裂) + 开运算 (去噪)
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)

        bg = (s < self.BG_SAT_MAX) & (v > self.BG_VAL_MIN)
        colored = s > self.SAT_THRESHOLD
        dark = v < self.VAL_LOW_THRESHOLD

        mask = (~bg) & (colored | dark)
        mask = mask.astype(np.uint8) * 255

        # 形态学清理
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        return mask

    # ================================================================
    # 连通域
    # ================================================================

    @staticmethod
    def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
        """保留面积最大的连通域, 去除孔洞阴影等小噪点。"""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return mask

        # label 0 是背景, 从 label 1 开始找最大面积
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.zeros_like(mask)
        out[labels == largest] = 255
        return out

    # ================================================================
    # 骨架化
    # ================================================================

    @staticmethod
    def _skeletonize(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
        """骨架化: 将导线掩码细化为单像素宽的骨架线。

        优先使用 cv2.ximgproc.thinning (Zhang-Suen, 更精确),
        不可用时回退到形态学迭代腐蚀法。
        """
        # 优先: opencv-contrib 的 Zhang-Suen thinning
        try:
            return cv2.ximgproc.thinning(
                mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except (AttributeError, cv2.error):
            pass

        # 回退: 形态学骨架提取
        skel = np.zeros_like(mask)
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        tmp = mask.copy()
        for _ in range(max_iter):
            eroded = cv2.erode(tmp, elem)
            opened = cv2.dilate(eroded, elem)
            diff = cv2.subtract(tmp, opened)
            skel = cv2.bitwise_or(skel, diff)
            tmp = eroded
            if cv2.countNonZero(tmp) == 0:
                break
        return skel

    # ================================================================
    # 端点检测
    # ================================================================

    @staticmethod
    def _find_skeleton_endpoints(skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """检测骨架端点: 8-邻域中恰好有 1 个邻居的骨架像素。

        Returns:
            [(row, col), ...] 端点坐标列表
        """
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)

        binary = (skeleton > 0).astype(np.uint8)
        neighbors = cv2.filter2D(binary, cv2.CV_32S, kernel)

        mask = (binary > 0) & (neighbors == 1)
        return list(zip(*np.where(mask)))  # (row, col) tuples

    @staticmethod
    def _find_farthest_skeleton_pair(
        skeleton: np.ndarray,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """回退方案: 端点检测不到 >=2 个点时, 取骨架上最远的两个像素。

        使用凸包将 O(n^2) 搜索缩小到 O(h^2), h 为凸包顶点数。
        """
        pts = np.argwhere(skeleton > 0)  # (row, col)
        if len(pts) < 2:
            return None

        if len(pts) > 30:
            xy = pts[:, ::-1].astype(np.float32)
            hull = cv2.convexHull(xy).squeeze()
            if hull.ndim == 1:
                hull = hull.reshape(1, -1)
            cands = [(int(p[1]), int(p[0])) for p in hull]
        else:
            cands = [tuple(p) for p in pts]

        best_d, best = 0, None
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                d = (cands[i][0] - cands[j][0]) ** 2 + \
                    (cands[i][1] - cands[j][1]) ** 2
                if d > best_d:
                    best_d = d
                    best = (cands[i], cands[j])
        return best

    @staticmethod
    def _pick_farthest_pair(
        endpoints: List[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """从多个端点中选出欧氏距离最远的一对 (即导线两端连接点)。"""
        if len(endpoints) == 2:
            return (endpoints[0], endpoints[1])

        best_d, best = 0, (endpoints[0], endpoints[1])
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                d = (endpoints[i][0] - endpoints[j][0]) ** 2 + \
                    (endpoints[i][1] - endpoints[j][1]) ** 2
                if d > best_d:
                    best_d = d
                    best = (endpoints[i], endpoints[j])
        return best

    # ================================================================
    # 颜色分类
    # ================================================================

    def _classify_from_mask(
        self, hsv: np.ndarray, wire_mask: np.ndarray,
    ) -> str:
        """基于导线掩码内的 HSV 像素分布, 投票分类导线颜色。

        对每种预定义颜色范围, 计算匹配像素占比, 取最高者。
        """
        wire_px_count = cv2.countNonZero(wire_mask)
        if wire_px_count < 10:
            return 'unknown'

        best_color, best_ratio = 'unknown', 0.0

        for name in self._COLOR_PRIORITY:
            count = 0
            for lo, hi in self._COLOR_RANGES[name]:
                lo_arr = np.array(lo, dtype=np.uint8)
                hi_arr = np.array(hi, dtype=np.uint8)
                in_range = cv2.inRange(hsv, lo_arr, hi_arr)
                count += cv2.countNonZero(in_range & wire_mask)

            ratio = count / wire_px_count
            if ratio > best_ratio and ratio > self.COLOR_MATCH_MIN_RATIO:
                best_ratio = ratio
                best_color = name

        return best_color
