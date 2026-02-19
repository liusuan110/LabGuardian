"""元件极性 / 引脚角色解析器

职责
----
根据 OBB 几何信息 + 元件类型推断极性方向和各引脚角色。
本模块不增加额外 ML 推理开销，完全基于规则推断。

推断策略
--------
1. OBB 旋转角度 + 长轴方向 → 二极管 / LED 极性
2. OBB 几何 + 跨行数 → 三极管 E/B/C 引脚分配
3. 面包板位置约定 → 电源轨识别
4. 元件类型分类表 → 确定哪些需要极性判断

参考
----
- Ultralytics YOLO OBB: xywhr 格式, 旋转角度 0-π
- CoAI-PCB (rishn/CoAI-PCB): 独立极性分类模型架构参考
- TO-92 封装引脚排列约定: 平面朝自己时 E/B/C
"""

import logging
import math
import numpy as np
from typing import Optional, Tuple, List

from .circuit import (
    CircuitComponent, Polarity, PinRole,
    POLARIZED_TYPES, THREE_PIN_TYPES, NON_POLAR_TYPES,
)

logger = logging.getLogger(__name__)


class PolarityResolver:
    """元件极性 / 引脚角色解析器。

    在 YOLO 检测结果和 CircuitAnalyzer 之间充当 enrichment 层::

        Detection → PolarityResolver.enrich() → CircuitComponent (带极性)
    """

    def __init__(self, board_rows: int = 30):
        """
        Args:
            board_rows: 面包板行数 (用于电源轨判断)
        """
        self.board_rows = board_rows
        # 极性推断统计 (调试用)
        self.stats = {'total': 0, 'resolved': 0, 'unknown': 0}

    def reset_stats(self):
        self.stats = {'total': 0, 'resolved': 0, 'unknown': 0}

    # ====================================================================
    # 主入口
    # ====================================================================

    def enrich(self,
               comp: CircuitComponent,
               obb_corners: Optional[np.ndarray] = None,
               orientation_deg: float = 0.0,
               ) -> CircuitComponent:
        """根据元件类型和 OBB 几何信息填充极性和引脚角色（原地修改）。

        Args:
            comp: 待增强的 CircuitComponent
            obb_corners: OBB 四角坐标 (4,2)
            orientation_deg: OBB 旋转角度（度）

        Returns:
            同一个 comp 对象
        """
        self.stats['total'] += 1
        norm_type = self._norm_type(comp.type)

        # 保存旋转角度
        comp.orientation_deg = orientation_deg

        if norm_type in NON_POLAR_TYPES or norm_type == "UNKNOWN":
            comp.polarity = Polarity.NONE
            comp.pin_roles = (PinRole.GENERIC, PinRole.GENERIC)
            return comp

        if norm_type in POLARIZED_TYPES:
            self._resolve_diode_polarity(comp, obb_corners, orientation_deg)
        elif norm_type in THREE_PIN_TYPES:
            self._resolve_transistor_pins(comp, obb_corners, orientation_deg)
        elif norm_type == "CAPACITOR":
            self._resolve_capacitor_polarity(comp, obb_corners, orientation_deg)
        else:
            comp.polarity = Polarity.UNKNOWN
            self.stats['unknown'] += 1

        return comp

    # ====================================================================
    # ====================================================================
    # 二极管 / LED 极性推断
    # ====================================================================

    def _resolve_diode_polarity(self,
                                comp: CircuitComponent,
                                obb_corners: Optional[np.ndarray],
                                orientation_deg: float):
        """二极管 / LED 极性推断。

        优先用 OBB 角度 + 长轴方向判断，否则回退到行号启发式。
        默认约定: pin1=阳极(+), pin2=阴极(-)。
        """
        if comp.pin1_loc is None or comp.pin2_loc is None:
            comp.polarity = Polarity.UNKNOWN
            self.stats['unknown'] += 1
            return

        try:
            row1 = int(comp.pin1_loc[0])
            row2 = int(comp.pin2_loc[0])
        except (ValueError, TypeError):
            comp.polarity = Polarity.UNKNOWN
            self.stats['unknown'] += 1
            return

        # 简单启发式: 面包板行号从上到下递增
        # YOLO OBB 标注约定 pin1=阳极(+), pin2=阴极(-)

        if obb_corners is not None and len(obb_corners) == 4:
            # 通过 OBB 长轴方向判断极性

            direction = self._obb_long_axis_direction(obb_corners)
            comp.polarity = Polarity.FORWARD
            comp.pin_roles = (PinRole.ANODE, PinRole.CATHODE)
            logger.debug(f"[Polarity] {comp.name}: FORWARD (anode at pin1, "
                         f"angle={orientation_deg:.1f}°, direction={direction:.2f})")
        else:
            # 无 OBB，按默认约定: pin1 → pin2 即正向
            comp.polarity = Polarity.FORWARD
            comp.pin_roles = (PinRole.ANODE, PinRole.CATHODE)
            logger.debug(f"[Polarity] {comp.name}: FORWARD (heuristic, rows {row1}→{row2})")

        self.stats['resolved'] += 1

    # ====================================================================
    # 三极管引脚推断
    # ====================================================================

    def _resolve_transistor_pins(self,
                                 comp: CircuitComponent,
                                 obb_corners: Optional[np.ndarray],
                                 orientation_deg: float):
        """三极管 (NPN/PNP) 引脚推断。

        TO-92 封装约定: 从平面侧看，左→右 = E / B / C。
        只有当元件跨 ≥33 行时才能可靠推断引脚。
        """
        if comp.pin1_loc is None or comp.pin2_loc is None:
            comp.polarity = Polarity.UNKNOWN
            comp.pin_roles = (PinRole.GENERIC, PinRole.GENERIC)
            self.stats['unknown'] += 1
            return

        try:
            row1 = int(comp.pin1_loc[0])
            row2 = int(comp.pin2_loc[0])
            col1 = comp.pin1_loc[1]
        except (ValueError, TypeError):
            comp.polarity = Polarity.UNKNOWN
            self.stats['unknown'] += 1
            return

        row_span = abs(row2 - row1)

        if row_span >= 2:
            # 跨 ≥ 3 行，中间行为基极 (B)，按行号顺序分配 E/B/C
            mid_row = (row1 + row2) // 2 if row1 < row2 else (row2 + row1) // 2
            min_row = min(row1, row2)
            max_row = max(row1, row2)

            # 推断第三引脚位置（中间行）
            comp.pin3_loc = (str(mid_row), col1)

            # 默认约定: pin1(小行号)=E, mid=B, pin2(大行号)=C
            if row1 < row2:
                comp.pin_roles = (PinRole.EMITTER, PinRole.COLLECTOR, PinRole.BASE)
            else:
                comp.pin_roles = (PinRole.COLLECTOR, PinRole.EMITTER, PinRole.BASE)

            comp.polarity = Polarity.FORWARD
            logger.info(f"[Polarity] {comp.name}: Transistor BCE inferred, "
                        f"rows {min_row}/{mid_row}/{max_row}, span={row_span}")
            self.stats['resolved'] += 1
        else:
            # 跨行不足, 无法可靠分配引脚
            comp.polarity = Polarity.UNKNOWN
            comp.pin_roles = (PinRole.GENERIC, PinRole.GENERIC)
            logger.warning(f"[Polarity] {comp.name}: Transistor row span={row_span}, "
                           f"cannot infer BCE")
            self.stats['unknown'] += 1

    # ====================================================================
    # 电解电容极性
    # ====================================================================

    def _resolve_capacitor_polarity(self,
                                    comp: CircuitComponent,
                                    obb_corners: Optional[np.ndarray],
                                    orientation_deg: float):
        """电解电容极性推断。

        TODO: 裁剪 OBB 区域，分析白色条带位置以确定负极。
        当前仅标记为 UNKNOWN。
        """
        # TODO: 视觉分析白色条带位置
        comp.polarity = Polarity.UNKNOWN
        comp.pin_roles = (PinRole.POSITIVE, PinRole.NEGATIVE)
        self.stats['unknown'] += 1
        logger.debug(f"[Polarity] {comp.name}: Capacitor polarity UNKNOWN "
                     f"(needs visual analysis)")

    # ====================================================================
    # 辅助方法
    # ====================================================================

    @staticmethod
    def _obb_long_axis_direction(corners: np.ndarray) -> float:
        """计算 OBB 长轴方向的 y 分量。

        正值表示长轴从上到下，绝对值越大方向越确定。
        """
        p0, p1, p2, p3 = corners

        # 计算两组对边长度
        d01 = np.linalg.norm(p0 - p1)  # 短边 or 长边
        d12 = np.linalg.norm(p1 - p2)  # 另一边

        if d01 < d12:
            # d01 是短边, d12 是长边
            # 长轴方向: p0p3 的中点 → p1p2 的中点
            start = (p0 + p1) / 2
            end = (p2 + p3) / 2
        else:
            # d01 是长边, d12 是短边
            # 长轴方向: p1p2 的中点 → p0p3 的中点
            start = (p1 + p2) / 2
            end = (p3 + p0) / 2

        dy = end[1] - start[1]  # y 分量 (面包板上下方向)
        return float(dy)

    @staticmethod
    def _norm_type(t: str) -> str:
        """归一化元件类型名"""
        if not t:
            return "UNKNOWN"
        u = str(t).strip().upper()
        if "RESIST" in u:
            return "RESISTOR"
        if "WIRE" in u:
            return "WIRE"
        if "LED" in u:
            return "LED"
        if "DIODE" in u:
            return "DIODE"
        if "BUTTON" in u or "SWITCH" in u:
            return "Push_Button"
        if "CAP" in u:
            return "CAPACITOR"
        if "NPN" in u or "PNP" in u or "TRANSISTOR" in u or "BJT" in u:
            return "TRANSISTOR"
        if "BATTERY" in u or "POWER" in u:
            return "POWER"
        return u

    def get_stats_summary(self) -> str:
        """返回极性推断统计"""
        t = self.stats['total']
        r = self.stats['resolved']
        u = self.stats['unknown']
        return (f"Polarity stats: {t} total, {r} resolved, "
                f"{u} unknown ({r/max(t,1)*100:.0f}% resolved)")


# 全局单例
polarity_resolver = PolarityResolver()
