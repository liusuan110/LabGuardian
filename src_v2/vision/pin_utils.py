"""
引脚定位统一工具模块
====================
集中管理:
  - 元件引脚物理先验 (延伸比例、搜索半径、最小跨度)
  - 面包板电气约束评分 (短路/跨度/同行惩罚)
  - 最佳引脚对选择

消除 PinHoleDetector._select_best_pair() 与
ImageAnalyzer._pick_best_pin_pair() 的代码重复。
"""

from typing import Tuple, List, Optional

# ============================================================
# 元件引脚物理先验表 (统一管理)
# ============================================================
# ext_ratio:    OBB 长轴方向引脚延伸比例 (补偿元件遮挡)
# search_mult:  视觉验证搜索半径 = 孔间距 × 此倍数
# min_span:     两引脚最小合理行跨度 (低于此值惩罚)
# min_ext_px:   最小延伸像素数 (防止 OBB 太小时引脚卡在体内)

COMPONENT_PIN_PROPS = {
    "Resistor":                {"ext_ratio": 0.18, "search_mult": 5.0, "min_span": 2},
    "LED":                     {"ext_ratio": 0.18, "search_mult": 4.0, "min_span": 1},
    "Diode":                   {"ext_ratio": 0.15, "search_mult": 4.0, "min_span": 2},
    "Ceramic_Capacitor":       {"ext_ratio": 0.15, "search_mult": 3.5, "min_span": 1},
    "Electrolytic_Capacitor":  {"ext_ratio": 0.15, "search_mult": 3.5, "min_span": 1},
    "Transistor":              {"ext_ratio": 0.15, "search_mult": 3.5, "min_span": 1},
    "IC":                      {"ext_ratio": 0.04, "search_mult": 3.5, "min_span": 1},
    "Potentiometer":           {"ext_ratio": 0.12, "search_mult": 3.5, "min_span": 1},
    "Wire":                    {"ext_ratio": 0.02, "search_mult": 3.0, "min_span": 0},
}

DEFAULT_PIN_PROPS = {"ext_ratio": 0.12, "search_mult": 3.5, "min_span": 1}
MIN_EXTENSION_PX = 6

# 电源轨行号 (不受同行惩罚)
RAIL_ROWS = {"1", "2", "63", "64", "65"}


def get_pin_props(comp_type: str) -> dict:
    """获取元件引脚属性, 未知类型返回默认值"""
    from logic.circuit import norm_component_type
    ntype = norm_component_type(comp_type)
    return COMPONENT_PIN_PROPS.get(ntype,
           COMPONENT_PIN_PROPS.get(comp_type, DEFAULT_PIN_PROPS))


# ============================================================
# 统一电气约束评分
# ============================================================

def score_electrical_constraints(
    loc1: Tuple[str, str],
    loc2: Tuple[str, str],
    comp_type: str,
) -> float:
    """评估引脚对的面包板电气约束, 返回惩罚分 (越大越差, 0=无惩罚).

    面包板规则:
      - 同行同侧 (a-e 或 f-j) = 导通组短路
      - 非 Wire 同行无跨缝 = 无意义
      - 行跨度过大 = 可能误匹配

    Args:
        loc1, loc2: 逻辑坐标 ("行号", "列字母")
        comp_type: 元件类型名

    Returns:
        惩罚分 (0=完美, >0=有惩罚, 负数=奖励合理配置)
    """
    from logic.circuit import norm_component_type

    ntype = norm_component_type(comp_type)
    is_wire = (ntype == "Wire")
    props = get_pin_props(comp_type)

    penalty = 0.0

    try:
        r1, r2 = int(loc1[0]), int(loc2[0])
        c1, c2 = loc1[1], loc2[1]
    except (ValueError, TypeError):
        return 0.0

    is_rail1 = str(r1) in RAIL_ROWS
    is_rail2 = str(r2) in RAIL_ROWS
    group1 = 'L' if c1 in 'abcde' else 'R'
    group2 = 'L' if c2 in 'abcde' else 'R'

    if is_wire:
        return 0.0  # 导线不受约束

    # 同行同侧 = 短路
    if r1 == r2 and group1 == group2 and not is_rail1 and not is_rail2:
        penalty += 100.0

    # 跨中缝同行: 合理但微罚
    elif r1 == r2 and group1 != group2:
        penalty += 5.0

    span = abs(r2 - r1)

    # 纯同行 (非电源轨)
    if span == 0 and not is_rail1 and not is_rail2:
        penalty += 50.0

    # 跨度过大
    elif span > 10:
        penalty += 20.0 + (span - 10) * 2.0

    # 跨度小于元件最小合理跨度
    elif span < props["min_span"] and not is_rail1 and not is_rail2:
        penalty += 15.0

    # 奖励合理跨度
    if 2 <= span <= 5:
        penalty -= 3.0

    return penalty


# ============================================================
# 最佳引脚对选择 (几何候选)
# ============================================================

def select_best_pin_pair(
    candidates1: List[Tuple[Tuple[str, str], float]],
    candidates2: List[Tuple[Tuple[str, str], float]],
    comp_type: str,
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """从 Top-K 候选中选最佳引脚对.

    评分 = 像素距离之和 + 电气约束惩罚

    Args:
        candidates1: [(逻辑坐标, 距离), ...] — 引脚 1 的候选
        candidates2: [(逻辑坐标, 距离), ...] — 引脚 2 的候选
        comp_type: 元件类型名

    Returns:
        (loc1, loc2) 最佳逻辑坐标对
    """
    best_score = float('inf')
    best_pair = (candidates1[0][0], candidates2[0][0])

    for loc1, dist1 in candidates1:
        for loc2, dist2 in candidates2:
            score = dist1 + dist2
            score += score_electrical_constraints(loc1, loc2, comp_type)

            if score < best_score:
                best_score = score
                best_pair = (loc1, loc2)

    return best_pair
