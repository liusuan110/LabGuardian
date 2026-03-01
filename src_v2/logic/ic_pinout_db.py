"""IC DIP 封装引脚数据库

职责
----
提供内置的 IC 引脚定义, 用于将 YOLO 检测到的 IC 芯片
映射为面包板上的多引脚电路元件。

工作流程:
  1. OCR 识别芯片型号 (e.g. "LM324")
  2. lookup_ic("LM324") → ICPackageInfo (14 引脚定义)
  3. get_ic_pin_locs(info, top_row) → 每个引脚的 (行号, 列字母) 坐标

DIP 封装在面包板上的插入约定:
  - IC 跨越中央沟槽, 左侧引脚插入 e 列, 右侧引脚插入 f 列
  - Pin 1 在左上角 (有缺口/圆点标记)
  - 左侧引脚从上到下: Pin 1, 2, ..., N/2
  - 右侧引脚从下到上: Pin N/2+1, N/2+2, ..., N

支持的 IC:
  - LM324:  四路运放, DIP-14
  - LM358:  双路运放, DIP-8
  - NE5532: 双路低噪运放, DIP-8
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ICPinDef:
    """单个引脚定义"""
    pin_number: int     # 物理引脚号 (1-based)
    name: str           # 引脚名称 (e.g. "1OUT", "VCC", "GND")
    function: str       # 功能分类:
                        #   "output"      — 运放输出
                        #   "input_inv"   — 反相输入 (-)
                        #   "input_non"   — 同相输入 (+)
                        #   "power_vcc"   — 电源正
                        #   "power_gnd"   — 电源地
                        #   "nc"          — 空脚


@dataclass
class ICPackageInfo:
    """IC 封装完整信息"""
    model: str                  # 主型号 (e.g. "LM324")
    package: str                # 封装类型 (e.g. "DIP-14")
    pin_count: int              # 引脚总数
    description: str            # 中文描述
    pins: List[ICPinDef]        # 引脚列表 (按 pin_number 升序)
    aliases: List[str] = field(default_factory=list)  # 型号别名


# ============================================================
# 内置引脚数据库
# ============================================================

def _build_db() -> Dict[str, ICPackageInfo]:
    """构建内置数据库 (模块加载时执行一次)"""
    db: Dict[str, ICPackageInfo] = {}

    # ---- LM324 — 四路运算放大器, DIP-14 ----
    #        ┌──U──┐
    #  1OUT ─┤1  14├─ 4OUT
    #  1IN- ─┤2  13├─ 4IN-
    #  1IN+ ─┤3  12├─ 4IN+
    #  VCC  ─┤4  11├─ GND
    #  2IN+ ─┤5  10├─ 3IN+
    #  2IN- ─┤6   9├─ 3IN-
    #  2OUT ─┤7   8├─ 3OUT
    #        └─────┘
    lm324 = ICPackageInfo(
        model="LM324",
        package="DIP-14",
        pin_count=14,
        description="四路运算放大器",
        aliases=["LM324N", "LM324AN", "LM324D", "LM2902"],
        pins=[
            ICPinDef(1,  "1OUT",  "output"),
            ICPinDef(2,  "1IN-",  "input_inv"),
            ICPinDef(3,  "1IN+",  "input_non"),
            ICPinDef(4,  "VCC",   "power_vcc"),
            ICPinDef(5,  "2IN+",  "input_non"),
            ICPinDef(6,  "2IN-",  "input_inv"),
            ICPinDef(7,  "2OUT",  "output"),
            ICPinDef(8,  "3OUT",  "output"),
            ICPinDef(9,  "3IN-",  "input_inv"),
            ICPinDef(10, "3IN+",  "input_non"),
            ICPinDef(11, "GND",   "power_gnd"),
            ICPinDef(12, "4IN+",  "input_non"),
            ICPinDef(13, "4IN-",  "input_inv"),
            ICPinDef(14, "4OUT",  "output"),
        ],
    )
    db["LM324"] = lm324

    # ---- LM358 — 双路运算放大器, DIP-8 ----
    #        ┌──U──┐
    #  1OUT ─┤1   8├─ VCC
    #  1IN- ─┤2   7├─ 2OUT
    #  1IN+ ─┤3   6├─ 2IN-
    #  GND  ─┤4   5├─ 2IN+
    #        └─────┘
    lm358 = ICPackageInfo(
        model="LM358",
        package="DIP-8",
        pin_count=8,
        description="双路运算放大器",
        aliases=["LM358N", "LM358P", "LM358D", "LM358AN"],
        pins=[
            ICPinDef(1, "1OUT",  "output"),
            ICPinDef(2, "1IN-",  "input_inv"),
            ICPinDef(3, "1IN+",  "input_non"),
            ICPinDef(4, "GND",   "power_gnd"),
            ICPinDef(5, "2IN+",  "input_non"),
            ICPinDef(6, "2IN-",  "input_inv"),
            ICPinDef(7, "2OUT",  "output"),
            ICPinDef(8, "VCC",   "power_vcc"),
        ],
    )
    db["LM358"] = lm358

    # ---- NE5532 — 双路低噪声运算放大器, DIP-8 ----
    #        ┌──U──┐
    #  1OUT ─┤1   8├─ VCC
    #  1IN- ─┤2   7├─ 2OUT
    #  1IN+ ─┤3   6├─ 2IN-
    #  GND  ─┤4   5├─ 2IN+
    #        └─────┘
    ne5532 = ICPackageInfo(
        model="NE5532",
        package="DIP-8",
        pin_count=8,
        description="双路低噪声运算放大器",
        aliases=["NE5532N", "NE5532P", "NE5532D", "SA5532"],
        pins=[
            ICPinDef(1, "1OUT",  "output"),
            ICPinDef(2, "1IN-",  "input_inv"),
            ICPinDef(3, "1IN+",  "input_non"),
            ICPinDef(4, "GND",   "power_gnd"),
            ICPinDef(5, "2IN+",  "input_non"),
            ICPinDef(6, "2IN-",  "input_inv"),
            ICPinDef(7, "2OUT",  "output"),
            ICPinDef(8, "VCC",   "power_vcc"),
        ],
    )
    db["NE5532"] = ne5532

    return db


# 全局数据库实例
IC_PINOUT_DB: Dict[str, ICPackageInfo] = _build_db()


# ============================================================
# 查询接口
# ============================================================

def lookup_ic(model_name: str) -> Optional[ICPackageInfo]:
    """按型号查找 IC 引脚定义 (大小写不敏感, 支持别名).

    Args:
        model_name: OCR 识别的型号, e.g. "LM324", "LM358N"

    Returns:
        ICPackageInfo 或 None (未找到)
    """
    if not model_name:
        return None

    upper = model_name.strip().upper()

    # 1. 精确匹配主型号
    if upper in IC_PINOUT_DB:
        return IC_PINOUT_DB[upper]

    # 2. 遍历别名
    for info in IC_PINOUT_DB.values():
        for alias in info.aliases:
            if upper == alias.upper():
                return info

    # 3. 前缀匹配 (e.g. "LM324A" 匹配 "LM324")
    for key, info in IC_PINOUT_DB.items():
        if upper.startswith(key):
            return info

    # 4. 主型号作为前缀 (e.g. "LM324" 匹配输入 "LM3241")
    for key, info in IC_PINOUT_DB.items():
        if key.startswith(upper) and len(upper) >= 4:
            return info

    logger.debug(f"[IC_DB] 未找到型号: {model_name}")
    return None


def get_ic_pin_locs(
    info: ICPackageInfo,
    top_row: int,
) -> List[Tuple[int, str, ICPinDef]]:
    """计算 IC 每个引脚在面包板上的 (行号, 列字母) 坐标.

    DIP 封装在面包板上的布局:
      - 左侧引脚 (Pin 1 .. Pin N/2): 插入 'e' 列, 行号从 top_row 递增
      - 右侧引脚 (Pin N/2+1 .. Pin N): 插入 'f' 列, 行号从底部递减到 top_row

    Example — LM324 (DIP-14), top_row=10:
      Pin 1  → (10, 'e')    Pin 14 → (10, 'f')
      Pin 2  → (11, 'e')    Pin 13 → (11, 'f')
      Pin 3  → (12, 'e')    Pin 12 → (12, 'f')
      Pin 4  → (13, 'e')    Pin 11 → (13, 'f')
      Pin 5  → (14, 'e')    Pin 10 → (14, 'f')
      Pin 6  → (15, 'e')    Pin 9  → (15, 'f')
      Pin 7  → (16, 'e')    Pin 8  → (16, 'f')

    Args:
        info: IC 封装信息
        top_row: IC Pin1 所在的面包板行号

    Returns:
        按 pin_number 排序的列表: [(行号, 列字母, ICPinDef), ...]
    """
    n = info.pin_count
    half = n // 2
    result: List[Tuple[int, str, ICPinDef]] = []

    # 按 pin_number 排序引脚定义
    sorted_pins = sorted(info.pins, key=lambda p: p.pin_number)

    for pdef in sorted_pins:
        pn = pdef.pin_number
        if pn <= half:
            # 左侧引脚: Pin 1..N/2 → 'e' 列, row = top_row + (pn - 1)
            row = top_row + (pn - 1)
            col = 'e'
        else:
            # 右侧引脚: Pin N/2+1..N → 'f' 列, row = top_row + (N - pn)
            row = top_row + (n - pn)
            col = 'f'
        result.append((row, col, pdef))

    return result


def get_supported_models() -> List[str]:
    """返回所有支持的 IC 型号列表"""
    return sorted(IC_PINOUT_DB.keys())
