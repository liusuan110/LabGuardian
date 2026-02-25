"""
风险分级引擎 — 将 CircuitValidator.diagnose() 输出映射到风险等级
================================================================

输入: diagnose() 返回的 List[str] 诊断文本
输出: (RiskLevel, list[str]) 风险等级 + 触发原因

不依赖 src_v2, 仅通过关键词匹配工作。
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class RiskLevel(str, Enum):
    """风险等级 — 决定教师端工位卡片颜色"""
    SAFE = "safe"           # 绿色: 无问题
    WARNING = "warning"     # 黄色: 需关注但不紧急
    DANGER = "danger"       # 红色: 立即干预, 存在元器件损坏风险


# ---- 关键词 → 风险等级映射 (按优先级排列) ----

_DANGER_KEYWORDS: List[str] = [
    "短路",
    "烧毁",
    "无限流电阻",
    "同一导通组",
    "可能损坏",
]

_WARNING_KEYWORDS: List[str] = [
    "极性未确定",
    "极性未知",
    "引脚缺失",
    "浮空",
    "孤立",
    "开路",
    "未正确跨行",
    "方向",
]


def classify_risk(diagnostics: List[str]) -> Tuple[RiskLevel, List[str]]:
    """
    将诊断文本列表分类为风险等级。

    Args:
        diagnostics: CircuitValidator.diagnose(analyzer) 的返回值

    Returns:
        (max_risk_level, risk_reasons) 其中 risk_reasons 是触发判定的诊断条目
    """
    if not diagnostics:
        return RiskLevel.SAFE, []

    max_level = RiskLevel.SAFE
    reasons: List[str] = []

    for diag in diagnostics:
        matched_level = _match_single(diag)
        if matched_level is not None:
            reasons.append(diag)
            if _level_priority(matched_level) > _level_priority(max_level):
                max_level = matched_level

    return max_level, reasons


def _match_single(diag: str) -> RiskLevel | None:
    """对单条诊断文本进行关键词匹配"""
    for keyword in _DANGER_KEYWORDS:
        if keyword in diag:
            return RiskLevel.DANGER

    for keyword in _WARNING_KEYWORDS:
        if keyword in diag:
            return RiskLevel.WARNING

    return None


def _level_priority(level: RiskLevel) -> int:
    """风险等级优先级 (数值越大越严重)"""
    return {
        RiskLevel.SAFE: 0,
        RiskLevel.WARNING: 1,
        RiskLevel.DANGER: 2,
    }.get(level, 0)
