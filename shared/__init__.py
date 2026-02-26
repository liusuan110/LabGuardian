"""
LabGuardian Shared — 学生端 / 教师端共用数据模型
================================================

此包不依赖 src_v2 中的任何模块, 可独立使用。
仅依赖 pydantic (数据校验) 和 Python 标准库。
"""

from .models import (
    ComponentInfo,
    StationHeartbeat,
    GuidanceMessage,
    BroadcastMessage,
)
from .risk import RiskLevel, classify_risk

__all__ = [
    "ComponentInfo",
    "StationHeartbeat",
    "GuidanceMessage",
    "BroadcastMessage",
    "RiskLevel",
    "classify_risk",
]
