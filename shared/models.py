"""
共享数据模型 — 学生端心跳上报 / 教师端指导消息
==============================================

所有网络传输的数据结构在此定义, 保证两端序列化/反序列化一致。
零依赖于 src_v2 (vision / logic / ai / gui_qt)。
"""

from __future__ import annotations

import time
from typing import List, Optional
from pydantic import BaseModel, Field


class ComponentInfo(BaseModel):
    """单个元器件信息 (简化版, 用于网络传输)"""
    name: str = ""                      # "R1", "LED1", "Q1"
    type: str = ""                      # "RESISTOR", "LED", "TRANSISTOR"
    polarity: str = "none"              # "forward" / "reverse" / "none" / "unknown"
    pin1: List[int | str] = Field(default_factory=list)   # [row, side] e.g. [5, "L"]
    pin2: List[int | str] = Field(default_factory=list)
    pin3: List[int | str] = Field(default_factory=list)   # 三极管第三引脚
    confidence: float = 0.0


class StationHeartbeat(BaseModel):
    """
    学生工位心跳包 — 每 2 秒由学生端 POST 到教师服务器

    包含: 元器件检测结果 + 电路验证进度 + 诊断问题 + 风险等级 + 系统状态
    """
    # ---- 工位身份 ----
    station_id: str                                 # "A03"
    student_name: str = ""                          # 可选

    # ---- 时间戳 ----
    timestamp: float = Field(default_factory=time.time)

    # ---- 元器件检测 ----
    component_count: int = 0
    net_count: int = 0
    components: List[ComponentInfo] = Field(default_factory=list)

    # ---- 电路验证 (来自 CircuitValidator) ----
    progress: float = 0.0                           # 0.0 ~ 1.0  子图匹配进度
    similarity: float = 0.0                         # GED 相似度
    match_level: str = ""                           # "L0" ~ "L3"
    missing_components: List[str] = Field(default_factory=list)

    # ---- 诊断 (来自 CircuitValidator.diagnose) ----
    diagnostics: List[str] = Field(default_factory=list)

    # ---- 风险分级 (来自 shared.risk) ----
    risk_level: str = "safe"                        # "safe" / "warning" / "danger"
    risk_reasons: List[str] = Field(default_factory=list)

    # ---- 电路快照 ----
    circuit_snapshot: str = ""                      # 截断的电路描述 (≤500 字符)

    # ---- 系统状态 ----
    fps: float = 0.0
    detector_ok: bool = True
    llm_backend: str = ""
    ocr_backend: str = ""

    # ---- 缩略图 (base64 JPEG, ~5KB) ----
    thumbnail_b64: str = ""


class GuidanceMessage(BaseModel):
    """教师 → 单个学生 的指导消息"""
    station_id: str
    type: str = "hint"                              # "hint" / "warning" / "approve"
    message: str
    sender: str = "Teacher"
    timestamp: float = Field(default_factory=time.time)


class BroadcastMessage(BaseModel):
    """教师 → 全班广播消息"""
    type: str = "broadcast"
    message: str
    sender: str = "Teacher"
    timestamp: float = Field(default_factory=time.time)
