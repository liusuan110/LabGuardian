"""
课堂状态管理器 — 维护全班工位的实时状态
========================================

职责:
  1. 接收并存储每台工位的心跳数据 (StationHeartbeat)
  2. 维护工位在线/离线判定 (10s 超时)
  3. 提供聚合查询: 排行榜、风险警报、班级统计
  4. 管理 WebSocket 连接 (教师→学生 指导消息推送)

线程安全: 使用 threading.Lock 保护共享状态,
FastAPI 路由在 asyncio 事件循环中调用, 通过 lock 序列化访问。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 避免直接引用 fastapi.WebSocket 类型, 用 Any 代替 (减少 import 耦合)
WebSocketConnection = Any


@dataclass
class StationState:
    """单个工位的完整状态"""

    # ---- 最新心跳 (dict 形式, 来自 StationHeartbeat.model_dump()) ----
    heartbeat: Dict[str, Any] = field(default_factory=dict)

    # ---- 时间线 ----
    first_seen: float = 0.0             # 首次上线时间戳
    last_seen: float = 0.0              # 最近心跳时间戳

    # ---- WebSocket 连接 (用于推送教师指导) ----
    websocket: Optional[WebSocketConnection] = None

    # ---- 累计统计 ----
    risk_event_count: int = 0           # 累计触发的风险事件次数
    peak_progress: float = 0.0          # 历史最高进度

    # ---- 教师指导记录 ----
    guidance_history: List[Dict[str, str]] = field(default_factory=list)


class ClassroomState:
    """
    全班课堂状态 (内存存储, 单次实验会话)

    使用方式:
        state = ClassroomState()
        state.update_station(heartbeat_dict)
        ranking = state.get_ranking()
    """

    ONLINE_TIMEOUT = 10.0  # 超过此秒数未收到心跳则判定离线

    def __init__(self):
        self._lock = threading.Lock()
        self._stations: Dict[str, StationState] = {}
        self._session_start: float = time.time()
        self._reference_circuit: Optional[Dict] = None

    # ================================================================
    # 心跳更新
    # ================================================================

    def update_station(self, heartbeat: Dict[str, Any]) -> List[str]:
        """
        更新工位状态, 返回新产生的警报消息列表。

        Args:
            heartbeat: StationHeartbeat.model_dump() 的结果

        Returns:
            new_alerts: 新产生的高风险警报 (仅在风险等级从非 danger 变为 danger 时触发)
        """
        station_id = heartbeat.get("station_id", "unknown")
        now = time.time()
        new_alerts: List[str] = []

        with self._lock:
            if station_id not in self._stations:
                self._stations[station_id] = StationState(
                    first_seen=now,
                )
                logger.info(f"[Classroom] 新工位上线: {station_id}")

            station = self._stations[station_id]
            old_risk = station.heartbeat.get("risk_level", "safe")

            # 更新心跳
            station.heartbeat = heartbeat
            station.last_seen = now

            # 更新峰值进度
            progress = heartbeat.get("progress", 0.0)
            if progress > station.peak_progress:
                station.peak_progress = progress

            # 风险事件计数
            new_risk = heartbeat.get("risk_level", "safe")
            if new_risk == "danger":
                station.risk_event_count += 1
                if old_risk != "danger":
                    student = heartbeat.get("student_name", station_id)
                    reasons = heartbeat.get("risk_reasons", [])
                    reason_text = reasons[0] if reasons else "检测到危险电路"
                    new_alerts.append(
                        f"{station_id} {student} — {reason_text}"
                    )

        return new_alerts

    # ================================================================
    # 查询接口
    # ================================================================

    def get_all_stations(self) -> Dict[str, Dict[str, Any]]:
        """获取全部工位状态 (用于教师端全班视图)"""
        now = time.time()
        result = {}
        with self._lock:
            for sid, state in self._stations.items():
                result[sid] = {
                    **state.heartbeat,
                    "online": (now - state.last_seen) < self.ONLINE_TIMEOUT,
                    "first_seen": state.first_seen,
                    "elapsed_s": now - state.first_seen,
                    "risk_event_count": state.risk_event_count,
                    "peak_progress": state.peak_progress,
                }
        return result

    def get_ranking(self) -> List[Dict[str, Any]]:
        """进度排行榜: 按 progress 降序, 同进度按用时升序"""
        stations = self.get_all_stations()
        ranking = []
        for sid, data in stations.items():
            ranking.append({
                "station_id": sid,
                "student_name": data.get("student_name", ""),
                "progress": data.get("progress", 0.0),
                "similarity": data.get("similarity", 0.0),
                "elapsed_s": data.get("elapsed_s", 0),
                "risk_event_count": data.get("risk_event_count", 0),
                "component_count": data.get("component_count", 0),
                "risk_level": data.get("risk_level", "safe"),
                "online": data.get("online", False),
            })
        ranking.sort(key=lambda x: (-x["progress"], x["elapsed_s"]))

        for i, entry in enumerate(ranking):
            entry["rank"] = i + 1

        return ranking

    def get_alerts(self) -> List[Dict[str, Any]]:
        """获取所有活跃风险警报 (risk_level != safe 的在线工位)"""
        stations = self.get_all_stations()
        alerts = []
        for sid, data in stations.items():
            if data.get("risk_level", "safe") != "safe" and data.get("online"):
                alerts.append({
                    "station_id": sid,
                    "student_name": data.get("student_name", ""),
                    "risk_level": data.get("risk_level", "warning"),
                    "risk_reasons": data.get("risk_reasons", []),
                    "diagnostics": data.get("diagnostics", []),
                    "progress": data.get("progress", 0.0),
                })
        # danger 排在前面
        alerts.sort(key=lambda x: (0 if x["risk_level"] == "danger" else 1))
        return alerts

    def get_stats(self) -> Dict[str, Any]:
        """班级聚合统计"""
        stations = self.get_all_stations()
        if not stations:
            return {
                "total_stations": 0, "online_count": 0, "completed_count": 0,
                "avg_progress": 0.0, "total_risk_events": 0,
                "danger_count": 0,
                "error_histogram": {},
                "session_duration_s": time.time() - self._session_start,
            }

        online_count = sum(1 for s in stations.values() if s.get("online"))
        completed = sum(1 for s in stations.values() if s.get("progress", 0) >= 1.0)
        progresses = [s.get("progress", 0.0) for s in stations.values()]
        total_risk = sum(s.get("risk_event_count", 0) for s in stations.values())

        # 错误类型直方图
        error_counter: Counter = Counter()
        for s in stations.values():
            for diag in s.get("diagnostics", []):
                # 提取错误类型关键信息 (冒号后的第一句)
                if ":" in diag:
                    error_type = diag.split(":", 1)[1].strip()[:30]
                else:
                    error_type = diag[:30]
                error_counter[error_type] += 1

        danger_count = sum(
            1 for s in stations.values()
            if s.get("risk_level") == "danger" and s.get("online")
        )

        return {
            "total_stations": len(stations),
            "online_count": online_count,
            "completed_count": completed,
            "avg_progress": sum(progresses) / len(progresses) if progresses else 0.0,
            "total_risk_events": total_risk,
            "danger_count": danger_count,
            "error_histogram": dict(error_counter.most_common(10)),
            "session_duration_s": time.time() - self._session_start,
        }

    # ================================================================
    # WebSocket 管理
    # ================================================================

    def register_websocket(self, station_id: str, ws: WebSocketConnection):
        """注册工位的 WebSocket 连接 (用于教师指导推送)"""
        with self._lock:
            if station_id in self._stations:
                self._stations[station_id].websocket = ws
            else:
                self._stations[station_id] = StationState(
                    first_seen=time.time(),
                    websocket=ws,
                )

    def unregister_websocket(self, station_id: str):
        """注销工位的 WebSocket 连接"""
        with self._lock:
            if station_id in self._stations:
                self._stations[station_id].websocket = None

    def get_websocket(self, station_id: str) -> Optional[WebSocketConnection]:
        """获取指定工位的 WebSocket (可能为 None)"""
        with self._lock:
            station = self._stations.get(station_id)
            return station.websocket if station else None

    def get_all_websockets(self) -> List[WebSocketConnection]:
        """获取所有活跃的 WebSocket 连接 (用于广播)"""
        with self._lock:
            return [
                s.websocket for s in self._stations.values()
                if s.websocket is not None
            ]

    def add_guidance_record(self, station_id: str, guidance: Dict[str, str]):
        """记录教师指导历史"""
        with self._lock:
            if station_id in self._stations:
                self._stations[station_id].guidance_history.append(guidance)

    # ================================================================
    # 参考电路
    # ================================================================

    def set_reference(self, reference: Dict):
        """设置本节课的参考电路"""
        with self._lock:
            self._reference_circuit = reference
            logger.info("[Classroom] 参考电路已设置")

    def get_reference(self) -> Optional[Dict]:
        """获取参考电路"""
        with self._lock:
            return self._reference_circuit

    # ================================================================
    # 会话管理
    # ================================================================

    def reset(self):
        """重置会话 (新的一节课)"""
        with self._lock:
            self._stations.clear()
            self._reference_circuit = None
            self._session_start = time.time()
            logger.info("[Classroom] 会话已重置")

    @property
    def station_count(self) -> int:
        with self._lock:
            return len(self._stations)

    @property
    def session_start(self) -> float:
        return self._session_start
