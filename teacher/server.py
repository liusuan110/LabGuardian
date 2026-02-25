"""
LabGuardian Teacher Server — FastAPI 后端
=========================================

职责:
  1. 接收学生工位心跳 (POST /api/heartbeat)
  2. 提供教师端查询 API (课堂总览/排行/警报/统计)
  3. WebSocket 双向通信 (教师指导 → 学生)
  4. 代理 MJPEG 视频流 (学生缩略图 / 未来扩展)
  5. 静态文件服务 (Vue SPA dist/)

启动方式:
  - 寄生模式: 由 src_v2/launcher.py --classroom 在 daemon 线程中启动
  - 独立模式: python teacher/run_teacher.py

依赖: fastapi, uvicorn, pydantic (不依赖 src_v2)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---- 路径设置 (确保 shared/ 可导入) ----
import sys
_TEACHER_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _TEACHER_DIR.parent
if str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

from shared.models import StationHeartbeat, GuidanceMessage, BroadcastMessage
from teacher.classroom import ClassroomState

logger = logging.getLogger(__name__)

# ================================================================
# FastAPI 应用
# ================================================================

app = FastAPI(
    title="LabGuardian Classroom",
    description="教师端课堂监控 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局课堂状态 (单例)
_classroom = ClassroomState()

# 最新帧缓存 (station_id → base64 JPEG)
_frame_lock = threading.Lock()
_frame_cache: Dict[str, str] = {}


def get_classroom() -> ClassroomState:
    """获取课堂状态单例"""
    return _classroom


# ================================================================
# 学生端上报
# ================================================================

@app.post("/api/heartbeat")
async def receive_heartbeat(heartbeat: StationHeartbeat):
    """
    接收学生工位心跳

    学生端每 2 秒 POST 一次, 包含元器件检测、电路验证、风险等级等。
    """
    data = heartbeat.model_dump()
    new_alerts = _classroom.update_station(data)

    # 缓存缩略图
    thumb = data.get("thumbnail_b64", "")
    if thumb:
        with _frame_lock:
            _frame_cache[heartbeat.station_id] = thumb

    return {"status": "ok", "new_alerts": len(new_alerts)}


# ================================================================
# 教师端查询 API
# ================================================================

@app.get("/api/classroom")
async def get_all_stations():
    """获取全班工位状态"""
    return _classroom.get_all_stations()


@app.get("/api/classroom/ranking")
async def get_ranking():
    """进度排行榜"""
    return _classroom.get_ranking()


@app.get("/api/classroom/alerts")
async def get_alerts():
    """活跃风险警报"""
    return _classroom.get_alerts()


@app.get("/api/classroom/stats")
async def get_stats():
    """班级聚合统计"""
    return _classroom.get_stats()


@app.get("/api/station/{station_id}")
async def get_station(station_id: str):
    """获取单个工位详情"""
    stations = _classroom.get_all_stations()
    if station_id in stations:
        return stations[station_id]
    return JSONResponse(status_code=404, content={"error": "station not found"})


@app.get("/api/station/{station_id}/thumbnail")
async def get_thumbnail(station_id: str):
    """获取工位最新缩略图 (base64)"""
    with _frame_lock:
        thumb = _frame_cache.get(station_id, "")
    if thumb:
        return {"thumbnail_b64": thumb}
    return JSONResponse(status_code=404, content={"error": "no thumbnail"})


# ================================================================
# 教师指导推送
# ================================================================

@app.post("/api/station/{station_id}/guidance")
async def send_guidance(station_id: str, msg: GuidanceMessage):
    """
    教师 → 单个学生 发送指导消息

    如果学生 WebSocket 已连接, 实时推送; 否则等学生下次轮询时获取。
    """
    guidance_data = msg.model_dump()
    _classroom.add_guidance_record(station_id, guidance_data)

    # 尝试 WebSocket 实时推送
    ws = _classroom.get_websocket(station_id)
    if ws:
        try:
            await ws.send_json(guidance_data)
            return {"status": "delivered"}
        except Exception:
            return {"status": "queued", "reason": "ws_send_failed"}

    return {"status": "queued", "reason": "ws_not_connected"}


@app.post("/api/classroom/broadcast")
async def broadcast(msg: BroadcastMessage):
    """教师 → 全班广播"""
    data = msg.model_dump()
    websockets = _classroom.get_all_websockets()
    sent = 0
    for ws in websockets:
        try:
            await ws.send_json(data)
            sent += 1
        except Exception:
            pass
    return {"status": "ok", "sent": sent, "total": len(websockets)}


# ================================================================
# 参考电路
# ================================================================

@app.post("/api/classroom/reference")
async def set_reference(body: dict):
    """设置本节课的参考电路 (JSON 格式)"""
    _classroom.set_reference(body)
    return {"status": "ok"}


@app.get("/api/classroom/reference")
async def get_reference():
    """获取参考电路"""
    ref = _classroom.get_reference()
    if ref:
        return ref
    return JSONResponse(status_code=404, content={"error": "no reference set"})


# ================================================================
# 会话管理
# ================================================================

@app.post("/api/classroom/reset")
async def reset_session():
    """重置课堂会话"""
    _classroom.reset()
    return {"status": "ok"}


# ================================================================
# WebSocket 端点 (学生连接, 接收教师指导)
# ================================================================

@app.websocket("/ws/station/{station_id}")
async def ws_station(websocket: WebSocket, station_id: str):
    """
    学生端 WebSocket 连接

    连接后保持, 教师发送指导时通过此通道实时推送到学生端。
    学生端也可以通过此通道上报额外信息。
    """
    await websocket.accept()
    _classroom.register_websocket(station_id, websocket)
    logger.info(f"[WS] 工位 {station_id} WebSocket 已连接")

    try:
        while True:
            # 保持连接, 等待学生端消息 (心跳或其他)
            data = await websocket.receive_text()
            # 学生端可以通过 WS 发送额外信息 (目前仅 keep-alive)
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        logger.info(f"[WS] 工位 {station_id} WebSocket 已断开")
    finally:
        _classroom.unregister_websocket(station_id)


# ================================================================
# Vue SPA 静态文件服务
# ================================================================

_FRONTEND_DIST = _TEACHER_DIR / "frontend" / "dist"

if _FRONTEND_DIST.exists():
    # 生产模式: 服务打包好的 Vue SPA
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="spa")
else:
    # 开发模式: 提示使用 npm run dev
    @app.get("/", response_class=HTMLResponse)
    async def dev_index():
        return """<!DOCTYPE html>
        <html><head><meta charset="utf-8"><title>LabGuardian Classroom</title>
        <style>body{background:#1b1e23;color:#e0e0e0;font-family:system-ui;
        display:flex;justify-content:center;align-items:center;height:100vh;margin:0}
        .box{text-align:center;max-width:500px}
        h1{color:#6c63ff}code{background:#272b33;padding:4px 8px;border-radius:4px}
        a{color:#6c63ff}</style></head><body>
        <div class="box">
        <h1>LabGuardian Classroom</h1>
        <p>前端未构建。请先构建 Vue SPA:</p>
        <p><code>cd teacher/frontend && npm install && npm run build</code></p>
        <p>或使用开发模式:</p>
        <p><code>cd teacher/frontend && npm run dev</code></p>
        <hr style="border-color:#333">
        <p>API 已就绪: <a href="/docs">/docs</a> (Swagger UI)</p>
        </div></body></html>"""


# ================================================================
# 启动函数 (供 launcher.py 调用)
# ================================================================

def start_teacher_server(host: str = "0.0.0.0", port: int = 8080):
    """
    在后台 daemon 线程中启动教师端服务器。

    Args:
        host: 监听地址 (0.0.0.0 = 局域网可访问)
        port: 监听端口

    Returns:
        thread: 服务器线程
    """
    import uvicorn

    thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={
            "host": host,
            "port": port,
            "log_level": "warning",
        },
        daemon=True,
        name="TeacherServer",
    )
    thread.start()
    logger.info(f"[Teacher] 教师端服务器启动: http://{host}:{port}")
    return thread
