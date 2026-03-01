"""
LabGuardian AppContext -- 服务注册中心
======================================

职责:
  1. 集中管理所有核心服务实例 (Detector / Analyzer / LLM / OCR ...)
  2. 提供线程安全的数据访问 (QReadWriteLock 保护 analyzer)
  3. 解耦 GUI 与业务逻辑 — MainWindow 不再直接构造任何 AI / Vision 模块
  4. 支持无 GUI 模式 (benchmark / 单元测试) 直接构造 AppContext 即可

架构角色:
  MainWindow ──depends──► AppContext ──owns──► Detector
                                              Stabilizer
                                              Analyzer
                                              LLMEngine
                                              OCREngine
                                              PolarityResolver
                                              CircuitValidator
                                              BreadboardCalibrator

线程模型:
  - ImageAnalysisWorker 线程: 写 analyzer (write lock)
  - 主线程 (UI):              读 analyzer (read lock),
                               调用 validate / ask_ai / show_netlist
  - LLMWorker 线程:           读 circuit_description 快照 (无需锁, 使用快照)

使用方式:
    ctx = AppContext()
    ctx.load_all()              # 加载全部模型

    # 线程安全的 analyzer 访问
    with ctx.write_lock():
        ctx.analyzer.reset()
        ctx.analyzer.add_component(comp)

    with ctx.read_lock():
        desc = ctx.analyzer.get_circuit_description()
"""

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from vision.detector import ComponentDetector
from vision.stabilizer import DetectionStabilizer
from vision.calibrator import BreadboardCalibrator, board_calibrator
from logic.circuit import CircuitAnalyzer
from logic.polarity import PolarityResolver
from logic.validator import CircuitValidator
from ai.llm_engine import LLMEngine
from ai.ocr_engine import OCREngine
from config import circuit as circuit_cfg

logger = logging.getLogger(__name__)


class ReadWriteLock:
    """
    读写锁 (纯 Python, 不依赖 Qt)

    - 多个读者可并发持有读锁
    - 写者独占, 与所有读/写互斥
    - 写者优先 (避免写饥饿)

    用于保护 CircuitAnalyzer:
      VideoWorker 线程需要每帧 write (reset + rebuild)
      主线程 (AI 问答 / 验证 / 网表导出) 需要 read
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._read_ready = threading.Condition(self._lock)
        self._readers = 0
        self._writers = 0
        self._write_waiters = 0

    def acquire_read(self):
        with self._lock:
            while self._writers > 0 or self._write_waiters > 0:
                self._read_ready.wait()
            self._readers += 1

    def release_read(self):
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        with self._lock:
            self._write_waiters += 1
            while self._readers > 0 or self._writers > 0:
                self._read_ready.wait()
            self._write_waiters -= 1
            self._writers += 1

    def release_write(self):
        with self._lock:
            self._writers -= 1
            self._read_ready.notify_all()


class AppContext:
    """
    LabGuardian 应用上下文 (服务注册中心)

    所有核心模块在此创建并统一管理生命周期。
    通过 read_lock / write_lock 保护跨线程共享状态。
    """

    def __init__(self):
        # ---- 感知层 ----
        self.detector = ComponentDetector()
        self.stabilizer = DetectionStabilizer(window_size=5, min_hits=3)
        self.calibrator: BreadboardCalibrator = board_calibrator  # 复用全局单例

        # ---- 推理层 ----
        self.analyzer = CircuitAnalyzer(
            rail_track_rows=circuit_cfg.rail_track_rows,
        )
        self.polarity = PolarityResolver()
        self.validator = CircuitValidator()

        # ---- 认知层 ----
        self.llm = LLMEngine()
        self.ocr = OCREngine()

        # ---- 线程安全: 读写锁 (保护 analyzer + stabilizer) ----
        self._rw_lock = ReadWriteLock()

        # ---- OCR 缓存 (线程安全接口) ----
        self._ocr_lock = threading.Lock()
        self._ocr_cache: Dict[str, str] = {}   # cache_key -> chip_model

        # ---- 幽灵线数据 (主线程写, 分析线程读) ----
        self._ghost_lock = threading.Lock()
        self._ar_missing_links: list = []

        # ---- RAG 已查询型号缓存 ----
        self._rag_queried_models: Set[str] = set()

        # ---- 最新电路描述快照 (分析线程写, AI 线程读) ----
        self._desc_lock = threading.Lock()
        self._circuit_description_snapshot: str = ""

        # ---- ROI 裁剪缓存 (校准后设置) ----
        self._roi_rect: Optional[Tuple[int, int, int, int]] = None  # (x1,y1,x2,y2)

    # ================================================================
    # 读写锁上下文管理器
    # ================================================================

    @contextmanager
    def read_lock(self):
        """获取读锁 (多读者并发)"""
        self._rw_lock.acquire_read()
        try:
            yield
        finally:
            self._rw_lock.release_read()

    @contextmanager
    def write_lock(self):
        """获取写锁 (独占)"""
        self._rw_lock.acquire_write()
        try:
            yield
        finally:
            self._rw_lock.release_write()

    # ================================================================
    # 电路描述快照 (无锁 AI 问答)
    # ================================================================

    def update_circuit_snapshot(self):
        """在写锁内调用, 将当前 analyzer 状态快照为文本描述。
        AI 问答时直接读快照, 不需要再获取锁。
        同时执行独立诊断, 将问题附加到快照中。
        """
        desc = self.analyzer.get_circuit_description()

        # 独立诊断 (不依赖参考电路)
        diag_issues = CircuitValidator.diagnose(self.analyzer)
        if diag_issues:
            desc += "\n自动诊断:\n"
            for issue in diag_issues:
                desc += f"  - {issue}\n"

        with self._desc_lock:
            self._circuit_description_snapshot = desc

    def get_circuit_snapshot(self) -> str:
        """获取最新的电路描述快照 (线程安全, 无阻塞)"""
        with self._desc_lock:
            return self._circuit_description_snapshot

    # ================================================================
    # OCR 缓存 (线程安全)
    # ================================================================

    def ocr_cache_get(self, key: str) -> Optional[str]:
        with self._ocr_lock:
            return self._ocr_cache.get(key)

    def ocr_cache_set(self, key: str, value: str):
        with self._ocr_lock:
            self._ocr_cache[key] = value

    def ocr_cache_has(self, key: str) -> bool:
        with self._ocr_lock:
            return key in self._ocr_cache

    def ocr_cache_clear(self):
        with self._ocr_lock:
            self._ocr_cache.clear()

    def get_ocr_cache_copy(self) -> Dict[str, str]:
        """返回 OCR 缓存的副本 (用于绘制标签)"""
        with self._ocr_lock:
            return dict(self._ocr_cache)

    # ================================================================
    # 幽灵线数据 (线程安全)
    # ================================================================

    def set_missing_links(self, links: list):
        with self._ghost_lock:
            self._ar_missing_links = list(links)

    def get_missing_links(self) -> list:
        with self._ghost_lock:
            return list(self._ar_missing_links)

    # ================================================================
    # 模型加载
    # ================================================================

    def load_all(self) -> dict:
        """加载全部模型, 返回状态摘要"""
        status = {}

        # YOLO
        if self.detector.load():
            status['detector'] = 'OK'
        else:
            status['detector'] = 'FAIL'

        # LLM + RAG
        llm_status = self.llm.load()
        status['llm'] = llm_status

        # OCR
        if self.ocr.initialize():
            status['ocr'] = self.ocr.backend_name
        else:
            status['ocr'] = 'NONE'

        return status

    # ================================================================
    # 电源轨标注 (学生交互, 线程安全)
    # ================================================================

    def set_rail_assignment(self, track_id: str, label: str):
        """学生标注某条轨道的用途 (线程安全)"""
        with self.write_lock():
            self.analyzer.set_rail_assignment(track_id, label)

    def clear_rail_assignments(self):
        """清除所有轨道标注 (线程安全)"""
        with self.write_lock():
            self.analyzer.clear_rail_assignments()

    def get_rail_track_ids(self) -> list:
        """返回所有轨道标识 (无需锁)"""
        return list(self.analyzer._rail_track_rows.keys())

    def get_rail_assignments(self) -> dict:
        """返回当前轨道标注的副本 (线程安全)"""
        with self.read_lock():
            return dict(self.analyzer.rail_assignments)

    def get_unassigned_active_rails(self) -> list:
        """返回有连接但未标注的轨道 (线程安全)"""
        with self.read_lock():
            return self.analyzer.get_unassigned_active_rails()

    # ================================================================
    # 清理
    # ================================================================

    def reset_analysis(self):
        """重置 analyzer / stabilizer / OCR 缓存 (需从主线程调用)"""
        with self.write_lock():
            self.analyzer.reset()
            self.stabilizer.clear()
        self.ocr_cache_clear()
        self.set_missing_links([])
        self._rag_queried_models.clear()
        self._circuit_description_snapshot = ""
