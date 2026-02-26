"""
LabGuardian 性能基准测试工具
=====================================

用法:
    cd src_v2
    python -m tools.benchmark                    # 运行全部基准测试
    python -m tools.benchmark --test yolo         # 仅测试 YOLO 推理
    python -m tools.benchmark --test pipeline     # 仅测试端到端管线
    python -m tools.benchmark --test llm          # 仅测试 LLM
    python -m tools.benchmark --test ocr          # 仅测试 OCR
    python -m tools.benchmark --frames 200        # 指定测试帧数
    python -m tools.benchmark --export report.json  # 导出 JSON 报告

竞赛评审 KPI 指标:
    - YOLO 推理延迟 (目标 <50ms/帧 on iGPU)
    - 端到端管线延迟: 帧输入 -> 标注输出 (目标 <100ms)
    - LLM 首 token 延迟 TTFT (目标 <3s on NPU)
    - OCR 识别延迟 (目标 <200ms/芯片)
    - 电路分析延迟 (目标 <10ms/帧)
    - 内存占用
"""

import sys
import time
import json
import argparse
import logging
import statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

# 确保 src_v2 在路径中
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class LatencyStats:
    """延迟统计"""
    name: str
    unit: str = "ms"
    samples: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else 0

    @property
    def p95(self) -> float:
        if not self.samples:
            return 0
        s = sorted(self.samples)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def p99(self) -> float:
        if not self.samples:
            return 0
        s = sorted(self.samples)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]

    @property
    def min_val(self) -> float:
        return min(self.samples) if self.samples else 0

    @property
    def max_val(self) -> float:
        return max(self.samples) if self.samples else 0

    @property
    def std(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "unit": self.unit,
            "count": self.count,
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
            "min": round(self.min_val, 2),
            "max": round(self.max_val, 2),
            "std": round(self.std, 2),
        }

    def summary(self) -> str:
        return (
            f"  {self.name}: "
            f"mean={self.mean:.1f}{self.unit}  "
            f"median={self.median:.1f}{self.unit}  "
            f"p95={self.p95:.1f}{self.unit}  "
            f"p99={self.p99:.1f}{self.unit}  "
            f"min={self.min_val:.1f}  max={self.max_val:.1f}  "
            f"n={self.count}"
        )


def _get_memory_mb() -> float:
    """获取当前进程内存占用 (MB)"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _generate_test_frame(width: int = 960, height: int = 960) -> np.ndarray:
    """生成模拟的测试帧 (带随机噪声)"""
    frame = np.random.randint(40, 200, (height, width, 3), dtype=np.uint8)
    return frame


# ============================================================
# YOLO 推理基准
# ============================================================

def benchmark_yolo(n_frames: int = 100) -> dict:
    """
    测量 YOLO 模型推理延迟

    返回:
        dict 包含延迟统计和 FPS
    """
    print("\n" + "=" * 60)
    print("YOLO Inference Benchmark")
    print("=" * 60)

    from vision.detector import ComponentDetector

    detector = ComponentDetector()
    if not detector.load():
        print("  [SKIP] YOLO model not loaded")
        return {"status": "skip", "reason": "model not loaded"}

    model_path = str(detector._model_path)
    print(f"  Model: {model_path}")
    print(f"  Frames: {n_frames}")

    # 预热 (前 5 帧不计入统计)
    warmup = 5
    frame = _generate_test_frame()
    for _ in range(warmup):
        detector.detect(frame)

    stats = LatencyStats("YOLO Inference", "ms")
    mem_before = _get_memory_mb()

    for i in range(n_frames):
        frame = _generate_test_frame()
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        t1 = time.perf_counter()
        stats.samples.append((t1 - t0) * 1000)

    mem_after = _get_memory_mb()
    fps = 1000.0 / stats.mean if stats.mean > 0 else 0

    print(stats.summary())
    print(f"  Throughput: {fps:.1f} FPS")
    print(f"  Memory: {mem_after:.0f} MB (delta: +{mem_after - mem_before:.0f} MB)")

    return {
        "status": "ok",
        "latency": stats.to_dict(),
        "fps": round(fps, 1),
        "memory_mb": round(mem_after, 1),
        "model": model_path,
    }


# ============================================================
# 端到端管线基准
# ============================================================

def benchmark_pipeline(n_frames: int = 100) -> dict:
    """
    测量完整帧处理管线延迟:
      detection -> stabilization -> annotation

    不含校准和电路分析 (需要真实面包板图像)
    """
    print("\n" + "=" * 60)
    print("End-to-End Pipeline Benchmark")
    print("=" * 60)

    from vision.detector import ComponentDetector
    from vision.stabilizer import DetectionStabilizer

    detector = ComponentDetector()
    if not detector.load():
        print("  [SKIP] YOLO model not loaded")
        return {"status": "skip"}

    stabilizer = DetectionStabilizer(window_size=5, min_hits=3)

    # 预热
    frame = _generate_test_frame()
    for _ in range(5):
        dets = detector.detect(frame)
        stabilizer.update(dets)
        detector.annotate_frame(frame, dets)

    stats_total = LatencyStats("Pipeline Total", "ms")
    stats_detect = LatencyStats("  Detection", "ms")
    stats_stable = LatencyStats("  Stabilization", "ms")
    stats_annot = LatencyStats("  Annotation", "ms")

    print(f"  Frames: {n_frames}")

    for i in range(n_frames):
        frame = _generate_test_frame()

        t_start = time.perf_counter()

        # Detection
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        t1 = time.perf_counter()
        stats_detect.samples.append((t1 - t0) * 1000)

        # Stabilization
        t0 = time.perf_counter()
        stable = stabilizer.update(dets)
        t1 = time.perf_counter()
        stats_stable.samples.append((t1 - t0) * 1000)

        # Annotation
        t0 = time.perf_counter()
        annotated = detector.annotate_frame(frame, stable)
        t1 = time.perf_counter()
        stats_annot.samples.append((t1 - t0) * 1000)

        t_end = time.perf_counter()
        stats_total.samples.append((t_end - t_start) * 1000)

    fps = 1000.0 / stats_total.mean if stats_total.mean > 0 else 0

    for s in [stats_total, stats_detect, stats_stable, stats_annot]:
        print(s.summary())
    print(f"  Throughput: {fps:.1f} FPS")

    return {
        "status": "ok",
        "total": stats_total.to_dict(),
        "detection": stats_detect.to_dict(),
        "stabilization": stats_stable.to_dict(),
        "annotation": stats_annot.to_dict(),
        "fps": round(fps, 1),
    }


# ============================================================
# LLM 基准测试
# ============================================================

def benchmark_llm() -> dict:
    """
    测量 LLM 推理性能:
      - TTFT (Time to First Token)
      - 总生成延迟
      - 生成 token/s
    """
    print("\n" + "=" * 60)
    print("LLM Inference Benchmark")
    print("=" * 60)

    from ai.llm_engine import LLMEngine

    llm = LLMEngine()
    status = llm.load()
    print(f"  Backend: {llm.backend_name}")
    print(f"  Load status: {status}")

    if not llm._active:
        print("  [SKIP] No LLM backend available")
        return {"status": "skip", "reason": "no backend"}

    test_questions = [
        "What is a resistor?",
        "Explain how a capacitor stores energy.",
        "What does the NE555 timer IC do?",
    ]

    stats_ttft = LatencyStats("TTFT", "ms")
    stats_total = LatencyStats("Total Generation", "ms")
    stats_tokens = LatencyStats("Output Length", "chars")

    mem_before = _get_memory_mb()

    for q in test_questions:
        print(f"  Q: {q[:50]}...")

        t0 = time.perf_counter()
        answer = llm.ask(q, circuit_context="")
        t1 = time.perf_counter()

        total_ms = (t1 - t0) * 1000
        stats_total.samples.append(total_ms)
        stats_tokens.samples.append(len(answer))

        # TTFT 估算: 如果是流式生成可以精确测量, 非流式用总时间近似
        stats_ttft.samples.append(total_ms)

        print(f"    A: {answer[:80]}...")
        print(f"    Time: {total_ms:.0f}ms, Length: {len(answer)} chars")

    mem_after = _get_memory_mb()

    for s in [stats_ttft, stats_total, stats_tokens]:
        print(s.summary())
    print(f"  Memory: {mem_after:.0f} MB")

    return {
        "status": "ok",
        "backend": llm.backend_name,
        "ttft": stats_ttft.to_dict(),
        "total_generation": stats_total.to_dict(),
        "output_length": stats_tokens.to_dict(),
        "memory_mb": round(mem_after, 1),
    }


# ============================================================
# OCR 基准测试
# ============================================================

def benchmark_ocr() -> dict:
    """
    测量 OCR 推理性能
    """
    print("\n" + "=" * 60)
    print("OCR Inference Benchmark")
    print("=" * 60)

    from ai.ocr_engine import OCREngine, preprocess_chip_crop

    ocr = OCREngine()
    if not ocr.initialize():
        print("  [SKIP] No OCR backend available")
        return {"status": "skip", "reason": "no backend"}

    print(f"  Backend: {ocr.backend_name}")

    # 生成模拟芯片裁剪图 (不同尺寸)
    crop_sizes = [(100, 50), (200, 80), (300, 120), (150, 60)]
    stats_recognize = LatencyStats("OCR Recognize", "ms")
    stats_preprocess = LatencyStats("OCR Preprocess", "ms")

    n_iters = 10

    for w, h in crop_sizes:
        for _ in range(n_iters):
            crop = np.random.randint(50, 220, (h, w, 3), dtype=np.uint8)
            # Add some text-like patterns
            import cv2
            cv2.putText(crop, "NE555", (10, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Preprocess
            t0 = time.perf_counter()
            processed = preprocess_chip_crop(crop)
            t1 = time.perf_counter()
            stats_preprocess.samples.append((t1 - t0) * 1000)

            # Recognize
            t0 = time.perf_counter()
            result = ocr.recognize_image(crop)
            t1 = time.perf_counter()
            stats_recognize.samples.append((t1 - t0) * 1000)

    for s in [stats_preprocess, stats_recognize]:
        print(s.summary())

    return {
        "status": "ok",
        "backend": ocr.backend_name,
        "preprocess": stats_preprocess.to_dict(),
        "recognize": stats_recognize.to_dict(),
    }


# ============================================================
# 电路分析基准
# ============================================================

def benchmark_circuit_analysis() -> dict:
    """
    测量电路分析模块性能:
      - add_component 吞吐量
      - build_topology_graph 延迟
      - VF2++ 同构匹配延迟
    """
    print("\n" + "=" * 60)
    print("Circuit Analysis Benchmark")
    print("=" * 60)

    from logic.circuit import CircuitAnalyzer, CircuitComponent, Polarity, PinRole
    from logic.validator import CircuitValidator

    n_components_list = [5, 10, 20, 50]
    stats_build = LatencyStats("Build Graph", "ms")
    stats_topo = LatencyStats("Build Topology", "ms")
    stats_vf2 = LatencyStats("VF2++ Compare", "ms")

    for n in n_components_list:
        # 创建参考电路
        ref = CircuitAnalyzer()
        for i in range(n):
            comp = CircuitComponent(
                name=f"R{i}" if i % 3 == 0 else (f"LED{i}" if i % 3 == 1 else f"Wire{i}"),
                type="RESISTOR" if i % 3 == 0 else ("LED" if i % 3 == 1 else "Wire"),
                pin1_loc=(str(i), "A"),
                pin2_loc=(str(i), "B"),
                confidence=0.9,
                polarity=Polarity.FORWARD if i % 3 == 1 else Polarity.NONE,
                pin_roles=(PinRole.ANODE, PinRole.CATHODE) if i % 3 == 1 else (PinRole.GENERIC, PinRole.GENERIC),
            )
            ref.add_component(comp)

        # 测量 build graph
        for _ in range(10):
            analyzer = CircuitAnalyzer()
            t0 = time.perf_counter()
            for i in range(n):
                comp = CircuitComponent(
                    name=f"R{i}",
                    type="RESISTOR",
                    pin1_loc=(str(i), "A"),
                    pin2_loc=(str(i), "B"),
                    confidence=0.9,
                )
                analyzer.add_component(comp)
            t1 = time.perf_counter()
            stats_build.samples.append((t1 - t0) * 1000)

        # 测量 build topology
        for _ in range(10):
            t0 = time.perf_counter()
            try:
                topo = ref.build_topology_graph()
            except Exception:
                pass
            t1 = time.perf_counter()
            stats_topo.samples.append((t1 - t0) * 1000)

        # 测量 VF2++ compare
        validator = CircuitValidator()
        validator.set_reference(ref)
        for _ in range(5):
            curr = CircuitAnalyzer()
            for i in range(n):
                comp = CircuitComponent(
                    name=f"R{i}" if i % 3 == 0 else (f"LED{i}" if i % 3 == 1 else f"Wire{i}"),
                    type="RESISTOR" if i % 3 == 0 else ("LED" if i % 3 == 1 else "Wire"),
                    pin1_loc=(str(i), "A"),
                    pin2_loc=(str(i), "B"),
                    confidence=0.9,
                    polarity=Polarity.FORWARD if i % 3 == 1 else Polarity.NONE,
                    pin_roles=(PinRole.ANODE, PinRole.CATHODE) if i % 3 == 1 else (PinRole.GENERIC, PinRole.GENERIC),
                )
                curr.add_component(comp)
            t0 = time.perf_counter()
            result = validator.compare(curr)
            t1 = time.perf_counter()
            stats_vf2.samples.append((t1 - t0) * 1000)

    print(f"  Component counts tested: {n_components_list}")
    for s in [stats_build, stats_topo, stats_vf2]:
        print(s.summary())

    return {
        "status": "ok",
        "build_graph": stats_build.to_dict(),
        "build_topology": stats_topo.to_dict(),
        "vf2_compare": stats_vf2.to_dict(),
    }


# ============================================================
# 报告生成
# ============================================================

def generate_report(results: dict, export_path: Optional[str] = None) -> str:
    """生成可读的基准测试报告"""

    lines = []
    lines.append("")
    lines.append("=" * 65)
    lines.append("  LabGuardian Performance Benchmark Report")
    lines.append("=" * 65)
    lines.append("")

    import platform
    lines.append(f"  Platform:  {platform.platform()}")
    lines.append(f"  Python:    {sys.version.split()[0]}")
    lines.append(f"  CPU:       {platform.processor() or 'N/A'}")

    try:
        import openvino
        lines.append(f"  OpenVINO:  {openvino.__version__}")
    except ImportError:
        lines.append("  OpenVINO:  not installed")

    mem = _get_memory_mb()
    lines.append(f"  Memory:    {mem:.0f} MB (current process)")
    lines.append("")

    # KPI 摘要表
    lines.append("-" * 65)
    lines.append("  KPI Summary")
    lines.append("-" * 65)
    lines.append(f"  {'Metric':<35} {'Value':>10} {'Target':>10}")
    lines.append(f"  {'-'*35} {'-'*10} {'-'*10}")

    yolo = results.get("yolo", {})
    if yolo.get("status") == "ok":
        lat = yolo["latency"]["mean"]
        fps = yolo["fps"]
        lines.append(f"  {'YOLO Inference (mean)':<35} {lat:>8.1f}ms {'<50ms':>10}")
        lines.append(f"  {'YOLO FPS':<35} {fps:>8.1f}   {'>20':>10}")

    pipeline = results.get("pipeline", {})
    if pipeline.get("status") == "ok":
        lat = pipeline["total"]["mean"]
        fps = pipeline["fps"]
        lines.append(f"  {'Pipeline Total (mean)':<35} {lat:>8.1f}ms {'<100ms':>10}")
        lines.append(f"  {'Pipeline FPS':<35} {fps:>8.1f}   {'>15':>10}")

    llm = results.get("llm", {})
    if llm.get("status") == "ok":
        ttft = llm["ttft"]["mean"]
        lines.append(f"  {'LLM TTFT (mean)':<35} {ttft:>8.0f}ms {'<3000ms':>10}")
        lines.append(f"  {'LLM Backend':<35} {llm['backend']:>10}")

    ocr = results.get("ocr", {})
    if ocr.get("status") == "ok":
        lat = ocr["recognize"]["mean"]
        lines.append(f"  {'OCR Recognize (mean)':<35} {lat:>8.1f}ms {'<200ms':>10}")

    circuit = results.get("circuit", {})
    if circuit.get("status") == "ok":
        lat = circuit["vf2_compare"]["mean"]
        lines.append(f"  {'VF2++ Compare (mean)':<35} {lat:>8.1f}ms {'<50ms':>10}")

    lines.append("")
    lines.append("=" * 65)

    report = "\n".join(lines)
    print(report)

    # 导出 JSON
    if export_path:
        import platform as pf
        full_report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "platform": pf.platform(),
            "python": sys.version.split()[0],
            "results": results,
        }
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"\n  Report exported to: {export_path}")

    return report


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LabGuardian Performance Benchmark"
    )
    parser.add_argument(
        "--test",
        choices=["yolo", "pipeline", "llm", "ocr", "circuit", "all"],
        default="all",
        help="Which benchmark to run (default: all)"
    )
    parser.add_argument(
        "--frames", type=int, default=100,
        help="Number of test frames for YOLO/pipeline (default: 100)"
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export JSON report to file"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    results = {}
    test = args.test

    if test in ("yolo", "all"):
        results["yolo"] = benchmark_yolo(args.frames)

    if test in ("pipeline", "all"):
        results["pipeline"] = benchmark_pipeline(args.frames)

    if test in ("circuit", "all"):
        results["circuit"] = benchmark_circuit_analysis()

    if test in ("llm", "all"):
        results["llm"] = benchmark_llm()

    if test in ("ocr", "all"):
        results["ocr"] = benchmark_ocr()

    generate_report(results, export_path=args.export)


if __name__ == "__main__":
    main()
