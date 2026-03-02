"""
LabGuardian 帧处理管线
======================
从 MainWindow 中提取的纯计算逻辑, 不依赖任何 Qt 组件.
在 VideoWorker 线程中执行.

职责:
  - YOLO 目标检测 (含 ROI 裁剪 + 帧差分跳帧)
  - 检测稳定化
  - 坐标映射 (帧像素 → 面包板逻辑坐标)
  - 电路分析 (构建 CircuitComponent 并加入 Analyzer)
  - 帧标注 (坐标标签 / OCR 标签)
  - OCR 芯片丝印识别 + RAG 自动检索
"""

import logging
import numpy as np
import cv2
from typing import Callable, Optional

from config import vision as vision_cfg, circuit as circuit_cfg, THREE_PIN_COMPONENTS
from app_context import AppContext
from logic.circuit import CircuitComponent, norm_component_type
from ai.ocr_engine import OCR_TARGET_CLASSES
from vision.wire_analyzer import WireAnalyzer

logger = logging.getLogger(__name__)


class FramePipeline:
    """帧处理管线 — 检测 → 稳定化 → 坐标映射 → 电路分析 → 标注

    纯计算类, 不依赖任何 Qt 组件. 在 VideoWorker 线程中执行.
    MainWindow 通过回调属性接收日志和 RAG 结果.
    """

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.current_detection: str = "None"
        self.wire_analyzer = WireAnalyzer()

        # ---- 回调 (由 MainWindow 设置) ----
        self.on_log: Optional[Callable[[str], None]] = None
        self.on_rag_result: Optional[Callable[[str, str, str], None]] = None
        # (chip_model, short_msg, detail_msg) — detail 用于聊天面板
        self.on_frame_processed: Optional[Callable[[np.ndarray], None]] = None
        # 处理完的帧 → 心跳线程用于缩略图
        self.on_rails_need_assignment: Optional[Callable[[list], None]] = None
        # 当电源轨有连接但未标注时触发, 参数为未标注的轨道ID列表

        # 防止重复提示: 上次通知的未标注轨道集合
        self._last_notified_rails: set = set()

    def _log(self, msg: str):
        """通过回调发送日志, 避免直接依赖 UI"""
        if self.on_log:
            self.on_log(msg)

    def _refine_wire_detections(self, frame: np.ndarray, detections: list):
        """对 Wire 类检测执行骨架端点分析 + 颜色分类, 覆写 OBB/HBB 估计。

        分析失败时保持原始引脚坐标不变 (安全回退)。
        """
        for det in detections:
            if norm_component_type(det.class_name) != "WIRE":
                continue
            try:
                endpoints, color = self.wire_analyzer.analyze_wire(
                    frame, det.bbox)
                if endpoints is not None:
                    det.pin1_pixel, det.pin2_pixel = endpoints
                det.wire_color = color
            except Exception as e:
                logger.debug(f"[WireAnalyzer] 分析失败: {e}")

    # ================================================================
    # 主入口: 帧处理
    # ================================================================

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        每帧回调: 检测 -> 稳定化 -> 坐标映射 -> 电路分析 -> 标注

        优化:
          - ROI 裁剪: 校准后只对面包板区域执行 YOLO, 推理速度提升 30-50%
          - 帧差分跳帧: 画面无变化时跳过 YOLO, 复用上帧检测结果

        线程安全:
          - 在 VideoWorker 线程中执行
          - 通过 ctx.write_lock() 保护 analyzer / stabilizer 的写操作
          - OCR 缓存通过 ctx.ocr_cache_* 线程安全方法访问
        """
        ctx = self.ctx

        if not ctx.detector.model:
            return frame

        conf = vision_cfg.conf_threshold
        skipped = False

        # ---- 帧差分检查: 画面无变化时跳过 YOLO ----
        if vision_cfg.frame_diff_enabled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_rect = ctx._roi_rect

            if ctx._prev_gray is not None:
                if roi_rect is not None:
                    rx1, ry1, rx2, ry2 = roi_rect
                    diff_region = cv2.absdiff(
                        gray[ry1:ry2, rx1:rx2],
                        ctx._prev_gray[ry1:ry2, rx1:rx2],
                    )
                else:
                    diff_region = cv2.absdiff(gray, ctx._prev_gray)

                mean_diff = float(np.mean(diff_region))

                if (mean_diff < vision_cfg.frame_diff_threshold and
                        ctx._frames_since_detect < vision_cfg.max_skip_frames):
                    skipped = True
                    ctx._frames_since_detect += 1
                    stable_dets = ctx._cached_stable_dets

            ctx._prev_gray = gray

        if not skipped:
            # ---- 1. YOLO 检测 (ROI 裁剪优化) ----
            roi_rect = ctx._roi_rect
            if vision_cfg.roi_enabled and roi_rect is not None:
                rx1, ry1, rx2, ry2 = roi_rect
                cropped = frame[ry1:ry2, rx1:rx2]
                detections = ctx.detector.detect(cropped, conf=conf)
                ctx.detector.offset_detections(detections, rx1, ry1)
            else:
                detections = ctx.detector.detect(frame, conf=conf)

            ctx._frames_since_detect = 0

            # ---- 2-4. 稳定化 + 坐标映射 + 电路分析 (写锁保护) ----
            with ctx.write_lock():
                stable_dets = ctx.stabilizer.update(detections)
                ctx._cached_stable_dets = stable_dets

                # ---- Wire 骨架端点精炼 + 颜色分类 ----
                self._refine_wire_detections(frame, stable_dets)

                if ctx.calibrator.is_calibrated and stable_dets:
                    ctx.analyzer.reset()
                    k = circuit_cfg.pin_candidate_k

                    for det in stable_dets:
                        if det.pin1_pixel and det.pin2_pixel:
                            cands1 = ctx.calibrator.frame_pixel_to_logic_candidates(
                                *det.pin1_pixel, k=k)
                            cands2 = ctx.calibrator.frame_pixel_to_logic_candidates(
                                *det.pin2_pixel, k=k)

                            if not cands1 or not cands2:
                                continue

                            loc1, loc2 = self.pick_best_pin_pair(
                                cands1, cands2, det.class_name)

                            if (loc1 and loc2 and
                                    loc1[0] != "Groove" and loc2[0] != "Groove"):
                                comp = CircuitComponent(
                                    name=det.class_name,
                                    type=det.class_name,
                                    pin1_loc=loc1,
                                    pin2_loc=loc2,
                                    confidence=det.confidence,
                                )

                                ntype = norm_component_type(det.class_name)
                                if det.class_name in THREE_PIN_COMPONENTS or ntype in ("NPN", "PNP", "TRANSISTOR"):
                                    self.find_transistor_pin3(det)
                                    pin3 = getattr(det, '_pin3_loc', None)
                                    if pin3 is not None:
                                        comp.pin3_loc = pin3

                                obb_corners = det.obb_corners if det.is_obb else None
                                orientation = self.compute_obb_angle(obb_corners)
                                ctx.polarity.enrich(
                                    comp,
                                    obb_corners=obb_corners,
                                    orientation_deg=orientation,
                                )

                                ctx.analyzer.add_component(comp)

                    try:
                        net_count = ctx.analyzer.get_net_count()
                    except Exception:
                        net_count = 0

                    ctx.update_circuit_snapshot()

            # ---- 检查未标注的活跃电源轨 (读锁) ----
            with ctx.read_lock():
                unassigned = ctx.analyzer.get_unassigned_active_rails()
            if unassigned:
                unassigned_set = set(unassigned)
                if unassigned_set != self._last_notified_rails:
                    self._last_notified_rails = unassigned_set
                    if self.on_rails_need_assignment:
                        self.on_rails_need_assignment(unassigned)

        # ---- 5. 绘制检测结果 ----
        annotated = ctx.detector.annotate_frame(frame, stable_dets)

        if not skipped and ctx.calibrator.is_calibrated and stable_dets:
            for det in stable_dets:
                if det.pin1_pixel and det.pin2_pixel:
                    loc1 = ctx.calibrator.frame_pixel_to_logic(*det.pin1_pixel)
                    loc2 = ctx.calibrator.frame_pixel_to_logic(*det.pin2_pixel)
                    if loc1 and loc2 and loc1[0] != "Groove" and loc2[0] != "Groove":
                        info = f"{loc1[0]}-{loc2[0]}"
                        x1, y1 = det.bbox[:2]
                        cv2.putText(annotated, info, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            try:
                cv2.putText(annotated, f"Nets: {net_count}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            except Exception:
                pass

        if skipped:
            cv2.putText(annotated, "SKIP", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        # ---- 7. OCR 芯片丝印识别 ----
        ctx._ocr_frame_skip += 1
        if (not skipped and ctx.ocr.is_ready and stable_dets and
                ctx._ocr_frame_skip >= ctx._ocr_interval):
            ctx._ocr_frame_skip = 0
            self.run_ocr(frame, stable_dets, annotated)

        # ---- 8. 绘制 OCR 标签 ----
        self.draw_ocr_labels(annotated, stable_dets)

        # ---- 9. 更新检测状态 ----
        if stable_dets:
            top = max(stable_dets, key=lambda d: d.confidence)
            self.current_detection = top.class_name
        else:
            self.current_detection = "None"

        # ---- 10. 通知处理完成 (心跳缩略图) ----
        if self.on_frame_processed:
            self.on_frame_processed(annotated)

        return annotated

    # ================================================================
    # 引脚选择
    # ================================================================

    @staticmethod
    def pick_best_pin_pair(candidates1, candidates2, comp_type):
        """基于面包板约束从候选孔洞中选择最佳引脚对。

        评分规则:
          1. 距离分 — 两引脚到候选孔洞的像素距离之和 (越小越好)
          2. 同导通组惩罚 — 非Wire元件两脚同组 → 短路, 重罚
          3. 同行惩罚 — 非Wire元件两脚同行且同侧 → 不合理
          4. 跨度异常惩罚 — 行跨度 > 阈值 → 轻罚
        """
        ctype = norm_component_type(comp_type)
        is_wire = (ctype == "WIRE")

        best_score = float('inf')
        best_pair = (candidates1[0][0], candidates2[0][0])

        for loc1, dist1 in candidates1:
            for loc2, dist2 in candidates2:
                score = dist1 + dist2

                row1, col1 = loc1
                row2, col2 = loc2

                try:
                    r1, r2 = int(row1), int(row2)
                except (ValueError, TypeError):
                    continue

                group1 = 'L' if col1 in ('a', 'b', 'c', 'd', 'e') else 'R'
                group2 = 'L' if col2 in ('a', 'b', 'c', 'd', 'e') else 'R'
                same_group = (r1 == r2 and group1 == group2)

                if not is_wire:
                    if same_group:
                        score += circuit_cfg.pin_same_group_penalty
                    elif r1 == r2:
                        score += circuit_cfg.pin_same_row_penalty * 0.3

                    span = abs(r2 - r1)
                    if span > circuit_cfg.pin_large_span_threshold:
                        score += 20.0

                if score < best_score:
                    best_score = score
                    best_pair = (loc1, loc2)

        return best_pair

    @staticmethod
    def compute_obb_angle(corners) -> float:
        """从 OBB 四角坐标计算旋转角度 (度)。无 OBB 时返回 0.0。"""
        if corners is None or len(corners) < 4:
            return 0.0
        p0, p1, p2, p3 = corners
        d01 = np.linalg.norm(p0 - p1)
        d12 = np.linalg.norm(p1 - p2)
        if d01 > d12:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        else:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return float(np.degrees(np.arctan2(dy, dx)))

    def find_transistor_pin3(self, det) -> None:
        """为三极管检测结果搜索第三引脚 (基极) 的实际孔洞位置。

        在 pin1 和 pin2 之间的线段上搜索校准器已检测到的孔洞,
        找到距线段最近且严格位于两端点之间的孔洞作为 pin3。
        """
        ctx = self.ctx
        cal = ctx.calibrator

        if (not cal.is_calibrated or cal.matrix is None or
                not cal.hole_centers or len(cal.hole_centers) < 10):
            return

        p1_frame = np.array([[[det.pin1_pixel[0], det.pin1_pixel[1]]]],
                            dtype=np.float32)
        p2_frame = np.array([[[det.pin2_pixel[0], det.pin2_pixel[1]]]],
                            dtype=np.float32)
        p1_warp = cv2.perspectiveTransform(p1_frame, cal.matrix)[0][0]
        p2_warp = cv2.perspectiveTransform(p2_frame, cal.matrix)[0][0]

        seg = p2_warp - p1_warp
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-5:
            return

        seg_unit = seg / seg_len

        best_hole = None
        best_dist_to_mid = float('inf')

        if cal.row_centers is not None and len(cal.row_centers) > 1:
            avg_spacing = float(np.mean(np.diff(cal.row_centers)))
        else:
            avg_spacing = 20.0
        perp_threshold = avg_spacing * 0.6

        mid_point = (p1_warp + p2_warp) / 2

        for hx, hy in cal.hole_centers:
            h = np.array([hx, hy], dtype=np.float32)
            v = h - p1_warp
            t = float(np.dot(v, seg_unit))

            margin = seg_len * 0.15
            if t <= margin or t >= seg_len - margin:
                continue

            proj_point = p1_warp + seg_unit * t
            perp_dist = float(np.linalg.norm(h - proj_point))

            if perp_dist > perp_threshold:
                continue

            dist_to_mid = float(np.linalg.norm(h - mid_point))
            if dist_to_mid < best_dist_to_mid:
                best_dist_to_mid = dist_to_mid
                best_hole = (hx, hy)

        if best_hole is not None:
            pin3_logic = cal.hole_to_logic(best_hole[0], best_hole[1])
            if pin3_logic is not None and pin3_logic[0] != "Groove":
                det._pin3_loc = pin3_logic
                logger.debug(f"[Pin3] {det.class_name}: found pin3 at "
                             f"hole ({best_hole[0]:.0f}, {best_hole[1]:.0f}) "
                             f"→ {pin3_logic}")
            else:
                det._pin3_loc = None
        else:
            det._pin3_loc = None

    # ================================================================
    # OCR 子系统
    # ================================================================

    def run_ocr(self, frame: np.ndarray, detections: list,
                annotated: np.ndarray):
        """对芯片类检测结果执行 OCR 识别丝印, 新识别出型号时自动查询 RAG"""
        ctx = self.ctx
        target_classes = {c.upper() for c in OCR_TARGET_CLASSES}
        for det in detections:
            if det.class_name.upper() not in target_classes:
                continue
            cx = (det.bbox[0] + det.bbox[2]) // 2
            cy = (det.bbox[1] + det.bbox[3]) // 2
            cache_key = f"{cx//50}_{cy//50}"
            if ctx.ocr_cache_has(cache_key):
                continue
            try:
                result = ctx.ocr.recognize_chip(frame, det.bbox)
                if result.chip_model:
                    ctx.ocr_cache_set(cache_key, result.chip_model)
                    logger.info(f"[OCR] {det.class_name} -> {result.chip_model}")
                    self.auto_rag_lookup(result.chip_model)
            except Exception as e:
                logger.warning(f"[OCR] 识别出错: {e}")

    def auto_rag_lookup(self, chip_model: str):
        """当 OCR 识别出新芯片型号时, 自动查询 RAG 知识库.

        通过 on_log 和 on_rag_result 回调通知 UI, 不直接操作 GUI 组件.
        """
        ctx = self.ctx
        if chip_model.upper() in ctx._rag_queried_models:
            return
        ctx._rag_queried_models.add(chip_model.upper())

        if not ctx.llm.rag_ready:
            self._log(f"识别到芯片: {chip_model} (RAG 未就绪, 跳过知识检索)")
            return

        try:
            query = f"{chip_model} 引脚定义 功能 使用方法"
            results = ctx.llm.rag.query(query, top_k=2, min_score=0.3)
            if results:
                best = results[0]
                snippet = best["text"][:150].replace("\n", " ").strip()
                if len(best["text"]) > 150:
                    snippet += "..."
                self._log(f"识别到 [{chip_model}] - {snippet}")

                # 通过回调通知 UI 显示详细信息
                if self.on_rag_result:
                    detail = (
                        f"**自动识别**: 检测到芯片 **{chip_model}**\n"
                        f"> {snippet}\n\n"
                        f"输入 `{chip_model} 引脚` 可查看详细引脚定义"
                    )
                    short = f"识别到芯片 {chip_model}: {snippet}"
                    self.on_rag_result(chip_model, detail, short)

                logger.info(f"[RAG] 自动检索 {chip_model}: 相关度 {best['score']:.1%}")
            else:
                self._log(f"识别到芯片: {chip_model} (知识库暂无相关信息)")
        except Exception as e:
            logger.warning(f"[RAG] 自动检索 {chip_model} 出错: {e}")

    def draw_ocr_labels(self, frame: np.ndarray, detections: list):
        """在帧上绘制已识别的芯片型号标签"""
        ocr_cache = self.ctx.get_ocr_cache_copy()
        if not ocr_cache:
            return
        target_classes = {c.upper() for c in OCR_TARGET_CLASSES}
        for det in detections:
            if det.class_name.upper() not in target_classes:
                continue
            cx = (det.bbox[0] + det.bbox[2]) // 2
            cy = (det.bbox[1] + det.bbox[3]) // 2
            cache_key = f"{cx//50}_{cy//50}"
            model_name = ocr_cache.get(cache_key)
            if model_name:
                x1, y2 = det.bbox[0], det.bbox[3]
                label = f"[{model_name}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y2 + 2), (x1 + tw + 4, y2 + th + 8),
                              (128, 64, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y2 + th + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

