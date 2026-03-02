"""
图片分析引擎
============
职责:
  - 接收 1-3 张面包板俯拍图片
  - 对每张图片独立执行 YOLO 检测 (高分辨率 1280)
  - Wire 骨架端点精炼 + 颜色分类
  - 多图检测结果 IoU 融合 (处理遮挡)
  - 坐标映射 → 电路拓扑建模 → 结构化报告生成
  - OCR 芯片丝印识别

替代 FramePipeline, 用于 Image-only 模式 (无视频流).
"""

import logging
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import Counter

from config import vision as vision_cfg, circuit as circuit_cfg, THREE_PIN_COMPONENTS, IC_DIP_COMPONENTS
from app_context import AppContext
from logic.circuit import CircuitComponent, PinRole, norm_component_type
from logic.validator import CircuitValidator
from ai.ocr_engine import OCR_TARGET_CLASSES
from vision.wire_analyzer import WireAnalyzer
from vision.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """一次分析的完整结果"""
    annotated_image: np.ndarray             # 标注后的图片 (用于显示)
    detections: list                        # 融合后的检测列表
    report: str                             # 结构化文本报告
    component_count: int = 0
    net_count: int = 0
    ocr_results: Dict[str, str] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


class ImageAnalyzer:
    """图片分析引擎 — 一次性分析 1-3 张图片, 返回结构化结果.

    替代 FramePipeline, 用于 Image-only 架构.
    不涉及任何 Qt / GUI 组件.
    """

    # IoU 融合阈值: IoU > 此值则视为同一检测目标
    IOU_MERGE_THRESHOLD = 0.3

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.wire_analyzer = WireAnalyzer()

    # ================================================================
    # 主入口
    # ================================================================

    def analyze(
        self,
        images: List[np.ndarray],
        conf: float = 0.25,
        imgsz: int = 1280,
        progress_callback=None,
    ) -> AnalysisResult:
        """对 1-3 张图片执行完整分析流程.

        Args:
            images: BGR 图片列表 (1-3 张)
            conf: YOLO 置信度阈值
            imgsz: YOLO 推理分辨率
            progress_callback: fn(str) 进度回调

        Returns:
            AnalysisResult 完整分析结果
        """
        if not images:
            return AnalysisResult(
                annotated_image=np.zeros((480, 640, 3), dtype=np.uint8),
                detections=[],
                report="未提供图片.",
            )

        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        ctx = self.ctx
        primary = images[0]

        # ---- 1. 校准 (使用第一张图片) ----
        _progress("校准面包板...")
        if not ctx.calibrator.is_calibrated:
            from vision.calibrator import board_calibrator
            board_calibrator.auto_detect_board(primary)

        # ---- 2. 逐张检测 ----
        all_det_lists: List[List[Detection]] = []
        for i, img in enumerate(images):
            _progress(f"检测中 ({i + 1}/{len(images)})...")
            dets = self._detect_single(img, conf, imgsz)
            self._refine_wires(img, dets)
            all_det_lists.append(dets)

        # ---- 3. 多图融合 ----
        _progress("融合检测结果...")
        if len(all_det_lists) == 1:
            merged = all_det_lists[0]
        else:
            merged = self._fuse_detections(all_det_lists)

        # ---- 4. 坐标映射 + 电路分析 ----
        _progress("电路分析...")
        net_count, report = self._build_circuit(merged, primary)

        # ---- 5. OCR ----
        _progress("芯片识别...")
        ocr_results = self._run_ocr(primary, merged)

        # ---- 6. 标注 ----
        _progress("生成标注图...")
        annotated = self._annotate(primary, merged)

        # ---- 7. 综合报告 ----
        issues = self._collect_issues()
        component_count = len([
            d for d in merged
            if norm_component_type(d.class_name) != "WIRE"
        ])
        full_report = self._generate_report(
            merged, net_count, ocr_results, issues, len(images),
        )

        _progress("分析完成")

        return AnalysisResult(
            annotated_image=annotated,
            detections=merged,
            report=full_report,
            component_count=component_count,
            net_count=net_count,
            ocr_results=ocr_results,
            issues=issues,
        )

    # ================================================================
    # 子步骤
    # ================================================================

    def _detect_single(
        self, image: np.ndarray, conf: float, imgsz: int,
    ) -> List[Detection]:
        """对单张图片执行 YOLO 检测 (含 ROI 裁剪)."""
        ctx = self.ctx
        if ctx.detector.model is None:
            return []

        # 临时覆盖 imgsz
        old_imgsz = vision_cfg.imgsz
        vision_cfg.imgsz = imgsz
        try:
            roi_rect = ctx._roi_rect
            if vision_cfg.roi_enabled and roi_rect is not None:
                rx1, ry1, rx2, ry2 = roi_rect
                cropped = image[ry1:ry2, rx1:rx2]
                dets = ctx.detector.detect(cropped, conf=conf)
                ctx.detector.offset_detections(dets, rx1, ry1)
            else:
                dets = ctx.detector.detect(image, conf=conf)
        finally:
            vision_cfg.imgsz = old_imgsz

        return dets

    def _refine_wires(self, image: np.ndarray, detections: List[Detection]):
        """对 Wire 类检测执行骨架端点分析 + 颜色分类 (原地修改)."""
        for det in detections:
            if norm_component_type(det.class_name) != "WIRE":
                continue
            try:
                endpoints, color = self.wire_analyzer.analyze_wire(
                    image, det.bbox)
                if endpoints is not None:
                    det.pin1_pixel, det.pin2_pixel = endpoints
                det.wire_color = color
            except Exception as e:
                logger.debug(f"[WireAnalyzer] 分析失败: {e}")

    def _fuse_detections(
        self, det_lists: List[List[Detection]],
    ) -> List[Detection]:
        """多图检测结果 IoU 融合.

        算法:
          - 以第一张图的检测为基准列表
          - 对后续图片的每个检测, 在基准中寻找 IoU 最高的匹配
          - IoU > 阈值: 合并 (保留高置信度, 平均引脚位置)
          - 无匹配: 作为新检测加入基准 (可能被前图遮挡)
        """
        if not det_lists:
            return []

        base = list(det_lists[0])  # 拷贝

        for extra_dets in det_lists[1:]:
            for det in extra_dets:
                best_iou = 0.0
                best_idx = -1

                for i, base_det in enumerate(base):
                    iou = self._compute_iou(det.bbox, base_det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

                if best_iou >= self.IOU_MERGE_THRESHOLD and best_idx >= 0:
                    # 合并: 保留置信度更高的, 平均引脚位置
                    bd = base[best_idx]
                    if det.confidence > bd.confidence:
                        bd.confidence = det.confidence
                        bd.bbox = det.bbox
                        if det.is_obb:
                            bd.is_obb = True
                            bd.obb_corners = det.obb_corners
                    # 平均引脚位置
                    if det.pin1_pixel and bd.pin1_pixel:
                        bd.pin1_pixel = (
                            (det.pin1_pixel[0] + bd.pin1_pixel[0]) / 2,
                            (det.pin1_pixel[1] + bd.pin1_pixel[1]) / 2,
                        )
                    if det.pin2_pixel and bd.pin2_pixel:
                        bd.pin2_pixel = (
                            (det.pin2_pixel[0] + bd.pin2_pixel[0]) / 2,
                            (det.pin2_pixel[1] + bd.pin2_pixel[1]) / 2,
                        )
                else:
                    # 新检测 (前图可能被遮挡)
                    base.append(det)

        return base

    @staticmethod
    def _compute_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """计算两个 bbox 的 IoU."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def _build_circuit(
        self, detections: List[Detection], image: np.ndarray,
    ) -> Tuple[int, str]:
        """坐标映射 + 电路拓扑建模, 返回 (网络数, 电路描述快照)."""
        ctx = self.ctx

        with ctx.write_lock():
            ctx.analyzer.reset()

            if not ctx.calibrator.is_calibrated or not detections:
                return 0, ""

            k = circuit_cfg.pin_candidate_k

            for det in detections:
                if not det.pin1_pixel or not det.pin2_pixel:
                    continue

                cands1 = ctx.calibrator.frame_pixel_to_logic_candidates(
                    *det.pin1_pixel, k=k)
                cands2 = ctx.calibrator.frame_pixel_to_logic_candidates(
                    *det.pin2_pixel, k=k)

                if not cands1 or not cands2:
                    continue

                loc1, loc2 = self._pick_best_pin_pair(
                    cands1, cands2, det.class_name)

                if (loc1 and loc2 and
                        loc1[0] != "Groove" and loc2[0] != "Groove"):

                    ntype = norm_component_type(det.class_name)

                    # IC 多引脚: OCR 识别型号 → 引脚数据库 → 全引脚映射
                    if ntype == "IC_DIP":
                        ic_comp = self._build_ic_component(det, image, loc1, loc2)
                        if ic_comp:
                            obb_corners = det.obb_corners if det.is_obb else None
                            orientation = self._compute_obb_angle(obb_corners)
                            ctx.polarity.enrich(
                                ic_comp,
                                obb_corners=obb_corners,
                                orientation_deg=orientation,
                            )
                            ctx.analyzer.add_component(ic_comp)
                            continue
                        # OCR 失败或型号未知 → 回退到 2-pin 处理

                    comp = CircuitComponent(
                        name=det.class_name,
                        type=det.class_name,
                        pin1_loc=loc1,
                        pin2_loc=loc2,
                        confidence=det.confidence,
                    )

                    # 三端元件 (三极管/电位器): 搜索第三引脚
                    if (det.class_name in THREE_PIN_COMPONENTS or
                            ntype in ("NPN", "PNP", "TRANSISTOR", "POTENTIOMETER")):
                        self._find_3pin_middle(det)
                        pin3 = getattr(det, '_pin3_loc', None)
                        if pin3 is not None:
                            comp.pin3_loc = pin3

                    obb_corners = det.obb_corners if det.is_obb else None
                    orientation = self._compute_obb_angle(obb_corners)
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
            report = ctx.get_circuit_snapshot()

        return net_count, report

    def _run_ocr(
        self, image: np.ndarray, detections: List[Detection],
    ) -> Dict[str, str]:
        """对芯片类检测执行 OCR 识别, 返回 {cache_key: model_name}."""
        ctx = self.ctx
        results: Dict[str, str] = {}

        if not ctx.ocr.is_ready:
            return results

        target_classes = {c.upper() for c in OCR_TARGET_CLASSES}

        for det in detections:
            if det.class_name.upper() not in target_classes:
                continue
            cx = (det.bbox[0] + det.bbox[2]) // 2
            cy = (det.bbox[1] + det.bbox[3]) // 2
            cache_key = f"{cx // 50}_{cy // 50}"

            try:
                result = ctx.ocr.recognize_chip(image, det.bbox)
                if result.chip_model:
                    results[cache_key] = result.chip_model
                    ctx.ocr_cache_set(cache_key, result.chip_model)
                    logger.info(f"[OCR] {det.class_name} -> {result.chip_model}")
            except Exception as e:
                logger.warning(f"[OCR] 识别出错: {e}")

        return results

    def _annotate(
        self, image: np.ndarray, detections: List[Detection],
    ) -> np.ndarray:
        """在图片上绘制检测标注 + 坐标标签 + OCR 标签."""
        ctx = self.ctx
        annotated = ctx.detector.annotate_frame(image, detections)

        # 坐标标签
        if ctx.calibrator.is_calibrated:
            for det in detections:
                if det.pin1_pixel and det.pin2_pixel:
                    loc1 = ctx.calibrator.frame_pixel_to_logic(*det.pin1_pixel)
                    loc2 = ctx.calibrator.frame_pixel_to_logic(*det.pin2_pixel)
                    if loc1 and loc2 and loc1[0] != "Groove" and loc2[0] != "Groove":
                        info = f"{loc1[0]}-{loc2[0]}"
                        x1, y1 = det.bbox[:2]
                        cv2.putText(annotated, info, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 255), 1)

        # 网络数
        try:
            with ctx.read_lock():
                net_count = ctx.analyzer.get_net_count()
            cv2.putText(annotated, f"Nets: {net_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        except Exception:
            pass

        # OCR 标签
        self._draw_ocr_labels(annotated, detections)

        return annotated

    def _draw_ocr_labels(self, frame: np.ndarray, detections: List[Detection]):
        """在帧上绘制已识别的芯片型号标签."""
        ocr_cache = self.ctx.get_ocr_cache_copy()
        if not ocr_cache:
            return
        target_classes = {c.upper() for c in OCR_TARGET_CLASSES}
        for det in detections:
            if det.class_name.upper() not in target_classes:
                continue
            cx = (det.bbox[0] + det.bbox[2]) // 2
            cy = (det.bbox[1] + det.bbox[3]) // 2
            cache_key = f"{cx // 50}_{cy // 50}"
            model_name = ocr_cache.get(cache_key)
            if model_name:
                x1, y2 = det.bbox[0], det.bbox[3]
                label = f"[{model_name}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y2 + 2), (x1 + tw + 4, y2 + th + 8),
                              (128, 64, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y2 + th + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _collect_issues(self) -> List[str]:
        """收集分析诊断问题."""
        try:
            with self.ctx.read_lock():
                return CircuitValidator.diagnose(self.ctx.analyzer)
        except Exception:
            return []

    def _generate_report(
        self,
        detections: List[Detection],
        net_count: int,
        ocr_results: Dict[str, str],
        issues: List[str],
        image_count: int,
    ) -> str:
        """生成结构化分析报告."""
        # 元件统计
        type_counts = Counter()
        wire_count = 0
        for det in detections:
            ntype = norm_component_type(det.class_name)
            if ntype == "WIRE":
                wire_count += 1
            else:
                type_counts[ntype] += 1

        total = len(detections)
        comp_str = ", ".join(f"{t}x{c}" for t, c in sorted(type_counts.items()))
        if wire_count:
            comp_str += f", Wirex{wire_count}" if comp_str else f"Wirex{wire_count}"

        report = f"分析结果 ({image_count} 张图片)\n"
        report += "=" * 40 + "\n"
        report += f"检测到 {total} 个目标: {comp_str}\n"
        report += f"电气网络数: {net_count}\n\n"

        # 电路描述 (来自 CircuitAnalyzer)
        desc = self.ctx.get_circuit_snapshot()
        if desc:
            report += desc + "\n"

        # OCR 结果
        if ocr_results:
            report += "\n芯片识别:\n"
            for key, model in ocr_results.items():
                report += f"  {model}\n"

        # 诊断问题
        if issues:
            report += "\n诊断问题:\n"
            for issue in issues:
                report += f"  - {issue}\n"

        return report

    # ================================================================
    # 辅助方法 (从 FramePipeline 提取)
    # ================================================================

    @staticmethod
    def _pick_best_pin_pair(candidates1, candidates2, comp_type):
        """基于面包板约束从候选孔洞中选择最佳引脚对."""
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
    def _compute_obb_angle(corners) -> float:
        """从 OBB 四角坐标计算旋转角度 (度)."""
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

    def _build_ic_component(
        self, det, image: np.ndarray,
        loc1: Tuple[str, str], loc2: Tuple[str, str],
    ) -> Optional[CircuitComponent]:
        """构建 IC 多引脚 CircuitComponent.

        策略:
          1. OCR 识别 IC 型号
          2. 查询引脚数据库获取引脚定义
          3. 从 loc1/loc2 推导 IC 顶部行号
          4. 计算所有引脚的 (row, col) 坐标 (DIP 封装约定)
          5. 创建带 pin_locs + pin_roles 的 CircuitComponent

        如果 OCR 失败或型号不在数据库中, 返回 None (回退到 2-pin).
        """
        from logic.ic_pinout_db import lookup_ic, get_ic_pin_locs

        # 1. OCR 识别型号
        model_name = None
        if self.ctx.ocr.is_ready:
            try:
                result = self.ctx.ocr.recognize_chip(image, det.bbox)
                if result.chip_model:
                    model_name = result.chip_model
            except Exception:
                pass

        # 也检查 OCR 缓存
        if not model_name:
            cx = (det.bbox[0] + det.bbox[2]) // 2
            cy = (det.bbox[1] + det.bbox[3]) // 2
            cache_key = f"{cx // 50}_{cy // 50}"
            model_name = self.ctx.ocr_cache_get(cache_key)

        if not model_name:
            return None

        # 2. 查询引脚数据库
        ic_info = lookup_ic(model_name)
        if ic_info is None:
            return None

        # 3. 推导顶部行号
        try:
            row1, row2 = int(loc1[0]), int(loc2[0])
        except (ValueError, TypeError):
            return None
        top_row = min(row1, row2)

        # 4. 计算所有引脚坐标
        pin_locations = get_ic_pin_locs(ic_info, top_row)
        pin_locs = [(str(row), col) for row, col, _ in pin_locations]

        # 5. 构建 pin_roles
        role_map = {
            "power_vcc": PinRole.IC_VCC,
            "power_gnd": PinRole.IC_GND,
            "output": PinRole.IC_OUTPUT,
            "input_inv": PinRole.IC_INPUT_INV,
            "input_non": PinRole.IC_INPUT_NON,
        }
        pin_roles = []
        for _, _, pdef in pin_locations:
            pin_roles.append(role_map.get(pdef.function, PinRole.IC_PIN))

        comp = CircuitComponent(
            name=det.class_name,
            type="IC_DIP",
            pin1_loc=pin_locs[0] if pin_locs else loc1,
            pin2_loc=pin_locs[1] if len(pin_locs) > 1 else loc2,
            confidence=det.confidence,
            pin_locs=pin_locs,
            pin_roles=tuple(pin_roles),
            ic_model=model_name,
        )

        logger.info(f"[IC] Built {model_name} ({ic_info.pin_count}-pin) "
                     f"at rows {top_row}-{top_row + ic_info.pin_count // 2 - 1}")
        return comp

    def _find_3pin_middle(self, det) -> None:
        """为三端元件 (三极管/电位器) 搜索第三引脚 (中间) 的孔洞位置."""
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
            else:
                det._pin3_loc = None
        else:
            det._pin3_loc = None
