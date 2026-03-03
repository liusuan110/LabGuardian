"""
LabGuardian 无头测试脚本
========================
目的: 在不启动 GUI 的情况下，测试完整的检测+校准+电路分析流水线。
用法: python test_pipeline.py <image_path>
"""

import sys
import os
import logging
import time
from pathlib import Path

# 路径设置
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(SRC_DIR)

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_pipeline")

import cv2
import numpy as np


def run_test(image_path: str):
    """运行完整流水线测试"""
    logger.info("=" * 60)
    logger.info("LabGuardian Pipeline Test")
    logger.info("=" * 60)

    # 1. 加载图片 (Windows 中文路径兼容)
    logger.info(f"加载图片: {image_path}")
    img_data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if image is None:
        logger.error(f"无法加载图片: {image_path}")
        return
    h, w = image.shape[:2]
    logger.info(f"图片尺寸: {w}x{h}")

    # 2. 初始化 AppContext
    logger.info("初始化 AppContext...")
    t0 = time.time()
    from app_context import AppContext
    ctx = AppContext()
    logger.info(f"AppContext 创建完成 ({time.time() - t0:.2f}s)")

    # 3. 加载 YOLO 模型
    logger.info("加载 YOLO 模型...")
    t0 = time.time()
    yolo_ok = ctx.detector.load()
    logger.info(f"YOLO 加载: {'成功' if yolo_ok else '失败'} ({time.time() - t0:.2f}s)")
    if not yolo_ok:
        logger.error("YOLO 模型加载失败，无法继续测试")
        return

    logger.info(f"YOLO 模型类型: {'OBB' if ctx.detector.is_obb_model else 'HBB'}")
    if ctx.detector.model:
        try:
            logger.info(f"类别: {ctx.detector.model.names}")
        except Exception:
            pass

    # 4. 测试自动校准
    logger.info("-" * 40)
    logger.info("测试自动校准...")
    t0 = time.time()
    cal_ok = ctx.calibrator.auto_calibrate(image)
    cal_time = time.time() - t0
    logger.info(f"自动校准: {'成功' if cal_ok else '失败'} ({cal_time:.2f}s)")
    if cal_ok:
        hole_count = len(ctx.calibrator.hole_centers)
        logger.info(f"检测到 {hole_count} 个孔洞")
        if hasattr(ctx.calibrator, 'row_centers') and ctx.calibrator.row_centers is not None:
            logger.info(f"行中心数: {len(ctx.calibrator.row_centers)}")
        if hasattr(ctx.calibrator, 'col_centers') and ctx.calibrator.col_centers is not None:
            logger.info(f"列中心数: {len(ctx.calibrator.col_centers)}")
    else:
        logger.warning("自动校准失败, 将使用全图回退")
        margin = 0.05
        corners = np.array([
            [w * margin, h * margin],
            [w * (1 - margin), h * margin],
            [w * (1 - margin), h * (1 - margin)],
            [w * margin, h * (1 - margin)],
        ], dtype=np.float32)
        ctx.calibrator.calibrate(corners)
        warped = ctx.calibrator.warp(image)
        ctx.calibrator.detect_holes(warped)
        logger.info(f"回退校准: {len(ctx.calibrator.hole_centers)} 孔洞")

    # 5. 运行 YOLO 检测
    logger.info("-" * 40)
    logger.info("运行 YOLO 检测...")
    t0 = time.time()
    detections = ctx.detector.detect(image, conf=0.25)
    det_time = time.time() - t0
    logger.info(f"检测完成: {len(detections)} 个目标 ({det_time:.2f}s)")

    # 统计各类别
    from collections import Counter
    type_counts = Counter(d.class_name for d in detections)
    for cls_name, count in sorted(type_counts.items()):
        logger.info(f"  {cls_name}: {count}")

    # 6. 视觉引脚-孔洞占用检测 (新方法)
    logger.info("-" * 40)
    logger.info("视觉引脚-孔洞占用检测 (PinHoleDetector)...")
    from vision.pin_hole_detector import PinHoleDetector
    phd = PinHoleDetector()
    t0 = time.time()
    occupancy_map = phd.detect_occupied_holes(image, ctx.calibrator)
    phd_time = time.time() - t0
    logger.info(f"占用检测完成: {len(occupancy_map)} 孔洞分析, 耗时 {phd_time:.2f}s")

    if occupancy_map:
        occ_holes = [(k, v) for k, v in occupancy_map.items() if v > phd.OCCUPANCY_THRESHOLD]
        occ_holes.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"被占用孔洞数: {len(occ_holes)} (阈值>{phd.OCCUPANCY_THRESHOLD:.2f})")
        for (hx, hy), score in occ_holes[:20]:
            loc = ctx.calibrator.hole_to_logic(hx, hy)
            loc_str = f"Row{loc[0]}-{loc[1]}" if loc else "N/A"
            logger.info(f"  {loc_str} score={score:.3f}")

        # 保存占用状态可视化图
        try:
            inv_matrix = np.linalg.inv(ctx.calibrator.matrix)
            vis_img = image.copy()
            for (hx, hy), score in occupancy_map.items():
                pt_w = np.array([[[hx, hy]]], dtype=np.float32)
                pt_f = cv2.perspectiveTransform(pt_w, inv_matrix)[0][0]
                fx, fy = int(pt_f[0]), int(pt_f[1])
                if score > phd.OCCUPANCY_THRESHOLD:
                    # 被占用: 红色圆, 大小按分数缩放
                    radius = max(5, int(score * 12))
                    cv2.circle(vis_img, (fx, fy), radius, (0, 0, 255), 2)
                    loc = ctx.calibrator.hole_to_logic(hx, hy)
                    if loc:
                        cv2.putText(vis_img, f"{loc[0]}-{loc[1]}", (fx+8, fy-3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                else:
                    # 空孔洞: 小绿点
                    cv2.circle(vis_img, (fx, fy), 3, (0, 180, 0), -1)
            occ_vis_path = str(Path(image_path).parent / f"occupancy_debug_{Path(image_path).stem}.jpg")
            cv2.imwrite(occ_vis_path, vis_img)
            logger.info(f"占用状态可视化已保存: {occ_vis_path}")
        except Exception as e:
            logger.warning(f"可视化保存失败: {e}")

    # 7. 对比: 视觉引脚 vs 几何引脚
    logger.info("-" * 40)
    logger.info("引脚定位对比 (视觉 vs 几何):")
    from vision.wire_analyzer import WireAnalyzer
    wire_analyzer = WireAnalyzer()

    for i, det in enumerate(detections):
        # 对 Wire 执行骨架分析
        if det.class_name.upper() == "WIRE" or det.class_name == "Wire":
            try:
                endpoints, color = wire_analyzer.analyze_wire(image, det.bbox)
                if endpoints is not None:
                    det.pin1_pixel, det.pin2_pixel = endpoints
                det.wire_color = color
            except Exception as e:
                logger.debug(f"Wire 分析失败: {e}")

        if not det.pin1_pixel or not det.pin2_pixel:
            continue

        # 几何方法
        geo_loc1_str = "N/A"
        geo_loc2_str = "N/A"
        if ctx.calibrator.is_calibrated:
            geo_loc1 = ctx.calibrator.frame_pixel_to_logic(*det.pin1_pixel)
            geo_loc1_str = f"Row{geo_loc1[0]}-{geo_loc1[1]}" if geo_loc1 else "N/A"
            geo_loc2 = ctx.calibrator.frame_pixel_to_logic(*det.pin2_pixel)
            geo_loc2_str = f"Row{geo_loc2[0]}-{geo_loc2[1]}" if geo_loc2 else "N/A"

        # 视觉方法
        vis_loc1_str = "N/A"
        vis_loc2_str = "N/A"
        if occupancy_map and det.class_name.upper() != "WIRE":
            try:
                vloc1, vloc2 = phd.find_component_pins(
                    det, ctx.calibrator, occupancy_map, det.class_name)
                vis_loc1_str = f"Row{vloc1[0]}-{vloc1[1]}" if vloc1 else "N/A"
                vis_loc2_str = f"Row{vloc2[0]}-{vloc2[1]}" if vloc2 else "N/A"
            except Exception as e:
                vis_loc1_str = f"ERR:{e}"

        match_pin1 = "✓" if geo_loc1_str == vis_loc1_str else "✗"
        match_pin2 = "✓" if geo_loc2_str == vis_loc2_str else "✗"
        wire_info = f" color={det.wire_color}" if det.wire_color else ""

        logger.info(
            f"  [{i}] {det.class_name} conf={det.confidence:.2f}{wire_info}\n"
            f"       几何: pin1={geo_loc1_str}, pin2={geo_loc2_str}\n"
            f"       视觉: pin1={vis_loc1_str}, pin2={vis_loc2_str}  "
            f"[{match_pin1}/{match_pin2}]"
        )

    # 8. 运行完整图片分析流水线 (含视觉引脚检测)
    logger.info("-" * 40)
    logger.info("运行完整 ImageAnalyzer 流水线 (含视觉引脚检测)...")
    t0 = time.time()

    # 重置校准以测试完整流程
    ctx.calibrator.reset()
    from vision.image_analyzer import ImageAnalyzer
    analyzer = ImageAnalyzer(ctx)
    result = analyzer.analyze([image], conf=0.25, imgsz=1280)
    pipeline_time = time.time() - t0
    logger.info(f"流水线完成 ({pipeline_time:.2f}s)")

    if result:
        logger.info(f"检测到 {result.component_count} 个元件")
        logger.info(f"电气网络数: {result.net_count}")
        logger.info(f"检测目标总数: {len(result.detections)}")

        # 保存标注图
        output_path = str(Path(image_path).parent / f"test_result_{Path(image_path).stem}.jpg")
        cv2.imwrite(output_path, result.annotated_image)
        logger.info(f"标注结果已保存: {output_path}")

        # 打印报告
        logger.info("-" * 40)
        logger.info("分析报告:")
        for line in result.report.split('\n'):
            logger.info(f"  {line}")

        # 导出网表
        logger.info("-" * 40)
        with ctx.read_lock():
            if ctx.analyzer.components:
                netlist = ctx.analyzer.export_netlist()
                logger.info(f"网表导出: {len(netlist.get('components', []))} 元件, "
                           f"{len(netlist.get('nets', {}))} 网络")
                for comp in netlist.get('components', []):
                    pins_str = ", ".join(
                        f"{p.get('role','?')}@{p.get('loc','?')}" for p in comp.get('pins', [])
                    )
                    logger.info(f"  {comp['name']} ({comp['type']}): {pins_str}")
            else:
                logger.warning("没有检测到元件，无法导出网表")
    else:
        logger.error("分析返回 None")

    logger.info("=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # 默认使用第一张数据集图片
        default = str(PROJECT_ROOT / "dataset" / "images" / "采集" / "采集" / "1-自然光-01.jpg")
        if os.path.exists(default):
            img_path = default
        else:
            print("用法: python test_pipeline.py <image_path>")
            print(f"默认路径不存在: {default}")
            sys.exit(1)

    run_test(img_path)
