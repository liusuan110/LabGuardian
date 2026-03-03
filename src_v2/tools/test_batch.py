"""
LabGuardian 批量验证脚本
========================
对数据集中多张图片运行完整检测+视觉引脚检测流水线，
逐一输出每个元件的引脚识别结果，并保存可视化标注图。
"""

import sys, os, logging, time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(SRC_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_test")

import cv2
import numpy as np


def load_image(path: str):
    """Windows 中文路径兼容加载"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def draw_pin_annotations(image, det, calibrator, phd, occupancy_map, idx):
    """在图上详细标注引脚位置"""
    vis = image.copy()
    inv_m = np.linalg.inv(calibrator.matrix)

    # 标注所有被占用孔洞
    for (hx, hy), score in occupancy_map.items():
        pt_w = np.array([[[hx, hy]]], dtype=np.float32)
        pt_f = cv2.perspectiveTransform(pt_w, inv_m)[0][0]
        fx, fy = int(pt_f[0]), int(pt_f[1])
        if score > phd.OCCUPANCY_THRESHOLD:
            radius = max(4, int(score * 10))
            cv2.circle(vis, (fx, fy), radius, (0, 0, 255), 2)
        else:
            cv2.circle(vis, (fx, fy), 2, (0, 140, 0), -1)

    return vis


def run_batch(image_paths: list):
    """对一批图片运行测试"""

    # 初始化
    from app_context import AppContext
    ctx = AppContext()

    logger.info("加载 YOLO 模型...")
    if not ctx.detector.load():
        logger.error("YOLO 模型加载失败")
        return

    logger.info(f"YOLO 类型: {'OBB' if ctx.detector.is_obb_model else 'HBB'}")
    try:
        logger.info(f"类别: {ctx.detector.model.names}")
    except Exception:
        pass

    from vision.pin_hole_detector import PinHoleDetector
    from vision.wire_analyzer import WireAnalyzer
    from vision.image_analyzer import ImageAnalyzer

    phd = PinHoleDetector()
    wire_analyzer = WireAnalyzer()

    output_dir = PROJECT_ROOT / "test_results"
    output_dir.mkdir(exist_ok=True)

    summary = []

    for img_idx, img_path in enumerate(image_paths):
        stem = Path(img_path).stem
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"[{img_idx+1}/{len(image_paths)}] {Path(img_path).name}")
        logger.info("=" * 70)

        image = load_image(img_path)
        if image is None:
            logger.error(f"  加载失败: {img_path}")
            continue
        h, w = image.shape[:2]
        logger.info(f"  尺寸: {w}x{h}")

        # --- 自动校准 ---
        ctx.calibrator.reset()
        t0 = time.time()
        cal_ok = ctx.calibrator.auto_calibrate(image)
        cal_time = time.time() - t0
        hole_count = len(ctx.calibrator.hole_centers) if cal_ok else 0
        logger.info(f"  校准: {'成功' if cal_ok else '失败'} "
                    f"({cal_time:.2f}s, {hole_count} 孔洞)")
        if not cal_ok:
            logger.warning("  跳过 (校准失败)")
            continue

        # --- YOLO 检测 ---
        t0 = time.time()
        detections = ctx.detector.detect(image, conf=0.25)
        det_time = time.time() - t0
        logger.info(f"  YOLO: {len(detections)} 目标 ({det_time:.2f}s)")
        for d in detections:
            logger.info(f"    {d.class_name} conf={d.confidence:.2f}")

        # --- Wire 骨架分析 ---
        for det in detections:
            if det.class_name.upper() == "WIRE" or det.class_name == "Wire":
                try:
                    endpoints, color = wire_analyzer.analyze_wire(image, det.bbox)
                    if endpoints is not None:
                        det.pin1_pixel, det.pin2_pixel = endpoints
                    det.wire_color = color
                except Exception:
                    pass

        # --- 视觉引脚占用检测 ---
        t0 = time.time()
        occupancy_map = phd.detect_occupied_holes(image, ctx.calibrator)
        phd_time = time.time() - t0
        occ_holes = [(k, v) for k, v in occupancy_map.items()
                     if v > phd.OCCUPANCY_THRESHOLD]
        occ_holes.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"  占用检测: {len(occ_holes)} 被占用 / {len(occupancy_map)} 总计 ({phd_time:.2f}s)")

        # 打印 Top 占用孔洞
        logger.info(f"  Top 被占用孔洞:")
        for (hx, hy), score in occ_holes[:15]:
            loc = ctx.calibrator.hole_to_logic(hx, hy)
            loc_str = f"Row{loc[0]}-{loc[1]}" if loc else "N/A"
            logger.info(f"    {loc_str:>12s}  score={score:.3f}")

        # --- 逐元件引脚识别 ---
        logger.info(f"  " + "-" * 50)
        logger.info(f"  逐元件引脚识别结果:")

        img_results = []
        for i, det in enumerate(detections):
            if det.pin1_pixel is None or det.pin2_pixel is None:
                logger.info(f"    [{i}] {det.class_name}: 无引脚估计")
                continue

            # 几何方法
            geo1 = ctx.calibrator.frame_pixel_to_logic(*det.pin1_pixel)
            geo2 = ctx.calibrator.frame_pixel_to_logic(*det.pin2_pixel)
            geo1_str = f"Row{geo1[0]}-{geo1[1]}" if geo1 else "N/A"
            geo2_str = f"Row{geo2[0]}-{geo2[1]}" if geo2 else "N/A"

            # 视觉方法
            vis1_str, vis2_str = "N/A", "N/A"
            is_wire = det.class_name.upper() == "WIRE" or det.class_name == "Wire"
            if occupancy_map and not is_wire:
                try:
                    vloc1, vloc2 = phd.find_component_pins(
                        det, ctx.calibrator, occupancy_map, det.class_name
                    )
                    vis1_str = f"Row{vloc1[0]}-{vloc1[1]}" if vloc1 else "N/A"
                    vis2_str = f"Row{vloc2[0]}-{vloc2[1]}" if vloc2 else "N/A"
                except Exception as e:
                    vis1_str = f"ERR:{e}"

            wire_info = f" ({det.wire_color})" if hasattr(det, 'wire_color') and det.wire_color else ""
            logger.info(f"    [{i}] {det.class_name}{wire_info} conf={det.confidence:.2f}")
            logger.info(f"         几何引脚: pin1={geo1_str}, pin2={geo2_str}")
            if not is_wire:
                logger.info(f"         视觉引脚: pin1={vis1_str}, pin2={vis2_str}")
                match1 = "Match" if geo1_str == vis1_str else "DIFF"
                match2 = "Match" if geo2_str == vis2_str else "DIFF"
                logger.info(f"         比较: pin1=[{match1}], pin2=[{match2}]")

            img_results.append({
                'det_idx': i,
                'class': det.class_name,
                'conf': det.confidence,
                'geo_pin1': geo1_str, 'geo_pin2': geo2_str,
                'vis_pin1': vis1_str, 'vis_pin2': vis2_str,
            })

        # --- 运行完整流水线并保存标注图 ---
        logger.info(f"  " + "-" * 50)
        logger.info(f"  运行完整 ImageAnalyzer 流水线...")
        ctx.calibrator.reset()
        analyzer = ImageAnalyzer(ctx)
        t0 = time.time()
        result = analyzer.analyze([image], conf=0.25, imgsz=1280)
        pipeline_time = time.time() - t0

        if result:
            logger.info(f"  流水线完成: {result.component_count} 元件, "
                        f"{result.net_count} 网络 ({pipeline_time:.2f}s)")

            # 保存标注图
            out_path = str(output_dir / f"result_{stem}.jpg")
            cv2.imwrite(out_path, result.annotated_image)
            logger.info(f"  标注图已保存: {out_path}")

            # 保存占用可视化图
            try:
                occ_vis = image.copy()
                inv_m = np.linalg.inv(ctx.calibrator.matrix)
                for (hx, hy), score in occupancy_map.items():
                    pt_w = np.array([[[hx, hy]]], dtype=np.float32)
                    pt_f = cv2.perspectiveTransform(pt_w, inv_m)[0][0]
                    fx, fy = int(pt_f[0]), int(pt_f[1])
                    if score > phd.OCCUPANCY_THRESHOLD:
                        r = max(4, int(score * 10))
                        cv2.circle(occ_vis, (fx, fy), r, (0, 0, 255), 2)
                        loc = ctx.calibrator.hole_to_logic(hx, hy)
                        if loc:
                            cv2.putText(occ_vis, f"{loc[0]}-{loc[1]}",
                                        (fx+8, fy-3), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.3, (0, 0, 255), 1)
                    else:
                        cv2.circle(occ_vis, (fx, fy), 2, (0, 140, 0), -1)
                occ_path = str(output_dir / f"occupancy_{stem}.jpg")
                cv2.imwrite(occ_path, occ_vis)
                logger.info(f"  占用可视化已保存: {occ_path}")
            except Exception as e:
                logger.warning(f"  占用可视化失败: {e}")

            # 打印流水线报告 (关键行)
            for line in result.report.split('\n'):
                if 'pin' in line.lower() or 'Row' in line or 'Net' in line:
                    logger.info(f"    {line.strip()}")

        summary.append({
            'image': Path(img_path).name,
            'components': len(detections),
            'occupied_holes': len(occ_holes),
            'results': img_results,
        })

    # --- 汇总 ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("汇总")
    logger.info("=" * 70)
    for s in summary:
        logger.info(f"  {s['image']}: {s['components']} 元件, "
                    f"{s['occupied_holes']} 被占用孔洞")
        for r in s['results']:
            is_wire = r['class'].upper() == 'WIRE'
            pin_info = (f"几何[{r['geo_pin1']},{r['geo_pin2']}]" +
                        ("" if is_wire else f" 视觉[{r['vis_pin1']},{r['vis_pin2']}]"))
            logger.info(f"    {r['class']} conf={r['conf']:.2f}: {pin_info}")

    logger.info(f"\n输出保存在: {output_dir}")


if __name__ == "__main__":
    base = PROJECT_ROOT / "dataset" / "images" / "processed"

    # 选择对应用户3张附件的图片:
    # 图1: 竖版面包板, LED+电阻+小元件 → scene1
    # 图2: 横版面包板, 复杂电路(电位器+导线+电解电容+三极管+电阻) → scene4
    # 图3: 竖版面包板, IC芯片+电阻+导线 → scene3
    test_images = [
        str(base / "scene1_natural_01.jpg"),
        str(base / "scene4_natural_01.jpg"),
        str(base / "scene3_natural_01.jpg"),
    ]

    # 额外选取清晰俯视图: 每个场景的自然光第2张
    extra_images = [
        str(base / "scene1_natural_02.jpg"),
        str(base / "scene2_natural_01.jpg"),
        str(base / "scene5_natural_01.jpg"),
    ]

    # 检查文件存在性
    valid_images = []
    for p in test_images + extra_images:
        if os.path.exists(p):
            valid_images.append(p)
        else:
            logger.warning(f"图片不存在: {p}")

    if not valid_images:
        logger.error("没有找到有效图片!")
        sys.exit(1)

    logger.info(f"测试 {len(valid_images)} 张图片")
    run_batch(valid_images)
