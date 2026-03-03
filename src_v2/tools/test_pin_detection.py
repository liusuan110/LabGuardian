"""
测试脚本: 对比引脚检测方法
=========================
方法 A: OBB 几何延伸 + 最近孔洞候选 (旧方法)
方法 B: YOLO pin 直接检测 + 孔洞映射 (新模型)
方法 C: 视觉孔洞占用分析 (PinHoleDetector)

对同一张图片运行三种方法, 可视化对比结果.
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path

# 确保 src_v2 在路径中
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from ultralytics import YOLO
from config import vision as vision_cfg, circuit as circuit_cfg, find_best_yolo_model


def load_image(path_str: str) -> np.ndarray:
    """支持中文路径的图片加载"""
    buf = np.fromfile(path_str, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法加载图片: {path_str}")
    return img


def save_image(path_str: str, img: np.ndarray):
    """支持中文路径的图片保存"""
    ext = os.path.splitext(path_str)[1]
    _, buf = cv2.imencode(ext, img)
    buf.tofile(path_str)


def run_test(image_path: str):
    """对一张图片运行完整的对比测试"""
    
    print("=" * 70)
    print("  LabGuardian 引脚检测对比测试")
    print("=" * 70)
    
    # ---- 加载图片 ----
    print(f"\n[1] 加载图片: {Path(image_path).name}")
    img = load_image(image_path)
    h, w = img.shape[:2]
    print(f"    尺寸: {w}x{h}")
    
    # ---- 加载 OBB 模型 (元件检测) ----
    print("\n[2] 加载 OBB 元件检测模型...")
    obb_model_path = find_best_yolo_model()
    obb_model = YOLO(str(obb_model_path))
    print(f"    模型: {obb_model_path.name}")
    
    # ---- 加载引脚检测模型 (新) ----
    print("\n[3] 加载 YOLO 引脚检测模型 (新训练)...")
    pin_model_path = Path(SRC_DIR).parent / "runs" / "detect" / "runs" / "pin_detect" / "lab_guardian_pins_v2" / "weights" / "best.pt"
    if not pin_model_path.exists():
        pin_model_path = Path(SRC_DIR).parent / "runs" / "detect" / "runs" / "pin_detect" / "lab_guardian_pins_v1" / "weights" / "best.pt"
    pin_model = YOLO(str(pin_model_path))
    print(f"    模型: {pin_model_path.name} ({pin_model_path.parent.parent.name})")
    
    # ---- 校准面包板 ----
    print("\n[4] 校准面包板...")
    from vision.calibrator import BreadboardCalibrator
    calibrator = BreadboardCalibrator()
    cal_ok = calibrator.auto_calibrate(img)
    if cal_ok:
        hole_count = len(calibrator.hole_centers)
        print(f"    校准成功: {hole_count} 个孔洞检测到")
        if calibrator.row_centers is not None:
            print(f"    行数: {len(calibrator.row_centers)}, 列数: {len(calibrator.col_centers) if calibrator.col_centers is not None else '?'}")
    else:
        print("    校准失败!")
        return
    
    # ---- 方法 A: OBB 元件检测 + 几何引脚 ----
    print("\n[5] 方法 A: OBB 检测 + 几何引脚延伸...")
    t0 = time.time()
    obb_results = obb_model(img, verbose=False, conf=0.25, imgsz=960)
    t_obb = time.time() - t0
    
    from vision.detector import ComponentDetector
    detector = ComponentDetector(obb_model_path)
    detector.model = obb_model
    detector.is_obb_model = (getattr(obb_model, 'task', 'detect') == 'obb')
    
    obb_dets = detector.detect(img, conf=0.25)
    print(f"    检测到 {len(obb_dets)} 个元件 ({t_obb:.2f}s)")
    
    geom_pins = []
    for det in obb_dets:
        if det.pin1_pixel and det.pin2_pixel:
            loc1 = calibrator.frame_pixel_to_logic(*det.pin1_pixel)
            loc2 = calibrator.frame_pixel_to_logic(*det.pin2_pixel)
            geom_pins.append({
                'component': det.class_name,
                'conf': det.confidence,
                'pin1_pixel': det.pin1_pixel,
                'pin2_pixel': det.pin2_pixel,
                'pin1_logic': loc1,
                'pin2_logic': loc2,
                'method': 'A_geometric',
            })
            print(f"    [{det.class_name} {det.confidence:.2f}] "
                  f"Pin1={loc1} Pin2={loc2}")
    
    # ---- 方法 B: YOLO 引脚直接检测 ----
    print(f"\n[6] 方法 B: YOLO 引脚直接检测 (新模型)...")
    t0 = time.time()
    pin_results = pin_model(img, verbose=False, conf=0.15, imgsz=960)
    t_pin = time.time() - t0
    
    pin_dets_raw = []
    pin_result = pin_results[0]
    if pin_result.boxes is not None:
        for box in pin_result.boxes:
            cls_id = int(box.cls[0])
            cls_name = pin_result.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # 引脚检测: bbox 底部中心 = 插入点 (引脚从上方插入)
            # 或者使用中心点
            insert_x = cx
            insert_y = cy  # 使用中心, 因为标注的就是引脚插入孔洞区域
            
            loc = calibrator.frame_pixel_to_logic(insert_x, insert_y)
            pin_dets_raw.append({
                'class': cls_name,
                'conf': conf,
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy),
                'insert_point': (insert_x, insert_y),
                'logic': loc,
            })
    
    print(f"    检测到 {len(pin_dets_raw)} 个引脚 ({t_pin:.2f}s)")
    for pd in sorted(pin_dets_raw, key=lambda x: x['conf'], reverse=True):
        print(f"    [{pd['class']} {pd['conf']:.2f}] "
              f"center=({pd['center'][0]:.0f},{pd['center'][1]:.0f}) "
              f"-> {pd['logic']}")
    
    # ---- 方法 C: 视觉孔洞占用检测 (PinHoleDetector) ----
    print(f"\n[7] 方法 C: 视觉孔洞占用检测...")
    from vision.pin_hole_detector import PinHoleDetector
    phd = PinHoleDetector()
    t0 = time.time()
    occupancy_map = phd.detect_occupied_holes(img, calibrator)
    t_occ = time.time() - t0
    
    occupied = [(pos, score) for pos, score in occupancy_map.items() 
                if score > phd.OCCUPANCY_THRESHOLD]
    print(f"    分析 {len(occupancy_map)} 孔洞, "
          f"{len(occupied)} 个疑似被占用 ({t_occ:.2f}s)")
    
    for (hx, hy), score in sorted(occupied, key=lambda x: -x[1])[:15]:
        loc = calibrator.hole_to_logic(hx, hy)
        print(f"    [{score:.3f}] -> {loc}")
    
    # ---- 方法 B 精确映射: 将引脚检测关联到元件 ----
    print(f"\n[8] 方法 B 精确关联: 将引脚分配给元件...")
    pin_to_comp = associate_pins_to_components(pin_dets_raw, obb_dets, calibrator)
    
    for comp_name, pins in pin_to_comp.items():
        if len(pins) >= 2:
            print(f"    [{comp_name}]")
            for p in pins:
                print(f"      {p['class']} (conf={p['conf']:.2f}) -> {p['logic']}")
    
    # ---- 生成对比标注图 ----
    print(f"\n[9] 生成标注图...")
    vis = create_comparison_visualization(
        img, obb_dets, pin_dets_raw, occupied, calibrator, geom_pins
    )
    
    out_dir = Path(SRC_DIR).parent / "test_results"
    out_dir.mkdir(exist_ok=True)
    
    basename = Path(image_path).stem
    out_path = str(out_dir / f"{basename}_pin_comparison.jpg")
    save_image(out_path, vis)
    print(f"    保存到: {out_path}")
    
    # ---- 生成仅引脚检测标注图 ----
    vis_pins = create_pin_only_visualization(img, pin_dets_raw, calibrator)
    out_path2 = str(out_dir / f"{basename}_pins_only.jpg")
    save_image(out_path2, vis_pins)
    print(f"    仅引脚: {out_path2}")
    
    # ---- 总结 ----
    print(f"\n{'='*70}")
    print(f"  总结对比")
    print(f"{'='*70}")
    print(f"  方法 A (几何延伸):   {len(geom_pins)} 组引脚对, {t_obb:.2f}s")
    print(f"  方法 B (YOLO引脚):   {len(pin_dets_raw)} 个引脚, {t_pin:.2f}s")
    print(f"  方法 C (视觉占用):   {len(occupied)} 个被占用, {t_occ:.2f}s")
    print(f"{'='*70}")


def associate_pins_to_components(pin_dets, obb_dets, calibrator):
    """将引脚检测结果关联到最近的元件"""
    result = {}
    
    for det in obb_dets:
        comp_key = f"{det.class_name}_{det.confidence:.2f}"
        x1, y1, x2, y2 = det.bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        
        # 预计引脚搜索区域: 元件 bbox 外扩 50%
        expand = max(x2-x1, y2-y1) * 0.5
        search_x1 = x1 - expand
        search_y1 = y1 - expand
        search_x2 = x2 + expand
        search_y2 = y2 + expand
        
        nearby_pins = []
        for pd in pin_dets:
            px, py = pd['center']
            if search_x1 <= px <= search_x2 and search_y1 <= py <= search_y2:
                dist = np.sqrt((px-cx)**2 + (py-cy)**2)
                nearby_pins.append({**pd, 'dist_to_comp': dist})
        
        # 按置信度排序, 取前 N 个
        nearby_pins.sort(key=lambda x: -x['conf'])
        result[comp_key] = nearby_pins[:4]
    
    return result


def create_comparison_visualization(img, obb_dets, pin_dets, occupied, calibrator, geom_pins):
    """生成对比标注图"""
    vis = img.copy()
    
    # 绘制 OBB 检测框 (灰色)
    for det in obb_dets:
        if det.is_obb and det.obb_corners is not None:
            pts = det.obb_corners.astype(int)
            cv2.polylines(vis, [pts], True, (180, 180, 180), 2)
        else:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 180, 180), 2)
        
        label = f"{det.class_name} {det.confidence:.2f}"
        x1, y1 = det.bbox[:2]
        cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 方法 A: 几何引脚 (蓝色)
    for gp in geom_pins:
        p1 = tuple(int(v) for v in gp['pin1_pixel'])
        p2 = tuple(int(v) for v in gp['pin2_pixel'])
        cv2.circle(vis, p1, 8, (255, 0, 0), 2)  # 蓝色空心圆
        cv2.circle(vis, p2, 8, (255, 0, 0), 2)
        if gp['pin1_logic']:
            text = f"A:{gp['pin1_logic'][0]}-{gp['pin1_logic'][1]}"
            cv2.putText(vis, text, (p1[0]+10, p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
        if gp['pin2_logic']:
            text = f"A:{gp['pin2_logic'][0]}-{gp['pin2_logic'][1]}"
            cv2.putText(vis, text, (p2[0]+10, p2[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
    
    # 方法 B: YOLO 引脚检测 (绿色)
    for pd in pin_dets:
        x1, y1, x2, y2 = pd['bbox']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = int(pd['center'][0]), int(pd['center'][1])
        cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)  # 绿色实心
        
        label = f"B:{pd['class']} {pd['conf']:.2f}"
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
        if pd['logic'] and pd['logic'][0] != 'Groove':
            loc_text = f"{pd['logic'][0]}-{pd['logic'][1]}"
            cv2.putText(vis, loc_text, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    
    # 方法 C: 视觉占用 (红色小点)
    if occupied and calibrator.matrix is not None:
        try:
            inv_m = np.linalg.inv(calibrator.matrix)
            for (hx, hy), score in occupied:
                pts = np.array([[[hx, hy]]], dtype=np.float32)
                frame_pts = cv2.perspectiveTransform(pts, inv_m)[0][0]
                fx, fy = int(frame_pts[0]), int(frame_pts[1])
                radius = max(3, int(score * 6))
                cv2.circle(vis, (fx, fy), radius, (0, 0, 255), -1)  # 红色
        except Exception:
            pass
    
    # 图例
    legend_y = 30
    cv2.putText(vis, "A: Geometric (blue)", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(vis, "B: YOLO Pin (green)", (20, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, "C: Visual Occ (red)", (20, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return vis


def create_pin_only_visualization(img, pin_dets, calibrator):
    """生成仅引脚检测的标注图"""
    vis = img.copy()
    
    for pd in pin_dets:
        x1, y1, x2, y2 = pd['bbox']
        # 颜色根据类别
        color_map = {
            'pin1': (0, 255, 0), 'pin2': (0, 200, 255), 'pin3': (255, 0, 0),
            'pin4': (255, 0, 255), 'pin5': (0, 255, 255), 'pin6': (255, 255, 0),
            'pin7': (128, 0, 255), 'pin8': (255, 128, 0),
        }
        color = color_map.get(pd['class'], (128, 128, 128))
        
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cx, cy = int(pd['center'][0]), int(pd['center'][1])
        cv2.circle(vis, (cx, cy), 4, color, -1)
        
        label = f"{pd['class']} {pd['conf']:.2f}"
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if pd['logic'] and pd['logic'][0] != 'Groove':
            loc_text = f"Row{pd['logic'][0]}-{pd['logic'][1]}"
            cv2.putText(vis, loc_text, (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    cv2.putText(vis, f"YOLO Pin Detections: {len(pin_dets)}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return vis


if __name__ == "__main__":
    # 默认测试图片: 原始采集第一张
    default_imgs = [
        r"D:\desktop\inter\LabGuardian\dataset\images\采集\采集\1-自然光-01.jpg",
        r"D:\desktop\inter\LabGuardian\dataset\images\采集\采集\1-台灯-01.jpg",
    ]
    
    # 也测试 roboflow 测试集中的图片
    test_dir = Path(r"D:\desktop\inter\LabGuardian\dataset\images\---------.v1i.yolov8\test\images")
    if test_dir.exists():
        test_imgs = list(test_dir.glob("*.jpg"))[:2]
        default_imgs.extend([str(p) for p in test_imgs])
    
    for img_path in default_imgs:
        if os.path.exists(img_path):
            print(f"\n\n{'#'*70}")
            print(f"# 测试图片: {Path(img_path).name}")
            print(f"{'#'*70}\n")
            try:
                run_test(img_path)
            except Exception as e:
                print(f"  错误: {e}")
                import traceback
                traceback.print_exc()
            print()
