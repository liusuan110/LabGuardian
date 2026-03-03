"""
LabGuardian YOLOv8-OBB 完整训练管线
=============================================

本脚本覆盖从数据准备到模型部署的完整流程:
  Step 1: 数据集校验与预处理
  Step 2: 数据增强策略配置
  Step 3: 模型训练 (YOLOv8-OBB)
  Step 4: 训练结果可视化与评估
  Step 5: 模型导出 (OpenVINO)

用法:
    # 完整流程
    python scripts/train_pipeline.py

    # 只训练
    python scripts/train_pipeline.py --skip-check

    # 小规模试跑 (验证环境)
    python scripts/train_pipeline.py --dry-run

    # 恢复中断的训练
    python scripts/train_pipeline.py --resume runs/obb/lab_guardian_v3/weights/last.pt

什么是 Jupyter Notebook?
------------------------
Jupyter Notebook (.ipynb) 是一种交互式文档，可以把代码、运行结果、
文字说明混在一起。你可以一个单元格(cell)一个单元格地运行代码，
立刻看到输出结果，非常适合调试和学习。

在 VS Code 中使用 Jupyter:
  1. 安装 "Jupyter" 扩展 (微软官方)
  2. Ctrl+Shift+P → "Create: New Jupyter Notebook"
  3. 把本脚本的每个 STEP 复制到一个 cell 里运行即可

或者直接运行本 .py 脚本，效果一样。
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

# ============================================================
# 路径设置
# ============================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"

# 类别定义 — 严格匹配 Roboflow data.yaml 的 ID 顺序
CLASS_NAMES = [
    "Ceramic_Capacitor",       # 0: 陶瓷电容
    "IC",                      # 1: 集成电路 (DIP 封装)
    "LED",                     # 2: 发光二极管
    "Transistor",              # 3: 三极管 (TO-92)
    "Diode",                   # 4: 二极管
    "Electrolytic_Capacitor",  # 5: 电解电容
    "Potentiometer",           # 6: 电位器
    "Resistor",                # 7: 电阻
    "Wire",                    # 8: 导线
]


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 1: 数据集校验与预处理                              ║
# ╚══════════════════════════════════════════════════════════╝

def check_dataset(dataset_dir: Path, fix: bool = False) -> dict:
    """校验数据集完整性，返回统计信息。

    检查项:
      - 每张图片是否有对应的标签文件
      - 标签格式是否正确 (OBB: class x1 y1 x2 y2 x3 y3 x4 y4)
      - 类别 ID 是否在合法范围内
      - 坐标值是否归一化到 [0, 1]
    """
    stats = {
        "total_images": 0,
        "total_labels": 0,
        "missing_labels": [],
        "orphan_labels": [],
        "format_errors": [],
        "class_distribution": Counter(),
        "bbox_count": 0,
        "needs_obb_fix": 0,   # xywh 格式需转为 OBB
    }

    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"

        if not img_dir.exists():
            continue

        image_files = set()
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_files.update(f.stem for f in img_dir.glob(ext))

        label_files = set(f.stem for f in lbl_dir.glob("*.txt")) if lbl_dir.exists() else set()

        stats["total_images"] += len(image_files)
        stats["total_labels"] += len(label_files)

        # 缺失标签
        for img in image_files - label_files:
            stats["missing_labels"].append(f"{split}/{img}")

        # 孤立标签
        for lbl in label_files - image_files:
            stats["orphan_labels"].append(f"{split}/{lbl}")

        # 逐文件检查标签格式
        for lbl_file in sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []:
            lines = lbl_file.read_text().strip().splitlines()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue

                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    stats["format_errors"].append(
                        f"{split}/{lbl_file.name}:{i+1} 解析失败")
                    continue

                # 类别检查
                if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                    stats["format_errors"].append(
                        f"{split}/{lbl_file.name}:{i+1} 类别 {cls_id} 超出范围 [0,{len(CLASS_NAMES)-1}]")

                # 格式检查
                if len(coords) == 4:
                    # xywh 格式 → 需要转为 OBB
                    stats["needs_obb_fix"] += 1
                elif len(coords) == 8:
                    # OBB 四点格式 → 正确
                    pass
                else:
                    stats["format_errors"].append(
                        f"{split}/{lbl_file.name}:{i+1} "
                        f"坐标数量 {len(coords)} 不是 4(xywh) 也不是 8(OBB)")
                    continue

                # 坐标范围检查
                for v in coords:
                    if v < -0.01 or v > 1.01:
                        stats["format_errors"].append(
                            f"{split}/{lbl_file.name}:{i+1} 坐标 {v:.4f} 超出 [0,1]")
                        break

                stats["class_distribution"][CLASS_NAMES[cls_id]] += 1
                stats["bbox_count"] += 1

    # 打印报告
    print("\n" + "=" * 60)
    print("📋 数据集校验报告")
    print("=" * 60)
    print(f"  图片: {stats['total_images']}  标签: {stats['total_labels']}")
    print(f"  标注框: {stats['bbox_count']}")

    if stats["class_distribution"]:
        print("\n  类别分布:")
        for cls, cnt in stats["class_distribution"].most_common():
            bar = "█" * min(cnt, 50)
            print(f"    {cls:15s} {cnt:5d}  {bar}")

    if stats["missing_labels"]:
        print(f"\n  ⚠️ 缺失标签: {len(stats['missing_labels'])} 张图片无标签")
        for m in stats["missing_labels"][:5]:
            print(f"    - {m}")

    if stats["needs_obb_fix"] > 0:
        print(f"\n  ⚠️ 有 {stats['needs_obb_fix']} 个标注是 xywh 格式，需要转为 OBB 四点格式")
        if fix:
            print("  → 正在自动修复...")
            _fix_xywh_to_obb(dataset_dir)

    if stats["format_errors"]:
        print(f"\n  ❌ 格式错误: {len(stats['format_errors'])}")
        for e in stats["format_errors"][:10]:
            print(f"    - {e}")

    if not stats["format_errors"] and not stats["missing_labels"]:
        print("\n  ✅ 数据集校验通过！")

    print("=" * 60)
    return stats


def _fix_xywh_to_obb(dataset_dir: Path):
    """将 xywh 格式自动转换为 OBB 四点格式。"""
    fixed = 0
    for split in ["train", "val", "test"]:
        lbl_dir = dataset_dir / split / "labels"
        if not lbl_dir.exists():
            continue

        for lbl_file in lbl_dir.glob("*.txt"):
            lines = lbl_file.read_text().strip().splitlines()
            new_lines = []
            modified = False

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    x, y, w, h = [float(v) for v in parts[1:]]
                    # xywh → 四角坐标
                    x1, y1 = x - w/2, y - h/2
                    x2, y2 = x + w/2, y - h/2
                    x3, y3 = x + w/2, y + h/2
                    x4, y4 = x - w/2, y + h/2
                    new_lines.append(
                        f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
                        f"{x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}")
                    modified = True
                else:
                    new_lines.append(line.strip())

            if modified:
                lbl_file.write_text("\n".join(new_lines) + "\n")
                fixed += 1

    print(f"  → 已修复 {fixed} 个文件")


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 2: 生成 data.yaml                                 ║
# ╚══════════════════════════════════════════════════════════╝

def generate_data_yaml(dataset_dir: Path) -> Path:
    """根据实际目录结构生成/更新 data.yaml。

    data.yaml 是 YOLO 训练的数据集描述文件，告诉模型:
      - 训练集/验证集的图片在哪
      - 一共有多少个类别
      - 每个类别叫什么名字
    """
    yaml_path = dataset_dir / "data.yaml"

    # 自动检测可用的 split
    splits = {}
    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / split / "images"
        if img_dir.exists() and any(img_dir.iterdir()):
            splits[split] = f"{split}/images"

    if "train" not in splits:
        raise FileNotFoundError(
            f"未找到训练集! 请将图片放到 {dataset_dir}/train/images/")

    # 如果没有单独的 val，用 train 代替（不推荐，但可先跑通）
    if "val" not in splits:
        print("  ⚠️ 未找到验证集 (val/)，将使用 train 代替。")
        print("  → 建议: 拍照时留出 10-20% 的图片放到 val/images/ 和 val/labels/")
        splits["val"] = splits["train"]

    content = f"""# LabGuardian 数据集配置
# 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}

path: {dataset_dir.as_posix()}
train: {splits['train']}
val: {splits['val']}
"""
    if "test" in splits:
        content += f"test: {splits['test']}\n"

    content += f"""
nc: {len(CLASS_NAMES)}
names:
"""
    for name in CLASS_NAMES:
        content += f"- {name}\n"

    yaml_path.write_text(content, encoding="utf-8")
    print(f"  ✅ data.yaml 已生成: {yaml_path}")
    return yaml_path


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 3: 训练模型                                       ║
# ╚══════════════════════════════════════════════════════════╝

def train_model(
    data_yaml: Path,
    epochs: int = 200,
    imgsz: int = 960,
    batch: int = 8,
    device: str = "0",
    model_size: str = "s",
    resume: str = None,
    dry_run: bool = False,
    project_name: str = "lab_guardian_v3",
):
    """启动 YOLOv8-OBB 训练。

    关键概念解释
    ------------
    epochs (轮次):
        整个数据集被完整训练一遍叫一个 epoch。
        200 epochs = 所有图片被看 200 遍。
        一般 100-300 即可，配合 early stopping。

    imgsz (输入分辨率):
        训练时图片被缩放到的尺寸。越大越精确但越慢越耗显存。
        - 640: 速度快，适合调试
        - 960: 推荐，面包板元件较小需要高分辨率
        - 1280: 精度最高，需要大显存

    batch (批大小):
        每次送入 GPU 的图片数量。受显存限制:
        - 4: 4GB 显存
        - 8: 8GB 显存 (推荐)
        - 16: 16GB+ 显存

    model_size (模型尺寸):
        - "n" (nano): 最快，精度最低，适合调试
        - "s" (small): 速度/精度平衡 (推荐)
        - "m" (medium): 更精确但更慢
        - "l" (large): 最精确，需要大显存

    patience (早停):
        如果验证指标连续 N 个 epoch 没有提升，自动停止训练。
        防止过拟合（模型记住了训练图片但不能泛化到新图片）。

    OBB vs 普通检测:
        OBB (Oriented Bounding Box) = 旋转边界框
        普通检测只能画水平矩形，OBB 可以画任意角度的矩形。
        面包板上的元件可能斜着放，所以用 OBB 更准确，
        而且 OBB 的短边中点天然就是元件引脚位置！

    数据增强 (Augmentation):
        训练时自动对图片做随机变换（翻转、旋转、色彩调整等），
        让模型见到更多变化，提升泛化能力。YOLO 内置了大量增强策略。
    """
    from ultralytics import YOLO

    # 选择基础权重
    weights = f"yolov8{model_size}-obb.pt"
    if resume:
        weights = resume
        print(f"  🔄 恢复训练: {resume}")
    else:
        print(f"  📦 基础权重: {weights}")

    print(f"  📁 数据集:   {data_yaml}")
    print(f"  🔢 Epochs:   {epochs}")
    print(f"  📐 ImgSz:    {imgsz}")
    print(f"  📊 Batch:    {batch}")
    print(f"  💻 Device:   {device}")

    if dry_run:
        print("\n  🧪 Dry-run 模式: 只训练 3 个 epoch 验证环境")
        epochs = 3
        batch = 2

    model = YOLO(weights)

    # ---- 开始训练 ----
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name=project_name,
        exist_ok=True,          # 覆盖同名实验
        patience=50,            # 50 epoch 无提升则早停
        save_period=20,         # 每 20 epoch 保存一次 checkpoint

        # ---- 数据增强参数 ----
        # 这些是面包板场景优化过的值
        degrees=15.0,           # 随机旋转 ±15° (元件可能稍有倾斜)
        translate=0.1,          # 随机平移 10%
        scale=0.3,              # 随机缩放 ±30% (模拟不同拍摄距离)
        shear=2.0,              # 微小剪切变换
        perspective=0.0005,     # 轻微透视变换 (模拟不同拍摄角度)
        flipud=0.0,             # 禁止上下翻转 (面包板有上下方向)
        fliplr=0.5,             # 50% 概率左右翻转
        mosaic=1.0,             # Mosaic 增强 (4 图拼接)
        mixup=0.1,              # 10% MixUp (两图混合)
        hsv_h=0.015,            # 色调微调 (不同灯光)
        hsv_s=0.5,              # 饱和度变化
        hsv_v=0.4,              # 亮度变化 (模拟不同光照)
        erasing=0.2,            # 20% 随机擦除 (模拟遮挡)

        # ---- 优化器参数 ----
        optimizer="auto",       # 自动选择 (SGD/AdamW)
        lr0=0.01,               # 初始学习率
        lrf=0.01,               # 最终学习率衰减比
        warmup_epochs=5,        # 预热轮次

        # ---- 其他 ----
        workers=4,              # 数据加载线程
        seed=42,                # 随机种子 (保证可复现)
        verbose=True,
    )

    # 训练结果路径
    run_dir = Path(f"runs/obb/{project_name}")
    best_pt = run_dir / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print(f"  最佳模型: {best_pt}")
    print(f"  训练日志: {run_dir}")
    print(f"  下一步:   python scripts/train_pipeline.py --evaluate {best_pt}")
    print("=" * 60)

    return best_pt


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 4: 评估模型                                       ║
# ╚══════════════════════════════════════════════════════════╝

def evaluate_model(model_path: Path, data_yaml: Path, imgsz: int = 960):
    """在验证集上评估模型，输出关键指标。

    关键指标解释
    ------------
    mAP50 (mean Average Precision @ IoU=0.5):
        所有类别在 IoU≥50% 时的平均精度。
        通俗说: 模型检测到的框和真实框重叠超过一半就算对。
        - > 0.7: 可用
        - > 0.85: 良好
        - > 0.95: 优秀

    mAP50-95:
        在 IoU 从 0.5 到 0.95 (步长 0.05) 范围内取平均。
        比 mAP50 更严格，是目标检测的标准指标。

    Precision (精确率):
        模型说"这是电阻"的时候，真的是电阻的比例。
        精确率低 = 误报多 (把不是元件的东西检成元件)。

    Recall (召回率):
        所有真实电阻中，被模型找到的比例。
        召回率低 = 漏检多 (有元件但没检测到)。

    每类 AP:
        各类别单独的精度。可以看出哪类检测好、哪类需要更多数据。
    """
    from ultralytics import YOLO

    print(f"\n📊 评估模型: {model_path}")
    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=8,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("📊 评估结果摘要")
    print("=" * 60)
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    # 每类精度
    if hasattr(metrics.box, 'ap_class_index'):
        print("\n  每类 AP50:")
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = CLASS_NAMES[int(cls_idx)] if int(cls_idx) < len(CLASS_NAMES) else f"class_{cls_idx}"
            ap = metrics.box.ap50[i]
            bar = "█" * int(ap * 30)
            print(f"    {cls_name:15s} {ap:.3f}  {bar}")

    print("=" * 60)
    return metrics


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 5: 导出模型 (OpenVINO)                            ║
# ╚══════════════════════════════════════════════════════════╝

def export_model(model_path: Path, format: str = "openvino", imgsz: int = 960):
    """将训练好的模型导出为部署格式。

    OpenVINO 导出说明
    -----------------
    比赛平台是 Intel DK-2500 (Core Ultra 5 225U)，自带 NPU。
    OpenVINO 可以让模型在 Intel CPU/GPU/NPU 上高效推理。

    导出后会生成一个目录，包含 .xml 和 .bin 文件，
    放到 models/ 目录下即可被 LabGuardian 自动加载。
    """
    from ultralytics import YOLO

    print(f"\n📦 导出模型: {model_path} → {format}")
    model = YOLO(str(model_path))

    export_path = model.export(
        format=format,
        imgsz=imgsz,
        half=False,     # INT8/FP16 量化在 OpenVINO 端做更灵活
    )

    print(f"  ✅ 导出完成: {export_path}")

    # 复制到 models/ 目录
    dst = MODELS_DIR / f"lab_guardian_obb_{format}"
    if Path(export_path).is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(export_path, dst)
        print(f"  📁 已复制到: {dst}")

    return export_path


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 6: 快速预测测试                                   ║
# ╚══════════════════════════════════════════════════════════╝

def quick_predict(model_path: Path, image_path: Path, imgsz: int = 960):
    """用训练好的模型对单张图片做预测，可视化结果。"""
    from ultralytics import YOLO
    import cv2

    print(f"\n🔍 预测测试: {image_path}")
    model = YOLO(str(model_path))

    results = model(
        str(image_path),
        imgsz=imgsz,
        conf=0.25,
        save=True,
        save_txt=True,
    )

    result = results[0]
    n_det = len(result.obb) if result.obb is not None else len(result.boxes) if result.boxes is not None else 0
    print(f"  检测到 {n_det} 个元件")
    print(f"  结果保存在 runs/obb/predict/")

    return results


# ╔══════════════════════════════════════════════════════════╗
# ║  主函数                                                  ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="LabGuardian YOLOv8-OBB 训练管线",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset", type=str,
                        default=str(DATASET_DIR),
                        help="数据集目录路径")
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮次 (默认 200)")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="输入分辨率 (默认 960)")
    parser.add_argument("--batch", type=int, default=8,
                        help="批大小 (默认 8，显存不够就改小)")
    parser.add_argument("--device", type=str, default="0",
                        help="设备: '0'=GPU, 'cpu'=CPU")
    parser.add_argument("--model-size", type=str, default="s",
                        choices=["n", "s", "m", "l"],
                        help="模型尺寸: n/s/m/l (默认 s)")
    parser.add_argument("--name", type=str, default="lab_guardian_v3",
                        help="实验名称")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的 checkpoint 路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="快速试跑 (3 epoch)")
    parser.add_argument("--skip-check", action="store_true",
                        help="跳过数据集校验")
    parser.add_argument("--fix-labels", action="store_true",
                        help="自动修复 xywh→OBB 格式")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="评估指定模型 (跳过训练)")
    parser.add_argument("--export", type=str, default=None,
                        help="导出指定模型为 OpenVINO")
    parser.add_argument("--predict", type=str, default=None,
                        help="用模型预测一张图片")
    parser.add_argument("--predict-model", type=str, default=None,
                        help="预测用的模型路径")

    args = parser.parse_args()
    dataset_dir = Path(args.dataset)

    print("=" * 60)
    print("🔬 LabGuardian YOLOv8-OBB 训练管线")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ---- 仅评估 ----
    if args.evaluate:
        yaml_path = dataset_dir / "data.yaml"
        if not yaml_path.exists():
            yaml_path = generate_data_yaml(dataset_dir)
        evaluate_model(Path(args.evaluate), yaml_path, args.imgsz)
        return

    # ---- 仅导出 ----
    if args.export:
        export_model(Path(args.export), "openvino", args.imgsz)
        return

    # ---- 仅预测 ----
    if args.predict:
        model = args.predict_model or str(MODELS_DIR / "yolov8n-obb.pt")
        quick_predict(Path(model), Path(args.predict), args.imgsz)
        return

    # ---- 完整流程 ----

    # Step 1: 数据集校验
    if not args.skip_check:
        print("\n📋 Step 1: 数据集校验")
        stats = check_dataset(dataset_dir, fix=args.fix_labels)
        if stats["total_images"] == 0:
            print("\n❌ 数据集为空！请先准备数据：")
            print(f"   图片放到: {dataset_dir}/train/images/")
            print(f"   标签放到: {dataset_dir}/train/labels/")
            print("   标签格式: class_id x1 y1 x2 y2 x3 y3 x4 y4")
            print("   (每行一个标注，坐标归一化到 0-1)")
            return

    # Step 2: 生成 data.yaml
    print("\n📝 Step 2: 生成 data.yaml")
    yaml_path = generate_data_yaml(dataset_dir)

    # Step 3: 训练
    print("\n🚀 Step 3: 开始训练")
    best_pt = train_model(
        data_yaml=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        model_size=args.model_size,
        resume=args.resume,
        dry_run=args.dry_run,
        project_name=args.name,
    )

    # Step 4: 评估
    if best_pt and best_pt.exists():
        print("\n📊 Step 4: 评估模型")
        evaluate_model(best_pt, yaml_path, args.imgsz)

        # Step 5: 导出
        print("\n📦 Step 5: 导出 OpenVINO 格式")
        try:
            export_model(best_pt, "openvino", args.imgsz)
        except Exception as e:
            print(f"  ⚠️ 导出失败: {e}")
            print("  → 可以稍后手动执行: python scripts/train_pipeline.py --export <model_path>")


if __name__ == "__main__":
    main()
