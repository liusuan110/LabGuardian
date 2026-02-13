"""
YOLOv8-OBB 训练脚本
用法: python -m tools.train_obb
"""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_DIR


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-OBB model")
    parser.add_argument("--data", type=str,
                        default=str(DATASET_DIR / "data.yaml"),
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0",
                        help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--name", type=str, default="lab_guardian_v2")
    parser.add_argument("--weights", type=str, default="yolov8s-obb.pt",
                        help="Pretrained weights (nano/small/medium)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (0=disabled)")
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"[Train] Data:    {args.data}")
    print(f"[Train] Weights: {args.weights}")
    print(f"[Train] Epochs:  {args.epochs}")
    print(f"[Train] ImgSz:   {args.imgsz}")

    model = YOLO(args.weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        exist_ok=True,
        device=args.device,
        patience=args.patience,
    )

    print("[Train] Done!")
    print(f"[Train] Best model: runs/obb/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
