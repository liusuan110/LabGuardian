"""
OBB 标注辅助工具
用法: python -m tools.annotate_helper
"""

import cv2
import os
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_DIR, COMPONENT_CLASSES, CLASS_COLORS

# 数据集目录
TRAIN_DIR = DATASET_DIR / "train"
IMG_DIR = TRAIN_DIR / "images"
LABEL_DIR = TRAIN_DIR / "labels"

LABEL_DIR.mkdir(parents=True, exist_ok=True)


class OBBAnnotator:
    """交互式 OBB 标注工具"""

    def __init__(self):
        self.image_files = [
            f for f in os.listdir(IMG_DIR)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        if not self.image_files:
            print(f"Error: No images found in {IMG_DIR}")
            return

        self.current_idx = 0
        self.points = []
        self.current_class = 5  # 默认 Wire
        self.labels = []
        self.load_image()

    def load_image(self):
        self.img_name = self.image_files[self.current_idx]
        img_path = str(IMG_DIR / self.img_name)
        self.original_img = cv2.imread(img_path)
        self.display_img = self.original_img.copy()
        print(f"\n--- Annotating: {self.img_name} ---")
        print("Controls: [1-6] class, [Click] 4 corners, [S] save, [C] clear, [N] next, [Q] quit")
        self.redraw()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            if len(self.points) == 4:
                self.save_current_shape()
                self.points = []
            self.redraw()

    def save_current_shape(self):
        h, w = self.original_img.shape[:2]
        pts_norm = []
        for p in self.points:
            pts_norm.extend([p[0] / w, p[1] / h])
        label_line = f"{self.current_class} " + " ".join(f"{v:.6f}" for v in pts_norm)
        self.labels.append(label_line)
        print(f"  Added: {COMPONENT_CLASSES[self.current_class]} ({label_line[:30]}...)")

    def redraw(self):
        self.display_img = self.original_img.copy()
        for p in self.points:
            cv2.circle(self.display_img, p, 3, (0, 0, 255), -1)
        cv2.imshow("OBB Annotator", self.display_img)

    def save_labels(self):
        label_path = LABEL_DIR / (Path(self.img_name).stem + ".txt")
        with open(label_path, 'w') as f:
            f.write("\n".join(self.labels) + "\n")
        print(f"  Saved {len(self.labels)} labels to {label_path}")

    def run(self):
        cv2.namedWindow("OBB Annotator")
        cv2.setMouseCallback("OBB Annotator", self.mouse_callback)

        while True:
            cv2.imshow("OBB Annotator", self.display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif ord('1') <= key <= ord('6'):
                self.current_class = key - ord('1')
                print(f"  Class: {COMPONENT_CLASSES[self.current_class]}")
            elif key == ord('s'):
                self.save_labels()
            elif key == ord('c'):
                self.labels = []
                self.points = []
                print("  Cleared.")
                self.redraw()
            elif key == ord('n'):
                self.current_idx = (self.current_idx + 1) % len(self.image_files)
                self.labels = []
                self.points = []
                self.load_image()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    annotator = OBBAnnotator()
    if annotator.image_files:
        annotator.run()
