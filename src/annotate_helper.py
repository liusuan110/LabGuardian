import cv2
import os
import numpy as np

# 配置区域
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'OneShot_Demo_Dataset', 'train')
IMG_DIR = os.path.join(DATASET_DIR, 'images')
LABEL_DIR = os.path.join(DATASET_DIR, 'labels')
CLASSES = ["CAPACITOR", "DIODE", "LED", "RESISTOR", "Push_Button", "Wire"]
COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]

os.makedirs(LABEL_DIR, exist_ok=True)

class OBBAnnotator:
    def __init__(self):
        self.image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not self.image_files:
            print(f"错误：在 {IMG_DIR} 没有找到图片！请先将演示图片放入该文件夹。")
            return
            
        self.current_idx = 0
        self.points = []
        self.current_class = 5 # 默认为 Wire (导线)
        self.labels = [] # List of "cls x1 y1 x2 y2 x3 y3 x4 y4"
        self.dragging = False
        
        self.load_image()

    def load_image(self):
        self.img_name = self.image_files[self.current_idx]
        img_path = os.path.join(IMG_DIR, self.img_name)
        self.original_img = cv2.imread(img_path)
        self.display_img = self.original_img.copy()
        print(f"\n--- 正在标注: {self.img_name} ---")
        print("操作指南:")
        print("  [1-6]: 选择类别 (1:CAP, 2:DIODE, 3:LED, 4:RES, 5:BTN, 6:Wire)")
        print("  [鼠标左键]: 依次点击元件的 4 个角点")
        print("  [S]: 保存当前标注")
        print("  [C]: 清除所有标注")
        print("  [N]: 下一张 / [Q]: 退出")
        self.redraw()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            if len(self.points) == 4:
                # 完成一个 OBB
                self.save_current_shape()
                self.points = []
            self.redraw()
        elif event == cv2.EVENT_MOUSEMOVE:
             self.temp_pos = (x,y)
             if len(self.points) > 0:
                 self.redraw()

    def save_current_shape(self):
        # 归一化坐标
        h, w = self.original_img.shape[:2]
        pts_norm = []
        for p in self.points:
            pts_norm.append(f"{p[0]/w:.6f} {p[1]/h:.6f}")
        
        line = f"{self.current_class} {' '.join(pts_norm)}"
        self.labels.append((self.current_class, self.points, line))
        print(f"已添加: {CLASSES[self.current_class]}")

    def redraw(self):
        self.display_img = self.original_img.copy()
        
        # 画已保存的
        for cls, pts, _ in self.labels:
            pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.display_img, [pts_np], True, COLORS[cls], 2)
            # 标类别
            cv2.putText(self.display_img, CLASSES[cls], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cls], 2)

        # 画当前正在点的
        for p in self.points:
            cv2.circle(self.display_img, p, 3, (255, 255, 255), -1)
        
        if len(self.points) > 0 and hasattr(self, 'temp_pos'):
             # 画辅助线
             cv2.line(self.display_img, self.points[-1], self.temp_pos, (200,200,200), 1)

        # 状态栏
        info = f"Class: {CLASSES[self.current_class]}  |  Count: {len(self.labels)}"
        cv2.putText(self.display_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("OBB Annotator (Custom)", self.display_img)

    def save_txt(self):
        txt_name = os.path.splitext(self.img_name)[0] + ".txt"
        txt_path = os.path.join(LABEL_DIR, txt_name)
        with open(txt_path, "w") as f:
            for _, _, line in self.labels:
                f.write(line + "\n")
        print(f"✅ 已保存标签文件: {txt_path}")

    def run(self):
        cv2.namedWindow("OBB Annotator (Custom)")
        cv2.setMouseCallback("OBB Annotator (Custom)", self.mouse_callback)
        
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord('q'): # ESC/Q
                break
            elif key == ord('s'):
                self.save_txt()
            elif key == ord('c'):
                self.labels = []
                self.points = []
                self.redraw()
            elif key >= ord('1') and key <= ord('6'):
                self.current_class = key - ord('1')
                self.redraw()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = OBBAnnotator()
    if app.image_files:
        app.run()
