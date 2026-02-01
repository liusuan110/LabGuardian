import cv2
import numpy as np

class BreadboardMap:
    def __init__(self):
        # 面包板标准参数 (以标准830孔面包板为例)
        # 上下两个电源轨，中间是元件区(通常63行)
        # 我们这里定义一个简化的逻辑坐标系：
        # Row: 1-63 (Standard) or 1-30 (Half-size)
        # Col: A-E, F-J (对应区域)
        self.rows = 30 # Modified for Half-size Breadboard (Demo Image)
        self.cols_per_side = 5 # a-e, f-j
        self.matrix = None # 透视变换矩阵
        self.width = 0
        self.height = 0
        
        # 预设的输出图像大小 (宽, 高)，建议设大一点以保证精度
        self.output_size = (600, 800) # Adjusted aspect ratio for half-size

        # --- Hole grid (孔洞网格) ---
        # 在校准后的俯视图(warped)上检测孔洞中心点，并用于后续 pin->hole 吸附
        self.hole_centers = []  # list[(x,y)] in warped coordinates
        self.row_centers = None # np.ndarray shape (rows,)
        self.col_centers = None # np.ndarray shape (10,)

    def reset_holes(self):
        self.hole_centers = []
        self.row_centers = None
        self.col_centers = None

    def detect_holes(self, warped_bgr):
        """Detect breadboard hole centers on the warped (top-down) image.

        Returns:
            int: number of detected holes
        """
        self.reset_holes()
        if warped_bgr is None or self.matrix is None:
            return 0

        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Holes are dark; invert threshold helps blob detector
        thr = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 3
        )
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        # 经验范围：基于 output_size=(600,800) 的孔洞面积（会随清晰度变化）
        params.minArea = 15
        params.maxArea = 400
        params.filterByCircularity = True
        params.minCircularity = 0.2
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(thr)
        if not keypoints:
            return 0

        h, w = thr.shape[:2]

        # 过滤掉电源轨等区域（经验裁剪，避免把电源轨孔也算进来影响聚类）
        x_min, x_max = int(0.08 * w), int(0.92 * w)
        y_min, y_max = int(0.08 * h), int(0.92 * h)

        centers = []
        for kp in keypoints:
            x, y = kp.pt
            if x_min <= x <= x_max and y_min <= y <= y_max:
                centers.append((float(x), float(y)))

        if len(centers) < 50:
            # 太少通常意味着阈值/光照不合适
            self.hole_centers = centers
            return len(self.hole_centers)

        self.hole_centers = centers

        # --- 估计 row/col 中心（不依赖 sklearn） ---
        ys = np.array([c[1] for c in centers], dtype=np.float32)
        xs = np.array([c[0] for c in centers], dtype=np.float32)

        def quantile_bin_centers(values, bins):
            values = np.sort(values)
            n = len(values)
            if n < bins:
                return None
            bin_size = max(1, n // bins)
            centers_out = []
            for i in range(bins):
                start = i * bin_size
                end = (i + 1) * bin_size if i < bins - 1 else n
                seg = values[start:end]
                centers_out.append(float(np.median(seg)))
            return np.array(centers_out, dtype=np.float32)

        self.row_centers = quantile_bin_centers(ys, self.rows)
        self.col_centers = quantile_bin_centers(xs, 10)

        return len(self.hole_centers)

    def nearest_hole(self, wx, wy):
        """Return nearest hole center (hx,hy) in warped coordinates, or None."""
        if not self.hole_centers:
            return None
        pts = np.array(self.hole_centers, dtype=np.float32)
        dx = pts[:, 0] - float(wx)
        dy = pts[:, 1] - float(wy)
        idx = int(np.argmin(dx * dx + dy * dy))
        return (float(pts[idx, 0]), float(pts[idx, 1]))

    def hole_to_logic(self, hx, hy):
        """Map a hole center (warped) to (row_str, col_char) if row/col centers exist."""
        if self.row_centers is None or self.col_centers is None:
            return None
        row_idx = int(np.argmin(np.abs(self.row_centers - float(hy)))) + 1
        col_idx = int(np.argmin(np.abs(self.col_centers - float(hx))))
        if col_idx <= 4:
            col_char = chr(ord('a') + col_idx)
        else:
            col_char = chr(ord('f') + (col_idx - 5))
        row_idx = max(1, min(self.rows, row_idx))
        return (f"{row_idx}", col_char)

    def warped_point_to_logic(self, wx, wy):
        """Prefer hole-based mapping; fall back to linear mapping."""
        hole = self.nearest_hole(wx, wy)
        if hole is not None:
            loc = self.hole_to_logic(hole[0], hole[1])
            if loc is not None:
                return loc
        return self.pixel_to_logic(wx, wy)

    def logic_to_warped(self, row_str, col_char):
        """Map (row_str, col_char) to an approximate (wx,wy) using learned row/col centers.
        Returns None if not available.
        """
        if self.row_centers is None or self.col_centers is None:
            return None
        try:
            row = int(row_str)
        except Exception:
            return None
        row = max(1, min(self.rows, row))
        wy = float(self.row_centers[row - 1])
        if col_char <= 'e':
            col_idx = ord(col_char) - ord('a')
        else:
            col_idx = 5 + (ord(col_char) - ord('f'))
        col_idx = max(0, min(9, col_idx))
        wx = float(self.col_centers[col_idx])
        return (wx, wy)

    def get_perspective_matrix(self, src_points):
        """
        计算透视变换矩阵
        src_points: 原始图像中选取的4个点 (numpy array)
        顺序：左上 -> 右上 -> 右下 -> 左下
        """
        # 定义目标图像中的4个角点 (将图像拉伸为垂直长方形)
        w, h = self.output_size
        dst_points = np.array([
            [0, 0],       # 左上
            [w, 0],       # 右上
            [w, h],       # 右下
            [0, h]        # 左下
        ], dtype="float32")

        if len(src_points) != 4:
            return None
        
        # 计算透视变换矩阵
        self.matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.width = w
        self.height = h
        return self.matrix

    def apply_transform(self, image):
        """
        应用透视变换，返回矫正后的“俯视图”
        """
        if self.matrix is None:
            return image
        
        warped = cv2.warpPerspective(image, self.matrix, (self.width, self.height))
        return warped

    def pixel_to_logic(self, x_pixel, y_pixel):
        """
        将像素坐标转换为面包板孔位逻辑坐标 (Row, Col)
        注意：输入的 x, y 是矫正后图像(Top-Down View)上的坐标！
        
        【重要修正 - Day 3】
        用户指出面包板导通规则是 "竖向导通，横向不导通"。
        这意味着面包板在画面中是 "Landscape" 放置的（或者用户的理解模式为此）：
        - 每一列 (Vertical Strip) 是一个节点 (Net)
        - 每一行 (Horizontal) 是绝缘的
        
        我们需要交换 Row 和 Col 的几何映射逻辑：
        Original (Portrait):
           y -> Row ID (1..30) [Connected]
           x -> Col ID (a..e)
           
        New (Landscape / Vertical Conducting):
           x -> Row ID (1..30) [Connected]  <-- The "Row Number" now represents the vertical strip index
           y -> Col ID (a..j)               <-- The "Col Char" is just position within the strip
        """
        if self.matrix is None:
            return None

        # --- 重新定义几何映射 ---
        # 假设：矫正后的图片是 Landscape (宽 > 高)
        # 宽度被 30 (或63) 列平分 -> 这些是导通的 Strips
        
        # 计算 "Row Number" (逻辑上的 Row，物理上的 X轴列)
        # 1-30
        strip_width = self.width / self.rows
        logic_row_idx = int(x_pixel / strip_width) + 1
        
        # 计算 "Col Char" (逻辑上的 Col，物理上的 Y轴位置)
        # 上半部分 a-e, 下半部分 f-j, 中间有沟槽
        # Y轴: Top(0) -> Bottom(Height)
        
        split_ratio = 0.5
        groove_height = self.height * 0.1 # 沟槽高度
        hole_pitch_y = (self.height - groove_height) / 10 # 10个孔位在Y轴
        
        # 判断是在沟槽上方还是下方
        # 假设 Top is a-e, Bottom is f-j
        if y_pixel < (self.height / 2 - groove_height / 2):
             # 上半区 a-e (Note: usually 'a' is top, 'e' is near groove)
             col_idx = int(y_pixel / hole_pitch_y)
             side = 'Top'
             col_name = chr(ord('a') + col_idx)
             
        elif y_pixel > (self.height / 2 + groove_height / 2):
             # 下半区 f-j
             effective_y = y_pixel - (self.height / 2 + groove_height / 2)
             col_idx = int(effective_y / hole_pitch_y)
             side = 'Bottom'
             col_name = chr(ord('f') + col_idx)
        else:
             return "Groove", 0

        if logic_row_idx < 1: logic_row_idx = 1
        if logic_row_idx > self.rows: logic_row_idx = self.rows

        return f"{logic_row_idx}", col_name

# 全局单例，方便调用
board_map = BreadboardMap()

# --- 单元测试 ---
if __name__ == "__main__":
    # 模拟一个鼠标回调来选点
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print(f"Selected point: {x}, {y}")
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    cap = cv2.VideoCapture(0) # 打开摄像头
    
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)
    
    print("【Step 1】请按顺序点击面包板的4个角：左上 -> 右上 -> 右下 -> 左下")
    print("【Step 2】点完4个点后，按 'c' 键计算变换矩阵")
    print("【Step 3】之后点击 'Warpped' 窗口，测试坐标映射")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 画出已选的点
        for p in points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
            
        cv2.imshow("Calibration", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c') and len(points) == 4:
            # 计算矩阵
            src_pts = np.float32(points)
            board_map.get_perspective_matrix(src_pts)
            print("矩阵计算完成！打开预览窗口...")
            
            # 开启预览循环
            while True:
                ret, frame = cap.read()
                warped = board_map.apply_transform(frame)
                
                # 画出网格线方便调试
                h, w = warped.shape[:2]
                for i in range(1, 64): # 画行线
                    y = int(i * (h / 63))
                    cv2.line(warped, (0, y), (w, y), (50, 50, 50), 1)
                
                cv2.imshow("Warped View", warped)
                
                # 在 warp 窗口测试鼠标点击
                def test_click(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        res = board_map.pixel_to_logic(x, y)
                        print(f"Pixel: ({x},{y}) -> Logic: {res}")
                        
                cv2.setMouseCallback("Warped View", test_click)
                
                if cv2.waitKey(1) == ord('q'):
                    break
            break
            
    cap.release()
    cv2.destroyAllWindows()
