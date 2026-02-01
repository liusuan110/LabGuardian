import cv2
import sys
import threading
import time
from datetime import datetime
from queue import Queue
import os
import glob
import numpy as np
import networkx as nx
from tkinter import filedialog, simpledialog # 新增 simpledialog

# 图形界面库
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk

# AI 库
from ultralytics import YOLO
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
from openai import OpenAI # 新增 OpenAI 库支持

# --- Cloud LLM 配置 ---
USE_CLOUD_LLM = True # 开启云端 AI 模式 (更智能，支持中文)

# 请在下方填入您的 API Key (例如 DeepSeek, Moonshot, Alibaba DashScope 等)
# 这里预设为 DeepSeek 的配置，如果您有其他 OpenAI 兼容的 Key 也可以填
CLOUD_API_KEY = "sk-756ee8992b8342a6926bc3b5a90e90a9" # <--- 请在此处填入您的 Key
CLOUD_BASE_URL = "https://api.deepseek.com" # 或者 api.moonshot.cn 等
CLOUD_MODEL_NAME = "deepseek-chat" # 或者 moonshot-v1-8k, qwen-turbo 等

# --- 引入新模块 ---
from calibration import board_map
from circuit_logic import CircuitAnalyzer, CircuitComponent, validator
from schematic_viz import SchematicGenerator # 新增可视化工器
analyzer = CircuitAnalyzer()

# --- 配置修正区域 ---
# 获取当前脚本所在的 src 目录绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 动态查找 YOLO 模型 (best.pt)
# 我们假设 runs 文件夹位于 src 目录下
runs_dir = os.path.join(BASE_DIR, 'runs', 'detect')
print(f"Debug: Looking for YOLO models in {runs_dir}")

# 查找 lab_guardian* 下所有的 best.pt
candidates = glob.glob(os.path.join(runs_dir, 'lab_guardian*', 'weights', 'best.pt'))

if candidates:
    # 优先寻找包含 "oneshot" 的模型，因为这些是针对演示优化的
    oneshot_candidates = [c for c in candidates if "oneshot" in c]
    if oneshot_candidates:
        YOLO_MODEL_PATH = max(oneshot_candidates, key=os.path.getmtime)
        print(f"Debug: Found Dedicated Demo Model: {YOLO_MODEL_PATH}")
    else:
        # 否则找通用的最新的
        YOLO_MODEL_PATH = max(candidates, key=os.path.getmtime)
        print(f"Debug: Found YOLO model: {YOLO_MODEL_PATH}")
else:
    # 如果没找到，尝试一个硬编码的默认路径 (回退)
    print("Debug: No custom model found, attempting default path...")
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, "runs/detect/lab_guardian_v1/weights/best.pt")

# 2. 动态设置 LLM 模型路径 (使用绝对路径解决 from_pretrained 报错)
LLM_MODEL_PATH = os.path.join(BASE_DIR, "openvino_tinyllama_model")
print(f"Debug: LLM Model Path set to: {LLM_MODEL_PATH}")

# --- 主程序类 ---
class LabGuardianApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LabGuardian - Intel Embedded AI Assistant")
        self.root.geometry("1400x900")
        
        # 智能配置检查
        global CLOUD_API_KEY
        if USE_CLOUD_LLM and "placeholder" in CLOUD_API_KEY:
             # 尝试弹窗请求 Key
             key = simpledialog.askstring("DeepSeek API 配置", 
                                        "检测到您开启了云端 AI 模式。\n请输入您的 DeepSeek API Key (sk-...):\n(如果不输入将只能使用功能受限的本地模型)",
                                        parent=self.root)
             if key and key.startswith("sk-"):
                 CLOUD_API_KEY = key.strip()
                 print(f"Debug: API Key set via Dialog.")
        
        # 状态变量
        self.is_running = True
        self.current_detection = "None"
        self.llm_thinking = False
        self.analyzer = analyzer # 初始化电路分析器实例
        
        # 输入源控制
        self.input_source = "camera" # 'camera' 或 'image'
        self.static_frame = None     # 存储加载的静态图片

        # 校准状态
        self.calibration_requested = False
        
        # 初始化界面
        self.setup_ui()
        
        # 初始化 AI 引擎 (后台加载，防止卡死界面)
        self.log("正在初始化 AI 引擎 (YOLO + LLM)...")
        threading.Thread(target=self.load_models, daemon=True).start()
        
    def setup_ui(self):
        # 布局：左边是摄像头，右边是控制台和聊天框
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：视频区
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=3)
        
        self.video_label = ttk.Label(left_frame, text="摄像头初始化中...", background="black", foreground="white")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # 右侧：交互区
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)
        
        # 标题和状态
        status_frame = ttk.Labelframe(right_frame, text="System Status")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Loading AI Models...", font=("Arial", 12, "bold"))
        self.status_label.pack(pady=10)
        
        self.detect_label = ttk.Label(status_frame, text="Current Object: None", font=("Arial", 10))
        self.detect_label.pack(pady=5)
        
        # 聊天记录
        chat_frame = ttk.Labelframe(right_frame, text="AI Assistant Chat")
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.chat_history = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_history.insert(tk.END, "System: Welcome to LabGuardian.\n")
        self.chat_history.config(state=tk.DISABLED) # 只读
        
        # 按钮区
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        # 新增：用户输入框
        input_frame = ttk.Frame(btn_frame)
        input_frame.pack(fill=tk.X, pady=2)
        ttk.Label(input_frame, text="Ask:").pack(side=tk.LEFT)
        self.user_input = ttk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.user_input.bind("<Return>", lambda e: self.ask_ai_thread()) # 回车发送
        
        self.ask_btn = ttk.Button(btn_frame, text="🔍 Ask AI (Circuit Aware)", command=self.ask_ai_thread)
        self.ask_btn.pack(fill=tk.X, pady=5)
        
        # 新增 Calibration 按钮
        calib_btn = ttk.Button(btn_frame, text="📐 Calibrate Camera", command=self.start_calibration)
        calib_btn.pack(fill=tk.X, pady=5)
        
        # 新增加载图片按钮
        load_btn = ttk.Button(btn_frame, text="📂 Load Test Image", command=self.load_test_image)
        load_btn.pack(fill=tk.X, pady=5)
        
        # 新增显示网表按钮
        netlist_btn = ttk.Button(btn_frame, text="📝 Show Circuit Netlist", command=self.show_netlist)
        netlist_btn.pack(fill=tk.X, pady=5)
        
        # 新增显示原理图按钮
        draw_btn = ttk.Button(btn_frame, text="🎨 Draw Schematic", command=self.draw_schematic)
        draw_btn.pack(fill=tk.X, pady=5)

        # --- Demo Controls (容错控制) ---
        ctrl_frame = ttk.Labelframe(btn_frame, text="Demo Controls")
        ctrl_frame.pack(fill=tk.X, pady=10)
        
        # 1. 置信度滑块
        ttk.Label(ctrl_frame, text="AI Sensitivity (Conf)").pack(anchor='w', padx=5)
        self.conf_slider = ttk.Scale(ctrl_frame, from_=0.01, to=0.99, orient=tk.HORIZONTAL)
        self.conf_slider.set(0.25) # 默认值
        self.conf_slider.pack(fill=tk.X, padx=5, pady=2)
        
        # 2. 重置按钮
        ttk.Button(ctrl_frame, text="🔄 Reset Circuit Analyzer", command=self.reset_analyzer).pack(fill=tk.X, pady=5)
        
        # --- Debug / Validation Zone ---
        debug_frame = ttk.Labelframe(btn_frame, text="Circuit Debugger")
        debug_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(debug_frame, text="⭐ Set as Gold Ref", command=self.set_golden_ref).pack(fill=tk.X, pady=2)
        ttk.Button(debug_frame, text="✅ Validate Current", command=self.validate_circuit).pack(fill=tk.X, pady=2)
        
        quit_btn = ttk.Button(btn_frame, text="Quit Application", command=self.close_app)
        quit_btn.pack(fill=tk.X)

    def reset_analyzer(self):
        if hasattr(self, 'analyzer'):
            self.analyzer.reset()
            self.log("🔄 Circuit Analyzer Reset. Cleared all connections.")
            
    def set_golden_ref(self):
        if hasattr(self, 'analyzer') and self.analyzer.components:
            validator.set_reference(self.analyzer)
            self.log("✅ Current circuit saved as Golden Reference.")
            self.log(f"   (Components: {len(self.analyzer.components)})")
        else:
            self.log("⚠️ Analyzer empty, cannot set reference.")

    def validate_circuit(self):
        if hasattr(self, 'analyzer'):
            self.log("Running validation...")
            results = validator.compare(self.analyzer)
            self.log("--- Validation Report ---")
            for msg in results:
                self.log(msg)
            self.log("-----------------------")
        else:
            self.log("System not ready.")

    def show_netlist(self):
        if hasattr(self, 'analyzer'):
            netlist = self.analyzer.get_circuit_description()
            self.log("--- Generated Netlist ---")
            self.log(netlist)
            self.log("------------------------")
        else:
            self.log("System not ready or analyzer not initialized.")

    def draw_schematic(self):
        if hasattr(self, 'analyzer'):
            try:
                self.log("Generating Schematic...")
                # 注意：Matplotlib 绘图可能会阻塞 GUI，视情况可能需要独立进程
                # 但这里作为演示先直接调用
                generator = SchematicGenerator(self.analyzer)
                generator.generate_schematic(show=True)
                self.log("✅ Schematic Drawn.")
            except Exception as e:
                self.log(f"Schematic Error: {e}")
        else:
            self.log("System not ready.")

    def log(self, text):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {text}\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def load_models(self):
        try:
            # 1. 加载 YOLO
            self.status_label.config(text="Loading Vision Model...", foreground="orange")
            # 尝试加载上次训练的模型，如果没找到则用通用的
            try:
                self.yolo_model = YOLO(YOLO_MODEL_PATH) 
            except:
                self.log(f"Warning: Could not find {YOLO_MODEL_PATH}, using yolov8n.pt")
                self.yolo_model = YOLO("yolov8n.pt")
                
            self.log("✅ Vision Model Loaded.")
            
            # 2. 加载 LLM
            self.status_label.config(text="Loading Language Model...", foreground="orange")
            
            if USE_CLOUD_LLM:
                if not CLOUD_API_KEY or "placeholder" in CLOUD_API_KEY:
                    self.log("⚠️ Cloud AI enabled but Key missing.")
                    self.log("Running in Vision-Only mode until Key is added.")
                    self.llm_client = None
                else:
                    try:
                        self.llm_client = OpenAI(api_key=CLOUD_API_KEY, base_url=CLOUD_BASE_URL)
                        self.log(f"✅ Cloud AI Ready: {CLOUD_MODEL_NAME}")
                    except Exception as e:
                        self.log(f"Cloud AI Error: {e}")
                        self.llm_client = None
            else:
                # 关键修改：此时传入的是绝对路径，所以不会再被误判为 repo id
                self.llm_model = OVModelForCausalLM.from_pretrained(LLM_MODEL_PATH, device="GPU")
                self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
                self.log("✅ Local Language Model Loaded.")
            
            self.status_label.config(text="System Ready - Active", foreground="green")
            
            # 3. 开启摄像头线程
            threading.Thread(target=self.video_loop, daemon=True).start()
            
        except Exception as e:
            self.log(f"Error loading models: {e}")
            self.status_label.config(text="System Error", foreground="red")

    def start_calibration(self):
        self.log("Starting calibration... Please click 4 corners in the popup window.")
        self.calibration_requested = True

    def load_test_image(self):
        file_path = filedialog.askopenfilename(title="Select Circuit Image", filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                self.static_frame = img
                self.input_source = "image"
                self.log(f"Loaded image: {os.path.basename(file_path)}")
                self.status_label.config(text="Mode: Test Image", foreground="blue")
            else:
                self.log("Error: Failed to load image.")

    def video_loop(self):
        cap = cv2.VideoCapture(0)
        # 降低分辨率以提高性能
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 校准用的临时变量
        calib_points = []
        def mouse_callback(event, x, y, flags, param):
            # param 是当前的缩放比例 scale
            current_scale = param if param else 1.0
            
            # 左键：校准选点
            if event == cv2.EVENT_LBUTTONDOWN and self.calibration_requested:
                real_x = int(x / current_scale)
                real_y = int(y / current_scale)
                calib_points.append([real_x, real_y])
                print(f"Calibration Point: {real_x}, {real_y}")
            
            # 右键：演示数据采集 (获取元件坐标)
            elif event == cv2.EVENT_RBUTTONDOWN:
                real_x = int(x / current_scale)
                real_y = int(y / current_scale)
                # 打印出可以直接复制到代码里的格式
                print(f"DEMO_DATA: [ {real_x}, {real_y}, {real_x+50}, {real_y+80} ] # Clicked Center")
                # 在画面上画个临时的圈反馈一下
                cv2.circle(frame, (real_x, real_y), 5, (0, 0, 255), -1)
                cv2.imshow(win_name, frame)

        while self.is_running:
            # --- 1. 获取每一帧 ---
            if self.input_source == "image" and self.static_frame is not None:
                # 使用加载的静态图片，必须 copy 否则后续绘图会污染原图
                frame = self.static_frame.copy()
                # 为了防止静态图太大撑爆屏幕，可以 resize (可选)
                # frame = cv2.resize(frame, (1024, 768)) 
                time.sleep(0.05) # 静态图不需要高速刷新
            else:
                # 使用摄像头
                ret, frame = cap.read()
                if not ret: break

            # --- 校准模式 ---
            if self.calibration_requested:
                win_name = "Calibrate: Click 4 corners (TL->TR->BR->BL)"
                
                # 计算缩放比例，同时适应屏幕宽和高
                h, w = frame.shape[:2]
                max_w = 1000
                max_h = 700
                
                scale_w = max_w / w
                scale_h = max_h / h
                scale = min(scale_w, scale_h, 1.0) # 取最小比例，且不放大
                
                new_w = int(w * scale)
                new_h = int(h * scale)
                disp_frame = cv2.resize(frame, (new_w, new_h))
                
                cv2.namedWindow(win_name) 
                # 传递 scale 给回调
                cv2.setMouseCallback(win_name, mouse_callback, param=scale)
                
                # 绘制已选点 (需要转换回屏幕坐标)
                draw_frame = disp_frame.copy()
                for i, p in enumerate(calib_points):
                    sx = int(p[0] * scale)
                    sy = int(p[1] * scale)
                    cv2.circle(draw_frame, (sx, sy), 5, (0, 0, 255), -1)
                    cv2.putText(draw_frame, str(i+1), (sx+10, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                cv2.imshow(win_name, draw_frame)
                cv2.waitKey(1)
                
                if len(calib_points) == 4:
                    import numpy as np
                    try:
                        # calib_points 存储的是真实坐标，直接用于计算矩阵
                        src_pts = np.float32(calib_points)
                        board_map.get_perspective_matrix(src_pts)
                        self.log("✅ Calibration successful! Board mapped.")
                        
                        # --- 弹窗显示校准后的效果 (带网格) ---
                        warped = board_map.apply_transform(frame)
                        
                        # 结果窗口也需要缩放显示
                        h_w, w_w = warped.shape[:2]
                        # 结果窗口同样应用最大宽高限制
                        s_w = max_w / w_w
                        s_h = max_h / h_w
                        scale_res = min(s_w, s_h, 1.0)
                            
                        warped_disp = cv2.resize(warped, (int(w_w * scale_res), int(h_w * scale_res)))
                        h_wd, w_wd = warped_disp.shape[:2]
                        
                        # 画行线 (Dynamic Rows)
                        rows_count = getattr(board_map, 'rows', 63)
                        for i in range(1, rows_count + 1): 
                            y = int(i * (h_wd / rows_count))
                            cv2.line(warped_disp, (0, y), (w_wd, y), (50, 50, 50), 1)
                            
                        cv2.imshow("Calibration Result (Press Any Key to Close)", warped_disp)
                        cv2.waitKey(0) # 等待按键
                        try:
                            if cv2.getWindowProperty("Calibration Result (Press Any Key to Close)", cv2.WND_PROP_VISIBLE) >= 1:
                                cv2.destroyWindow("Calibration Result (Press Any Key to Close)")
                        except:
                            pass
                        
                    except Exception as e:
                        self.log(f"Calibration failed: {e}")
                    
                    try:
                        cv2.destroyWindow(win_name)
                    except:
                        pass
                    
                    self.calibration_requested = False
                    calib_points = [] # 清空以备下次使用
                continue # 暂停主界面的更新，专注于校准窗口

            # YOLO 推理
            if hasattr(self, 'yolo_model'):
                # 获取滑块的当前值作为动态阈值
                current_conf = 0.25
                if hasattr(self, 'conf_slider'):
                    current_conf = self.conf_slider.get()

                # 使用动态阈值进行推理
                results = self.yolo_model(frame, verbose=False, conf=current_conf) 
                annotated_frame = results[0].plot()

                # LOG detected objects for debugging
                if results[0].boxes:
                    det_info = [f"{results[0].names[int(b.cls[0])]}({float(b.conf[0]):.2f})" for b in results[0].boxes]
                    current_det_str = ",".join(det_info)
                    if not hasattr(self, 'last_det_log') or self.last_det_log != current_det_str:
                        print(f"DEBUG DETECT: {current_det_str}") 
                        self.last_det_log = current_det_str
                
                # --- 新增 Day 2 逻辑: 将检测框映射回电路逻辑坐标 (Advanced 2-Pin Logic) ---
                if hasattr(self, 'analyzer'):
                    self.analyzer = analyzer # 确保引用的是全局导入的那个实例，或者在这里复位
                    self.analyzer.reset() # 每一帧重新构建电路图

                if board_map.matrix is not None and results[0].boxes:
                    for box in results[0].boxes:
                        # 获取边界框坐标 (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls_id = int(box.cls[0])
                        label = results[0].names[cls_id]
                        
                        # --- 核心升级：元件物理指纹库 (Component Fingerprint Logic) ---
                        w = x2 - x1
                        h = y2 - y1
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        
                        pin1_pixel, pin2_pixel = None, None
                        
                        # 1. 轴向元件 (Axial): 电阻、二极管、导线
                        # 特点：引脚位于长轴的两端，像哑铃一样
                        if label in ["Wire", "RESISTOR", "DIODE", "Resistor"]: 
                            if w > h: # 横向摆放
                                margin = w * 0.05 # 5% 的边缘内缩，更精确
                                pin1_pixel = (x1 + margin, cy)
                                pin2_pixel = (x2 - margin, cy)
                            else: # 纵向摆放
                                margin = h * 0.05
                                pin1_pixel = (cx, y1 + margin)
                                pin2_pixel = (cx, y2 - margin)
                                
                        # 2. 径向/直插元件 (Radial): LED, 电容
                        # 特点：引脚通常都在底部，但在俯视图中，它们往往跨越 1-2 个孔
                        # 我们假设它们垂直于面包板插入
                        elif label in ["LED", "CAPACITOR"]:
                            # 即使是圆形的LED，我们也尝试寻找它的长轴方向来确定引脚分布
                            if h > w: # 主要是纵向
                                offset = h * 0.25 # 引脚不像电阻那样在最边缘
                                pin1_pixel = (cx, y1 + offset)
                                pin2_pixel = (cx, y2 - offset)
                            else: # 主要是横向
                                offset = w * 0.25
                                pin1_pixel = (cx - offset, cy)
                                pin2_pixel = (cx + offset, cy)
                                
                        # 3. 封装元件 (Package): 按钮
                        # 特点：四脚方块。对于电路逻辑，我们关注它是否跨接了不同的 Net。
                        # 通常按钮会跨接中间的沟槽，或者连接非连通的行。
                        # 我们取其物理上的“上边缘中心”和“下边缘中心”作为等效引脚。
                        elif "Button" in label:
                            margin = h * 0.15
                            pin1_pixel = (cx, y1 + margin)
                            pin2_pixel = (cx, y2 - margin)
                            
                        # 4. 默认回退逻辑
                        else:
                            if w > h:
                                pin1_pixel = (x1 + w*0.1, cy)
                                pin2_pixel = (x2 - w*0.1, cy)
                            else:
                                pin1_pixel = (cx, y1 + h*0.1)
                                pin2_pixel = (cx, y2 - h*0.1)

                        # 定义变换函数 (Pixel -> Warped -> Logic)
                        def get_logic_loc(px, py):
                            src_point = np.array([[[px, py]]], dtype=np.float32)
                            dst_point = cv2.perspectiveTransform(src_point, board_map.matrix)
                            wx, wy = dst_point[0][0]
                            return board_map.pixel_to_logic(wx, wy)

                        loc1 = get_logic_loc(*pin1_pixel)
                        loc2 = get_logic_loc(*pin2_pixel)
                        
                        # 如果两个引脚都在有效区域
                        if loc1 and loc2 and loc1[0] != "Groove" and loc2[0] != "Groove":
                            # 添加到电路分析器
                            comp = CircuitComponent(f"{label}", label, loc1, loc2)
                            self.analyzer.add_component(comp)
                            
                            # 绘制 "虚拟连接" (绿色线条代表系统认为它们已连接)
                            cv2.line(annotated_frame, (int(pin1_pixel[0]), int(pin1_pixel[1])), 
                                     (int(pin2_pixel[0]), int(pin2_pixel[1])), (0, 255, 0), 2)
                            
                            # 显示引脚及其所在的行号
                            cv2.circle(annotated_frame, (int(pin1_pixel[0]), int(pin1_pixel[1])), 4, (255, 0, 0), -1)
                            cv2.circle(annotated_frame, (int(pin2_pixel[0]), int(pin2_pixel[1])), 4, (0, 0, 255), -1)
                            
                            # 文字标注: "R1: 15-20"
                            info = f"{loc1[0]}-{loc2[0]}"
                            cv2.putText(annotated_frame, info, (int(x1), int(y1)-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # --- 分析当前电路并显示简报 ---
                    try:
                         # 简单的显示连通组数
                        connected_nets = list(nx.connected_components(self.analyzer.graph))
                        status_text = f"Nets Found: {len(connected_nets)}"
                        cv2.putText(annotated_frame, status_text, (20, 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                                   
                        # (可选) 如果检测到 Row10 和 Row20 连通，显示大大的 "Success"
                        # 这是演示的一个 trick
                        if self.analyzer.validate_connection('10', '20'): # 示例检查
                             cv2.putText(annotated_frame, "Circuit Closed!", (20, 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                    except Exception as e:
                        print(f"Analysis Error: {e}")


                # 更新当前检测到的物体
                if results[0].boxes:
                    # 获取置信度最高的类别
                    cls_id = int(results[0].boxes.cls[0])
                    obj_name = results[0].names[cls_id]
                    self.current_detection = obj_name
                else:
                    self.current_detection = "None"
                    
                # 在主线程更新界面文本
                self.root.after(0, lambda: self.detect_label.config(text=f"Detected: {self.current_detection}"))
            else:
                annotated_frame = frame

            # 转换颜色 BGR -> RGB 用于 tkinter 显示
            cv_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            
            # 缩放图片以适应左侧视频显示区域 (防止图片过大只显示局部)
            # 获取左侧 Label 的当前大小
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            # 如果窗口刚启动尚未layout完成，给一个默认值
            if label_width < 100: label_width = 800
            if label_height < 100: label_height = 600
            
            # 保持比例缩放
            pil_image.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)
            
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # 更新视频 Label
            self.root.after(0, lambda img=tk_image: self.update_video_label(img))
            
            # 稍微休眠，释放 CPU
            time.sleep(0.01)
            
        cap.release()

    def update_video_label(self, img):
        self.video_label.configure(image=img)
        self.video_label.image = img

    def ask_ai_thread(self):
        if self.llm_thinking:
            return
            
        # 优先使用用户输入的问题
        custom_q = self.user_input.get().strip()
        if custom_q:
             self.user_input.delete(0, tk.END) # 清空输入框
             threading.Thread(target=self.process_question, args=(self.current_detection, custom_q), daemon=True).start()
        elif self.current_detection != "None":
             threading.Thread(target=self.process_question, args=(self.current_detection, None), daemon=True).start()
        else:
             self.log("System: No object detected and no question typed.")

    def process_question(self, obj_name, custom_question=None):
        self.llm_thinking = True
        self.ask_btn.state(["disabled"])
        
        # 1. 获取电路拓扑描述
        circuit_context = ""
        if hasattr(self, 'analyzer'):
            circuit_context = self.analyzer.get_circuit_description()
        
        # 2. 构造 Prompt (中文优化)
        if custom_question:
            question = custom_question
            system_prompt = f"""你是一个专业的电子实验室助手。
你拥有计算机视觉系统提供的实时电路网表数据：
{circuit_context}

请基于此电路状态回答用户问题。
- 如果被问及连接，请根据 Net (网络) 信息判断。
- Push_Button = 按钮开关, Wire = 导线。
- 请用**中文**回答，像一个人类助教一样自然。"""
        else:
            # 默认问题
            question = f"我正在看一个 {obj_name}，请告诉我它在这个电路中是怎么连接的？"
            system_prompt = f"你是一个实验室助手。当前电路连接如下：\n{circuit_context}\n请用中文简要描述 {obj_name} 的连接情况。"
        
        self.log(f"User: {question}")
        self.log("AI: 正在思考..." if USE_CLOUD_LLM else "AI: Local Thinking...")
        
        try:
            if USE_CLOUD_LLM and hasattr(self, 'llm_client') and self.llm_client:
                # --- Cloud AI 调用 ---
                response = self.llm_client.chat.completions.create(
                    model=CLOUD_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                final_answer = response.choices[0].message.content.strip()
                self.log(f"AI: {final_answer}")
            
            elif hasattr(self, 'llm_model'):
                # --- Local AI 调用 (Fallback) ---
                # 构造输入 (英文 Prompt 以保证 TinyLlama 效果)
                messages = [
                    {"role": "system", "content": "You are a helpful lab assistant. Answer based on the circuit netlist provided."},
                    {"role": "user", "content": f"Context: {circuit_context}\nQuestion: {question}"},
                ]
                input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.llm_model.device)
                
                output = self.llm_model.generate(input_ids, max_new_tokens=100, temperature=0.7)
                answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
                final_answer = answer.split("<|assistant|>")[-1].strip()
                self.log(f"AI (Local): {final_answer}")
            else:
                self.log("AI Error: No model available.")
                
        except Exception as e:
            self.log(f"Error generating answer: {e}")
            
        self.llm_thinking = False
        self.root.after(0, lambda: self.ask_btn.state(["!disabled"]))

    def close_app(self):
        self.is_running = False
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = LabGuardianApp(root)
    root.mainloop()
