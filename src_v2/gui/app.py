"""
LabGuardian 主窗口
职责：组装各面板，协调各模块间的交互
从原 main.py 的 941 行巨石中拆出为纯 UI 编排层
"""

import threading
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
import networkx as nx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import gui as gui_cfg, vision as vision_cfg
from vision.detector import ComponentDetector
from vision.calibrator import board_calibrator
from vision.stabilizer import DetectionStabilizer
from logic.circuit import CircuitAnalyzer, CircuitComponent
from logic.polarity import polarity_resolver
from logic.validator import validator
from logic.schematic import SchematicGenerator
from ai.llm_engine import LLMEngine
from gui.video_panel import VideoPanel
from gui.chat_panel import ChatPanel


class LabGuardianApp:
    """
    主应用类 — 负责协调各模块
    
    模块依赖关系:
        VideoPanel → on_frame_callback → Detector + Calibrator + Analyzer
        ChatPanel → LLMEngine
        Buttons → Validator / SchematicGenerator
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(gui_cfg.window_title)
        self.root.geometry(f"{gui_cfg.window_size[0]}x{gui_cfg.window_size[1]}")

        # 核心模块实例
        self.detector = ComponentDetector()
        self.stabilizer = DetectionStabilizer(window_size=5, min_hits=3)
        self.analyzer = CircuitAnalyzer()
        self.llm = LLMEngine()

        # 状态
        self.is_running = True
        self.current_detection = "None"
        self.llm_thinking = False
        self.calibration_requested = False
        self.ar_missing_links = []

        # 搭建 UI
        self._setup_ui()

        # 后台加载 AI 模型
        self.chat.log("正在初始化 AI 引擎...")
        threading.Thread(target=self._load_models, daemon=True).start()

    # ========================================================
    # UI 搭建
    # ========================================================

    def _setup_ui(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：视频
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=3)
        self.video = VideoPanel(left_frame, on_frame_callback=self._process_frame)

        # 右侧：交互
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)

        # 状态栏
        status_frame = ttk.Labelframe(right_frame, text="System Status")
        status_frame.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(status_frame, text="Loading...",
                                      font=("Arial", 12, "bold"))
        self.status_label.pack(pady=10)
        self.detect_label = ttk.Label(status_frame, text="Detected: None",
                                      font=("Arial", 10))
        self.detect_label.pack(pady=5)

        # 按钮区 (底部)
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # 聊天面板
        self.chat = ChatPanel(right_frame)

        # 按钮
        self.ask_btn = ttk.Button(btn_frame, text="🔍 Ask AI",
                                  command=self._ask_ai_thread)
        self.ask_btn.pack(fill=tk.X, pady=5)
        self.chat.bind_send(self._ask_ai_thread)

        ttk.Button(btn_frame, text="📐 Calibrate Camera",
                   command=self._start_calibration).pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="📂 Load Test Image",
                   command=self._load_image).pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="📝 Show Netlist",
                   command=self._show_netlist).pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="🎨 Draw Schematic",
                   command=self._draw_schematic).pack(fill=tk.X, pady=5)

        # Demo 控制
        ctrl_frame = ttk.Labelframe(btn_frame, text="Demo Controls")
        ctrl_frame.pack(fill=tk.X, pady=10)

        ttk.Label(ctrl_frame, text="AI Sensitivity (Conf)").pack(anchor='w', padx=5)
        self.conf_slider = ttk.Scale(ctrl_frame, from_=0.01, to=0.99,
                                     orient=tk.HORIZONTAL)
        self.conf_slider.set(vision_cfg.conf_threshold)
        self.conf_slider.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(ctrl_frame, text="🔄 Reset Analyzer",
                   command=self._reset_analyzer).pack(fill=tk.X, pady=5)

        # 验证区
        debug_frame = ttk.Labelframe(btn_frame, text="Circuit Debugger")
        debug_frame.pack(fill=tk.X, pady=10)

        ttk.Button(debug_frame, text="⭐ Set as Gold Ref",
                   command=self._set_golden_ref).pack(fill=tk.X, pady=2)
        ttk.Button(debug_frame, text="💾 Save Template",
                   command=self._save_template).pack(fill=tk.X, pady=2)
        ttk.Button(debug_frame, text="📂 Load Template",
                   command=self._load_template).pack(fill=tk.X, pady=2)
        ttk.Button(debug_frame, text="✅ Validate",
                   command=self._validate_circuit).pack(fill=tk.X, pady=2)

        ttk.Button(btn_frame, text="Quit",
                   command=self._close).pack(fill=tk.X)

    # ========================================================
    # 模型加载
    # ========================================================

    def _load_models(self):
        try:
            self.status_label.config(text="Loading Vision Model...", foreground="orange")
            if self.detector.load():
                self.chat.log("✅ Vision Model Loaded.")
            else:
                self.chat.log("⚠️ Vision Model failed, using fallback.")

            self.status_label.config(text="Loading Language Model...", foreground="orange")
            llm_status = self.llm.load()
            self.chat.log(f"✅ {llm_status}")

            self.status_label.config(text="System Ready", foreground="green")
            self.video.start_video_thread()

        except Exception as e:
            self.chat.log(f"Error: {e}")
            self.status_label.config(text="System Error", foreground="red")

    # ========================================================
    # 帧处理管线 (核心逻辑)
    # ========================================================

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        每帧回调：检测 → 稳定化 → 坐标映射 → 电路分析 → 标注
        
        Args:
            frame: 原始 BGR 帧
        Returns:
            标注后的帧
        """
        if not self.detector.model:
            return frame

        conf = self.conf_slider.get()

        # 1. YOLO 检测
        detections = self.detector.detect(frame, conf=conf)

        # 2. 多帧稳定化
        stable_dets = self.stabilizer.update(detections)

        # 3. 绘制检测结果
        annotated = self.detector.annotate_frame(frame, stable_dets)

        # 4. 坐标映射 + 电路分析
        if board_calibrator.is_calibrated and stable_dets:
            self.analyzer.reset()

            for det in stable_dets:
                if det.pin1_pixel and det.pin2_pixel:
                    loc1 = board_calibrator.frame_pixel_to_logic(*det.pin1_pixel)
                    loc2 = board_calibrator.frame_pixel_to_logic(*det.pin2_pixel)

                    if (loc1 and loc2 and
                            loc1[0] != "Groove" and loc2[0] != "Groove"):
                        comp = CircuitComponent(
                            name=det.class_name,
                            type=det.class_name,
                            pin1_loc=loc1,
                            pin2_loc=loc2,
                            confidence=det.confidence,
                        )

                        # 极性推断
                        obb_corners = det.obb_corners if det.is_obb else None
                        polarity_resolver.enrich(
                            comp,
                            obb_corners=obb_corners,
                            orientation_deg=0.0,
                        )

                        self.analyzer.add_component(comp)

                        # 标注逻辑坐标
                        info = f"{loc1[0]}-{loc2[0]}"
                        x1, y1 = det.bbox[:2]
                        cv2.putText(annotated, info, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 网络简报
            try:
                net_count = self.analyzer.get_net_count()
                cv2.putText(annotated, f"Nets: {net_count}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            except Exception:
                pass

            # AR 幽灵线
            self._draw_ghost_wires(annotated)

        # 5. 更新检测状态
        if stable_dets:
            top = max(stable_dets, key=lambda d: d.confidence)
            self.current_detection = top.class_name
        else:
            self.current_detection = "None"

        self.root.after(0, lambda: self.detect_label.config(
            text=f"Detected: {self.current_detection}"))

        return annotated

    def _draw_ghost_wires(self, frame: np.ndarray):
        """绘制缺失连接的幽灵线引导"""
        if not self.ar_missing_links:
            return

        for pin1_loc, pin2_loc in self.ar_missing_links:
            try:
                p1 = board_calibrator.logic_to_frame_pixel(pin1_loc[0], pin1_loc[1])
                p2 = board_calibrator.logic_to_frame_pixel(pin2_loc[0], pin2_loc[1])
                if p1 and p2:
                    cv2.arrowedLine(frame, p1, p2, (0, 255, 255), 3, tipLength=0.2)
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    cv2.putText(frame, "MISSING", mid,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            except Exception:
                pass

    # ========================================================
    # 用户操作回调
    # ========================================================

    def _start_calibration(self):
        self.chat.log("Calibration: 请在弹出窗口中点击面包板4个角点 (左上→右上→右下→左下)")
        # 用独立线程避免阻塞 GUI
        threading.Thread(target=self._calibration_flow, daemon=True).start()

    def _calibration_flow(self):
        """校准交互流程 (在 OpenCV 窗口中完成)"""
        # 获取当前帧
        if self.video.input_source == "image" and self.video.static_frame is not None:
            frame = self.video.static_frame.copy()
        else:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self.chat.log("Error: Cannot capture frame for calibration.")
                return

        points = []
        win_name = "Calibrate: Click 4 corners (TL->TR->BR->BL)"

        h, w = frame.shape[:2]
        max_w, max_h = 1000, 700
        scale = min(max_w / w, max_h / h, 1.0)
        disp = cv2.resize(frame, (int(w * scale), int(h * scale)))

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                real_x, real_y = int(x / scale), int(y / scale)
                points.append([real_x, real_y])

        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, on_click)

        while len(points) < 4:
            draw = disp.copy()
            for i, p in enumerate(points):
                sx, sy = int(p[0] * scale), int(p[1] * scale)
                cv2.circle(draw, (sx, sy), 5, (0, 0, 255), -1)
                cv2.putText(draw, str(i + 1), (sx + 10, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(win_name, draw)
            if cv2.waitKey(50) == ord('q'):
                cv2.destroyWindow(win_name)
                return

        cv2.destroyWindow(win_name)

        # 计算变换矩阵
        src_pts = np.float32(points)
        board_calibrator.calibrate(src_pts)

        # 执行孔洞检测
        warped = board_calibrator.warp(frame)
        hole_count = board_calibrator.detect_holes(warped)
        self.chat.log(f"✅ Calibration done. Holes detected: {hole_count}")

    def _load_image(self):
        name = self.video.load_test_image()
        if name:
            self.chat.log(f"Loaded: {name}")
            self.status_label.config(text="Mode: Test Image", foreground="blue")

    def _show_netlist(self):
        netlist = self.analyzer.get_circuit_description()
        self.chat.log("--- Netlist ---")
        self.chat.log(netlist)

    def _draw_schematic(self):
        try:
            gen = SchematicGenerator(self.analyzer)
            gen.generate_schematic(show=True)
            self.chat.log("✅ Schematic drawn.")
        except Exception as e:
            self.chat.log(f"Schematic error: {e}")

    def _reset_analyzer(self):
        self.analyzer.reset()
        self.stabilizer.clear()
        self.chat.log("🔄 Analyzer reset.")

    def _set_golden_ref(self):
        if self.analyzer.components:
            validator.set_reference(self.analyzer)
            self.chat.log(f"⭐ Reference set ({len(self.analyzer.components)} components)")
        else:
            self.chat.log("⚠️ No components to set as reference.")

    def _save_template(self):
        path = filedialog.asksaveasfilename(
            title="Save Template", defaultextension=".json",
            filetypes=[("LabGuardian Template", "*.json")]
        )
        if path:
            validator.save_reference(path)
            self.chat.log(f"💾 Saved: {Path(path).name}")

    def _load_template(self):
        path = filedialog.askopenfilename(
            title="Load Template",
            filetypes=[("LabGuardian Template", "*.json")]
        )
        if path:
            validator.load_reference(path)
            self.chat.log(f"📂 Loaded: {Path(path).name}")

    def _validate_circuit(self):
        self.chat.log("Running validation...")
        results = validator.compare(self.analyzer)

        self.chat.log("--- Validation Report ---")
        for msg in results['errors']:
            self.chat.log(msg)

        self.ar_missing_links = results.get('missing_links', [])
        if self.ar_missing_links:
            self.chat.log(f"⚠️ AR Guide: {len(self.ar_missing_links)} missing links shown.")
        self.chat.log("-------------------------")

    # ========================================================
    # AI 问答
    # ========================================================

    def _ask_ai_thread(self):
        if self.llm_thinking:
            return
        question = self.chat.get_user_input()
        if not question and self.current_detection == "None":
            self.chat.log("No object detected and no question typed.")
            return
        threading.Thread(
            target=self._ask_ai, args=(question,), daemon=True
        ).start()

    def _ask_ai(self, question: str):
        self.llm_thinking = True
        self.ask_btn.state(["disabled"])

        circuit_context = self.analyzer.get_circuit_description()

        if question:
            self.chat.log(f"User: {question}")
        else:
            question = f"我正在看一个 {self.current_detection}，请描述它在电路中的连接。"
            self.chat.log(f"User: {question}")

        self.chat.log("AI: 正在思考...")

        try:
            answer = self.llm.ask(question, circuit_context)
            self.chat.log(f"AI: {answer}")
        except Exception as e:
            self.chat.log(f"AI Error: {e}")

        self.llm_thinking = False
        self.root.after(0, lambda: self.ask_btn.state(["!disabled"]))

    # ========================================================
    # 生命周期
    # ========================================================

    def _close(self):
        self.is_running = False
        self.video.stop()
        self.root.destroy()
