"""
LabGuardian 全局配置模块
所有可配置参数集中管理，支持环境变量覆盖，跨平台兼容
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# 路径配置 — 全部使用 pathlib，跨平台兼容
# ============================================================

# 项目根目录 (src_v2/ 的上一级)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# src_v2 目录
SRC_DIR = Path(__file__).resolve().parent

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"

# 模型搜索路径 (按优先级排序)
MODEL_SEARCH_DIRS = [
    SRC_DIR / "runs" / "obb",
    SRC_DIR / "runs" / "detect",
    PROJECT_ROOT / "runs" / "obb",
    PROJECT_ROOT / "runs" / "detect",
]

# 默认 YOLO 预训练权重 (回退用)
DEFAULT_YOLO_WEIGHTS = MODELS_DIR / "yolov8n.pt"

# OpenVINO LLM 模型路径
LOCAL_LLM_MODEL_DIR = MODELS_DIR / "tinyllama_openvino"

# 数据集目录
DATASET_DIR = PROJECT_ROOT / "dataset"

# RAG 知识库目录
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_DB_DIR = KNOWLEDGE_BASE_DIR / "chroma_db"
EMBEDDING_MODEL_DIR = MODELS_DIR / "text2vec_chinese"


# ============================================================
# RAG 知识库配置
# ============================================================

@dataclass
class RAGConfig:
    """
    RAG (检索增强生成) 配置

    知识库目录:
      - builtin/    内置知识 (三极管/运放/实验指南, 随代码分发)
      - user_docs/  用户自添文档 (PDF/TXT/MD, 放入即可)
      - chroma_db/  ChromaDB 持久化目录 (自动生成)
    """
    enabled: bool = True                                    # 是否启用 RAG
    embedding_model: str = "shibing624/text2vec-base-chinese"  # HuggingFace 模型名
    fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 400                                   # 文本分块大小 (字符)
    chunk_overlap: int = 80                                 # 分块重叠字符数
    top_k: int = 5                                          # 检索返回数量
    min_score: float = 0.35                                 # 最低相似度过滤


# ============================================================
# 视觉检测配置
# ============================================================

@dataclass
class VisionConfig:
    """YOLO 推理参数"""
    conf_threshold: float = 0.25          # 置信度阈值
    iou_threshold: float = 0.5            # NMS IoU 阈值
    imgsz: int = 960                      # 推理分辨率
    device: str = "0"                     # 推理设备: "0"(GPU), "cpu", "GPU"(OpenVINO iGPU)
    model_name_hint: str = "lab_guardian"  # 自动搜索模型时的名称关键词
    preferred_model: str = "lab_guardian_oneshot_v1"  # 优先加载的模型名

    # ROI 裁剪 (校准后只检测面包板区域)
    roi_enabled: bool = True              # 启用 ROI 裁剪优化
    roi_padding: int = 30                 # ROI 外扩像素 (防止边缘元件被截断)

    # 帧差分跳帧 (无变化时复用上帧结果)
    frame_diff_enabled: bool = True       # 启用帧差分优化
    frame_diff_threshold: float = 5.0     # 灰度均值差 < 此值则跳帧
    max_skip_frames: int = 15             # 最大连续跳帧数 (安全网, 强制重新检测)


# ============================================================
# 面包板校准配置
# ============================================================

@dataclass
class CalibrationConfig:
    """面包板参数"""
    rows: int = 30                         # 面包板行数 (30=半尺寸, 63=全尺寸)
    cols_per_side: int = 5                 # 每侧列数 (a-e, f-j)
    output_size: tuple = (600, 800)        # 透视变换输出尺寸 (宽, 高)

    # --- Blob 检测参数 (Level 1) ---
    blob_min_area: int = 15
    blob_max_area: int = 400
    blob_min_circularity: float = 0.2
    min_hole_count: int = 50               # 低于此值回退到线性映射

    # --- 多参数扫描 (Level 1 Ensemble) ---
    # 每组: (min_area, max_area, min_circularity)
    blob_param_sweep: tuple = (
        (8, 600, 0.10),    # 宽松: 适合模糊/低对比度/远距离
        (15, 400, 0.20),   # 默认: 通用
        (25, 300, 0.40),   # 严格: 减少误检
    )

    # --- CLAHE 预处理参数 ---
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple = (8, 8)

    # --- 多阈值二值化 ---
    adaptive_block_sizes: tuple = (21, 31, 41)  # 自适应阈值窗口
    adaptive_c_values: tuple = (2, 3, 5)        # 自适应阈值偏移

    # --- Hough 圆检测参数 (Level 2) ---
    hough_dp: float = 1.2                  # 分辨率累加器比率
    hough_min_dist: int = 12               # 检测圆心最小距离
    hough_param1: int = 50                 # Canny 高阈值
    hough_param2: int = 18                 # 累加器阈值 (越小越灵敏)
    hough_min_radius: int = 2              # 最小圆半径
    hough_max_radius: int = 18             # 最大圆半径

    # --- 轮廓检测参数 (Level 3) ---
    contour_min_area: int = 10
    contour_max_area: int = 500
    contour_min_circularity: float = 0.3

    # --- 去重 & 网格拟合参数 ---
    dedup_radius: float = 5.0              # 候选点去重半径 (px)
    grid_fit_tolerance: float = 0.35       # 网格拟合容差 (相对于间距)


# ============================================================
# 摄像头配置
# ============================================================

@dataclass
class CameraConfig:
    """摄像头参数，支持环境变量覆盖"""
    device_id: int = 0                     # 摄像头设备号
    width: int = 640                       # 采集宽度
    height: int = 480                      # 采集高度
    backend: Optional[str] = None          # 后端: None(自动), "v4l2", "dshow"

    def __post_init__(self):
        # 支持环境变量覆盖
        self.device_id = int(os.environ.get("LG_CAMERA_ID", self.device_id))
        self.width = int(os.environ.get("LG_CAMERA_W", self.width))
        self.height = int(os.environ.get("LG_CAMERA_H", self.height))

        # 自动选择后端
        if self.backend is None:
            if sys.platform == "linux":
                self.backend = "v4l2"
            elif sys.platform == "win32":
                self.backend = "dshow"

    @property
    def cv2_backend(self):
        """返回 OpenCV VideoCapture 后端常量"""
        import cv2
        backends = {
            "v4l2": cv2.CAP_V4L2,
            "dshow": cv2.CAP_DSHOW,
            "gstreamer": cv2.CAP_GSTREAMER,
        }
        return backends.get(self.backend, cv2.CAP_ANY)


# ============================================================
# LLM 配置
# ============================================================

@dataclass
class LLMConfig:
    """
    大语言模型配置

    降级策略 (竞赛离线模式):
      1. Local NPU  — Qwen2.5-1.5B INT4 via openvino_genai (首选)
      2. Local GPU   — 同上, 回退到 iGPU
      3. Local CPU   — 同上, 回退到 CPU
      4. Rule        — 领域规则模板引擎 (零依赖兜底)

    降级策略 (开发/联网模式):
      1. Cloud       — DeepSeek / Qwen 等 OpenAI 兼容 API
      2-4. 同上

    API Key 从环境变量读取, 永远不要硬编码
    """
    # ---- 竞赛模式 (跳过 Cloud, 纯离线) ----
    competition_mode: bool = True           # 竞赛环境无网络, 跳过所有 Cloud 尝试

    use_cloud: bool = True                  # competition_mode=True 时自动禁用
    cloud_provider: str = "deepseek"        # 提供商

    # 云端 API 参数
    cloud_api_key: str = ""
    cloud_base_url: str = "https://api.deepseek.com"
    cloud_model_name: str = "deepseek-chat"

    # 本地模型参数
    local_model_path: str = ""
    local_device: str = "NPU"              # 首选设备: "NPU", "GPU", "CPU"
    local_device_fallback: tuple = ("NPU", "GPU", "CPU")  # 降级链
    max_tokens: int = 300
    temperature: float = 0.7

    # NPU 优化参数 (针对 Intel Core Ultra 5 225U)
    npu_warm_up: bool = True               # 启动时预热 NPU (避免首次推理卡顿)
    npu_cache_dir: str = ".npucache"       # NPU 编译缓存目录 (加速二次启动)

    # 本地模型搜索路径 (按优先级)
    local_model_search_names: tuple = (
        "qwen2.5_1.5b_ov",         # Qwen2.5-1.5B-Instruct INT4 (首选)
        "minicpm_1b_ov",            # MiniCPM-1B-sft INT4
        "phi3_mini_ov",             # Phi-3-mini-4k-instruct INT4
        "tinyllama_openvino",       # TinyLlama (旧兜底, 中文差)
    )

    def __post_init__(self):
        # 竞赛模式强制禁用 Cloud
        if self.competition_mode:
            self.use_cloud = False

        # API Key 优先从环境变量读取
        self.cloud_api_key = os.environ.get("LG_API_KEY",
                             os.environ.get("DEEPSEEK_API_KEY", self.cloud_api_key))
        self.cloud_base_url = os.environ.get("LG_API_BASE_URL", self.cloud_base_url)
        self.cloud_model_name = os.environ.get("LG_API_MODEL", self.cloud_model_name)
        self.local_device = os.environ.get("LG_LLM_DEVICE", self.local_device)
        self.competition_mode = os.environ.get(
            "LG_COMPETITION_MODE", str(self.competition_mode)
        ).lower() in ("true", "1", "yes")

        # 自动搜索最佳本地模型
        self.local_model_path = self._find_best_local_model()

    def _find_best_local_model(self) -> str:
        """按优先级搜索本地模型目录"""
        for name in self.local_model_search_names:
            candidate = MODELS_DIR / name
            if candidate.exists() and (candidate / "openvino_model.xml").exists():
                return str(candidate)
            # 也检查 .bin 文件 (旧格式)
            if candidate.exists() and any(candidate.glob("*.bin")):
                return str(candidate)
        # 回退到默认路径
        return str(LOCAL_LLM_MODEL_DIR)

    @property
    def is_cloud_ready(self) -> bool:
        return bool(self.cloud_api_key and self.cloud_api_key.startswith("sk-"))


# ============================================================
# OCR 识别配置
# ============================================================

@dataclass
class OCRConfig:
    """
    OCR 芯片丝印识别配置

    降级策略: PaddleOCR → EasyOCR → 无 OCR
    """
    enabled: bool = True                   # 是否启用 OCR
    backend: str = "auto"                  # "auto" | "paddle" | "easyocr"
    lang: str = "en"                       # 识别语言 (芯片丝印主要是英文+数字)
    preprocess: bool = True                # 是否预处理增强
    crop_padding: float = 0.05             # 裁剪填充比例
    min_confidence: float = 0.3            # OCR 最低置信度
    # 触发 OCR 的 YOLO 检测类别 (检测到这些类别时自动 OCR)
    target_classes: tuple = (
        "IC", "CHIP", "DIP", "TRANSISTOR", "NPN", "PNP", "BJT",
        "NE555", "LM358", "OPAMP", "REGULATOR",
    )


# ============================================================
# GUI 配置
# ============================================================

@dataclass
class GUIConfig:
    """界面参数"""
    window_title: str = "LabGuardian — Intel Embedded AI Assistant"
    window_size: tuple = (1400, 900)
    video_max_display: tuple = (1000, 700)  # 视频区最大显示尺寸
    font_family: str = "Consolas"
    font_size: int = 10


# ============================================================
# 课堂模式配置
# ============================================================

@dataclass
class ClassroomConfig:
    """
    课堂监控配置 (学生端 ↔ 教师端)

    启用方式:
      命令行: python launcher.py --classroom
      环境变量: LG_CLASSROOM=true

    角色:
      - Hub (is_hub=True): 本机同时运行教师服务器 + 学生端
      - Client (is_hub=False): 本机仅运行学生端, 心跳上报到 Hub
    """
    enabled: bool = False                           # 是否启用课堂模式
    station_id: str = ""                            # 工位编号 (空=自动生成)
    student_name: str = ""                          # 学生姓名 (可选)
    server_url: str = "http://localhost:8080"       # 教师服务器地址
    server_port: int = 8080                         # 教师服务器端口 (Hub 模式)
    server_host: str = "0.0.0.0"                    # 教师服务器监听地址
    is_hub: bool = False                            # 本机是否运行教师服务器
    heartbeat_interval: float = 2.0                 # 心跳间隔 (秒)
    thumbnail_quality: int = 70                     # 缩略图 JPEG 质量
    thumbnail_size: tuple = (160, 120)              # 缩略图分辨率

    def __post_init__(self):
        self.enabled = os.environ.get(
            "LG_CLASSROOM", str(self.enabled)
        ).lower() in ("true", "1", "yes")
        self.station_id = os.environ.get("LG_STATION_ID", self.station_id)
        self.student_name = os.environ.get("LG_STUDENT_NAME", self.student_name)
        self.server_url = os.environ.get("LG_SERVER_URL", self.server_url)
        self.server_port = int(os.environ.get("LG_SERVER_PORT", self.server_port))
        self.is_hub = os.environ.get(
            "LG_IS_HUB", str(self.is_hub)
        ).lower() in ("true", "1", "yes")

        # 自动生成 station_id
        if not self.station_id:
            import socket
            self.station_id = socket.gethostname()[:12]


# ============================================================
# 检测类别定义
# ============================================================

# 类别列表 (与训练 data.yaml 中的顺序一致)
COMPONENT_CLASSES = [
    "CAPACITOR",
    "DIODE", 
    "LED",
    "RESISTOR",
    "Push_Button",
    "Wire",
]

# 类别颜色 (BGR)
CLASS_COLORS = {
    "CAPACITOR":   (0, 0, 255),
    "DIODE":       (0, 255, 0),
    "LED":         (255, 0, 0),
    "RESISTOR":    (0, 255, 255),
    "Push_Button": (255, 255, 0),
    "Wire":        (255, 0, 255),
}

# 两端元件 (需要提取两个引脚坐标)
TWO_PIN_COMPONENTS = {"Wire", "RESISTOR", "DIODE", "Resistor", "LED", "CAPACITOR"}

# 三端元件 (需要推断第三引脚)
THREE_PIN_COMPONENTS = {"TRANSISTOR", "NPN", "PNP", "BJT"}


# ============================================================
# 电路逻辑配置 (极性/引脚/拓扑)
# ============================================================

@dataclass
class CircuitConfig:
    """电路拓扑分析 & 极性推断参数"""

    # --- 极性推断 ---
    enable_polarity: bool = True           # 是否启用极性推断
    polarity_conf_threshold: float = 0.4   # 极性推断所需的最低检测置信度

    # --- 面包板电源轨 ---
    power_rail_rows: tuple = (1, 2, 29, 30)   # 被视为电源轨的行号
    vcc_rail_rows: tuple = (1, 2)              # VCC 轨行号 (红色)
    gnd_rail_rows: tuple = (29, 30)            # GND 轨行号 (蓝色)

    # --- 三极管 (TO-92) 引脚映射 ---
    # TO-92 封装: 从平面侧看, 左→右 = E/B/C
    # 面包板上: 行号从小到大 ↔ 物理从左到右
    transistor_row_span_min: int = 2       # 三极管最小跨行数 (3行跨2)
    transistor_default_pinout: str = "EBC" # 默认引脚排列 (可选: "EBC", "BCE", "CBE")

    # --- 验证比较 ---
    topology_match_check_polarity: bool = True  # VF2++ 匹配时是否检查极性
    position_tolerance_rows: int = 2       # 位置启发式匹配的行容差

    # --- 引脚遮挡补偿 ---
    pin_candidate_k: int = 3              # 每个引脚返回的候选孔洞数量
    pin_same_group_penalty: float = 100.0  # 两引脚在同一导通组的惩罚分
    pin_same_row_penalty: float = 50.0     # 非Wire元件两引脚在同一行的惩罚分
    pin_large_span_threshold: int = 10     # 行跨度 > 此值视为异常

    # --- 网表导出 ---
    netlist_include_confidence: bool = True  # 网表中是否包含置信度
    netlist_format: str = "json"            # 导出格式: "json" | "spice_like"


# ============================================================
# 全局配置实例 (单例)
# ============================================================

vision = VisionConfig()
calibration = CalibrationConfig()
camera = CameraConfig()
llm = LLMConfig()
ocr = OCRConfig()
gui = GUIConfig()
circuit = CircuitConfig()
rag = RAGConfig()
classroom = ClassroomConfig()


# ============================================================
# 辅助函数
# ============================================================

def find_best_yolo_model() -> Path:
    """
    自动搜索最佳 YOLO 模型权重文件
    优先级: 环境变量 > runs/ 下自训练模型 > models/ 下 .pt 文件
    """
    import glob as _glob

    # 0. 环境变量指定路径
    env_path = os.environ.get("LG_MODEL_PATH")
    if env_path and Path(env_path).exists():
        print(f"[Config] 使用环境变量指定的模型: {env_path}")
        return Path(env_path)

    # 1. 搜索 runs/ 下的自训练模型
    candidates = []
    for search_dir in MODEL_SEARCH_DIRS:
        pattern = str(search_dir / f"{vision.model_name_hint}*" / "weights" / "best.pt")
        candidates.extend(_glob.glob(pattern))

    if not candidates:
        # 2. 搜索 models/ 目录下的 .pt 文件
        if MODELS_DIR.exists():
            pt_files = list(MODELS_DIR.glob("*.pt"))
            if pt_files:
                # 优先选 obb 模型
                obb = [f for f in pt_files if "obb" in f.name]
                chosen = obb[0] if obb else pt_files[0]
                print(f"[Config] 使用 models/ 目录下的模型: {chosen}")
                return chosen
        print(f"[Config] 未找到任何模型，回退到默认: {DEFAULT_YOLO_WEIGHTS}")
        return DEFAULT_YOLO_WEIGHTS

    # 1. 优先：精确匹配 preferred_model
    for c in candidates:
        if vision.preferred_model in c:
            print(f"[Config] 找到指定模型: {c}")
            return Path(c)

    # 2. 次选：包含 "oneshot" 的最新模型
    oneshot = [c for c in candidates if "oneshot" in c]
    if oneshot:
        best = max(oneshot, key=os.path.getmtime)
        print(f"[Config] 找到 OneShot 模型: {best}")
        return Path(best)

    # 3. 回退：最新的模型
    best = max(candidates, key=os.path.getmtime)
    print(f"[Config] 使用最新模型: {best}")
    return Path(best)


def print_config_summary():
    """打印当前配置摘要（调试用）"""
    print("=" * 50)
    print("LabGuardian Configuration Summary")
    print("=" * 50)
    print(f"  Platform:    {sys.platform}")
    print(f"  Project Root:{PROJECT_ROOT}")
    print(f"  Camera:      device={camera.device_id}, {camera.width}x{camera.height}, backend={camera.backend}")
    print(f"  Vision:      conf={vision.conf_threshold}, imgsz={vision.imgsz}")
    print(f"  LLM:         cloud={llm.use_cloud}, ready={llm.is_cloud_ready}")
    print(f"  Calibration: rows={calibration.rows}, output={calibration.output_size}")
    if classroom.enabled:
        print(f"  Classroom:   station={classroom.station_id}, hub={classroom.is_hub}, server={classroom.server_url}")
    print("=" * 50)
