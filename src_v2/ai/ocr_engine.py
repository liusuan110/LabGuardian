"""
OCR 文字识别引擎模块
职责：识别芯片表面丝印文字（如 "NE555", "S8050", "LM358"）

技术栈:
  - 首选: PaddleOCR (OpenVINO 后端, 支持 Intel NPU/iGPU 加速)
  - 回退: EasyOCR (纯 PyTorch, 通用性好)
  - 兜底: 无 OCR (返回空结果, 不影响主流程)

工作流程:
  1. YOLO 检测到 IC/芯片类元件 → 获取 bbox 区域
  2. 裁剪 + 预处理 (灰度/增强/二值化)
  3. OCR 识别丝印文字 → 返回型号字符串
  4. (可选) 将型号传给 RAG 知识库查询芯片信息

使用方式:
    from ai.ocr_engine import OCREngine
    ocr = OCREngine()
    ocr.initialize()
    result = ocr.recognize_chip(frame, bbox=(x1, y1, x2, y2))
    print(result.text)  # "NE555"

支持的芯片类别 (触发 OCR 的 YOLO 类别):
  - IC, CHIP, DIP, NE555, LM358, TRANSISTOR, NPN, PNP 等
  - 任何新增的芯片类别只需加入 OCR_TARGET_CLASSES 即可
"""

import logging
import re
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class OCRResult:
    """单次 OCR 识别结果"""
    text: str                                # 识别出的完整文本
    chip_model: str = ""                     # 解析出的芯片型号 (如 "NE555")
    confidence: float = 0.0                  # OCR 置信度 (0~1)
    bbox_in_crop: Optional[Tuple] = None     # 文字在裁剪图中的位置
    raw_results: Optional[list] = None       # OCR 引擎原始返回

    @property
    def has_model(self) -> bool:
        """是否成功解析出芯片型号"""
        return bool(self.chip_model)


# ============================================================
# 芯片型号解析器
# ============================================================

# 常见芯片型号正则 (匹配 NE555, LM358, S8050, TL072, UA741, 2N3904 等)
CHIP_MODEL_PATTERNS = [
    # NE555, LM358, LM741, UA741, TL072, TL074, NE5532
    re.compile(r'\b(NE|LM|UA|TL|AD|OP|MC|SN|CD|HEF|TC|MAX|MCP)\s*\d{3,5}[A-Z]?\b', re.IGNORECASE),
    # S8050, S8550, SS8050, SS8550
    re.compile(r'\b(SS?)\s*(\d{4})\b', re.IGNORECASE),
    # 2N3904, 2N3906, 2N2222, BC547, BC557
    re.compile(r'\b(2N|BC|BD|BF|2SC|2SA|2SD)\s*\d{3,5}[A-Z]?\b', re.IGNORECASE),
    # 78xx / 79xx 稳压芯片 (7805, 7812, 7905)
    re.compile(r'\b(78|79)\d{2}\b'),
    # ATmega, ATtiny 系列
    re.compile(r'\b(AT)\s*(mega|tiny)\s*\d+[A-Z]?\b', re.IGNORECASE),
    # 74HC 系列逻辑芯片
    re.compile(r'\b(74|54)\s*(HC|LS|HCT|AC|ALS|F)\s*\d{2,4}\b', re.IGNORECASE),
    # STM32 系列
    re.compile(r'\bSTM32[A-Z]\d{3}[A-Z]{2}\b', re.IGNORECASE),
    # 通用: 2-3字母 + 3-5数字 (兜底)
    re.compile(r'\b[A-Z]{1,3}\d{3,5}[A-Z]?\b'),
]


def parse_chip_model(text: str) -> str:
    """
    从 OCR 文本中解析芯片型号

    Args:
        text: OCR 识别的原始文本 (可能包含噪声)

    Returns:
        最可能的芯片型号字符串, 如 "NE555"; 未识别到返回 ""
    """
    if not text:
        return ""

    # 清理: 去掉多余空格, 统一大写
    cleaned = " ".join(text.split()).upper()
    # variant1: O→0, I→1 (常见 OCR 混淆)
    variant1 = cleaned.replace("O", "0").replace("I", "1").replace("l", "1")
    # variant2: 额外将数字上下文中的 S→5, B→8, G→6 (OCR 对数字/字母混淆)
    variant2 = _fix_digit_letter_confusion(cleaned)
    # 但保留原始文本也尝试匹配
    candidates = []

    for pattern in CHIP_MODEL_PATTERNS:
        for t in [text.upper(), variant1, variant2]:
            # 使用 finditer 获取完整匹配 (非仅捕获组)
            for m in pattern.finditer(t):
                candidate = m.group(0).strip().replace(" ", "")
                if len(candidate) >= 3:
                    candidates.append(candidate)

    if not candidates:
        # 最后尝试: 直接对原始文本做 typo 表查询
        # (如 "NES55" → NE555 在 typo 表中但没被正则匹配到)
        direct_fix = _fix_common_typos(text.upper().replace(" ", ""))
        if direct_fix != text.upper().replace(" ", ""):
            return direct_fix
        return ""

    # 优先选最长的匹配 (通常更精确)
    best = max(candidates, key=len)
    # 修正: 如果 O/0 和 I/1 被误替换, 对常见型号做映射
    best = _fix_common_typos(best)
    return best


def _fix_common_typos(model: str) -> str:
    """修正 OCR 常见识别错误"""
    fixes = {
        "NE555": ["NE5S5", "NES55", "NE5SS", "NE55S", "NESS5", "NE5S"],
        "LM358": ["LM3S8", "LM35B", "LM3SB", "LM358N"],
        "LM741": ["LM74I", "LM74L"],
        "S8050": ["S80S0", "SB050", "58050", "S805O", "S8O5O"],
        "S8550": ["S85S0", "SB550", "58550", "S855O", "S8S5O"],
        "SS8050": ["SS80S0", "5S8050", "SS805O"],
        "SS8550": ["SS85S0", "5S8550", "SS855O"],
        "TL072": ["TL0T2", "TLO72", "TL0T2"],
        "TL074": ["TL0T4", "TLO74"],
        "NE5532": ["NES532", "NE55S2", "NE553Z", "NESS32"],
        "UA741": ["UA74I", "UA74L"],
        "2N3904": ["ZN3904", "2N39O4"],
        "2N3906": ["ZN3906", "2N39O6"],
        "7805": ["T805", "78O5"],
        "7812": ["T812", "78I2"],
    }
    model_upper = model.upper()
    for correct, typos in fixes.items():
        if model_upper in typos or model_upper == correct:
            return correct
    return model


def _fix_digit_letter_confusion(text: str) -> str:
    """
    在数字上下文中修正 OCR 字母/数字混淆

    规则: 当一个字母夹在数字之间或紧跟数字后方时, 尝试替换为对应数字
    S→5, B→8, G→6, Z→2, O→0, I→1, D→0, Q→0, T→7
    """
    # 这些替换只在 "字母前后都有数字" 或 "前缀字母之后" 的上下文中进行
    # 简单策略: 先用已知前缀分割, 对数字部分做替换
    LETTER_TO_DIGIT = str.maketrans("OISBGZDQT", "015860207")

    # 尝试分离前缀 (1-3 个字母) + 后续 (混合数字/字母)
    m = re.match(r'^([A-Z]{1,3})(.+)$', text)
    if m:
        prefix = m.group(1)
        suffix = m.group(2)
        # 对 suffix 中的字母做数字替换
        fixed_suffix = suffix.translate(LETTER_TO_DIGIT)
        return prefix + fixed_suffix

    return text


# ============================================================
# 图像预处理
# ============================================================

def preprocess_chip_crop(image: np.ndarray, target_height: int = 64) -> np.ndarray:
    """
    对裁剪出的芯片区域做 OCR 预处理

    步骤:
      1. 转灰度
      2. CLAHE 对比度增强
      3. 自适应高斯二值化
      4. 缩放到目标高度 (保持宽高比)

    Args:
        image: BGR 裁剪图
        target_height: 输出图像高度 (px)

    Returns:
        预处理后的灰度图 (适合 OCR 输入)
    """
    if image is None or image.size == 0:
        return image

    # 灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # 自适应二值化
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 5
    )

    # 缩放 (保持宽高比)
    h, w = binary.shape[:2]
    if h > 0:
        scale = target_height / h
        new_w = max(int(w * scale), 1)
        binary = cv2.resize(binary, (new_w, target_height), interpolation=cv2.INTER_LINEAR)

    return binary


def crop_detection_region(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          padding: float = 0.1) -> np.ndarray:
    """
    从帧中裁剪检测区域, 带少量填充

    Args:
        frame: 完整帧 (BGR)
        bbox: (x1, y1, x2, y2)
        padding: 填充比例 (相对于 bbox 尺寸)

    Returns:
        裁剪出的区域 (BGR)
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    # 加 padding
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return frame[y1:y2, x1:x2].copy()


# ============================================================
# OCR 后端: PaddleOCR
# ============================================================

class PaddleOCRBackend:
    """
    PaddleOCR 后端 (v3.4.0 新 API)

    支持 OpenVINO 加速推理:
      - 检测模型 (det): 文字区域检测
      - 识别模型 (rec): 文字内容识别
      - 方向分类 (textline_orientation): 文字方向分类

    安装: pip install paddlepaddle==3.2.2 paddleocr>=3.4.0

    ⚠ 已知问题 (Paddle#77340, PaddleOCR#17539):
      PaddlePaddle 3.3.0 的 PIR (Program Intermediate Representation) 执行器
      在将 pir::ArrayAttribute<pir::DoubleAttribute> 转换为运行时属性时缺少实现,
      当 CPU 推理启用 oneDNN 加速时触发 NotImplementedError.
      修复方案: 将 paddlepaddle 降级到 3.2.2 (社区多人验证有效)
    """

    def __init__(self):
        self._ocr = None
        self._ready = False

    def load(self) -> bool:
        """加载 PaddleOCR 模型 (PaddleOCR v3.4+ predict API)"""
        try:
            import os
            # 跳过模型源连接检查, 加快启动
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR

            # PaddleOCR 3.4+ 新 API: 参数名全部重构
            # - use_doc_orientation_classify: 文档方向分类 (芯片不需要)
            # - use_doc_unwarping: 文档去畸变 (芯片不需要)
            # - use_textline_orientation: 文字行方向 (丝印可能旋转, 启用)
            self._ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                text_det_thresh=0.3,
                text_rec_score_thresh=0.3,
            )

            # 用小图做一次推理验证 (验证 Paddle 推理引擎正常工作)
            test_img = np.zeros((50, 150, 3), dtype=np.uint8)
            _ = list(self._ocr.predict(test_img))

            self._ready = True
            logger.info("[OCR-Paddle] ✅ PaddleOCR 加载成功")
            print("[OCR-Paddle] ✅ PaddleOCR 加载成功 (v3.4+ predict API)")
            return True
        except ImportError:
            logger.info("[OCR-Paddle] PaddlePaddle/PaddleOCR 未安装")
            return False
        except Exception as e:
            logger.warning(f"[OCR-Paddle] 加载失败: {e}")
            print(f"[OCR-Paddle] ❌ 加载失败: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def recognize(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        对图像执行 OCR (PaddleOCR 3.4+ predict API)

        Args:
            image: BGR 或灰度图

        Returns:
            [(text, confidence), ...] 按置信度降序
        """
        if not self._ready:
            return []
        try:
            results = list(self._ocr.predict(image))
            texts = []
            for res in results:
                # PaddleOCR 3.4+ 返回 OCRResult (dict 子类)
                # 包含 rec_texts, rec_scores 等 key
                rec_texts = None
                rec_scores = None

                if isinstance(res, dict):
                    rec_texts = res.get('rec_texts')
                    rec_scores = res.get('rec_scores')
                elif hasattr(res, 'rec_texts') and hasattr(res, 'rec_scores'):
                    rec_texts = res.rec_texts
                    rec_scores = res.rec_scores

                if rec_texts and rec_scores:
                    for txt, score in zip(rec_texts, rec_scores):
                        txt = str(txt).strip()
                        if txt:
                            texts.append((txt, float(score)))

            texts.sort(key=lambda x: x[1], reverse=True)
            return texts
        except Exception as e:
            logger.warning(f"[OCR-Paddle] 识别错误: {e}")
            return []


# ============================================================
# OCR 后端: EasyOCR (回退)
# ============================================================

class EasyOCRBackend:
    """
    EasyOCR 后端 (PyTorch 原生)

    优势: 安装简单, 无需 PaddlePaddle
    劣势: 首次加载稍慢, 不支持 NPU 加速

    安装: pip install easyocr
    """

    def __init__(self):
        self._reader = None
        self._ready = False

    def load(self) -> bool:
        try:
            import easyocr
            self._reader = easyocr.Reader(
                ["en"],                     # 英文 (芯片丝印)
                gpu=False,
                verbose=False,
            )
            self._ready = True
            logger.info("[OCR-Easy] ✅ EasyOCR 加载成功")
            print("[OCR-Easy] ✅ EasyOCR 加载成功 (lang=en)")
            return True
        except ImportError:
            logger.info("[OCR-Easy] EasyOCR 未安装")
            return False
        except Exception as e:
            logger.warning(f"[OCR-Easy] 加载失败: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def recognize(self, image: np.ndarray) -> List[Tuple[str, float]]:
        if not self._ready:
            return []
        try:
            results = self._reader.readtext(image)
            texts = []
            for (bbox, text, conf) in results:
                text = text.strip()
                if text:
                    texts.append((text, conf))
            texts.sort(key=lambda x: x[1], reverse=True)
            return texts
        except Exception as e:
            logger.warning(f"[OCR-Easy] 识别错误: {e}")
            return []


# ============================================================
# 统一 OCR 引擎
# ============================================================

# 触发 OCR 的 YOLO 检测类别 (检测到这些类别时自动裁剪+OCR)
OCR_TARGET_CLASSES = {
    "IC", "CHIP", "DIP", "TRANSISTOR", "NPN", "PNP", "BJT",
    "NE555", "LM358", "OPAMP", "REGULATOR",
    # 当 YOLO 类别列表扩展时, 在此添加芯片相关类别即可
}


class OCREngine:
    """
    统一 OCR 引擎

    降级策略:
      1. PaddleOCR (首选, 精度高, 支持 OpenVINO)
      2. EasyOCR (回退, 安装简单)
      3. 无 OCR (兜底, 返回空结果, 不阻塞主流程)

    使用方式:
        ocr = OCREngine()
        ocr.initialize()

        # 方式 1: 直接对裁剪图识别
        result = ocr.recognize_chip(frame, bbox=(100, 200, 300, 400))

        # 方式 2: 对 Detection 列表批量识别
        results = ocr.recognize_from_detections(frame, detections)
    """

    def __init__(self):
        self._paddle = PaddleOCRBackend()
        self._easy = EasyOCRBackend()
        self._active = None
        self._backend_name = "None"

    def initialize(self) -> bool:
        """
        初始化 OCR 引擎, 按优先级尝试后端

        Returns:
            True if at least one backend loaded
        """
        # ⚠ Windows 上 PaddlePaddle 和 PyTorch 有 DLL 冲突:
        #   如果 paddle 先加载，torch 的 shm.dll 会找不到依赖。
        #   必须确保 torch 先导入。(EasyOCR 依赖 torch)
        try:
            import torch  # noqa: F401 — 确保 torch DLL 先加载
        except ImportError:
            pass

        # 1. PaddleOCR
        if self._paddle.load():
            self._active = self._paddle
            self._backend_name = "PaddleOCR"
            return True

        # 2. EasyOCR 回退
        if self._easy.load():
            self._active = self._easy
            self._backend_name = "EasyOCR"
            return True

        # 3. 无 OCR
        logger.warning("[OCR] ❌ 无可用 OCR 后端. 安装: pip install paddleocr 或 pip install easyocr")
        print("[OCR] ❌ 无 OCR 后端. 安装: pip install paddleocr 或 pip install easyocr")
        self._backend_name = "None"
        return False

    @property
    def is_ready(self) -> bool:
        return self._active is not None and self._active.is_ready

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def recognize_chip(self, frame: np.ndarray,
                       bbox: Tuple[int, int, int, int],
                       preprocess: bool = True) -> OCRResult:
        """
        识别单个芯片区域的丝印文字

        Args:
            frame: 完整帧 (BGR)
            bbox: 芯片检测框 (x1, y1, x2, y2)
            preprocess: 是否进行预处理增强

        Returns:
            OCRResult 包含识别文本和解析出的型号
        """
        if not self.is_ready:
            return OCRResult(text="", confidence=0.0)

        # 1. 裁剪
        crop = crop_detection_region(frame, bbox, padding=0.05)
        if crop.size == 0:
            return OCRResult(text="", confidence=0.0)

        # 2. 识别 (先用原图, 如果失败再用预处理图)
        raw_results = self._active.recognize(crop)

        if not raw_results and preprocess:
            # 预处理后重试
            processed = preprocess_chip_crop(crop)
            # 转回 3 通道 (OCR 引擎可能需要)
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            raw_results = self._active.recognize(processed)

        if not raw_results:
            return OCRResult(text="", confidence=0.0, raw_results=[])

        # 3. 合并所有识别文本
        all_text = " ".join([t for t, c in raw_results])
        avg_conf = sum(c for _, c in raw_results) / len(raw_results)

        # 4. 解析芯片型号
        chip_model = parse_chip_model(all_text)

        return OCRResult(
            text=all_text,
            chip_model=chip_model,
            confidence=avg_conf,
            raw_results=raw_results,
        )

    def recognize_from_detections(self, frame: np.ndarray,
                                  detections: list,
                                  target_classes: set = None) -> dict:
        """
        对检测结果列表中的芯片类元件批量执行 OCR

        Args:
            frame: 完整帧 (BGR)
            detections: List[Detection] 来自 YOLO 检测器
            target_classes: 需要 OCR 的类别集合 (None 则使用默认)

        Returns:
            {detection_index: OCRResult, ...}
            只包含触发了 OCR 的检测结果
        """
        if not self.is_ready:
            return {}

        targets = target_classes or OCR_TARGET_CLASSES
        results = {}

        for i, det in enumerate(detections):
            # 检查是否为芯片类元件
            if det.class_name.upper() in {c.upper() for c in targets}:
                ocr_result = self.recognize_chip(frame, det.bbox)
                if ocr_result.text:
                    results[i] = ocr_result
                    logger.info(
                        f"[OCR] {det.class_name} → \"{ocr_result.text}\" "
                        f"(型号: {ocr_result.chip_model or '未识别'}, "
                        f"置信度: {ocr_result.confidence:.2f})"
                    )

        return results

    def recognize_image(self, image: np.ndarray) -> OCRResult:
        """
        对任意图像直接 OCR (不裁剪)

        用于调试或手动传入已裁剪的芯片图片
        """
        if not self.is_ready:
            return OCRResult(text="", confidence=0.0)

        raw_results = self._active.recognize(image)
        if not raw_results:
            return OCRResult(text="", confidence=0.0, raw_results=[])

        all_text = " ".join([t for t, c in raw_results])
        avg_conf = sum(c for _, c in raw_results) / len(raw_results)
        chip_model = parse_chip_model(all_text)

        return OCRResult(
            text=all_text,
            chip_model=chip_model,
            confidence=avg_conf,
            raw_results=raw_results,
        )
