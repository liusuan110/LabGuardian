from .llm_engine import LLMEngine, RuleBasedBackend

try:
    from .rag_engine import RAGEngine
except ImportError:
    RAGEngine = None  # chromadb / sentence-transformers 未安装时优雅降级

try:
    from .ocr_engine import OCREngine, OCRResult, parse_chip_model
except ImportError:
    OCREngine = None  # paddleocr / easyocr 未安装时优雅降级
    OCRResult = None
    parse_chip_model = None
