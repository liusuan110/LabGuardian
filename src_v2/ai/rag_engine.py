"""
RAG 检索增强生成引擎 (v1)
职责：构建本地向量知识库，支持芯片手册/模拟电路实验离线问答

技术栈:
  - Embedding: sentence-transformers (all-MiniLM-L6-v2 或 text2vec-base-chinese)
  - Vector Store: ChromaDB (本地持久化, 零服务器依赖)
  - 文档加载: 支持 .txt / .md / .pdf

使用方式:
    from ai.rag_engine import RAGEngine
    rag = RAGEngine()
    rag.build_index()                       # 首次构建
    context = rag.query("8050三极管放大倍数")  # 检索相关片段
    # 将 context 拼接给 LLM 作为增强上下文

知识库目录结构:
    knowledge_base/
    ├── builtin/          # 内置知识 (随代码分发, .md/.txt)
    │   ├── transistor_8050.md
    │   ├── transistor_8550.md
    │   ├── op_amp_basics.md
    │   └── analog_lab_guide.md
    ├── user_docs/        # 用户自添 PDF/文档 (放入即可)
    └── chroma_db/        # ChromaDB 持久化目录 (自动生成)
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ============================================================
# 路径常量
# ============================================================

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
BUILTIN_DIR = KNOWLEDGE_BASE_DIR / "builtin"
USER_DOCS_DIR = KNOWLEDGE_BASE_DIR / "user_docs"
CHROMA_DB_DIR = KNOWLEDGE_BASE_DIR / "chroma_db"

# Embedding 模型路径 (本地缓存)
EMBEDDING_MODEL_DIR = PROJECT_ROOT / "models" / "text2vec_chinese"

# 默认 Embedding 模型名 (若本地无缓存, 从 HuggingFace 下载)
DEFAULT_EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 分块参数
DEFAULT_CHUNK_SIZE = 400       # 每个文本块的字符数
DEFAULT_CHUNK_OVERLAP = 80     # 块之间重叠字符数


class DocumentChunker:
    """
    文档分块器
    将长文档切分为适合 Embedding 的小块, 保留上下文重叠
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: dict = None) -> List[dict]:
        """
        将纯文本切分为块

        Returns:
            [{"text": "...", "metadata": {...}}, ...]
        """
        if not text.strip():
            return []

        chunks = []
        # 先按段落分割, 再按 chunk_size 合并
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        current_chunk = ""
        for para in paragraphs:
            # 如果单段太长, 强制按字符切分
            if len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                for i in range(0, len(para), self.chunk_size - self.chunk_overlap):
                    sub = para[i:i + self.chunk_size]
                    chunks.append(sub)
                continue

            if len(current_chunk) + len(para) + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                # 保留重叠
                overlap_text = current_chunk[-(self.chunk_overlap):] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap_text + "\n" + para if overlap_text else para
            else:
                current_chunk = current_chunk + "\n" + para if current_chunk else para

        if current_chunk.strip():
            chunks.append(current_chunk)

        # 附加 metadata
        base_meta = metadata or {}
        result = []
        for i, chunk in enumerate(chunks):
            meta = {**base_meta, "chunk_index": i}
            result.append({"text": chunk.strip(), "metadata": meta})

        return result

    def load_and_chunk_file(self, filepath: Path) -> List[dict]:
        """
        加载单个文件并分块

        支持: .txt, .md, .pdf
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"文件不存在: {filepath}")
            return []

        metadata = {
            "source": filepath.name,
            "source_path": str(filepath),
            "file_type": filepath.suffix.lower(),
        }

        text = ""
        suffix = filepath.suffix.lower()

        if suffix in (".txt", ".md"):
            text = filepath.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            text = self._extract_pdf_text(filepath)
        else:
            logger.warning(f"不支持的文件类型: {suffix} ({filepath.name})")
            return []

        if not text.strip():
            logger.warning(f"文件内容为空: {filepath.name}")
            return []

        logger.info(f"[RAG] 加载文档: {filepath.name} ({len(text)} chars)")
        return self.chunk_text(text, metadata)

    @staticmethod
    def _extract_pdf_text(filepath: Path) -> str:
        """从 PDF 提取纯文本 (需要 PyMuPDF 或 pdfplumber)"""
        # 优先尝试 PyMuPDF (fitz)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(filepath))
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except ImportError:
            pass

        # 回退: pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(filepath)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except ImportError:
            logger.warning(
                "[RAG] 无法读取 PDF: 请安装 PyMuPDF (pip install pymupdf) "
                "或 pdfplumber (pip install pdfplumber)"
            )
            return ""


class RAGEngine:
    """
    RAG 检索增强生成引擎

    核心流程:
      1. build_index() — 扫描 knowledge_base/ 目录, 分块 → Embedding → 存入 ChromaDB
      2. query()       — 用户问题 → Embedding → 在 ChromaDB 中检索 Top-K 相关块
      3. get_context() — 将检索结果拼接为 LLM 可用的上下文字符串

    特性:
      - 增量更新: 基于文件哈希, 只处理新增/修改的文件
      - 双目录: builtin/ (内置知识) + user_docs/ (用户自添)
      - 优雅降级: ChromaDB 不可用时返回空上下文, 不影响主流程
    """

    def __init__(self, persist_dir: Path = None, embedding_model: str = None):
        self._persist_dir = persist_dir or CHROMA_DB_DIR
        self._embedding_model_name = embedding_model or DEFAULT_EMBEDDING_MODEL
        self._collection = None  # ChromaDB collection
        self._client = None      # ChromaDB client
        self._embed_fn = None    # Embedding function
        self._chunker = DocumentChunker()
        self._ready = False

    # ================================================================
    # 初始化
    # ================================================================

    def initialize(self) -> bool:
        """
        初始化 RAG 引擎: 加载 Embedding 模型 + 连接 ChromaDB

        Returns:
            True if ready, False if failed (will degrade gracefully)
        """
        try:
            self._embed_fn = self._load_embedding_function()
            self._client, self._collection = self._connect_chromadb()
            self._ready = True
            doc_count = self._collection.count()
            logger.info(f"[RAG] ✅ 初始化成功 | 知识库文档数: {doc_count}")
            print(f"[RAG] ✅ 初始化成功 | 知识库已有 {doc_count} 个文本块")
            return True
        except Exception as e:
            logger.error(f"[RAG] ❌ 初始化失败: {e}")
            print(f"[RAG] ❌ 初始化失败: {e}")
            self._ready = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def doc_count(self) -> int:
        if self._collection:
            return self._collection.count()
        return 0

    # ================================================================
    # Embedding 模型加载
    # ================================================================

    def _load_embedding_function(self):
        """
        加载 Embedding 模型, 返回 ChromaDB 兼容的 embedding function

        优先级:
          1. 本地缓存目录 (models/text2vec_chinese/)
          2. HuggingFace text2vec-base-chinese (中文最优)
          3. all-MiniLM-L6-v2 (英文回退, 体积小)
        """
        from chromadb.utils import embedding_functions

        # 尝试 1: 本地缓存的中文模型
        if EMBEDDING_MODEL_DIR.exists():
            logger.info(f"[RAG] 使用本地 Embedding 模型: {EMBEDDING_MODEL_DIR}")
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=str(EMBEDDING_MODEL_DIR)
            )

        # 尝试 2: 从 HuggingFace 加载中文模型
        try:
            logger.info(f"[RAG] 加载 Embedding 模型: {self._embedding_model_name}")
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model_name
            )
            # 验证可用
            _ = ef(["测试"])
            logger.info("[RAG] 中文 Embedding 模型加载成功")
            return ef
        except Exception as e:
            logger.warning(f"[RAG] 中文模型加载失败: {e}, 尝试英文回退模型")

        # 尝试 3: 英文回退模型
        logger.info(f"[RAG] 使用回退 Embedding 模型: {FALLBACK_EMBEDDING_MODEL}")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=FALLBACK_EMBEDDING_MODEL
        )

    # ================================================================
    # ChromaDB 连接
    # ================================================================

    def _connect_chromadb(self):
        """连接 ChromaDB (本地持久化模式)"""
        import chromadb

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self._persist_dir))

        collection = client.get_or_create_collection(
            name="labguardian_knowledge",
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )

        return client, collection

    # ================================================================
    # 索引构建 (增量)
    # ================================================================

    def build_index(self, force_rebuild: bool = False) -> dict:
        """
        扫描 knowledge_base/ 目录, 分块并存入 ChromaDB

        Args:
            force_rebuild: True 时清空旧索引重新构建

        Returns:
            {"added": int, "skipped": int, "total": int}
        """
        if not self._ready:
            logger.error("[RAG] 引擎未初始化, 请先调用 initialize()")
            return {"added": 0, "skipped": 0, "total": 0}

        if force_rebuild:
            logger.info("[RAG] 强制重建: 清空旧索引")
            self._client.delete_collection("labguardian_knowledge")
            self._collection = self._client.get_or_create_collection(
                name="labguardian_knowledge",
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )

        # 收集所有文档文件
        doc_files = self._scan_documents()
        if not doc_files:
            logger.warning("[RAG] 未找到任何知识文档")
            return {"added": 0, "skipped": 0, "total": 0}

        added = 0
        skipped = 0

        for filepath in doc_files:
            file_hash = self._file_hash(filepath)

            # 增量检查: 如果哈希已存在, 跳过
            if not force_rebuild and self._is_file_indexed(file_hash):
                skipped += 1
                continue

            # 加载并分块
            chunks = self._chunker.load_and_chunk_file(filepath)
            if not chunks:
                continue

            # 写入 ChromaDB
            ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
            documents = [c["text"] for c in chunks]
            metadatas = [{**c["metadata"], "file_hash": file_hash} for c in chunks]

            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            added += len(chunks)
            logger.info(f"[RAG] 已索引: {filepath.name} → {len(chunks)} 块")

        total = self._collection.count()
        result = {"added": added, "skipped": skipped, "total": total}
        print(f"[RAG] 索引构建完成: 新增 {added} 块, 跳过 {skipped} 文件, 总计 {total} 块")
        return result

    def _scan_documents(self) -> List[Path]:
        """扫描 builtin/ 和 user_docs/ 下的所有支持文件"""
        supported = {".txt", ".md", ".pdf"}
        files = []

        for scan_dir in [BUILTIN_DIR, USER_DOCS_DIR]:
            if not scan_dir.exists():
                scan_dir.mkdir(parents=True, exist_ok=True)
                continue
            for f in scan_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in supported:
                    files.append(f)

        logger.info(f"[RAG] 扫描到 {len(files)} 个文档文件")
        return files

    def _is_file_indexed(self, file_hash: str) -> bool:
        """检查某文件是否已被索引 (基于哈希)"""
        try:
            results = self._collection.get(
                where={"file_hash": file_hash},
                limit=1,
            )
            return len(results["ids"]) > 0
        except Exception:
            return False

    @staticmethod
    def _file_hash(filepath: Path) -> str:
        """计算文件内容哈希 (用于增量更新判断)"""
        h = hashlib.md5()
        h.update(filepath.read_bytes())
        return h.hexdigest()[:12]

    # ================================================================
    # 检索查询
    # ================================================================

    def query(self, question: str, top_k: int = 5) -> List[dict]:
        """
        检索与问题最相关的知识块

        Args:
            question: 用户问题
            top_k: 返回前 K 个最相关块

        Returns:
            [{"text": "...", "source": "...", "score": float}, ...]
        """
        if not self._ready or self._collection.count() == 0:
            return []

        try:
            results = self._collection.query(
                query_texts=[question],
                n_results=min(top_k, self._collection.count()),
            )

            retrieved = []
            for i in range(len(results["ids"][0])):
                doc_text = results["documents"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # ChromaDB cosine distance: 0=最相似, 2=最不相似
                similarity = 1.0 - distance / 2.0

                retrieved.append({
                    "text": doc_text,
                    "source": metadata.get("source", "unknown"),
                    "score": round(similarity, 3),
                    "metadata": metadata,
                })

            return retrieved

        except Exception as e:
            logger.error(f"[RAG] 查询失败: {e}")
            return []

    def get_context(self, question: str, top_k: int = 5,
                    min_score: float = 0.3) -> str:
        """
        检索并拼接为 LLM 可用的上下文字符串

        Args:
            question: 用户问题
            top_k: 最多返回 K 个块
            min_score: 最低相似度过滤

        Returns:
            格式化的参考资料字符串, 可直接嵌入 system prompt
        """
        results = self.query(question, top_k)
        if not results:
            return ""

        # 按相似度过滤
        filtered = [r for r in results if r["score"] >= min_score]
        if not filtered:
            return ""

        # 拼接为结构化上下文
        context_parts = ["[参考资料 - 来自本地知识库]"]
        for i, r in enumerate(filtered, 1):
            source = r["source"]
            score = r["score"]
            context_parts.append(
                f"--- 参考 {i} (来源: {source}, 相关度: {score:.0%}) ---\n{r['text']}"
            )
        context_parts.append("[参考资料结束]")

        return "\n\n".join(context_parts)

    # ================================================================
    # 管理工具
    # ================================================================

    def list_indexed_sources(self) -> List[str]:
        """列出已索引的所有文档来源"""
        if not self._ready:
            return []
        try:
            all_data = self._collection.get(include=["metadatas"])
            sources = set()
            for meta in all_data["metadatas"]:
                if "source" in meta:
                    sources.add(meta["source"])
            return sorted(sources)
        except Exception:
            return []

    def delete_source(self, source_name: str) -> int:
        """删除指定来源的所有文档块"""
        if not self._ready:
            return 0
        try:
            # 获取该来源的所有 ID
            results = self._collection.get(
                where={"source": source_name},
            )
            ids = results["ids"]
            if ids:
                self._collection.delete(ids=ids)
                logger.info(f"[RAG] 已删除来源 '{source_name}' 的 {len(ids)} 个块")
            return len(ids)
        except Exception as e:
            logger.error(f"[RAG] 删除失败: {e}")
            return 0

    def reset(self):
        """完全重置知识库 (删除 ChromaDB 数据)"""
        if self._client:
            try:
                self._client.delete_collection("labguardian_knowledge")
                logger.info("[RAG] 知识库已重置")
            except Exception:
                pass
        self._ready = False
