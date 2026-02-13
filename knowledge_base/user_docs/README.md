# 将用户自定义文档（PDF/TXT/MD）放入此目录
# RAG 引擎会自动扫描并索引这些文件
#
# 支持格式:
#   - .md  (Markdown)
#   - .txt (纯文本)
#   - .pdf (需要安装 pymupdf 或 pdfplumber)
#
# 建议:
#   - 芯片 datasheet PDF 放在这里
#   - 实验指导书放在这里
#   - 添加文件后运行: python -m ai.rag_engine --rebuild
