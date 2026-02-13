"""
RAG 知识库管理 CLI 工具

用法:
  python build_knowledge_base.py                # 增量构建索引
  python build_knowledge_base.py --rebuild      # 强制重建索引
  python build_knowledge_base.py --query "8050引脚"  # 测试检索
  python build_knowledge_base.py --list         # 列出已索引文档
  python build_knowledge_base.py --stats        # 索引统计信息

提示:
  - 首次运行会下载 Embedding 模型 (~100MB), 之后使用本地缓存
  - 内置知识文档在 knowledge_base/builtin/ 目录
  - 用户自添文档放入 knowledge_base/user_docs/ 目录
  - 支持 .md / .txt / .pdf 格式
"""

import argparse
import sys
from pathlib import Path

# 确保项目路径在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai.rag_engine import RAGEngine


def main():
    parser = argparse.ArgumentParser(
        description="LabGuardian RAG 知识库管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重建整个索引 (清空旧数据)")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="测试检索: 输入问题, 返回相关知识片段")
    parser.add_argument("--list", "-l", action="store_true",
                        help="列出已索引的文档来源")
    parser.add_argument("--stats", action="store_true",
                        help="显示索引统计信息")
    parser.add_argument("--top-k", type=int, default=5,
                        help="检索返回的最大结果数 (默认: 5)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互式检索模式 (持续输入问题)")

    args = parser.parse_args()

    # 初始化 RAG 引擎
    print("=" * 60)
    print("  LabGuardian RAG 知识库管理工具")
    print("=" * 60)
    print()

    rag = RAGEngine()
    print("[1/2] 正在初始化 RAG 引擎 (加载 Embedding 模型)...")
    if not rag.initialize():
        print("❌ RAG 引擎初始化失败! 请检查依赖是否安装:")
        print("   pip install chromadb sentence-transformers")
        sys.exit(1)

    # --- 列出已索引文档 ---
    if args.list:
        sources = rag.list_indexed_sources()
        if sources:
            print(f"\n已索引的文档 ({len(sources)} 个):")
            for s in sources:
                print(f"  📄 {s}")
        else:
            print("\n知识库为空, 请先构建索引:")
            print("  python build_knowledge_base.py")
        return

    # --- 统计信息 ---
    if args.stats:
        count = rag.doc_count
        sources = rag.list_indexed_sources()
        print(f"\n📊 索引统计:")
        print(f"   文本块总数: {count}")
        print(f"   文档来源数: {len(sources)}")
        for s in sources:
            print(f"   - {s}")
        return

    # --- 构建/重建索引 ---
    if args.query is None and not args.interactive:
        print("[2/2] 正在构建知识库索引...")
        result = rag.build_index(force_rebuild=args.rebuild)
        print()
        print(f"✅ 索引构建完成!")
        print(f"   新增块数: {result['added']}")
        print(f"   跳过文件: {result['skipped']}")
        print(f"   索引总量: {result['total']}")
        print()
        print("提示: 运行以下命令测试检索效果:")
        print('  python build_knowledge_base.py --query "8050三极管引脚"')
        print('  python build_knowledge_base.py --query "运放反相放大器增益"')
        print('  python build_knowledge_base.py -i   # 交互模式')
        return

    # --- 确保有索引数据 ---
    if rag.doc_count == 0:
        print("⚠️ 知识库为空, 先构建索引...")
        rag.build_index()
        print()

    # --- 单次查询 ---
    if args.query:
        _do_query(rag, args.query, args.top_k)
        return

    # --- 交互模式 ---
    if args.interactive:
        print("\n📝 交互式检索模式 (输入 'quit' 退出)")
        print("-" * 40)
        while True:
            try:
                question = input("\n🔍 问题: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    print("Bye!")
                    break
                if not question:
                    continue
                _do_query(rag, question, args.top_k)
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break


def _do_query(rag: RAGEngine, question: str, top_k: int):
    """执行一次检索并打印结果"""
    print(f"\n🔍 检索: \"{question}\"")
    print("-" * 50)

    results = rag.query(question, top_k=top_k)

    if not results:
        print("  (无相关结果)")
        return

    for i, r in enumerate(results, 1):
        score = r["score"]
        source = r["source"]
        text = r["text"]
        # 截断显示
        display_text = text[:200] + "..." if len(text) > 200 else text
        print(f"\n  [{i}] 相关度: {score:.1%} | 来源: {source}")
        print(f"  {display_text}")

    # 也打印合成的上下文
    context = rag.get_context(question, top_k=top_k)
    if context:
        print(f"\n{'=' * 50}")
        print("📋 合成上下文 (可直接嵌入 LLM prompt):")
        print(context[:500] + "..." if len(context) > 500 else context)


if __name__ == "__main__":
    main()
