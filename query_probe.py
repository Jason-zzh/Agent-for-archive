#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查询探针：对 pdf_processor 保存的 Chroma 向量库做相似检索，直观打印每条查询的 top-k 结果，便于肉眼评估检索质量。
使用与入库相同的 BGE 模型编码查询并检索。
"""

import argparse
import os
import sys
from pathlib import Path

# 与 pdf_processor 一致，便于国内拉模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BGE_MODEL_NAME = "BAAI/bge-large-zh-v1.5"


def _load_vector_store(vector_db_dir: str):
    """加载 Chroma 与 BGE Embedding，返回 LangChain VectorStore。"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("需要: pip install langchain-huggingface sentence-transformers", file=sys.stderr)
        return None
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        print("需要: pip install langchain-community chromadb", file=sys.stderr)
        return None
    path = Path(vector_db_dir)
    if not path.is_dir():
        print(f"错误：向量库目录不存在: {path}", file=sys.stderr)
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name=BGE_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(path),
        embedding_function=embeddings,
        collection_name="pdf_chunks",
    )


def _format_result(rank: int, score: float, doc, max_content: int = 400):
    """单条检索结果格式化为可读字符串。Chroma 返回的 score 为 L2 距离，越小越相似。"""
    meta = (doc.metadata or {}) if hasattr(doc, "metadata") else {}
    page = meta.get("page_label") or meta.get("page_number", "?")
    cat = meta.get("category", "?")
    content = (doc.page_content or "") if hasattr(doc, "page_content") else str(doc)
    if len(content) > max_content:
        content = content[:max_content].rstrip() + "…"
    content = content.replace("\n", " ").strip()
    return f"  #{rank}  距离={score:.4f}  页={page}  类型={cat}\n    {content}"


def run_queries(vector_store, queries: list[str], k: int = 5, max_content: int = 400):
    """对每个查询做 similarity_search_with_score，打印结果。"""
    sep = "─" * 60
    for i, q in enumerate(queries, 1):
        q = q.strip()
        if not q:
            continue
        print(f"\n{sep}")
        print(f"【探针 {i}】 {q}")
        print(sep)
        try:
            hits = vector_store.similarity_search_with_score(q, k=k)
        except Exception as e:
            print(f"  检索失败: {e}")
            continue
        if not hits:
            print("  （无结果）")
            continue
        for rank, (doc, score) in enumerate(hits, 1):
            print(_format_result(rank, score, doc, max_content=max_content))
    print(f"\n{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="对 Chroma 向量库做查询探针，肉眼评估检索效果")
    parser.add_argument("vector_db", help="Chroma 向量库目录（与 pdf_processor --vector-db 相同）")
    parser.add_argument("-q", "--query", action="append", dest="queries", default=[], help="查询语句，可多次指定")
    parser.add_argument("-f", "--query-file", default=None, help="从文件读取查询（一行一条）")
    parser.add_argument("-k", type=int, default=5, help="每条查询返回 top-k 条，默认 5")
    parser.add_argument("--max-content", type=int, default=400, help="每条结果展示的最大字符数，默认 400")
    args = parser.parse_args()

    queries = list(args.queries) if args.queries else []
    if args.query_file:
        p = Path(args.query_file)
        if p.is_file():
            queries.extend([line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()])
        else:
            print(f"错误：查询文件不存在: {p}", file=sys.stderr)
            sys.exit(1)
    if not queries:
        print("请通过 -q \"查询\" 或 -f 查询文件 指定至少一条查询。", file=sys.stderr)
        sys.exit(1)

    print("正在加载向量库与 BGE 模型…")
    vs = _load_vector_store(args.vector_db)
    if vs is None:
        sys.exit(1)
    print("开始检索…")
    run_queries(vs, queries, k=args.k, max_content=args.max_content)


if __name__ == "__main__":
    main()
