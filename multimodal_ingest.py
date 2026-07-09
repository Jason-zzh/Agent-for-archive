#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: ingest image captions into the existing Chroma vector store.

This script reads the JSONL produced by image_captioner.py, converts each image
caption into a LangChain Document, and stores the documents in the same
`pdf_chunks` Chroma collection used by pdf_processor.py and ai_qa.py.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep embedding downloads consistent with pdf_processor.py/query_probe.py.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BGE_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
DEFAULT_COLLECTION = "pdf_chunks"


@dataclass
class SimpleDocument:
    page_content: str
    metadata: dict[str, Any]


def _as_int(value: Any) -> int | None:
    """Return value as int when possible; Chroma metadata must stay primitive."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _join_entities(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(x).strip() for x in value if str(x).strip())
    if value is None:
        return ""
    return str(value).strip()


def _build_page_content(item: dict[str, Any]) -> str:
    caption = str(item.get("caption") or "").strip()
    image_type = str(item.get("image_type") or "").strip()
    entities = _join_entities(item.get("entities"))
    image_path = str(item.get("image_path") or "").strip()

    lines = ["【图片内容】", caption]
    if image_type:
        lines.append(f"图片类型：{image_type}")
    if entities:
        lines.append(f"关键实体：{entities}")
    if image_path:
        lines.append(f"图片路径：{image_path}")
    return "\n".join(line for line in lines if line)


def _build_metadata(item: dict[str, Any]) -> dict[str, Any]:
    pdf_page = _as_int(item.get("pdf_page"))
    page_label = str(pdf_page) if pdf_page is not None else str(item.get("pdf_page") or "?")

    metadata: dict[str, Any] = {
        "modality": "image",
        "category": "FigureCaption",
        "page_label": page_label,
        "printed_page": str(item.get("printed_page") or ""),
        "image_path": str(item.get("image_path") or ""),
        "file_name": str(item.get("file_name") or ""),
        "image_type": str(item.get("image_type") or ""),
        "entities": _join_entities(item.get("entities")),
        "source": str(item.get("source") or "image_captioner"),
    }
    if pdf_page is not None:
        metadata["page_number"] = pdf_page
    return metadata


def _doc_id(doc: Any) -> str:
    meta = doc.metadata or {}
    page = meta.get("page_label", "?")
    image_path = meta.get("image_path") or meta.get("file_name") or "unknown"
    return f"image-caption:{page}:{image_path}"


def load_image_caption_docs(jsonl_path: Path, limit: int | None = None) -> list[Any]:
    """Load image-caption JSONL rows as LangChain Document objects."""
    try:
        from langchain_core.documents import Document
    except ImportError:
        Document = SimpleDocument

    docs: list[Any] = []
    skipped = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped += 1
                print(f"跳过第 {line_no} 行：JSON 解析失败 - {exc}", file=sys.stderr)
                continue

            caption = str(item.get("caption") or "").strip()
            if not caption:
                skipped += 1
                print(f"跳过第 {line_no} 行：caption 为空", file=sys.stderr)
                continue

            docs.append(Document(page_content=_build_page_content(item), metadata=_build_metadata(item)))
            if limit is not None and len(docs) >= limit:
                break

    if skipped:
        print(f"已跳过 {skipped} 条无效记录。")
    return docs


def ingest_documents(
    docs: list[Any],
    vector_db_path: Path,
    model_name: str = BGE_MODEL_NAME,
    collection_name: str = DEFAULT_COLLECTION,
) -> None:
    """Append documents to a persistent Chroma collection."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("需要: pip install langchain-huggingface sentence-transformers", file=sys.stderr)
        sys.exit(1)
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        print("需要: pip install langchain-community chromadb", file=sys.stderr)
        sys.exit(1)

    if not docs:
        print("没有可入库的图片 caption。")
        return

    vector_db_path.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        persist_directory=str(vector_db_path),
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    vector_store.add_documents(docs, ids=[_doc_id(doc) for doc in docs])
    print(f"已写入 {len(docs)} 条图片 caption 到 Chroma: {vector_db_path} / {collection_name}")


def preview_docs(docs: list[Any], max_content: int = 260) -> None:
    sep = "-" * 60
    print(f"共读取 {len(docs)} 条图片 caption。预览前 5 条：")
    for idx, doc in enumerate(docs[:5], 1):
        meta = doc.metadata or {}
        content = (doc.page_content or "").replace("\n", " ").strip()
        if len(content) > max_content:
            content = content[:max_content].rstrip() + "..."
        print(sep)
        print(
            f"#{idx} 页={meta.get('page_label', '?')} "
            f"类型={meta.get('category', '?')} "
            f"图片={meta.get('image_path', '')}"
        )
        print(content)
    if docs:
        print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: 将图片 caption JSONL 追加写入 Chroma 向量库")
    parser.add_argument("captions", nargs="?", default="image_captions.jsonl", help="image_captioner.py 输出的 JSONL 文件")
    parser.add_argument("--vector-db", default="chroma_db", help="Chroma 向量库目录，默认 chroma_db")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection 名称，默认 pdf_chunks")
    parser.add_argument("--model", default=BGE_MODEL_NAME, help=f"Embedding 模型，默认 {BGE_MODEL_NAME}")
    parser.add_argument("--limit", type=int, default=None, help="只读取前 N 条 caption，便于小样本测试")
    parser.add_argument("--dry-run", action="store_true", help="只读取和预览，不写入 Chroma")
    args = parser.parse_args()

    captions_path = Path(args.captions)
    if not captions_path.is_file():
        print(f"错误：caption JSONL 不存在: {captions_path}", file=sys.stderr)
        sys.exit(1)

    docs = load_image_caption_docs(captions_path, limit=args.limit)
    preview_docs(docs)

    if args.dry_run:
        print("dry-run 模式：未写入向量库。")
        return

    ingest_documents(
        docs,
        vector_db_path=Path(args.vector_db),
        model_name=args.model,
        collection_name=args.collection,
    )


if __name__ == "__main__":
    main()
