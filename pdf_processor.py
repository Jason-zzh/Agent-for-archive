#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 PyMuPDF 从 PDF 提取段落级文本，可选提取内嵌图片、文本清洗、智能切片与向量化入库。
解析结果：控制台截断显示，-o 文件为完整内容；支持文本清洗（换行规范化、过滤页眉页脚、合并短段）。
跨页合并的段落会标注多页码（page_numbers / page_label），便于 QA 时引用「第 5、6、7 页」。

使用方式
--------
  # 仅解析并输出到控制台
  python pdf_processor.py 文档.pdf

  # 解析 + 完整结果写入文件
  python pdf_processor.py 文档.pdf -o result.txt

  # 解析 + 文本清洗 + 智能切片 + 向量化入库（供 ai_qa / query_probe 使用）
  python pdf_processor.py 文档.pdf --vector-db chroma_db -o result.txt

  # 同时提取带 caption 的内嵌图片到 image 目录
  python pdf_processor.py 文档.pdf --raw-images -i image -o result.txt

  # 不做清洗、不做切片（仅段落/表格级提取）
  python pdf_processor.py 文档.pdf --no-clean --no-chunk -o result.txt

  # 查看全部参数
  python pdf_processor.py -h
"""

import argparse
import os
import sys
from pathlib import Path

# HuggingFace 镜像（在加载任何 HF 模型前设置，便于国内下载）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Windows 下确保控制台能正确显示中英文（UTF-8）
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr != sys.stdout:
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def _table_data_to_markdown(rows: list) -> str:
    """将表格数据（二维列表）转为 Markdown 表格字符串。"""
    if not rows:
        return ""
    # 单元格内 | 换行符等需要转义或替换，避免破坏表格
    def cell_text(c):
        if c is None:
            return ""
        s = str(c).strip().replace("\n", " ").replace("\r", " ")
        return s.replace("|", "\\|")
    normalized = [[cell_text(c) for c in row] for row in rows]
    if not normalized:
        return ""
    # 统一列数（按第一行）
    ncols = max(len(r) for r in normalized)
    for r in normalized:
        r.extend([""] * (ncols - len(r)))
    lines = []
    for i, row in enumerate(normalized):
        line = "| " + " | ".join(row[:ncols]) + " |"
        lines.append(line)
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in range(ncols)) + " |")
    return "\n".join(lines)


def _bbox_inside_any(block_bbox, table_rects: list, tol: float = 2.0) -> bool:
    """判断文本块 bbox 是否落在任一表格区域内，是则视为表格内文字不再单独输出。"""
    if not block_bbox or len(block_bbox) < 4:
        return False
    bx0, by0, bx1, by1 = block_bbox[0], block_bbox[1], block_bbox[2], block_bbox[3]
    for tr in table_rects:
        if hasattr(tr, "x0"):
            tx0, ty0, tx1, ty1 = tr.x0, tr.y0, tr.x1, tr.y1
        else:
            tx0, ty0, tx1, ty1 = tr[0], tr[1], tr[2], tr[3]
        if bx0 >= tx0 - tol and by0 >= ty0 - tol and bx1 <= tx1 + tol and by1 <= ty1 + tol:
            return True
    return False


def _extract_paragraphs_and_tables_pymupdf(pdf_path: Path) -> list[tuple[str, int, str]]:
    """用 PyMuPDF 提取段落与表格，按页面内垂直位置排序。落在表格区域内的文字不再单独输出，避免重复。"""
    try:
        import fitz
    except ImportError:
        print("请先安装 PyMuPDF: pip install pymupdf", file=sys.stderr)
        return []
    items: list[tuple[float, int, str, str]] = []  # (y0, page_no, content, category)
    with fitz.open(str(pdf_path)) as doc:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_no = page_idx + 1
            # 先收集本页所有表格的 bbox，用于过滤重复的段落
            table_rects: list = []
            try:
                finder = page.find_tables()
                for table in finder:
                    r = getattr(table, "bbox", None) or getattr(table, "rect", None)
                    if r is not None:
                        table_rects.append(r)
            except AttributeError:
                pass
            raw = page.get_text("dict", clip=page.rect)
            blocks = raw.get("blocks") or []
            for block in blocks:
                bbox = block.get("bbox")
                if table_rects and _bbox_inside_any(bbox, table_rects):
                    continue  # 该块在表格内，只保留表格输出，不单独作为段落
                y0 = bbox[1] if bbox and len(bbox) >= 2 else 0.0
                lines = block.get("lines") or []
                parts: list[str] = []
                for line in lines:
                    for span in line.get("spans") or []:
                        t = span.get("text") or ""
                        if t:
                            parts.append(t)
                text = " ".join(parts).strip()
                if text:
                    items.append((y0, page_no, text, "Paragraph"))
            try:
                finder = page.find_tables()
                for table in finder:
                    try:
                        data = table.extract()
                        if data:
                            md = _table_data_to_markdown(data)
                            if md:
                                bbox = getattr(table, "bbox", None) or getattr(table, "rect", None)
                                ty = bbox[1] if bbox and len(bbox) >= 2 else 0.0
                                items.append((ty, page_no, md, "Table"))
                    except Exception:
                        continue
            except AttributeError:
                pass
    items.sort(key=lambda x: (x[1], x[0]))
    return [(content, page_no, category) for _, page_no, content, category in items]


def _caption_below_rect(page, img_rect, text_blocks, max_gap: float = 80) -> str:
    """在页面文本块中找位于图片下方、最可能为 caption 的文本（如「图 1.3 PCS 7 现场级」）。"""
    x0, y0, x1, y1 = img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1
    best_text = ""
    best_y = 1e9
    for block in text_blocks:
        bbox = block.get("bbox")
        if not bbox:
            continue
        bx0, by0, bx1, by1 = bbox
        # caption 通常在图下方：块顶部略低于图底，且水平有重叠
        if by0 < y1 - 5:
            continue
        gap = by0 - y1
        if gap > max_gap:
            continue
        if bx1 < x0 or bx0 > x1:
            continue
        lines = block.get("lines") or []
        text = " ".join(
            span.get("text", "")
            for line in lines
            for span in line.get("spans", [])
        ).strip()
        if not text:
            continue
        if by0 < best_y:
            best_y = by0
            best_text = text
    return best_text


def _extract_raw_images_pymupdf(pdf_path: Path, image_dir: Path, use_caption: bool = True) -> int:
    """用 PyMuPDF (fitz) 提取 PDF 内嵌的原始图片，经 Pillow 解码后统一保存为 PNG；仅保存能匹配到 caption（图下方文本）的图，无 caption 的图不保存。"""
    import io
    try:
        import fitz
    except ImportError:
        print("请先安装 PyMuPDF: pip install pymupdf", file=sys.stderr)
        return 0
    try:
        from PIL import Image as PILImage
    except ImportError:
        print("请先安装 Pillow: pip install Pillow", file=sys.stderr)
        return 0
    image_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with fitz.open(str(pdf_path)) as doc:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            image_list = page.get_images()
            text_blocks = (page.get_text("dict", clip=page.rect).get("blocks", []) if use_caption else [])
            page_dir = image_dir / f"p{page_idx + 1}"
            page_dir.mkdir(parents=True, exist_ok=True)
            used_names = set()
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue
                image_bytes = base_image["image"]
                base_name = None
                if use_caption and text_blocks:
                    try:
                        rects = page.get_image_rects(xref)
                        if rects:
                            caption = _caption_below_rect(page, rects[0], text_blocks)
                            if caption:
                                base_name = _sanitize_filename_for_caption(caption)
                                if len(base_name) < 4:
                                    base_name = None
                    except Exception:
                        pass
                if not base_name:
                    continue
                if base_name in used_names:
                    base_name = f"{base_name}_{img_idx + 1}"
                used_names.add(base_name)
                name = f"{base_name}.png"
                if len(name) > 200:
                    base_name = base_name[: 195]
                    name = f"{base_name}.png"
                out_path = page_dir / name
                try:
                    img = PILImage.open(io.BytesIO(image_bytes))
                    if img.mode in ("RGBA", "P"):
                        img.save(out_path, "PNG")
                    else:
                        img.save(out_path, "PNG")
                    saved += 1
                except Exception as e:
                    print(f"警告：保存 {out_path} 失败: {e}", file=sys.stderr)
    return saved


class _StderrFilter:
    """过滤 PDF 解析时产生的颜色空间警告。"""
    def __init__(self, target):
        self.target = target
    def write(self, msg):
        if "Cannot set non-stroke color" in msg or "expected 4 components" in msg:
            return
        self.target.write(msg)
    def flush(self):
        self.target.flush()


def _sanitize_filename_for_caption(caption: str, max_len: int = 60) -> str:
    """把 caption 转成合法文件名，保留中文和空格；限制长度避免 File name too long。"""
    invalid = r'\/:*?"<>|'
    s = caption.strip().replace("\n", " ").replace("\r", " ")
    for c in invalid:
        s = s.replace(c, "_")
    s = "".join(c for c in s if c.isprintable() or c in " _-").strip(" ._")
    if not s:
        return ""
    return s[:max_len].rstrip(" ._") if len(s) > max_len else s


def _get_element_type(doc) -> str:
    """从 doc.metadata 取元素类型。"""
    return doc.metadata.get("category") or doc.metadata.get("element_type") or "Paragraph"


# 智能切片：优先按段落、句子切分，避免从句子中间截断
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 120
CHUNK_SEPARATORS = ["\n\n", "\n", "。", ". ", " ", ""]


def _chunk_docs(docs: list, chunk_size: int, chunk_overlap: int):
    """使用 RecursiveCharacterTextSplitter 对 docs 做智能切片；Table 类型整块保留不切。"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            return docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNK_SEPARATORS,
        length_function=len,
    )
    result: list = []
    buffer: list = []
    for doc in docs:
        if (doc.metadata or {}).get("category") == "Table":
            if buffer:
                result.extend(splitter.split_documents(buffer))
                buffer = []
            result.append(doc)
        else:
            buffer.append(doc)
    if buffer:
        result.extend(splitter.split_documents(buffer))
    return result


# 向量化入库：BAAI/bge-large-zh-v1.5 + Chroma
BGE_MODEL_NAME = "BAAI/bge-large-zh-v1.5"


def _embed_and_store(docs: list, vector_db_path: Path, model_name: str = BGE_MODEL_NAME) -> None:
    """用 BGE 中文模型将 docs 向量化并持久化到 Chroma（使用 langchain-huggingface，避免弃用警告）。"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("向量化需要安装: pip install langchain-huggingface sentence-transformers chromadb", file=sys.stderr)
        return
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        print("向量库需要安装: pip install langchain-community chromadb", file=sys.stderr)
        return

    if not docs:
        return
    vector_db_path = Path(vector_db_path)
    vector_db_path.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    persist_dir = str(vector_db_path)
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="pdf_chunks",
    )
    print(f"已向量化 {len(docs)} 个 chunk 并写入: {persist_dir}")


def process_pdf(
    pdf_path: str,
    show_metadata: bool = False,
    quiet_warnings: bool = True,
    output_file: str | None = None,
    max_display_chars: int = 500,
    save_images_dir: str | None = None,
    extract_raw_images: bool = False,
    do_clean: bool = True,
    do_chunk: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    vector_db_dir: str | None = None,
) -> None:
    """使用 PyMuPDF 提取段落级文本；可选文本清洗、智能切片、向量化入库（BGE + Chroma）。"""
    path = Path(pdf_path)
    if not path.exists():
        print(f"错误：文件不存在 - {path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix.lower() != ".pdf":
        print(f"警告：扩展名不是 .pdf，仍尝试解析：{path}", file=sys.stderr)

    if extract_raw_images:
        if not save_images_dir:
            print("错误：使用 --raw-images 时请指定 -i/--save-images 目录", file=sys.stderr)
            sys.exit(1)
        image_dir = Path(save_images_dir)
        n = _extract_raw_images_pymupdf(path, image_dir)
        print(f"已用 PyMuPDF 提取 {n} 张原始图片到: {image_dir}")
        save_images_dir = None

    if save_images_dir and not extract_raw_images:
        print("提示：要保存图片请加 --raw-images。", file=sys.stderr)
        save_images_dir = None

    header = f"\n{'='*60}\n正在解析: {path.name}\n提取: PyMuPDF 段落级文本\n{'='*60}\n"
    print(header)

    # 使用 PyMuPDF 提取段落级文本，转为 Document 列表（与 text_cleaning / 输出格式兼容）
    try:
        from langchain_core.documents import Document
    except ImportError:
        class Document:
            __slots__ = ("page_content", "metadata")
            def __init__(self, page_content: str, metadata: dict):
                self.page_content = page_content
                self.metadata = metadata

    items = _extract_paragraphs_and_tables_pymupdf(path)
    docs = [
        Document(page_content=content, metadata={"page_number": p, "category": cat})
        for content, p, cat in items
    ]

    if not docs:
        print("未提取到任何内容。")
        return

    # 文本清洗：去换行截断、过滤页眉页脚、合并短小元素
    if do_clean:
        try:
            from text_cleaning import clean_docs
            docs = clean_docs(
                docs,
                normalize_breaks=True,
                drop_headers_footers=True,
                merge_short=True,
                merge_max_chars=50,
            )
        except ImportError:
            pass  # 无 text_cleaning 时跳过清洗

    # 智能切片（Chunking）：按段落、句子切分，避免从中间截断，便于后续 RAG/模型上下文
    if do_chunk and docs:
        n_before = len(docs)
        docs = _chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if len(docs) != n_before:
            header += f"切片: chunk_size={chunk_size}, overlap={chunk_overlap} → {len(docs)} 块\n"
            print(f"已切片为 {len(docs)} 块（chunk_size={chunk_size}, overlap={chunk_overlap}）")

    # 为每条 doc 生成 page_label（单页为 "5"，多页为 "5,6,7"），便于检索结果与 QA 展示
    for doc in docs:
        meta = doc.metadata or {}
        if "page_label" in meta:
            continue
        pages = meta.get("page_numbers")
        if pages is not None:
            meta["page_label"] = ",".join(str(p) for p in sorted(pages))
        else:
            p = meta.get("page_number")
            meta["page_label"] = str(p) if p is not None else "?"
        doc.metadata = meta

    # 向量化入库：BAAI/bge-large-zh-v1.5 做 Embedding，Chroma 持久化
    if vector_db_dir and docs:
        _embed_and_store(docs, Path(vector_db_dir), model_name=BGE_MODEL_NAME)

    for i, doc in enumerate(docs, 1):
        el_type = _get_element_type(doc)
        text = (doc.page_content or "").strip()
        meta_str = f"\n    元数据: {doc.metadata}" if show_metadata and doc.metadata else ""
        display_text = text if len(text) <= max_display_chars else text[:max_display_chars] + "..."
        print(f"[{i}] 类型: {el_type}\n    内容: {display_text}{meta_str}\n")

    summary = f"{'='*60}\n共提取 {len(docs)} 个元素\n{'='*60}\n"
    print(summary)

    if output_file:
        out_path = Path(output_file)
        full_content = "".join(
            f"[{i}] 类型: {_get_element_type(doc)}\n    内容: {(doc.page_content or '').strip()}\n"
            + (f"    元数据: {doc.metadata}\n" if show_metadata else "")
            + "\n"
            for i, doc in enumerate(docs, 1)
        )
        out_path.write_text(header + full_content + summary, encoding="utf-8")
        print(f"完整结果已写入: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="PyMuPDF 段落级文本提取，可选提图与文本清洗")
    parser.add_argument("pdf_path", nargs="?", default=None, help="PDF 文件路径")
    parser.add_argument("-m", "--metadata", action="store_true", help="输出元数据")
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        default=None,
        help="将完整解析结果写入文件",
    )
    parser.add_argument(
        "-i", "--save-images",
        metavar="DIR",
        nargs="?",
        const="image",
        default=None,
        help="与 --raw-images 一起用时，将 PyMuPDF 提取的内嵌图片保存到该目录（默认 image）",
    )
    parser.add_argument(
        "--no-quiet-warnings",
        action="store_true",
        help="不屏蔽 PDF 颜色空间等底层警告",
    )
    parser.add_argument(
        "--raw-images",
        action="store_true",
        help="用 PyMuPDF 提取 PDF 内嵌的原始图片到 -i 目录（需同时指定 -i）",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="不做文本清洗（默认会做：去换行截断、过滤页眉页脚、合并短小元素）",
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="不做智能切片（默认使用 RecursiveCharacterTextSplitter 按段落/句子切块）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        metavar="N",
        help=f"切片块大小字符数（默认 {DEFAULT_CHUNK_SIZE}，建议 500–800）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        metavar="N",
        help=f"切片重叠字符数（默认 {DEFAULT_CHUNK_OVERLAP}，建议 100–150）",
    )
    parser.add_argument(
        "--vector-db",
        metavar="DIR",
        default=None,
        help="将 chunk 用 BAAI/bge-large-zh-v1.5 向量化并存入 Chroma 目录（需安装 langchain-community sentence-transformers chromadb）",
    )
    args = parser.parse_args()

    if not args.pdf_path:
        parser.print_help()
        print("\n示例: python pdf_processor.py 你的文档.pdf -o result.txt")
        print("      python pdf_processor.py 你的文档.pdf -i image --raw-images")
        sys.exit(0)

    process_pdf(
        args.pdf_path,
        show_metadata=args.metadata,
        quiet_warnings=not args.no_quiet_warnings,
        output_file=args.output,
        save_images_dir=args.save_images,
        extract_raw_images=args.raw_images,
        do_clean=not args.no_clean,
        do_chunk=not args.no_chunk,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_db_dir=args.vector_db,
    )


if __name__ == "__main__":
    main()
