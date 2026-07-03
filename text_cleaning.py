# -*- coding: utf-8 -*-
"""
文本清洗模块：处理 Unstructured 提取出的 docs，去除换行截断、页眉页脚、合并短小元素。
"""

import re
from typing import Any


# 视为“文本”可参与合并的类别（Title / NarrativeText / UncategorizedText 等）
TEXT_CATEGORIES = frozenset({"Title", "NarrativeText", "UncategorizedText", "Text", "Header", "Paragraph"})


def normalize_line_breaks(text: str) -> str:
    """
    去除段落内不该断开的换行符：单换行改为空格，保留真实段落分隔（\\n\\n）。
    """
    if not text or not isinstance(text, str):
        return text
    # 按双换行拆成段落，段落内单换行替为空格，再合回
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = []
    for p in paragraphs:
        p = p.replace("\r\n", "\n").replace("\r", "\n")
        p = re.sub(r"(?<!\n)\n(?!\n)", " ", p)
        p = re.sub(r" +", " ", p).strip()
        if p:
            cleaned.append(p)
    return "\n\n".join(cleaned)


def is_likely_header_footer(text: str) -> bool:
    """
    判断是否为页眉/页脚或孤立页码：仅数字、常见版权/手册标题等。
    """
    s = (text or "").strip()
    if not s:
        return True
    # 纯页码：仅数字，可能带空格或少量后缀
    if re.match(r"^\d{1,5}\s*$", s):
        return True
    # 极短且多为数字（如 "1 - 8 页 1-8" 的片段）
    if len(s) <= 3 and re.search(r"\d", s) and not re.search(r"[a-zA-Z\u4e00-\u9fff]{2,}", s):
        return True
    # 常见页眉/页脚模式（不区分大小写）
    patterns = [
        r"^PCS\s*7\s+.*(手册|Manual|V\d+\.\d+)\s*\|",  # 手册名 + 版本
        r"Copyright\s*\d{4}\s*[©©]\s*Siemens",
        r"Siemens\s+AG\s*$",
        r"All\s+rights\s+reserved",
        r"I\s+IA\s+AS\s+S\s+SUP\s+PA",  # 部门缩写
        r"^D-\d+\s+\w+",  # 德文地址如 D-76181 Karlsruhe
        r"^Karlsruhe.*\d{4}",
        r"^Siemens\s+AG\s*$",
        r"免责声明\s*$",
        r"内容如有变动，恕不事先通知",
        r"未经明确的书面授权",
        r"^\d+\s*-\s*\d+\s*页\s*\d+-\d+",  # “1 - 8 页 1-8”
    ]
    for pat in patterns:
        if re.search(pat, s, re.IGNORECASE):
            return True
    return False


def get_category(doc: Any) -> str:
    """从 doc.metadata 取类别，与 pdf_processor 一致。"""
    meta = getattr(doc, "metadata", None) or {}
    return meta.get("category") or meta.get("element_type") or "Text"


def merge_short_elements(
    docs: list,
    max_chars: int = 50,
    text_categories: frozenset[str] | None = None,
) -> list:
    """
    将连续的、字数极少的文本类元素合并为一个。
    text_categories：视为可合并的类别集合，默认 Title/NarrativeText/UncategorizedText/Text/Header。
    """
    if text_categories is None:
        text_categories = TEXT_CATEGORIES
    if not docs:
        return []

    try:
        from langchain_core.documents import Document
    except ImportError:
        Document = None

    out: list = []
    i = 0
    while i < len(docs):
        doc = docs[i]
        content = (getattr(doc, "page_content", None) or "").strip()
        meta = dict(getattr(doc, "metadata", None) or {})
        cat = get_category(doc)

        # 非文本类或已较长：直接加入
        if cat not in text_categories or len(content) > max_chars:
            out.append(doc)
            i += 1
            continue

        # 收集后续连续短文本，并收集所有出现过的页码（跨页时保留全部）
        run_contents = [content]
        run_meta = dict(meta)
        run_pages = []
        p0 = meta.get("page_number")
        if p0 is not None:
            run_pages.append(int(p0) if isinstance(p0, (int, float)) else p0)
        j = i + 1
        while j < len(docs):
            next_doc = docs[j]
            next_content = (getattr(next_doc, "page_content", None) or "").strip()
            next_cat = get_category(next_doc)
            next_meta = getattr(next_doc, "metadata", None) or {}
            if next_cat not in text_categories or len(next_content) > max_chars:
                break
            run_contents.append(next_content)
            pn = next_meta.get("page_number")
            if pn is not None:
                run_pages.append(int(pn) if isinstance(pn, (int, float)) else pn)
            j += 1

        if run_pages:
            run_meta["page_numbers"] = sorted(set(run_pages))
            run_meta["page_number"] = run_meta["page_numbers"][0]  # 兼容仅读 page_number 的代码
        merged_text = " ".join(run_contents)
        if Document is not None:
            run_meta["category"] = "MergedText"
            out.append(Document(page_content=merged_text, metadata=run_meta))
        else:
            doc.page_content = merged_text
            doc.metadata = run_meta
            out.append(doc)
        i = j
    return out


def clean_docs(
    docs: list,
    *,
    normalize_breaks: bool = True,
    drop_headers_footers: bool = True,
    merge_short: bool = True,
    merge_max_chars: int = 50,
) -> list:
    """
    对 Unstructured 提取的 docs 做全文清洗。

    - normalize_breaks: 段落内单换行改为空格，保留 \\n\\n。
    - drop_headers_footers: 过滤疑似页眉/页脚和孤立页码。
    - merge_short: 合并连续字数 <= merge_max_chars 的文本类元素。
    """
    if not docs:
        return []

    try:
        from langchain_core.documents import Document
    except ImportError:
        Document = None

    result = []
    for doc in docs:
        content = getattr(doc, "page_content", None) or ""
        meta = dict(getattr(doc, "metadata", None) or {})

        cat = meta.get("category") or ""
        if drop_headers_footers and cat != "Table" and is_likely_header_footer(content):
            continue
        if normalize_breaks and cat != "Table":
            content = normalize_line_breaks(content)
        if not content.strip():
            continue
        if Document is not None:
            meta = dict(getattr(doc, "metadata", None) or {})
            result.append(Document(page_content=content, metadata=meta))
        else:
            doc.page_content = content
            result.append(doc)

    if merge_short:
        result = merge_short_elements(result, max_chars=merge_max_chars)
    return result
