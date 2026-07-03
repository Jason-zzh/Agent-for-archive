#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批处理问题列表：从文本文件逐行读取问题，调用与 ai_qa 相同的 RAG 链，
将每题的问答与耗时写入输出文件（控制台同步打印进度）。
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr != sys.stdout:
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _read_questions(path: Path) -> list[str]:
    """读取问题文件：一行一题，空行与 # 开头行跳过。"""
    raw = path.read_text(encoding="utf-8")
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def main():
    parser = argparse.ArgumentParser(description="从文件批量问答，结果写入另一文件")
    parser.add_argument("vector_db", help="Chroma 向量库目录")
    parser.add_argument("questions_file", help="问题列表文件（一行一个问题）")
    parser.add_argument("-o", "--output", required=True, metavar="FILE", help="输出结果文件路径")
    parser.add_argument("-m", "--model", default=None, help="LLM 模型名")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("-k", type=int, default=20, help="检索 top-k，默认 20")
    parser.add_argument("--local", action="store_true", help="使用本地 Ollama DeepSeek")
    parser.add_argument("--show-chunks", action="store_true", help="每题回答前在控制台打印检索 chunk（不写入输出文件）")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        metavar="秒",
        help="每题结束后、下一题开始前暂停的秒数，缓解云端 API 限流/503（默认 0）",
    )
    args = parser.parse_args()

    qpath = Path(args.questions_file)
    if not qpath.is_file():
        print(f"错误：问题文件不存在: {qpath}", file=sys.stderr)
        sys.exit(1)

    questions = _read_questions(qpath)
    if not questions:
        print("错误：未读取到任何问题（请检查文件内容或非空行）。", file=sys.stderr)
        sys.exit(1)

    from ai_qa import _build_chain, _invoke_qa, _load_embeddings, _load_llm, _load_vector_store

    print("正在加载 Embedding、向量库与 LLM…", file=sys.stderr)
    embeddings = _load_embeddings()
    if embeddings is None:
        sys.exit(1)
    vs = _load_vector_store(args.vector_db, embeddings)
    if vs is None:
        sys.exit(1)
    llm = _load_llm(model=args.model, base_url=args.base_url, use_local=args.local)
    if llm is None:
        sys.exit(1)
    built = _build_chain(vs, llm, k=args.k)
    if built is None:
        sys.exit(1)
    chain, har, combine_chain = built

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines_out: list[str] = []
    lines_out.append("=" * 60)
    lines_out.append(f"批次问答结果（共 {len(questions)} 题）")
    lines_out.append(f"问题文件: {qpath.resolve()}")
    lines_out.append(f"向量库: {Path(args.vector_db).resolve()}")
    lines_out.append("=" * 60)
    lines_out.append("")

    batch_t0 = time.perf_counter()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] 处理中…", file=sys.stderr)
        inp = {"input": q, "chat_history": []}
        try:
            t0 = time.perf_counter()
            result = _invoke_qa(inp, chain, har, combine_chain, args.show_chunks)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            lines_out.append(f"【问题 {i}】")
            lines_out.append(q)
            lines_out.append(f"耗时: 失败")
            lines_out.append(f"【回答】")
            lines_out.append(f"错误: {e}")
            lines_out.append("")
            lines_out.append("-" * 60)
            lines_out.append("")
            print(f"  失败: {e}", file=sys.stderr)
            if i < len(questions) and args.sleep > 0:
                time.sleep(args.sleep)
            continue

        answer = result.get("answer", "") if isinstance(result, dict) else (result or "")
        lines_out.append(f"【问题 {i}】")
        lines_out.append(q)
        lines_out.append(f"耗时: {elapsed:.2f} 秒")
        lines_out.append("【回答】")
        lines_out.append(answer if answer else "(空)")
        lines_out.append("")
        lines_out.append("-" * 60)
        lines_out.append("")

        if i < len(questions) and args.sleep > 0:
            time.sleep(args.sleep)

    batch_elapsed = time.perf_counter() - batch_t0
    lines_out.append(f"全部完成，总耗时 {batch_elapsed:.2f} 秒")

    text = "\n".join(lines_out)
    out_path.write_text(text, encoding="utf-8")
    print(f"已写入: {out_path.resolve()}（总耗时 {batch_elapsed:.2f} 秒）", file=sys.stderr)


if __name__ == "__main__":
    main()
