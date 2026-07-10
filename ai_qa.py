#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PCS7 知识库 AI 问答：基于 Chroma 向量库的检索增强生成（RAG），支持多轮对话与历史记忆。
- 提示词模板：通用长文解析角色、语气、拒答逻辑（见 prompt_template.py）
- 记忆模块：LangChain 对话记忆，支持指代消解（如「它」指 CFC）
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BGE_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
# 本地 Ollama 默认地址与模型名；远程 API 地址和 key 必须通过参数或环境变量提供。
OLLAMA_BASE_URL = "http://localhost:11434/v1"
LOCAL_DEEPSEEK_MODEL = "deepseek-r1:14b"  # 可选: deepseek-r1:7b, deepseek-coder:6.7b 等
MULTIMODAL_DOCUMENT_PROMPT_TEMPLATE = (
    "【来源】\n"
    "页码: {page_label}\n"
    "类型: {category}\n"
    "模态: {modality}\n"
    "图片路径: {image_path}\n\n"
    "{page_content}"
)


def _load_embeddings():
    """加载 BGE 中文 Embedding。"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("需要: pip install langchain-huggingface sentence-transformers", file=sys.stderr)
        return None
    return HuggingFaceEmbeddings(
        model_name=BGE_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_vector_store(vector_db_dir: str, embeddings):
    """加载 Chroma 向量库。"""
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        print("需要: pip install langchain-community chromadb", file=sys.stderr)
        return None
    path = Path(vector_db_dir)
    if not path.is_dir():
        print(f"错误：向量库目录不存在: {path}", file=sys.stderr)
        return None
    return Chroma(
        persist_directory=str(path),
        embedding_function=embeddings,
        collection_name="pdf_chunks",
    )


def _load_llm(model: str = None, base_url: str = None, api_key: str = None, use_local: bool = False):
    """加载 LLM（OpenAI 兼容接口；use_local=True 时走本地 Ollama）。"""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("需要: pip install langchain-openai", file=sys.stderr)
        return None
    if use_local:
        base_url = base_url or os.environ.get("OPENAI_API_BASE", OLLAMA_BASE_URL)
        model = model or os.environ.get("LLM_MODEL", LOCAL_DEEPSEEK_MODEL)
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
    else:
        model = model or os.environ.get("LLM_MODEL")
        base_url = base_url or os.environ.get("OPENAI_API_BASE")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not model:
            print("错误：请通过 --model 或环境变量 LLM_MODEL 指定远程模型名。", file=sys.stderr)
            return None
        if not base_url:
            print("错误：请通过 --base-url 或环境变量 OPENAI_API_BASE 指定 OpenAI 兼容 API 地址。", file=sys.stderr)
            return None
        if not api_key:
            print("错误：请通过环境变量 OPENAI_API_KEY 指定 API key。", file=sys.stderr)
            return None
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.2,
        max_retries=3,
        request_timeout=120,
    )


def _normalize_retrieved_docs(docs):
    """为检索到的每条文档补全 QA 所需的多模态 metadata。"""
    for d in docs:
        m = dict(getattr(d, "metadata", None) or {})
        if "page_label" not in m:
            m["page_label"] = str(m.get("page_number", "?"))
        if "category" not in m:
            m["category"] = "Paragraph"
        if "image_path" not in m:
            m["image_path"] = ""
        if "modality" not in m:
            m["modality"] = "image" if m.get("image_path") else "text"
        d.metadata = m
    return docs


def _format_document_for_prompt(doc) -> str:
    """把单条检索结果格式化为多模态来源块，便于测试和人工调试。"""
    meta = dict(getattr(doc, "metadata", None) or {})
    if "page_label" not in meta:
        meta["page_label"] = str(meta.get("page_number", "?"))
    if "category" not in meta:
        meta["category"] = "Paragraph"
    if "image_path" not in meta:
        meta["image_path"] = ""
    if "modality" not in meta:
        meta["modality"] = "image" if meta.get("image_path") else "text"
    content = (getattr(doc, "page_content", None) or "").strip()
    return MULTIMODAL_DOCUMENT_PROMPT_TEMPLATE.format(
        page_label=meta.get("page_label", "?"),
        category=meta.get("category", "Paragraph"),
        modality=meta.get("modality", "text"),
        image_path=meta.get("image_path", ""),
        page_content=content,
    )


def _print_chunks(docs, max_chars: int = 400):
    """打印本轮检索到的 chunk（距离排序已由检索器返回顺序体现）。"""
    sep = "─" * 56
    print(f"\n【检索到的 chunk 共 {len(docs)} 条】")
    for i, d in enumerate(docs, 1):
        meta = getattr(d, "metadata", None) or {}
        page = meta.get("page_label") or meta.get("page_number", "?")
        cat = meta.get("category", "?")
        modality = meta.get("modality", "?")
        image_path = meta.get("image_path") or ""
        text = (getattr(d, "page_content", None) or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        text = text.replace("\n", " ")
        line = f"{sep}\n  #{i}  页={page}  类型={cat}  模态={modality}"
        if image_path:
            line += f"\n      图片路径={image_path}"
        print(f"{line}\n  {text}")
    print(sep + "\n")


def _build_chain(vector_store, llm, k: int = 5):
    """构建历史感知检索 + 回答链。返回 (整条链, history_aware_retriever, combine_docs_chain)。"""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

    try:
        from langchain.chains import create_history_aware_retriever, create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
    except ImportError:
        try:
            from langchain.chains.history_aware_retriever import create_history_aware_retriever
            from langchain.chains.retrieval import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain
        except ImportError:
            print("需要: pip install langchain", file=sys.stderr)
            return None

    from prompt_template import LONG_DOCUMENT_ANALYSIS_PROMPT, USER_TURN_REMINDER

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 历史感知检索：将含指代的追问改写为独立查询
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "你根据对话历史将用户最新问题改写为可独立理解的完整问题。若已清晰则直接返回原问题。只输出改写后的问题。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, rewrite_prompt)

    # 回答链：注入上下文 + 历史 + 问题（context 由 create_stuff_documents_chain 自动填入文档）
    qa_system = (
        LONG_DOCUMENT_ANALYSIS_PROMPT
        + "\n\n基于以下参考文档回答用户问题。若文档中无相关内容，请按拒答规则明确说明。\n\n参考文档：\n{context}"
    )
    # 每条用户轮次前加约束提醒，便于本地模型（如 Ollama）遵守 system 要求
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", USER_TURN_REMINDER + "{input}"),
    ])
    # 每条文档带多模态来源信息，避免模型把文本、表格、图片 caption 混为一类证据。
    document_prompt = PromptTemplate.from_template(MULTIMODAL_DOCUMENT_PROMPT_TEMPLATE)
    combine_docs_chain = create_stuff_documents_chain(
        llm, qa_prompt, document_prompt=document_prompt
    )

    # 检索后为文档补全 page_label（兼容旧向量库只有 page_number 的情况）
    from langchain_core.runnables import RunnableLambda

    def _retrieve_and_normalize(x):
        docs = _normalize_retrieved_docs(history_aware_retriever.invoke(x))
        return {**x, "context": docs}

    retrieval_chain = RunnableLambda(_retrieve_and_normalize) | combine_docs_chain
    return retrieval_chain, history_aware_retriever, combine_docs_chain


def _invoke_qa(inp: dict, chain, har, combine_chain, show_chunks: bool):
    """单次问答调用：可选先打印检索 chunk，再生成回答。"""
    if show_chunks:
        docs = _normalize_retrieved_docs(har.invoke(inp))
        _print_chunks(docs)
        return combine_chain.invoke({**inp, "context": docs})
    return chain.invoke(inp)


def run_interactive(vector_db: str, model: str = None, base_url: str = None, k: int = 5, use_local: bool = False, show_chunks: bool = False):
    """交互式多轮问答。"""
    print("正在加载 Embedding 模型与向量库…")
    embeddings = _load_embeddings()
    if embeddings is None:
        return 1
    vs = _load_vector_store(vector_db, embeddings)
    if vs is None:
        return 1
    print("正在加载 LLM…" + ("（本地 Ollama）" if use_local else ""))
    llm = _load_llm(model=model, base_url=base_url, use_local=use_local)
    if llm is None:
        return 1
    built = _build_chain(vs, llm, k=k)
    if built is None:
        return 1
    chain, har, combine_chain = built

    from langchain_community.chat_message_histories import ChatMessageHistory

    memory = ChatMessageHistory()
    print("\n" + "=" * 60)
    print("知识库问答（多轮对话，输入 quit/exit 退出）")
    if show_chunks:
        print("（已开启：每次回答前显示检索到的 chunk）")
    print("=" * 60)

    while True:
        try:
            q = input("\n您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("再见。")
            break

        chat_history = memory.messages
        inp = {"input": q, "chat_history": chat_history}
        try:
            t0 = time.perf_counter()
            result = _invoke_qa(inp, chain, har, combine_chain, show_chunks)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"错误: {e}")
            continue

        answer = result.get("answer", "") if isinstance(result, dict) else (result or "")
        print(f"\n助手: {answer}")
        print(f"（本次回答耗时 {elapsed:.2f} 秒）")

        memory.add_user_message(q)
        memory.add_ai_message(answer)
    return 0


def run_single(vector_db: str, question: str, model: str = None, base_url: str = None, k: int = 5, use_local: bool = False, show_chunks: bool = False):
    """单次问答（无历史）。"""
    embeddings = _load_embeddings()
    if embeddings is None:
        return 1
    vs = _load_vector_store(vector_db, embeddings)
    if vs is None:
        return 1
    llm = _load_llm(model=model, base_url=base_url, use_local=use_local)
    if llm is None:
        return 1
    built = _build_chain(vs, llm, k=k)
    if built is None:
        return 1
    chain, har, combine_chain = built

    inp = {"input": question, "chat_history": []}
    t0 = time.perf_counter()
    result = _invoke_qa(inp, chain, har, combine_chain, show_chunks)
    elapsed = time.perf_counter() - t0
    answer = result.get("answer", "") if isinstance(result, dict) else (result or "")
    print(answer)
    print(f"（本次回答耗时 {elapsed:.2f} 秒）")
    return 0


def main():
    parser = argparse.ArgumentParser(description="PCS7 知识库 AI 问答（RAG + 多轮记忆）")
    parser.add_argument("vector_db", help="Chroma 向量库目录")
    parser.add_argument("-q", "--query", default=None, help="单次问答时的问题（不指定则进入交互模式）")
    parser.add_argument("-m", "--model", default=None, help="LLM 模型名；远程模式默认读取 LLM_MODEL")
    parser.add_argument("--base-url", default=None, help="API base URL，如 Ollama: http://localhost:11434/v1")
    parser.add_argument("-k", type=int, default=20, help="检索 top-k 条文档")
    parser.add_argument("--local", action="store_true", help="使用本地 Ollama，需先安装 Ollama 并拉取模型")
    parser.add_argument("--show-chunks", action="store_true", help="回答前打印本轮检索到的 chunk（页码、类型、内容预览）")
    args = parser.parse_args()

    if args.query:
        sys.exit(run_single(args.vector_db, args.query, model=args.model, base_url=args.base_url, k=args.k, use_local=args.local, show_chunks=args.show_chunks))
    else:
        sys.exit(run_interactive(args.vector_db, model=args.model, base_url=args.base_url, k=args.k, use_local=args.local, show_chunks=args.show_chunks))


if __name__ == "__main__":
    main()
