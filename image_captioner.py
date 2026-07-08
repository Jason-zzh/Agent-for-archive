#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Caption extracted PDF figures with a Qwen-VL compatible vision model.

The script reads images saved by pdf_processor.py under image/p{pdf_page}/,
uses the directory name as the authoritative PDF page number, and writes
one JSON object per line to image_captions.jsonl.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://llm-ff0m6y205ofqzyru.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_PROMPT = """你是一个面向技术文档和工程手册的图像理解助手。

请只分析图片本身，不要猜测 PDF 页码。页码会由程序从文件路径提供。

请输出严格 JSON，不要使用 Markdown 代码块，不要添加额外解释。JSON 字段如下：
{
  "caption": "用中文概括图片内容，适合被向量检索；如果是流程图/架构图/软件截图，请说明主要对象、关系和用途。",
  "image_type": "从 diagram、chart、table_screenshot、software_screenshot、photo、formula、other 中选择一个",
  "entities": ["图片中重要的中文或英文术语，最多 12 个"]
}
"""

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _configure_utf8_stdio() -> None:
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            if sys.stderr != sys.stdout:
                sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _iter_images(image_dir: Path) -> list[Path]:
    images = [
        p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(images, key=lambda p: str(p).lower())


def _pdf_page_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        m = re.fullmatch(r"p(\d+)", part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _image_to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model response JSON is not an object")
    return data


def _normalize_caption_payload(data: dict[str, Any]) -> dict[str, Any]:
    caption = str(data.get("caption") or "").strip()
    image_type = str(data.get("image_type") or "other").strip() or "other"
    entities_raw = data.get("entities") or []
    if isinstance(entities_raw, str):
        entities = [e.strip() for e in re.split(r"[,，;；\n]", entities_raw) if e.strip()]
    elif isinstance(entities_raw, list):
        entities = [str(e).strip() for e in entities_raw if str(e).strip()]
    else:
        entities = []
    return {
        "caption": caption,
        "image_type": image_type,
        "entities": entities[:12],
    }


def _build_client(api_key: str, base_url: str):
    try:
        from openai import OpenAI
    except ImportError:
        print("需要安装 openai: pip install openai", file=sys.stderr)
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


def caption_image(client, model: str, image_path: Path, prompt: str, temperature: float) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_uri(image_path)}},
                ],
            }
        ],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return _normalize_caption_payload(_extract_json(content))


def _load_done_paths(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.is_file():
        return done
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        image_path = item.get("image_path")
        if image_path:
            done.add(str(image_path))
    return done


def main() -> int:
    _configure_utf8_stdio()
    _load_env_file()

    parser = argparse.ArgumentParser(description="Caption PDF figures with Qwen-VL and save JSONL output")
    parser.add_argument("image_dir", nargs="?", default="image", help="image root directory, default: image")
    parser.add_argument("-o", "--output", default="image_captions.jsonl", help="output JSONL file")
    parser.add_argument("--model", default=os.environ.get("QWEN_VL_MODEL", DEFAULT_MODEL), help="Qwen-VL model name")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN_VL_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL,
        help="Qwen-VL OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_VL_API_KEY"),
        help="API key; defaults to DASHSCOPE_API_KEY or QWEN_VL_API_KEY",
    )
    parser.add_argument("--prompt-file", default=None, help="optional custom prompt file")
    parser.add_argument("--limit", type=int, default=None, help="maximum number of images to process")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between API calls")
    parser.add_argument("--temperature", type=float, default=0.0, help="model temperature")
    parser.add_argument("--overwrite", action="store_true", help="overwrite output instead of resuming")
    parser.add_argument("--dry-run", action="store_true", help="print planned images without calling the API")
    args = parser.parse_args()
    args.model = (args.model or "").strip()
    args.base_url = (args.base_url or "").strip()
    args.api_key = (args.api_key or "").strip()

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"错误：图片目录不存在: {image_dir}", file=sys.stderr)
        return 1

    images = _iter_images(image_dir)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        print(f"未找到图片: {image_dir}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    done_paths = set() if args.overwrite else _load_done_paths(output_path)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT

    todo = [p for p in images if str(p.as_posix()) not in done_paths and str(p) not in done_paths]
    print(f"图片总数: {len(images)}，待处理: {len(todo)}，输出: {output_path}")

    if args.dry_run:
        for p in todo:
            print(f"pdf_page={_pdf_page_from_path(p)} image_path={p.as_posix()}")
        return 0

    if not args.api_key:
        print("错误：请设置 DASHSCOPE_API_KEY 或 QWEN_VL_API_KEY。", file=sys.stderr)
        return 1

    masked_key = args.api_key[:4] + "..." + args.api_key[-4:] if len(args.api_key) > 8 else "***"
    print(f"使用模型: {args.model}")
    print(f"使用 API: {args.base_url}")
    print(f"使用 Key: {masked_key}")

    client = _build_client(args.api_key, args.base_url)
    if client is None:
        return 1

    mode = "w" if args.overwrite else "a"
    with output_path.open(mode, encoding="utf-8") as f:
        for idx, image_path in enumerate(todo, 1):
            pdf_page = _pdf_page_from_path(image_path)
            rel_path = image_path.as_posix()
            try:
                payload = caption_image(client, args.model, image_path, prompt, args.temperature)
                item = {
                    "pdf_page": pdf_page,
                    "printed_page": "",
                    "image_path": rel_path,
                    "file_name": image_path.name,
                    "caption": payload["caption"],
                    "image_type": payload["image_type"],
                    "entities": payload["entities"],
                    "source": args.model,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{idx}/{len(todo)}] OK p{pdf_page}: {image_path.name}")
            except Exception as e:
                print(f"[{idx}/{len(todo)}] 失败: {image_path} - {e}", file=sys.stderr)
            if args.sleep > 0 and idx < len(todo):
                time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
