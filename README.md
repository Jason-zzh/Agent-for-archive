# Long Document RAG Assistant

Agent for Archive is a local long-document RAG assistant for PDF-based question answering. It extracts paragraphs, tables, and optional figures from PDFs, cleans and chunks the content, stores embeddings in Chroma, and answers questions with OpenAI-compatible LLMs through LangChain.

一个面向长文档的本地 RAG 问答工具。它可以从 PDF 中提取段落、表格和可选图片，清洗并切分文本，写入 Chroma 向量库，然后通过 LangChain 和 OpenAI 兼容模型进行基于文档的问答。

## Features

- PDF 段落和表格提取，基于 PyMuPDF。
- 可选提取带 caption 的 PDF 内嵌图片。
- 文本清洗：去除异常换行、过滤页眉页脚、合并短文本片段。
- 智能切片：按段落、句子和字符边界切分长文档。
- Chroma 向量库入库，使用 `BAAI/bge-large-zh-v1.5` embedding。
- RAG 问答：支持单次问答和多轮对话。
- 历史感知检索：可把“它”“这个”等追问改写成完整问题后再检索。
- 可打印检索到的 chunk，便于调试 embedding 和召回效果。
- Prompt 模板独立在 `prompt_template.py` 中，便于修改成不同领域的长文档解析助手。

## Use Cases

- Technical manuals and engineering documents
- Academic papers and reports
- Long PDF archives
- Course materials and project documentation
- Internal knowledge-base documents

## Project Structure

```text
.
├── ai_qa.py                    # RAG 问答入口：加载向量库、检索 chunk、调用 LLM 回答
├── batch_qa.py                 # 批量问答：读取问题文件并输出 answers.txt
├── extract_answer_times.py     # 从批量问答结果中提取每题耗时
├── filter_short_name_images.py # 过滤图片目录中的短文件名图片路径
├── image_captioner.py          # Phase 1：调用视觉模型生成图片 caption JSONL
├── multimodal_ingest.py        # Phase 2：将图片 caption 写入 Chroma，支持图文检索
├── pdf_processor.py            # PDF 解析、清洗、切片、向量化入库
├── prompt_template.py          # 通用长文档解析 prompt 模板
├── query_probe.py              # 检索探针：查看某个问题召回了哪些 chunk
├── requirements.txt            # Python 依赖
├── text_cleaning.py            # 文本清洗工具
└── LICENSE                     # MIT license
```

## Installation

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

国内环境下，脚本默认设置了 HuggingFace 镜像：

```python
HF_ENDPOINT=https://hf-mirror.com
```

The first run may download the `BAAI/bge-large-zh-v1.5` embedding model. Make sure that HuggingFace access or the configured mirror is available.

首次运行时可能会下载 `BAAI/bge-large-zh-v1.5` embedding 模型，请确保 HuggingFace 或镜像站点可访问。

## Configuration

默认使用 OpenAI 兼容接口。远程服务的 API 地址、模型名和 API key 都需要通过环境变量或命令行参数提供。

远程 API 示例：

```bash
export OPENAI_API_KEY=your-api-key
export OPENAI_API_BASE=https://your-openai-compatible-endpoint/v1
export LLM_MODEL=your-model-name
```

本地 Ollama 示例：

```bash
ollama pull deepseek-r1:14b
python ai_qa.py chroma_db --local
```

Windows PowerShell 示例：

```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_API_BASE="https://your-openai-compatible-endpoint/v1"
$env:LLM_MODEL="your-model-name"
```

You can also keep API settings in a `.env` file:

```env
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name
```

当前脚本不会自动加载 `.env`，请先用你的 shell、`direnv` 或其他环境变量工具加载后再运行。`.env` 已被 `.gitignore` 忽略，避免误提交密钥。

## Quick Start

Minimal workflow:

```bash
python pdf_processor.py document.pdf --vector-db chroma_db -o result.txt
python ai_qa.py chroma_db -q "请总结这份文档的核心内容"
```

With image captions:

```bash
python image_captioner.py image -o image_captions.jsonl
python multimodal_ingest.py image_captions.jsonl --vector-db chroma_db
python ai_qa.py chroma_db -q "文档中的关键界面截图说明了什么？"
```

### 1. Process a PDF

仅解析并输出到控制台：

```bash
python pdf_processor.py document.pdf
```

解析并保存完整文本结果：

```bash
python pdf_processor.py document.pdf -o result.txt
```

解析、清洗、切片，并写入 Chroma 向量库：

```bash
python pdf_processor.py document.pdf --vector-db chroma_db -o result.txt
```

可选：提取带 caption 的图片：

```bash
python pdf_processor.py document.pdf --raw-images -i image -o result.txt
```

### 2. Ask Questions

单次问答：

```bash
python ai_qa.py chroma_db -q "请总结这份文档的核心内容"
```

交互式多轮问答：

```bash
python ai_qa.py chroma_db
```

查看每次回答前检索到的文档片段：

```bash
python ai_qa.py chroma_db --show-chunks
```

多模态 QA 会把每条检索结果以来源块形式注入模型，包含页码、类型、模态和图片路径。引用图片证据时，回答应同时给出图片路径。

### 3. Probe Retrieval Quality

`query_probe.py` 用于检查某个问题会召回哪些文档片段，适合调试 embedding、chunk 切分和检索效果。

```bash
python query_probe.py chroma_db "文档的主要结论是什么？" -k 5
```

多模态入库后，探针结果会同时展示页码、chunk 类型、模态和图片路径，便于确认问题是否召回了 `FigureCaption`：

```text
#1  距离=0.3201  页=13  类型=FigureCaption  模态=image  顺序=4
    bbox=120.50,88.00,420.00,260.25
    图片路径=image/p13/system_architecture.png
    【图片内容】...
```

### 4. Ingest Image Captions

如果已经通过 `image_captioner.py` 生成了 `image_captions.jsonl`，可以把图片说明追加写入同一个 Chroma collection：

```bash
python multimodal_ingest.py image_captions.jsonl --vector-db chroma_db
```

先预览不入库：

```bash
python multimodal_ingest.py image_captions.jsonl --dry-run --limit 5
```

### 5. Batch QA

准备一个问题文件，每行一个问题：

```text
请总结文档结构
文档中提到的关键流程有哪些？
```

运行批量问答：

```bash
python batch_qa.py chroma_db questions.txt -o answers.txt
```

提取每题耗时：

```bash
python extract_answer_times.py answers.txt --stats --csv answer_times.csv
```

## How It Works

```text
PDF
 ↓
PyMuPDF extraction
 ↓
Text cleaning
 ↓
Chunking
 ↓
BGE embedding
 ↓
Chroma vector store
 ↓
Retriever
 ↓
LLM answer with page references
```

1. `pdf_processor.py` 使用 PyMuPDF 提取 PDF 文本和表格。
2. `text_cleaning.py` 清洗文本，减少页眉页脚、碎片和异常换行。
3. 文档被切分为适合 RAG 的 chunks。
4. 文本和表格 chunk 保留版面 metadata，包括 `bbox`、`reading_order`、`page_width` 和 `page_height`。
5. chunks 使用 BGE embedding 转成向量并写入 Chroma。
6. `ai_qa.py` 接收问题后，先在多轮对话中把追问改写成完整问题。
7. Chroma 检索 top-k 相关 chunks。
8. 检索到的 chunks 连同页码、类型、模态和图片路径一起注入 prompt。
9. LLM 只基于参考文档回答，并按模板要求标注页码、图片证据或拒答。

Layout metadata uses Chroma-safe primitive values. For example, `bbox` is stored as a comma-separated string, while `bbox_x0`, `bbox_y0`, `bbox_x1`, and `bbox_y1` are also stored as numeric fields for future filtering or layout-aware ranking.

## Prompt Customization

编辑 `prompt_template.py` 中的 `LONG_DOCUMENT_ANALYSIS_PROMPT` 可以改变助手角色、语气、拒答规则和回答格式。

默认模板强调：

- 仅依据参考文档作答；
- 文档无相关内容时明确拒答；
- 回答需要标注来源页码；
- 长文总结应按主题、章节或逻辑层次组织。

## Retrieval Debugging

Use `--show-chunks` or `query_probe.py` to inspect retrieved chunks before trusting an answer. This is useful for diagnosing chunking issues, embedding mismatch, and poor recall.

```bash
python ai_qa.py chroma_db --show-chunks
python query_probe.py chroma_db "文档的主要结论是什么？" -k 5
```

## Limitations

- The assistant is only as reliable as the retrieved chunks. Poor chunking or weak retrieval can lead to incomplete answers.
- Scanned PDFs require extractable text or OCR handled before ingestion; this project currently focuses on PyMuPDF-based extraction.
- Very large documents may require tuning `--chunk-size`, `--chunk-overlap`, and `-k`.
- API availability, rate limits, and model behavior depend on the OpenAI-compatible provider you configure.

## Acknowledgement

This project uses PyMuPDF for PDF parsing, Chroma for vector storage, BGE embeddings for retrieval, and LangChain for RAG orchestration.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
