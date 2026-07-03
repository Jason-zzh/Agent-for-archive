# Long Document RAG Assistant

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

## Project Structure

```text
.
├── ai_qa.py                    # RAG 问答入口：加载向量库、检索 chunk、调用 LLM 回答
├── batch_qa.py                 # 批量问答：读取问题文件并输出 answers.txt
├── extract_answer_times.py     # 从批量问答结果中提取每题耗时
├── filter_short_name_images.py # 过滤图片目录中的短文件名图片路径
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

## Configuration

默认使用 OpenAI 兼容接口。可以使用 DeepSeek、OpenAI、Ollama 或其他兼容服务。

DeepSeek 示例：

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_API_BASE=https://api.deepseek.com
export LLM_MODEL=deepseek-chat
```

本地 Ollama 示例：

```bash
ollama pull deepseek-r1:14b
python ai_qa.py chroma_db --local
```

Windows PowerShell 示例：

```powershell
$env:OPENAI_API_KEY="sk-xxx"
$env:OPENAI_API_BASE="https://api.deepseek.com"
$env:LLM_MODEL="deepseek-chat"
```

## Quick Start

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

### 3. Probe Retrieval Quality

`query_probe.py` 用于检查某个问题会召回哪些文档片段，适合调试 embedding、chunk 切分和检索效果。

```bash
python query_probe.py chroma_db "文档的主要结论是什么？" -k 5
```

### 4. Batch QA

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

1. `pdf_processor.py` 使用 PyMuPDF 提取 PDF 文本和表格。
2. `text_cleaning.py` 清洗文本，减少页眉页脚、碎片和异常换行。
3. 文档被切分为适合 RAG 的 chunks。
4. chunks 使用 BGE embedding 转成向量并写入 Chroma。
5. `ai_qa.py` 接收问题后，先在多轮对话中把追问改写成完整问题。
6. Chroma 检索 top-k 相关 chunks。
7. 检索到的 chunks 连同页码一起注入 prompt。
8. LLM 只基于参考文档回答，并按模板要求标注页码或拒答。

## Prompt Customization

编辑 `prompt_template.py` 中的 `LONG_DOCUMENT_ANALYSIS_PROMPT` 可以改变助手角色、语气、拒答规则和回答格式。

默认模板强调：

- 仅依据参考文档作答；
- 文档无相关内容时明确拒答；
- 回答需要标注来源页码；
- 长文总结应按主题、章节或逻辑层次组织。

## Generated Files

以下文件或目录通常是运行产物，不建议提交到 Git：

- `chroma_db/`
- `image/`
- `answers.txt`
- `result.txt`
- `result_raw.txt`
- `answer_times.csv`
- `.env`

`.gitignore` 已默认忽略这些内容。

## License

This project is licensed under the MIT License. See `LICENSE` for details.
