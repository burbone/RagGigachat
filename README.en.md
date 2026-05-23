# RAG Assistant

> Upload a document — ask a question. Answers are grounded strictly in the file's content.

![Interface screenshot](screenshot.png)

[Русская версия](README.md)

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [CLI](#cli)
- [Environment Variables](#environment-variables)
- [Requirements](#requirements)

---

## Features

- **Formats:** PDF, DOCX, TXT, MD
- **Hybrid search:** BM25 + vector embeddings
- **Smart chunking:** parent-child strategy, tables are never split
- **Disk cache:** embeddings persist between sessions
- **Answer generation:** GigaChat

---

## Quick Start

```bash
git clone https://github.com/username/rag-assistant.git
cd rag-assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # add your GIGACHAT_CREDENTIALS

python main.py serve
```

Open [http://localhost:8000](http://localhost:8000)

---

## CLI

Index a document:

```bash
python main.py index document.pdf
```

Ask a question:

```bash
python main.py ask document.pdf "What is the commission?"
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|---|---|
| `GIGACHAT_CREDENTIALS` | GigaChat API key |
| `LOCAL_EMBEDDING_MODEL` | Embedding model name |
| `USE_RERANKER` | Enable reranker: `true` / `false` |

---

## Requirements

- Python 3.11+
- Dependencies: `requirements.txt`
