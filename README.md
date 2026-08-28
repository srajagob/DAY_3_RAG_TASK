# DAY_3_RAG_TASK

## Task 1: Baseline PDF RAG with ChromaDB

A minimal RAG pipeline: PDF ingestion (via **MarkItDown**, page-by-page so page numbers are
preserved) → chunking → embeddings → **ChromaDB** persistent vector store → retrieval →
source-cited answer generation.

### How it works

1. **`ingest.py`**
   - Splits each PDF into single pages using `pypdf` (this is what lets us track page numbers,
     since MarkItDown itself converts a whole file to Markdown without per-page boundaries).
   - Converts each single-page PDF to Markdown text with `MarkItDown`.
   - Chunks each page's text with `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 150).
   - Attaches metadata to every chunk: `source` (file name), `page`, `chunk_index`.
   - Embeds chunks locally with a `sentence-transformers` model (`all-MiniLM-L6-v2`) and
     upserts them into a persistent ChromaDB collection stored in `chroma_db/`.

2. **`query.py`**
   - Embeds the user question and retrieves the top-k (default 4) most similar chunks.
   - Builds a context block where every chunk is prefixed with its citation:
     `[Source: quarterly_report.pdf | Page 14]`.
   - Sends a prompt to the LLM instructing it to answer **only** from the context and to cite
     `[Source: <file> | Page <page>]` after every claim.
   - If `OPENAI_API_KEY` is not set, it prints the assembled prompt/context instead of calling
     an LLM, so the pipeline is fully runnable without any API key.

### Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and optionally set `OPENAI_API_KEY` to enable LLM-generated,
cited answers (otherwise the raw cited context is printed).

### Usage

```powershell
# 1. Drop your PDF files into data/pdfs/, then ingest them:
python ingest.py

# 2. Ask a question - answers will include [Source: file | Page N] citations:
python query.py "What was the revenue in Q3?"
```

### Project structure

```
config.py        # paths, chunking, embedding, and retrieval settings
ingest.py         # PDF -> per-page MarkItDown -> chunk -> embed -> ChromaDB
query.py          # retrieve top-k chunks -> cited context -> LLM answer
data/pdfs/        # put source PDFs here
chroma_db/        # persistent ChromaDB store (generated, gitignored)
```