"""
Task 1: Baseline PDF RAG - Ingestion pipeline.

Pipeline:
1. Split each source PDF into individual single-page PDFs (so page numbers are preserved).
2. Parse each page's content to Markdown text using MarkItDown.
3. Chunk each page's text with a recursive character splitter.
4. Attach metadata (source file name, page number, chunk index) to every chunk.
5. Embed chunks and upsert them into a persistent ChromaDB collection.

Usage:
    python ingest.py                 # ingest all PDFs in data/pdfs/
    python ingest.py path/to/one.pdf # ingest a single PDF
"""
import sys
import tempfile
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown
from pypdf import PdfReader, PdfWriter

from config import (
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PDF_DIR,
)

_md = MarkItDown(enable_plugins=False)


def extract_pages_markdown(pdf_path: Path) -> list[str]:
    """Split a PDF into single pages and convert each page to markdown text via MarkItDown."""
    reader = PdfReader(str(pdf_path))
    page_texts: list[str] = []
    total_pages = len(reader.pages)

    for page_index in range(total_pages):
        writer = PdfWriter()
        writer.add_page(reader.pages[page_index])

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            writer.write(tmp)
            tmp_path = Path(tmp.name)

        try:
            result = _md.convert(str(tmp_path))
            page_texts.append(result.text_content or "")
        finally:
            tmp_path.unlink(missing_ok=True)

        if (page_index + 1) % 10 == 0 or (page_index + 1) == total_pages:
            print(f"  Parsed page {page_index + 1}/{total_pages}", flush=True)

    return page_texts


def chunk_pdf(pdf_path: Path) -> tuple[list[str], list[dict], list[str]]:
    """Chunk a single PDF, returning (documents, metadatas, ids) ready for ChromaDB."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    page_texts = extract_pages_markdown(pdf_path)
    for page_number, page_text in enumerate(page_texts, start=1):
        if not page_text.strip():
            continue

        chunks = splitter.split_text(page_text)
        for chunk_index, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk_index": chunk_index,
                }
            )
            ids.append(f"{pdf_path.stem}_p{page_number}_c{chunk_index}")

    return documents, metadatas, ids


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_pdfs(pdf_paths: list[Path]) -> None:
    collection = get_collection()

    for pdf_path in pdf_paths:
        print(f"Ingesting: {pdf_path.name}")
        documents, metadatas, ids = chunk_pdf(pdf_path)
        if not documents:
            print(f"  No extractable text found in {pdf_path.name}, skipping.")
            continue

        # Upsert in batches to avoid overly large single calls.
        batch_size = 100
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            collection.upsert(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
        print(f"  Indexed {len(documents)} chunks across {len(set(m['page'] for m in metadatas))} pages.")

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = [Path(arg) for arg in sys.argv[1:]]
    else:
        paths = sorted(PDF_DIR.glob("*.pdf"))

    if not paths:
        print(f"No PDF files found. Place .pdf files in {PDF_DIR} or pass a path as an argument.")
        sys.exit(1)

    ingest_pdfs(paths)
