"""
Baseline RAG pipeline:
  ingest()  — convert PDFs → chunks → embeddings → ChromaDB
  query()   — embed question, retrieve top-k chunks, return answer with sources
"""

import os
import sys
import textwrap
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import chunk_file

# ── config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"   # fast, 384-dim, good for retrieval
COLLECTION    = "rag_documents"
CHROMA_DIR    = "./chroma_db"
CHUNK_SIZE    = 1000
OVERLAP       = 200
TOP_K         = 5
# ─────────────────────────────────────────────────────────────────────────────


def _get_collection() -> tuple[chromadb.Collection, SentenceTransformer]:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    model = SentenceTransformer(EMBED_MODEL)
    return collection, model


# ── ingestion ─────────────────────────────────────────────────────────────────

def ingest(paths: list[str], chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> int:
    """
    Ingest one or more files into ChromaDB.
    Returns the total number of chunks stored.
    """
    collection, model = _get_collection()

    total = 0
    for path in paths:
        path = str(Path(path).resolve())
        print(f"[ingest] processing: {path}")

        chunks = chunk_file(path, chunk_size, overlap)
        if not chunks:
            print(f"  ↳ no text extracted, skipping.")
            continue

        texts     = [c["text"] for c in chunks]
        # stable IDs: source path + chunk index prevents re-ingesting duplicates
        ids       = [f"{path}::chunk_{c['chunk_id']}" for c in chunks]
        metadatas = [
            {
                "source":     path,
                "filename":   os.path.basename(path),
                "chunk_id":   c["chunk_id"],
                "start_word": c["start_word"],
                "end_word":   c["end_word"],
                "word_count": c["word_count"],
            }
            for c in chunks
        ]

        print(f"  ↳ embedding {len(texts)} chunks …")
        embeddings = model.encode(texts, show_progress_bar=True).tolist()

        # upsert so re-running is safe
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total += len(chunks)
        print(f"  ↳ stored {len(chunks)} chunks from '{os.path.basename(path)}'")

    print(f"\n[ingest] done — {total} total chunks in collection '{COLLECTION}'")
    return total


# ── retrieval ─────────────────────────────────────────────────────────────────

def query(question: str, top_k: int = TOP_K) -> dict:
    """
    Retrieve the top-k most relevant chunks for `question`.
    Returns a dict with 'question', 'chunks', and 'sources'.
    """
    collection, model = _get_collection()

    if collection.count() == 0:
        raise RuntimeError("Collection is empty — run ingest() first.")

    q_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    seen_sources = {}
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1 - dist, 4)   # cosine distance → similarity
        chunks.append({
            "text":       doc,
            "similarity": similarity,
            "source":     meta["source"],
            "filename":   meta["filename"],
            "chunk_id":   meta["chunk_id"],
            "start_word": meta["start_word"],
            "end_word":   meta["end_word"],
        })
        seen_sources[meta["filename"]] = meta["source"]

    return {
        "question": question,
        "chunks":   chunks,
        "sources":  seen_sources,   # {filename: full_path}
    }


def format_result(result: dict, max_text_width: int = 100) -> str:
    """Pretty-print a query result with attributed sources."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"QUESTION: {result['question']}")
    lines.append("=" * 70)

    for i, chunk in enumerate(result["chunks"], 1):
        lines.append(f"\n[{i}] similarity={chunk['similarity']:.4f}  "
                     f"file={chunk['filename']}  chunk={chunk['chunk_id']}  "
                     f"words {chunk['start_word']}–{chunk['end_word']}")
        lines.append("-" * 70)
        wrapped = textwrap.fill(chunk["text"], width=max_text_width)
        lines.append(wrapped)

    lines.append("\n" + "=" * 70)
    lines.append("SOURCES")
    lines.append("=" * 70)
    for fname, fpath in result["sources"].items():
        lines.append(f"  • {fname}")
        lines.append(f"    {fpath}")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _usage():
    print(
        "Usage:\n"
        "  python rag_pipeline.py ingest <file1.pdf> [file2.pdf ...]\n"
        "  python rag_pipeline.py query  \"your question here\"\n"
        "  python rag_pipeline.py query  \"your question here\" --top-k 3\n"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "ingest":
        files = sys.argv[2:]
        ingest(files)

    elif command == "query":
        question = sys.argv[2]
        top_k = TOP_K
        if "--top-k" in sys.argv:
            top_k = int(sys.argv[sys.argv.index("--top-k") + 1])
        result = query(question, top_k=top_k)
        print(format_result(result))

    else:
        _usage()
        sys.exit(1)
