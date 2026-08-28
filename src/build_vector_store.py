"""
Stage 4 of the baseline RAG pipeline: Vector Storage.

Loads the chunks (data/chunks/*.jsonl) and their precomputed embeddings
(data/embeddings/*.npy, from src/embed_documents.py) into a persistent
Chroma (https://github.com/chroma-core/chroma) collection, ready for
similarity search in the retrieval stage.

Usage:
    python src/build_vector_store.py                  # (re)build the collection from data/chunks + data/embeddings
    python src/build_vector_store.py --query "..."     # sanity-check search against the built collection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from context import format_citation

CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "chunks"
EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"
VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store"

COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_collection(client: chromadb.ClientAPI):
    # Embeddings are already L2-normalized, so cosine and dot-product search are equivalent.
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def build(client: chromadb.ClientAPI) -> None:
    collection = get_collection(client)

    for chunk_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        embedding_path = EMBEDDINGS_DIR / f"{chunk_path.stem}.npy"
        if not embedding_path.exists():
            print(f"[skip] {chunk_path.name}: no matching embeddings at {embedding_path.name}. Run src/embed_documents.py first.")
            continue

        records = load_chunks(chunk_path)
        embeddings = np.load(embedding_path)
        if len(records) != embeddings.shape[0]:
            print(f"[skip] {chunk_path.name}: {len(records)} chunks but {embeddings.shape[0]} embeddings (out of sync).")
            continue

        collection.upsert(
            ids=[r["id"] for r in records],
            embeddings=embeddings.tolist(),
            documents=[r["text"] for r in records],
            metadatas=[
                {"source": r["source"], "chunk_index": r["chunk_index"], "page": r.get("page") or 0}
                for r in records
            ],
        )
        print(f"[ok] {chunk_path.name} -> collection '{COLLECTION_NAME}' ({len(records)} chunks)")

    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} vectors total.")


def query(client: chromadb.ClientAPI, text: str, top_k: int = 5) -> None:
    results = retrieve(client, text, top_k)
    for rank, r in enumerate(results, start=1):
        print(f"\n#{rank} [{r['id']}] {format_citation(r['metadata'])} distance={r['distance']:.4f}")
        print(r["document"][:300].replace("\n", " ") + ("..." if len(r["document"]) > 300 else ""))


def retrieve(client: chromadb.ClientAPI, text: str, top_k: int = 5, model: SentenceTransformer | None = None) -> list[dict]:
    """Embed `text` and return the top_k nearest chunks as {id, document, metadata, distance} dicts."""
    collection = get_collection(client)
    model = model or SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([text], normalize_embeddings=True).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return [
        {"id": chunk_id, "document": document, "metadata": metadata, "distance": distance}
        for chunk_id, document, metadata, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Run a sanity-check similarity search instead of (re)building the store")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    if args.query:
        query(client, args.query, args.top_k)
        return

    build(client)


if __name__ == "__main__":
    main()
