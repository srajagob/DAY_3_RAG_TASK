"""
Stage 3 of the baseline RAG pipeline: Embeddings.

Uses sentence-transformers (https://github.com/UKPLab/sentence-transformers)
to embed each chunk produced by src/chunk_documents.py into a dense vector,
ready for storage in a vector index/database in the next stage.

Default model: sentence-transformers/all-MiniLM-L6-v2
- 384-dim, fast on CPU, strong general-purpose baseline for RAG.
- Swap via --model for higher quality, e.g. BAAI/bge-base-en-v1.5 (768-dim).

Usage:
    python src/embed_documents.py                      # embed everything in data/chunks
    python src/embed_documents.py path/to/chunks.jsonl  # embed a single chunk file
    python src/embed_documents.py --model BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "chunks"
EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed_file(model: SentenceTransformer, source: Path, output_dir: Path) -> Path:
    records = load_chunks(source)
    texts = [r["text"] for r in records]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,  # so cosine similarity == dot product
        convert_to_numpy=True,
    ).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    np.save(output_dir / f"{stem}.npy", embeddings)

    # ids/metadata kept alongside so a vector index can be rebuilt with row alignment.
    with (output_dir / f"{stem}.meta.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"id": r["id"], "source": r["source"], "chunk_index": r["chunk_index"]}) + "\n")

    return output_dir / f"{stem}.npy"


def embed_directory(model: SentenceTransformer, input_dir: Path, output_dir: Path) -> None:
    for source in sorted(input_dir.glob("*.jsonl")):
        if source.name.endswith(".meta.jsonl"):
            continue
        start = time.perf_counter()
        output_path = embed_file(model, source, output_dir)
        elapsed = time.perf_counter() - start
        n = np.load(output_path).shape[0]
        print(f"[ok] {source.name} -> {output_path.relative_to(output_dir.parent.parent)} ({n} vectors, {elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="Single .jsonl chunk file to embed")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    print(f"Embedding dimension: {model.get_embedding_dimension()}")

    if args.path:
        source = Path(args.path).resolve()
        output_path = embed_file(model, source, EMBEDDINGS_DIR)
        print(f"[ok] {source} -> {output_path}")
        return

    if not CHUNKS_DIR.exists() or not any(CHUNKS_DIR.glob("*.jsonl")):
        print(f"No chunk files found in {CHUNKS_DIR}. Run src/chunk_documents.py first.")
        return

    embed_directory(model, CHUNKS_DIR, EMBEDDINGS_DIR)


if __name__ == "__main__":
    main()
