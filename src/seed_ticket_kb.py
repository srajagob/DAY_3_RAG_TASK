"""
Seeds the `ticket_kb` Chroma collection with a handful of historical
resolved-ticket / RCA entries so ticket_resolver_graph.py's RAG retrieval
step has something realistic to match against.

Usage:
    python src/seed_ticket_kb.py
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store"
TICKET_KB_COLLECTION = "ticket_kb"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

KB_ENTRIES = [
    {
        "id": "kb-tcl-001",
        "source": "RCA-2025-114 (Tcl scripting)",
        "text": (
            "Symptom: PD flow Tcl script raised 'invalid command name \"print\"' during STA report generation. "
            "Root cause: engineer used Python-style `print` instead of Tcl's `puts` command; Tcl has no built-in "
            "`print` proc. Fix: replace every `print ...` call with `puts ...`. Validated by re-running the script "
            "against the same testcase with no remaining 'invalid command name' errors."
        ),
    },
    {
        "id": "kb-tcl-002",
        "source": "RCA-2024-087 (Tcl scripting)",
        "text": (
            "Symptom: floorplan init Tcl proc failed with 'wrong # args: should be \"init_floorplan die_w die_h "
            "util\"'. Root cause: proc argument list was missing a closing brace, so Tcl treated the body as part "
            "of the argument list. Fix: close the arg list properly: `proc init_floorplan {die_w die_h util} { ... }`."
        ),
    },
    {
        "id": "kb-python-001",
        "source": "RCA-2025-042 (Python automation)",
        "text": (
            "Symptom: nightly regression Python script crashed with IndentationError on a dict comprehension. "
            "Root cause: mixed tabs and spaces after a copy-paste from a wiki page. Fix: normalize to 4-space "
            "indentation throughout the file; re-ran the script successfully."
        ),
    },
    {
        "id": "kb-cpp-001",
        "source": "RCA-2024-201 (C++ timing model)",
        "text": (
            "Symptom: timing-model C++ plugin failed to compile with 'expected ';' before '}' token'. Root cause: "
            "a missing semicolon after a struct definition used elsewhere in the same header. Fix: add the missing "
            "semicolon after the struct closing brace."
        ),
    },
    {
        "id": "kb-network-001",
        "source": "RCA-2023-330 (IT infrastructure)",
        "text": (
            "Symptom: users could not reach internal VPN gateway while traveling. Root cause: expired VPN client "
            "certificate. Fix: IT re-issued the certificate; not a coding issue, handled by desktop support."
        ),
    },
]


def main() -> None:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = client.get_or_create_collection(name=TICKET_KB_COLLECTION, metadata={"hnsw:space": "cosine"})

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([e["text"] for e in KB_ENTRIES], normalize_embeddings=True).tolist()

    collection.upsert(
        ids=[e["id"] for e in KB_ENTRIES],
        embeddings=embeddings,
        documents=[e["text"] for e in KB_ENTRIES],
        metadatas=[{"source": e["source"]} for e in KB_ENTRIES],
    )
    print(f"Seeded '{TICKET_KB_COLLECTION}' with {collection.count()} entries at {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
