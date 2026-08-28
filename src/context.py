"""
Shared context-formatting helpers for the generation stage.

Deterministically attaches source/page citations to retrieved chunks,
rather than relying on the LLM to invent or preserve them correctly.
"""

from __future__ import annotations


def format_citation(metadata: dict) -> str:
    """Render a chunk's metadata as e.g. "[Source: sdm_v1.pdf | Page 14]"."""
    source = metadata.get("source", "unknown")
    page = metadata.get("page")
    if page:
        return f"[Source: {source} | Page {page}]"
    return f"[Source: {source}]"


def build_context(documents: list[str], metadatas: list[dict]) -> str:
    """Join retrieved chunks into a single prompt-ready context block, each tagged with its citation."""
    blocks = [f"{format_citation(metadata)}\n{document}" for document, metadata in zip(documents, metadatas)]
    return "\n\n---\n\n".join(blocks)
