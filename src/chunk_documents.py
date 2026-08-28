"""
Stage 2 of the baseline RAG pipeline: Chunking.

Splits parsed Markdown documents (data/parsed/*.md) into overlapping,
token-sized chunks suitable for embedding. Uses a recursive splitter
(paragraph -> line -> sentence -> word) so chunks break on natural
boundaries instead of mid-sentence, sized by token count via tiktoken.

Usage:
    python src/chunk_documents.py                    # chunk everything in data/parsed
    python src/chunk_documents.py path/to/file.md     # chunk a single file
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import tiktoken

PARSED_DIR = Path(__file__).resolve().parent.parent / "data" / "parsed"
CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "chunks"

CHUNK_SIZE_TOKENS = 450
CHUNK_OVERLAP_TOKENS = 70
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

ENCODING = tiktoken.get_encoding("cl100k_base")

SOURCE_MARKER_RE = re.compile(r"<!-- source-file:(.*?) -->\n")
PAGE_MARKER_RE = re.compile(r"<!-- page:(\d+) -->\n?")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def _split_on_separator(text: str, separators: list[str]) -> list[str]:
    """Recursively split text using the first separator that fits, falling back to finer ones."""
    if not separators:
        return [text]

    separator, remaining_separators = separators[0], separators[1:]
    parts = text.split(separator) if separator else list(text)

    pieces: list[str] = []
    for part in parts:
        if not part:
            continue
        if count_tokens(part) > CHUNK_SIZE_TOKENS:
            pieces.extend(_split_on_separator(part, remaining_separators))
        else:
            pieces.append(part)
    return pieces


def _merge_with_overlap(pieces: list[str], separator_join: str = "\n\n") -> list[str]:
    """Greedily pack small pieces into chunks up to CHUNK_SIZE_TOKENS, carrying overlap forward."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = count_tokens(piece)

        if current and current_tokens + piece_tokens > CHUNK_SIZE_TOKENS:
            chunks.append(separator_join.join(current))

            # Carry trailing pieces forward as overlap for the next chunk.
            overlap: list[str] = []
            overlap_tokens = 0
            for prev_piece in reversed(current):
                prev_tokens = count_tokens(prev_piece)
                if overlap_tokens + prev_tokens > CHUNK_OVERLAP_TOKENS:
                    break
                overlap.insert(0, prev_piece)
                overlap_tokens += prev_tokens
            current, current_tokens = overlap, overlap_tokens

        current.append(piece)
        current_tokens += piece_tokens

    if current:
        chunks.append(separator_join.join(current))

    return chunks


def chunk_text(text: str) -> list[str]:
    pieces = _split_on_separator(text, SEPARATORS)
    return _merge_with_overlap(pieces)


def _extract_source_name(text: str, fallback: str) -> tuple[str, str]:
    """Pull the original filename left by parse_documents.py, stripping the marker from the text."""
    match = SOURCE_MARKER_RE.match(text)
    if not match:
        return fallback, text
    return match.group(1), text[match.end():]


def _split_into_pages(text: str) -> list[tuple[int | None, str]]:
    """Split on "<!-- page:N -->" markers so each resulting segment maps to one PDF page."""
    matches = list(PAGE_MARKER_RE.finditer(text))
    if not matches:
        return [(None, text)]

    pages = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pages.append((int(match.group(1)), text[start:end]))
    return pages


def chunk_file(source: Path, output_dir: Path) -> Path:
    raw_text = source.read_text(encoding="utf-8")
    original_source, raw_text = _extract_source_name(raw_text, fallback=source.name)
    pages = _split_into_pages(raw_text)

    output_path = output_dir / f"{source.stem}.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        chunk_index = 0
        for page_number, page_text in pages:
            for chunk in chunk_text(page_text):
                if not chunk.strip():
                    continue
                record = {
                    "id": f"{source.stem}::{chunk_index}",
                    "source": original_source,
                    "chunk_index": chunk_index,
                    "page": page_number,
                    "token_count": count_tokens(chunk),
                    "text": chunk,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_index += 1
    return output_path


def chunk_directory(input_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for source in sorted(input_dir.glob("*.md")):
        output_path = chunk_file(source, output_dir)
        with output_path.open(encoding="utf-8") as f:
            n_chunks = sum(1 for _ in f)
        outputs.append(output_path)
        print(f"[ok] {source.name} -> {output_path.relative_to(output_dir.parent.parent)} ({n_chunks} chunks)")
    return outputs


def main() -> None:
    if len(sys.argv) > 1:
        source = Path(sys.argv[1]).resolve()
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = chunk_file(source, CHUNKS_DIR)
        print(f"[ok] {source} -> {output_path}")
        return

    if not PARSED_DIR.exists() or not any(PARSED_DIR.glob("*.md")):
        print(f"No parsed Markdown files found in {PARSED_DIR}. Run src/parse_documents.py first.")
        return

    chunk_directory(PARSED_DIR, CHUNKS_DIR)


if __name__ == "__main__":
    main()
