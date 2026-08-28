"""
Stage 1 of the baseline RAG pipeline: Document Parsing.

Uses Microsoft's MarkItDown (https://github.com/microsoft/markitdown) to
convert heterogeneous source documents (PDF, DOCX, PPTX, XLSX, HTML, ...)
into clean Markdown text, which is the ideal format for downstream
chunking + embedding steps in a RAG pipeline.

PDFs are handled specially: pages are extracted one at a time (reusing
MarkItDown's own per-page table/form detection) and tagged with
"<!-- page:N -->" markers, so later stages can cite the source page number.

Usage:
    python src/parse_documents.py                 # parse everything in data/raw
    python src/parse_documents.py path/to/file.pdf  # parse a single file
"""

from __future__ import annotations

import sys
from pathlib import Path

import ftfy
import pdfplumber
from markitdown import MarkItDown
from markitdown.converters._pdf_converter import (
    _extract_form_content_from_words,
    _merge_partial_numbering_lines,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PARSED_DIR = Path(__file__).resolve().parent.parent / "data" / "parsed"

SOURCE_MARKER = "<!-- source-file:{name} -->\n"
PAGE_MARKER = "<!-- page:{page} -->"


def _parse_pdf_with_page_markers(source: Path) -> str:
    """Extract PDF text page-by-page so each page can later be tagged with its number."""
    page_chunks: list[str] = []
    with pdfplumber.open(source) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            content = _extract_form_content_from_words(page) or (page.extract_text() or "")
            content = content.strip()
            if content:
                page_chunks.append(f"{PAGE_MARKER.format(page=page_idx)}\n{content}")
            page.close()
    return _merge_partial_numbering_lines("\n\n".join(page_chunks))


def parse_file(md: MarkItDown, source: Path, output_dir: Path) -> Path:
    """Convert a single document to Markdown and save it next to the others."""
    if source.suffix.lower() == ".pdf":
        text = _parse_pdf_with_page_markers(source)
    else:
        text = md.convert(str(source)).text_content

    # PDF extractors sometimes emit UTF-8 mis-decoded as cp1252 (e.g. Â®, â€™); ftfy repairs it.
    text = SOURCE_MARKER.format(name=source.name) + ftfy.fix_text(text)
    output_path = output_dir / f"{source.stem}.md"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def parse_directory(md: MarkItDown, input_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for source in sorted(input_dir.iterdir()):
        if source.is_file() and source.name != ".gitkeep":
            try:
                outputs.append(parse_file(md, source, output_dir))
                print(f"[ok]   {source.name} -> {outputs[-1].relative_to(output_dir.parent.parent)}")
            except Exception as exc:  # noqa: BLE001 - report and continue with other files
                print(f"[fail] {source.name}: {exc}")
    return outputs


def main() -> None:
    md = MarkItDown(enable_plugins=False)

    if len(sys.argv) > 1:
        source = Path(sys.argv[1]).resolve()
        PARSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = parse_file(md, source, PARSED_DIR)
        print(f"[ok] {source} -> {output_path}")
        return

    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(f"No files found in {RAW_DIR}. Add some documents there and re-run.")
        return

    parse_directory(md, RAW_DIR, PARSED_DIR)


if __name__ == "__main__":
    main()
