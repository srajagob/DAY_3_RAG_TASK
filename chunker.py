from markitdown import MarkItDown


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Sliding window chunking: each chunk is chunk_size words, advancing by (chunk_size - overlap) words."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    step = chunk_size - overlap
    chunks = []
    chunk_id = 0
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(chunk_words),
            "word_count": len(chunk_words),
            "start_word": start,
            "end_word": min(end, len(words)),
        })
        chunk_id += 1
        start += step

    return chunks


def chunk_file(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Convert a file to markdown via MarkItDown, then apply sliding window chunking."""
    md = MarkItDown()
    result = md.convert(file_path)
    text = result.text_content
    chunks = chunk_text(text, chunk_size, overlap)
    for chunk in chunks:
        chunk["source"] = file_path
    return chunks


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <file_path> [chunk_size] [overlap]")
        print("       chunk_size defaults to 1000 words")
        print("       overlap    defaults to 200 words")
        sys.exit(1)

    file_path = sys.argv[1]
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    chunks = chunk_file(file_path, chunk_size, overlap)
    print(f"Total chunks: {len(chunks)}  (chunk_size={chunk_size}, overlap={overlap})\n")
    for chunk in chunks:
        print(json.dumps(chunk, indent=2))
        print("-" * 60)
