"""
Shared configuration for the baseline PDF RAG pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "pdf_rag_baseline"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Embeddings (local, no API key required)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Retrieval
TOP_K = 4

# LLM (optional - only used if OPENAI_API_KEY is set)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PDF_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
