"""
Stateful RAG pipeline built with LangGraph.

Graph flow:
  retrieve → grade_documents ─┬─(sufficient relevance)──► generate → END
                               ├─(poor retrieval, retry)──► rewrite_query → retrieve
                               └─(all attempts exhausted)─► fallback → END

LLM backend (optional, in priority order):
  1. OpenAI  – set OPENAI_API_KEY  + OPENAI_MODEL  (default gpt-4o-mini)
  2. Ollama  – set OLLAMA_BASE_URL + OLLAMA_MODEL   (default llama3)
  3. Extractive (always available) – returns top-ranked chunks verbatim
"""

import os
import textwrap
from typing import TypedDict

import chromadb
from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer

# ── tunables ──────────────────────────────────────────────────────────────────
EMBED_MODEL         = "all-MiniLM-L6-v2"
CHROMA_DIR          = "./chroma_db"
COLLECTION          = "rag_documents"
TOP_K               = 6           # chunks fetched per retrieval attempt
RELEVANCE_THRESHOLD = 0.28        # cosine-similarity cutoff for a "relevant" chunk
MIN_RELEVANT        = 2           # minimum relevant chunks needed before generating
MAX_ATTEMPTS        = 2           # retrieval+rewrite cycles before fallback
# ─────────────────────────────────────────────────────────────────────────────


# ── shared singletons (loaded once) ──────────────────────────────────────────
_embed_model: SentenceTransformer | None = None
_chroma_collection = None


def _get_resources():
    global _embed_model, _chroma_collection
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )
    return _embed_model, _chroma_collection


# ── LLM loader (lazy, optional) ───────────────────────────────────────────────

def _load_llm():
    """Return an LLM or None if no backend is configured."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"[llm] using OpenAI model: {model}")
            return ChatOpenAI(model=model, temperature=0)
        except ImportError:
            pass

    if os.getenv("OLLAMA_BASE_URL"):
        try:
            from langchain_community.chat_models import ChatOllama
            model = os.getenv("OLLAMA_MODEL", "llama3")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            print(f"[llm] using Ollama model: {model} @ {base_url}")
            return ChatOllama(model=model, base_url=base_url)
        except ImportError:
            pass

    print("[llm] no LLM configured — using extractive mode")
    return None


# ── graph state ───────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question:          str
    active_query:      str              # may differ from question after rewrite
    documents:         list[dict]       # raw retrieved chunks with similarity scores
    relevant_docs:     list[dict]       # filtered subset that passed grading
    answer:            str
    sources:           dict             # {filename: full_path}
    attempt:           int
    fallback_triggered: bool


# ── node: retrieve ────────────────────────────────────────────────────────────

def retrieve(state: RAGState) -> dict:
    query = state.get("active_query") or state["question"]
    attempt = state.get("attempt", 0) + 1
    print(f"\n[retrieve] attempt {attempt} | query: {query!r}")

    model, collection = _get_resources()
    if collection.count() == 0:
        raise RuntimeError("ChromaDB collection is empty — run ingest first.")

    embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({
            "text":       text,
            "similarity": round(1 - dist, 4),
            "source":     meta.get("source", ""),
            "filename":   meta.get("filename", ""),
            "chunk_id":   meta.get("chunk_id", -1),
            "start_word": meta.get("start_word", 0),
            "end_word":   meta.get("end_word", 0),
        })

    print(f"  ↳ fetched {len(docs)} chunks  "
          f"(scores: {[d['similarity'] for d in docs]})")
    return {"documents": docs, "attempt": attempt, "active_query": query}


# ── node: grade_documents ─────────────────────────────────────────────────────

def grade_documents(state: RAGState) -> dict:
    docs = state["documents"]
    relevant = [d for d in docs if d["similarity"] >= RELEVANCE_THRESHOLD]
    print(f"[grade]    {len(relevant)}/{len(docs)} chunks pass threshold "
          f"(≥{RELEVANCE_THRESHOLD})")
    return {"relevant_docs": relevant}


# ── node: rewrite_query ───────────────────────────────────────────────────────

def rewrite_query(state: RAGState) -> dict:
    original  = state["question"]
    previous  = state.get("active_query", original)
    top_terms = " ".join(
        w for d in state["documents"][:2]
        for w in d["text"].split()[:10]
        if w.isalpha() and len(w) > 4
    )
    # Expand the query with salient terms from the top retrieved snippets
    expanded = f"{original} {top_terms}".strip()
    print(f"[rewrite]  '{previous}' → '{expanded[:80]}…'")
    return {"active_query": expanded}


# ── node: generate ────────────────────────────────────────────────────────────

def generate(state: RAGState) -> dict:
    docs     = state["relevant_docs"] or state["documents"][:MIN_RELEVANT]
    question = state["question"]
    llm      = _load_llm()

    sources = {}
    for d in docs:
        sources[d["filename"]] = d["source"]

    if llm:
        context = "\n\n---\n\n".join(
            f"[Source: {d['filename']} | chunk {d['chunk_id']} | "
            f"similarity {d['similarity']}]\n{d['text']}"
            for d in docs
        )
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=(
                "You are a precise assistant. Answer the question using ONLY "
                "the provided context. If the context is insufficient, say so. "
                "Always cite the source filename(s) at the end of your answer."
            )),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        response = llm.invoke(messages)
        answer = response.content
    else:
        # Extractive: stitch the top relevant passages together
        passages = []
        for d in docs:
            snippet = " ".join(d["text"].split()[:150])
            passages.append(
                f"[{d['filename']} | chunk {d['chunk_id']} | "
                f"similarity {d['similarity']}]\n{snippet}"
            )
        answer = (
            f"Most relevant passages for: \"{question}\"\n\n"
            + "\n\n".join(passages)
        )

    print(f"[generate] answer produced ({len(answer)} chars)")
    return {"answer": answer, "sources": sources, "fallback_triggered": False}


# ── node: fallback ────────────────────────────────────────────────────────────

def fallback(state: RAGState) -> dict:
    question = state["question"]
    best = state["documents"][:1]
    hint = ""
    if best:
        hint = (f"\n\nClosest match found (similarity={best[0]['similarity']}):\n"
                f"  File: {best[0]['filename']}  chunk {best[0]['chunk_id']}\n"
                f"  {' '.join(best[0]['text'].split()[:40])}…")
    answer = (
        f"Could not find sufficiently relevant content for: \"{question}\".\n"
        f"All {state['attempt']} retrieval attempt(s) returned chunks below the "
        f"relevance threshold ({RELEVANCE_THRESHOLD}).{hint}\n\n"
        f"Suggestions:\n"
        f"  • Rephrase your question with different keywords.\n"
        f"  • Lower RELEVANCE_THRESHOLD in rag_graph.py.\n"
        f"  • Ingest additional documents."
    )
    sources = {d["filename"]: d["source"] for d in state["documents"][:3]}
    print("[fallback] triggered — returning fallback response")
    return {"answer": answer, "sources": sources, "fallback_triggered": True}


# ── routing ───────────────────────────────────────────────────────────────────

def route_after_grading(state: RAGState) -> str:
    n_relevant = len(state.get("relevant_docs", []))
    attempt    = state.get("attempt", 1)

    if n_relevant >= MIN_RELEVANT:
        return "generate"
    if attempt < MAX_ATTEMPTS:
        return "rewrite_query"
    return "fallback"


# ── build graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(RAGState)

    g.add_node("retrieve",        retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query",   rewrite_query)
    g.add_node("generate",        generate)
    g.add_node("fallback",        fallback)

    g.add_edge(START,              "retrieve")
    g.add_edge("retrieve",         "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate":     "generate",
            "rewrite_query": "rewrite_query",
            "fallback":     "fallback",
        },
    )
    g.add_edge("rewrite_query",   "retrieve")
    g.add_edge("generate",        END)
    g.add_edge("fallback",        END)

    return g.compile()


# ── public API ────────────────────────────────────────────────────────────────

app = build_graph()


def ask(question: str) -> dict:
    """Run the full RAG graph and return the final state."""
    initial: RAGState = {
        "question":          question,
        "active_query":      question,
        "documents":         [],
        "relevant_docs":     [],
        "answer":            "",
        "sources":           {},
        "attempt":           0,
        "fallback_triggered": False,
    }
    return app.invoke(initial)


def format_answer(state: dict, width: int = 100) -> str:
    lines = ["=" * 70]
    tag = " [FALLBACK]" if state.get("fallback_triggered") else ""
    lines.append(f"QUESTION: {state['question']}{tag}")
    lines.append("=" * 70)
    lines.append(textwrap.fill(state["answer"], width=width))
    lines.append("\n" + "-" * 70)
    lines.append("SOURCES")
    lines.append("-" * 70)
    for fname, fpath in state["sources"].items():
        lines.append(f"  • {fname}")
        lines.append(f"    {fpath}")
    lines.append("=" * 70)
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python rag_graph.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result   = ask(question)
    print("\n" + format_answer(result))
