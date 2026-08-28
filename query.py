"""
Task 1: Baseline PDF RAG - Retrieval + generation with source attribution.

Usage:
    python query.py "What was the revenue in Q3?"
"""
import sys

from ingest import get_collection
from config import OPENAI_API_KEY, OPENAI_MODEL, TOP_K

CITATION_TEMPLATE = "[Source: {source} | Page {page}]"

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the provided context.
Every claim in your answer MUST be followed by its citation in the form [Source: <file> | Page <page>].
If the answer is not contained in the context, say you don't know.

Context:
{context}

Question: {question}

Answer (with citations):"""


def retrieve(query: str, top_k: int = TOP_K):
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "score": 1 - dist,  # cosine similarity from distance
            }
        )
    return chunks


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for chunk in chunks:
        citation = CITATION_TEMPLATE.format(source=chunk["source"], page=chunk["page"])
        blocks.append(f"{citation}\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    if not OPENAI_API_KEY:
        return (
            "[No OPENAI_API_KEY set - returning assembled context instead of an LLM answer]\n\n"
            + prompt
        )

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def answer_query(question: str) -> str:
    chunks = retrieve(question)
    if not chunks:
        return "No relevant chunks found. Have you run ingest.py yet?"
    return generate_answer(question, chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question"')
        sys.exit(1)

    user_question = " ".join(sys.argv[1:])
    print(answer_query(user_question))
