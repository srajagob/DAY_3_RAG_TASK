"""
Corrective RAG (CRAG) graph, built with LangGraph on top of the baseline pipeline.

Extends plain retrieval-then-generate with a self-correction loop:

    retrieve -> grade_documents -> [relevant?] -> generate
                                 -> [irrelevant] -> transform_query -> web_search -> generate

An LLM (local Ollama model) grades each retrieved chunk for relevance to the
question. If too few chunks survive grading, the query is rewritten for web
search and a DuckDuckGo fallback fills in for the missing context. The
generation node always answers using the shared context.build_context()
formatter, so citations ("[Source: ... | Page ...]" or "[Source: web:...]")
are attached deterministically rather than left to the LLM.

Usage:
    python src/crag_graph.py "your question here"
"""

from __future__ import annotations

import argparse
import os
from typing import Literal, TypedDict

import chromadb
from ddgs import DDGS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from build_vector_store import EMBEDDING_MODEL, VECTOR_STORE_DIR, retrieve
from context import build_context, format_citation

# Corporate HTTP(S)_PROXY env vars intercept localhost traffic too; exempt Ollama's local server.
for _no_proxy_var in ("NO_PROXY", "no_proxy"):
    os.environ[_no_proxy_var] = ",".join(
        filter(None, [os.environ.get(_no_proxy_var, ""), "localhost", "127.0.0.1"])
    )

OLLAMA_MODEL = "llama3.1:8b"
RETRIEVAL_TOP_K = 5
MIN_RELEVANT_CHUNKS = 1  # below this, fall back to web search
WEB_SEARCH_RESULTS = 3


class Document(TypedDict):
    id: str
    document: str
    metadata: dict
    distance: float


class GraphState(TypedDict):
    query: str
    documents: list[Document]
    relevance_score: float  # fraction of retrieved chunks graded relevant, 0.0-1.0
    web_search_flag: bool
    generation: str


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)


class GradeDocument(BaseModel):
    """Binary relevance grade for a single retrieved chunk."""

    binary_score: Literal["yes", "no"] = Field(description="'yes' if the chunk is relevant to the question, else 'no'")


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether a retrieved document chunk is relevant to a user question. "
            "Give a binary score 'yes' or 'no'. 'yes' means the chunk contains information that helps answer "
            "the question, even partially. Be lenient: if the chunk is topically related, grade it 'yes'.",
        ),
        ("human", "Retrieved chunk:\n\n{document}\n\nUser question: {query}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite user questions into short, keyword-focused web search queries. "
            "Output ONLY the rewritten query, nothing else.",
        ),
        ("human", "{query}"),
    ]
)

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a technical assistant. Answer the user's question using ONLY the provided context. "
            "Each context block starts with a citation tag like '[Source: file.pdf | Page N]' or "
            "'[Source: web:URL]'. When you use information from a block, you MUST keep its citation tag "
            "inline next to the claim it supports, exactly as written. If the context does not contain the "
            "answer, say you don't know instead of guessing.",
        ),
        ("human", "Context:\n\n{context}\n\nQuestion: {query}"),
    ]
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: GraphState) -> GraphState:
    """Fetch candidate chunks from the Chroma vector store built in src/build_vector_store.py."""
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    documents = retrieve(client, state["query"], top_k=RETRIEVAL_TOP_K)
    return {**state, "documents": documents}


def grade_documents_node(state: GraphState) -> GraphState:
    """Grade each retrieved chunk for relevance with an LLM and keep only the relevant ones."""
    grader = _llm().with_structured_output(GradeDocument)

    relevant_documents: list[Document] = []
    for doc in state["documents"]:
        grade = grader.invoke(GRADE_PROMPT.format_messages(document=doc["document"], query=state["query"]))
        if grade.binary_score == "yes":
            relevant_documents.append(doc)

    total = len(state["documents"])
    relevance_score = (len(relevant_documents) / total) if total else 0.0
    web_search_flag = len(relevant_documents) < MIN_RELEVANT_CHUNKS

    print(f"[grade] {len(relevant_documents)}/{total} chunks graded relevant (score={relevance_score:.2f})")

    return {
        **state,
        "documents": relevant_documents,
        "relevance_score": relevance_score,
        "web_search_flag": web_search_flag,
    }


def decide_to_generate(state: GraphState) -> Literal["generate", "transform_query"]:
    """Conditional edge: skip straight to generation if we have relevant context, else self-correct."""
    return "transform_query" if state["web_search_flag"] else "generate"


def transform_query_node(state: GraphState) -> GraphState:
    """Rewrite the user's question into a better web-search query."""
    rewritten = _llm().invoke(REWRITE_PROMPT.format_messages(query=state["query"])).content.strip()
    print(f"[rewrite] '{state['query']}' -> '{rewritten}'")
    return {**state, "query": rewritten}


def web_search_node(state: GraphState) -> GraphState:
    """Fallback retrieval: search the web (DuckDuckGo) when local chunks were graded irrelevant."""
    with DDGS() as ddgs:
        results = list(ddgs.text(state["query"], max_results=WEB_SEARCH_RESULTS))

    web_documents: list[Document] = [
        {
            "id": f"web::{i}",
            "document": r.get("body", ""),
            "metadata": {"source": f"web:{r.get('href', 'unknown')}", "page": 0},
            "distance": 0.0,
        }
        for i, r in enumerate(results)
    ]
    print(f"[web-search] found {len(web_documents)} results for fallback context")
    return {**state, "documents": state["documents"] + web_documents}


def generate_node(state: GraphState) -> GraphState:
    """Answer strictly from the validated context, preserving citation tags for traceability."""
    if not state["documents"]:
        return {**state, "generation": "I don't have enough relevant information to answer this question."}

    context = build_context(
        documents=[d["document"] for d in state["documents"]],
        metadatas=[d["metadata"] for d in state["documents"]],
    )
    answer = _llm().invoke(GENERATION_PROMPT.format_messages(context=context, query=state["query"])).content
    return {**state, "generation": answer}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("transform_query", transform_query_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents", decide_to_generate, {"generate": "generate", "transform_query": "transform_query"}
    )
    graph.add_edge("transform_query", "web_search")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Question to answer via the Corrective RAG graph")
    args = parser.parse_args()

    app = build_graph()
    final_state = app.invoke(
        {"query": args.query, "documents": [], "relevance_score": 0.0, "web_search_flag": False, "generation": ""}
    )

    print("\n=== ANSWER ===")
    print(final_state["generation"])


if __name__ == "__main__":
    main()
