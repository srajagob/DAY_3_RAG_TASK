"""
Agentic Ticketing Resolver — LangGraph workflow (TO-BE AI-assisted workflow).

Scope guardrail: only bug tickets about CODE (Tcl / Python / C++) inside the
Physical Design (PD) implementation domain are handled. Everything else is
routed to "out_of_scope" untouched.

Pipeline (see build_graph() for the exact node/edge wiring):

    ingest_ticket
        -> categorize                              (LLM: language, domain, issue_type, severity)
        -> [domain_gate]                            conditional
             out_of_scope  -----------------------------------------> update_ticket -> END
             in_scope -> retrieve_similar            (RAG: embed + query ticket_kb)
                      -> grade_similarity             (LLM grader, like CRAG)
                      -> [similarity_gate]            conditional
                           no_match     -> manual_rca -----------------\
                           has_reference (non-syntax) -> manual_rca ----+--> root_cause_report
                           auto_debuggable (syntax + known fix)         |
                             -> reproduce_issue                         |
                             -> attempt_fix                             |
                             -> validate_fix                            |
                             -> [fix_gate]           conditional        |
                                  fixed        -> root_cause_report <---/
                                  retry        -> attempt_fix (loop, bounded)
                                  give_up      -> manual_rca -> root_cause_report
        -> root_cause_report                        (LLM: root cause + suggested fix, structured)
        -> [confidence_gate]                         conditional
             auto_apply    -> update_ticket (status=resolved, fix applied)   -> END
             human_review  -> update_ticket (status=in_progress, needs review) -> END

Usage:
    python src/ticket_resolver_graph.py --diagram              # print the mermaid diagram only
    python src/ticket_resolver_graph.py --demo syntax           # run the bundled auto-fixable demo ticket
    python src/ticket_resolver_graph.py --demo unrelated        # run the out-of-scope demo ticket
    python src/ticket_resolver_graph.py --ticket-id 3           # pull a real ticket from the Ticketing System API
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

from context import build_context

for _no_proxy_var in ("NO_PROXY", "no_proxy"):
    os.environ[_no_proxy_var] = ",".join(
        filter(None, [os.environ.get(_no_proxy_var, ""), "localhost", "127.0.0.1"])
    )

OLLAMA_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store"
TICKET_KB_COLLECTION = "ticket_kb"
TOP_K_SIMILAR = 3
MAX_FIX_ATTEMPTS = 2
AUTO_APPLY_CONFIDENCE = 0.8
TICKETING_API_BASE = os.environ.get("TICKETING_API_BASE", "http://localhost:8000/api")

SUPPORTED_LANGUAGES = {"python", "tcl", "cpp"}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SimilarCase(TypedDict):
    id: str
    document: str
    metadata: dict
    distance: float
    relevant: bool


class GraphState(TypedDict, total=False):
    ticket_id: int | None
    title: str
    description: str
    language: str
    domain: str
    issue_type: str
    severity: str
    in_scope: bool
    similar_cases: list[SimilarCase]
    best_match: SimilarCase | None
    auto_debug_eligible: bool
    fix_attempts: int
    reproduced: bool
    original_code: str
    candidate_code: str
    fix_validated: bool
    root_cause_summary: str
    contributing_factors: list[str]
    evidence: list[str]
    recommended_next_steps: list[str]
    suggested_fix: str
    confidence: float
    resolution_path: str  # "auto_apply" | "human_review" | "out_of_scope"
    ticket_update_status: str


# ---------------------------------------------------------------------------
# LLM setup + structured schemas
# ---------------------------------------------------------------------------

def _llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)


class Categorization(BaseModel):
    language: Literal["python", "tcl", "cpp", "other"] = Field(description="Primary programming language of the reported issue")
    domain: Literal["physical_design", "other"] = Field(description="'physical_design' if about PD implementation (place & route, timing, floorplanning, STA, flows), else 'other'")
    issue_type: Literal["syntax_error", "logic_error", "environment_issue", "performance", "other"] = Field(
        description="'syntax_error' only if the bug is a clear-cut parse/compile error reproducible from a code snippet"
    )
    severity: Literal["low", "medium", "high", "urgent"] = Field(description="Estimated severity")


CATEGORIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You triage engineering bug tickets submitted to a Physical Design (PD) implementation team's "
            "internal ticketing system. Every ticket that reports a problem with a Tcl/Python/C++ script, tool "
            "flow, or automation used by this team is 'physical_design' domain BY DEFAULT, even if the ticket "
            "text itself doesn't mention PD terminology (place & route, timing, floorplanning, STA) — treat the "
            "submitting team as the context. Only classify domain='other' for tickets that are clearly NOT about "
            "code at all (e.g. hardware/facilities requests, network/VPN access, account/dashboard access). "
            "For issue_type, 'syntax_error' covers parse errors, unmatched braces/quotes, AND invalid/unknown "
            "command or function names — i.e. any error reproducible directly from the code without needing "
            "external system state. Classify the ticket strictly using the given schema.",
        ),
        ("human", "Title: {title}\n\nDescription:\n{description}"),
    ]
)


class RelevanceGrade(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="'yes' if this historical case describes the same or a closely related issue")


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade whether a historical ticket/KB entry is relevant to a new bug report. "
            "Answer 'yes' only if the root cause or symptom clearly matches.",
        ),
        ("human", "New ticket:\n{ticket}\n\nHistorical case:\n{case}"),
    ]
)


class ReproductionResult(BaseModel):
    reproducible: bool = Field(description="True if a concrete, specific error/bug can be pinpointed in the code")
    error_message: str = Field(description="The specific error a real interpreter/compiler would raise, with line/command reference")


REPRODUCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a {language} interpreter/compiler simulator. No local {language} toolchain is available, so "
            "you must statically analyze the code and predict the exact error it would raise (unmatched braces, "
            "invalid/unknown command or function names, type mismatches, undefined variables, etc.). Be specific: "
            "name the offending line/command. If the code truly has no defect, say reproducible=false.",
        ),
        ("human", "Code:\n```\n{code}\n```"),
    ]
)


class FixValidation(BaseModel):
    valid: bool = Field(description="True if the candidate code no longer contains the originally identified error")
    remaining_issues: str = Field(description="Any remaining problems, or 'none'")


VALIDATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a {language} interpreter/compiler simulator (no local toolchain available). Check whether "
            "the candidate code below still exhibits the original error. Be strict.",
        ),
        ("human", "Original error:\n{error}\n\nCandidate code:\n```\n{code}\n```"),
    ]
)


class FixProposal(BaseModel):
    fixed_code: str = Field(description="The corrected code snippet, complete and ready to test")
    explanation: str = Field(description="One or two sentences on what was wrong and what changed")


FIX_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior {language} engineer fixing a syntax error. "
            "Return ONLY the corrected code and a short explanation, using the given schema.",
        ),
        (
            "human",
            "Broken code:\n```\n{code}\n```\n\nError observed:\n{error}\n\n"
            "Similar historical fix for reference (may be empty):\n{reference}",
        ),
    ]
)


class RootCauseReport(BaseModel):
    root_cause_summary: str
    contributing_factors: list[str]
    recommended_next_steps: list[str]
    confidence: float = Field(description="0-1 confidence that the root cause/fix is correct")

    @field_validator("confidence")
    @classmethod
    def _normalize_confidence(cls, value: float) -> float:
        # some models answer with a 0-100 percentage despite instructions; normalize defensively
        if value > 1:
            value = value / 100
        return max(0.0, min(1.0, value))


REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write the final engineering-facing root cause report for a bug ticket. "
            "Be concrete and reference the evidence provided. Do not invent facts not in the context. "
            "`confidence` MUST be a decimal between 0 and 1 (e.g. 0.8), never a percentage like 80.",
        ),
        (
            "human",
            "Ticket:\nTitle: {title}\nDescription: {description}\nIssue type: {issue_type}\n\n"
            "Evidence / context:\n{context}\n\n"
            "Auto-debug outcome (if any): {debug_outcome}",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def ingest_ticket_node(state: GraphState) -> GraphState:
    """Normalize the incoming ticket payload before any AI processing."""
    return {**state, "fix_attempts": 0}


def categorize_node(state: GraphState) -> GraphState:
    """LLM categorization: language, PD domain relevance, issue type, severity."""
    result = _llm().with_structured_output(Categorization).invoke(
        CATEGORIZE_PROMPT.format_messages(title=state["title"], description=state["description"])
    )
    in_scope = result.language in SUPPORTED_LANGUAGES and result.domain == "physical_design"
    print(f"[categorize] language={result.language} domain={result.domain} issue_type={result.issue_type} in_scope={in_scope}")
    return {
        **state,
        "language": result.language,
        "domain": result.domain,
        "issue_type": result.issue_type,
        "severity": result.severity,
        "in_scope": in_scope,
    }


def domain_gate(state: GraphState) -> Literal["in_scope", "out_of_scope"]:
    """Conditional edge: only coding tickets (Tcl/Python/C++) in the PD domain proceed."""
    return "in_scope" if state["in_scope"] else "out_of_scope"


def out_of_scope_node(state: GraphState) -> GraphState:
    """Tag tickets outside the coding/PD domain for manual triage instead of AI processing."""
    return {**state, "resolution_path": "out_of_scope"}


def retrieve_similar_node(state: GraphState) -> GraphState:
    """RAG step: embed the ticket and search the ticket/RCA/knowledge-base vector store."""
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = client.get_or_create_collection(name=TICKET_KB_COLLECTION, metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        print("[retrieve] ticket_kb is empty — no historical cases to compare against")
        return {**state, "similar_cases": []}

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_text = f"{state['title']}\n{state['description']}"
    query_embedding = model.encode([query_text], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K_SIMILAR)

    cases: list[SimilarCase] = [
        {"id": cid, "document": doc, "metadata": meta, "distance": dist, "relevant": False}
        for cid, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
    print(f"[retrieve] found {len(cases)} candidate historical case(s)")
    return {**state, "similar_cases": cases}


def grade_similarity_node(state: GraphState) -> GraphState:
    """LLM grader (CRAG-style): keep only cases that genuinely match this ticket's root cause/symptom."""
    grader = _llm().with_structured_output(RelevanceGrade)
    ticket_text = f"{state['title']}\n{state['description']}"

    graded: list[SimilarCase] = []
    for case in state.get("similar_cases", []):
        grade = grader.invoke(GRADE_PROMPT.format_messages(ticket=ticket_text, case=case["document"]))
        graded.append({**case, "relevant": grade.binary_score == "yes"})

    relevant = [c for c in graded if c["relevant"]]
    best_match = min(relevant, key=lambda c: c["distance"]) if relevant else None
    auto_debug_eligible = bool(best_match) and state["issue_type"] == "syntax_error"

    print(f"[grade] {len(relevant)}/{len(graded)} relevant; auto_debug_eligible={auto_debug_eligible}")
    return {
        **state,
        "similar_cases": graded,
        "best_match": best_match,
        "auto_debug_eligible": auto_debug_eligible,
    }


def similarity_gate(state: GraphState) -> Literal["auto_debug", "manual_rca"]:
    """Conditional edge: syntax issues with a matching historical fix go through auto-debugging."""
    return "auto_debug" if state["auto_debug_eligible"] else "manual_rca"


def _extract_code_block(text: str) -> str:
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def reproduce_issue_node(state: GraphState) -> GraphState:
    """Reproduce the reported bug: real parse check for Python, LLM-simulated static analysis otherwise."""
    code = _extract_code_block(state["description"])

    if state["language"] == "python":
        try:
            ast.parse(code)
            reproduced, error_message = False, "code parsed without error (could not reproduce)"
        except SyntaxError as exc:
            reproduced, error_message = True, f"{exc.msg} at line {exc.lineno}"
    else:
        # No local Tcl/C++ toolchain on this machine: fall back to an LLM acting as a static analyzer.
        result = _llm().with_structured_output(ReproductionResult).invoke(
            REPRODUCE_PROMPT.format_messages(language=state["language"], code=code)
        )
        reproduced, error_message = result.reproducible, result.error_message

    print(f"[reproduce] language={state['language']} reproduced={reproduced} -> {error_message}")
    return {**state, "reproduced": reproduced, "original_code": code, "candidate_code": code, "root_cause_summary": error_message}


def attempt_fix_node(state: GraphState) -> GraphState:
    """LLM proposes a corrected code snippet, guided by the matched historical fix.

    Always fixes from original_code (not a possibly-truncated prior candidate) so retries
    don't compound on a broken attempt.
    """
    reference = state["best_match"]["document"] if state.get("best_match") else ""
    proposal = _llm().with_structured_output(FixProposal).invoke(
        FIX_PROMPT.format_messages(
            language=state["language"],
            code=state["original_code"],
            error=state["root_cause_summary"],
            reference=reference,
        )
    )
    attempts = state.get("fix_attempts", 0) + 1
    print(f"[attempt_fix] attempt #{attempts}: {proposal.explanation}")
    return {**state, "candidate_code": proposal.fixed_code, "suggested_fix": proposal.fixed_code, "fix_attempts": attempts}


def validate_fix_node(state: GraphState) -> GraphState:
    """Re-run the same reproduction check against the patched code to confirm the fix."""
    original_len = len(state.get("original_code", "").strip())
    candidate_len = len(state["candidate_code"].strip())
    if original_len > 40 and candidate_len < 0.4 * original_len:
        # small local models occasionally truncate long snippets instead of returning the full fix;
        # never trust a candidate that dropped most of the original code, regardless of language.
        print(f"[validate] rejected: candidate_code looks truncated ({candidate_len}/{original_len} chars)")
        return {**state, "fix_validated": False}

    if state["language"] == "python":
        try:
            ast.parse(state["candidate_code"])
            validated = True
        except SyntaxError:
            validated = False
    else:
        result = _llm().with_structured_output(FixValidation).invoke(
            VALIDATE_PROMPT.format_messages(
                language=state["language"], error=state["root_cause_summary"], code=state["candidate_code"]
            )
        )
        validated = result.valid

    print(f"[validate] fix_validated={validated}")
    return {**state, "fix_validated": validated}


def fix_gate(state: GraphState) -> Literal["fixed", "retry", "give_up"]:
    """Conditional edge: accept the fix, retry within budget, or fall back to manual RCA."""
    if state["fix_validated"]:
        return "fixed"
    if state["fix_attempts"] < MAX_FIX_ATTEMPTS:
        return "retry"
    return "give_up"


def manual_rca_node(state: GraphState) -> GraphState:
    """No safe auto-fix available: prepare hints for the human engineer instead of applying code changes."""
    return {**state, "resolution_path": "human_review"}


def root_cause_report_node(state: GraphState) -> GraphState:
    """Generate the final, clearly-labelled root cause + suggested fix report for the ticket."""
    context_blocks = [c["document"] for c in state.get("similar_cases", []) if c["relevant"]]
    context = build_context(
        documents=context_blocks,
        metadatas=[c["metadata"] for c in state.get("similar_cases", []) if c["relevant"]],
    ) or "No relevant historical cases were found."

    debug_outcome = "not attempted"
    if "fix_validated" in state:
        debug_outcome = (
            f"auto-debug {'succeeded' if state['fix_validated'] else 'exhausted retries'} "
            f"after {state.get('fix_attempts', 0)} attempt(s)"
        )

    report = _llm().with_structured_output(RootCauseReport).invoke(
        REPORT_PROMPT.format_messages(
            title=state["title"],
            description=state["description"],
            issue_type=state["issue_type"],
            context=context,
            debug_outcome=debug_outcome,
        )
    )

    evidence = [f"Historical case: {c['metadata'].get('source', c['id'])}" for c in state.get("similar_cases", []) if c["relevant"]]
    if state.get("fix_validated"):
        evidence.append("Auto-debug: fix re-validated by re-running the reproduction check")

    print(f"[report] confidence={report.confidence:.2f}")
    return {
        **state,
        "root_cause_summary": report.root_cause_summary,
        "contributing_factors": report.contributing_factors,
        "recommended_next_steps": report.recommended_next_steps,
        "evidence": evidence,
        "confidence": report.confidence,
        "suggested_fix": state.get("suggested_fix", ""),
    }


def _confidence_decision(state: GraphState) -> Literal["auto_apply", "human_review"]:
    if state.get("fix_validated") and state.get("confidence", 0) >= AUTO_APPLY_CONFIDENCE:
        return "auto_apply"
    return "human_review"


def confidence_gate(state: GraphState) -> Literal["auto_apply", "human_review"]:
    """Conditional edge: only auto-apply when the fix was validated AND confidence clears the bar."""
    return _confidence_decision(state)


def update_ticket_node(state: GraphState) -> GraphState:
    """Write the AI findings back onto the ticket via the Ticketing System REST API."""
    # conditional-edge routing decisions (confidence_gate) aren't persisted to state by LangGraph,
    # so out_of_scope is read back from state while auto_apply/human_review is recomputed here.
    path = "out_of_scope" if state.get("resolution_path") == "out_of_scope" else _confidence_decision(state)

    if path == "out_of_scope":
        note = "[AI Triage] Ticket is outside the coding/Physical-Design scope of the automated resolver. Routed for manual triage."
        payload = {"status": "in_progress"}
    else:
        fix_block = f"\n\nSuggested fix:\n```\n{state['suggested_fix']}\n```" if state.get("suggested_fix") else ""
        note = (
            "[AI Resolution Summary]\n"
            f"Root cause: {state.get('root_cause_summary', 'n/a')}\n"
            f"Contributing factors: {', '.join(state.get('contributing_factors', [])) or 'n/a'}\n"
            f"Evidence: {', '.join(state.get('evidence', [])) or 'n/a'}\n"
            f"Recommended next steps: {', '.join(state.get('recommended_next_steps', [])) or 'n/a'}\n"
            f"Confidence: {state.get('confidence', 0):.2f}"
            f"{fix_block}"
        )
        payload = {"status": "resolved" if path == "auto_apply" else "in_progress"}

    if state.get("ticket_id") is not None:
        try:
            with httpx.Client(base_url=TICKETING_API_BASE, timeout=5.0) as client:
                current = client.get(f"/tickets/{state['ticket_id']}").json()
                payload["description"] = f"{current['description']}\n\n{note}"
                client.patch(f"/tickets/{state['ticket_id']}", json=payload)
            ticket_update_status = "updated"
        except httpx.HTTPError as exc:
            ticket_update_status = f"failed: {exc}"
    else:
        ticket_update_status = "skipped (no ticket_id; dry run)"

    print(f"[update_ticket] path={path} status={ticket_update_status}\n{note}")
    return {**state, "ticket_update_status": ticket_update_status}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(GraphState)
    graph.add_node("ingest_ticket", ingest_ticket_node)
    graph.add_node("categorize", categorize_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("retrieve_similar", retrieve_similar_node)
    graph.add_node("grade_similarity", grade_similarity_node)
    graph.add_node("reproduce_issue", reproduce_issue_node)
    graph.add_node("attempt_fix", attempt_fix_node)
    graph.add_node("validate_fix", validate_fix_node)
    graph.add_node("manual_rca", manual_rca_node)
    graph.add_node("root_cause_report", root_cause_report_node)
    graph.add_node("update_ticket", update_ticket_node)

    graph.add_edge(START, "ingest_ticket")
    graph.add_edge("ingest_ticket", "categorize")
    graph.add_conditional_edges(
        "categorize", domain_gate, {"in_scope": "retrieve_similar", "out_of_scope": "out_of_scope"}
    )
    graph.add_edge("out_of_scope", "update_ticket")

    graph.add_edge("retrieve_similar", "grade_similarity")
    graph.add_conditional_edges(
        "grade_similarity", similarity_gate, {"auto_debug": "reproduce_issue", "manual_rca": "manual_rca"}
    )

    graph.add_edge("reproduce_issue", "attempt_fix")
    graph.add_edge("attempt_fix", "validate_fix")
    graph.add_conditional_edges(
        "validate_fix",
        fix_gate,
        {"fixed": "root_cause_report", "retry": "attempt_fix", "give_up": "manual_rca"},
    )

    graph.add_edge("manual_rca", "root_cause_report")
    graph.add_conditional_edges(
        "root_cause_report", confidence_gate, {"auto_apply": "update_ticket", "human_review": "update_ticket"}
    )
    graph.add_edge("update_ticket", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Demo tickets + CLI
# ---------------------------------------------------------------------------

DEMO_TICKETS: dict[str, GraphState] = {
    "syntax": {
        "ticket_id": None,
        "title": "Tcl script fails with 'wrong # args' during floorplan init",
        "description": (
            "Our place-and-route init_floorplan.tcl step fails during physical design implementation.\n"
            "```\n"
            "proc init_floorplan {die_w die_h util\n"
            "    puts \"Initializing floorplan $die_w x $die_h at $util utilization\"\n"
            "}\n"
            "```\n"
            "Error: wrong # args: should be \"init_floorplan die_w die_h util\""
        ),
    },
    "unrelated": {
        "ticket_id": None,
        "title": "Conference room projector not turning on",
        "description": "The projector in room Cedar does not power on when the HDMI cable is connected.",
    },
}


def _initial_state(seed: GraphState) -> GraphState:
    return {
        "ticket_id": seed.get("ticket_id"),
        "title": seed["title"],
        "description": seed["description"],
        "similar_cases": [],
        "best_match": None,
        "fix_attempts": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagram", action="store_true", help="Print the compiled graph as Mermaid and exit")
    parser.add_argument("--demo", choices=sorted(DEMO_TICKETS), help="Run a bundled demo ticket end-to-end")
    parser.add_argument("--ticket-id", type=int, help="Fetch and resolve a real ticket from the Ticketing System API")
    args = parser.parse_args()

    app = build_graph()

    if args.diagram:
        print(app.get_graph().draw_mermaid())
        return

    if args.ticket_id is not None:
        with httpx.Client(base_url=TICKETING_API_BASE, timeout=5.0) as client:
            ticket = client.get(f"/tickets/{args.ticket_id}").json()
        seed: GraphState = {"ticket_id": ticket["id"], "title": ticket["title"], "description": ticket["description"]}
    else:
        seed = DEMO_TICKETS[args.demo or "syntax"]

    final_state = app.invoke(_initial_state(seed))

    print("\n=== FINAL STATE ===")
    for key in ("resolution_path", "root_cause_summary", "suggested_fix", "confidence", "ticket_update_status"):
        if key in final_state:
            print(f"{key}: {final_state[key]}")


if __name__ == "__main__":
    main()
