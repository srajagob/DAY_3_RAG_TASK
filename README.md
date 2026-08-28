# DAY_3_RAG_TASK

Two things live in this repo:

1. **Baseline + Corrective RAG pipeline** (Day 3 exercise) — parse → chunk → embed → vector store → retrieve/grade/generate.
2. **Agentic Ticketing Resolver** (Day 4 capstone) — a LangGraph agent that triages, RAG-matches, auto-debugs, and writes fixes back to a real ticketing system.

Both share the same Python environment, embedding model, and Chroma vector store directory (`data/vector_store`), just different collections (`rag_chunks` vs `ticket_kb`).

---

## 1. Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Windows note: if your clone path is deeply nested (e.g. under OneDrive), `torch`'s
> license file paths can exceed `MAX_PATH`. Create the venv somewhere shorter
> (e.g. `C:\venvs\<name>`) if you hit `WinError 206`.

Copy `.env.example` to `.env` and adjust if Ollama isn't on the default local address:

```
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

Pull the model once: `ollama pull llama3.1:8b`, and make sure `ollama serve` is running.

---

## 2. Baseline + Corrective RAG pipeline

Source docs → clean Markdown → token-sized chunks → embeddings → Chroma vector store → retrieval-augmented answers.

| Stage | Script | What it does |
|---|---|---|
| 1. Parse | [src/parse_documents.py](src/parse_documents.py) | Converts heterogeneous files (`data/raw/*`) to clean Markdown via MarkItDown. PDFs are paged with `<!-- page:N -->` markers for citation. |
| 2. Chunk | [src/chunk_documents.py](src/chunk_documents.py) | Splits `data/parsed/*.md` into overlapping, token-sized chunks (450 tokens, 70 overlap, `tiktoken` cl100k_base), breaking on paragraph/line/sentence boundaries. |
| 3. Embed | [src/embed_documents.py](src/embed_documents.py) | Embeds `data/chunks/*.jsonl` with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, swappable via `--model`). |
| 4. Vector store | [src/build_vector_store.py](src/build_vector_store.py) | Loads chunks + embeddings into a persistent Chroma collection `rag_chunks` (`data/vector_store`, cosine similarity). |
| Shared | [src/context.py](src/context.py) | Formats retrieved chunks into citation-tagged context blocks (`[Source: file.pdf \| Page N]`) for the generation prompt. |
| Corrective RAG graph | [src/crag_graph.py](src/crag_graph.py) | LangGraph pipeline: `retrieve → grade_documents → (generate \| transform_query → web_search → generate)`. An LLM grades each chunk's relevance; if too few pass, the query is rewritten and a DuckDuckGo web search fills the gap before generation. |

Run the pipeline end to end:

```powershell
python src/parse_documents.py
python src/chunk_documents.py
python src/embed_documents.py
python src/build_vector_store.py
python src/crag_graph.py "your question here"
```

---

## 3. Agentic Ticketing Resolver

An end-to-end LangGraph agent that reads a bug ticket, decides if it's in scope, checks whether a similar issue has been solved before, and — for reproducible syntax errors — automatically reproduces, fixes, and validates the bug before writing the result back onto the real ticket.

Code: [src/ticket_resolver_graph.py](src/ticket_resolver_graph.py). Diagram: [ticket_resolver_workflow.jpg](ticket_resolver_workflow.jpg) (regenerate with [src/render_workflow_diagram.py](src/render_workflow_diagram.py)).

### Scope guardrail

Only tickets about **code** (Python / Tcl / C++) inside a **Physical Design (PD) implementation** team's tooling are processed end-to-end. Everything else (hardware, facilities, VPN/account access, etc.) is routed to `out_of_scope` and left for manual triage — untouched otherwise.

### Workflow

```
ingest_ticket
    -> categorize                              (LLM: language, domain, issue_type, severity)
    -> [domain_gate]
         out_of_scope  -----------------------------------------> update_ticket -> END
         in_scope -> retrieve_similar            (RAG: embed + query ticket_kb)
                  -> grade_similarity             (LLM grader, CRAG-style)
                  -> [similarity_gate]
                       no match / non-syntax -> manual_rca ----------\
                       syntax + known fix                             +--> root_cause_report
                         -> reproduce_issue                           |
                         -> attempt_fix                                |
                         -> validate_fix                               |
                         -> [fix_gate]                                  |
                              fixed   -> root_cause_report <------------/
                              retry   -> attempt_fix (loop, bounded)
                              give_up -> manual_rca
    -> root_cause_report                        (LLM: root cause + suggested fix, structured)
    -> [confidence_gate]
         auto_apply    -> update_ticket (status=resolved, fix written back)   -> END
         human_review  -> update_ticket (status=in_progress, flagged for review) -> END
```

### Nodes

| Node | LLM call? | Purpose |
|---|---|---|
| `ingest_ticket` | no | Initializes graph state |
| `categorize` | yes | Classifies `language` / `domain` / `issue_type` / `severity` |
| `out_of_scope` | no | Flags tickets outside the coding/PD domain |
| `retrieve_similar` | no (embedding model only) | Embeds the ticket and queries the `ticket_kb` Chroma collection for similar historical cases |
| `grade_similarity` | yes | CRAG-style relevance grading of each retrieved case; picks the best match |
| `reproduce_issue` | yes, except Python | Real `ast.parse()` for Python; LLM-simulated static analysis for Tcl/C++ (no local toolchain) |
| `attempt_fix` | yes | Proposes corrected code, guided by the matched historical fix |
| `validate_fix` | yes, except Python | Re-checks the candidate fix; rejects clearly truncated/garbled candidates (< 40% of original length) before trusting them |
| `manual_rca` | no | Marks the ticket for human-reviewed root-cause analysis (no auto-fix attempted) |
| `root_cause_report` | yes | Produces the final structured report: root cause, contributing factors, evidence, next steps, suggested fix, confidence |
| `update_ticket` | no | PATCHes the real ticketing system with the AI findings and a status change |

### Conditionals

- **`domain_gate`** — in-scope (Python/Tcl/C++ + PD domain) vs `out_of_scope`.
- **`similarity_gate`** — auto-debug only when a graded-relevant historical case exists **and** `issue_type == syntax_error`; otherwise `manual_rca`.
- **`fix_gate`** — `fixed` → report; `retry` (bounded, `MAX_FIX_ATTEMPTS=2`) → loop to `attempt_fix`; `give_up` → `manual_rca`.
- **`confidence_gate`** — `auto_apply` (ticket marked `resolved`) only if the fix was validated **and** confidence ≥ `AUTO_APPLY_CONFIDENCE=0.8`; otherwise `human_review` (ticket marked `in_progress` with the AI note attached).

### Integration with the Ticketing System

Talks to the [G2_Ticketing_System](../ai_class/G2_Ticketing_System) REST API (`GET/PATCH /api/tickets/{id}`), default `http://localhost:8000/api` (override with the `TICKETING_API_BASE` env var). No ticket ID → dry run (prints the result, doesn't write anything).

### Usage

```powershell
python src/seed_ticket_kb.py                        # seed the ticket_kb collection with sample RCA cases
python src/ticket_resolver_graph.py --diagram        # print the compiled graph as Mermaid
python src/ticket_resolver_graph.py --demo syntax     # bundled auto-fixable Tcl demo ticket (dry run)
python src/ticket_resolver_graph.py --demo unrelated  # bundled out-of-scope demo ticket (dry run)
python src/ticket_resolver_graph.py --ticket-id 5     # resolve a real ticket and write the result back
python src/render_workflow_diagram.py                # regenerate ticket_resolver_workflow.jpg
```

### Known limitations (from live trial runs)

- No local Tcl or C++ toolchain on this machine, so `reproduce_issue`/`validate_fix` fall back to LLM-simulated static analysis for those languages instead of real compilation.
- `grade_similarity` (the CRAG-style relevance grader) is noticeably inconsistent with `llama3.1:8b` — the same ticket vs. the same KB entry can grade relevant on one run and not on another. A larger/more consistent model is recommended before relying on this for production auto-apply decisions.
- Small local models occasionally return truncated/garbled "fixed code"; `validate_fix` guards against this with a length-ratio check, but it's a symptom worth watching.

---

## 4. Repository layout

```
DAY_3_RAG_TASK/
  data/
    raw/          source documents (pdf, html, ...)
    parsed/       MarkItDown output (.md)
    chunks/       chunked text (.jsonl)
    embeddings/   chunk embeddings (.npy + .meta.jsonl)
    vector_store/ persistent Chroma DB (collections: rag_chunks, ticket_kb)
  src/
    parse_documents.py, chunk_documents.py, embed_documents.py,
    build_vector_store.py, context.py, crag_graph.py   # baseline + corrective RAG
    ticket_resolver_graph.py, seed_ticket_kb.py,
    render_workflow_diagram.py                          # agentic ticketing resolver
  ticket_resolver_workflow.jpg                           # workflow diagram
  Day_4_AI_Assisted_Engineering_Improvement_Playbook_Template.docx
  requirements.txt
  .env.example
```
