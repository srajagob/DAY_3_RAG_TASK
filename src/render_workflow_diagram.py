"""
Renders the ticket_resolver_graph.py LangGraph workflow as a standalone JPG,
independent of any Mermaid/graphviz backend or internet access.

Usage:
    python src/render_workflow_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "ticket_resolver_workflow.jpg"

NODE_FACE = "#EAF1FB"
NODE_EDGE = "#2C5F9E"
GATE_FACE = "#FFF3D6"
GATE_EDGE = "#B8860B"
TERMINAL_FACE = "#DDEAD1"
TERMINAL_EDGE = "#4C7A3D"
ARROW_COLOR = "#444444"


def box(ax, xy, w, h, text, face=NODE_FACE, edge=NODE_EDGE, fontsize=9.5):
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, facecolor=face, edgecolor=edge, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, wrap=True, zorder=3)
    return patch


def diamond(ax, xy, w, h, text, fontsize=9):
    x, y = xy
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    patch = Polygon(pts, closed=True, linewidth=1.4, facecolor=GATE_FACE, edgecolor=GATE_EDGE, zorder=2)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3)
    return patch


def terminal(ax, xy, text, w=1.6, h=0.55):
    box(ax, xy, w, h, text, face=TERMINAL_FACE, edge=TERMINAL_EDGE, fontsize=9.5)


def arrow(ax, start, end, label=None, curve=0.0, label_dx=0.0, label_dy=0.0, color=ARROW_COLOR):
    patch = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={curve}",
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, color=color, zorder=1,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2 + label_dx
        my = (start[1] + end[1]) / 2 + label_dy
        ax.text(
            mx, my, label, ha="center", va="center", fontsize=8, color="#333333",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85), zorder=4,
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 17))
    ax.set_xlim(0, 11)
    ax.set_ylim(-4.0, 15)
    ax.axis("off")
    ax.set_title(
        "Agentic Ticketing Resolver \u2014 LangGraph Workflow (TO-BE)",
        fontsize=14, fontweight="bold", pad=14,
    )

    CX = 5.5  # center column
    LX = 1.6  # left column (out-of-scope short-circuit)
    RX = 9.4  # right column (manual RCA / human path)

    W, H = 3.2, 0.65

    # Main spine
    terminal(ax, (CX, 14.2), "START")
    box(ax, (CX, 13.2), W, H, "ingest_ticket")
    box(ax, (CX, 12.1), W, H, "categorize\n(LLM: language / PD domain / issue_type / severity)", fontsize=8.5)
    diamond(ax, (CX, 10.7), 4.2, 1.1, "in PD-coding\nscope?", fontsize=8.5)
    box(ax, (CX, 9.3), W, H, "retrieve_similar\n(RAG: embed + query ticket_kb)", fontsize=8.5)
    box(ax, (CX, 8.2), W, H, "grade_similarity\n(LLM grader, CRAG-style)", fontsize=8.5)
    diamond(ax, (CX, 6.8), 4.6, 1.2, "similar case +\nsyntax issue?", fontsize=8.5)
    box(ax, (CX, 5.4), W, H, "reproduce_issue\n(parse/compile check)", fontsize=8.5)
    box(ax, (CX, 4.3), W, H, "attempt_fix\n(LLM proposes patch)", fontsize=8.5)
    box(ax, (CX, 3.2), W, H, "validate_fix\n(re-run reproduction check)", fontsize=8.5)
    diamond(ax, (CX, 1.8), 4.0, 1.1, "fix result?", fontsize=8.5)
    box(ax, (CX, 0.4), W + 0.4, H + 0.2,
        "root_cause_report\n(root cause, contributing factors,\nevidence, next steps, suggested fix, confidence)",
        fontsize=8)
    diamond(ax, (CX, -0.9), 4.6, 1.1, "confidence \u2265 0.8\n& fix validated?", fontsize=8.5)

    # Side nodes
    box(ax, (LX, 9.3), 2.6, H, "out_of_scope\n(manual triage)", fontsize=8.5)
    box(ax, (RX, 7.2), 2.6, H, "manual_rca\n(LLM root-cause hypotheses)", fontsize=8)
    UT = (CX, -2.2)
    box(ax, UT, 2.8, H, "update_ticket\n(PATCH ticketing API + audit note)", fontsize=8.5)
    terminal(ax, (CX, -3.4), "END")

    # Main spine edges
    arrow(ax, (CX, 13.87), (CX, 13.52))
    arrow(ax, (CX, 12.87), (CX, 12.42))
    arrow(ax, (CX, 11.77), (CX, 11.25))
    arrow(ax, (CX, 10.15), (CX, 9.62))
    arrow(ax, (CX, 8.97), (CX, 8.52))
    arrow(ax, (CX, 7.87), (CX, 7.4))
    arrow(ax, (CX, 6.2), (CX, 5.72))
    arrow(ax, (CX, 5.07), (CX, 4.62))
    arrow(ax, (CX, 3.97), (CX, 3.52))
    arrow(ax, (CX, 2.87), (CX, 2.35))
    arrow(ax, (CX, 1.25), (CX, 0.72))
    arrow(ax, (CX, 0.05), (CX, -0.35))
    arrow(ax, (CX, -1.45), (CX, -1.85))
    arrow(ax, (CX, -2.52), (CX, -3.15))

    # domain_gate branches
    arrow(ax, (CX - 2.1, 10.7), (LX, 9.62), label="out_of_scope", label_dx=-0.9, label_dy=0.3)
    arrow(ax, (LX, 8.97), UT, curve=-0.12, label="status=in_progress", label_dx=-2.3, label_dy=-1.5)

    # similarity_gate branches (enters manual_rca from the top)
    arrow(ax, (CX + 2.3, 6.8), (RX, 7.52), label="no match /\nnon-syntax", label_dx=1.15, label_dy=0.55)
    arrow(ax, (RX, 6.88), (CX + 1.9, 0.55), curve=0.15, label="manual_rca", label_dx=1.75, label_dy=3.3)

    # fix_gate branches (give_up enters manual_rca from below)
    arrow(ax, (CX + 2.0, 1.8), (RX, 6.9), curve=-0.3, label="give_up", label_dx=1.9, label_dy=-1.4)
    arrow(ax, (CX - 2.0, 1.8), (CX - 1.75, 4.3), curve=0.4, label="retry (<2)", label_dx=-1.1, label_dy=1.0)

    # confidence_gate branches
    arrow(ax, (CX - 2.3, -0.9), (CX - 1.4, -2.2), curve=-0.25, label="auto_apply \u2192 resolved", label_dx=-1.7, label_dy=-0.05)
    arrow(ax, (CX + 2.3, -0.9), (CX + 1.4, -2.2), curve=0.25, label="human_review \u2192 in_progress", label_dx=1.9, label_dy=-0.05)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, format="jpg", dpi=200)
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
