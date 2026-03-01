"""
UK Immigration Assistant — LangGraph Workflow  (updated)

Changes from original:
  1. decomposition_agent added as the new entry node (runs before router)
  2. retriever_agent swapped for the HyDE-enhanced version from hyde_retriever.py
  3. LangSmith tracing is configured at module import time via src/tracing.py

Graph structure:

    START
      │
      ▼
  [Decomposition]   ← NEW: atomises compound queries
      │
      ▼
  [Router] ── needs_clarification? ── YES ──► [Clarify] ──► END
      │
      NO
      │
      ▼
  [Retriever-HyDE]  ← UPDATED: HyDE vector search
      │
      ▼
  [Analyst] ── low_confidence? ── YES ──► [Human Review]
      │                                         │
      NO                                        │
      │                                         │
      ▼                                         │
  [Responder] ◄─────────────────────────────────┘
      │
      ▼
     END
"""

from typing import Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .state       import AgentState
from .tracing     import configure_tracing          # ← LangSmith
from .decomposition import decomposition_agent      # ← new decomposition node
from .hyde_retriever import retriever_agent         # ← HyDE-enhanced retriever

from .workers import (
    router_agent,
    analyst_agent,
    response_agent,
    clarification_node,
    human_review_node,
)

# Activate LangSmith tracing (no-op if LANGSMITH_API_KEY is absent)
configure_tracing(project_name="japapolicy-ai")


# Conditional edge functions
def should_clarify(state: AgentState) -> Literal["clarify", "retrieve"]:
    if state.get("needs_clarification", False):
        return "clarify"
    return "retrieve"


def check_confidence(state: AgentState) -> Literal["review", "respond"]:
    if state.get("confidence_score", 0.5) < 0.5:
        return "review"
    return "respond"


# Graph builder
def build_graph(with_memory: bool = True):
    """
    Build and compile the LangGraph workflow.

    New node order:
        START → decomposition → router → (clarify | retriever)
              → analyst → (human_review | responder) → END
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("decomposition", decomposition_agent)   # ← new
    workflow.add_node("router",        router_agent)
    workflow.add_node("retriever",     retriever_agent)       # ← HyDE version
    workflow.add_node("analyst",       analyst_agent)
    workflow.add_node("responder",     response_agent)
    workflow.add_node("clarify",       clarification_node)
    workflow.add_node("human_review",  human_review_node)

    # Entry: decomposition runs first
    workflow.add_edge(START, "decomposition")

    # Decomposition → Router (always)
    workflow.add_edge("decomposition", "router")

    # Router → clarify or retrieve
    workflow.add_conditional_edges(
        "router",
        should_clarify,
        {"clarify": "clarify", "retrieve": "retriever"},
    )

    workflow.add_edge("clarify", END)

    # Retriever → Analyst
    workflow.add_edge("retriever", "analyst")

    # Analyst → human review or responder
    workflow.add_conditional_edges(
        "analyst",
        check_confidence,
        {"review": "human_review", "respond": "responder"},
    )

    workflow.add_edge("human_review", "responder")
    workflow.add_edge("responder", END)

    # Compile
    if with_memory:
        memory = MemorySaver()
        app = workflow.compile(
            checkpointer=memory,
            interrupt_before=["human_review"],
        )
    else:
        app = workflow.compile()

    return app


# Default instance (no memory — used by tests / CLI)
graph = build_graph(with_memory=False)


def visualize_graph():
    """Print ASCII graph structure."""
    print("""
    ┌──────────────────────────────────────────────────────────┐
    │       UK Immigration Assistant — Agent Graph (v2)        │
    └──────────────────────────────────────────────────────────┘

                            START
                              │
                              ▼
                    ┌──────────────────┐
                    │  Decomposition   │  ← atomises compound queries
                    └────────┬─────────┘
                             │
                             ▼
                       ┌──────────┐
                       │  Router  │
                       └────┬─────┘
                            │
             ┌──────────────┴──────────────┐
             │                             │
       needs_clarification?          NO (proceed)
             │                             │
             ▼                             ▼
       ┌──────────┐              ┌─────────────────┐
       │ Clarify  │              │ Retriever (HyDE)│  ← HyDE vector search
       └────┬─────┘              └────────┬────────┘
            │                            │
            ▼                            ▼
           END                    ┌──────────┐
                                  │ Analyst  │
                                  └────┬─────┘
                                       │
                        ┌──────────────┴──────────────┐
                        │                             │
                  low_confidence?              HIGH (proceed)
                        │                             │
                        ▼                             │
                ┌─────────────┐                       │
                │Human Review │                       │
                └──────┬──────┘                       │
                       └──────────────┬───────────────┘
                                      ▼
                               ┌──────────┐
                               │Responder │
                               └────┬─────┘
                                    │
                                    ▼
                                   END
    """)