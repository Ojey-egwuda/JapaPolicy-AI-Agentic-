"""
UK Immigration Assistant - LangGraph Workflow
Orchestrates the multi-agent system.
"""

from typing import Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .workers import (
    router_agent,
    retriever_agent,
    analyst_agent,
    response_agent,
    clarification_node,
    human_review_node
)


def should_clarify(state: AgentState) -> Literal["clarify", "retrieve"]:
    """Conditional edge: check if clarification is needed."""
    if state.get("needs_clarification", False):
        return "clarify"
    return "retrieve"


def check_confidence(state: AgentState) -> Literal["review", "respond"]:
    """Conditional edge: route based on confidence score."""
    confidence = state.get("confidence_score", 0.5)
    if confidence < 0.5:
        return "review"
    return "respond"


def build_graph(with_memory: bool = True):
    """
    Build the LangGraph workflow for the immigration assistant.
    
    Graph Structure:
    
        START
          │
          ▼
       [Router] ─── needs_clarification? ─── YES ──► [Clarify] ──► END
          │                                   
          NO                                  
          │                                   
          ▼                                   
      [Retriever]                             
          │                                   
          ▼                                   
       [Analyst] ─── low_confidence? ─── YES ──► [Human Review]
          │                                            │
          NO                                           │
          │                                            │
          ▼                                            │
      [Response] ◄─────────────────────────────────────┘
          │
          ▼
         END
    """
    
    # Initialize the graph with state schema
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("router", router_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("responder", response_agent)
    workflow.add_node("clarify", clarification_node)
    workflow.add_node("human_review", human_review_node)
    
    # Set entry point
    workflow.add_edge(START, "router")
    
    # Router conditional edges
    workflow.add_conditional_edges(
        "router",
        should_clarify,
        {
            "clarify": "clarify",
            "retrieve": "retriever"
        }
    )
    
    # Clarification ends the flow (waits for user input)
    workflow.add_edge("clarify", END)
    
    # Retriever -> Analyst
    workflow.add_edge("retriever", "analyst")
    
    # Analyst conditional edges based on confidence
    workflow.add_conditional_edges(
        "analyst",
        check_confidence,
        {
            "review": "human_review",
            "respond": "responder"
        }
    )
    
    # Human review -> Response
    workflow.add_edge("human_review", "responder")
    
    # Response -> End
    workflow.add_edge("responder", END)
    
    # Compile the graph
    if with_memory:
        memory = MemorySaver()
        app = workflow.compile(
            checkpointer=memory,
            interrupt_before=["human_review"] 
        )
    else:
        app = workflow.compile()
    
    return app


# Create default graph instance
graph = build_graph(with_memory=False)


def visualize_graph():
    """Generate ASCII visualization of the graph."""
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │           UK Immigration Assistant - Agent Graph         │
    └─────────────────────────────────────────────────────────┘
    
                            START
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
        ┌──────────┐                 ┌──────────┐
        │ Clarify  │                 │Retriever │
        └────┬─────┘                 └────┬─────┘
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
                  ┌─────────────┐                      │
                  │Human Review │                      │
                  └──────┬──────┘                      │
                         │                             │
                         └──────────┬──────────────────┘
                                    │
                                    ▼
                             ┌──────────┐
                             │ Responder│
                             └────┬─────┘
                                  │
                                  ▼
                                 END
    """)