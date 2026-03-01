"""
UK Immigration Assistant - Shared State
Defines the AgentState TypedDict used across all agents.
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state across all agents in the immigration assistant."""
    
    
    # Conversation
    messages: Annotated[list, add_messages]
    query: str
    
    # Router outputs
    query_type: Optional[str]  # visa_eligibility, switching, extension, ilr, citizenship, general
    visa_category: Optional[str]  # skilled_worker, student, family, visitor, etc.
    needs_clarification: bool
    clarification_question: Optional[str]
    decomposed_queries: List[str]
    
    # Retriever outputs
    retrieved_docs: List[Dict[str, Any]]  # Vector search results
    web_results: List[Dict[str, Any]]  # Web search results from gov.uk
    tool_results: List[Dict[str, Any]]  # Results from date_calculator, eligibility_checker
    
    # Analyst outputs
    analysis: Optional[str]
    key_requirements: List[str]
    confidence_score: float
    
    # Response outputs
    final_response: Optional[str]
    sources_cited: List[Dict[str, Any]]