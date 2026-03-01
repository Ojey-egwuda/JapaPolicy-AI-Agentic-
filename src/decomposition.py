"""
JapaPolicy AI — Query Decomposition

A dedicated decomposition step that runs BEFORE the router classifies the
query type. Its only job is to break compound immigration questions into
independent, atomic sub-queries so each one can be retrieved separately.

WHY THIS MATTERS
────────────────
The original router generates decomposed_queries heuristically — it works
for simple questions but misses compound eligibility scenarios like:

  "Can I work full-time on my Graduate visa and does my employer need to
   sponsor me if I want to stay longer?"

That is actually THREE separate retrieval units:
  1. Graduate visa work rights — full-time employment permitted
  2. Graduate visa employer sponsorship requirement
  3. Switching from Graduate to Skilled Worker (to 'stay longer')

Fetching all three independently, then synthesising, produces a far more
accurate answer than fetching with the combined query string.

INTEGRATION
───────────
1. Add `decomposition_agent` as a node in src/graph.py BEFORE "router"
2. Wire:  START → decomposition → router → (existing flow)
3. The router reads state["decomposed_queries"] — no other change needed
   in workers.py, graph.py, or state.py (state already has the field).
"""

import os
import re
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .state import AgentState


# LLM factory
def get_llm(temperature: float = 0.0):
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        temperature=temperature,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )


# Decomposition prompt
DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at breaking down complex UK immigration questions
into simple, atomic retrieval queries.

Your ONLY job is to identify EVERY distinct piece of information the user needs,
and write ONE search query per piece. Each query must be self-contained and
answerable on its own.

RULES:
- Produce between 1 and 4 sub-queries (never more than 4)
- Each sub-query must be a SHORT noun phrase or question (max 12 words)
- If the original question is already simple, return it as a single sub-query
- Focus on what documents need to be FOUND, not on answering the question
- Include the visa route in each sub-query for precision

EXAMPLES:

Input: "Can I work full-time on a Graduate visa and does my employer need to sponsor me?"
Output: {{
  "sub_queries": [
    "Graduate visa work rights full-time employment",
    "Graduate visa employer sponsorship requirement",
    "Switching Graduate visa to Skilled Worker route"
  ],
  "is_compound": true
}}

Input: "What is the minimum salary for a Skilled Worker visa?"
Output: {{
  "sub_queries": [
    "Skilled Worker visa minimum salary threshold 2024"
  ],
  "is_compound": false
}}

Input: "My Skilled Worker visa expires in 2 weeks, my extension is pending, can I still work and can I travel to Nigeria?"
Output: {{
  "sub_queries": [
    "Section 3C leave right to work pending application",
    "Section 3C leave travel outside UK restrictions",
    "Skilled Worker visa extension timeline processing"
  ],
  "is_compound": true
}}

Respond with valid JSON only."""),
    ("human", "{query}"),
])


# Fallback: rule-based decomposer
COMPOUND_PATTERNS = [
    r"\band\b",
    r"\balso\b",
    r"\bfurthermore\b",
    r"\bin addition\b",
    r"\bbesides\b",
    r"\bas well as\b",
    r"\bboth\b",
    r"\bwhat about\b",
]

def _rule_based_decompose(query: str) -> List[str]:
    """
    Lightweight fallback decomposer.
    Splits on conjunctions and cleans up the parts.
    """
    parts = re.split(r"\band\b|\balso\b|\bfurthermore\b|\bin addition\b", query, flags=re.IGNORECASE)
    cleaned = [p.strip(" ?,;") for p in parts if len(p.strip()) > 8]
    return cleaned if len(cleaned) > 1 else [query]


def _is_compound(query: str) -> bool:
    """Quick check for compound indicators before calling the LLM."""
    q = query.lower()
    return any(re.search(p, q) for p in COMPOUND_PATTERNS) or len(query.split()) > 18


# Main decomposition agent
def decomposition_agent(state: AgentState) -> Dict[str, Any]:
    """
    Pre-router node that atomises compound immigration queries.

    Writes to state["decomposed_queries"].
    The downstream router and retriever read from this field as before.
    """
    query = state["query"]
    print(f"\n✂️  [Decomposition Agent] Analysing query structure…")

    # Skip LLM call for simple queries
    if not _is_compound(query):
        print(f"   └─ Simple query — no decomposition needed")
        return {"decomposed_queries": [query]}

    print(f"   ├─ Compound query detected — decomposing…")

    try:
        llm    = get_llm(temperature=0.0)
        chain  = DECOMPOSITION_PROMPT | llm | JsonOutputParser()
        result = chain.invoke({"query": query})

        sub_queries = result.get("sub_queries", [])
        is_compound = result.get("is_compound", False)

        # Sanity checks
        if not sub_queries or not isinstance(sub_queries, list):
            raise ValueError("LLM returned no sub_queries")

        # Deduplicate and cap at 4
        seen   = set()
        unique = []
        for sq in sub_queries:
            sq = sq.strip()
            if sq and sq.lower() not in seen:
                seen.add(sq.lower())
                unique.append(sq)
            if len(unique) == 4:
                break

        # Always ensure the original query is represented
        if query not in unique:
            unique.insert(0, query)

        unique = unique[:4]

        print(f"   ├─ Compound: {is_compound}  →  {len(unique)} sub-queries")
        for i, sq in enumerate(unique, 1):
            print(f"   │   {i}. {sq}")
        print(f"   └─ Decomposition complete")

        return {"decomposed_queries": unique}

    except Exception as e:
        print(f"   ├─ ⚠️  LLM decomposition failed ({e}), using rule-based fallback")
        fallback = _rule_based_decompose(query)
        print(f"   └─ Fallback produced {len(fallback)} sub-queries")
        return {"decomposed_queries": fallback}