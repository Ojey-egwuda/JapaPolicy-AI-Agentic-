"""
JapaPolicy AI — HyDE Retriever (Hypothetical Document Embeddings)

Replaces the retriever_agent in src/workers.py.

HOW IT WORKS
────────────
1. The LLM generates a short *hypothetical* regulatory passage that would
   answer the user's query — e.g. "To be eligible for ILR on the Skilled
   Worker route, the applicant must have completed 5 continuous years…"
2. That hypothetical passage (not the raw user question) is used as the
   ChromaDB search vector.
3. Because the embedding space for answer-shaped text sits much closer to
   the embedding space for regulatory source passages than question-shaped
   text does, retrieval precision improves measurably.

INTEGRATION
───────────
In src/workers.py, replace the existing `retriever_agent` function with
the one defined here, then add the HyDE helper above it.
Keep all other agents (router, analyst, response) unchanged.
"""

import os
from datetime import datetime
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import AgentState
from .tools import (
    search_immigration_docs,
    search_govuk_updates,
    calculate_visa_dates,
    check_basic_eligibility,
)


# ── LLM helper (reuse the same factory as workers.py) ────────────────────────

def get_llm(temperature: float = 0.0):
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        temperature=temperature,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ── HyDE prompt ──────────────────────────────────────────────────────────────

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a UK immigration policy expert writing official guidance.

Write a SHORT (3-5 sentence) regulatory passage that directly answers the
question below — as if it appeared in an official Home Office guidance document.

Rules:
- Write as a factual statement, NOT a question
- Include specific figures, thresholds, or timeframes where relevant
- Use formal regulatory language ("an applicant must…", "leave is granted…")
- Do NOT say "I don't know" — always produce a plausible passage
- Do NOT add disclaimers or caveats about seeking legal advice

The passage will be used ONLY as a search vector to find real documents.
It will never be shown to the user."""),
    ("human", "{query}"),
])


def generate_hypothetical_passage(query: str) -> str:
    """
    Generate a hypothetical regulatory answer to use as the HyDE search vector.
    Falls back to the raw query on any error.
    """
    try:
        llm   = get_llm(temperature=0.2)   # Slight temperature for lexical variety
        chain = HYDE_PROMPT | llm | StrOutputParser()
        passage = chain.invoke({"query": query})
        print(f"   │  🧪 HyDE passage ({len(passage)} chars): {passage[:120]}…")
        return passage
    except Exception as e:
        print(f"   │  ⚠️  HyDE generation failed ({e}), falling back to raw query")
        return query


# ── Retriever agent (HyDE-enhanced) ──────────────────────────────────────────

def retriever_agent(state: AgentState) -> Dict[str, Any]:
    """
    AGENT 2: HyDE-enhanced Retriever.

    Replaces the original retriever_agent in src/workers.py.
    All downstream agents (analyst, response) remain unchanged.
    """
    print("\n🔍 [Retriever Agent — HyDE] Gathering information…")

    query         = state["query"]
    query_lower   = query.lower()
    visa_category = state.get("visa_category", "skilled_worker")
    decomposed    = state.get("decomposed_queries", [query])

    all_docs      = []
    all_web       = []
    tool_results  = []

    # ── Step 1: Generate HyDE vector ────────────────────────────────────────
    print("   ├─ 🧪 Generating HyDE search vector…")
    hyde_vector = generate_hypothetical_passage(query)

    # ── Step 2: Vector search — HyDE vector + each sub-query ────────────────
    # Search once with the HyDE vector (highest signal), then with sub-queries
    search_inputs = [hyde_vector] + [
        f"{visa_category} {sq}" if visa_category not in ("unknown", "other", "")
        else sq
        for sq in decomposed[:2]          # Limit to 2 sub-queries after HyDE
    ]

    for i, search_text in enumerate(search_inputs):
        label = "HyDE vector" if i == 0 else f"sub-query {i}"
        print(f"   ├─ 📚 Doc search [{label}]: '{search_text[:60]}…'")
        try:
            results = search_immigration_docs.invoke({
                "query":     search_text,
                "n_results": 5,
            })
            if results and "No relevant documents" not in results:
                all_docs.append({
                    "sub_query": search_text,
                    "results":   results,
                    "search_type": label,
                })
        except Exception as e:
            print(f"   │  ⚠️  Doc search error: {e}")

    # ── Step 3: Web search for recent gov.uk updates ─────────────────────────
    print("   ├─ 🌐 Searching gov.uk for recent updates…")
    web_keywords = []
    if "section 3c" in query_lower:
        web_keywords.append("Section 3C leave")
    if "extension" in query_lower or "pending" in query_lower:
        web_keywords.append("visa extension pending")
    if "work" in query_lower:
        web_keywords.append("right to work")

    web_query = (
        f"UK immigration {visa_category} "
        f"{' '.join(web_keywords) if web_keywords else query[:50]}"
    )
    try:
        web_results = search_govuk_updates.invoke({"query": web_query})
        if web_results and "error" not in web_results.lower() and len(web_results) > 50:
            all_web.append({"query": web_query, "results": web_results})
            print(f"   │  ✅ Web search: {len(web_results)} chars")
        else:
            print("   │  ⚠️  Web search: no useful results")
    except Exception as e:
        print(f"   │  ⚠️  Web search failed: {e}")

    # ── Step 4: Date calculator (if timing / dates mentioned) ────────────────
    date_keywords = [
        "expire", "expiry", "expires", "ilr", "settlement", "how long",
        "days outside", "absence", "continuous residence", "5 years",
        "extension", "pending", "section 3c", "when can i",
    ]
    if any(kw in query_lower for kw in date_keywords):
        print("   ├─ 📅 Running date calculator…")
        try:
            route = (
                visa_category
                if visa_category in ("skilled_worker", "health_care",
                                     "global_talent", "family", "student")
                else "skilled_worker"
            )
            date_result = calculate_visa_dates.invoke({
                "visa_start_date":    "2024-01-01",
                "visa_length_years":  5,
                "days_outside_uk":    0,
                "route":              route,
            })
            tool_results.append({
                "tool":   "date_calculator",
                "result": f"Visa Date Calculations:\n{date_result}",
            })
            print("   │  ✅ Date calculator: ILR timeline generated")
        except Exception as e:
            print(f"   │  ⚠️  Date calc error: {e}")

    # ── Step 5: Eligibility checker (if eligibility question) ────────────────
    eligibility_keywords = [
        "eligible", "eligibility", "qualify", "can i apply",
        "requirements", "minimum salary", "do i need", "am i allowed",
    ]
    if any(kw in query_lower for kw in eligibility_keywords):
        print("   ├─ ✅ Running eligibility pre-check…")
        try:
            visa_for_check = (
                visa_category
                if visa_category in ("skilled_worker", "health_care",
                                     "student", "graduate")
                else "skilled_worker"
            )
            elig_result = check_basic_eligibility.invoke({
                "visa_type":    visa_for_check,
                "salary":       None,
                "has_job_offer": None,
                "english_level": None,
                "has_sponsor":  None,
            })
            tool_results.append({
                "tool":   "eligibility_checker",
                "result": f"Eligibility Pre-Check for {visa_for_check}:\n{elig_result}",
            })
            print("   │  ✅ Eligibility checker: requirements retrieved")
        except Exception as e:
            print(f"   │  ⚠️  Eligibility check error: {e}")

    print(
        f"   └─ Retrieved: {len(all_docs)} doc sets "
        f"| {len(all_web)} web results "
        f"| {len(tool_results)} tool outputs"
    )

    return {
        "retrieved_docs": all_docs,
        "web_results":    all_web,
        "tool_results":   tool_results,
    }