"""
UK Immigration Assistant - Agent Workers

Changes from previous version:
  • analyst_agent: doc context trimmed from 3000 → 1500 chars per doc set
  • analyst_agent: web context trimmed from 2000 → 1000 chars
  • Reduces analyst token count from ~8,000 back to ~5,000-6,000
  • Cuts analyst latency from ~23s back to ~13s
"""

import os
import re
from datetime import datetime
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

from .state import AgentState
from .tools import (
    search_immigration_docs,
    search_govuk_updates,
    calculate_visa_dates,
    check_basic_eligibility
)


# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.0):
    """Get configured LLM instance."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
        temperature=temperature,
        api_key=os.getenv("GOOGLE_API_KEY")
    )


# ── Policy awareness ──────────────────────────────────────────────────────────

MAJOR_POLICY_CHANGES = """
CRITICAL POLICY CHANGES (be aware of these):

1. STUDENT DEPENDANTS (January 2024):
   - BEFORE Jan 2024: Most students could bring dependants
   - AFTER Jan 2024: ONLY PhD/research students and government-funded scholarship holders

2. SKILLED WORKER SALARY (4 April 2024):
   - General threshold: £38,700
   - New entrant threshold: £30,960 (70% of going rate)

3. CARE WORKER DEPENDANTS (11 March 2024):
   - NEW care workers can NO LONGER bring dependants

4. SECTION 3C LEAVE:
   - If you apply to extend BEFORE your visa expires, your conditions continue
   - You can keep working on same terms while decision is pending
   - DO NOT travel outside UK/Ireland/Channel Islands/Isle of Man

5. MAINTENANCE FUNDS (2024):
   - Student in London: £1,334/month
   - Student outside London: £1,023/month
"""


def get_freshness_warning(query: str) -> str:
    """Return a warning if the query relates to recently changed policy."""
    query_lower = query.lower()
    warnings = []

    if any(w in query_lower for w in ["student", "dependant", "dependent"]) and "skilled" not in query_lower:
        warnings.append("⚠️ Student dependant rules changed significantly in January 2024.")

    if any(w in query_lower for w in ["salary", "threshold", "minimum"]) and \
       any(w in query_lower for w in ["skilled", "worker", "work"]):
        warnings.append("⚠️ Skilled Worker salary thresholds changed on 4 April 2024. General: £38,700, New entrant: £30,960.")

    if any(w in query_lower for w in ["care worker", "carer", "6135", "6136"]):
        warnings.append("⚠️ Care workers applying on/after 11 March 2024 cannot bring dependants.")

    return " ".join(warnings) if warnings else ""


def is_temporal_query(query: str) -> bool:
    temporal_indicators = [
        "2022", "2023", "2024", "2025", "2026", "2027",
        "last year", "next year", "will change", "used to be",
        "before", "after", "when did", "compare", "changed"
    ]
    return any(i in query.lower() for i in temporal_indicators)


# ── AGENT 1: Router ───────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert UK immigration query router. Analyze the user's question and classify it.

Your job is to:
1. Identify the type of immigration query
2. Detect the visa category (or infer the most likely one)
3. Break down complex questions into 2-3 searchable sub-queries
4. NEVER ask for clarification unless the query is completely meaningless

Query Types:
- visa_eligibility: Requirements, exemptions, who can apply
- visa_switching: Changing from one visa to another
- visa_extension: Extending current visa, Section 3C leave, pending applications
- ilr_application: Indefinite Leave to Remain / settlement
- citizenship: British citizenship / naturalization
- general_info: General immigration questions

Visa Categories:
- skilled_worker: Work visas, salary questions, sponsorship (DEFAULT for work questions)
- health_care: Health and Care Worker visa, NHS workers
- student: Student visa, studying in UK
- graduate: Graduate visa, post-study work
- family: Spouse, partner, parent, child visas
- visitor: Standard Visitor visa
- global_talent: Global Talent visa
- other: Multiple types or unclear

SMART INFERENCE (use instead of asking for clarification):
- "visa expires" + "extension" + "pending" + "work" → visa_extension + skilled_worker
- "Section 3C" → visa_extension
- English/degree/test → skilled_worker
- Salary/job/employer/sponsor → skilled_worker
- Study/university/course → student
- Spouse/partner/marriage → family
- ILR/settlement/permanent → ilr_application

For decomposed_queries, create 2-3 SPECIFIC search queries that will find relevant documents.

Respond with valid JSON:
{{
    "query_type": "string",
    "visa_category": "string",
    "needs_clarification": false,
    "clarification_question": "",
    "decomposed_queries": ["specific search query 1", "specific search query 2"]
}}"""),
    ("human", "{query}")
])


def router_agent(state: AgentState) -> Dict[str, Any]:
    """
    AGENT 1: Router
    Classifies the query and determines routing strategy.
    Preserves decomposed_queries from the decomposition agent.
    """
    print("\n🔀 [Router Agent] Analyzing query...")

    llm   = get_llm(temperature=0.0)
    chain = ROUTER_PROMPT | llm | JsonOutputParser()

    query       = state["query"]
    query_lower = query.lower()

    try:
        result = chain.invoke({"query": query})

        # Only ask clarification for genuinely meaningless queries
        genuinely_meaningless = (
            len(query_lower.strip()) < 5 or
            query_lower.strip() in ["hi", "hello", "help", "?", "visa", "uk", "yes", "no"]
        )

        needs_clarification = genuinely_meaningless

        visa_category = result.get("visa_category", "other")
        query_type    = result.get("query_type", "general_info")

        # Override based on keywords
        if "section 3c" in query_lower or "3c leave" in query_lower:
            query_type = "visa_extension"
        if "pending" in query_lower and ("extension" in query_lower or "application" in query_lower):
            query_type = "visa_extension"

        if visa_category in ["unknown", "", None, "other"]:
            if any(w in query_lower for w in ["skilled", "worker", "work", "job", "salary", "sponsor", "employer"]):
                visa_category = "skilled_worker"
            elif any(w in query_lower for w in ["student", "study", "university", "course"]):
                visa_category = "student"
            elif any(w in query_lower for w in ["spouse", "partner", "husband", "wife", "family", "marriage"]):
                visa_category = "family"
            elif any(w in query_lower for w in ["graduate", "psw", "post-study"]):
                visa_category = "graduate"
            elif any(w in query_lower for w in ["health", "care", "nhs", "nurse"]):
                visa_category = "health_care"
            elif any(w in query_lower for w in ["ilr", "settlement", "indefinite"]):
                visa_category = "skilled_worker"
            else:
                visa_category = "skilled_worker"

        # ── Preserve decomposition agent output ──────────────────────────────
        # If decomposition_agent already produced sub-queries, keep them.
        # Only fall back to router's own decomposition if state is empty.
        existing_decomposition = state.get("decomposed_queries", [])

        if existing_decomposition:
            decomposed = existing_decomposition
        else:
            decomposed = result.get("decomposed_queries", [])
            if not decomposed:
                decomposed = []
                if "section 3c" in query_lower or "pending" in query_lower:
                    decomposed.append("Section 3C leave pending application rights")
                    decomposed.append("Right to work while visa extension pending")
                if "expire" in query_lower or "extension" in query_lower:
                    decomposed.append("Visa extension application before expiry")
                if not decomposed:
                    decomposed = [query]

        print(f"   ├─ Query Type: {query_type}")
        print(f"   ├─ Visa Category: {visa_category}")
        print(f"   ├─ Decomposed: {len(decomposed)} queries")
        print(f"   └─ Needs Clarification: {needs_clarification}")

        return {
            "query_type":             query_type,
            "visa_category":          visa_category,
            "needs_clarification":    needs_clarification,
            "clarification_question": result.get("clarification_question", "") if needs_clarification else "",
            "decomposed_queries":     decomposed,
        }

    except Exception as e:
        print(f"   └─ ❌ Router Error: {e}, using defaults")
        existing = state.get("decomposed_queries", [])
        return {
            "query_type":             "general_info",
            "visa_category":          "skilled_worker",
            "needs_clarification":    False,
            "clarification_question": "",
            "decomposed_queries":     existing if existing else [query],
        }


# ── AGENT 2: Retriever ────────────────────────────────────────────────────────
# NOTE: retriever_agent is defined in hyde_retriever.py and imported by graph.py
# This file intentionally does not define retriever_agent to avoid conflicts.


# ── AGENT 3: Analyst ──────────────────────────────────────────────────────────

ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a UK immigration policy analyst. Analyze the retrieved documents and provide accurate information.

TODAY'S DATE: {current_date}

{policy_awareness}

YOUR TASK:
1. Extract relevant information from documents, web results, and tool outputs
2. Identify key requirements with EXACT figures when available
3. Note any dates, deadlines, or timeframes
4. Provide a CONFIDENT analysis when documents contain the answer
5. Only mark as uncertain if information is genuinely missing

CONFIDENCE SCORING:
- 0.85-0.95: Documents clearly answer the question with specific details
- 0.70-0.84: Good information found, minor gaps
- 0.50-0.69: Partial information, some uncertainty
- Below 0.50: Information not found or contradictory

IMPORTANT:
- If documents contain Section 3C information, confidence should be HIGH
- If web results confirm recent policy, boost confidence
- If tool results provide calculations, include them

Format your response as:

**Key Requirements:**
- [Specific requirement with figure/date if available]
- [Another requirement]

**Analysis:**
[Detailed analysis citing sources where possible]

**Confidence:** [0.0-1.0]
[Brief justification]"""),
    ("human", """Query: {query}
Query Type: {query_type}
Visa Category: {visa_category}

Retrieved Documents:
{retrieved_docs}

Web Search Results:
{web_results}

Tool Results:
{tool_results}

Provide your analysis:""")
])


def analyst_agent(state: AgentState) -> Dict[str, Any]:
    """
    AGENT 3: Analyst
    Analyses all retrieved information.
    Context is trimmed to keep token count under control.
    """
    print("\n🔬 [Analyst Agent] Analyzing information...")

    current_date = datetime.now().strftime("%Y-%m-%d")
    llm   = get_llm(temperature=0.0)
    chain = ANALYST_PROMPT | llm | StrOutputParser()

    # ── Format retrieved docs — 1500 chars per doc set (was 3000) ────────────
    docs_text = ""
    for doc_set in state.get("retrieved_docs", []):
        docs_text += f"\n--- Search: {doc_set['sub_query'][:50]} ---\n"
        docs_text += doc_set["results"][:1500]

    # ── Format web results — 1000 chars (was 2000) ───────────────────────────
    web_text = ""
    for web_set in state.get("web_results", []):
        web_text += "\n--- Web Results ---\n"
        web_text += web_set["results"][:1000]

    # ── Format tool results ───────────────────────────────────────────────────
    tool_text = ""
    for tool_result in state.get("tool_results", []):
        tool_text += f"\n--- {tool_result['tool'].replace('_', ' ').title()} ---\n"
        tool_text += tool_result["result"]

    has_docs  = bool(docs_text.strip())
    has_web   = bool(web_text.strip())
    has_tools = bool(tool_text.strip())

    if not has_docs and not has_web and not has_tools:
        print("   └─ ⚠️ No context available")
        return {
            "analysis":         "No relevant documents were retrieved for this query.",
            "key_requirements": [],
            "confidence_score": 0.0,
        }

    try:
        analysis = chain.invoke({
            "query":            state["query"],
            "query_type":       state.get("query_type", "general"),
            "visa_category":    state.get("visa_category", "unknown"),
            "retrieved_docs":   docs_text if has_docs else "No documents retrieved.",
            "web_results":      web_text  if has_web  else "No web results.",
            "tool_results":     tool_text if has_tools else "No tool results.",
            "current_date":     current_date,
            "policy_awareness": MAJOR_POLICY_CHANGES,
        })

        # Extract confidence score
        confidence = 0.85  # Strong default when docs are present

        if "confidence:" in analysis.lower():
            try:
                conf_match = re.search(r'confidence[:\s]+([0-9.]+)', analysis.lower())
                if conf_match:
                    extracted = float(conf_match.group(1))
                    if 0 <= extracted <= 1:
                        confidence = extracted
            except Exception:
                pass

        query_lower = state["query"].lower()

        # Boost for Section 3C queries with matching docs
        if "section 3c" in query_lower or "3c leave" in query_lower:
            if "section 3c" in docs_text.lower() or "3c" in docs_text.lower():
                confidence = max(confidence, 0.85)

        # Small boost for gov.uk web results
        if has_web and "gov.uk" in web_text.lower():
            confidence = min(confidence + 0.05, 0.95)

        # Reduce for missing info
        critical_missing = [
            "no relevant documents",
            "no information found",
            "cannot provide analysis",
            "no documents retrieved",
        ]
        for phrase in critical_missing:
            if phrase in analysis.lower():
                confidence = min(confidence, 0.3)
                break

        # Extract key requirements
        key_reqs = []
        if "**key requirements" in analysis.lower():
            try:
                parts = analysis.split("**Key Requirements:**")
                if len(parts) > 1:
                    req_section = parts[1]
                    for end_marker in ["**Analysis", "**Confidence", "**Information"]:
                        if end_marker in req_section:
                            req_section = req_section.split(end_marker)[0]
                            break
                    for line in req_section.split("\n"):
                        line = line.strip()
                        if line.startswith(("-", "•", "*")):
                            clean = line.lstrip("-•* ").strip()
                            if clean and len(clean) > 10:
                                key_reqs.append(clean)
            except Exception as e:
                print(f"   ├─ Warning extracting requirements: {e}")

        print(f"   ├─ Confidence Score: {confidence:.2f}")
        print(f"   ├─ Key Requirements: {len(key_reqs)}")
        print(f"   └─ Sources: docs={has_docs}, web={has_web}, tools={has_tools}")

        return {
            "analysis":         analysis,
            "key_requirements": key_reqs[:10],
            "confidence_score": confidence,
        }

    except Exception as e:
        print(f"   └─ ❌ Error: {e}")
        return {
            "analysis":         f"Analysis error: {str(e)}",
            "key_requirements": [],
            "confidence_score": 0.0,
        }


# ── AGENT 4: Responder ────────────────────────────────────────────────────────

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are JapaPolicy AI, an expert UK Immigration Policy Assistant.

Your role is to provide clear, accurate, and helpful answers about UK immigration.

RESPONSE GUIDELINES:

1. **Start with a direct answer** - Answer the user's question immediately
2. **Explain the relevant law/policy** - e.g., Section 3C leave, visa requirements
3. **Include specific details** - Dates, figures, conditions from the analysis
4. **Cite sources** - Use [Source: filename, Page X] when available
5. **Add next steps** - Practical advice for the user

CONFIDENCE-BASED TONE:

HIGH (0.8+): Provide confident, detailed answer with full explanation
MEDIUM (0.5-0.8): Provide answer with note to verify on gov.uk
LOW (<0.5): State limitations clearly, recommend official sources

SECTION 3C LEAVE (if relevant):
- Explain that in-time applications extend existing leave
- Confirm work rights continue on same conditions
- Warn about travel restrictions
- Note that leave continues until decision is made

FORMATTING:
- Use ### headers for sections when the answer is complex
- Use bullet points for lists
- Keep paragraphs short and readable
- Include practical next steps

Always end with a note about verifying on gov.uk for important decisions."""),
    ("human", """User Question: {query}

Query Type: {query_type}
Visa Category: {visa_category}
Confidence: {confidence_score} ({confidence_label})

Analysis:
{analysis}

Key Requirements:
{key_requirements}

Freshness Warning: {freshness_warning}

Provide a helpful, accurate response:""")
])


def response_agent(state: AgentState) -> Dict[str, Any]:
    """AGENT 4: Response Generator"""
    print("\n💬 [Response Agent] Generating response...")

    llm   = get_llm(temperature=0.3)
    chain = RESPONSE_PROMPT | llm | StrOutputParser()

    confidence = state.get("confidence_score", 0.5)

    if confidence >= 0.8:
        confidence_label = "HIGH - Well-supported by documents"
    elif confidence >= 0.6:
        confidence_label = "MEDIUM - Good information with some gaps"
    elif confidence >= 0.4:
        confidence_label = "LOW-MEDIUM - Limited information"
    else:
        confidence_label = "LOW - Very limited information found"

    key_reqs      = state.get("key_requirements", [])
    key_reqs_text = "\n".join([f"- {req}" for req in key_reqs]) if key_reqs else "No specific requirements extracted."
    freshness_warning = get_freshness_warning(state["query"])

    try:
        response = chain.invoke({
            "query":             state["query"],
            "query_type":        state.get("query_type", "general"),
            "visa_category":     state.get("visa_category", "unknown"),
            "analysis":          state.get("analysis", "No analysis available."),
            "key_requirements":  key_reqs_text,
            "confidence_score":  f"{confidence:.2f}",
            "confidence_label":  confidence_label,
            "freshness_warning": freshness_warning or "None",
        })

        if "gov.uk" not in response.lower():
            response += "\n\n---\n📌 **Verify on gov.uk** - Immigration rules change frequently. Always check the official website before making decisions."

        # Extract sources from retrieved docs
        sources_cited = []
        for doc_set in state.get("retrieved_docs", []):
            results = doc_set.get("results", "")
            source_matches = re.findall(r'Source:\s*([^,\n]+),?\s*[Pp]age\s*(\d+)?', results)
            for match in source_matches[:3]:
                sources_cited.append({
                    "reference": f"Source: {match[0]}, Page {match[1] if len(match) > 1 else 'N/A'}"
                })

        print(f"   └─ Response generated ({len(response)} chars)")

        return {
            "final_response": response,
            "sources_cited":  sources_cited[:5],
        }

    except Exception as e:
        print(f"   └─ ❌ Error: {e}")
        return {
            "final_response": (
                "I apologise, but I encountered an error generating the response. "
                "For accurate information on UK immigration, please check gov.uk directly.\n\n"
                f"Error: {str(e)}"
            ),
            "sources_cited": [],
        }


# ── Clarification node ────────────────────────────────────────────────────────

def clarification_node(state: AgentState) -> Dict[str, Any]:
    """Handle cases where clarification is genuinely needed (rare)."""
    print("\n❓ [Clarification] Requesting more information...")
    question = state.get("clarification_question", "Could you please provide more details about your immigration question?")
    return {
        "final_response": f"To provide accurate information, I need a bit more detail:\n\n{question}\n\nYou can also check gov.uk for comprehensive immigration guidance.",
        "sources_cited":  [],
    }


# ── Human review node ─────────────────────────────────────────────────────────

def human_review_node(state: AgentState) -> Dict[str, Any]:
    """Flag response for human review when confidence is very low."""
    print("\n⚠️ [Human Review] Low confidence - adding notice...")
    analysis = state.get("analysis", "")
    analysis += (
        "\n\n⚠️ **Note:** This response has lower confidence. "
        "Please verify with official gov.uk sources."
    )
    return {"analysis": analysis}