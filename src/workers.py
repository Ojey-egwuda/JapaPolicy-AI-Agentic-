"""
UK Immigration Assistant - Agent Workers
All agents defined in a single module for clarity.

FIXES APPLIED:
- Anti-hallucination guardrails
- Temporal awareness (current date injection)
- Lower default confidence
- Uncertainty detection
- Freshness warnings
- Stricter factual grounding
"""

import os
import json
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


# ============================================================
# LLM INITIALIZATION
# ============================================================

def get_llm(temperature: float = 0.0):
    """Get configured LLM instance."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        temperature=temperature,
        api_key=os.getenv("GOOGLE_API_KEY")
    )


# ============================================================
# POLICY CHANGE AWARENESS
# ============================================================

MAJOR_POLICY_CHANGES = """
CRITICAL POLICY CHANGES (you MUST be aware of these):

1. STUDENT DEPENDANTS (January 2024):
   - BEFORE Jan 2024: Most students could bring dependants
   - AFTER Jan 2024: ONLY PhD/research students and government-funded scholarship holders can bring dependants
   - This was a MAJOR restriction

2. SKILLED WORKER SALARY (4 April 2024):
   - General threshold increased from £26,200 to £38,700
   - New entrant threshold: £30,960 (70% of going rate)
   - Some transitional protections for existing visa holders

3. CARE WORKER DEPENDANTS (11 March 2024):
   - NEW care workers (SOC 6135, 6136) can NO LONGER bring dependants
   - Only those who applied BEFORE 11 March 2024 can still bring dependants

4. MAINTENANCE FUNDS (Current as of 2024):
   - London: £1,334/month (9 months = £12,006)
   - Outside London: £1,023/month (9 months = £9,207)

5. ASYLUM SEEKERS WORK RIGHTS:
   - Cannot work for first 12 months
   - After 12 months: can apply for permission to work
   - Restricted to Shortage Occupation List jobs only
"""


def get_freshness_warning(query: str) -> str:
    """
    Return a warning if the query relates to recently changed policy.
    """
    query_lower = query.lower()
    warnings = []
    
    if any(word in query_lower for word in ["student", "dependant", "dependent"]):
        warnings.append(
            "⚠️ IMPORTANT: Student dependant rules changed significantly in January 2024. "
            "Most students can NO LONGER bring dependants - only PhD/research students and "
            "government-funded scholarship holders qualify."
        )
    
    if any(word in query_lower for word in ["salary", "threshold", "minimum"]) and "skilled" in query_lower:
        warnings.append(
            "⚠️ IMPORTANT: Skilled Worker salary thresholds increased on 4 April 2024. "
            "General threshold is now £38,700. New entrant threshold is £30,960."
        )
    
    if any(word in query_lower for word in ["care worker", "care home", "social care", "6135", "6136"]):
        warnings.append(
            "⚠️ IMPORTANT: Care workers applying on/after 11 March 2024 can NO LONGER bring dependants."
        )
    
    if any(word in query_lower for word in ["maintenance", "funds", "bank statement", "savings"]):
        warnings.append(
            "⚠️ IMPORTANT: Maintenance requirements as of 2024 are £1,334/month (London) "
            "or £1,023/month (outside London). Verify current figures on gov.uk."
        )
    
    if any(word in query_lower for word in ["asylum", "refugee", "protection"]):
        warnings.append(
            "⚠️ IMPORTANT: Asylum seekers cannot work for the first 12 months. "
            "After 12 months, they may apply for permission to work in Shortage Occupation List jobs only."
        )
    
    return "\n".join(warnings) if warnings else ""


def is_temporal_query(query: str) -> bool:
    """Check if query asks about past/future rules."""
    temporal_indicators = [
        "2022", "2023", "2024", "2025", "2026", "2027",
        "last year", "next year", "will change", "used to be",
        "before", "after", "when did", "compare", "history",
        "changed", "new rules", "old rules"
    ]
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in temporal_indicators)


# ============================================================
# AGENT 1: ROUTER AGENT
# ============================================================

class RouterOutput(BaseModel):
    """Structured output for the router agent."""
    query_type: str = Field(
        description="Type of immigration query"
    )
    visa_category: str = Field(
        default="unknown",
        description="Visa category if identifiable"
    )
    needs_clarification: bool = Field(
        default=False,
        description="True ONLY if the question is completely unanswerable"
    )
    clarification_question: str = Field(
        default="",
        description="Question to ask if clarification absolutely needed"
    )
    decomposed_queries: List[str] = Field(
        default_factory=list,
        description="List of sub-queries to research"
    )
    is_temporal: bool = Field(
        default=False,
        description="True if query asks about past/future rules"
    )


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert UK immigration query router. Analyze the user's question and classify it.

Your job is to:
1. Identify the type of immigration query
2. Detect the visa category if mentioned (or infer the most likely one)
3. Break down complex questions into searchable sub-queries
4. Identify if this is a TEMPORAL query (asking about past/future rules)
5. ALMOST NEVER ask for clarification - just make reasonable assumptions

Query Types:
- visa_eligibility: Questions about who can apply, requirements, exemptions
- visa_switching: Questions about changing from one visa to another
- visa_extension: Questions about extending current visa
- ilr_application: Questions about Indefinite Leave to Remain / settlement
- citizenship: Questions about British citizenship / naturalization
- general_info: General immigration questions

Visa Categories:
- skilled_worker: Skilled Worker visa - DEFAULT for work/salary/English questions
- health_care: Health and Care Worker visa
- student: Student visa
- graduate: Graduate visa
- family: Family visas (spouse, partner, parent, child)
- visitor: Standard Visitor visa
- global_talent: Global Talent visa
- asylum: Asylum/refugee related queries
- other: Other or multiple types

TEMPORAL QUERIES:
Set is_temporal=true if the query:
- Asks about specific years (2022, 2024, 2026, etc.)
- Compares rules across different time periods
- Asks "what changed" or "when did X change"
- Asks about future rules

INFERENCE RULES (use instead of asking for clarification):
- English/degree/test questions → "skilled_worker"
- Salary/job/employer questions → "skilled_worker"
- Study/university questions → "student"
- Spouse/partner/marriage questions → "family"
- Asylum/refugee questions → "asylum"
- Care worker/NHS questions → "health_care"

Respond with valid JSON:
{{
    "query_type": "string",
    "visa_category": "string", 
    "needs_clarification": false,
    "clarification_question": "",
    "decomposed_queries": ["search query 1", "search query 2"],
    "is_temporal": false
}}"""),
    ("human", "{query}")
])


def router_agent(state: AgentState) -> Dict[str, Any]:
    """
    AGENT 1: Router
    Classifies the query and determines routing strategy.
    Rarely asks for clarification - infers and searches.
    """
    print("\n🔀 [Router Agent] Analyzing query...")
    
    llm = get_llm(temperature=0.0)
    chain = ROUTER_PROMPT | llm | JsonOutputParser()
    
    query = state["query"]
    query_lower = query.lower()
    
    try:
        result = chain.invoke({"query": query})
        
        # OVERRIDE: Force needs_clarification to False for almost all queries
        genuinely_meaningless = (
            len(query_lower.strip()) < 5 or
            query_lower.strip() in ["hi", "hello", "help", "?", "visa", "uk", "yes", "no"]
        )
        
        needs_clarification = genuinely_meaningless
        
        # Detect temporal query
        is_temporal = result.get("is_temporal", False) or is_temporal_query(query)
        
        # Infer visa category if not detected
        visa_category = result.get("visa_category", "other")
        if visa_category in ["unknown", "", None]:
            visa_category = "other"
        
        # Smart inference based on keywords
        if visa_category == "other":
            if any(word in query_lower for word in ["asylum", "refugee", "protection", "persecution"]):
                visa_category = "asylum"
            elif any(word in query_lower for word in ["english", "language", "test", "ielts", "degree", "salary", "job", "work", "employer", "sponsor", "cos"]):
                visa_category = "skilled_worker"
            elif any(word in query_lower for word in ["study", "university", "course", "student", "cas"]):
                visa_category = "student"
            elif any(word in query_lower for word in ["spouse", "partner", "husband", "wife", "marriage", "family", "child"]):
                visa_category = "family"
            elif any(word in query_lower for word in ["graduate", "post-study", "psw"]):
                visa_category = "graduate"
            elif any(word in query_lower for word in ["health", "care", "nhs", "nurse", "doctor", "6135", "6136"]):
                visa_category = "health_care"
            elif any(word in query_lower for word in ["global talent", "exceptional", "endorsement"]):
                visa_category = "global_talent"
        
        # Ensure we have decomposed queries
        decomposed = result.get("decomposed_queries", [])
        if not decomposed or len(decomposed) == 0:
            decomposed = [query]
        
        # Add temporal context to queries if needed
        if is_temporal:
            # Extract years mentioned
            years_mentioned = [y for y in ["2022", "2023", "2024", "2025", "2026"] if y in query]
            if years_mentioned:
                decomposed = [f"{q} {' '.join(years_mentioned)}" for q in decomposed]
        
        print(f"   ├─ Query Type: {result.get('query_type', 'general_info')}")
        print(f"   ├─ Visa Category: {visa_category}")
        print(f"   ├─ Is Temporal: {is_temporal}")
        print(f"   └─ Needs Clarification: {needs_clarification}")
        
        return {
            "query_type": result.get("query_type", "general_info"),
            "visa_category": visa_category,
            "needs_clarification": needs_clarification,
            "clarification_question": result.get("clarification_question", "") if needs_clarification else "",
            "decomposed_queries": decomposed
        }
    except Exception as e:
        print(f"   └─ ❌ Router Error: {e}, proceeding with defaults")
        return {
            "query_type": "general_info",
            "visa_category": "skilled_worker",
            "needs_clarification": False,
            "clarification_question": "",
            "decomposed_queries": [query]
        }


# ============================================================
# AGENT 2: RETRIEVER AGENT
# ============================================================

def retriever_agent(state: AgentState) -> Dict[str, Any]:
    """
    AGENT 2: Retriever
    Searches documents and web for relevant information.
    """
    print("\n🔍 [Retriever Agent] Gathering information...")
    
    query = state["query"]
    visa_category = state.get("visa_category", "")
    decomposed = state.get("decomposed_queries", [query])
    
    all_docs = []
    all_web_results = []
    
    # Get freshness warning for this query
    freshness_warning = get_freshness_warning(query)
    if freshness_warning:
        print(f"   ├─ {freshness_warning[:80]}...")
    
    # Search for each decomposed query
    for i, sub_query in enumerate(decomposed[:3]):
        print(f"   ├─ Searching: '{sub_query[:50]}...'")
        
        # Tool 1: Vector search
        enhanced_query = f"{visa_category} {sub_query}" if visa_category not in ["unknown", "other", ""] else sub_query
        doc_results = search_immigration_docs.invoke({
            "query": enhanced_query,
            "n_results": 5  # Increased for better coverage
        })
        
        if doc_results and "No relevant documents" not in doc_results:
            all_docs.append({
                "sub_query": sub_query,
                "results": doc_results
            })
    
    # Tool 2: Web search for recent updates
    print(f"   ├─ Searching web for recent updates...")
    web_query = f"UK immigration {visa_category} {query} 2024 2025 site:gov.uk"
    
    try:
        web_results = search_govuk_updates.invoke({"query": web_query})
        if web_results and "error" not in web_results.lower():
            all_web_results.append({
                "query": web_query,
                "results": web_results
            })
    except Exception as e:
        print(f"   ├─ ⚠️ Web search failed: {e}")
    
    print(f"   └─ Retrieved {len(all_docs)} document sets, {len(all_web_results)} web results")
    
    return {
        "retrieved_docs": all_docs,
        "web_results": all_web_results
    }


# ============================================================
# AGENT 3: ANALYST AGENT (ANTI-HALLUCINATION)
# ============================================================

ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a UK immigration policy analyst. Your job is to analyze retrieved documents and extract key information.

CRITICAL RULES - NEVER VIOLATE:
1. ONLY state facts that are EXPLICITLY written in the retrieved documents
2. NEVER invent, predict, or speculate about future law changes unless the EXACT rule is published and retrieved
3. If a date or figure is not in the documents, say "not specified in retrieved documents"
4. If documents conflict, state BOTH versions and flag the conflict
5. NEVER say "rules will change" unless you can cite the exact published Statement of Changes
6. If asked about a topic and NO relevant documents were retrieved, say "No information retrieved on this topic"

TEMPORAL AWARENESS:
- Today's date is {current_date}
- Immigration law changes frequently - flag if retrieved docs may be outdated
- Do NOT assume rules from one year apply to another unless explicitly stated

Analyze the provided context and:
1. Identify relevant eligibility requirements WITH EXACT FIGURES where available
2. Extract key dates, deadlines, and timeframes FROM THE DOCUMENTS ONLY
3. List required documents
4. Note any policy changes WITH THEIR EFFECTIVE DATES
5. Flag ambiguities or MISSING information
6. Assess confidence: HIGH only if exact figures/rules are in documents

Format your response as:
**Key Requirements:**
- [requirement 1 with exact figure if available]
- [requirement 2]

**Analysis:**
[Your analysis - ONLY from retrieved documents]

**Information Gaps:**
[What was NOT found in the documents]

**Confidence:** [0.0-1.0]
[Explanation - lower if figures are missing or docs may be outdated]"""),
    ("human", """Query: {query}
Query Type: {query_type}
Visa Category: {visa_category}
Current Date: {current_date}

Retrieved Documents:
{retrieved_docs}

Recent Web Results:
{web_results}

Provide your analysis (ONLY from the above documents, NO speculation):""")
])


def analyst_agent(state: AgentState) -> Dict[str, Any]:
    """
    AGENT 3: Analyst
    Analyzes retrieved information - STRICT anti-hallucination.
    """
    print("\n🔬 [Analyst Agent] Analyzing information...")
    
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    llm = get_llm(temperature=0.0)  # Zero temperature for factual accuracy
    chain = ANALYST_PROMPT | llm | StrOutputParser()
    
    # Format retrieved docs
    docs_text = ""
    for doc_set in state.get("retrieved_docs", []):
        docs_text += f"\n--- Sub-query: {doc_set['sub_query']} ---\n"
        docs_text += doc_set["results"]
    
    # Format web results
    web_text = ""
    for web_set in state.get("web_results", []):
        web_text += f"\n--- Web Search ---\n"
        web_text += web_set["results"]
    
    if not docs_text and not web_text:
        print("   └─ ⚠️ No context available for analysis")
        return {
            "analysis": "No relevant documents were retrieved for this query. Cannot provide analysis without source documents.",
            "key_requirements": [],
            "confidence_score": 0.0
        }
    
    try:
        analysis = chain.invoke({
            "query": state["query"],
            "query_type": state.get("query_type", "general"),
            "visa_category": state.get("visa_category", "unknown"),
            "retrieved_docs": docs_text or "No documents retrieved.",
            "web_results": web_text or "No web results available.",
            "current_date": current_date
        })
        
        # Extract confidence score - DEFAULT TO MEDIUM, not high
        confidence = 0.6  # Default to medium, not high
        if "confidence:" in analysis.lower():
            try:
                conf_section = analysis.lower().split("confidence:")[1][:20]
                for word in conf_section.split():
                    try:
                        confidence = float(word.strip())
                        break
                    except ValueError:
                        continue
            except:
                pass
        
        # REDUCE confidence if analysis contains uncertainty markers
        uncertainty_markers = [
            "not specified", "not found", "no information", "unclear",
            "may be outdated", "conflict", "ambiguous", "not in the documents",
            "cannot confirm", "not retrieved"
        ]
        
        for marker in uncertainty_markers:
            if marker in analysis.lower():
                confidence = min(confidence, 0.5)
                break
        
        # Extract key requirements
        key_reqs = []
        if "**key requirements:**" in analysis.lower():
            try:
                req_section = analysis.split("**Key Requirements:**")[1].split("**")[0]
                for line in req_section.split("\n"):
                    line = line.strip()
                    if line.startswith("-") or line.startswith("•"):
                        key_reqs.append(line[1:].strip())
            except:
                pass
        
        print(f"   ├─ Confidence Score: {confidence:.2f}")
        print(f"   └─ Key Requirements Found: {len(key_reqs)}")
        
        return {
            "analysis": analysis,
            "key_requirements": key_reqs[:10],
            "confidence_score": confidence
        }
    except Exception as e:
        print(f"   └─ ❌ Error: {e}")
        return {
            "analysis": f"Analysis error: {str(e)}",
            "key_requirements": [],
            "confidence_score": 0.0
        }

# ============================================================
# AGENT 4: RESPONSE AGENT (CAUTIOUS)
# ============================================================

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are JapaPolicy AI, a UK Immigration Policy Assistant.

CRITICAL RULES - FOLLOW STRICTLY:
1. ONLY state facts from the analysis - NEVER invent rules or figures
2. If the analysis says "not specified" or "not found", YOU must say the same
3. NEVER predict future law changes unless citing a PUBLISHED Statement of Changes with its reference number
4. If confidence is LOW, clearly state limitations BEFORE giving any information
5. When figures conflict, present BOTH and tell user to verify on gov.uk
6. NEVER use phrases like "rules will change" or "expected to change" without published law

RESPONSE STRUCTURE:
- Start with direct answer (if known) or "I cannot confirm..." (if not in documents)
- Cite sources with [Source: filename, Page X] format
- End with "Verify on gov.uk" for any uncertain information

Confidence Level: {confidence}
- High (≥0.8): Answer confidently WITH citations
- Medium (0.5-0.8): Answer with caveats, recommend verification
- Low (<0.5): State limitations clearly, recommend official sources

FORBIDDEN PHRASES (never use these without published law):
- "From [future date], the rules will..."
- "The government plans to..."
- "Expected changes include..."
- "It is anticipated that..."

REQUIRED PHRASES when information is missing:
- "This specific figure is not in the retrieved documents"
- "I could not find information on [X] in the available sources"
- "Please verify current figures on gov.uk as immigration rules change frequently"
"""),
    ("human", """User Question: {query}

Query Type: {query_type}
Visa Category: {visa_category}

Analysis (use ONLY this information):
{analysis}

Key Requirements:
{key_requirements}

Provide your response (ONLY from the analysis above, NO invention):""")
])


def response_agent(state: AgentState) -> Dict[str, Any]:
    """
    AGENT 4: Response Generator
    Creates cautious, factual responses with appropriate caveats.
    """
    print("\n💬 [Response Agent] Generating response...")
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    llm = get_llm(temperature=0.2)  # Low temperature for accuracy
    chain = RESPONSE_PROMPT | llm | StrOutputParser()
    
    confidence = state.get("confidence_score", 0.5)
    
    if confidence >= 0.8:
        confidence_label = "HIGH - Information well-supported by documents"
    elif confidence >= 0.6:
        confidence_label = "MEDIUM - Some information found but gaps exist"
    elif confidence >= 0.4:
        confidence_label = "LOW-MEDIUM - Limited information, significant gaps"
    else:
        confidence_label = "LOW - Very limited information found"
    
    key_reqs_text = "\n".join([f"- {req}" for req in state.get("key_requirements", [])]) or "No specific requirements extracted from documents."
    
    freshness_warning = get_freshness_warning(state["query"])
    
    try:
        response = chain.invoke({
            "query": state["query"],
            "query_type": state.get("query_type", "general"),
            "visa_category": state.get("visa_category", "unknown"),
            "analysis": state.get("analysis", "No analysis available."),
            "key_requirements": key_reqs_text,
            "confidence": f"{confidence:.2f}",
            "confidence_label": confidence_label,
            "current_date": current_date,
            "freshness_warning": freshness_warning or "None"
        })
        
        # Add freshness warning to response if relevant
        if freshness_warning and freshness_warning not in response:
            response = f"{freshness_warning}\n\n{response}"
        
        # Ensure gov.uk recommendation is present
        if "gov.uk" not in response.lower():
            response += "\n\n**Important:** Immigration rules change frequently. Always verify current requirements on the official gov.uk website before making any decisions or applications."
        
        # Extract sources from retrieved docs
        sources_cited = []
        for doc_set in state.get("retrieved_docs", []):
            results = doc_set.get("results", "")
            if "Source:" in results:
                for line in results.split("\n"):
                    if "Source:" in line and ("Page" in line or "page" in line):
                        sources_cited.append({"reference": line.strip()[:200]})
        
        print(f"   └─ Response generated ({len(response)} chars)")
        
        return {
            "final_response": response,
            "sources_cited": sources_cited[:5]
        }
    except Exception as e:
        print(f"   └─ ❌ Error: {e}")
        return {
            "final_response": (
                "I apologize, but I encountered an error generating the response. "
                "For accurate and up-to-date information on UK immigration, please check the official gov.uk website directly.\n\n"
                f"Error: {str(e)}"
            ),
            "sources_cited": []
        }


# ============================================================
# CLARIFICATION NODE
# ============================================================

def clarification_node(state: AgentState) -> Dict[str, Any]:
    """
    Handle cases where clarification is needed.
    This should RARELY be triggered with the new router.
    """
    print("\n❓ [Clarification] Requesting more information...")
    
    question = state.get("clarification_question", "Could you please provide more details about your immigration question?")
    
    return {
        "final_response": f"To provide accurate information, I need a bit more detail:\n\n{question}\n\nAlternatively, you can check gov.uk for comprehensive immigration guidance.",
        "sources_cited": []
    }


# ============================================================
# HUMAN REVIEW NODE
# ============================================================

def human_review_node(state: AgentState) -> Dict[str, Any]:
    """
    Flag response for human review when confidence is low.
    """
    print("\n⚠️ [Human Review] Low confidence - adding disclaimer...")
    
    analysis = state.get("analysis", "")
    analysis += (
        "\n\n⚠️ **LOW CONFIDENCE NOTICE:** "
        "The retrieved documents may not fully address this query. "
        "Information provided should be verified with official sources before relying on it."
    )
    
    return {
        "analysis": analysis
    }