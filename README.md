# 🇬🇧 JapaPolicy AI

🚀 <a href="https://japapolicy.streamlit.app/" target="_blank" rel="noopener noreferrer">Live App</a>

👉 https://japapolicy.streamlit.app/

> **Your Intelligent UK Immigration Assistant** — An Agentic RAG system that answers complex UK immigration questions using multi-agent orchestration and retrieval-augmented generation.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io)
[![LangSmith](https://img.shields.io/badge/LangSmith-Observability-purple.svg)](https://smith.langchain.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Example Queries](#-example-queries)
- [Performance](#-performance)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

**JapaPolicy AI** is an intelligent assistant that helps users navigate the complex UK immigration system. Built using a 5-agent architecture with LangGraph, it processes over 60 official UK government documents to provide accurate, cited answers to immigration queries.

The system uses **Agentic RAG** (Retrieval-Augmented Generation) — combining the power of large language models with a specialised knowledge base of UK immigration rules, guidance documents, and policy updates. It features **HyDE** (Hypothetical Document Embeddings) for precision retrieval and **Query Decomposition** for handling complex multi-part questions.

### Why "Japa"?

"Japa" is Nigerian slang meaning "to run" or "to relocate abroad" — commonly used when discussing emigration. This tool helps people planning to "japa" to the UK by providing reliable immigration information.

---

## ✨ Features

| Feature                        | Description                                                          |
| ------------------------------ | -------------------------------------------------------------------- |
| 🤖 **5-Agent Pipeline**        | Decomposition → Router → Retriever → Analyst → Responder             |
| 🧪 **HyDE Retrieval**          | Generates hypothetical answers to improve vector search precision    |
| ✂️ **Query Decomposition**     | Breaks compound questions into atomic sub-queries before retrieval   |
| 📚 **60+ Official Documents**  | Processes UK gov guidance, Immigration Rules, and policy documents   |
| 🔍 **Hybrid Search**           | Combines semantic search with BM25 keyword matching via RRF          |
| 🌐 **Web Search**              | Fetches latest policy updates from gov.uk via Tavily                 |
| 💬 **Conversation Memory**     | Maintains context across multiple questions                          |
| 📊 **Confidence Scoring**      | Rates answer reliability (High/Medium/Low)                           |
| 📝 **Source Citations**        | Every answer includes document references and page numbers           |
| 📡 **LangSmith Observability** | Full trace monitoring, latency breakdown, and token tracking         |
| 🌙 **Dark Mode Support**       | Modern UI that adapts to system theme                                |
| ⚡ **ChromaDB Warmup**         | Pre-loads vector database at startup to eliminate cold-start latency |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ✂️  DECOMPOSITION AGENT                         │
│  • Detects compound multi-part questions                        │
│  • Breaks into 2-4 atomic sub-queries                           │
│  • Rule-based fallback if LLM decomposition fails               │
│  • Skips LLM call entirely for simple queries                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     🔀 ROUTER AGENT                              │
│  • Classifies query type (eligibility, switching, ILR, etc.)    │
│  • Identifies visa category (Skilled Worker, Student, etc.)     │
│  • Preserves sub-queries from Decomposition Agent               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 🔍 RETRIEVER AGENT (HyDE)                        │
│  • Generates hypothetical regulatory passage (HyDE vector)      │
│  • Searches ChromaDB with HyDE vector + atomic sub-queries      │
│  • Performs hybrid search (semantic cosine + BM25 via RRF)      │
│  • Fetches recent policy updates from gov.uk                    │
│  • Runs date calculator and eligibility checker tools           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    🔬 ANALYST AGENT                              │
│  • Synthesises documents, web results, and tool outputs         │
│  • Extracts key requirements with exact figures                 │
│  • Identifies policy change dates and deadlines                 │
│  • Assigns confidence score (0.0–1.0)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    💬 RESPONSE AGENT                             │
│  • Generates user-friendly structured response                  │
│  • Adds source citations with document names and page numbers   │
│  • Adjusts tone and certainty based on confidence score         │
│  • Includes next steps and gov.uk verification note             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FINAL ANSWER                              │
│  Confidence badge · Source citations · gov.uk verification note │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Component           | Technology                                         |
| ------------------- | -------------------------------------------------- |
| **LLM**             | Google Gemini 2.5 Flash                            |
| **Agent Framework** | LangGraph                                          |
| **Vector Database** | ChromaDB (cosine similarity)                       |
| **Embeddings**      | sentence-transformers/all-mpnet-base-v2 (768 dims) |
| **Keyword Search**  | BM25Okapi with Reciprocal Rank Fusion              |
| **Web Search**      | Tavily API                                         |
| **Observability**   | LangSmith                                          |
| **Frontend**        | Streamlit                                          |
| **Language**        | Python 3.10+                                       |

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- ~2GB disk space for embeddings and database

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/japapolicy-ai.git
cd japapolicy-ai
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install langsmith
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-2.5-flash

# Optional — web search (recommended)
TAVILY_API_KEY=tvly-your_tavily_api_key_here

# Optional — observability (recommended)
LANGSMITH_API_KEY=ls__your_key_here
LANGSMITH_PROJECT=japapolicy-ai
LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com
```

**Get API Keys:**

- Google API Key: [Google AI Studio](https://aistudio.google.com/app/apikey)
- Tavily API Key: [Tavily](https://tavily.com/)
- LangSmith Key: [LangSmith](https://smith.langchain.com) → Settings → API Keys

### Step 5: Add Immigration Documents

Place UK immigration PDF documents in the `./data/` folder. Recommended documents:

- Immigration Rules Appendix Skilled Worker
- Immigration Rules Appendix Skilled Occupations (Tables 1, 2, 3)
- Home Office caseworker guidance (Skilled Worker, Student, Family)
- Section 3C and 3D Leave guidance
- ILR guidance documents
- Gov.uk visa guidance pages (saved as PDF)

### Step 6: Build the Vector Database

```bash
python build_db.py
```

Expected output:

```
📄 Found 60 PDF files
...
📊 Total pages loaded: 2,366
✅ DATABASE BUILD COMPLETE
   • Total chunks: 2,366
   • Hybrid search: ✅ Enabled
```

---

## 🚀 Usage

### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Option 2: Command Line — Test Mode

```bash
python -m src.app --test
```

### Option 3: Command Line — Interactive Mode

```bash
python -m src.app
```

### Option 4: Python API

```python
from src.app import AgenticRAGAssistant

assistant = AgenticRAGAssistant(enable_memory=True)

result = assistant.invoke(
    "What is the minimum salary for a Skilled Worker visa?",
    thread_id="my_session"
)

print(result["answer"])
print(f"Confidence: {result['confidence']}")
print(f"Query type: {result['query_type']}")
```

---

## 📁 Project Structure

```
japapolicy-ai/
│
├── data/                       # UK immigration PDF documents
│   └── *.pdf
│
├── chroma_db/                  # Vector database (auto-generated)
│
├── src/
│   ├── __init__.py
│   ├── app.py                  # Main application & AgenticRAGAssistant class
│   ├── state.py                # AgentState TypedDict definition
│   ├── tools.py                # Tool implementations (search, calculate, etc.)
│   ├── workers.py              # Router, Analyst, Responder agents
│   ├── decomposition.py        # Decomposition agent (pre-router)
│   ├── hyde_retriever.py       # HyDE-enhanced retriever agent
│   ├── graph.py                # LangGraph workflow definition
│   ├── tracing.py              # LangSmith observability setup
│   └── vectordb.py             # ChromaDB wrapper with hybrid search + caching
│
├── streamlit_app.py            # Streamlit web interface
├── build_db.py                 # Database builder script
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── UK.png                      # UK flag image for UI
└── README.md                   # This file
```

---

## ⚙️ How It Works

### 1. Decomposition Agent

Runs before the router on every query. Detects compound multi-part questions (e.g. "can I work AND can I travel?") and breaks them into independent atomic sub-queries. Simple queries skip the LLM call entirely for speed. Falls back to rule-based splitting if the LLM fails.

### 2. Router Agent

Classifies the query type and visa category. Preserves the decomposed sub-queries from the Decomposition Agent rather than overwriting them. Almost never asks for clarification — infers context from keywords.

- **Query Types:** visa_eligibility, visa_switching, visa_extension, ilr_application, citizenship, general_info
- **Visa Categories:** skilled_worker, health_care, student, graduate, family, visitor, global_talent

### 3. Retriever Agent (HyDE)

The core retrieval innovation. Instead of searching with the raw user question, it first generates a **hypothetical regulatory passage** that would answer the query — then uses that passage as the search vector. This closes the embedding space gap between questions and regulatory document text, significantly improving retrieval precision.

Retrieval pipeline per query:

1. Generate HyDE hypothetical passage (LLM call)
2. Vector search with HyDE passage → typically 87–93% cosine similarity
3. Vector search with each atomic sub-query → 80–87% similarity
4. Hybrid search: semantic cosine + BM25 keyword via Reciprocal Rank Fusion
5. Web search for recent gov.uk policy updates (Tavily)
6. Date calculator and eligibility checker tools where relevant

### 4. Analyst Agent

Synthesises all retrieved context — documents, web results, tool outputs — into a structured analysis. Assigns a confidence score (0.0–1.0) based on source quality. Context is trimmed to ~1,500 chars per document set to keep token counts efficient.

### 5. Response Agent

Generates the final user-facing answer with section headers, bullet points, source citations (document name + page number), and a gov.uk verification note. Tone adjusts based on confidence score.

### Tools Available

| Tool                      | Purpose                                                 |
| ------------------------- | ------------------------------------------------------- |
| `search_immigration_docs` | Hybrid vector + BM25 search through 60+ documents       |
| `search_govuk_updates`    | Live web search for recent policy changes via Tavily    |
| `calculate_visa_dates`    | ILR eligibility dates, visa expiry, absence compliance  |
| `check_basic_eligibility` | Pre-check salary, sponsorship, and English requirements |

---

## 💬 Example Queries

**Simple eligibility**

- "What is the minimum salary for a Skilled Worker visa?"
- "Am I exempt from the English language test with a Nigerian degree?"
- "Can I bring my family on a Student visa?"

**Visa switching**

- "Can I switch from a Graduate visa to a Skilled Worker visa inside the UK?"
- "Can a visitor switch to a spouse visa without leaving the UK?"

**Settlement (ILR)**

- "How long do I need to be on a Skilled Worker visa before applying for ILR?"
- "I spent 210 days outside the UK across 5 years — am I still eligible for ILR?"

**Section 3C / pending applications**

- "My visa expires next week but my extension is pending — can I still work?"
- "My sponsor's licence was suspended while my application is pending — what happens?"

**Complex compound queries**

- "I'm on a Graduate visa expiring in 6 weeks, I have a job offer for £35,000, my employer has a sponsor licence — can I switch to Skilled Worker, and if my application is pending can I still work and travel to Nigeria?"

---

## 📊 Performance

Benchmarked using LangSmith traces across 10 complex immigration queries:

| Node                             | Typical Latency              |
| -------------------------------- | ---------------------------- |
| Decomposition                    | 0–2s (0s for simple queries) |
| Router                           | ~1.5–2s                      |
| Retriever (HyDE + 2 sub-queries) | ~8–10s                       |
| Analyst                          | ~10–13s                      |
| Responder                        | ~5–6s                        |
| **Total end-to-end**             | **~28–35s**                  |

**Retrieval quality:**

- HyDE vector search: 87–93% cosine similarity on official documents
- Without HyDE (raw query): often 0 results on the same queries

**Confidence distribution across 10 stress-test queries:**

- HIGH (≥0.8): 8/10 queries
- MEDIUM (0.6–0.8): 2/10 queries
- LOW: 0/10 queries

---

## ⚠️ Limitations

| Limitation             | Description                                                                   |
| ---------------------- | ----------------------------------------------------------------------------- |
| **Not Legal Advice**   | This is an AI assistant, not a qualified immigration adviser                  |
| **Document Freshness** | Answers depend on the documents in your database — rebuild periodically       |
| **SOC Table Lookup**   | Specific going rates require Appendix Skilled Occupations Tables 1–3 in data/ |
| **Processing Times**   | Live UKVI processing times require Tavily API for accurate results            |
| **Complex Cases**      | Edge cases may require professional consultation                              |
| **UK Only**            | Focused exclusively on UK immigration                                         |

---

## 🔮 Future Improvements

- [ ] Contextual compression to further reduce analyst token cost
- [ ] Semantic caching for repeated queries
- [ ] Document freshness checking with auto-rebuild alerts
- [ ] User feedback mechanism with LangSmith annotation
- [ ] API endpoint for third-party integration
- [ ] Multi-language support
- [ ] Analytics dashboard
- [ ] Support for document uploads by users

---

## 👨‍💻 Author

**Ojonugwa Egwuda**

- LinkedIn: [linkedin.com/in/egwudaojonugwa](https://www.linkedin.com/in/egwudaojonugwa/)
- GitHub: [github.com/ojey-egwuda](https://github.com/Ojey-egwuda)

---

## 📜 Disclaimer

> **⚠️ IMPORTANT: This tool is for informational purposes only.**
>
> JapaPolicy AI is an AI-powered assistant and does **NOT** provide legal advice. The information provided:
>
> - May be incomplete, outdated, or inaccurate
> - Should **ALWAYS** be verified on [gov.uk](https://www.gov.uk/browse/visas-immigration)
> - Does not replace consultation with a qualified immigration adviser
> - Cannot assess individual circumstances or applications
>
> For complex immigration matters, consult a registered [OISC adviser](https://www.gov.uk/find-an-immigration-adviser) or solicitor.
>
> The developers accept no liability for decisions made based on information provided by this tool.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- UK Home Office for publishing immigration guidance
- LangChain team for LangGraph framework
- Anthropic for Claude assistance in development
- The Nigerian tech community for inspiration ("Japa" culture 🚀)

---

<div align="center">

**Made with ❤️ for the immigrant community**

🇬🇧 🇳🇬 🌍

</div>
