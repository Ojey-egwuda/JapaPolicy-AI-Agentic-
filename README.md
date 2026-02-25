# 🇬🇧 JapaPolicy AI

🚀 <a href="https://japapolicy.streamlit.app/" target="_blank" rel="noopener noreferrer">Live App</a>

👉 https://japapolicy.streamlit.app/

> **Your Intelligent UK Immigration Assistant** — An Agentic RAG system that answers complex UK immigration questions using multi-agent orchestration and retrieval-augmented generation.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io)
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
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

**JapaPolicy AI** is an intelligent assistant that helps users navigate the complex UK immigration system. Built using a multi-agent architecture with LangGraph, it processes over 50 official UK government documents to provide accurate, cited answers to immigration queries.

The system uses **Agentic RAG** (Retrieval-Augmented Generation) — combining the power of large language models with a specialized knowledge base of UK immigration rules, guidance documents, and policy updates.

### Why "Japa"?

"Japa" is Nigerian slang meaning "to run" or "to relocate abroad" — commonly used when discussing emigration. This tool helps people planning to "japa" to the UK by providing reliable immigration information.

---

## ✨ Features

| Feature                       | Description                                                        |
| ----------------------------- | ------------------------------------------------------------------ |
| 🤖 **Multi-Agent System**     | 4 specialized AI agents working together                           |
| 📚 **50+ Official Documents** | Processes UK gov guidance, Immigration Rules, and policy documents |
| 🔍 **Hybrid Search**          | Combines semantic search with BM25 keyword matching                |
| 🌐 **Web Search**             | Fetches latest policy updates from gov.uk                          |
| 💬 **Conversation Memory**    | Maintains context across multiple questions                        |
| 📊 **Confidence Scoring**     | Rates answer reliability (High/Medium/Low)                         |
| 📝 **Source Citations**       | Every answer includes document references                          |
| 🌙 **Dark Mode Support**      | Modern UI that adapts to system theme                              |
| ⚡ **Query Routing**          | Automatically classifies and routes queries                        |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     🔀 ROUTER AGENT                              │
│  • Classifies query type (eligibility, switching, ILR, etc.)    │
│  • Identifies visa category (Skilled Worker, Student, etc.)     │
│  • Decomposes complex queries into sub-queries                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    🔍 RETRIEVER AGENT                            │
│  • Searches vector database (ChromaDB)                          │
│  • Performs hybrid search (semantic + BM25)                     │
│  • Fetches web updates from gov.uk                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    🔬 ANALYST AGENT                              │
│  • Extracts key requirements from documents                     │
│  • Identifies dates, deadlines, figures                         │
│  • Assigns confidence score                                     │
│  • Flags policy changes and ambiguities                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    💬 RESPONSE AGENT                             │
│  • Generates user-friendly response                             │
│  • Adds source citations                                        │
│  • Adjusts tone based on confidence                             │
│  • Includes next steps and recommendations                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FINAL ANSWER                              │
│  With confidence badge, citations, and gov.uk verification note │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Component           | Technology                              |
| ------------------- | --------------------------------------- |
| **LLM**             | Google Gemini 2.0 Flash                 |
| **Agent Framework** | LangGraph                               |
| **Vector Database** | ChromaDB                                |
| **Embeddings**      | sentence-transformers/all-mpnet-base-v2 |
| **Web Search**      | Tavily API                              |
| **Frontend**        | Streamlit                               |
| **Language**        | Python 3.10+                            |

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
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-2.0-flash

# Optional (for web search)
TAVILY_API_KEY=tvly-your_tavily_api_key_here
```

**Get API Keys:**

- Google API Key: [Google AI Studio](https://makersuite.google.com/app/apikey)
- Tavily API Key: [Tavily](https://tavily.com/) (optional, for web search)

### Step 5: Add Immigration Documents

Place UK immigration PDF documents in the `./data/` folder:

```
data/
├── Skilled_Worker_Guidance.pdf
├── Student_Visa_Rules.pdf
├── Family_Visa_Guidance.pdf
├── ILR_Requirements.pdf
└── ... (other official documents)
```

**Recommended documents to include:**

- Immigration Rules (Appendix Skilled Worker, Student, etc.)
- Home Office caseworker guidance
- Gov.uk visa guidance pages (saved as PDF)

### Step 6: Build the Vector Database

```bash
python build_db.py
```

This will:

- Process all PDFs in `./data/`
- Create embeddings using sentence-transformers
- Store in ChromaDB at `./chroma_db/`

**Expected output:**

```
Loading documents from ./data...
Found 52 PDF files
Processing: Skilled_Worker_Guidance.pdf...
...
Total pages processed: 1,731
Building vector database...
Database built successfully!
```

---

## 🚀 Usage

### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Option 2: Command Line Interface

```bash
# Interactive mode
python -m src.app

# Test mode (runs predefined test queries)
python -m src.app --test
```

### Option 3: Python API

```python
from src.app import AgenticRAGAssistant

# Initialize
assistant = AgenticRAGAssistant(enable_memory=True)

# Ask a question
result = assistant.invoke(
    "What is the minimum salary for a Skilled Worker visa?",
    thread_id="my_session"
)

print(result["answer"])
print(f"Confidence: {result['confidence']}")
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
│   ├── workers.py              # All 4 agent implementations
│   ├── graph.py                # LangGraph workflow definition
│   └── vectordb.py             # ChromaDB wrapper with hybrid search
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

### 1. Router Agent

Classifies incoming queries and determines the best approach:

- **Query Types:** visa_eligibility, visa_switching, visa_extension, ilr_application, citizenship, general_info
- **Visa Categories:** skilled_worker, health_care, student, graduate, family, visitor, global_talent

### 2. Retriever Agent

Gathers relevant information using multiple tools:

- **Vector Search:** Semantic search through immigration documents
- **Web Search:** Latest updates from gov.uk (via Tavily)
- **Hybrid Search:** Combines semantic + keyword (BM25) matching

### 3. Analyst Agent

Processes retrieved information:

- Extracts key requirements and figures
- Identifies dates and deadlines
- Flags policy changes
- Assigns confidence score (0.0-1.0)

### 4. Response Agent

Generates the final answer:

- Structures response clearly
- Adds source citations
- Adjusts tone based on confidence
- Includes verification recommendations

### Tools Available

| Tool                      | Purpose                                 |
| ------------------------- | --------------------------------------- |
| `search_immigration_docs` | Vector database search                  |
| `search_govuk_updates`    | Web search for recent policy changes    |
| `calculate_visa_dates`    | Calculate ILR eligibility, visa expiry  |
| `check_basic_eligibility` | Pre-check visa eligibility requirements |

---

## 💬 Example Queries

The system handles a wide range of UK immigration questions:

**Visa Eligibility**

- "What is the minimum salary for a Skilled Worker visa?"
- "Can I bring my family on a Student visa?"
- "Am I exempt from the English test with a Nigerian degree?"

**Visa Switching**

- "Can I switch from Graduate to Skilled Worker visa?"
- "Can a visitor switch to a spouse visa inside the UK?"

**Settlement (ILR)**

- "How long do I need to work to get ILR?"
- "What happens if I spend 200 days outside the UK?"

**Complex Scenarios**

- "My visa expires next week but my extension is pending. Can I work?"
- "What happens if I overstay by 45 days?"

---

## ⚠️ Limitations

| Limitation             | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| **Not Legal Advice**   | This is an AI assistant, not a qualified immigration adviser |
| **Document Freshness** | Answers depend on the documents in your database             |
| **Complex Cases**      | Edge cases may require professional consultation             |
| **No Case Assessment** | Cannot assess individual applications                        |
| **UK Only**            | Focused exclusively on UK immigration                        |

---

## 🔮 Future Improvements

- [ ] Add more authoritative facts for common queries
- [ ] Implement document freshness checking
- [ ] Add user feedback mechanism
- [ ] Create API endpoint for integration
- [ ] Add multi-language support
- [ ] Implement caching for frequent queries
- [ ] Add analytics dashboard
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
