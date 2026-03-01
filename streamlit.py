"""
JapaPolicy AI - UK Immigration Agentic RAG Assistant
Modern Streamlit Interface with Dark Mode Support
"""

import streamlit as st
import time
import uuid
import base64
from datetime import datetime
from src.app import AgenticRAGAssistant


# PAGE CONFIGURATION
st.set_page_config(
    page_title="JapaPolicy AI | UK Immigration Assistant",
    page_icon="🇬🇧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# HELPER FUNCTIONS
def get_image_base64(image_path):
    """Converts a local image to a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None

# Load UK flag image
uk_flag_base64 = get_image_base64("UK.png")

# CUSTOM CSS STYLING (DARK MODE FRIENDLY)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root CSS Variables for Theme Support */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-tertiary: #e9ecef;
        --text-primary: #1a1a2e;
        --text-secondary: #495057;
        --text-muted: #6c757d;
        --border-color: #dee2e6;
        --accent-primary: #1e3a5f;
        --accent-secondary: #2d5a87;
        --accent-light: rgba(45, 90, 135, 0.1);
        --success: #28a745;
        --warning: #ffc107;
        --danger: #dc3545;
        --shadow: rgba(0, 0, 0, 0.1);
    }
    
    /* Dark Mode Variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-tertiary: #0f3460;
            --text-primary: #eaeaea;
            --text-secondary: #b8b8b8;
            --text-muted: #888888;
            --border-color: #2d3748;
            --accent-primary: #4a9eff;
            --accent-secondary: #63b3ff;
            --accent-light: rgba(74, 158, 255, 0.15);
            --success: #48bb78;
            --warning: #ecc94b;
            --danger: #fc8181;
            --shadow: rgba(0, 0, 0, 0.3);
        }
    }
    
    /* Streamlit Dark Mode Detection */
    [data-testid="stAppViewContainer"][data-theme="dark"],
    .stApp[data-theme="dark"] {
        --bg-primary: #1a1a2e;
        --bg-secondary: #16213e;
        --bg-tertiary: #0f3460;
        --text-primary: #eaeaea;
        --text-secondary: #b8b8b8;
        --text-muted: #888888;
        --border-color: #2d3748;
        --accent-primary: #4a9eff;
        --accent-secondary: #63b3ff;
        --accent-light: rgba(74, 158, 255, 0.15);
        --shadow: rgba(0, 0, 0, 0.3);
    }
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header Styles */
    .header-container {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px var(--shadow);
    }
    
    .header-title {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .header-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        color: white;
        display: inline-block;
        margin-top: 0.75rem;
    }
    
    /* Chat Container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Chat Message Styles */
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: var(--accent-light);
        border-left: 4px solid var(--accent-primary);
        margin-left: 1rem;
        color: var(--text-primary);
    }
    
    .assistant-message {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px var(--shadow);
        color: var(--text-primary);
    }
    
    .message-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.5rem;
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    
    .message-content {
        line-height: 1.6;
        color: var(--text-primary);
    }
    
    /* Confidence Badge */
    .confidence-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 8px;
    }
    
    .confidence-high {
        background: rgba(40, 167, 69, 0.15);
        color: var(--success);
        border: 1px solid var(--success);
    }
    
    .confidence-medium {
        background: rgba(255, 193, 7, 0.15);
        color: var(--warning);
        border: 1px solid var(--warning);
    }
    
    .confidence-low {
        background: rgba(220, 53, 69, 0.15);
        color: var(--danger);
        border: 1px solid var(--danger);
    }
    
    /* Category Tags */
    .category-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 500;
        background: var(--bg-tertiary);
        color: var(--text-secondary);
        margin-right: 6px;
        border: 1px solid var(--border-color);
    }
    
    /* Welcome Card */
    .welcome-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .welcome-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }
    
    .welcome-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .welcome-text {
        color: var(--text-muted);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Clear Button Styling */
    .clear-btn-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 0.5rem;
    }
    
    .clear-btn {
        background: transparent;
        border: 1px solid var(--border-color);
        color: var(--text-muted);
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    
    .clear-btn:hover {
        background: var(--danger);
        border-color: var(--danger);
        color: white;
    }
    
    /* Custom Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 1rem 0;
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 0.5rem;
    }
    
    .typing-dot {
        width: 6px;
        height: 6px;
        background: var(--accent-primary);
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
        30% { transform: translateY(-8px); opacity: 1; }
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    /* Button Styling */
    .stButton > button {
        border-radius: 8px;
        padding: 0.4rem 1rem;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        border: 1px solid var(--border-color);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px var(--shadow);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-secondary);
        background: var(--bg-tertiary);
        border-radius: 8px;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input,
    .stChatInput > div > div > textarea {
        border-radius: 12px;
        border: 1px solid var(--border-color);
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        color: var(--accent-primary);
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        padding: 1rem 0;
        color: var(--text-muted);
        font-size: 0.75rem;
    }
    
    .footer-text a {
        color: var(--accent-primary);
        text-decoration: none;
    }
    
    .footer-text a:hover {
        text-decoration: underline;
    }
    
    /* Warning/Info Boxes */
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid var(--warning);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: var(--text-primary);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-tertiary);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--text-muted);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
</style>
""", unsafe_allow_html=True)


# INITIALIZE SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"

if "assistant" not in st.session_state:
    st.session_state.assistant = None

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# INITIALIZE ASSISTANT
@st.cache_resource
def initialize_assistant():
    """Initialize the Agentic RAG Assistant (cached)."""
    try:
        assistant = AgenticRAGAssistant(
            enable_memory=True,
            enable_hitl=False
        )
        return assistant
    except Exception as e:
        st.error(f"❌ Failed to initialize assistant: {e}")
        return None

# Load assistant
if st.session_state.assistant is None:
    with st.spinner("🚀 Initializing JapaPolicy AI..."):
        st.session_state.assistant = initialize_assistant()

assistant = st.session_state.assistant

# SIDEBAR
with st.sidebar:
    # Logo and Title
    if uk_flag_base64:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0;">
            <img src="data:image/png;base64,{uk_flag_base64}" width="50" style="border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
            <h2 style="margin: 0.5rem 0; color: var(--text-primary);">JapaPolicy AI</h2>
            <p style="color: var(--text-muted); font-size: 0.8rem;">UK Immigration Assistant</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.5rem;">🇬🇧</div>
            <h2 style="margin: 0.5rem 0;">JapaPolicy AI</h2>
            <p style="color: var(--text-muted); font-size: 0.8rem;">UK Immigration Assistant</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    
    
    # Quick Actions
    st.markdown("##### ⚡ Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear chat history"):
            st.session_state.messages = []
            st.session_state.show_welcome = True
            if assistant:
                assistant.clear_history(st.session_state.thread_id)
            st.rerun()
    
    with col2:
        if st.button("🔄 New", use_container_width=True, help="Start new session"):
            st.session_state.messages = []
            st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state.total_queries = 0
            st.session_state.show_welcome = True
            st.rerun()
    
    st.markdown("---")
    
    # Popular Topics
    st.markdown("##### 🏷️ Quick Topics")
    
    topics = [
        ("💼", "Skilled Worker"),
        ("🎓", "Student Visa"),
        ("👨‍👩‍👧", "Family Visa"),
        ("🏠", "ILR / Settlement"),
        ("🔄", "Visa Switching"),
    ]
    
    for icon, topic in topics:
        if st.button(f"{icon} {topic}", use_container_width=True, key=f"topic_{topic}"):
            st.session_state.pending_question = f"What are the requirements for a {topic} visa?"
            st.session_state.show_welcome = False
            st.rerun()
    
    st.markdown("---")
    
    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **JapaPolicy AI** uses:
        - 🤖 5 AI Agents
        - 📚 60+ Official Docs
        - 🔍 Hybrid RAG Search
        
        ⚠️ *Not legal advice. Verify on gov.uk*
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-text">
        Built by <a href="https://www.linkedin.com/in/egwudaojonugwa/" target="_blank">Ojonugwa Egwuda</a><br>
        © 2026 JapaPolicy AI
    </div>
    """, unsafe_allow_html=True)

# MAIN CONTENT

# Header
if uk_flag_base64:
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">
            <img src="data:image/png;base64,{uk_flag_base64}" width="40" style="border-radius: 4px;">
            JapaPolicy AI
        </h1>
        <p class="header-subtitle">Your Intelligent UK Immigration Assistant</p>
        <span class="header-badge">✨ Powered by Agentic RAG</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🇬🇧 JapaPolicy AI</h1>
        <p class="header-subtitle">Your Intelligent UK Immigration Assistant</p>
        <span class="header-badge">✨ Powered by Agentic RAG</span>
    </div>
    """, unsafe_allow_html=True)

# Check if assistant is available
if not assistant:
    st.error("""
    ⚠️ **Assistant Initialization Failed**
    
    Please ensure:
    1. Run `python build_db.py` first
    2. Check `./chroma_db` exists
    3. Verify `.env` has required API keys
    """)
    st.stop()


# CLEAR BUTTON (Above Chat)
if len(st.session_state.messages) > 0:
    col1, col2, col3 = st.columns([6, 1, 1])
    with col2:
        if st.button("🗑️", help="Clear conversation", key="clear_top"):
            st.session_state.messages = []
            st.session_state.show_welcome = True
            if assistant:
                assistant.clear_history(st.session_state.thread_id)
            st.rerun()
    with col3:
        st.caption(f"{len(st.session_state.messages)} msgs")


# WELCOME SCREEN
if st.session_state.show_welcome and len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">👋</div>
        <div class="welcome-title">Welcome to JapaPolicy AI!</div>
        <div class="welcome-text">
            Ask me anything about UK immigration visas,<br>eligibility, requirements, and more.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("##### 💡 Try asking:")
    
    examples = [
        ("💼", "Skilled Worker Visa", "What is the minimum salary requirement for a Skilled Worker visa?"),
        ("🔄", "Visa Switching", "Can I switch from a Graduate visa to a Skilled Worker visa?"),
        ("🏠", "Settlement (ILR)", "How long do I need to be in the UK to apply for ILR?"),
        ("📋", "Section 3C Leave", "My visa expires soon but my extension is pending. Can I still work?"),
        ("🎓", "English Test", "I have a degree taught in English. Am I exempt from the English test?"),
        ("⚠️", "Overstaying", "What happens if someone overstays their visa by 45 days?"),
    ]
    
    cols = st.columns(2)
    for idx, (icon, title, question) in enumerate(examples):
        with cols[idx % 2]:
            if st.button(f"{icon} {title}", key=f"ex_{idx}", use_container_width=True, help=question):
                st.session_state.pending_question = question
                st.session_state.show_welcome = False
                st.rerun()

# CHAT HISTORY

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">👤 You</div>
            <div class="message-content">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        meta = message.get("metadata", {})
        confidence = meta.get("confidence", "medium")
        confidence_emoji = meta.get("confidence_emoji", "🟡")
        query_type = meta.get("query_type", "")
        visa_category = meta.get("visa_category", "")
        
        conf_class = f"confidence-{confidence}"
        
        # Header with confidence
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-header">
                🤖 JapaPolicy AI
                <span class="confidence-badge {conf_class}">{confidence_emoji} {confidence.upper()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Category tags
        if query_type or visa_category:
            tags_html = ""
            if query_type and query_type not in ["unknown", "error"]:
                tags_html += f'<span class="category-tag">📂 {query_type.replace("_", " ").title()}</span>'
            if visa_category and visa_category not in ["unknown", "other"]:
                tags_html += f'<span class="category-tag">🏷️ {visa_category.replace("_", " ").title()}</span>'
            if tags_html:
                st.markdown(f'<div style="margin-bottom: 0.5rem;">{tags_html}</div>', unsafe_allow_html=True)
        
        # Message content
        st.markdown(message["content"])
        
        # Details expander
        with st.expander("📊 Details"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Confidence", f"{meta.get('confidence_score', 0):.0%}")
            with c2:
                st.metric("Type", query_type.replace("_", " ").title()[:15] if query_type else "N/A")
            with c3:
                st.metric("Category", visa_category.replace("_", " ").title()[:15] if visa_category else "N/A")
            
            key_reqs = meta.get("key_requirements", [])
            if key_reqs:
                st.markdown("**📋 Key Points:**")
                for req in key_reqs[:3]:
                    st.markdown(f"• {req[:100]}...")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# PROCESS PENDING QUESTION

if st.session_state.pending_question:
    pending = st.session_state.pending_question
    st.session_state.pending_question = None
    
    st.session_state.messages.append({"role": "user", "content": pending})
    
    with st.spinner("🔍 Researching..."):
        result = assistant.invoke(pending, thread_id=st.session_state.thread_id)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", "Sorry, an error occurred."),
        "metadata": {
            "confidence": result.get("confidence", "low"),
            "confidence_emoji": result.get("confidence_emoji", "🔴"),
            "confidence_score": result.get("confidence_score", 0),
            "query_type": result.get("query_type", ""),
            "visa_category": result.get("visa_category", ""),
            "key_requirements": result.get("key_requirements", []),
            "sources": result.get("sources", [])
        }
    })
    
    st.session_state.total_queries += 1
    st.rerun()


# CHAT INPUT

user_input = st.chat_input("Ask about UK immigration...")

if user_input:
    st.session_state.show_welcome = False
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("🔍 Researching..."):
        try:
            result = assistant.invoke(user_input, thread_id=st.session_state.thread_id)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("answer", "Sorry, an error occurred."),
                "metadata": {
                    "confidence": result.get("confidence", "low"),
                    "confidence_emoji": result.get("confidence_emoji", "🔴"),
                    "confidence_score": result.get("confidence_score", 0),
                    "query_type": result.get("query_type", ""),
                    "visa_category": result.get("visa_category", ""),
                    "key_requirements": result.get("key_requirements", []),
                    "sources": result.get("sources", [])
                }
            })
            
            st.session_state.total_queries += 1
            
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}. Please try again.",
                "metadata": {"confidence": "low", "confidence_emoji": "🔴"}
            })
    
    st.rerun()



# FOOTER
st.markdown("---")

cols = st.columns(3)
with cols[0]:
    st.caption("📚 60+ Official Documents")
with cols[1]:
    st.caption("🤖 5 AI Agents")
with cols[2]:
    st.caption("🔍 Hybrid RAG Search")

st.markdown("""
<style>
    .footer-container {
        background: var(--bg-secondary, #f8f9fa);
        border: 1px solid var(--border-color, #dee2e6);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .footer-container h4 {
        color: var(--text-primary, #1a1a2e);
        margin-bottom: 0.5rem;
    }
    .footer-container p {
        color: var(--text-secondary, #495057);
        margin-bottom: 0.5rem;
    }
    .footer-container a {
        color: var(--accent-primary, #1e3a5f);
        text-decoration: none;
        font-weight: 600;
    }
    .footer-container a:hover {
        text-decoration: underline;
    }
    .footer-container small {
        color: var(--text-muted, #6c757d);
    }
</style>

<div class='footer-container'>
    <h4>🤖 Contact the Developer</h4>
    <p>Connect with <strong>Ojonugwa Egwuda</strong> on 
        <a href="https://www.linkedin.com/in/egwudaojonugwa/" target="_blank">LinkedIn</a>
    </p>
    <small>© 2026 JapaPolicy AI | Built with ❤️ using Streamlit & Claude</small>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 0.5rem 0; color: var(--text-muted, #adb5bd); font-size: 0.75rem;">
    ⚠️ <strong>Disclaimer:</strong> This is an AI assistant, not legal advice. 
    Always verify on <a href="https://www.gov.uk/browse/visas-immigration" target="_blank" style="color: var(--accent-primary, #1e3a5f);">gov.uk</a>
</div>
""", unsafe_allow_html=True)