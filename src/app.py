"""
UK Immigration Agentic RAG Assistant — app.py

Changes from previous version:
  • ChromaDB pre-warmed at startup so first user query is not slow
"""

import os
import uuid
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from .state   import AgentState
from .graph   import graph, build_graph, visualize_graph
from .tracing import get_run_metadata

load_dotenv()


class AgenticRAGAssistant:
    """
    Agentic RAG Assistant for UK Immigration queries.
    Multi-agent LangGraph workflow: Decomposition → Router → Retriever (HyDE)
    → Analyst → Responder.
    """

    def __init__(self, enable_memory: bool = True, enable_hitl: bool = False):
        print("\n Initializing Agentic RAG Assistant…")
        print("=" * 60)

        self.enable_memory = enable_memory
        self.enable_hitl   = enable_hitl
        self.graph         = build_graph(with_memory=enable_memory)
        self.conversation_history: Dict[str, list] = {}

        print("Agent Graph Compiled")
        print("Agents: Decomposition → Router → Retriever (HyDE) → Analyst → Responder")
        print("Tools:  Vector Search (HyDE), Web Search, Date Calculator, Eligibility Checker")
        print(f"Memory: {'Enabled' if enable_memory else 'Disabled'}")
        print(f"Human-in-the-loop: {'Enabled' if enable_hitl else 'Disabled'}")

        # ── Pre-warm ChromaDB ─────────────────────────────────────────────────
        # Forces ChromaDB to load all 2,366 chunks into memory at startup so
        # the first user query is not penalised with a 30-70s cold-start cost.
        print("🔥 Pre-warming vector database...")
        try:
            from .tools import get_vector_db
            db = get_vector_db()
            db.collection.query(
                query_embeddings=[[0.0] * 768],
                n_results=1,
                include=[]
            )
            print("✅ Vector database warmed up and ready")
        except Exception as e:
            print(f"⚠️  Warmup failed (non-critical): {e}")

        print("=" * 60)
        print("Agentic RAG Assistant Ready\n")

    # ── Conversation helpers ──────────────────────────────────────────────────

    def _get_conversation_context(self, thread_id: str, max_turns: int = 5) -> str:
        if thread_id not in self.conversation_history:
            return ""
        history = self.conversation_history[thread_id][-max_turns:]
        if not history:
            return ""
        parts = []
        for turn in history:
            parts.append(f"User: {turn['question']}")
            parts.append(f"Assistant: {turn['answer'][:500]}…")
        return "\n".join(parts)

    def _add_to_history(self, thread_id: str, question: str, answer: str, metadata: Dict[str, Any] = None):
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
        self.conversation_history[thread_id].append({
            "question": question,
            "answer":   answer,
            "metadata": metadata or {},
        })
        if len(self.conversation_history[thread_id]) > 20:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-20:]

    def _enhance_query_with_context(self, question: str, thread_id: str) -> str:
        follow_up_indicators = [
            len(question.split()) < 10,
            question.lower().startswith(("yes", "no", "what about", "and ", "also", "how about")),
            "it"       in question.lower() and len(question.split()) < 15,
            "that"     in question.lower() and len(question.split()) < 15,
            "the visa" in question.lower() and "what" not in question.lower(),
        ]
        if any(follow_up_indicators) and thread_id in self.conversation_history:
            history = self.conversation_history[thread_id]
            if history:
                last_q = history[-1]["question"]
                return f"Previous question: {last_q}\nFollow-up question: {question}"
        return question

    # ── Main invoke ───────────────────────────────────────────────────────────

    def invoke(self, question: str, thread_id: str = None) -> Dict[str, Any]:
        """
        Process a user question through the full agent pipeline.

        Args:
            question:  User's immigration question
            thread_id: Optional thread ID for conversation continuity

        Returns:
            Dict with answer, sources, confidence, and metrics
        """
        if not question or not question.strip():
            return self._empty_response("Please enter a valid question.")

        question  = question.strip()
        thread_id = thread_id or str(uuid.uuid4())

        enhanced_question = self._enhance_query_with_context(question, thread_id)

        initial_state: AgentState = {
            "messages":               [],
            "query":                  enhanced_question,
            "query_type":             None,
            "visa_category":          None,
            "needs_clarification":    False,
            "clarification_question": None,
            "decomposed_queries":     [],
            "retrieved_docs":         [],
            "web_results":            [],
            "tool_results":           [],
            "analysis":               None,
            "key_requirements":       [],
            "confidence_score":       0.0,
            "final_response":         None,
            "sources_cited":          [],
        }

        config: Dict[str, Any] = {}
        if self.enable_memory:
            config["configurable"] = {"thread_id": thread_id}
        config["metadata"] = get_run_metadata()

        print(f"\n{'='*60}")
        print(f"Query: {question[:80]}…")
        print(f"{'='*60}")

        try:
            result = self.graph.invoke(initial_state, config)

            confidence_score = result.get("confidence_score", 0.5)
            if confidence_score >= 0.8:
                confidence, emoji = "high",   "🟢"
            elif confidence_score >= 0.6:
                confidence, emoji = "medium", "🟡"
            else:
                confidence, emoji = "low",    "🔴"

            print(f"\n{'='*60}")
            print(f"Response Generated | Confidence: {emoji} {confidence.upper()}")
            print(f"{'='*60}\n")

            answer = result.get("final_response", "No response generated.")

            self._add_to_history(
                thread_id=thread_id,
                question=question,
                answer=answer,
                metadata={
                    "confidence":    confidence,
                    "query_type":    result.get("query_type"),
                    "visa_category": result.get("visa_category"),
                },
            )

            return {
                "answer":           answer,
                "sources":          result.get("sources_cited", []),
                "confidence":       confidence,
                "confidence_emoji": emoji,
                "confidence_score": round(confidence_score, 3),
                "query_type":       result.get("query_type", "unknown"),
                "visa_category":    result.get("visa_category", "unknown"),
                "key_requirements": result.get("key_requirements", []),
                "analysis":         result.get("analysis", ""),
                "search_type":      "agentic-hyde",
                "thread_id":        thread_id,
            }

        except Exception as e:
            print(f"\nError during processing: {e}")
            return self._empty_response(f"An error occurred: {str(e)}")

    # ── History management ────────────────────────────────────────────────────

    def clear_history(self, thread_id: str = None):
        if thread_id:
            self.conversation_history.pop(thread_id, None)
            print(f"Cleared history for thread: {thread_id}")
        else:
            self.conversation_history = {}
            print("Cleared all conversation history")

    def get_history(self, thread_id: str) -> list:
        return self.conversation_history.get(thread_id, [])

    def _empty_response(self, message: str) -> Dict[str, Any]:
        return {
            "answer":           message,
            "sources":          [],
            "confidence":       "low",
            "confidence_emoji": "🔴",
            "confidence_score": 0.0,
            "query_type":       "error",
            "visa_category":    "unknown",
            "key_requirements": [],
            "analysis":         "",
            "search_type":      "none",
            "thread_id":        None,
        }

    def visualize(self):
        visualize_graph()


# Convenience factory
def create_assistant(**kwargs) -> AgenticRAGAssistant:
    return AgenticRAGAssistant(**kwargs)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    test_mode = "--test" in sys.argv or "-t" in sys.argv

    assistant = AgenticRAGAssistant(enable_memory=True, enable_hitl=False)

    if test_mode:
        print("\nRunning test query…")
        result = assistant.invoke("What is the minimum salary for a Skilled Worker visa?")
        print(f"\n{result['confidence_emoji']} Confidence: {result['confidence']}")
        print(f"\n{result['answer']}")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("UK Immigration Assistant — Interactive Mode")
    print("=" * 60)
    print("Type 'quit' to stop | 'clear' to clear history | 'new' for new session")
    print("=" * 60 + "\n")

    thread_id = f"session_{uuid.uuid4().hex[:8]}"
    print(f"Session ID: {thread_id}\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break
        if question.lower() == "clear":
            assistant.clear_history(thread_id)
            print("Conversation history cleared.\n")
            continue
        if question.lower() == "new":
            thread_id = f"session_{uuid.uuid4().hex[:8]}"
            print(f"New session: {thread_id}\n")
            continue
        if not question:
            continue

        result = assistant.invoke(question, thread_id=thread_id)
        print(f"\n{result['confidence_emoji']} Confidence: {result['confidence'].upper()}")
        if result.get("query_type") not in (None, "error"):
            print(f"📂 {result['query_type']} | 🏷️ {result['visa_category']}")
        print(f"\nAssistant:\n{result['answer']}\n")
        print("-" * 60 + "\n")