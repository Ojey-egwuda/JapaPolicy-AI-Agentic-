"""
UK Immigration Agentic RAG Assistant
Main application entry point with conversation memory support.
"""

import os
import uuid
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from .state import AgentState
from .graph import graph, build_graph, visualize_graph

load_dotenv()


class AgenticRAGAssistant:
    """
    Agentic RAG Assistant for UK Immigration queries.
    Uses a multi-agent LangGraph workflow with 4 specialized agents.
    """
    
    def __init__(self, enable_memory: bool = True, enable_hitl: bool = False):
        """
        Initialize the agentic assistant.
        
        Args:
            enable_memory: Enable conversation memory across invocations
            enable_hitl: Enable human-in-the-loop for low confidence responses
        """
        print("\n Initializing Agentic RAG Assistant...")
        print("=" * 60)
        
        self.enable_memory = enable_memory
        self.enable_hitl = enable_hitl
        
        # Build the graph with options
        self.graph = build_graph(with_memory=enable_memory)
        
        # Store conversation history for context
        self.conversation_history: Dict[str, list] = {}
        
        print("Agent Graph Compiled")
        print("Agents: Router → Retriever → Analyst → Responder")
        print("Tools: Vector Search, Web Search, Date Calculator, Eligibility Checker")
        print(f"Memory: {'Enabled' if enable_memory else 'Disabled'}")
        print(f"Human-in-the-loop: {'Enabled' if enable_hitl else 'Disabled'}")
        print("=" * 60)
        print("Agentic RAG Assistant Ready\n")
    
    def _get_conversation_context(self, thread_id: str, max_turns: int = 5) -> str:
        """
        Get recent conversation history for context.
        
        Args:
            thread_id: Conversation thread identifier
            max_turns: Maximum number of previous turns to include
            
        Returns:
            Formatted conversation history string
        """
        if thread_id not in self.conversation_history:
            return ""
        
        history = self.conversation_history[thread_id][-max_turns:]
        
        if not history:
            return ""
        
        context_parts = []
        for turn in history:
            context_parts.append(f"User: {turn['question']}")
            context_parts.append(f"Assistant: {turn['answer'][:500]}...")  # Truncate long answers
        
        return "\n".join(context_parts)
    
    def _add_to_history(self, thread_id: str, question: str, answer: str, metadata: Dict[str, Any] = None):
        """
        Add a conversation turn to history.
        
        Args:
            thread_id: Conversation thread identifier
            question: User's question
            answer: Assistant's response
            metadata: Optional metadata about the response
        """
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
        
        self.conversation_history[thread_id].append({
            "question": question,
            "answer": answer,
            "metadata": metadata or {}
        })
        
        # Keep only last 20 turns per thread
        if len(self.conversation_history[thread_id]) > 20:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-20:]
    
    def _enhance_query_with_context(self, question: str, thread_id: str) -> str:
        """
        Enhance the current question with conversation context if needed.
        
        Args:
            question: Current user question
            thread_id: Conversation thread identifier
            
        Returns:
            Enhanced question with context if applicable
        """
        # Check if question seems like a follow-up (short, references previous context)
        follow_up_indicators = [
            len(question.split()) < 10,  # Short question
            question.lower().startswith(("yes", "no", "what about", "and ", "also", "how about")),
            "it" in question.lower() and len(question.split()) < 15,
            "that" in question.lower() and len(question.split()) < 15,
            "the visa" in question.lower() and "what" not in question.lower(),
        ]
        
        if any(follow_up_indicators) and thread_id in self.conversation_history:
            # Get last turn for context
            history = self.conversation_history[thread_id]
            if history:
                last_turn = history[-1]
                last_question = last_turn["question"]
                
                # Combine context with current question
                enhanced = f"Previous question: {last_question}\nFollow-up question: {question}"
                return enhanced
        
        return question

    def invoke(self, question: str, thread_id: str = None) -> Dict[str, Any]:
        """
        Process a user question through the agent pipeline.
        
        Args:
            question: User's immigration question
            thread_id: Optional thread ID for conversation continuity
        
        Returns:
            Dictionary with answer, sources, confidence, and metrics
        """
        if not question or not question.strip():
            return self._empty_response("Please enter a valid question.")
        
        question = question.strip()
        
        # Generate thread_id if not provided
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        # Enhance question with conversation context if needed
        enhanced_question = self._enhance_query_with_context(question, thread_id)
        
        # Prepare initial state
        initial_state: AgentState = {
            "messages": [],
            "query": enhanced_question,
            "query_type": None,
            "visa_category": None,
            "needs_clarification": False,
            "clarification_question": None,
            "decomposed_queries": [],
            "retrieved_docs": [],
            "web_results": [],
            "analysis": None,
            "key_requirements": [],
            "confidence_score": 0.0,
            "final_response": None,
            "sources_cited": []
        }
        
        # Configure thread for memory
        config = {}
        if self.enable_memory and thread_id:
            config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\n{'='*60}")
        print(f"Query: {question[:80]}...")
        print(f"{'='*60}")
        
        try:
            # Run the graph
            result = self.graph.invoke(initial_state, config)
            
            # Extract results
            confidence_score = result.get("confidence_score", 0.5)
            
            if confidence_score >= 0.8:
                confidence, emoji = "high", "🟢"
            elif confidence_score >= 0.6:
                confidence, emoji = "medium", "🟡"
            else:
                confidence, emoji = "low", "🔴"
            
            print(f"\n{'='*60}")
            print(f"Response Generated | Confidence: {emoji} {confidence.upper()}")
            print(f"{'='*60}\n")
            
            answer = result.get("final_response", "No response generated.")
            
            # Add to conversation history
            self._add_to_history(
                thread_id=thread_id,
                question=question,
                answer=answer,
                metadata={
                    "confidence": confidence,
                    "query_type": result.get("query_type"),
                    "visa_category": result.get("visa_category")
                }
            )
            
            return {
                "answer": answer,
                "sources": result.get("sources_cited", []),
                "confidence": confidence,
                "confidence_emoji": emoji,
                "confidence_score": round(confidence_score, 3),
                "query_type": result.get("query_type", "unknown"),
                "visa_category": result.get("visa_category", "unknown"),
                "key_requirements": result.get("key_requirements", []),
                "analysis": result.get("analysis", ""),
                "search_type": "agentic",
                "thread_id": thread_id
            }
            
        except Exception as e:
            print(f"\nError during processing: {e}")
            return self._empty_response(f"An error occurred: {str(e)}")
    
    def clear_history(self, thread_id: str = None):
        """
        Clear conversation history.
        
        Args:
            thread_id: Specific thread to clear, or None to clear all
        """
        if thread_id:
            if thread_id in self.conversation_history:
                del self.conversation_history[thread_id]
                print(f"Cleared history for thread: {thread_id}")
        else:
            self.conversation_history = {}
            print("Cleared all conversation history")
    
    def get_history(self, thread_id: str) -> list:
        """
        Get conversation history for a thread.
        
        Args:
            thread_id: Conversation thread identifier
            
        Returns:
            List of conversation turns
        """
        return self.conversation_history.get(thread_id, [])
    
    def _empty_response(self, message: str) -> Dict[str, Any]:
        """Return standard empty/error response structure."""
        return {
            "answer": message,
            "sources": [],
            "confidence": "low",
            "confidence_emoji": "🔴",
            "confidence_score": 0.0,
            "query_type": "error",
            "visa_category": "unknown",
            "key_requirements": [],
            "analysis": "",
            "search_type": "none",
            "thread_id": None
        }
    
    def visualize(self):
        """Print the agent graph structure."""
        visualize_graph()


# Convenience function for backward compatibility
def create_assistant(**kwargs) -> AgenticRAGAssistant:
    """Factory function to create the assistant."""
    return AgenticRAGAssistant(**kwargs)



# INTERACTIVE MODE
if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    interactive = "--interactive" in sys.argv or "-i" in sys.argv
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    
    # Initialize assistant with memory enabled
    assistant = AgenticRAGAssistant(enable_memory=True, enable_hitl=False)
    
    if test_mode:
        # Run a single test query
        print("\nRunning test query...")
        result = assistant.invoke("What is the minimum salary for a Skilled Worker visa?")
        print(f"\n{result['confidence_emoji']} Confidence: {result['confidence']}")
        print(f"\n{result['answer']}")
        sys.exit(0)
    
    # Interactive mode (default)
    print("\n" + "=" * 60)
    print("UK Immigration Assistant - Interactive Mode")
    print("=" * 60)
    print("Ask any UK immigration question.")
    print("Commands:")
    print("  • Type 'quit' or 'exit' to stop")
    print("  • Type 'clear' to clear conversation history")
    print("  • Type 'new' to start a new conversation")
    print("  • Type 'history' to view conversation history")
    print("=" * 60 + "\n")
    
    # Generate a session thread ID
    thread_id = f"session_{uuid.uuid4().hex[:8]}"
    print(f"Session ID: {thread_id}\n")
    
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        
        # Handle commands
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if question.lower() == 'clear':
            assistant.clear_history(thread_id)
            print("Conversation history cleared.\n")
            continue
        
        if question.lower() == 'new':
            thread_id = f"session_{uuid.uuid4().hex[:8]}"
            print(f"Started new conversation. Session ID: {thread_id}\n")
            continue
        
        if question.lower() == 'history':
            history = assistant.get_history(thread_id)
            if history:
                print("\nConversation History:")
                print("-" * 40)
                for i, turn in enumerate(history, 1):
                    print(f"{i}. Q: {turn['question'][:60]}...")
                    print(f"   A: {turn['answer'][:100]}...")
                    print()
            else:
                print("No conversation history yet.\n")
            continue
        
        if question.lower() == 'help':
            print("\nAvailable Commands:")
            print("quit/exit/q - Exit the assistant")
            print("clear - Clear conversation history")
            print("new - Start a new conversation")
            print("history - View conversation history")
            print("help - Show this help message\n")
            continue
        
        if not question:
            print("Please enter a question.\n")
            continue
        
        # Process the question
        result = assistant.invoke(question, thread_id=thread_id)
        
        # Display response
        print(f"\n{result['confidence_emoji']} Confidence: {result['confidence'].upper()}")
        
        if result.get('query_type') and result['query_type'] != 'error':
            print(f"📂 Type: {result['query_type']} | 🏷️ Category: {result['visa_category']}")
        
        print(f"\nAssistant:\n{result['answer']}")
        
        # Show key requirements if available
        if result.get('key_requirements') and len(result['key_requirements']) > 0:
            print(f"\nKey Requirements:")
            for req in result['key_requirements'][:5]:
                print(f"{req}")
        
        print("\n" + "-" * 60 + "\n")