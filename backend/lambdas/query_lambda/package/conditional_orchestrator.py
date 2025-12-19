"""
Conditional Orchestrator - Agent with dynamic tool selection
"""

from agents.core import Agent, tool
from agents.retriever import RetrieverAgent
from typing import List, Dict
import logging

LOG = logging.getLogger(__name__)

class ConditionalOrchestrator(Agent):
    """
    Orchestrator that enables/disables tools based on user preferences
    """
    def __init__(self, user_id: str, chat_id: str, history: List[Dict], model_id: str, 
                 enabled_tools: List[str], doc_chat_only: bool = True):
        self.user_id = user_id
        self.chat_id = chat_id
        self.history = history
        self.doc_chat_only = doc_chat_only
        
        # Build tools list dynamically
        tools = []
        tool_descriptions = []
        
        if 'call_retriever' in enabled_tools:
            tools.append(self.call_retriever)
            tool_descriptions.append("- call_retriever: Search the knowledge base for specific information (people, projects, policies, etc.)")
        
        if 'search_web' in enabled_tools:
            tools.append(self.search_web)
            tool_descriptions.append("- search_web: Search the internet for real-time, current information")
        
        # Build dynamic system prompt
        if tool_descriptions:
            prompt = f"""You are an intelligent assistant with access to the following tools:

{chr(10).join(tool_descriptions)}

CRITICAL INSTRUCTIONS:
1. You are a RAG (Retrieval Augmented Generation) assistant. Your primary knowledge source is the 'call_retriever' tool.
2. If the user asks ANY specific question (e.g., about a person, project, code, policy, or specific topic), you MUST check `call_retriever` first to see if it's in the knowledge base.
3. ONLY answer from your general internal knowledge if the user says "Hi", "Hello", or asks a purely general question like "What is 2+2?" or "Tell me a joke".
4. If you are unsure, err on the side of using `call_retriever`.
5. Cite sources when using tool results.

Examples:
- "What is the refund policy?" → Use call_retriever (It's a specific policy)
- "What is the weather?" → Use search_web"""
        else:
            prompt = "You are a helpful, conversational assistant."
        
        super().__init__(
            name="ConditionalOrchestrator",
            model_id=model_id,
            instructions=prompt,
            tools=tools
        )
    
    @tool
    def call_retriever(self, query: str):
        """
        Search uploaded documents for relevant information.
        Args:
            query: The search query to find relevant documents
        """
        LOG.info(f"-> Tool Call: call_retriever('{query}')")
        try:
            agent = RetrieverAgent(self.user_id, self.chat_id, self.model_id)
            # OPTIMIZATION: Stop after search, don't let Retriever synthesize. 
            # Let Orchestrator synthesize the final answer.
            response = agent.run(query, history=[], stop_after_tool=True)
            LOG.info(f"<- Retriever success (Length: {len(response)})")
            return response
        except Exception as e:
            LOG.error(f"Retriever failed: {e}")
            return f"Document search failed: {str(e)}"
    
    @tool
    def search_web(self, query: str):
        """
        Search the internet for current, real-time information.
        Args:
            query: The search query for finding web information
        """
        LOG.info(f"-> Tool Call: search_web('{query}')")
        try:
            # Import here to avoid circular dependency
            from query_rag_bedrock import perform_web_search
            results = perform_web_search(query, num_results=3)
            return results
        except Exception as e:
            LOG.error(f"Web search failed: {e}")
            return f"Web search unavailable: {str(e)}"
