"""LangGraph definition for the chat workflow."""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END

from diabetic_api.agents.state import ChatState, create_initial_state
from diabetic_api.agents.nodes.router import RouterAgent, route_decision_edge
from diabetic_api.agents.nodes.query_gen import QueryGenAgent, should_retry_query
from diabetic_api.agents.nodes.research import ResearchAgent, StreamingResearchAgent
from diabetic_api.agents.llm import get_llm, get_llm_info
from diabetic_api.core.config import get_settings
from diabetic_api.db.unit_of_work import UnitOfWork

logger = logging.getLogger(__name__)


# Type alias for compiled graph
ChatGraph = Any  # CompiledGraph type


def build_graph(llm: BaseChatModel | None = None) -> ChatGraph:
    """
    Build the chat workflow graph.
    
    The graph implements the following workflow based on N8N:
    
    1. **Router** - Analyzes user message, decides workflow path
       - query: MongoDB only, no interpretation
       - research_query: Query → Research (most common)
       - research_full_query: Query + full data → Research
       - research_full: Research with existing data
       - research: Research with chat history only
    
    2. **Query Generator** - Creates MongoDB aggregation pipeline
       - Supports retry on error (up to 2 attempts)
       - Uses full PumpData schema
       - Handles timezone conversion
    
    3. **Research Agent** - Interprets results, generates response
       - Supportive, educational tone
       - Markdown-formatted output
       - Includes health context
    
    ```
    ┌─────────┐
    │ Router  │
    └────┬────┘
         │
    ┌────┴────┐
    │ Decision │
    └────┬────┘
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
    ┌──────────┐     ┌──────────┐
    │ QueryGen │     │ Research │
    └────┬─────┘     └────┬─────┘
         │                │
         │ (retry?)       │
         ▼                │
    ┌──────────┐          │
    │ Research │          │
    └────┬─────┘          │
         │                │
         └────────────────┘
                │
                ▼
              [END]
    ```
    
    Args:
        llm: Language model to use (auto-configured if not provided)
        
    Returns:
        Compiled StateGraph ready for execution
    """
    settings = get_settings()
    
    # Get LLM if not provided
    if llm is None:
        if not settings.is_llm_configured:
            logger.warning("No LLM configured - returning placeholder graph")
            return PlaceholderGraph()
        
        llm = get_llm(settings)
        llm_info = get_llm_info(settings)
        logger.info(f"Building graph with {llm_info['provider']}: {llm_info['model']}")
    
    # Initialize agents
    router = RouterAgent(llm)
    query_gen = QueryGenAgent(llm, max_retries=2)
    research = ResearchAgent(llm)
    
    # Build state graph
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("router", router)
    graph.add_node("query_gen", query_gen)
    graph.add_node("research", research)
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Router → conditional routing
    graph.add_conditional_edges(
        "router",
        route_decision_edge,
        {
            "query_gen": "query_gen",
            "research": "research",
        },
    )
    
    # Query Gen → conditional (retry or continue to research)
    graph.add_conditional_edges(
        "query_gen",
        should_retry_query,
        {
            "retry": "query_gen",  # Retry with error context
            "continue": "research",  # Proceed to interpret results
        },
    )
    
    # Research → END
    graph.add_edge("research", END)
    
    # Compile and return
    compiled = graph.compile()
    logger.info("Chat graph compiled successfully")
    
    return compiled


def get_compiled_graph() -> ChatGraph:
    """
    Get a compiled graph instance with default settings.
    
    Caches the graph for reuse.
    
    Returns:
        Compiled ChatGraph ready for execution
    """
    return build_graph()


class StreamingChatGraph:
    """
    Wrapper for chat graph with streaming support.
    
    Provides async iteration over response chunks for SSE streaming.
    """
    
    def __init__(self, llm: BaseChatModel | None = None):
        """
        Initialize streaming graph wrapper.
        
        Args:
            llm: Language model (auto-configured if not provided)
        """
        settings = get_settings()
        
        if llm is None and settings.is_llm_configured:
            llm = get_llm(settings)
        
        self.llm = llm
        self.graph = build_graph(llm) if llm else None
        self.streaming_research = StreamingResearchAgent(llm) if llm else None
    
    async def astream(
        self,
        message: str,
        history: list[dict],
        uow: UnitOfWork,
        session_id: str = "",
    ):
        """
        Stream chat response.
        
        Runs the full graph (router → query_gen → research) and streams
        the final research response.
        
        Args:
            message: User's message
            history: Chat history
            uow: Unit of Work for database access
            session_id: Chat session ID
            
        Yields:
            Response text chunks
        """
        if self.graph is None or self.streaming_research is None:
            # No LLM configured - yield placeholder
            async for chunk in PlaceholderGraph().astream({
                "message": message,
                "history": history,
            }):
                yield chunk.get("content", "")
            return
        
        # Create initial state
        state = create_initial_state(
            message=message,
            history=history,
            uow=uow,
            session_id=session_id,
        )
        
        logger.info(f"Starting chat stream for: {message[:50]}...")
        
        # Run graph and stream the response
        try:
            final_state = state
            response_yielded = False
            
            async for event in self.graph.astream(state):
                # Track state updates from each node
                for node_name, node_output in event.items():
                    logger.debug(f"Node '{node_name}' completed")
                    
                    if isinstance(node_output, dict):
                        final_state = {**final_state, **node_output}
                        
                        # Log any errors for debugging
                        if node_output.get("last_error"):
                            logger.warning(f"Node '{node_name}' error: {node_output['last_error']}")
                    
                    # If we hit research node, break - we'll stream it ourselves
                    if node_name == "research":
                        # Yield the response that was generated
                        response = node_output.get("response", "")
                        if response:
                            # Yield in chunks for streaming effect
                            words = response.split(" ")
                            buffer = ""
                            for i, word in enumerate(words):
                                buffer += word
                                if i < len(words) - 1:
                                    buffer += " "
                                # Yield every few words for smooth streaming
                                if len(buffer) > 20 or i == len(words) - 1:
                                    yield buffer
                                    buffer = ""
                        return
            
        except Exception as e:
            import traceback
            logger.error(f"Graph execution error: {e}")
            logger.error(traceback.format_exc())
            
            # If we have partial state with query results, mention that
            error_context = ""
            if final_state.get("last_error"):
                error_context = f"\n\nQuery error: {final_state['last_error']}"
            
            yield f"I apologize, but I encountered an error processing your request.{error_context}\n\nPlease try rephrasing your question."


class PlaceholderGraph:
    """Placeholder graph for when LLM is not configured."""
    
    async def astream(self, state: dict):
        """
        Stream placeholder response.
        
        Args:
            state: Input state with message
            
        Yields:
            Dict with content chunks
        """
        message = state.get("message", "your question")
        
        response = f"""🚧 **AI agents are not fully configured.**

Your question: *{message}*

To enable AI-powered responses:

1. **Set up your LLM provider** in `.env`:

   For **OpenAI**:
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-your-key-here
   ```

   For **Google Gemini**:
   ```
   LLM_PROVIDER=gemini
   GOOGLE_API_KEY=your-key-here
   ```

2. **Restart the server**

The LangGraph agents (Router → QueryGen → Research) will then process your questions about your diabetic data!

---

*Once configured, you can ask things like:*
- "What's my average blood sugar this week?"
- "How many boluses did I give yesterday?"
- "Show me my carb to insulin ratio"
"""
        
        # Yield in chunks for streaming effect
        words = response.split(" ")
        buffer = ""
        for i, word in enumerate(words):
            buffer += word
            if i < len(words) - 1:
                buffer += " "
            if len(buffer) > 30 or i == len(words) - 1:
                yield {"content": buffer}
                buffer = ""
    
    async def ainvoke(self, state: dict) -> dict:
        """Non-streaming invoke."""
        chunks = []
        async for chunk in self.astream(state):
            chunks.append(chunk.get("content", ""))
        return {"response": "".join(chunks)}
