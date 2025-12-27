"""LangGraph definition for the chat workflow."""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END

from diabetic_api.agents.state import ChatState, create_initial_state
from diabetic_api.agents.nodes.router import RouterAgent
from diabetic_api.agents.nodes.query_gen import QueryGenAgent, should_retry_query
from diabetic_api.agents.nodes.research import ResearchAgent, StreamingResearchAgent
from diabetic_api.agents.nodes.full_data import FullDataNode
from diabetic_api.agents.llm import get_llm, get_llm_info
from diabetic_api.core.config import get_settings
from diabetic_api.db.unit_of_work import UnitOfWork

logger = logging.getLogger(__name__)


def route_after_router(state: ChatState) -> str:
    """
    Determine next node after router based on workflow decision.

    Workflow routing:
    - query (0): query_gen → research (no full data needed)
    - research_query (1): query_gen → research
    - research_full_query (2): full_data → query_gen → research
    - research_full (3): full_data → research
    - research (4): research only

    Args:
        state: Current graph state

    Returns:
        Next node name: "query_gen", "full_data", or "research"
    """
    decision = state.get("route_decision")

    if decision is None:
        logger.warning("No route decision found, defaulting to research")
        return "research"

    workflow = decision.get("workflow", "research")
    need_query = decision.get("need_mongo_query", "no")
    data_strategy = decision.get("data_pass_strategy", "n/a")

    logger.info(f"Routing: workflow={workflow}, need_query={need_query}, data_strategy={data_strategy}")

    # Workflows that need full data first
    if workflow in ("research_full", "research_full_query"):
        return "full_data"

    # Workflows that need query (but not full data)
    if need_query == "yes" or workflow in ("query", "research_query"):
        return "query_gen"

    # Default to research only
    return "research"


def route_after_full_data(state: ChatState) -> str:
    """
    Determine next node after full_data fetch.

    Args:
        state: Current graph state

    Returns:
        "query_gen" if also needs query, else "research"
    """
    decision = state.get("route_decision")

    if decision is None:
        return "research"

    workflow = decision.get("workflow", "research")

    # research_full_query needs both full data AND query
    if workflow == "research_full_query":
        return "query_gen"

    # research_full goes directly to research
    return "research"


# Type alias for compiled graph
ChatGraph = Any  # CompiledGraph type


def build_graph(llm: BaseChatModel | None = None) -> ChatGraph:
    """
    Build the chat workflow graph.

    The graph implements the following workflow based on N8N:

    1. **Router** - Analyzes user message, decides workflow path
       - query (0): MongoDB only, no interpretation
       - research_query (1): Query → Research (most common)
       - research_full_query (2): Full Data + Query → Research
       - research_full (3): Full Data → Research
       - research (4): Research with chat history only

    2. **Full Data** - Fetches 90-day dataset (sensor, basal, bolus)
       - Used for research_full and research_full_query paths

    3. **Query Generator** - Creates MongoDB aggregation pipeline
       - Supports retry on error (up to 2 attempts)
       - Uses full PumpData schema
       - Handles timezone conversion

    4. **Research Agent** - Interprets results, generates response
       - Supportive, educational tone
       - Markdown-formatted output
       - Includes health context

    ```
    ┌─────────┐
    │ Router  │
    └────┬────┘
         │
    ┌────┴─────────────────────────────┐
    │           Decision               │
    └────┬─────────┬───────────────────┘
         │         │         │
    ┌────┴────┐    │    ┌────┴────┐
    │FullData │    │    │QueryGen │
    └────┬────┘    │    └────┬────┘
         │         │         │
    ┌────┴────┐    │         │ (retry?)
    │         │    │         │
    │QueryGen │    ▼         ▼
    │ (opt)   │ ┌──────────────┐
    └────┬────┘ │   Research   │
         │      └──────────────┘
         ▼             │
    ┌──────────┐       │
    │ Research │       │
    └────┬─────┘       │
         │             │
         └─────────────┘
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

    # Initialize agents/nodes
    router = RouterAgent(llm)
    full_data = FullDataNode(days=90)
    query_gen = QueryGenAgent(llm, max_retries=2)
    research = ResearchAgent(llm)

    # Build state graph
    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("router", router)
    graph.add_node("full_data", full_data)
    graph.add_node("query_gen", query_gen)
    graph.add_node("research", research)

    # Set entry point
    graph.set_entry_point("router")

    # Router → conditional routing (3 possible paths)
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "full_data": "full_data",  # research_full or research_full_query
            "query_gen": "query_gen",  # query or research_query
            "research": "research",  # research only
        },
    )

    # Full Data → conditional (may need query or go direct to research)
    graph.add_conditional_edges(
        "full_data",
        route_after_full_data,
        {
            "query_gen": "query_gen",  # research_full_query
            "research": "research",  # research_full
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

    # Compile with recursion limit as additional safeguard against infinite loops
    # This limits total node transitions (not just query_gen retries)
    # Max expected path: router(1) + full_data(1) + query_gen(3 with retries) + research(1) = 6
    # Set to 10 for safety margin
    compiled = graph.compile()

    logger.info("Chat graph compiled successfully with loop protection")

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
        # IMPORTANT: recursion_limit prevents infinite loops (billing protection)
        # Max expected: router(1) + full_data(1) + query_gen(3 with retries) + research(1) = 6
        # Set to 15 for safety margin
        config = {"recursion_limit": 15}

        try:
            final_state = state
            response_yielded = False

            async for event in self.graph.astream(state, config=config):
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
