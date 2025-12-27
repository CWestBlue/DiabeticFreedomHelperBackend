"""LangGraph state definitions."""

from typing import Annotated, Any, TypedDict
from operator import add

from diabetic_api.db.unit_of_work import UnitOfWork


class RouteDecision(TypedDict):
    """Router agent output."""
    
    need_mongo_query: str  # "yes" or "no"
    need_research_agent: str  # "yes" or "no"
    query_results_useful: str  # "yes", "no", or "n/a"
    data_pass_strategy: str  # "query_only", "full_data_and_query", or "n/a"
    workflow: str  # "query", "research_query", "research_full_query", "research_full", "research"
    workflow_index: int  # 0-4


class FullDataset(TypedDict):
    """Full dataset for research_full workflows."""

    sensorData: str  # CSV of sensor glucose readings (3-hour averages)
    basalData: str  # CSV of basal rate changes
    bolusData: str  # CSV of bolus/carb entries


class ChatState(TypedDict, total=False):
    """
    State that flows through the LangGraph.

    This state is passed between nodes and accumulates data
    as the graph executes.
    """

    # Input
    message: str  # User's current message
    history: list[dict]  # Previous chat messages
    session_id: str  # Chat session ID

    # Database access
    uow: UnitOfWork  # Unit of Work for database operations

    # Usage tracking (optional - for cost control)
    # Note: Using Any to avoid LangGraph runtime type resolution issues
    usage_service: Any  # UsageService | None - for tracking LLM calls

    # Router output
    route_decision: RouteDecision | None

    # Query generator output
    generated_query: str | None  # MongoDB aggregation pipeline as JSON string
    query_results: list[dict[str, Any]] | None  # Query execution results
    last_error: str | None  # Last query execution error (for retry)

    # Full data output (for research_full and research_full_query workflows)
    full_data: FullDataset | None  # Complete 90-day dataset

    # Research agent output
    research_context: str | None  # Additional context for research

    # Final output
    response: str  # Final response to stream to user
    response_chunks: Annotated[list[str], add]  # Accumulated response chunks

    # Internal tracking
    _retry_count: int  # Query retry counter


def create_initial_state(
    message: str,
    history: list[dict],
    uow: UnitOfWork,
    session_id: str = "",
    usage_service: Any = None,
) -> ChatState:
    """
    Create initial state for graph execution.

    Args:
        message: User's message
        history: Chat history
        uow: Unit of Work instance
        session_id: Chat session ID
        usage_service: Usage service for tracking LLM calls (optional)

    Returns:
        Initial ChatState
    """
    return ChatState(
        message=message,
        history=history,
        session_id=session_id,
        uow=uow,
        usage_service=usage_service,
        route_decision=None,
        generated_query=None,
        query_results=None,
        last_error=None,
        full_data=None,
        research_context=None,
        response="",
        response_chunks=[],
        _retry_count=0,
    )

