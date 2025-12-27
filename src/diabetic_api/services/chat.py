"""Chat service for AI-powered conversations."""

import logging
import re
from collections.abc import AsyncGenerator

from diabetic_api.db.unit_of_work import UnitOfWork
from diabetic_api.models.chat import ChatMessage
from diabetic_api.agents.graph import StreamingChatGraph
from diabetic_api.core.config import get_settings

logger = logging.getLogger(__name__)

# Regex patterns for filtering internal AI messages (matching N8N's filtering logic)
# 1. Router JSON responses like {"need_mongo_query": "yes", ...}
ROUTER_JSON_PATTERN = re.compile(r'^\s*\{.*"need_mongo_query"', re.DOTALL)

# 2. MongoDB aggregation pipelines like [{"$addFields": ...}]
MONGO_PIPELINE_PATTERN = re.compile(r'^\s*\[\s*\{\s*"\$[A-Za-z][A-Za-z0-9_]*', re.DOTALL)

# 3. Code blocks containing MongoDB pipelines
CODE_BLOCK_PIPELINE_PATTERN = re.compile(
    r'```[A-Za-z0-9_-]*\s*\[\s*\{\s*"\$[A-Za-z][A-Za-z0-9_]*',
    re.DOTALL
)


class ChatService:
    """
    Service for AI chat functionality.
    
    Orchestrates the LangGraph agents and manages chat sessions.
    Provides streaming responses for real-time display.
    """

    def __init__(self, uow: UnitOfWork, graph: StreamingChatGraph | None = None):
        """
        Initialize chat service.
        
        Args:
            uow: Unit of Work instance
            graph: Streaming chat graph (auto-created if not provided)
        """
        self.uow = uow
        self._graph = graph

    @property
    def graph(self) -> StreamingChatGraph:
        """Get or create the chat graph."""
        if self._graph is None:
            self._graph = StreamingChatGraph()
        return self._graph

    async def process_message(
        self,
        session_id: str,
        message: str,
    ) -> AsyncGenerator[str, None]:
        """
        Process a chat message and stream the response.
        
        Runs the full LangGraph workflow:
        1. Router decides workflow path
        2. Query Generator creates MongoDB pipeline (if needed)
        3. Research Agent interprets and responds
        
        Args:
            session_id: Chat session ID
            message: User's message
            
        Yields:
            Response chunks as they're generated
        """
        logger.info(f"Processing message for session {session_id}: {message[:50]}...")
        
        # Get chat history
        history = await self.uow.sessions.get_messages(session_id)
        
        # Convert to dict format for the graph
        history_dicts = [
            {
                "role": msg.get("role", "user"),
                "text": msg.get("text", ""),
                "timestamp": msg.get("timestamp"),
            }
            for msg in history
        ]
        
        # Save user message first
        await self.uow.sessions.add_message(session_id, message, role="user")
        
        # Stream response from graph
        full_response = ""
        try:
            async for chunk in self.graph.astream(
                message=message,
                history=history_dicts,
                uow=self.uow,
                session_id=session_id,
            ):
                if chunk:
                    full_response += chunk
                    yield chunk
            
            # Save assistant response
            if full_response:
                await self.uow.sessions.add_message(
                    session_id, full_response, role="assistant"
                )
                logger.info(f"Saved response ({len(full_response)} chars)")
            
            # Update session title from first message if this is a new session
            if len(history) == 0:
                await self.uow.sessions.update_title_from_first_message(session_id)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            yield error_msg
            await self.uow.sessions.add_message(session_id, error_msg, role="assistant")

    async def process_message_sync(
        self,
        session_id: str,
        message: str,
    ) -> str:
        """
        Process a chat message and return complete response (non-streaming).
        
        Args:
            session_id: Chat session ID
            message: User's message
            
        Returns:
            Complete response text
        """
        chunks = []
        async for chunk in self.process_message(session_id, message):
            chunks.append(chunk)
        return "".join(chunks)

    async def create_session(
        self,
        title: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """
        Create a new chat session.
        
        Args:
            title: Optional session title
            session_id: Optional session ID (for client-generated IDs)
            
        Returns:
            Session ID
        """
        created_id = await self.uow.sessions.create(title, session_id=session_id)
        logger.info(f"Created session: {created_id}")
        return created_id

    async def get_session(self, session_id: str) -> dict | None:
        """
        Get a session with filtered messages.
        
        Filters out internal AI messages (router decisions, MongoDB pipelines)
        and deduplicates consecutive human messages, matching N8N's chat
        history filtering behavior.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data with filtered messages, or None if not found
        """
        session = await self.uow.sessions.get_session_with_messages(session_id)
        if session and "messages" in session:
            session["messages"] = self._filter_messages(session["messages"])
        return session

    def _filter_messages(self, messages: list[dict]) -> list[dict]:
        """
        Filter out internal AI messages and deduplicate human messages.
        
        Matches N8N's chat history filtering logic:
        1. Keep all human/user messages
        2. Filter out AI messages that are:
           - Router JSON responses ({"need_mongo_query": ...})
           - MongoDB aggregation pipelines ([{"$...}])
           - Code blocks containing pipelines
        3. Deduplicate consecutive identical human messages
        
        Args:
            messages: List of normalized messages
            
        Returns:
            Filtered and deduplicated message list
        """
        filtered = []
        last_human_text = None
        
        for msg in messages:
            role = msg.get("role", "")
            text = msg.get("text", "")
            
            if role in ("user", "human"):
                # Deduplicate consecutive human messages
                if text == last_human_text:
                    continue
                last_human_text = text
                filtered.append(msg)
            elif role in ("assistant", "ai"):
                # Filter out internal AI messages
                if self._is_internal_ai_message(text):
                    continue
                filtered.append(msg)
                # Reset last_human_text when we see an AI message
                last_human_text = None
            else:
                # Keep other message types as-is
                filtered.append(msg)
        
        return filtered

    def _is_internal_ai_message(self, text: str) -> bool:
        """
        Check if an AI message is an internal message that should be hidden.
        
        Args:
            text: Message content
            
        Returns:
            True if this is an internal message that should be filtered
        """
        if not text:
            return False
        
        # Check against patterns
        if ROUTER_JSON_PATTERN.search(text):
            return True
        if MONGO_PIPELINE_PATTERN.search(text):
            return True
        if CODE_BLOCK_PIPELINE_PATTERN.search(text):
            return True
        
        return False

    async def get_all_sessions(
        self,
        limit: int = 50,
        skip: int = 0,
    ) -> list[dict]:
        """
        Get all chat sessions.
        
        Args:
            limit: Maximum sessions to return
            skip: Number to skip for pagination
            
        Returns:
            List of session summaries
        """
        return await self.uow.sessions.get_all_sessions(limit=limit, skip=skip)

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if deleted
        """
        deleted = await self.uow.sessions.delete_session(session_id)
        if deleted:
            logger.info(f"Deleted session: {session_id}")
        return deleted
