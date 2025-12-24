"""Chat service for AI-powered conversations."""

import logging
from collections.abc import AsyncGenerator

from diabetic_api.db.unit_of_work import UnitOfWork
from diabetic_api.models.chat import ChatMessage
from diabetic_api.agents.graph import StreamingChatGraph
from diabetic_api.core.config import get_settings

logger = logging.getLogger(__name__)


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

    async def create_session(self, title: str | None = None) -> str:
        """
        Create a new chat session.
        
        Args:
            title: Optional session title
            
        Returns:
            New session ID
        """
        session_id = await self.uow.sessions.create(title)
        logger.info(f"Created session: {session_id}")
        return session_id

    async def get_session(self, session_id: str) -> dict | None:
        """
        Get a session with messages.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data or None if not found
        """
        return await self.uow.sessions.get_session_with_messages(session_id)

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
