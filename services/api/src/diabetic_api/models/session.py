"""Pydantic models for chat session schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    """Request model for creating a chat session."""

    title: str | None = Field(None, max_length=100, description="Optional session title")


class ChatSession(BaseModel):
    """Chat session summary (without full message history)."""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    last_message_preview: str | None = None

    @classmethod
    def from_mongo(cls, doc: dict) -> "ChatSession":
        """Create from MongoDB aggregation result."""
        # Get stored title or generate from last message preview
        stored_title = doc.get("title")
        last_preview = doc.get("last_message_preview")
        
        # Generate title if not set or is default
        if not stored_title or stored_title == "New Chat":
            title = cls._generate_title(last_preview) if last_preview else "New Chat"
        else:
            title = stored_title
        
        return cls(
            session_id=doc.get("session_id", str(doc.get("_id", ""))),
            title=title,
            created_at=doc.get("created_at", datetime.utcnow()),
            updated_at=doc.get("updated_at", datetime.utcnow()),
            message_count=doc.get("message_count", 0),
            last_message_preview=last_preview,
        )
    
    @staticmethod
    def _generate_title(message: str) -> str:
        """Generate a title from a message."""
        if not message:
            return "New Chat"
        
        cleaned = message.strip()
        if not cleaned:
            return "New Chat"
        
        # Take first 40 chars and truncate nicely
        if len(cleaned) > 40:
            return cleaned[:40] + "..."
        return cleaned


class SessionMessage(BaseModel):
    """Message in a chat session."""
    
    role: str  # "user" or "assistant"
    text: str
    timestamp: datetime | None = None
    message_id: str | None = None


class ChatSessionDetail(ChatSession):
    """Chat session with full message history."""

    messages: list[SessionMessage] = Field(default_factory=list)

    @classmethod
    def from_mongo(cls, doc: dict) -> "ChatSessionDetail":
        """Create from MongoDB document."""
        # Get stored title or generate from messages
        stored_title = doc.get("title")
        messages = doc.get("messages", [])
        
        # Try to get first user message for title generation
        first_user_msg = next(
            (m.get("text", "") for m in messages if m.get("role") == "user"),
            None,
        )
        
        if not stored_title or stored_title == "New Chat":
            title = cls._generate_title(first_user_msg) if first_user_msg else "New Chat"
        else:
            title = stored_title
        
        # Convert messages to SessionMessage format
        session_messages = [
            SessionMessage(
                role=m.get("role", "user"),
                text=m.get("text", ""),
                timestamp=m.get("timestamp"),
                message_id=m.get("message_id"),
            )
            for m in messages
            if m.get("text")  # Skip empty messages
        ]
        
        return cls(
            session_id=doc.get("session_id", str(doc.get("_id", ""))),
            title=title,
            created_at=doc.get("created_at", datetime.utcnow()),
            updated_at=doc.get("updated_at", datetime.utcnow()),
            message_count=doc.get("message_count", len(session_messages)),
            last_message_preview=None,
            messages=session_messages,
        )


class SessionListResponse(BaseModel):
    """Response model for listing sessions."""

    sessions: list[ChatSession]
    total: int

