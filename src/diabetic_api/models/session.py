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
        return cls(
            session_id=doc.get("session_id", str(doc.get("_id", ""))),
            title=doc.get("title", "New Chat"),
            created_at=doc.get("created_at", datetime.utcnow()),
            updated_at=doc.get("updated_at", datetime.utcnow()),
            message_count=doc.get("message_count", 0),
            last_message_preview=doc.get("last_message_preview"),
        )


class ChatSessionDetail(ChatSession):
    """Chat session with full message history."""

    messages: list[dict] = Field(default_factory=list)

    @classmethod
    def from_mongo(cls, doc: dict) -> "ChatSessionDetail":
        """Create from MongoDB document."""
        return cls(
            session_id=doc.get("session_id", str(doc.get("_id", ""))),
            title=doc.get("title", "New Chat"),
            created_at=doc.get("created_at", datetime.utcnow()),
            updated_at=doc.get("updated_at", datetime.utcnow()),
            message_count=doc.get("message_count", 0),
            last_message_preview=None,
            messages=doc.get("messages", []),
        )


class SessionListResponse(BaseModel):
    """Response model for listing sessions."""

    sessions: list[ChatSession]
    total: int

