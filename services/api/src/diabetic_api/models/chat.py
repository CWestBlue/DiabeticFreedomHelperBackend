"""Pydantic models for chat-related API schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, description="User's message")
    session_id: str | None = Field(
        None,
        description="Session ID. If not provided, a new session is created.",
    )


class ChatMessage(BaseModel):
    """Individual chat message."""

    text: str
    role: str = Field(..., pattern="^(user|assistant)$")
    timestamp: datetime
    message_id: str | None = None

    @classmethod
    def from_mongo(cls, doc: dict) -> "ChatMessage":
        """Create from MongoDB document."""
        return cls(
            text=doc.get("text", ""),
            role=doc.get("role", "user"),
            timestamp=doc.get("timestamp", datetime.utcnow()),
            message_id=doc.get("message_id"),
        )


class ChatResponse(BaseModel):
    """Response model for chat endpoint (non-streaming)."""

    message: str
    session_id: str
    message_id: str | None = None


class ChatStreamChunk(BaseModel):
    """Individual chunk in SSE stream."""

    content: str
    done: bool = False
    session_id: str | None = None
    error: str | None = None

