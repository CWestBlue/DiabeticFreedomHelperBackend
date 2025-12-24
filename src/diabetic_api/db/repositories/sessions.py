"""Repository for ChatSessions collection."""

from datetime import datetime
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from .base import BaseRepository


class SessionRepository(BaseRepository):
    """
    Repository for chat sessions and messages.
    
    Handles CRUD operations for chat sessions and their message history.
    """

    def __init__(self, collection: AsyncIOMotorCollection):
        super().__init__(collection)

    async def create(self, title: str | None = None) -> str:
        """
        Create a new chat session.
        
        Args:
            title: Optional session title (auto-generated if not provided)
            
        Returns:
            Created session ID
        """
        now = datetime.utcnow()
        session = {
            "title": title or "New Chat",
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }
        return await self.insert_one(session)

    async def get_all_sessions(
        self,
        limit: int = 50,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Get all chat sessions (without messages).
        
        Args:
            limit: Maximum sessions to return
            skip: Number to skip for pagination
            
        Returns:
            List of session summaries
        """
        pipeline = [
            {"$sort": {"updated_at": -1}},
            {"$skip": skip},
            {"$limit": limit},
            {
                "$project": {
                    "session_id": {"$toString": "$_id"},
                    "title": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "message_count": 1,
                    "last_message_preview": {"$arrayElemAt": ["$messages.text", -1]},
                }
            },
        ]
        return await self.aggregate(pipeline, limit=limit)

    async def get_session_with_messages(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        """
        Get a session with all its messages.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session document with messages, or None if not found
        """
        try:
            doc = await self.collection.find_one({"_id": ObjectId(session_id)})
            if doc:
                doc["session_id"] = str(doc.pop("_id"))
            return doc
        except Exception:
            return None

    async def get_messages(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get messages for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum messages to return
            
        Returns:
            List of messages (most recent last)
        """
        try:
            doc = await self.collection.find_one(
                {"_id": ObjectId(session_id)},
                {"messages": {"$slice": -limit}},
            )
            return doc.get("messages", []) if doc else []
        except Exception:
            return []

    async def add_message(
        self,
        session_id: str,
        text: str,
        role: str = "user",
    ) -> bool:
        """
        Add a message to a session.
        
        Args:
            session_id: Session ID
            text: Message content
            role: 'user' or 'assistant'
            
        Returns:
            True if message was added
        """
        now = datetime.utcnow()
        message = {
            "text": text,
            "role": role,
            "timestamp": now,
            "message_id": str(ObjectId()),
        }
        
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(session_id)},
                {
                    "$push": {"messages": message},
                    "$inc": {"message_count": 1},
                    "$set": {"updated_at": now},
                },
            )
            return result.modified_count > 0
        except Exception:
            return False

    async def update_title(self, session_id: str, title: str) -> bool:
        """
        Update session title.
        
        Args:
            session_id: Session ID
            title: New title
            
        Returns:
            True if title was updated
        """
        return await self.update_one(session_id, {"title": title})

    async def update_title_from_first_message(self, session_id: str) -> bool:
        """
        Auto-generate title from first user message.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if title was updated
        """
        doc = await self.find_by_id(session_id)
        if not doc:
            return False
        
        messages = doc.get("messages", [])
        if not messages:
            return False
        
        # Get first user message
        first_user_msg = next(
            (m for m in messages if m.get("role") == "user"),
            None,
        )
        
        if not first_user_msg:
            return False
        
        # Truncate to 50 chars for title
        title = first_user_msg.get("text", "")[:50]
        if len(first_user_msg.get("text", "")) > 50:
            title += "..."
        
        return await self.update_title(session_id, title)

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if session was deleted
        """
        return await self.delete_one(session_id)

