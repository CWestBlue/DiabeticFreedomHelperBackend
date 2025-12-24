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

    async def create(
        self,
        title: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """
        Create a new chat session.
        
        Args:
            title: Optional session title (auto-generated if not provided)
            session_id: Optional session ID (for client-generated IDs like UUIDs)
            
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
        
        # If session_id is provided, store it as a separate field
        # This allows client-generated UUIDs alongside MongoDB ObjectIds
        if session_id:
            session["session_id"] = session_id
            await self.collection.insert_one(session)
            return session_id
        
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
                    # Use session_id field if present, otherwise convert _id
                    "session_id": {
                        "$ifNull": ["$session_id", {"$toString": "$_id"}]
                    },
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
            session_id: Session ID (ObjectId or client UUID)
            
        Returns:
            Session document with messages, or None if not found
        """
        doc = await self._find_session(session_id)
        if doc:
            # Normalize to always have session_id field
            if "session_id" not in doc:
                doc["session_id"] = str(doc.get("_id", ""))
            doc.pop("_id", None)
        return doc

    async def _find_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Find a session by ID, checking both _id and session_id fields.
        
        Args:
            session_id: Session ID (ObjectId or client UUID)
            
        Returns:
            Session document or None
        """
        # First try to find by session_id field (client-generated UUID)
        doc = await self.collection.find_one({"session_id": session_id})
        if doc:
            return doc
        
        # Fall back to _id (ObjectId)
        try:
            doc = await self.collection.find_one({"_id": ObjectId(session_id)})
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
            session_id: Session ID (ObjectId or client UUID)
            limit: Maximum messages to return
            
        Returns:
            List of messages (most recent last)
        """
        # First try to find by session_id field (client UUID)
        doc = await self.collection.find_one(
            {"session_id": session_id},
            {"messages": {"$slice": -limit}},
        )
        if doc:
            return doc.get("messages", [])
        
        # Fall back to _id (ObjectId)
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
            session_id: Session ID (ObjectId or client UUID)
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
        
        update = {
            "$push": {"messages": message},
            "$inc": {"message_count": 1},
            "$set": {"updated_at": now},
        }
        
        # Try session_id field first (client UUID)
        result = await self.collection.update_one(
            {"session_id": session_id},
            update,
        )
        if result.modified_count > 0:
            return True
        
        # Fall back to _id (ObjectId)
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(session_id)},
                update,
            )
            return result.modified_count > 0
        except Exception:
            return False

    async def update_title(self, session_id: str, title: str) -> bool:
        """
        Update session title.
        
        Args:
            session_id: Session ID (ObjectId or client UUID)
            title: New title
            
        Returns:
            True if title was updated
        """
        # Try session_id field first (client UUID)
        result = await self.collection.update_one(
            {"session_id": session_id},
            {"$set": {"title": title}},
        )
        if result.modified_count > 0:
            return True
        
        # Fall back to _id (ObjectId)
        return await self.update_one(session_id, {"title": title})

    async def update_title_from_first_message(self, session_id: str) -> bool:
        """
        Auto-generate title from first user message.
        
        Args:
            session_id: Session ID (ObjectId or client UUID)
            
        Returns:
            True if title was updated
        """
        doc = await self._find_session(session_id)
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
            session_id: Session ID (ObjectId or client UUID)
            
        Returns:
            True if session was deleted
        """
        # Try session_id field first (client UUID)
        result = await self.collection.delete_one({"session_id": session_id})
        if result.deleted_count > 0:
            return True
        
        # Fall back to _id (ObjectId)
        return await self.delete_one(session_id)

