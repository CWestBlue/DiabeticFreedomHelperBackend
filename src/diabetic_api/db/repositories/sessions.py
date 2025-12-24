"""Repository for ChatHistory collection (N8N compatible)."""

from datetime import datetime
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from .base import BaseRepository


class SessionRepository(BaseRepository):
    """
    Repository for chat sessions and messages.
    
    Handles CRUD operations for chat sessions and their message history.
    Compatible with N8N's memoryMongoDbChat format.
    
    N8N Format:
        - sessionId (camelCase)
        - messages[].type: "human" | "ai"
        - messages[].data.content: message text
    
    New Format:
        - session_id (snake_case)
        - messages[].role: "user" | "assistant"
        - messages[].text: message text
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
        
        # Store both formats for compatibility
        if session_id:
            session["session_id"] = session_id
            session["sessionId"] = session_id  # N8N format
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
                    # Check all possible session ID fields
                    "session_id": {
                        "$ifNull": [
                            "$session_id",
                            {"$ifNull": ["$sessionId", {"$toString": "$_id"}]}
                        ]
                    },
                    "title": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "message_count": {"$size": {"$ifNull": ["$messages", []]}},
                    # Get last message preview (handle both formats)
                    "last_message_preview": {
                        "$let": {
                            "vars": {
                                "lastMsg": {"$arrayElemAt": ["$messages", -1]}
                            },
                            "in": {
                                "$ifNull": [
                                    "$$lastMsg.text",  # New format
                                    "$$lastMsg.data.content"  # N8N format
                                ]
                            }
                        }
                    },
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
                doc["session_id"] = doc.get("sessionId", str(doc.get("_id", "")))
            doc.pop("_id", None)
            
            # Normalize messages to our format
            doc["messages"] = self._normalize_messages(doc.get("messages", []))
        return doc

    async def _find_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Find a session by ID, checking multiple ID fields.
        
        Args:
            session_id: Session ID (ObjectId or client UUID)
            
        Returns:
            Session document or None
        """
        # Try session_id field first (new format)
        doc = await self.collection.find_one({"session_id": session_id})
        if doc:
            return doc
        
        # Try sessionId field (N8N format)
        doc = await self.collection.find_one({"sessionId": session_id})
        if doc:
            return doc
        
        # Fall back to _id (ObjectId)
        try:
            doc = await self.collection.find_one({"_id": ObjectId(session_id)})
            return doc
        except Exception:
            return None

    def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        """
        Normalize messages from N8N format to our format.
        
        N8N: {"type": "human", "data": {"content": "..."}}
        Ours: {"role": "user", "text": "..."}
        """
        normalized = []
        for msg in messages:
            # Already in our format
            if "role" in msg and "text" in msg:
                normalized.append(msg)
                continue
            
            # N8N format
            msg_type = msg.get("type", "")
            content = msg.get("data", {}).get("content", "")
            
            if msg_type and content:
                role = "user" if msg_type == "human" else "assistant"
                normalized.append({
                    "role": role,
                    "text": content,
                    "timestamp": msg.get("timestamp", datetime.utcnow()),
                })
        
        return normalized

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
            List of messages (most recent last), normalized to our format
        """
        doc = await self._find_session(session_id)
        if doc:
            messages = doc.get("messages", [])[-limit:]
            return self._normalize_messages(messages)
        return []

    async def add_message(
        self,
        session_id: str,
        text: str,
        role: str = "user",
    ) -> bool:
        """
        Add a message to a session.
        
        Stores in both formats for compatibility.
        
        Args:
            session_id: Session ID (ObjectId or client UUID)
            text: Message content
            role: 'user' or 'assistant'
            
        Returns:
            True if message was added
        """
        now = datetime.utcnow()
        
        # Create message in N8N-compatible format
        msg_type = "human" if role == "user" else "ai"
        message = {
            # N8N format
            "type": msg_type,
            "data": {"content": text},
            # Our format (for easier reading)
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
        
        # Try session_id field first (new format)
        result = await self.collection.update_one(
            {"session_id": session_id},
            update,
        )
        if result.modified_count > 0:
            return True
        
        # Try sessionId field (N8N format)
        result = await self.collection.update_one(
            {"sessionId": session_id},
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
        # Try session_id field first (new format)
        result = await self.collection.update_one(
            {"session_id": session_id},
            {"$set": {"title": title}},
        )
        if result.modified_count > 0:
            return True
        
        # Try sessionId field (N8N format)
        result = await self.collection.update_one(
            {"sessionId": session_id},
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
        
        messages = self._normalize_messages(doc.get("messages", []))
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
        text = first_user_msg.get("text", "")
        title = text[:50]
        if len(text) > 50:
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
        # Try session_id field first (new format)
        result = await self.collection.delete_one({"session_id": session_id})
        if result.deleted_count > 0:
            return True
        
        # Try sessionId field (N8N format)
        result = await self.collection.delete_one({"sessionId": session_id})
        if result.deleted_count > 0:
            return True
        
        # Fall back to _id (ObjectId)
        return await self.delete_one(session_id)
