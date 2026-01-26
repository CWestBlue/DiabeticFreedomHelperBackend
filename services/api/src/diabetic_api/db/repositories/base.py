"""Base repository class with common database operations."""

from typing import Any, TypeVar, Generic
from datetime import datetime

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseRepository(Generic[T]):
    """
    Base repository providing common CRUD operations.
    
    Subclasses should set the `model_class` attribute to enable
    automatic document-to-model conversion.
    """

    model_class: type[T] | None = None

    def __init__(self, collection: AsyncIOMotorCollection):
        """
        Initialize repository with a MongoDB collection.
        
        Args:
            collection: Motor collection instance
        """
        self.collection = collection

    def _to_model(self, doc: dict[str, Any]) -> T | dict[str, Any]:
        """Convert MongoDB document to Pydantic model if model_class is set."""
        if doc is None:
            return None
        if self.model_class is not None:
            # Convert ObjectId to string for id field
            if "_id" in doc:
                doc["id"] = str(doc.pop("_id"))
            return self.model_class.model_validate(doc)
        return doc

    def _to_models(self, docs: list[dict[str, Any]]) -> list[T | dict[str, Any]]:
        """Convert list of MongoDB documents to models."""
        return [self._to_model(doc) for doc in docs if doc is not None]

    async def find_by_id(self, id: str) -> T | dict[str, Any] | None:
        """
        Find document by ID.
        
        Args:
            id: Document ObjectId as string
            
        Returns:
            Document as model or dict, or None if not found
        """
        try:
            doc = await self.collection.find_one({"_id": ObjectId(id)})
            return self._to_model(doc) if doc else None
        except Exception:
            return None

    async def find_many(
        self,
        filter: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        limit: int = 100,
        skip: int = 0,
    ) -> list[T | dict[str, Any]]:
        """
        Find multiple documents matching filter.
        
        Args:
            filter: MongoDB query filter
            sort: List of (field, direction) tuples
            limit: Maximum documents to return
            skip: Number of documents to skip
            
        Returns:
            List of documents as models or dicts
        """
        cursor = self.collection.find(filter or {})
        
        if sort:
            cursor = cursor.sort(sort)
        
        cursor = cursor.skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return self._to_models(docs)

    async def find_one(self, filter: dict[str, Any]) -> T | dict[str, Any] | None:
        """
        Find single document matching filter.
        
        Args:
            filter: MongoDB query filter
            
        Returns:
            Document as model or dict, or None if not found
        """
        doc = await self.collection.find_one(filter)
        return self._to_model(doc) if doc else None

    async def insert_one(self, document: dict[str, Any]) -> str:
        """
        Insert a single document.
        
        Args:
            document: Document to insert
            
        Returns:
            Inserted document ID as string
        """
        if "created_at" not in document:
            document["created_at"] = datetime.utcnow()
        
        result = await self.collection.insert_one(document)
        return str(result.inserted_id)

    async def update_one(
        self,
        id: str,
        update: dict[str, Any],
        upsert: bool = False,
    ) -> bool:
        """
        Update a single document by ID.
        
        Args:
            id: Document ObjectId as string
            update: Update operations (will be wrapped in $set if not an operator)
            upsert: Create document if it doesn't exist
            
        Returns:
            True if document was modified
        """
        # Wrap in $set if not already an operator
        if not any(key.startswith("$") for key in update.keys()):
            update = {"$set": update}
        
        # Add updated_at timestamp
        if "$set" in update:
            update["$set"]["updated_at"] = datetime.utcnow()
        
        result = await self.collection.update_one(
            {"_id": ObjectId(id)},
            update,
            upsert=upsert,
        )
        return result.modified_count > 0 or result.upserted_id is not None

    async def delete_one(self, id: str) -> bool:
        """
        Delete a single document by ID.
        
        Args:
            id: Document ObjectId as string
            
        Returns:
            True if document was deleted
        """
        result = await self.collection.delete_one({"_id": ObjectId(id)})
        return result.deleted_count > 0

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """
        Count documents matching filter.
        
        Args:
            filter: MongoDB query filter
            
        Returns:
            Document count
        """
        return await self.collection.count_documents(filter or {})

    async def aggregate(
        self,
        pipeline: list[dict[str, Any]],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Execute aggregation pipeline.
        
        Args:
            pipeline: MongoDB aggregation pipeline
            limit: Maximum results to return
            
        Returns:
            Aggregation results
        """
        cursor = self.collection.aggregate(pipeline)
        return await cursor.to_list(length=limit)

