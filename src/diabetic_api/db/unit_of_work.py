"""Unit of Work pattern for managing repository access."""

from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from .repositories.pump_data import PumpDataRepository
from .repositories.sessions import SessionRepository


class UnitOfWork:
    """
    Unit of Work pattern implementation.
    
    Groups repository access and provides a single injection point for services.
    Also handles dynamic queries for LLM-generated pipelines.
    
    Usage:
        uow = UnitOfWork(db)
        readings = await uow.pump_data.get_glucose_readings(start, end)
        messages = await uow.sessions.get_messages(session_id)
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize Unit of Work with database instance.
        
        Args:
            db: Motor database instance
        """
        self._db = db
        self._pump_data: PumpDataRepository | None = None
        self._sessions: SessionRepository | None = None

    @property
    def pump_data(self) -> PumpDataRepository:
        """
        Get PumpData repository (lazy loaded).
        
        Returns:
            PumpDataRepository instance
        """
        if self._pump_data is None:
            self._pump_data = PumpDataRepository(self._db["PumpData"])
        return self._pump_data

    @property
    def sessions(self) -> SessionRepository:
        """
        Get Sessions repository (lazy loaded).
        
        Returns:
            SessionRepository instance
        """
        if self._sessions is None:
            self._sessions = SessionRepository(self._db["ChatSessions"])
        return self._sessions

    async def run_aggregation(
        self,
        collection: str,
        pipeline: list[dict[str, Any]],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Execute dynamic aggregation pipeline on any collection.
        
        Used for LLM-generated MongoDB queries that need direct execution.
        
        Args:
            collection: Collection name
            pipeline: MongoDB aggregation pipeline
            limit: Maximum results to return
            
        Returns:
            Aggregation results
        """
        cursor = self._db[collection].aggregate(pipeline)
        return await cursor.to_list(length=limit)

    async def run_find(
        self,
        collection: str,
        filter: dict[str, Any],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Execute dynamic find query on any collection.
        
        Used for LLM-generated find queries that need direct execution.
        
        Args:
            collection: Collection name
            filter: MongoDB query filter
            limit: Maximum results to return
            
        Returns:
            Query results
        """
        cursor = self._db[collection].find(filter).limit(limit)
        return await cursor.to_list(length=limit)

    def get_collection(self, name: str):
        """
        Get raw collection for advanced operations.
        
        Args:
            name: Collection name
            
        Returns:
            Motor collection instance
        """
        return self._db[name]

