"""Unit of Work pattern for managing repository access."""

from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from .repositories.food_scans import FoodScanRepository
from .repositories.meal_estimates import MealEstimateRepository
from .repositories.pump_data import PumpDataRepository
from .repositories.scan_artifacts import ScanArtifactRepository
from .repositories.sessions import SessionRepository
from diabetic_api.services.gridfs_storage import GridFSStorageService


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
        self._meal_estimates: MealEstimateRepository | None = None
        self._food_scans: FoodScanRepository | None = None
        self._scan_artifacts: ScanArtifactRepository | None = None
        self._gridfs: GridFSStorageService | None = None

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
        
        Uses 'ChatHistory' collection to maintain compatibility with existing N8N data.
        
        Returns:
            SessionRepository instance
        """
        if self._sessions is None:
            self._sessions = SessionRepository(self._db["ChatHistory"])
        return self._sessions

    @property
    def meal_estimates(self) -> MealEstimateRepository:
        """
        Get MealEstimate repository (lazy loaded).
        
        Uses 'meal_estimates' collection (separate from pump_data).
        
        Returns:
            MealEstimateRepository instance
        """
        if self._meal_estimates is None:
            self._meal_estimates = MealEstimateRepository(self._db["meal_estimates"])
        return self._meal_estimates

    @property
    def food_scans(self) -> FoodScanRepository:
        """
        Get FoodScan repository (lazy loaded).
        
        Uses 'food_scans' collection for scan metadata and results.
        
        Returns:
            FoodScanRepository instance
        """
        if self._food_scans is None:
            self._food_scans = FoodScanRepository(self._db["food_scans"])
        return self._food_scans

    @property
    def scan_artifacts(self) -> ScanArtifactRepository:
        """
        Get ScanArtifact repository (lazy loaded).
        
        Uses 'scan_artifacts' collection with TTL for images/depth data.
        Only populated when user opts in.
        
        Returns:
            ScanArtifactRepository instance
        """
        if self._scan_artifacts is None:
            self._scan_artifacts = ScanArtifactRepository(self._db["scan_artifacts"])
        return self._scan_artifacts

    @property
    def gridfs(self) -> GridFSStorageService:
        """
        Get GridFS storage service (lazy loaded).
        
        Used for storing binary scan artifacts (RGB images, depth maps).
        Provides efficient storage for files exceeding BSON limits.
        
        Returns:
            GridFSStorageService instance
        """
        if self._gridfs is None:
            self._gridfs = GridFSStorageService(self._db)
        return self._gridfs

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

