"""MongoDB connection management using Motor async driver."""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


class MongoDB:
    """
    MongoDB connection manager.
    
    Uses a singleton pattern to maintain a single connection pool
    across the application lifecycle.
    """

    client: AsyncIOMotorClient | None = None
    _db_name: str = "diabetic_db"

    @classmethod
    def connect(cls, uri: str, db_name: str = "diabetic_db") -> None:
        """
        Initialize MongoDB connection.
        
        Args:
            uri: MongoDB connection URI
            db_name: Database name to use
        """
        cls.client = AsyncIOMotorClient(uri)
        cls._db_name = db_name

    @classmethod
    def close(cls) -> None:
        """Close MongoDB connection."""
        if cls.client is not None:
            cls.client.close()
            cls.client = None

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        """
        Get the MongoDB client.
        
        Raises:
            RuntimeError: If MongoDB is not connected
        """
        if cls.client is None:
            raise RuntimeError("MongoDB not connected. Call MongoDB.connect() first.")
        return cls.client

    @classmethod
    def get_database(cls, name: str | None = None) -> AsyncIOMotorDatabase:
        """
        Get a database instance.
        
        Args:
            name: Database name (uses default if not provided)
            
        Returns:
            AsyncIOMotorDatabase instance
        """
        client = cls.get_client()
        return client[name or cls._db_name]

    @classmethod
    def is_connected(cls) -> bool:
        """Check if MongoDB is connected."""
        return cls.client is not None

