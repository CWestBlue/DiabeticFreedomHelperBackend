"""GridFS Storage Service for Food Scan Artifacts.

Provides binary storage for scan images (RGB + depth maps) using MongoDB GridFS.
Used when users opt in via opt_in_store_artifacts=true.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorGridFSBucket

logger = logging.getLogger(__name__)

# Default bucket name for scan artifacts
GRIDFS_BUCKET_NAME = "scan_artifacts_fs"

# Default TTL for GridFS files (7 days)
DEFAULT_GRIDFS_TTL_DAYS = 7


class GridFSStorageError(Exception):
    """Exception raised for GridFS storage operations."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class GridFSStorageService:
    """
    Service for storing and retrieving binary files using MongoDB GridFS.

    Used for scan artifacts (RGB images, depth maps) that exceed the 16MB
    BSON document limit or need efficient binary storage.

    Usage:
        service = GridFSStorageService(db)
        file_id = await service.upload_file(data, "scan_123", "rgb", "image/jpeg")
        data = await service.download_file(file_id)
        await service.delete_file(file_id)
    """

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        bucket_name: str = GRIDFS_BUCKET_NAME,
    ):
        """
        Initialize GridFS storage service.

        Args:
            db: Motor database instance
            bucket_name: Name of the GridFS bucket
        """
        self._db = db
        self._bucket_name = bucket_name
        self._bucket: AsyncIOMotorGridFSBucket | None = None

    @property
    def bucket(self) -> AsyncIOMotorGridFSBucket:
        """Get or create the GridFS bucket (lazy initialization)."""
        if self._bucket is None:
            self._bucket = AsyncIOMotorGridFSBucket(
                self._db,
                bucket_name=self._bucket_name,
            )
        return self._bucket

    def generate_storage_uri(self, file_id: ObjectId | str) -> str:
        """
        Generate a storage URI for a GridFS file.

        Args:
            file_id: The GridFS file ID

        Returns:
            URI in format: gridfs://bucket_name/{file_id}
        """
        return f"gridfs://{self._bucket_name}/{str(file_id)}"

    @staticmethod
    def parse_storage_uri(uri: str) -> tuple[str, str] | None:
        """
        Parse a GridFS storage URI.

        Args:
            uri: URI in format gridfs://bucket_name/{file_id}

        Returns:
            Tuple of (bucket_name, file_id) or None if invalid
        """
        if not uri.startswith("gridfs://"):
            return None
        
        parts = uri[9:].split("/", 1)
        if len(parts) != 2:
            return None
        
        return parts[0], parts[1]

    async def upload_file(
        self,
        data: bytes,
        scan_id: str,
        artifact_type: str,
        content_type: str,
        width: int | None = None,
        height: int | None = None,
        bit_depth: int | None = None,
        ttl_days: int = DEFAULT_GRIDFS_TTL_DAYS,
    ) -> str:
        """
        Upload a file to GridFS.

        Args:
            data: Binary file data
            scan_id: Associated scan ID
            artifact_type: Type of artifact (rgb, depth_u16, confidence_u8)
            content_type: MIME type of the file
            width: Image width in pixels (optional)
            height: Image height in pixels (optional)
            bit_depth: Bit depth for depth maps (optional)
            ttl_days: Days until cleanup (stored in metadata)

        Returns:
            GridFS storage URI (gridfs://bucket/{file_id})

        Raises:
            GridFSStorageError: If upload fails
        """
        try:
            # Generate filename for identification
            filename = f"{scan_id}_{artifact_type}"
            
            # Calculate TTL expiry timestamp
            ttl_expires_at = datetime.utcnow() + timedelta(days=ttl_days)
            
            # Build metadata
            metadata = {
                "scan_id": scan_id,
                "artifact_type": artifact_type,
                "content_type": content_type,
                "uploaded_at": datetime.utcnow(),
                "ttl_expires_at": ttl_expires_at,
            }
            
            if width is not None:
                metadata["width"] = width
            if height is not None:
                metadata["height"] = height
            if bit_depth is not None:
                metadata["bit_depth"] = bit_depth
            
            # Upload to GridFS
            file_id = await self.bucket.upload_from_stream(
                filename,
                data,
                metadata=metadata,
            )
            
            storage_uri = self.generate_storage_uri(file_id)
            
            logger.info(
                f"Uploaded {artifact_type} for scan {scan_id}: "
                f"{len(data)} bytes -> {storage_uri}"
            )
            
            return storage_uri
            
        except Exception as e:
            logger.error(f"GridFS upload failed: {e}")
            raise GridFSStorageError(
                message=f"Failed to upload file: {e}",
                details={
                    "scan_id": scan_id,
                    "artifact_type": artifact_type,
                    "size": len(data),
                },
            ) from e

    async def download_file(self, file_id: str | ObjectId) -> bytes:
        """
        Download a file from GridFS.

        Args:
            file_id: GridFS file ID (string or ObjectId)

        Returns:
            File data as bytes

        Raises:
            GridFSStorageError: If download fails or file not found
        """
        try:
            if isinstance(file_id, str):
                file_id = ObjectId(file_id)
            
            # Open download stream
            grid_out = await self.bucket.open_download_stream(file_id)
            
            # Read all data
            data = await grid_out.read()
            
            logger.debug(f"Downloaded GridFS file {file_id}: {len(data)} bytes")
            
            return data
            
        except Exception as e:
            logger.error(f"GridFS download failed for {file_id}: {e}")
            raise GridFSStorageError(
                message=f"Failed to download file: {e}",
                details={"file_id": str(file_id)},
            ) from e

    async def download_by_uri(self, uri: str) -> bytes:
        """
        Download a file by its storage URI.

        Args:
            uri: Storage URI (gridfs://bucket/{file_id})

        Returns:
            File data as bytes

        Raises:
            GridFSStorageError: If URI is invalid or download fails
        """
        parsed = self.parse_storage_uri(uri)
        if not parsed:
            raise GridFSStorageError(
                message=f"Invalid storage URI: {uri}",
                details={"uri": uri},
            )
        
        bucket_name, file_id = parsed
        
        if bucket_name != self._bucket_name:
            raise GridFSStorageError(
                message=f"Bucket mismatch: expected {self._bucket_name}, got {bucket_name}",
                details={"uri": uri, "expected_bucket": self._bucket_name},
            )
        
        return await self.download_file(file_id)

    async def get_file_metadata(self, file_id: str | ObjectId) -> dict[str, Any] | None:
        """
        Get metadata for a GridFS file.

        Args:
            file_id: GridFS file ID

        Returns:
            File metadata dict or None if not found
        """
        try:
            if isinstance(file_id, str):
                file_id = ObjectId(file_id)
            
            # Query the files collection directly
            files_collection = self._db[f"{self._bucket_name}.files"]
            doc = await files_collection.find_one({"_id": file_id})
            
            if doc:
                return {
                    "file_id": str(doc["_id"]),
                    "filename": doc.get("filename"),
                    "length": doc.get("length"),
                    "upload_date": doc.get("uploadDate"),
                    **doc.get("metadata", {}),
                }
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get metadata for {file_id}: {e}")
            return None

    async def delete_file(self, file_id: str | ObjectId) -> bool:
        """
        Delete a file from GridFS.

        Args:
            file_id: GridFS file ID

        Returns:
            True if deleted, False if not found
        """
        try:
            if isinstance(file_id, str):
                file_id = ObjectId(file_id)
            
            await self.bucket.delete(file_id)
            logger.info(f"Deleted GridFS file {file_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to delete GridFS file {file_id}: {e}")
            return False

    async def delete_by_uri(self, uri: str) -> bool:
        """
        Delete a file by its storage URI.

        Args:
            uri: Storage URI (gridfs://bucket/{file_id})

        Returns:
            True if deleted, False otherwise
        """
        parsed = self.parse_storage_uri(uri)
        if not parsed:
            return False
        
        _, file_id = parsed
        return await self.delete_file(file_id)

    async def delete_scan_artifacts(self, scan_id: str) -> int:
        """
        Delete all GridFS files associated with a scan.

        Args:
            scan_id: The scan identifier

        Returns:
            Number of deleted files
        """
        try:
            # Find all files for this scan
            files_collection = self._db[f"{self._bucket_name}.files"]
            cursor = files_collection.find({"metadata.scan_id": scan_id})
            
            deleted_count = 0
            async for doc in cursor:
                try:
                    await self.bucket.delete(doc["_id"])
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete file {doc['_id']}: {e}")
            
            logger.info(f"Deleted {deleted_count} GridFS files for scan {scan_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete scan artifacts for {scan_id}: {e}")
            return 0

    async def cleanup_expired(self) -> int:
        """
        Delete GridFS files that have exceeded their TTL.

        Note: MongoDB's TTL index only works on regular collections, not GridFS.
        This method should be called periodically to clean up expired files.

        Returns:
            Number of deleted files
        """
        try:
            files_collection = self._db[f"{self._bucket_name}.files"]
            now = datetime.utcnow()
            
            # Find expired files
            cursor = files_collection.find({
                "metadata.ttl_expires_at": {"$lt": now}
            })
            
            deleted_count = 0
            async for doc in cursor:
                try:
                    await self.bucket.delete(doc["_id"])
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete expired file {doc['_id']}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired GridFS files")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"GridFS cleanup failed: {e}")
            return 0

    async def get_storage_stats(self) -> dict[str, Any]:
        """
        Get storage statistics for the GridFS bucket.

        Returns:
            Dictionary with storage statistics
        """
        try:
            files_collection = self._db[f"{self._bucket_name}.files"]
            
            # Aggregate stats by artifact type
            pipeline = [
                {
                    "$group": {
                        "_id": "$metadata.artifact_type",
                        "count": {"$sum": 1},
                        "total_bytes": {"$sum": "$length"},
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "artifact_type": "$_id",
                        "count": 1,
                        "total_bytes": 1,
                        "total_mb": {
                            "$round": [{"$divide": ["$total_bytes", 1048576]}, 2]
                        },
                    }
                },
            ]
            
            results = await files_collection.aggregate(pipeline).to_list(length=10)
            
            total_bytes = sum(r.get("total_bytes", 0) for r in results)
            total_count = sum(r.get("count", 0) for r in results)
            
            return {
                "bucket_name": self._bucket_name,
                "by_type": results,
                "total_files": total_count,
                "total_bytes": total_bytes,
                "total_mb": round(total_bytes / 1048576, 2),
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {
                "bucket_name": self._bucket_name,
                "error": str(e),
            }

    async def ensure_indexes(self) -> None:
        """
        Ensure required indexes exist on GridFS collections.

        Should be called during application startup.
        """
        try:
            files_collection = self._db[f"{self._bucket_name}.files"]
            
            # Index for querying by scan_id
            await files_collection.create_index(
                "metadata.scan_id",
                name="scan_id_idx",
            )
            
            # Index for TTL cleanup queries
            await files_collection.create_index(
                "metadata.ttl_expires_at",
                name="ttl_expires_idx",
            )
            
            # Index for querying by artifact type
            await files_collection.create_index(
                [("metadata.scan_id", 1), ("metadata.artifact_type", 1)],
                name="scan_artifact_type_idx",
            )
            
            logger.info(f"GridFS indexes ensured for bucket {self._bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to create GridFS indexes: {e}")
