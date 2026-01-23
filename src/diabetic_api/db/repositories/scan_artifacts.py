"""Repository for ScanArtifact collection (images/depth data with TTL)."""

from datetime import datetime, timedelta
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from diabetic_api.models.food_scan import ScanArtifact
from diabetic_api.services.gridfs_storage import GridFSStorageService

from .base import BaseRepository

# Default TTL for artifacts (7 days)
DEFAULT_ARTIFACT_TTL_DAYS = 7


class ScanArtifactRepository(BaseRepository[ScanArtifact]):
    """
    Repository for scan artifacts (images, depth maps).
    
    Stored in dedicated `scan_artifacts` collection with TTL index.
    Only populated when user opts in via `opt_in_store_artifacts=true`.
    
    TTL Index Setup:
        This collection should have a TTL index on `ttl_expires_at`:
        ```
        db.scan_artifacts.createIndex(
            {"ttl_expires_at": 1},
            {expireAfterSeconds: 0}
        )
        ```
    """

    model_class = ScanArtifact

    def __init__(self, collection: AsyncIOMotorCollection):
        super().__init__(collection)

    async def ensure_indexes(self) -> None:
        """
        Ensure required indexes exist on the collection.
        
        Should be called during application startup.
        """
        # TTL index for automatic cleanup
        await self.collection.create_index(
            "ttl_expires_at",
            expireAfterSeconds=0,
            name="ttl_cleanup_idx",
        )
        
        # Index for querying by scan_id
        await self.collection.create_index(
            "scan_id",
            name="scan_id_idx",
        )
        
        # Compound index for scan_id + artifact_type
        await self.collection.create_index(
            [("scan_id", 1), ("artifact_type", 1)],
            name="scan_artifact_type_idx",
        )

    async def store_artifact(
        self,
        scan_id: str,
        artifact_type: str,
        storage_uri: str,
        size_bytes: int,
        content_type: str,
        width: int | None = None,
        height: int | None = None,
        bit_depth: int | None = None,
        ttl_days: int = DEFAULT_ARTIFACT_TTL_DAYS,
    ) -> str:
        """
        Store a scan artifact record.
        
        Args:
            scan_id: Reference to parent scan
            artifact_type: Type of artifact (rgb, depth_u16, etc.)
            storage_uri: GridFS or blob storage URI
            size_bytes: Artifact size in bytes
            content_type: MIME type
            width: Image width (optional)
            height: Image height (optional)
            bit_depth: Bit depth for depth maps (optional)
            ttl_days: Days until automatic deletion
            
        Returns:
            Inserted document ID
        """
        now = datetime.utcnow()
        ttl_expires = now + timedelta(days=ttl_days)
        
        document = {
            "scan_id": scan_id,
            "artifact_type": artifact_type,
            "storage_uri": storage_uri,
            "size_bytes": size_bytes,
            "content_type": content_type,
            "created_at": now,
            "ttl_expires_at": ttl_expires,
        }
        
        if width is not None:
            document["width"] = width
        if height is not None:
            document["height"] = height
        if bit_depth is not None:
            document["bit_depth"] = bit_depth
        
        return await self.insert_one(document)

    async def get_artifacts_for_scan(
        self,
        scan_id: str,
    ) -> list[ScanArtifact]:
        """
        Get all artifacts for a scan.
        
        Args:
            scan_id: The scan identifier
            
        Returns:
            List of ScanArtifact objects
        """
        return await self.find_many(
            filter={"scan_id": scan_id},
            sort=[("artifact_type", 1)],
            limit=10,  # Max expected artifacts per scan
        )

    async def get_artifact_by_type(
        self,
        scan_id: str,
        artifact_type: str,
    ) -> ScanArtifact | None:
        """
        Get a specific artifact type for a scan.
        
        Args:
            scan_id: The scan identifier
            artifact_type: Type of artifact (rgb, depth_u16, etc.)
            
        Returns:
            ScanArtifact or None if not found
        """
        return await self.find_one({
            "scan_id": scan_id,
            "artifact_type": artifact_type,
        })

    async def delete_artifacts_for_scan(self, scan_id: str) -> int:
        """
        Delete all artifacts for a scan.
        
        Args:
            scan_id: The scan identifier
            
        Returns:
            Number of deleted documents
        """
        result = await self.collection.delete_many({"scan_id": scan_id})
        return result.deleted_count

    async def extend_ttl(
        self,
        scan_id: str,
        additional_days: int = DEFAULT_ARTIFACT_TTL_DAYS,
    ) -> int:
        """
        Extend TTL for all artifacts of a scan.
        
        Args:
            scan_id: The scan identifier
            additional_days: Days to add to current TTL
            
        Returns:
            Number of updated documents
        """
        new_expiry = datetime.utcnow() + timedelta(days=additional_days)
        
        result = await self.collection.update_many(
            {"scan_id": scan_id},
            {"$set": {"ttl_expires_at": new_expiry}},
        )
        return result.modified_count

    async def get_storage_stats(self, user_id: str | None = None) -> dict[str, Any]:
        """
        Get storage statistics for artifacts.
        
        Args:
            user_id: Optional user filter (requires join with food_scans)
            
        Returns:
            Dictionary with storage statistics
        """
        # For now, just get overall stats
        # User-specific stats would require a join with food_scans
        pipeline = [
            {
                "$group": {
                    "_id": "$artifact_type",
                    "count": {"$sum": 1},
                    "total_bytes": {"$sum": "$size_bytes"},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "artifact_type": "$_id",
                    "count": 1,
                    "total_bytes": 1,
                    "total_mb": {"$round": [{"$divide": ["$total_bytes", 1048576]}, 2]},
                }
            },
        ]
        
        results = await self.aggregate(pipeline, limit=10)
        
        total_bytes = sum(r.get("total_bytes", 0) for r in results)
        total_count = sum(r.get("count", 0) for r in results)
        
        return {
            "by_type": results,
            "total_artifacts": total_count,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / 1048576, 2),
        }

    async def cleanup_expired(self) -> int:
        """
        Manually cleanup expired artifacts.
        
        Note: MongoDB's TTL index handles this automatically, but this
        method can be used for immediate cleanup if needed.
        
        Returns:
            Number of deleted documents
        """
        result = await self.collection.delete_many({
            "ttl_expires_at": {"$lt": datetime.utcnow()}
        })
        return result.deleted_count

    # =========================================================================
    # GridFS-integrated methods
    # =========================================================================

    async def store_artifact_with_data(
        self,
        gridfs: GridFSStorageService,
        scan_id: str,
        artifact_type: str,
        data: bytes,
        content_type: str,
        width: int | None = None,
        height: int | None = None,
        bit_depth: int | None = None,
        ttl_days: int = DEFAULT_ARTIFACT_TTL_DAYS,
    ) -> str:
        """
        Store artifact binary data to GridFS and metadata to collection.
        
        This is the preferred method for storing artifacts as it:
        1. Uploads binary data to GridFS
        2. Stores metadata with GridFS URI reference
        
        Args:
            gridfs: GridFS storage service instance
            scan_id: Reference to parent scan
            artifact_type: Type of artifact (rgb, depth_u16, confidence_u8)
            data: Binary file data
            content_type: MIME type
            width: Image width (optional)
            height: Image height (optional)
            bit_depth: Bit depth for depth maps (optional)
            ttl_days: Days until automatic deletion
            
        Returns:
            Inserted metadata document ID
        """
        # Upload to GridFS first
        storage_uri = await gridfs.upload_file(
            data=data,
            scan_id=scan_id,
            artifact_type=artifact_type,
            content_type=content_type,
            width=width,
            height=height,
            bit_depth=bit_depth,
            ttl_days=ttl_days,
        )
        
        # Store metadata with GridFS URI reference
        return await self.store_artifact(
            scan_id=scan_id,
            artifact_type=artifact_type,
            storage_uri=storage_uri,
            size_bytes=len(data),
            content_type=content_type,
            width=width,
            height=height,
            bit_depth=bit_depth,
            ttl_days=ttl_days,
        )

    async def get_artifact_data(
        self,
        gridfs: GridFSStorageService,
        scan_id: str,
        artifact_type: str,
    ) -> tuple[bytes, ScanArtifact] | None:
        """
        Retrieve artifact binary data and metadata.
        
        Args:
            gridfs: GridFS storage service instance
            scan_id: The scan identifier
            artifact_type: Type of artifact (rgb, depth_u16, etc.)
            
        Returns:
            Tuple of (binary_data, metadata) or None if not found
        """
        # Get metadata first
        artifact = await self.get_artifact_by_type(scan_id, artifact_type)
        if not artifact:
            return None
        
        # Check if it's a GridFS URI
        if not artifact.storage_uri.startswith("gridfs://"):
            # Legacy non-GridFS artifact
            return None
        
        # Download from GridFS
        try:
            data = await gridfs.download_by_uri(artifact.storage_uri)
            return data, artifact
        except Exception:
            return None

    async def delete_artifacts_for_scan_with_gridfs(
        self,
        gridfs: GridFSStorageService,
        scan_id: str,
    ) -> tuple[int, int]:
        """
        Delete all artifacts for a scan, including GridFS files.
        
        Args:
            gridfs: GridFS storage service instance
            scan_id: The scan identifier
            
        Returns:
            Tuple of (metadata_deleted_count, gridfs_deleted_count)
        """
        # First delete GridFS files
        gridfs_deleted = await gridfs.delete_scan_artifacts(scan_id)
        
        # Then delete metadata documents
        metadata_deleted = await self.delete_artifacts_for_scan(scan_id)
        
        return metadata_deleted, gridfs_deleted
