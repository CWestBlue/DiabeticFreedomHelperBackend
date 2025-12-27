"""Sync API routes for CareLink data synchronization."""

from fastapi import APIRouter, HTTPException

from diabetic_api.api.dependencies import CareLinkServiceDep, SettingsDep
from diabetic_api.services.carelink import SyncResult

router = APIRouter()


@router.post("", response_model=dict)
async def trigger_sync(
    service: CareLinkServiceDep,
    settings: SettingsDep,
) -> dict:
    """
    Trigger a CareLink data sync.
    
    Downloads the latest data from Medtronic CareLink and imports it
    into the database. Uses the existing "Latest" timestamp as the
    start date to only fetch new data.
    
    Returns:
        Sync result with status and record count
        
    Raises:
        HTTPException: If CareLink is not configured
    """
    if not settings.is_carelink_configured:
        raise HTTPException(
            status_code=503,
            detail="CareLink sync not configured. Set CARELINK_USERNAME and CARELINK_PASSWORD.",
        )
    
    result: SyncResult = await service.sync()
    
    return result.to_dict()


@router.get("/status")
async def get_sync_status(settings: SettingsDep) -> dict:
    """
    Get the current sync configuration status.
    
    Returns:
        Configuration status information
    """
    return {
        "carelink_configured": settings.is_carelink_configured,
        "scheduled_sync_enabled": settings.sync_schedule_enabled,
        "schedule": {
            "day": settings.sync_schedule_day,
            "hour": settings.sync_schedule_hour,
        } if settings.sync_schedule_enabled else None,
    }

