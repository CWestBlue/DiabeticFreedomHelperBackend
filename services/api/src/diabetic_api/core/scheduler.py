"""Background task scheduler for automated sync operations."""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from diabetic_api.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Global scheduler instance
_scheduler: AsyncIOScheduler | None = None


async def run_carelink_sync() -> None:
    """
    Run CareLink sync as a scheduled task.
    
    This is called by APScheduler on the configured schedule.
    """
    from diabetic_api.db.mongo import MongoDB
    from diabetic_api.db.unit_of_work import UnitOfWork
    from diabetic_api.services.carelink import CareLinkSyncService
    
    settings = get_settings()
    
    if not settings.is_carelink_configured:
        logger.warning("Scheduled sync skipped: CareLink not configured")
        return
    
    logger.info("Starting scheduled CareLink sync...")
    
    try:
        db = MongoDB.get_database(settings.db_name)
        uow = UnitOfWork(db)
        service = CareLinkSyncService(uow)
        
        result = await service.sync()
        
        if result.success:
            logger.info(f"Scheduled sync completed: {result.message}")
        else:
            logger.error(f"Scheduled sync failed: {result.message}")
            if result.error:
                logger.error(f"Error details: {result.error}")
                
    except Exception as e:
        logger.exception(f"Scheduled sync error: {e}")


def get_day_of_week_name(day_num: int) -> str:
    """Convert day number (0=Monday) to cron day name."""
    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    return days[day_num % 7]


def start_scheduler(settings: Settings | None = None) -> AsyncIOScheduler | None:
    """
    Start the background scheduler if configured.
    
    Args:
        settings: Optional settings instance
        
    Returns:
        Scheduler instance if started, None otherwise
    """
    global _scheduler
    
    settings = settings or get_settings()
    
    if not settings.sync_schedule_enabled:
        logger.info("Scheduled sync is disabled")
        return None
    
    if not settings.is_carelink_configured:
        logger.warning("Scheduled sync enabled but CareLink not configured")
        return None
    
    _scheduler = AsyncIOScheduler()
    
    # Schedule weekly sync
    day_name = get_day_of_week_name(settings.sync_schedule_day)
    trigger = CronTrigger(
        day_of_week=day_name,
        hour=settings.sync_schedule_hour,
        minute=0,
    )
    
    _scheduler.add_job(
        run_carelink_sync,
        trigger=trigger,
        id="carelink_sync",
        name="Weekly CareLink Sync",
        replace_existing=True,
    )
    
    _scheduler.start()
    
    logger.info(
        f"Scheduler started: CareLink sync scheduled for "
        f"{day_name.upper()} at {settings.sync_schedule_hour:02d}:00"
    )
    
    return _scheduler


def stop_scheduler() -> None:
    """Stop the background scheduler if running."""
    global _scheduler
    
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
    
    _scheduler = None


def get_scheduler() -> AsyncIOScheduler | None:
    """Get the current scheduler instance."""
    return _scheduler

