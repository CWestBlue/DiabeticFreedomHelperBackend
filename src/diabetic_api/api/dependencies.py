"""FastAPI dependency injection factories."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from diabetic_api.core.config import Settings, get_settings
from diabetic_api.db.mongo import MongoDB
from diabetic_api.db.unit_of_work import UnitOfWork
from diabetic_api.services.chat import ChatService
from diabetic_api.services.dashboard import DashboardService
from diabetic_api.services.upload import UploadService


# Type aliases for cleaner route signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_database() -> AsyncIOMotorDatabase:
    """
    Get the MongoDB database instance.
    
    Returns:
        Motor database instance
    """
    settings = get_settings()
    return MongoDB.get_database(settings.db_name)


def get_uow(db: AsyncIOMotorDatabase = Depends(get_database)) -> UnitOfWork:
    """
    Get Unit of Work instance.
    
    Args:
        db: Injected database instance
        
    Returns:
        UnitOfWork instance
    """
    return UnitOfWork(db)


# Type alias for UoW dependency
UoWDep = Annotated[UnitOfWork, Depends(get_uow)]


def get_chat_service(uow: UnitOfWork = Depends(get_uow)) -> ChatService:
    """
    Get ChatService instance.
    
    Args:
        uow: Injected Unit of Work
        
    Returns:
        ChatService instance with streaming graph
    """
    from diabetic_api.agents.graph import StreamingChatGraph
    
    graph = StreamingChatGraph()
    return ChatService(uow, graph=graph)


def get_dashboard_service(uow: UnitOfWork = Depends(get_uow)) -> DashboardService:
    """
    Get DashboardService instance.
    
    Args:
        uow: Injected Unit of Work
        
    Returns:
        DashboardService instance
    """
    return DashboardService(uow)


def get_upload_service(uow: UnitOfWork = Depends(get_uow)) -> UploadService:
    """
    Get UploadService instance.
    
    Args:
        uow: Injected Unit of Work
        
    Returns:
        UploadService instance
    """
    return UploadService(uow)


# Type aliases for service dependencies
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
DashboardServiceDep = Annotated[DashboardService, Depends(get_dashboard_service)]
UploadServiceDep = Annotated[UploadService, Depends(get_upload_service)]

