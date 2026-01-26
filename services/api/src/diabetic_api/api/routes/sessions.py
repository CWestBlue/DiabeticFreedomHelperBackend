"""Chat sessions API routes."""

from fastapi import APIRouter, HTTPException

from diabetic_api.api.dependencies import ChatServiceDep
from diabetic_api.models.session import (
    ChatSession,
    ChatSessionDetail,
    SessionCreate,
    SessionListResponse,
)

router = APIRouter()


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    service: ChatServiceDep,
    limit: int = 50,
    skip: int = 0,
):
    """
    Get all chat sessions.
    
    Returns a paginated list of session summaries (without full message history).
    
    - **limit**: Maximum number of sessions to return (default: 50)
    - **skip**: Number of sessions to skip for pagination
    """
    sessions_data = await service.get_all_sessions(limit=limit, skip=skip)
    
    sessions = [
        ChatSession.from_mongo(s) for s in sessions_data
    ]
    
    return SessionListResponse(
        sessions=sessions,
        total=len(sessions),  # TODO: Add total count query
    )


@router.post("", response_model=ChatSession)
async def create_session(
    service: ChatServiceDep,
    request: SessionCreate | None = None,
):
    """
    Create a new chat session.
    
    - **title**: Optional title for the session
    """
    title = request.title if request else None
    session_id = await service.create_session(title=title)
    
    # Fetch the created session
    session_data = await service.get_session(session_id)
    
    if session_data is None:
        raise HTTPException(status_code=500, detail="Failed to create session")
    
    return ChatSession.from_mongo(session_data)


@router.get("/{session_id}", response_model=ChatSessionDetail)
async def get_session(
    session_id: str,
    service: ChatServiceDep,
):
    """
    Get a specific chat session with full message history.
    
    - **session_id**: The session ID
    """
    session_data = await service.get_session(session_id)
    
    if session_data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return ChatSessionDetail.from_mongo(session_data)


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    service: ChatServiceDep,
):
    """
    Delete a chat session.
    
    - **session_id**: The session ID to delete
    """
    deleted = await service.delete_session(session_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deleted", "session_id": session_id}

