"""Chat API routes with SSE streaming."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from diabetic_api.api.dependencies import ChatServiceDep
from diabetic_api.models.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("", response_model=None)
async def chat(
    request: ChatRequest,
    service: ChatServiceDep,
):
    """
    Send a message and receive a streaming response.
    
    Uses Server-Sent Events (SSE) for real-time streaming.
    
    - **message**: The user's question or message
    - **session_id**: Optional session ID. Creates new session if not provided.
    
    Returns an SSE stream with chunks of the AI response.
    """
    # Create session if not provided
    if request.session_id is None:
        session_id = await service.create_session()
    else:
        # Check if session exists, create if it doesn't
        session = await service.get_session(request.session_id)
        if session is None:
            # Auto-create session with the provided ID
            session_id = await service.create_session(session_id=request.session_id)
        else:
            session_id = request.session_id

    async def generate():
        """Generate SSE events from chat response."""
        try:
            async for chunk in service.process_message(session_id, request.message):
                # Send content chunk
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "content": chunk,
                        "done": False,
                        "session_id": session_id,
                    }),
                }
            
            # Send completion event
            yield {
                "event": "message",
                "data": json.dumps({
                    "content": "",
                    "done": True,
                    "session_id": session_id,
                }),
            }
        except Exception as e:
            # Send error event
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "session_id": session_id,
                }),
            }

    return EventSourceResponse(generate())


@router.post("/sync", response_model=ChatResponse)
async def chat_sync(
    request: ChatRequest,
    service: ChatServiceDep,
):
    """
    Send a message and receive a complete response (non-streaming).
    
    Useful for clients that don't support SSE.
    
    - **message**: The user's question or message
    - **session_id**: Optional session ID. Creates new session if not provided.
    """
    # Create session if not provided
    if request.session_id is None:
        session_id = await service.create_session()
    else:
        # Check if session exists, create if it doesn't
        session = await service.get_session(request.session_id)
        if session is None:
            # Auto-create session with the provided ID
            session_id = await service.create_session(session_id=request.session_id)
        else:
            session_id = request.session_id

    # Collect full response
    full_response = ""
    async for chunk in service.process_message(session_id, request.message):
        full_response += chunk

    return ChatResponse(
        message=full_response,
        session_id=session_id,
    )

