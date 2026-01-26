"""Pytest configuration and fixtures."""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
from httpx import ASGITransport, AsyncClient

from diabetic_api.main import app


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """
    Create async test client.
    
    Usage:
        async def test_endpoint(client: AsyncClient):
            response = await client.get("/health")
            assert response.status_code == 200
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_glucose_data() -> list[dict]:
    """Sample glucose readings for testing."""
    from datetime import datetime, timedelta
    
    base_time = datetime.utcnow()
    return [
        {
            "Timestamp": base_time - timedelta(hours=i),
            "Sensor Glucose (mg/dL)": str(100 + i * 5),
        }
        for i in range(24)
    ]


@pytest.fixture
def sample_chat_messages() -> list[dict]:
    """Sample chat messages for testing."""
    from datetime import datetime
    
    return [
        {
            "text": "What's my average glucose?",
            "role": "user",
            "timestamp": datetime.utcnow(),
            "message_id": "msg_001",
        },
        {
            "text": "Based on your data, your average glucose is 120 mg/dL.",
            "role": "assistant",
            "timestamp": datetime.utcnow(),
            "message_id": "msg_002",
        },
    ]

