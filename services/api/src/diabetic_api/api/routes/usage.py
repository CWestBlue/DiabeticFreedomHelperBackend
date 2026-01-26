"""LLM Usage tracking API routes.

Provides endpoints to monitor API usage and check limits.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from diabetic_api.api.dependencies import UsageServiceDep

router = APIRouter()


class UsageStatsResponse(BaseModel):
    """Basic usage statistics response."""
    # Call metrics
    daily_calls: int
    daily_limit: int
    daily_remaining: int
    monthly_calls: int
    monthly_limit: int
    monthly_remaining: int
    # Token metrics
    daily_tokens: int
    daily_token_limit: int
    daily_tokens_remaining: int
    monthly_tokens: int
    monthly_token_limit: int
    monthly_tokens_remaining: int
    # Status
    is_limited: bool
    limit_reason: str | None


class DetailedUsageResponse(UsageStatsResponse):
    """Detailed usage statistics with breakdowns."""
    today: dict
    this_month: dict


@router.get("", response_model=UsageStatsResponse)
async def get_usage_stats(
    service: UsageServiceDep,
):
    """
    Get current LLM usage statistics (calls and tokens).
    
    **Call Metrics:**
    - **daily_calls**: Number of LLM calls made today
    - **daily_limit**: Maximum calls allowed per day (0 = unlimited)
    - **daily_remaining**: Calls remaining today (-1 = unlimited)
    - **monthly_calls**: Number of LLM calls made this month
    - **monthly_limit**: Maximum calls allowed per month (0 = unlimited)
    - **monthly_remaining**: Calls remaining this month (-1 = unlimited)
    
    **Token Metrics:**
    - **daily_tokens**: Total tokens (input + output) used today
    - **daily_token_limit**: Maximum tokens per day (0 = unlimited)
    - **daily_tokens_remaining**: Tokens remaining today (-1 = unlimited)
    - **monthly_tokens**: Total tokens used this month
    - **monthly_token_limit**: Maximum tokens per month (0 = unlimited)
    - **monthly_tokens_remaining**: Tokens remaining this month (-1 = unlimited)
    
    **Status:**
    - **is_limited**: True if currently at or over any limit
    - **limit_reason**: Human-readable reason if limited
    """
    stats = await service.get_stats()
    return UsageStatsResponse(**stats)


@router.get("/detailed", response_model=DetailedUsageResponse)
async def get_detailed_usage(
    service: UsageServiceDep,
):
    """
    Get detailed LLM usage statistics with breakdowns.
    
    Includes all basic stats plus:
    - **today**: Breakdown by agent, model, token counts
    - **this_month**: Monthly aggregates
    
    Useful for debugging and monitoring costs.
    """
    stats = await service.get_detailed_stats()
    return DetailedUsageResponse(**stats)


@router.get("/check")
async def check_usage_limit(
    service: UsageServiceDep,
):
    """
    Quick check if usage is within limits.
    
    Returns:
    - **ok**: True if within limits, false if limited
    - **message**: Status message
    
    Use this for quick health checks before expensive operations.
    """
    stats = await service.get_stats()
    
    if stats["is_limited"]:
        return {
            "ok": False,
            "message": stats["limit_reason"],
            "daily_remaining": stats["daily_remaining"],
            "monthly_remaining": stats["monthly_remaining"],
        }
    
    return {
        "ok": True,
        "message": "Usage within limits",
        "daily_remaining": stats["daily_remaining"],
        "monthly_remaining": stats["monthly_remaining"],
    }

