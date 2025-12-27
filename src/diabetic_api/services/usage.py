"""LLM Usage tracking and limiting service."""

import logging
from datetime import datetime, timezone
from typing import TypedDict

from motor.motor_asyncio import AsyncIOMotorDatabase

from diabetic_api.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def extract_token_usage(response) -> tuple[int, int]:
    """
    Extract token usage from a LangChain LLM response.
    
    Supports multiple response formats from different LLM providers.
    
    Args:
        response: LangChain AIMessage or similar response object
        
    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    input_tokens = 0
    output_tokens = 0
    
    # Try usage_metadata (Gemini, newer LangChain)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        metadata = response.usage_metadata
        if isinstance(metadata, dict):
            input_tokens = metadata.get("input_tokens", 0) or metadata.get("prompt_tokens", 0)
            output_tokens = metadata.get("output_tokens", 0) or metadata.get("completion_tokens", 0)
        logger.debug(f"Token usage from usage_metadata: in={input_tokens}, out={output_tokens}")
    
    # Try response_metadata (OpenAI, some providers)
    elif hasattr(response, "response_metadata") and response.response_metadata:
        metadata = response.response_metadata
        if isinstance(metadata, dict):
            # OpenAI format
            if "token_usage" in metadata:
                usage = metadata["token_usage"]
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            # Gemini format in response_metadata
            elif "usage_metadata" in metadata:
                usage = metadata["usage_metadata"]
                input_tokens = usage.get("prompt_token_count", 0) or usage.get("input_tokens", 0)
                output_tokens = usage.get("candidates_token_count", 0) or usage.get("output_tokens", 0)
        logger.debug(f"Token usage from response_metadata: in={input_tokens}, out={output_tokens}")
    
    return input_tokens, output_tokens


class UsageStats(TypedDict):
    """Usage statistics."""
    # Call-based metrics
    daily_calls: int
    daily_limit: int
    daily_remaining: int
    monthly_calls: int
    monthly_limit: int
    monthly_remaining: int
    # Token-based metrics
    daily_tokens: int
    daily_token_limit: int
    daily_tokens_remaining: int
    monthly_tokens: int
    monthly_token_limit: int
    monthly_tokens_remaining: int
    # Status
    is_limited: bool
    limit_reason: str | None


class UsageLimitExceeded(Exception):
    """Exception raised when usage limit is exceeded."""
    
    def __init__(self, limit_type: str, current: int, limit: int, metric: str = "calls"):
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        self.metric = metric
        super().__init__(
            f"{limit_type.capitalize()} LLM usage limit exceeded: "
            f"{current:,}/{limit:,} {metric} used"
        )


class UsageService:
    """
    Service for tracking and limiting LLM API usage.
    
    Stores usage data in MongoDB and enforces configurable limits
    to prevent runaway costs from excessive API calls.
    
    Usage:
        usage_service = UsageService(db)
        
        # Check before making a call
        await usage_service.check_limit()  # Raises UsageLimitExceeded if over
        
        # Record after successful call
        await usage_service.record_call(model="gemini-2.5-flash", tokens=150)
        
        # Get current stats
        stats = await usage_service.get_stats()
    """
    
    COLLECTION_NAME = "LLMUsage"
    
    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        settings: Settings | None = None,
    ):
        """
        Initialize usage service.
        
        Args:
            db: MongoDB database instance
            settings: Application settings (uses default if not provided)
        """
        self._db = db
        self._collection = db[self.COLLECTION_NAME]
        self._settings = settings or get_settings()
    
    @property
    def daily_limit(self) -> int:
        """Get daily call limit (0 = unlimited)."""
        return self._settings.daily_llm_call_limit
    
    @property
    def monthly_limit(self) -> int:
        """Get monthly call limit (0 = unlimited)."""
        return self._settings.monthly_llm_call_limit
    
    @property
    def daily_token_limit(self) -> int:
        """Get daily token limit (0 = unlimited)."""
        return self._settings.daily_token_limit
    
    @property
    def monthly_token_limit(self) -> int:
        """Get monthly token limit (0 = unlimited)."""
        return self._settings.monthly_token_limit
    
    @property
    def tracking_enabled(self) -> bool:
        """Check if usage tracking is enabled."""
        return self._settings.usage_tracking_enabled
    
    def _get_today_key(self) -> str:
        """Get today's date key (YYYY-MM-DD)."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    def _get_month_key(self) -> str:
        """Get current month key (YYYY-MM)."""
        return datetime.now(timezone.utc).strftime("%Y-%m")
    
    async def record_call(
        self,
        model: str = "unknown",
        input_tokens: int = 0,
        output_tokens: int = 0,
        agent: str = "unknown",
    ) -> None:
        """
        Record an LLM API call.
        
        Args:
            model: Model name used
            input_tokens: Number of input tokens (if available)
            output_tokens: Number of output tokens (if available)
            agent: Name of the agent that made the call
        """
        if not self.tracking_enabled:
            return
        
        today = self._get_today_key()
        month = self._get_month_key()
        now = datetime.now(timezone.utc)
        
        # Upsert daily record
        await self._collection.update_one(
            {"date": today, "type": "daily"},
            {
                "$inc": {
                    "calls": 1,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    f"by_agent.{agent}": 1,
                    f"by_model.{model}": 1,
                },
                "$set": {
                    "last_call": now,
                    "month": month,
                },
                "$setOnInsert": {
                    "date": today,
                    "type": "daily",
                    "created_at": now,
                },
            },
            upsert=True,
        )
        
        # Also update monthly summary for faster queries
        await self._collection.update_one(
            {"month": month, "type": "monthly"},
            {
                "$inc": {
                    "calls": 1,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "$set": {"last_call": now},
                "$setOnInsert": {
                    "month": month,
                    "type": "monthly",
                    "created_at": now,
                },
            },
            upsert=True,
        )
        
        logger.debug(f"Recorded LLM call: agent={agent}, model={model}")
    
    async def _get_daily_record(self) -> dict | None:
        """Get today's usage record (cached within request)."""
        today = self._get_today_key()
        return await self._collection.find_one({"date": today, "type": "daily"})
    
    async def _get_monthly_record(self) -> dict | None:
        """Get this month's usage record (cached within request)."""
        month = self._get_month_key()
        return await self._collection.find_one({"month": month, "type": "monthly"})
    
    async def get_daily_calls(self) -> int:
        """Get number of calls made today."""
        record = await self._get_daily_record()
        return record.get("calls", 0) if record else 0
    
    async def get_monthly_calls(self) -> int:
        """Get number of calls made this month."""
        record = await self._get_monthly_record()
        return record.get("calls", 0) if record else 0
    
    async def get_daily_tokens(self) -> int:
        """Get total tokens used today (input + output)."""
        record = await self._get_daily_record()
        if not record:
            return 0
        return record.get("input_tokens", 0) + record.get("output_tokens", 0)
    
    async def get_monthly_tokens(self) -> int:
        """Get total tokens used this month (input + output)."""
        record = await self._get_monthly_record()
        if not record:
            return 0
        return record.get("input_tokens", 0) + record.get("output_tokens", 0)
    
    async def check_limit(self) -> None:
        """
        Check if usage is within limits (calls and tokens).
        
        Optimized to use at most 2 DB queries (daily + monthly records).
        
        Raises:
            UsageLimitExceeded: If any daily or monthly limit is exceeded
        """
        if not self.tracking_enabled:
            return
        
        # Check if any limits are configured
        has_daily_limits = self.daily_limit > 0 or self.daily_token_limit > 0
        has_monthly_limits = self.monthly_limit > 0 or self.monthly_token_limit > 0
        
        if not has_daily_limits and not has_monthly_limits:
            return
        
        # Fetch records in batch (max 2 queries)
        daily_record = await self._get_daily_record() if has_daily_limits else None
        monthly_record = await self._get_monthly_record() if has_monthly_limits else None
        
        # Check daily limits
        if daily_record or has_daily_limits:
            daily_calls = daily_record.get("calls", 0) if daily_record else 0
            daily_tokens = (
                (daily_record.get("input_tokens", 0) + daily_record.get("output_tokens", 0))
                if daily_record else 0
            )
            
            if self.daily_limit > 0 and daily_calls >= self.daily_limit:
                raise UsageLimitExceeded("daily", daily_calls, self.daily_limit, "calls")
            
            if self.daily_token_limit > 0 and daily_tokens >= self.daily_token_limit:
                raise UsageLimitExceeded("daily", daily_tokens, self.daily_token_limit, "tokens")
        
        # Check monthly limits
        if monthly_record or has_monthly_limits:
            monthly_calls = monthly_record.get("calls", 0) if monthly_record else 0
            monthly_tokens = (
                (monthly_record.get("input_tokens", 0) + monthly_record.get("output_tokens", 0))
                if monthly_record else 0
            )
            
            if self.monthly_limit > 0 and monthly_calls >= self.monthly_limit:
                raise UsageLimitExceeded("monthly", monthly_calls, self.monthly_limit, "calls")
            
            if self.monthly_token_limit > 0 and monthly_tokens >= self.monthly_token_limit:
                raise UsageLimitExceeded("monthly", monthly_tokens, self.monthly_token_limit, "tokens")
    
    async def get_stats(self) -> UsageStats:
        """
        Get current usage statistics.
        
        Optimized to use exactly 2 DB queries (daily + monthly records).
        
        Returns:
            UsageStats with current usage and limits (calls and tokens)
        """
        # Fetch both records in parallel-ish (2 queries total)
        daily_record = await self._get_daily_record()
        monthly_record = await self._get_monthly_record()
        
        # Extract values from records
        daily_calls = daily_record.get("calls", 0) if daily_record else 0
        monthly_calls = monthly_record.get("calls", 0) if monthly_record else 0
        daily_tokens = (
            (daily_record.get("input_tokens", 0) + daily_record.get("output_tokens", 0))
            if daily_record else 0
        )
        monthly_tokens = (
            (monthly_record.get("input_tokens", 0) + monthly_record.get("output_tokens", 0))
            if monthly_record else 0
        )
        
        # Call remaining
        daily_remaining = (
            max(0, self.daily_limit - daily_calls) 
            if self.daily_limit > 0 
            else -1  # -1 indicates unlimited
        )
        monthly_remaining = (
            max(0, self.monthly_limit - monthly_calls)
            if self.monthly_limit > 0
            else -1
        )
        
        # Token remaining
        daily_tokens_remaining = (
            max(0, self.daily_token_limit - daily_tokens)
            if self.daily_token_limit > 0
            else -1
        )
        monthly_tokens_remaining = (
            max(0, self.monthly_token_limit - monthly_tokens)
            if self.monthly_token_limit > 0
            else -1
        )
        
        # Determine if limited (check all limits)
        is_limited = False
        limit_reason = None
        
        if self.daily_limit > 0 and daily_calls >= self.daily_limit:
            is_limited = True
            limit_reason = f"Daily call limit reached ({daily_calls:,}/{self.daily_limit:,})"
        elif self.monthly_limit > 0 and monthly_calls >= self.monthly_limit:
            is_limited = True
            limit_reason = f"Monthly call limit reached ({monthly_calls:,}/{self.monthly_limit:,})"
        elif self.daily_token_limit > 0 and daily_tokens >= self.daily_token_limit:
            is_limited = True
            limit_reason = f"Daily token limit reached ({daily_tokens:,}/{self.daily_token_limit:,})"
        elif self.monthly_token_limit > 0 and monthly_tokens >= self.monthly_token_limit:
            is_limited = True
            limit_reason = f"Monthly token limit reached ({monthly_tokens:,}/{self.monthly_token_limit:,})"
        
        return UsageStats(
            daily_calls=daily_calls,
            daily_limit=self.daily_limit,
            daily_remaining=daily_remaining,
            monthly_calls=monthly_calls,
            monthly_limit=self.monthly_limit,
            monthly_remaining=monthly_remaining,
            daily_tokens=daily_tokens,
            daily_token_limit=self.daily_token_limit,
            daily_tokens_remaining=daily_tokens_remaining,
            monthly_tokens=monthly_tokens,
            monthly_token_limit=self.monthly_token_limit,
            monthly_tokens_remaining=monthly_tokens_remaining,
            is_limited=is_limited,
            limit_reason=limit_reason,
        )
    
    async def get_detailed_stats(self) -> dict:
        """
        Get detailed usage statistics including breakdown by agent/model.
        
        Returns:
            Detailed stats dictionary
        """
        today = self._get_today_key()
        month = self._get_month_key()
        
        daily_record = await self._collection.find_one(
            {"date": today, "type": "daily"}
        )
        monthly_record = await self._collection.find_one(
            {"month": month, "type": "monthly"}
        )
        
        basic_stats = await self.get_stats()
        
        return {
            **basic_stats,
            "today": {
                "date": today,
                "calls": daily_record.get("calls", 0) if daily_record else 0,
                "input_tokens": daily_record.get("input_tokens", 0) if daily_record else 0,
                "output_tokens": daily_record.get("output_tokens", 0) if daily_record else 0,
                "by_agent": daily_record.get("by_agent", {}) if daily_record else {},
                "by_model": daily_record.get("by_model", {}) if daily_record else {},
                "last_call": daily_record.get("last_call") if daily_record else None,
            },
            "this_month": {
                "month": month,
                "calls": monthly_record.get("calls", 0) if monthly_record else 0,
                "input_tokens": monthly_record.get("input_tokens", 0) if monthly_record else 0,
                "output_tokens": monthly_record.get("output_tokens", 0) if monthly_record else 0,
            },
        }

