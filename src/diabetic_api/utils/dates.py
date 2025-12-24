"""Date and time utility functions."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# Chicago timezone for user-facing dates
CHICAGO_TZ = ZoneInfo("America/Chicago")
UTC_TZ = timezone.utc


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC_TZ)


def to_chicago_time(dt: datetime) -> datetime:
    """
    Convert datetime to Chicago timezone.
    
    Args:
        dt: Datetime to convert (assumed UTC if no timezone)
        
    Returns:
        Datetime in Chicago timezone
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_TZ)
    return dt.astimezone(CHICAGO_TZ)


def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC.
    
    Args:
        dt: Datetime to convert (assumed Chicago if no timezone)
        
    Returns:
        Datetime in UTC
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CHICAGO_TZ)
    return dt.astimezone(UTC_TZ)


def parse_date_range(
    time_range: str,
    end_date: datetime | None = None,
) -> tuple[datetime, datetime]:
    """
    Parse time range string into start and end datetimes.
    
    Args:
        time_range: One of 'week', 'month', '3months'
        end_date: End date (defaults to now)
        
    Returns:
        Tuple of (start_date, end_date) in UTC
    """
    if end_date is None:
        end_date = utc_now()
    elif end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=UTC_TZ)
    
    match time_range.lower():
        case "week":
            start_date = end_date - timedelta(days=7)
        case "month":
            start_date = end_date - timedelta(days=30)
        case "3months":
            start_date = end_date - timedelta(days=90)
        case _:
            # Default to week
            start_date = end_date - timedelta(days=7)
    
    return start_date, end_date


def format_duration(seconds: float) -> str:
    """
    Format seconds into human-readable duration.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 30m")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    
    minutes = int(seconds / 60)
    if minutes < 60:
        return f"{minutes}m"
    
    hours = int(minutes / 60)
    remaining_mins = minutes % 60
    
    if remaining_mins == 0:
        return f"{hours}h"
    return f"{hours}h {remaining_mins}m"

