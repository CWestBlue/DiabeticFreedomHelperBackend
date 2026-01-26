"""Custom exception classes for the API."""

from typing import Any


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(self, message: str, status_code: int = 400, details: Any = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class NotFoundError(APIError):
    """Resource not found."""

    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            message=f"{resource} with id '{resource_id}' not found",
            status_code=404,
            details={"resource": resource, "id": resource_id},
        )


class ValidationError(APIError):
    """Validation error."""

    def __init__(self, message: str, details: Any = None):
        super().__init__(message=message, status_code=422, details=details)


class DatabaseError(APIError):
    """Database operation error."""

    def __init__(self, message: str, details: Any = None):
        super().__init__(message=message, status_code=500, details=details)


class QueryGenerationError(APIError):
    """Error generating MongoDB query from LLM."""

    def __init__(self, message: str, last_error: str | None = None):
        super().__init__(
            message=message,
            status_code=500,
            details={"last_error": last_error},
        )

