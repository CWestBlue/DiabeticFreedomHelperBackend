"""Dashboard API routes.

These routes match the N8N webhook format expected by the Flutter app.
"""

from fastapi import APIRouter

from diabetic_api.api.dependencies import DashboardServiceDep
from diabetic_api.models.dashboard import (
    DashboardData,
    DashboardRequest,
    UploadDatesResponse,
)

router = APIRouter()


@router.post("", response_model=DashboardData)
async def get_dashboard(
    request: DashboardRequest,
    service: DashboardServiceDep,
):
    """
    Get complete dashboard data (N8N-compatible POST endpoint).

    This matches the N8N webhook format expected by the Flutter app.

    Request body:
    - **unit**: Time unit (day, week, month)
    - **unitAmount**: Number of units
    - **days**: Number of days to fetch data for
    - **requestedAt**: Reference date (optional, defaults to now)
    """
    return await service.get_dashboard_data(
        days=request.days,
        reference_date=request.requestedAt,
    )


@router.get("/dates", response_model=UploadDatesResponse)
async def get_upload_dates(
    service: DashboardServiceDep,
):
    """
    Get the date range of uploaded data.

    Returns the earliest and latest dates in the database.
    Useful for showing the user what data is available.
    """
    return await service.get_upload_dates()
