"""File upload API routes."""

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from diabetic_api.api.dependencies import UploadServiceDep
from diabetic_api.models.pump_data import UploadResult

router = APIRouter()


@router.post("", response_model=UploadResult)
async def upload_file(
    service: UploadServiceDep,
    file: UploadFile = File(..., description="Medtronic pump export CSV file"),
    uploadedAt: str | None = Form(
        None,
        description="ISO timestamp with timezone offset (e.g., '2025-01-15T10:30:00-05:00'). "
                    "Used to correctly convert local pump times to UTC.",
    ),
):
    """
    Upload a Medtronic pump export CSV file.
    
    Parses the CSV and imports data into the database.
    
    - **file**: CSV file from Medtronic CareLink or pump export
    - **uploadedAt**: ISO timestamp with timezone offset from client
    
    The uploadedAt timestamp is used to determine the timezone offset for
    converting local pump Date/Time values to UTC Timestamps.
    
    Returns statistics about the import including:
    - records_processed: Total rows parsed
    - records_inserted: Successfully inserted records
    - records_skipped: Rows skipped (empty or invalid)
    - errors: Any parsing errors encountered
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported",
        )
    
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read file: {str(e)}",
        )
    
    # Process the CSV
    result = await service.process_csv(content, file.filename, uploaded_at=uploadedAt)
    
    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=result.message,
        )
    
    return result

