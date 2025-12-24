"""Upload service for CSV data ingestion."""

import csv
import io
from datetime import datetime
from typing import BinaryIO

from diabetic_api.db.unit_of_work import UnitOfWork
from diabetic_api.models.pump_data import UploadResult


class UploadService:
    """
    Service for handling CSV file uploads.
    
    Parses Medtronic pump export CSVs and ingests them into MongoDB.
    """

    def __init__(self, uow: UnitOfWork):
        """
        Initialize upload service.
        
        Args:
            uow: Unit of Work instance
        """
        self.uow = uow

    async def process_csv(
        self,
        file_content: bytes,
        filename: str,
    ) -> UploadResult:
        """
        Process uploaded CSV file.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            Upload result with statistics
        """
        try:
            # Decode and parse CSV
            content = file_content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            
            records = []
            errors = []
            skipped = 0
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is 1)
                try:
                    record = self._parse_row(row)
                    if record:
                        records.append(record)
                    else:
                        skipped += 1
                except Exception as e:
                    errors.append(f"Row {row_num}: {str(e)}")
                    if len(errors) > 10:
                        errors.append("... (additional errors truncated)")
                        break
            
            # Insert records
            if records:
                collection = self.uow.get_collection("PumpData")
                result = await collection.insert_many(records)
                inserted = len(result.inserted_ids)
            else:
                inserted = 0
            
            return UploadResult(
                success=True,
                records_processed=len(records) + skipped,
                records_inserted=inserted,
                records_skipped=skipped,
                errors=errors,
                message=f"Successfully imported {inserted} records from {filename}",
            )
            
        except Exception as e:
            return UploadResult(
                success=False,
                errors=[str(e)],
                message=f"Failed to process {filename}: {str(e)}",
            )

    def _parse_row(self, row: dict) -> dict | None:
        """
        Parse a CSV row into a MongoDB document.
        
        Args:
            row: CSV row as dict
            
        Returns:
            Parsed document or None if row should be skipped
        """
        # Skip empty rows
        if not any(row.values()):
            return None
        
        # Parse timestamp
        date_str = row.get("Date", "")
        time_str = row.get("Time", "")
        
        if not date_str or not time_str:
            return None
        
        try:
            # Medtronic format: MM/DD/YY or YYYY/MM/DD
            timestamp = self._parse_datetime(date_str, time_str)
        except Exception:
            return None
        
        # Build document with all fields
        doc = {"Timestamp": timestamp}
        
        # Copy all original fields
        for key, value in row.items():
            if value and value.strip():
                # Try to convert numeric fields
                doc[key] = self._maybe_convert_number(value)
            else:
                doc[key] = ""
        
        return doc

    def _parse_datetime(self, date_str: str, time_str: str) -> datetime:
        """Parse Medtronic date and time strings."""
        # Try different formats
        date_formats = [
            "%m/%d/%y",  # MM/DD/YY
            "%Y/%m/%d",  # YYYY/MM/DD
            "%m/%d/%Y",  # MM/DD/YYYY
        ]
        
        date_part = None
        for fmt in date_formats:
            try:
                date_part = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        
        if date_part is None:
            raise ValueError(f"Cannot parse date: {date_str}")
        
        # Parse time (usually HH:MM:SS)
        time_formats = [
            "%H:%M:%S",
            "%H:%M",
        ]
        
        for fmt in time_formats:
            try:
                time_part = datetime.strptime(time_str, fmt)
                return date_part.replace(
                    hour=time_part.hour,
                    minute=time_part.minute,
                    second=time_part.second if hasattr(time_part, 'second') else 0,
                )
            except ValueError:
                continue
        
        # Return date only if time parse fails
        return date_part

    def _maybe_convert_number(self, value: str) -> str | float | int:
        """Try to convert string to number, return original if not possible."""
        value = value.strip()
        
        try:
            # Try integer first
            if "." not in value:
                return int(value)
        except ValueError:
            pass
        
        try:
            # Try float
            return float(value)
        except ValueError:
            pass
        
        return value

