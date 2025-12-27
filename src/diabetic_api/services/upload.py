"""Upload service for CSV data ingestion."""

import csv
import io
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import BinaryIO

from diabetic_api.db.unit_of_work import UnitOfWork
from diabetic_api.models.pump_data import UploadResult

logger = logging.getLogger(__name__)


class UploadService:
    """
    Service for handling CSV file uploads.
    
    Parses Medtronic pump export CSVs and ingests them into MongoDB.
    """
    
    # Header pattern for Medtronic CSV files
    HEADER_PATTERN = "Index,Date,Time"
    # Banner row pattern (separator lines like "-------...")
    BANNER_PATTERN = re.compile(r"^-{3,}")
    # Timezone offset pattern (e.g., "+05:00", "-05:00", "Z")
    OFFSET_PATTERN = re.compile(r"([+-]\d{2}:\d{2}|Z)$")
    # Fields to exclude when checking if a row is empty
    METADATA_FIELDS = frozenset([
        "Date", "Time", "Index", "_id", 
        "New Device Time", "Alert", "Event Marker"
    ])

    def __init__(self, uow: UnitOfWork):
        """
        Initialize upload service.
        
        Args:
            uow: Unit of Work instance
        """
        self.uow = uow

    def _clean_csv_content(self, content: str) -> str:
        """
        Clean Medtronic CSV by removing metadata, banners, and duplicate headers.
        
        Medtronic CareLink exports have:
        - Metadata rows before the actual data header
        - Banner rows (-------...) separating sections
        - Sometimes duplicate header rows
        
        This matches N8N's SheetCleanUp logic exactly:
        - Header row is trimmed before adding
        - Data rows keep original formatting
        
        Args:
            content: Raw CSV content as string
            
        Returns:
            Cleaned CSV content with only data rows
        """
        header_found = False
        cleaned_lines = []
        
        for line in content.splitlines():
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Look for the first header row
            if not header_found:
                if stripped.startswith(self.HEADER_PATTERN):
                    header_found = True
                    # N8N pushes the TRIMMED line for header
                    cleaned_lines.append(stripped)
                    logger.debug(f"Found header: {stripped[:80]}...")
                # Skip all metadata lines before header
                continue
            
            # Header already found - filter out unwanted rows
            # Skip banner rows (-------...)
            if self.BANNER_PATTERN.match(stripped):
                logger.debug(f"Skipping banner row: {stripped[:50]}")
                continue
            
            # Skip duplicate header rows
            if stripped.startswith(self.HEADER_PATTERN):
                logger.debug("Skipping duplicate header")
                continue
            
            # Keep normal data rows (original formatting, not trimmed)
            cleaned_lines.append(line)
        
        logger.debug(f"CSV cleanup: found header={header_found}, data_rows={len(cleaned_lines)-1 if header_found else 0}")
        return "\n".join(cleaned_lines)

    async def _get_existing_date_range(self) -> tuple[datetime | None, datetime | None]:
        """
        Get the date range of existing data in MongoDB.
        
        Returns:
            Tuple of (earliest, latest) datetimes, or (None, None) if no data exists
        """
        collection = self.uow.get_collection("PumpData")
        
        pipeline = [
            {"$match": {"Timestamp": {"$ne": None}}},
            {"$group": {
                "_id": None,
                "Earliest": {"$min": "$Timestamp"},
                "Latest": {"$max": "$Timestamp"},
            }},
            {"$project": {"_id": 0, "Earliest": 1, "Latest": 1}},
        ]
        
        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=1)
        
        if not results:
            return None, None
        
        result = results[0]
        return result.get("Earliest"), result.get("Latest")

    def _timestamp_overlaps(
        self,
        timestamp: datetime,
        mongo_start: datetime | None,
        mongo_end: datetime | None,
    ) -> bool:
        """
        Check if a single timestamp falls within the existing MongoDB date range.
        
        Uses closed interval test: timestamp >= mongoStart && timestamp <= mongoEnd
        
        This matches N8N's per-row overlap filtering in CleanDataAndGetDates.
        
        Args:
            timestamp: The timestamp to check (timezone-aware, UTC)
            mongo_start: Earliest timestamp in MongoDB (None if no data)
            mongo_end: Latest timestamp in MongoDB (None if no data)
            
        Returns:
            True if timestamp overlaps with existing data, False otherwise
        """
        if mongo_start is None or mongo_end is None:
            # No existing data, no overlap possible
            return False
        
        # Ensure MongoDB dates are timezone-aware (assume UTC if naive)
        if mongo_start.tzinfo is None:
            mongo_start = mongo_start.replace(tzinfo=timezone.utc)
        if mongo_end.tzinfo is None:
            mongo_end = mongo_end.replace(tzinfo=timezone.utc)
        
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        return timestamp >= mongo_start and timestamp <= mongo_end

    async def _cleanup_empty_rows(self) -> int:
        """
        Delete documents where all meaningful fields are empty.
        
        A row is considered "empty" if all fields except metadata fields
        (Date, Time, Index, _id, New Device Time, Alert, Event Marker)
        are null, empty string, or 0.
        
        Returns:
            Number of deleted documents
        """
        collection = self.uow.get_collection("PumpData")
        
        # Build list of metadata field names to exclude from emptiness check
        metadata_fields = list(self.METADATA_FIELDS)
        
        # This query finds documents where ALL non-metadata fields are empty
        # Uses $objectToArray to iterate over all fields and $filter to find
        # fields with meaningful values. If the filtered array is empty (size=0),
        # the document is considered empty and should be deleted.
        query = {
            "$expr": {
                "$eq": [
                    {
                        "$size": {
                            "$filter": {
                                "input": {"$objectToArray": "$$ROOT"},
                                "as": "f",
                                "cond": {
                                    "$and": [
                                        # Field is not in metadata list
                                        {"$not": {"$in": ["$$f.k", metadata_fields]}},
                                        # Field has a meaningful value (not null, "", or 0)
                                        {"$not": {"$in": ["$$f.v", [None, "", 0]]}},
                                    ]
                                },
                            }
                        }
                    },
                    0,  # If size is 0, all non-metadata fields are empty
                ]
            }
        }
        
        result = await collection.delete_many(query)
        return result.deleted_count

    def _extract_timezone_offset(self, uploaded_at: str | None) -> timezone:
        """
        Extract timezone offset from ISO timestamp string.
        
        Args:
            uploaded_at: ISO timestamp with offset (e.g., "2025-01-15T10:30:00-05:00")
            
        Returns:
            timezone object for the offset, defaults to UTC if not parseable
        """
        if not uploaded_at:
            return timezone.utc
        
        match = self.OFFSET_PATTERN.search(uploaded_at)
        if not match:
            return timezone.utc
        
        offset_str = match.group(1)
        
        if offset_str == "Z":
            return timezone.utc
        
        # Parse offset like "+05:00" or "-05:00"
        try:
            sign = 1 if offset_str[0] == "+" else -1
            hours = int(offset_str[1:3])
            minutes = int(offset_str[4:6])
            offset_delta = timedelta(hours=sign * hours, minutes=sign * minutes)
            return timezone(offset_delta)
        except (ValueError, IndexError):
            return timezone.utc

    async def process_csv(
        self,
        file_content: bytes,
        filename: str,
        uploaded_at: str | None = None,
    ) -> UploadResult:
        """
        Process uploaded CSV file.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            uploaded_at: ISO timestamp with timezone offset from client
            
        Returns:
            Upload result with statistics
        """
        try:
            # Decode and clean CSV (remove metadata, banners, duplicate headers)
            content = file_content.decode("utf-8")
            cleaned_content = self._clean_csv_content(content)
            
            if not cleaned_content.strip():
                return UploadResult(
                    success=False,
                    errors=["No valid data found in CSV after cleanup"],
                    message=f"Failed to process {filename}: No data header found",
                )
            
            reader = csv.DictReader(io.StringIO(cleaned_content))
            
            # Parse timezone from uploadedAt
            tz_offset = self._extract_timezone_offset(uploaded_at)
            
            # Get existing date range FIRST (for per-row overlap filtering)
            # This matches N8N's flow: GetUploadDates runs before parsing
            mongo_start, mongo_end = await self._get_existing_date_range()
            
            records = []
            errors = []
            skipped = 0
            overlapping = 0
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is 1)
                try:
                    record = self._parse_row(row, tz_offset)
                    if not record:
                        skipped += 1
                        continue
                    
                    # Per-row overlap check (matches N8N's CleanDataAndGetDates)
                    timestamp = record.get("Timestamp")
                    if timestamp and self._timestamp_overlaps(timestamp, mongo_start, mongo_end):
                        overlapping += 1
                        continue  # Skip this row, it overlaps with existing data
                    
                    records.append(record)
                except Exception as e:
                    errors.append(f"Row {row_num}: {str(e)}")
                    if len(errors) > 10:
                        errors.append("... (additional errors truncated)")
                        break
            
            if not records:
                if overlapping > 0:
                    # All valid rows were filtered due to overlap
                    start_str = mongo_start.strftime("%Y-%m-%d") if mongo_start else "N/A"
                    end_str = mongo_end.strftime("%Y-%m-%d") if mongo_end else "N/A"
                    return UploadResult(
                        success=False,
                        records_processed=overlapping + skipped,
                        records_skipped=overlapping + skipped,
                        errors=[f"All {overlapping} records overlap with existing data ({start_str} - {end_str})"],
                        message=f"No new data to import - all records already exist for dates {start_str} - {end_str}",
                    )
                # Include debugging info: how many rows were in cleaned CSV, how many skipped
                logger.warning(f"No valid records: skipped={skipped}, overlapping={overlapping}, errors={len(errors)}")
                all_errors = errors.copy() if errors else []
                all_errors.append(f"Stats: {skipped} rows skipped (empty/invalid), {overlapping} overlapping")
                return UploadResult(
                    success=False,
                    records_processed=skipped,
                    records_skipped=skipped,
                    errors=all_errors,
                    message=f"Failed to process {filename}: No valid data rows",
                )
            
            # Insert records
            collection = self.uow.get_collection("PumpData")
            result = await collection.insert_many(records)
            inserted = len(result.inserted_ids)
            
            # Post-insert cleanup: remove empty rows
            cleaned_up = await self._cleanup_empty_rows()
            
            # Build result message
            message = f"Successfully imported {inserted} records from {filename}"
            details = []
            if overlapping > 0:
                details.append(f"{overlapping} overlapping rows skipped")
            if cleaned_up > 0:
                details.append(f"{cleaned_up} empty rows cleaned up")
            if details:
                message += f" ({', '.join(details)})"
            
            return UploadResult(
                success=True,
                records_processed=len(records) + skipped + overlapping,
                records_inserted=inserted - cleaned_up,  # Adjust for cleaned rows
                records_skipped=skipped + overlapping + cleaned_up,
                errors=errors,
                message=message,
            )
            
        except Exception as e:
            return UploadResult(
                success=False,
                errors=[str(e)],
                message=f"Failed to process {filename}: {str(e)}",
            )

    def _parse_row(self, row: dict, tz_offset: timezone) -> dict | None:
        """
        Parse a CSV row into a MongoDB document.
        
        Args:
            row: CSV row as dict
            tz_offset: Timezone offset from client for converting to UTC
            
        Returns:
            Parsed document or None if row should be skipped
        """
        # Skip empty rows
        if not any(row.values()):
            logger.debug("Skipping empty row")
            return None
        
        # Parse timestamp
        date_str = row.get("Date", "")
        time_str = row.get("Time", "")
        
        if not date_str or not time_str:
            logger.debug(f"Skipping row: missing Date='{date_str}' or Time='{time_str}'")
            return None
        
        try:
            # Medtronic format: MM/DD/YY or YYYY/MM/DD
            # Convert to UTC using the client's timezone
            timestamp = self._parse_datetime_to_utc(date_str, time_str, tz_offset)
        except Exception as e:
            logger.debug(f"Skipping row: datetime parse error for Date='{date_str}' Time='{time_str}': {e}")
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

    def _parse_datetime_to_utc(
        self, 
        date_str: str, 
        time_str: str, 
        tz_offset: timezone,
    ) -> datetime:
        """
        Parse Medtronic date and time strings and convert to UTC.
        
        The pump stores times in local time. We use the client's timezone
        offset (from uploadedAt) to convert to UTC for consistent storage.
        
        Args:
            date_str: Date string (e.g., "01/15/25" or "2025/01/15")
            time_str: Time string (e.g., "10:30:00")
            tz_offset: Client's timezone offset
            
        Returns:
            datetime in UTC
        """
        # Try different date formats
        date_formats = [
            "%m/%d/%y",  # MM/DD/YY
            "%Y/%m/%d",  # YYYY/MM/DD
            "%m/%d/%Y",  # MM/DD/YYYY
        ]
        
        date_part = None
        for fmt in date_formats:
            try:
                date_part = datetime.strptime(date_str.strip(), fmt)
                break
            except ValueError:
                continue
        
        if date_part is None:
            raise ValueError(f"Cannot parse date: {date_str}")
        
        # Parse time (usually HH:MM:SS)
        time_str = time_str.strip() if time_str else "00:00:00"
        time_formats = [
            "%H:%M:%S",
            "%H:%M",
        ]
        
        time_part = None
        for fmt in time_formats:
            try:
                time_part = datetime.strptime(time_str, fmt)
                break
            except ValueError:
                continue
        
        if time_part is None:
            # Default to midnight if time parse fails
            time_part = datetime(1900, 1, 1, 0, 0, 0)
        
        # Combine date and time with the client's timezone
        local_dt = date_part.replace(
            hour=time_part.hour,
            minute=time_part.minute,
            second=time_part.second,
            tzinfo=tz_offset,
        )
        
        # Convert to UTC for storage
        return local_dt.astimezone(timezone.utc)

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

