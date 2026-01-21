"""CareLink API client for token-based data retrieval.

This module implements a direct REST API client for Medtronic CareLink,
based on the carelink-python library approach. It uses a pre-authenticated
token to bypass the login flow (and reCAPTCHA).

Usage:
    1. Log into CareLink manually in your browser
    2. Open DevTools > Application > Cookies
    3. Copy the 'auth_tmp_token' cookie value
    4. Set CARELINK_TOKEN in your .env file
    5. The API will auto-refresh the token as needed
"""

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, UTC
from typing import Any

import requests

logger = logging.getLogger(__name__)


# Constants
CARELINK_SERVER_EU = "carelink.minimed.eu"
CARELINK_SERVER_US = "carelink.minimed.com"
AUTH_TOKEN_COOKIE = "auth_tmp_token"
TOKEN_VALID_TO_COOKIE = "c_token_valid_to"
AUTH_EXPIRE_DEADLINE_MINUTES = 10


def parse_carelink_timestamp(timestamp_str: str | None) -> datetime | None:
    """
    Parse CareLink API timestamp to datetime object in UTC.
    
    CareLink uses various formats:
    - ISO 8601: "2026-01-19T11:45:00"
    - With timezone: "2026-01-19T11:45:00-05:00"
    - With Z suffix: "2026-01-19T11:45:00Z"
    - Sometimes: "Jan 19, 2026 11:45:00 AM"
    
    Args:
        timestamp_str: Timestamp string from API
        
    Returns:
        datetime object in UTC, or None if parsing fails
    """
    if not timestamp_str:
        return None
    
    # Common ISO 8601 formats
    iso_formats = [
        "%Y-%m-%dT%H:%M:%S",          # 2026-01-19T11:45:00
        "%Y-%m-%dT%H:%M:%SZ",         # 2026-01-19T11:45:00Z
        "%Y-%m-%dT%H:%M:%S.%f",       # 2026-01-19T11:45:00.000
        "%Y-%m-%dT%H:%M:%S.%fZ",      # 2026-01-19T11:45:00.000Z
    ]
    
    # Try Python's fromisoformat first (handles timezone offsets)
    try:
        # Handle Z suffix
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(timestamp_str)
        # Convert to UTC if timezone-aware
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc)
        # Assume UTC if naive
        return dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        pass
    
    # Try other formats
    for fmt in iso_formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    
    # Try verbose format (sometimes used in display fields)
    verbose_formats = [
        "%b %d, %Y %I:%M:%S %p",      # Jan 19, 2026 11:45:00 AM
        "%b %d, %Y %H:%M:%S",         # Jan 19, 2026 11:45:00
    ]
    for fmt in verbose_formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse timestamp: {timestamp_str}")
    return None


@dataclass
class CareLinkData:
    """Container for CareLink pump data."""
    
    raw_data: dict[str, Any]
    retrieved_at: datetime
    
    @property
    def last_sg(self) -> dict | None:
        """Get the last sensor glucose reading."""
        return self.raw_data.get("lastSG")
    
    @property
    def last_sg_value(self) -> int | None:
        """Get the last sensor glucose value in mg/dL."""
        sg = self.last_sg
        return sg.get("sg") if sg else None
    
    @property
    def active_insulin(self) -> dict | None:
        """Get active insulin on board."""
        return self.raw_data.get("activeInsulin")
    
    @property
    def basal_rate(self) -> float | None:
        """Get current basal rate."""
        return self.raw_data.get("currentServerTime")
    
    @property
    def sensor_glucose_readings(self) -> list[dict]:
        """Get list of sensor glucose readings."""
        return self.raw_data.get("sgs", [])
    
    @property
    def markers(self) -> list[dict]:
        """Get list of markers (boluses, meals, etc.)."""
        return self.raw_data.get("markers", [])
    
    def to_pump_records(self) -> list[dict]:
        """
        Convert CareLink data to pump record format for database storage.
        
        Maps the JSON data to the CSV-like format used by the existing
        upload service for consistency. Timestamps are converted to
        proper datetime objects in UTC for correct storage and deduplication.
        """
        records = []
        
        # Convert sensor glucose readings
        for sg in self.sensor_glucose_readings:
            sg_value = sg.get("sg")
            # Try different timestamp field names (API varies)
            timestamp_str = sg.get("datetime") or sg.get("deviceTime") or sg.get("sensorTime")
            
            if sg_value is not None and timestamp_str:
                timestamp = parse_carelink_timestamp(timestamp_str)
                if timestamp:
                    record = {
                        "Timestamp": timestamp,
                        "Sensor Glucose (mg/dL)": sg_value,
                        "ISIG Value": sg.get("isig"),
                        "Event Marker": None,
                        "Source": "carelink_api",
                    }
                    # Add trend if available
                    if sg.get("trend"):
                        record["Sensor Glucose Trend"] = sg.get("trend")
                    records.append(record)
        
        # Convert markers (boluses, meals, calibrations, etc.)
        for marker in self.markers:
            marker_type = marker.get("type")
            # Try different timestamp field names
            timestamp_str = (
                marker.get("dateTime") or 
                marker.get("datetime") or 
                marker.get("deviceTime")
            )
            
            if not timestamp_str:
                continue
            
            timestamp = parse_carelink_timestamp(timestamp_str)
            if not timestamp:
                continue
            
            record = {
                "Timestamp": timestamp,
                "Source": "carelink_api",
            }
            
            # Only add fields if they have actual values (not None, not empty)
            if marker_type == "INSULIN":
                # Bolus delivery - only add if we have a delivered amount
                delivered = marker.get("deliveredAmount")
                if delivered is not None:
                    record["Bolus Volume Delivered (U)"] = delivered
                    if marker.get("bolusType"):
                        record["Bolus Type"] = marker.get("bolusType")
                    if marker.get("programmedAmount"):
                        record["Bolus Volume Programmed (U)"] = marker.get("programmedAmount")
            elif marker_type == "MEAL":
                # Meal/carb entry - only add if we have carbs
                carbs = marker.get("amount")
                if carbs is not None:
                    record["BWZ Carb Input (grams)"] = carbs
            elif marker_type == "CALIBRATION":
                # Calibration - only add if we have a value
                cal_value = marker.get("value")
                if cal_value is not None:
                    record["BG Reading (mg/dL)"] = cal_value
                    record["Calibration BG (mg/dL)"] = cal_value
            elif marker_type == "AUTO_BASAL_DELIVERY":
                # Auto mode basal - only add if we have a rate
                basal = marker.get("bolusAmount")
                if basal is not None:
                    record["Basal Rate (U/h)"] = basal
            elif marker_type == "BG":
                # Blood glucose meter reading - only add if we have a value
                bg_value = marker.get("value")
                if bg_value is not None:
                    record["BG Reading (mg/dL)"] = bg_value
            elif marker_type == "EXERCISE":
                # Exercise marker
                duration = marker.get("duration")
                if duration is not None:
                    record["Event Marker"] = f"Exercise: {duration}min"
            elif marker_type == "NOTE":
                # Note marker
                text = marker.get("text")
                if text:
                    record["Event Marker"] = text
            
            # Only add record if it has actual data (more than just Timestamp + Source)
            if len(record) > 2:
                records.append(record)
        
        return records


class CareLinkApiClient:
    """
    REST API client for Medtronic CareLink.
    
    Uses token-based authentication to bypass the web login flow.
    Automatically refreshes the token before expiration.
    """
    
    def __init__(
        self,
        token: str,
        country_code: str = "us",
        patient_username: str | None = None,
    ):
        """
        Initialize CareLink API client.
        
        Args:
            token: Initial auth token from browser cookie (auth_tmp_token)
            country_code: Country code ('us' for US, others for EU)
            patient_username: Optional patient username (for care partners)
        """
        self._auth_token = token
        self._auth_token_validto: str | None = None
        self._country_code = country_code.lower()
        self._patient_username = patient_username
        
        # Session data (populated on first API call)
        self._session_user: dict | None = None
        self._session_profile: dict | None = None
        self._session_country_settings: dict | None = None
        self._session_monitor_data: dict | None = None
        self._session_patients: list | None = None
        
        # HTTP session
        self._http = requests.Session()
        
        # State
        self._initialized = False
        self._last_error: str | None = None
    
    @property
    def server(self) -> str:
        """Get the CareLink server URL based on country."""
        if self._country_code == "us":
            return CARELINK_SERVER_US
        return CARELINK_SERVER_EU
    
    @property
    def is_initialized(self) -> bool:
        """Check if the client has been initialized."""
        return self._initialized
    
    @property
    def last_error(self) -> str | None:
        """Get the last error message."""
        return self._last_error
    
    def _check_token(self) -> bool:
        """
        Validate and decode the initial token.
        
        Returns:
            True if token is valid, False otherwise
        """
        if not self._auth_token:
            logger.error("No auth token provided")
            return False
        
        try:
            # Decode JWT payload
            parts = self._auth_token.split(".")
            if len(parts) != 3:
                logger.error("Invalid token format (not a JWT)")
                return False
            
            payload_b64 = parts[1]
            # Add padding if needed
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            
            payload_bytes = base64.b64decode(payload_b64)
            payload = json.loads(payload_bytes.decode())
            
            # Get expiration timestamp
            exp = payload.get("exp")
            if not exp:
                logger.error("Token has no expiration")
                return False
            
            # Check if expired (with 10 minute buffer)
            exp_with_buffer = exp - 600
            now = time.time()
            
            if now > exp_with_buffer:
                time_ago = int(now - exp)
                logger.warning(f"Token expired {time_ago}s ago")
                return False
            
            # Store expiration time
            self._auth_token_validto = datetime.utcfromtimestamp(exp).strftime(
                "%a %b %d %H:%M:%S UTC %Y"
            )
            
            time_remaining = int(exp - now)
            logger.info(f"Token valid for {time_remaining}s (until {self._auth_token_validto})")
            return True
            
        except Exception as e:
            logger.error(f"Error decoding token: {e}")
            return False
    
    def _get_auth_header(self) -> str | None:
        """
        Get the Authorization header value, refreshing token if needed.
        
        Returns:
            Bearer token string, or None if token is invalid
        """
        if not self._auth_token_validto:
            # First time - validate token
            if not self._check_token():
                return None
        
        # Check if token needs refresh
        try:
            validto = datetime.strptime(
                self._auth_token_validto, "%a %b %d %H:%M:%S UTC %Y"
            )
            time_remaining = (validto - datetime.utcnow()).total_seconds()
            
            if time_remaining < AUTH_EXPIRE_DEADLINE_MINUTES * 60:
                logger.info(f"Token expires in {int(time_remaining)}s, refreshing...")
                if not self._refresh_token():
                    logger.error("Token refresh failed")
                    return None
        except Exception as e:
            logger.warning(f"Error checking token expiration: {e}")
        
        return f"Bearer {self._auth_token}"
    
    def _refresh_token(self) -> bool:
        """
        Refresh the authentication token.
        
        Returns:
            True if refresh successful, False otherwise
        """
        url = f"https://{self.server}/patient/sso/reauth"
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Authorization": f"Bearer {self._auth_token}",
        }
        
        try:
            response = self._http.post(url, headers=headers)
            
            if response.ok:
                # Get new token from cookies
                new_token = self._http.cookies.get(AUTH_TOKEN_COOKIE)
                new_validto = self._http.cookies.get(TOKEN_VALID_TO_COOKIE)
                
                if new_token:
                    self._auth_token = new_token
                    if new_validto:
                        self._auth_token_validto = new_validto
                    logger.info(f"Token refreshed, valid until {self._auth_token_validto}")
                    return True
                else:
                    logger.error("No new token in refresh response")
                    return False
            else:
                logger.error(f"Token refresh failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False
    
    def _api_get(self, path: str, params: dict | None = None) -> dict | None:
        """
        Make an authenticated GET request to the CareLink API.
        
        Args:
            path: API path (e.g., 'patient/users/me')
            params: Optional query parameters
            
        Returns:
            JSON response data, or None on error
        """
        auth = self._get_auth_header()
        if not auth:
            return None
        
        url = f"https://{self.server}/{path}"
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": auth,
        }
        
        try:
            response = self._http.get(url, headers=headers, params=params)
            
            if response.ok:
                return response.json()
            else:
                logger.error(f"API GET {path} failed: {response.status_code}")
                self._last_error = f"API error: {response.status_code}"
                return None
                
        except Exception as e:
            logger.error(f"API GET {path} error: {e}")
            self._last_error = str(e)
            return None
    
    def _api_post(self, url: str, data: dict | str) -> dict | None:
        """
        Make an authenticated POST request.
        
        Args:
            url: Full URL to POST to
            data: Request body (dict or JSON string)
            
        Returns:
            JSON response data, or None on error
        """
        auth = self._get_auth_header()
        if not auth:
            return None
        
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": auth,
        }
        
        if isinstance(data, dict):
            data = json.dumps(data)
        
        try:
            response = self._http.post(url, headers=headers, data=data)
            
            if response.ok:
                return response.json()
            else:
                logger.error(f"API POST failed: {response.status_code}")
                self._last_error = f"API error: {response.status_code}"
                return None
                
        except Exception as e:
            logger.error(f"API POST error: {e}")
            self._last_error = str(e)
            return None
    
    def initialize(self) -> bool:
        """
        Initialize the client by fetching session data.
        
        This validates the token and retrieves user/profile information
        needed for subsequent API calls.
        
        Returns:
            True if initialization successful, False otherwise
        """
        logger.info("Initializing CareLink API client...")
        
        # Validate token first
        if not self._check_token():
            self._last_error = "Invalid or expired token"
            return False
        
        try:
            # Get user info
            self._session_user = self._api_get("patient/users/me")
            if not self._session_user:
                self._last_error = "Failed to get user info"
                return False
            logger.info(f"User: {self._session_user.get('username')}")
            
            # Get profile
            self._session_profile = self._api_get("patient/users/me/profile")
            if not self._session_profile:
                self._last_error = "Failed to get profile"
                return False
            
            # Get country settings
            self._session_country_settings = self._api_get(
                "patient/countries/settings",
                params={"countryCode": self._country_code, "language": "en"},
            )
            if not self._session_country_settings:
                self._last_error = "Failed to get country settings"
                return False
            
            # Get monitor data
            self._session_monitor_data = self._api_get("patient/monitor/data")
            if not self._session_monitor_data:
                self._last_error = "Failed to get monitor data"
                return False
            
            # Get patients list (for care partners)
            self._session_patients = self._api_get("patient/m2m/links/patients")
            
            # Select patient if not specified
            if not self._patient_username:
                if self._session_patients:
                    # Care partner - select from linked patients
                    for patient in self._session_patients:
                        if patient.get("status") == "ACTIVE":
                            self._patient_username = patient.get("username")
                            logger.info(
                                f"Selected patient: {patient.get('firstName')} "
                                f"{patient.get('lastName')} ({self._patient_username})"
                            )
                            break
                else:
                    # Regular patient - use own username from profile
                    self._patient_username = self._session_profile.get("username")
                    logger.info(f"Using own username as patient: {self._patient_username}")
            
            self._initialized = True
            logger.info("CareLink API client initialized successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Initialization error: {e}")
            self._last_error = str(e)
            return False
    
    def get_recent_data(self) -> CareLinkData | None:
        """
        Get recent pump/sensor data from CareLink.
        
        Returns:
            CareLinkData object with pump data, or None on error
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        try:
            # Determine which endpoint to use based on device type
            device_family = self._session_monitor_data.get("deviceFamily", "")
            
            if self._country_code == "us" or "BLE" in device_family:
                # Use BLE endpoint for newer pumps
                endpoint = self._session_country_settings.get("blePereodicDataEndpoint")
                if not endpoint:
                    logger.error("No BLE endpoint in country settings")
                    return None
                
                logger.debug(f"Using BLE endpoint: {endpoint}")
                logger.debug(f"Device family: {device_family}")
                
                # Determine role
                role = self._session_user.get("role", "")
                if role in ["CARE_PARTNER", "CARE_PARTNER_OUS"]:
                    role = "carepartner"
                else:
                    role = "patient"
                
                # Build request
                request_data = {
                    "username": self._session_profile.get("username"),
                    "role": role,
                    "patientId": self._patient_username,
                }
                
                logger.debug(f"Request data: {request_data}")
                data = self._api_post(endpoint, request_data)
            else:
                # Use legacy endpoint for older pumps
                params = {
                    "cpSerialNumber": "NONE",
                    "msgType": "last24hours",
                    "requestTime": str(int(time.time() * 1000)),
                }
                data = self._api_get("patient/connect/data", params=params)
            
            if data:
                return CareLinkData(raw_data=data, retrieved_at=datetime.now(UTC))
            
            return None
            
        except Exception as e:
            logger.exception(f"Error getting recent data: {e}")
            self._last_error = str(e)
            return None
    
    def get_current_token(self) -> str:
        """
        Get the current auth token (may have been refreshed).
        
        Useful for persisting the token after refresh.
        """
        return self._auth_token
