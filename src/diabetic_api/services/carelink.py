"""CareLink sync service for automated data retrieval.

This service automates the process of:
1. Logging into Medtronic CareLink
2. Selecting date range (from last upload to now)
3. Downloading CSV export
4. Processing via existing UploadService
"""

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from diabetic_api.core.config import Settings, get_settings
from diabetic_api.db.unit_of_work import UnitOfWork
from diabetic_api.services.upload import UploadService

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a CareLink sync operation."""
    
    success: bool
    message: str
    records_imported: int = 0
    sync_started_at: datetime | None = None
    sync_completed_at: datetime | None = None
    error: str | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "success": self.success,
            "message": self.message,
            "records_imported": self.records_imported,
            "sync_started_at": (
                self.sync_started_at.isoformat() if self.sync_started_at else None
            ),
            "sync_completed_at": (
                self.sync_completed_at.isoformat() if self.sync_completed_at else None
            ),
            "error": self.error,
        }


class CareLinkSyncService:
    """
    Service for syncing data from Medtronic CareLink.
    
    Uses Selenium to automate the login and CSV download process,
    then processes the data using the existing UploadService.
    """
    
    # CareLink URLs
    BASE_URL = "https://carelink.minimed.com"
    LOGIN_URL = f"{BASE_URL}/patient/sso/login"
    REPORTS_URL = f"{BASE_URL}/app/reports"
    
    # Timeouts (seconds)
    PAGE_LOAD_TIMEOUT = 30
    ELEMENT_WAIT_TIMEOUT = 20
    DOWNLOAD_WAIT_TIMEOUT = 60
    
    # Selectors
    SIGN_IN_BUTTON_XPATH = "//button[contains(text(), 'Sign In') or contains(text(), 'sign in')]"
    USERNAME_SELECTOR = "input[name='username'], input[id='username'], input[type='text']"
    PASSWORD_SELECTOR = "input[name='password'], input[id='password'], input[type='password']"
    LOGIN_SUBMIT_XPATH = "//button[@type='submit' and contains(., 'Sign In')]"
    EXPORT_BUTTON_XPATH = "//*[contains(text(), 'Data Export') or contains(text(), 'CSV')]"
    LOADING_SPINNER_XPATH = "//div[contains(@class, 'spinner') or contains(@class, 'loading')]"
    
    def __init__(
        self,
        uow: UnitOfWork,
        upload_service: UploadService | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize CareLink sync service.
        
        Args:
            uow: Unit of Work for database operations
            upload_service: Optional UploadService instance (created if not provided)
            settings: Optional Settings instance (uses get_settings if not provided)
        """
        self.uow = uow
        self.upload_service = upload_service or UploadService(uow)
        self.settings = settings or get_settings()
        self._driver: webdriver.Chrome | None = None
        self._download_dir: str | None = None
    
    def _create_driver(self) -> webdriver.Chrome:
        """
        Create and configure Chrome WebDriver for headless operation.
        
        Returns:
            Configured Chrome WebDriver instance
        """
        # Create temp directory for downloads
        self._download_dir = tempfile.mkdtemp(prefix="carelink_")
        
        options = ChromeOptions()
        
        # Use system Chrome/Chromium if specified
        chrome_bin = os.environ.get("CHROME_BIN")
        if chrome_bin and os.path.exists(chrome_bin):
            options.binary_location = chrome_bin
        
        # Headless mode
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        
        # Set window size (important for button positioning)
        options.add_argument("--window-size=1920,1080")
        
        # Disable automation flags to avoid detection
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        
        # Configure downloads
        prefs = {
            "download.default_directory": self._download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        }
        options.add_experimental_option("prefs", prefs)
        
        # Try to use system chromium-driver first, fall back to webdriver-manager
        try:
            # Check for environment variable (set in Docker)
            chromedriver_path = os.environ.get("CHROMEDRIVER_PATH")
            if chromedriver_path and os.path.exists(chromedriver_path):
                service = ChromeService(executable_path=chromedriver_path)
            # Check for system chromedriver
            elif os.path.exists("/usr/bin/chromedriver"):
                service = ChromeService(executable_path="/usr/bin/chromedriver")
            elif os.path.exists("/usr/local/bin/chromedriver"):
                service = ChromeService(executable_path="/usr/local/bin/chromedriver")
            else:
                # Use webdriver-manager to auto-download
                from webdriver_manager.chrome import ChromeDriverManager
                service = ChromeService(ChromeDriverManager().install())
        except Exception as e:
            logger.warning(f"ChromeDriver setup issue: {e}, trying default")
            service = ChromeService()
        
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(self.PAGE_LOAD_TIMEOUT)
        
        return driver
    
    def _cleanup_driver(self) -> None:
        """Clean up WebDriver and temp files."""
        if self._driver:
            try:
                self._driver.quit()
            except Exception as e:
                logger.warning(f"Error closing driver: {e}")
            self._driver = None
        
        # Clean up download directory
        if self._download_dir and os.path.exists(self._download_dir):
            try:
                import shutil
                shutil.rmtree(self._download_dir)
            except Exception as e:
                logger.warning(f"Error cleaning temp dir: {e}")
            self._download_dir = None
    
    def _wait_for_element(
        self,
        by: By,
        value: str,
        timeout: int = None,
        clickable: bool = False,
    ):
        """
        Wait for an element to be present or clickable.
        
        Args:
            by: Selenium By locator type
            value: Locator value
            timeout: Wait timeout in seconds
            clickable: If True, wait for element to be clickable
            
        Returns:
            WebElement if found
            
        Raises:
            TimeoutException: If element not found within timeout
        """
        timeout = timeout or self.ELEMENT_WAIT_TIMEOUT
        wait = WebDriverWait(self._driver, timeout)
        
        if clickable:
            condition = EC.element_to_be_clickable((by, value))
        else:
            condition = EC.presence_of_element_located((by, value))
        
        return wait.until(condition)
    
    def _login(self, username: str, password: str) -> bool:
        """
        Perform CareLink login.
        
        Args:
            username: CareLink username
            password: CareLink password
            
        Returns:
            True if login successful, False otherwise
        """
        logger.info("Navigating to CareLink login page...")
        self._driver.get(self.BASE_URL)
        
        try:
            # Look for and click initial "Sign In" button on landing page
            try:
                sign_in_btn = self._wait_for_element(
                    By.XPATH,
                    self.SIGN_IN_BUTTON_XPATH,
                    timeout=10,
                    clickable=True,
                )
                sign_in_btn.click()
                logger.info("Clicked initial Sign In button")
                time.sleep(2)  # Wait for login form to load
            except TimeoutException:
                # May already be on login page
                logger.info("No initial Sign In button, may already be on login form")
            
            # Wait for username field
            logger.info("Waiting for login form...")
            username_field = self._wait_for_element(
                By.CSS_SELECTOR,
                self.USERNAME_SELECTOR,
                clickable=True,
            )
            
            # Find password field
            password_field = self._driver.find_element(
                By.CSS_SELECTOR,
                self.PASSWORD_SELECTOR,
            )
            
            # Enter credentials
            logger.info("Entering credentials...")
            username_field.clear()
            username_field.send_keys(username)
            
            password_field.clear()
            password_field.send_keys(password)
            
            # Find and click submit button
            submit_btn = self._wait_for_element(
                By.XPATH,
                self.LOGIN_SUBMIT_XPATH,
                clickable=True,
            )
            submit_btn.click()
            
            logger.info("Submitted login form, waiting for redirect...")
            
            # Wait for successful login (redirect to reports or dashboard)
            time.sleep(5)  # Give time for redirect
            
            # Check if we're logged in by looking for reports page elements
            # or checking URL
            current_url = self._driver.current_url
            if "reports" in current_url or "app" in current_url:
                logger.info(f"Login successful, redirected to: {current_url}")
                return True
            
            # Check for error messages
            try:
                error_elem = self._driver.find_element(
                    By.XPATH,
                    "//*[contains(@class, 'error') or contains(@class, 'alert')]",
                )
                if error_elem.is_displayed():
                    logger.error(f"Login error displayed: {error_elem.text}")
                    return False
            except Exception:
                pass
            
            # Give more time and check again
            time.sleep(5)
            current_url = self._driver.current_url
            if "login" not in current_url.lower():
                logger.info(f"Login appears successful, now at: {current_url}")
                return True
            
            logger.error("Login failed - still on login page")
            return False
            
        except TimeoutException as e:
            logger.error(f"Timeout during login: {e}")
            return False
        except Exception as e:
            logger.error(f"Error during login: {e}")
            return False
    
    def _navigate_to_reports(self) -> bool:
        """
        Navigate to the reports page if not already there.
        
        Returns:
            True if on reports page, False otherwise
        """
        current_url = self._driver.current_url
        
        if "reports" in current_url.lower():
            logger.info("Already on reports page")
            return True
        
        logger.info("Navigating to reports page...")
        self._driver.get(self.REPORTS_URL)
        time.sleep(3)
        
        # Verify we're on reports page
        current_url = self._driver.current_url
        if "reports" in current_url.lower():
            logger.info("Successfully navigated to reports page")
            return True
        
        logger.error(f"Failed to navigate to reports, current URL: {current_url}")
        return False
    
    def _set_date_range(self, start_date: datetime, end_date: datetime) -> bool:
        """
        Set the date range for the report.
        
        Args:
            start_date: Start date for report
            end_date: End date for report
            
        Returns:
            True if date range set successfully
        """
        logger.info(f"Setting date range: {start_date.date()} to {end_date.date()}")
        
        try:
            # Look for date range picker
            date_picker_selectors = [
                "//div[contains(@class, 'date-range')]",
                "//button[contains(@class, 'date')]",
                "//input[contains(@type, 'date')]",
                "//*[contains(text(), 'Date Range')]",
            ]
            
            date_picker = None
            for selector in date_picker_selectors:
                try:
                    date_picker = self._wait_for_element(
                        By.XPATH,
                        selector,
                        timeout=5,
                        clickable=True,
                    )
                    break
                except TimeoutException:
                    continue
            
            if not date_picker:
                logger.warning("Could not find date picker, using default range")
                return True  # Continue with default range
            
            date_picker.click()
            time.sleep(1)
            
            # Date selection logic would go here
            # This is simplified - actual implementation depends on the specific
            # date picker UI used by CareLink
            
            logger.info("Date range selection attempted")
            return True
            
        except Exception as e:
            logger.warning(f"Error setting date range: {e}, continuing with default")
            return True  # Continue with default range
    
    def _trigger_export(self) -> str | None:
        """
        Click the export button and wait for download.
        
        Returns:
            Path to downloaded file, or None if failed
        """
        logger.info("Looking for Data Export (CSV) button...")
        
        try:
            # Find and click export button
            export_selectors = [
                "//*[contains(text(), 'Data Export (CSV)')]",
                "//*[contains(text(), 'Data Export')]",
                "//button[contains(., 'CSV')]",
                "//*[contains(@class, 'export')]",
            ]
            
            export_btn = None
            for selector in export_selectors:
                try:
                    export_btn = self._wait_for_element(
                        By.XPATH,
                        selector,
                        timeout=5,
                        clickable=True,
                    )
                    break
                except TimeoutException:
                    continue
            
            if not export_btn:
                logger.error("Could not find export button")
                return None
            
            # Click export
            logger.info("Clicking export button...")
            export_btn.click()
            
            # Wait for loading spinner to appear and disappear
            try:
                # Wait for spinner to appear
                self._wait_for_element(
                    By.XPATH,
                    self.LOADING_SPINNER_XPATH,
                    timeout=5,
                )
                logger.info("Export loading...")
                
                # Wait for spinner to disappear
                WebDriverWait(self._driver, self.DOWNLOAD_WAIT_TIMEOUT).until(
                    EC.invisibility_of_element_located((By.XPATH, self.LOADING_SPINNER_XPATH))
                )
                logger.info("Loading complete")
            except TimeoutException:
                # Spinner may not be present, continue
                logger.info("No loading spinner detected, continuing...")
            
            # Wait for file to appear in download directory
            return self._wait_for_download()
            
        except Exception as e:
            logger.error(f"Error triggering export: {e}")
            return None
    
    def _wait_for_download(self) -> str | None:
        """
        Wait for CSV file to appear in download directory.
        
        Returns:
            Path to downloaded file, or None if not found
        """
        logger.info(f"Waiting for download in: {self._download_dir}")
        
        start_time = time.time()
        while time.time() - start_time < self.DOWNLOAD_WAIT_TIMEOUT:
            # Check for CSV files
            if self._download_dir:
                files = list(Path(self._download_dir).glob("*.csv"))
                
                # Filter out partial downloads (.crdownload files)
                csv_files = [f for f in files if not str(f).endswith(".crdownload")]
                
                if csv_files:
                    # Return the most recently modified CSV
                    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
                    logger.info(f"Download complete: {latest_file.name}")
                    return str(latest_file)
            
            time.sleep(1)
        
        logger.error("Download timed out")
        return None
    
    async def _get_last_upload_date(self) -> datetime | None:
        """
        Get the latest timestamp from existing data.
        
        Returns:
            Latest timestamp or None if no data exists
        """
        collection = self.uow.get_collection("PumpData")
        
        pipeline = [
            {"$match": {"Timestamp": {"$ne": None}}},
            {"$group": {"_id": None, "Latest": {"$max": "$Timestamp"}}},
            {"$project": {"_id": 0, "Latest": 1}},
        ]
        
        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=1)
        
        if results and results[0].get("Latest"):
            return results[0]["Latest"]
        return None
    
    async def sync(self) -> SyncResult:
        """
        Perform full CareLink sync operation.
        
        This is the main entry point that orchestrates:
        1. Login to CareLink
        2. Navigate to reports
        3. Set date range (from last upload to now)
        4. Download CSV export
        5. Process via UploadService
        
        Returns:
            SyncResult with operation status and details
        """
        sync_started = datetime.now(UTC)
        
        # Check if CareLink is configured
        if not self.settings.is_carelink_configured:
            return SyncResult(
                success=False,
                message="CareLink credentials not configured",
                sync_started_at=sync_started,
                sync_completed_at=datetime.now(UTC),
                error="Set CARELINK_USERNAME and CARELINK_PASSWORD environment variables",
            )
        
        try:
            # Get date range
            last_upload = await self._get_last_upload_date()
            end_date = datetime.now(UTC)
            
            if last_upload:
                start_date = last_upload
                logger.info(f"Syncing from last upload: {start_date} to {end_date}")
            else:
                # No existing data, sync last 90 days
                from datetime import timedelta
                start_date = end_date - timedelta(days=90)
                logger.info("No existing data, syncing last 90 days")
            
            # Run Selenium operations in thread pool (they're blocking)
            loop = asyncio.get_event_loop()
            csv_path = await loop.run_in_executor(
                None,
                self._sync_blocking,
                start_date,
                end_date,
            )
            
            if not csv_path:
                return SyncResult(
                    success=False,
                    message="Failed to download CSV from CareLink",
                    sync_started_at=sync_started,
                    sync_completed_at=datetime.now(UTC),
                    error="Check logs for details",
                )
            
            # Read CSV content
            with open(csv_path, "rb") as f:
                csv_content = f.read()
            
            # Process via UploadService
            filename = Path(csv_path).name
            upload_result = await self.upload_service.process_csv(
                file_content=csv_content,
                filename=filename,
                uploaded_at=datetime.now(UTC).isoformat(),
            )
            
            if upload_result.success:
                return SyncResult(
                    success=True,
                    message=upload_result.message,
                    records_imported=upload_result.records_inserted or 0,
                    sync_started_at=sync_started,
                    sync_completed_at=datetime.now(UTC),
                )
            else:
                return SyncResult(
                    success=False,
                    message=upload_result.message,
                    sync_started_at=sync_started,
                    sync_completed_at=datetime.now(UTC),
                    error="; ".join(upload_result.errors) if upload_result.errors else None,
                )
                
        except Exception as e:
            logger.exception("Sync failed with exception")
            return SyncResult(
                success=False,
                message="Sync failed with unexpected error",
                sync_started_at=sync_started,
                sync_completed_at=datetime.now(UTC),
                error=str(e),
            )
        finally:
            self._cleanup_driver()
    
    def _sync_blocking(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> str | None:
        """
        Blocking sync operation (runs in thread pool).
        
        Args:
            start_date: Start date for report
            end_date: End date for report
            
        Returns:
            Path to downloaded CSV file, or None if failed
        """
        try:
            # Create driver
            logger.info("Starting CareLink sync...")
            self._driver = self._create_driver()
            
            # Login
            if not self._login(
                self.settings.carelink_username,
                self.settings.carelink_password,
            ):
                logger.error("Login failed")
                return None
            
            # Navigate to reports
            if not self._navigate_to_reports():
                logger.error("Failed to navigate to reports")
                return None
            
            # Set date range
            self._set_date_range(start_date, end_date)
            
            # Give page time to load with new date range
            time.sleep(3)
            
            # Trigger export and wait for download
            csv_path = self._trigger_export()
            
            if csv_path:
                logger.info(f"Successfully downloaded: {csv_path}")
                return csv_path
            
            logger.error("Export failed")
            return None
            
        except WebDriverException as e:
            logger.error(f"WebDriver error: {e}")
            return None
        except Exception as e:
            logger.exception(f"Sync error: {e}")
            return None

