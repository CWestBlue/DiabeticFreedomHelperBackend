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
    
    # Selectors - Updated based on CareLink UI analysis
    # Initial landing page "Sign In" link/button
    SIGN_IN_LINK_SELECTORS = [
        "//a[contains(text(), 'Sign In')]",
        "//button[contains(text(), 'Sign In')]",
        "//a[contains(@href, 'login')]",
        "//span[contains(text(), 'Sign In')]/parent::*",
    ]
    # Login form fields
    USERNAME_SELECTORS = [
        "#username",
        "input[name='username']",
        "input[id='username']",
        "input[placeholder*='username' i]",
        "input[placeholder*='email' i]",
        "input[type='text']:first-of-type",
    ]
    PASSWORD_SELECTORS = [
        "#password",
        "input[name='password']",
        "input[id='password']",
        "input[type='password']",
    ]
    # Submit button
    LOGIN_SUBMIT_SELECTORS = [
        "//button[@type='submit']",
        "//button[contains(text(), 'Sign In')]",
        "//input[@type='submit']",
        "//button[contains(@class, 'submit')]",
        "//button[contains(@class, 'login')]",
    ]
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
        
        # Headless mode - can be disabled for debugging via CARELINK_HEADLESS=false in .env
        if self.settings.carelink_headless:
            options.add_argument("--headless=new")
            logger.info("Running in HEADLESS mode")
        else:
            logger.info("Running in VISIBLE mode (browser window will open)")
        
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        
        # Set window size (important for button positioning)
        options.add_argument("--window-size=1920,1080")
        
        # Anti-detection measures
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        
        # Set realistic user agent
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        options.add_argument(f"--user-agent={user_agent}")
        
        # Additional settings to appear more like real browser
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-extensions")
        options.add_argument("--enable-javascript")
        options.add_argument("--lang=en-US,en")
        
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
    
    def _find_element_multi(
        self,
        selectors: list[str],
        by_type: str = "css",
        timeout: int = 5,
    ):
        """
        Try multiple selectors until one works.
        
        Args:
            selectors: List of selectors to try
            by_type: "css" or "xpath"
            timeout: Timeout per selector
            
        Returns:
            WebElement if found, None otherwise
        """
        by = By.CSS_SELECTOR if by_type == "css" else By.XPATH
        
        for selector in selectors:
            try:
                element = WebDriverWait(self._driver, timeout).until(
                    EC.element_to_be_clickable((by, selector))
                )
                logger.debug(f"Found element with selector: {selector}")
                return element
            except TimeoutException:
                continue
        
        return None

    def _save_debug_screenshot(self, name: str) -> None:
        """Save a screenshot for debugging."""
        if self._download_dir:
            try:
                path = os.path.join(self._download_dir, f"debug_{name}.png")
                self._driver.save_screenshot(path)
                logger.info(f"Debug screenshot saved: {path}")
            except Exception as e:
                logger.warning(f"Could not save screenshot: {e}")

    def _wait_for_angular_load(self, timeout: int = 30) -> bool:
        """
        Wait for Angular/SPA to finish loading.
        
        CareLink uses a JavaScript SPA that renders content after page load.
        """
        logger.info("Waiting for SPA to render...")
        
        try:
            # Wait for document ready state
            WebDriverWait(self._driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            logger.info("Document ready state: complete")
            
            # Check if JavaScript is working
            js_test = self._driver.execute_script("return typeof window !== 'undefined'")
            logger.info(f"JavaScript working: {js_test}")
            
            # Check for Angular
            angular_check = self._driver.execute_script(
                """
                var hasTestabilities = typeof window.getAllAngularTestabilities === 'function';
                return {
                    hasAngular: typeof window.ng !== 'undefined',
                    hasAngularTestabilities: hasTestabilities,
                    bodyChildren: document.body ? document.body.children.length : 0,
                    inputCount: document.querySelectorAll('input').length
                };
                """
            )
            logger.info(f"Angular check: {angular_check}")
            
            # Wait for Angular to stabilize (if Angular app)
            if angular_check.get("hasAngularTestabilities"):
                try:
                    WebDriverWait(self._driver, 10).until(
                        lambda d: d.execute_script(
                            "return window.getAllAngularTestabilities().every(t => t.isStable());"
                        )
                    )
                    logger.info("Angular stabilized")
                except Exception as e:
                    logger.warning(f"Angular stabilization timeout: {e}")
            
            # Wait for inputs to appear (poll every second)
            logger.info("Polling for input elements...")
            for i in range(timeout):
                input_count = self._driver.execute_script(
                    "return document.querySelectorAll('input').length;"
                )
                if input_count > 0:
                    logger.info(f"Found {input_count} inputs after {i+1} seconds")
                    return True
                time.sleep(1)
            
            logger.warning(f"No inputs found after {timeout} seconds")
            return False
            
        except Exception as e:
            logger.warning(f"Error waiting for SPA: {e}")
            return False

    def _check_for_iframes(self) -> bool:
        """
        Check if login form is inside an iframe and switch to it.
        """
        try:
            iframes = self._driver.find_elements(By.TAG_NAME, "iframe")
            logger.info(f"Found {len(iframes)} iframes on page")
            
            for i, iframe in enumerate(iframes):
                try:
                    self._driver.switch_to.frame(iframe)
                    logger.info(f"Switched to iframe {i}")
                    
                    # Check if this iframe has login fields
                    inputs = self._driver.find_elements(By.TAG_NAME, "input")
                    if len(inputs) >= 2:
                        logger.info(
                            f"Iframe {i} has {len(inputs)} inputs - likely login"
                        )
                        return True
                    
                    # Switch back and try next iframe
                    self._driver.switch_to.default_content()
                except Exception as e:
                    logger.debug(f"Could not switch to iframe {i}: {e}")
                    self._driver.switch_to.default_content()
            
            return False
        except Exception as e:
            logger.debug(f"Error checking iframes: {e}")
            return False

    def _login(self, username: str, password: str) -> bool:
        """
        Perform CareLink login.
        
        Args:
            username: CareLink username
            password: CareLink password
            
        Returns:
            True if login successful, False otherwise
        """
        # Go to CareLink - may land on home page first
        login_url = f"{self.BASE_URL}/app/login"
        logger.info(f"Navigating to CareLink: {login_url}")
        self._driver.get(login_url)
        
        # Wait for page to load
        time.sleep(5)
        
        logger.info(f"Current URL: {self._driver.current_url}")
        logger.info(f"Page title: {self._driver.title}")
        self._save_debug_screenshot("01_landing_page")
        
        try:
            # Step 0: Click "Sign in" button on landing page
            logger.info("Looking for 'Sign in' button on landing page...")
            
            # Try CSS selectors first (more reliable for this page)
            sign_in_css_selectors = [
                "#landing-login-button-id",  # Exact ID from DevTools
                "[data-qa='landing-login-button']",  # Data attribute
                "button[id*='login']",
                "button.mat-primary",
            ]
            
            sign_in_btn = self._find_element_multi(
                sign_in_css_selectors,
                by_type="css",
                timeout=10,
            )
            
            # Fallback to XPath if CSS didn't work
            if not sign_in_btn:
                sign_in_xpath_selectors = [
                    "//button[.//span[contains(text(), 'Sign in')]]",
                    "//button[contains(@id, 'login')]",
                    "//button[contains(@data-qa, 'login')]",
                ]
                sign_in_btn = self._find_element_multi(
                    sign_in_xpath_selectors,
                    by_type="xpath",
                    timeout=5,
                )
            
            if sign_in_btn:
                logger.info("Found 'Sign in' button, clicking...")
                sign_in_btn.click()
                time.sleep(5)  # Wait for login form to load
                logger.info(f"After Sign in click - URL: {self._driver.current_url}")
                self._save_debug_screenshot("02_after_signin_click")
            else:
                logger.warning("No 'Sign in' button found, checking if already on login form")
            
            # Wait for SPA to render the login form
            self._wait_for_angular_load(timeout=30)
            
            # Check if login is in an iframe
            in_iframe = self._check_for_iframes()
            if in_iframe:
                logger.info("Login form found in iframe")
            
            # Log all input elements found
            all_inputs = self._driver.find_elements(By.TAG_NAME, "input")
            logger.info(f"Found {len(all_inputs)} input elements on page")
            for inp in all_inputs[:5]:  # Log first 5
                try:
                    inp_type = inp.get_attribute("type")
                    inp_name = inp.get_attribute("name")
                    inp_id = inp.get_attribute("id")
                    inp_placeholder = inp.get_attribute("placeholder")
                    logger.info(
                        f"  Input: type={inp_type}, name={inp_name}, "
                        f"id={inp_id}, placeholder={inp_placeholder}"
                    )
                except Exception:
                    pass
            
            self._save_debug_screenshot("02_after_wait")
            
            # Step 1: Find username field with extended selectors
            logger.info("Looking for username field...")
            extended_username_selectors = self.USERNAME_SELECTORS + [
                "input[type='email']",
                "input[autocomplete='username']",
                "input[autocomplete='email']",
                "input:not([type='password']):not([type='hidden']):not([type='submit'])",
            ]
            
            username_field = self._find_element_multi(
                extended_username_selectors,
                by_type="css",
                timeout=20,
            )
            
            if not username_field:
                # Try finding any visible text input
                logger.warning("Standard selectors failed, trying to find any text input...")
                try:
                    inputs = self._driver.find_elements(By.CSS_SELECTOR, "input")
                    for inp in inputs:
                        inp_type = inp.get_attribute("type") or "text"
                        skip_types = ["password", "hidden", "submit", "button"]
                        if inp_type not in skip_types and inp.is_displayed():
                            username_field = inp
                            html_preview = inp.get_attribute("outerHTML")[:100]
                            logger.info(f"Found fallback input: {html_preview}")
                            break
                except Exception as e:
                    logger.error(f"Fallback search failed: {e}")
            
            if not username_field:
                logger.error("Could not find username field!")
                logger.error(f"Current URL: {self._driver.current_url}")
                # Log full page source for debugging
                page_source = self._driver.page_source
                logger.error(f"Page source length: {len(page_source)}")
                logger.error(f"Page source preview: {page_source[:2000]}")
                self._save_debug_screenshot("03_no_username_field")
                return False
            
            logger.info("Found username field")
            
            # Step 3: Find password field
            password_field = self._find_element_multi(
                self.PASSWORD_SELECTORS,
                by_type="css",
                timeout=5,
            )
            
            if not password_field:
                logger.error("Could not find password field!")
                self._save_debug_screenshot("04_no_password_field")
                return False
            
            logger.info("Found password field")
            
            # Step 4: Enter credentials
            logger.info("Entering credentials...")
            username_field.clear()
            username_field.send_keys(username)
            time.sleep(0.5)
            
            password_field.clear()
            password_field.send_keys(password)
            time.sleep(0.5)
            
            self._save_debug_screenshot("05_credentials_entered")
            
            # Step 5: Find and click submit button
            logger.info("Looking for submit button...")
            submit_btn = self._find_element_multi(
                self.LOGIN_SUBMIT_SELECTORS,
                by_type="xpath",
                timeout=10,
            )
            
            if not submit_btn:
                logger.error("Could not find submit button!")
                self._save_debug_screenshot("06_no_submit_button")
                return False
            
            logger.info("Clicking submit button...")
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
        
        NOTE: CareLink's datepicker is a complex Angular Material component.
        For now, we skip date selection and use CareLink's default range.
        The UploadService handles deduplication, so extra data is filtered out.
        
        Args:
            start_date: Start date for report (logged but not yet used)
            end_date: End date for report (logged but not yet used)
            
        Returns:
            True (always continues with default range)
        """
        # Format dates for logging
        start_str = start_date.strftime("%m/%d/%Y")
        end_str = end_date.strftime("%m/%d/%Y")
        
        logger.info(f"Desired date range: {start_str} to {end_str}")
        logger.warning(
            "Date picker manipulation not yet implemented - "
            "using CareLink's default range. "
            "UploadService will filter duplicates."
        )
        
        # TODO: Implement proper date range selection
        # CareLink uses Angular Material date range picker which requires:
        # 1. Click date range bar to open calendar
        # 2. Navigate calendar to correct month
        # 3. Click start date
        # 4. Click end date
        # 5. Click Apply
        # For now, we rely on UploadService deduplication
        
        return True
    
    def _trigger_export(self) -> str | None:
        """
        Click the export button and wait for download.
        
        Returns:
            Path to downloaded file, or None if failed
        """
        logger.info("Looking for Data Export (CSV) button...")
        
        try:
            # First, dismiss any open datepicker/overlay by pressing Escape or clicking body
            try:
                # Wait for any datepicker backdrop to disappear
                backdrop_selector = "div.cdk-overlay-backdrop"
                WebDriverWait(self._driver, 3).until(
                    EC.invisibility_of_element_located((By.CSS_SELECTOR, backdrop_selector))
                )
                logger.info("Datepicker overlay dismissed")
            except TimeoutException:
                # Backdrop may still be present, try clicking body to dismiss
                try:
                    body = self._driver.find_element(By.TAG_NAME, "body")
                    # Press Escape key to close any open dialogs
                    from selenium.webdriver.common.keys import Keys
                    body.send_keys(Keys.ESCAPE)
                    time.sleep(0.5)
                    logger.info("Sent Escape key to dismiss overlay")
                except Exception:
                    pass
            
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
            
            # Click export using JavaScript to bypass any overlay issues
            logger.info("Clicking export button via JavaScript...")
            self._driver.execute_script("arguments[0].click();", export_btn)
            
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
        
        Uses the same query pattern as the dashboard service.
        
        Returns:
            Latest timestamp or None if no data exists
        """
        pipeline = [
            {"$match": {"Timestamp": {"$ne": None}}},
            {
                "$group": {
                    "_id": None,
                    "Earliest": {"$min": "$Timestamp"},
                    "Latest": {"$max": "$Timestamp"},
                }
            },
            {"$project": {"_id": 0, "Earliest": 1, "Latest": 1}},
        ]
        
        results = await self.uow.pump_data.aggregate(pipeline, limit=1)
        
        if results:
            earliest = results[0].get("Earliest")
            latest = results[0].get("Latest")
            logger.info(f"Database date range: Earliest={earliest}, Latest={latest}")
            return latest
        
        logger.info("No existing data found in database")
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

