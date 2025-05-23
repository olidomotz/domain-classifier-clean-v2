"""
Enhanced Scrapy crawler implementation for domain classification.
This module replaces the existing scrapy_crawler.py with improved capabilities for content extraction.
"""
import logging
import crochet
crochet.setup()
import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy import signals
from scrapy.signalmanager import dispatcher
from typing import Tuple, Optional
from urllib.parse import urlparse
import time
import re
from scrapy.http import HtmlResponse
import traceback

# Import selenium - these imports are handled safely with try/except
selenium_available = False
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.common.by import By
    selenium_available = True
    logging.info("Selenium is available for JavaScript rendering")
except ImportError:
    logging.warning("Selenium not installed. JavaScript rendering will be limited.")

# Set up logging
logger = logging.getLogger(__name__)

class RotatingUserAgentMiddleware:
    """Middleware to rotate user agents to avoid detection."""
    
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Edge/115.0.1901.200',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1',
        # Added more agents for better rotation
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/115.0.1901.203'
    ]
    
    def __init__(self):
        self.current_index = 0
    
    def process_request(self, request, spider):
        # Skip non-HTML resources
        if any(ext in request.url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3', '.pdf', '.css', '.js']):
            request.meta['skip_non_html'] = True
            return None
            
        # Rotate through user agents
        user_agent = self.USER_AGENTS[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.USER_AGENTS)
        
        # Set the User-Agent header
        request.headers['User-Agent'] = user_agent
        # Add Sec-CH-UA header to simulate modern browser
        request.headers['Sec-CH-UA'] = '"Google Chrome";v="115", "Chromium";v="115", "Not=A?Brand";v="24"'
        request.headers['Accept-Language'] = 'en-US,en;q=0.9'
        # Add additional headers to improve redirect handling
        request.headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        request.headers['Referer'] = 'https://www.google.com/'
        return None


class SmartRetryMiddleware:
    """Middleware for smart retry logic with different strategies."""
    
    def __init__(self, settings):
        self.max_retry_times = settings.getint('RETRY_TIMES', 4)  # INCREASED from 3
        self.retry_http_codes = set(settings.getlist('RETRY_HTTP_CODES', [500, 502, 503, 504, 408, 429, 403]))
        self.priority_adjust = settings.getint('RETRY_PRIORITY_ADJUST', -1)
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)
    
    def process_response(self, request, response, spider):
        # Skip non-HTML resources
        if request.meta.get('skip_non_html', False):
            return response
            
        # Check if response is in retry codes
        if request.meta.get('dont_retry', False):
            return response
            
        if response.status in self.retry_http_codes:
            retry_count = request.meta.get('retry_count', 0)
            
            if retry_count < self.max_retry_times:
                # Implement different retry strategies based on error code
                if response.status == 403:  # Forbidden
                    return self._handle_forbidden(request, response, spider, retry_count)
                elif response.status == 429:  # Too Many Requests
                    return self._handle_rate_limit(request, response, spider, retry_count)
                else:  # Standard retry
                    return self._do_retry(request, response, spider, retry_count)
        
        # IMPROVED: Handle redirect-like behavior in non-30x responses
        # Some sites use JavaScript or meta refresh instead of proper HTTP redirects
        if response.status == 200:
            # Check for meta refresh tags
            meta_refresh = response.xpath('//meta[@http-equiv="refresh"]/@content').get()
            if meta_refresh:
                try:
                    # Extract URL from meta refresh
                    url_match = re.search(r'url=([^;]+)', meta_refresh, re.IGNORECASE)
                    if url_match:
                        redirect_url = url_match.group(1).strip()
                        # Create a new request for the redirect URL
                        logger.info(f"Following meta refresh redirect to {redirect_url}")
                        new_request = request.replace(url=redirect_url)
                        new_request.meta['redirect_times'] = request.meta.get('redirect_times', 0) + 1
                        # Prevent redirect loops
                        if new_request.meta['redirect_times'] <= 10:  # Set maximum redirects
                            return new_request
                except Exception as e:
                    logger.warning(f"Error processing meta refresh: {e}")
            
            # Check for JavaScript redirects in the page content
            js_redirect = response.xpath('//script[contains(text(), "window.location") or contains(text(), "document.location")]/text()').get()
            if js_redirect:
                try:
                    # Extract URL from JavaScript redirect
                    url_match = re.search(r'(?:window|document)\.location(?:\.href)?\s*=\s*[\'"]([^\'"]+)[\'"]', js_redirect)
                    if url_match:
                        redirect_url = url_match.group(1).strip()
                        # Create a new request for the redirect URL
                        logger.info(f"Following JavaScript redirect to {redirect_url}")
                        new_request = request.replace(url=redirect_url)
                        new_request.meta['redirect_times'] = request.meta.get('redirect_times', 0) + 1
                        # Prevent redirect loops
                        if new_request.meta['redirect_times'] <= 10:  # Set maximum redirects
                            return new_request
                except Exception as e:
                    logger.warning(f"Error processing JavaScript redirect: {e}")
        
        return response
    
    def process_exception(self, request, exception, spider):
        # Skip non-HTML resources
        if request.meta.get('skip_non_html', False):
            return None
            
        # Handle connection-related exceptions
        retry_count = request.meta.get('retry_count', 0)
        
        if retry_count < self.max_retry_times:
            # Log the exception
            logger.info(f"Retrying {request.url} due to exception: {exception.__class__.__name__}")
            
            # Use appropriate delay based on retry count
            retry_delay = 2 ** retry_count  # Exponential backoff
            
            # Create a new request
            new_request = request.copy()
            new_request.meta['retry_count'] = retry_count + 1
            new_request.dont_filter = True
            new_request.priority = request.priority + self.priority_adjust
            
            # Add delay
            new_request.meta['download_slot'] = self._get_slot(request)
            new_request.meta['download_delay'] = retry_delay
            
            logger.info(f"Retry {retry_count+1}/{self.max_retry_times} for {request.url} with delay {retry_delay}s")
            
            return new_request
        
        return None
    
    def _handle_forbidden(self, request, response, spider, retry_count):
        """Handle 403 Forbidden responses with special strategy."""
        logger.info(f"Handling 403 Forbidden for {request.url}")
        
        # Create a new request with different User-Agent
        new_request = request.copy()
        new_request.meta['retry_count'] = retry_count + 1
        new_request.dont_filter = True
        new_request.priority = request.priority + self.priority_adjust
        
        # Add custom User-Agent
        import random
        desktop_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0'
        ]
        new_request.headers['User-Agent'] = random.choice(desktop_agents)
        
        # Add other headers that might help
        new_request.headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        new_request.headers['Accept-Language'] = 'en-US,en;q=0.5'
        new_request.headers['Cache-Control'] = 'no-cache'
        
        # Add delay
        retry_delay = 5 + (5 * retry_count)  # Longer delay for 403s
        new_request.meta['download_slot'] = self._get_slot(request)
        new_request.meta['download_delay'] = retry_delay
        
        logger.info(f"Retry {retry_count+1}/{self.max_retry_times} for {request.url} with delay {retry_delay}s")
        
        return new_request
    
    def _handle_rate_limit(self, request, response, spider, retry_count):
        """Handle 429 Too Many Requests with appropriate backoff."""
        logger.info(f"Handling rate limit (429) for {request.url}")
        
        # Create a new request
        new_request = request.copy()
        new_request.meta['retry_count'] = retry_count + 1
        new_request.dont_filter = True
        new_request.priority = request.priority + self.priority_adjust
        
        # Calculate backoff time - check for Retry-After header first
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                # Retry-After can be an integer or a date
                retry_delay = int(retry_after)
            except ValueError:
                # If it's a date, use a default delay
                retry_delay = 30 * (retry_count + 1)
        else:
            # Exponential backoff with jitter
            import random
            retry_delay = (2 ** retry_count) * 10 + random.uniform(0, 5)
        
        # Add delay
        new_request.meta['download_slot'] = self._get_slot(request)
        new_request.meta['download_delay'] = retry_delay
        
        logger.info(f"Rate limit retry {retry_count+1}/{self.max_retry_times} for {request.url} with delay {retry_delay}s")
        
        return new_request
    
    def _do_retry(self, request, response, spider, retry_count):
        """Standard retry logic."""
        # Create a new request
        new_request = request.copy()
        new_request.meta['retry_count'] = retry_count + 1
        new_request.dont_filter = True
        new_request.priority = request.priority + self.priority_adjust
        
        # Calculate delay with exponential backoff
        retry_delay = 2 ** retry_count
        
        # Add delay
        new_request.meta['download_slot'] = self._get_slot(request)
        new_request.meta['download_delay'] = retry_delay
        
        logger.info(f"Standard retry {retry_count+1}/{self.max_retry_times} for {request.url} with delay {retry_delay}s")
        
        return new_request
    
    def _get_slot(self, request):
        """Get download slot for the request."""
        return request.meta.get('download_slot') or urlparse(request.url).netloc


class JavaScriptMiddleware:
    """Middleware to handle JavaScript-heavy sites using Selenium."""
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.selenium_driver = None
        self.js_urls = set()  # Keep track of URLs processed with JS
        self.selenium_available = selenium_available
    
    @classmethod
    def from_crawler(cls, crawler):
        middleware = cls(crawler)
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        return middleware
    
    def spider_opened(self, spider):
        """Initialize Selenium WebDriver when spider is opened."""
        pass  # Lazy initialization when needed
    
    def spider_closed(self, spider):
        """Close the Selenium WebDriver when spider is closed."""
        if self.selenium_driver:
            logger.info("Closing Selenium WebDriver")
            try:
                self.selenium_driver.quit()
            except Exception as e:
                logger.error(f"Error closing Selenium: {e}")
    
    def _initialize_selenium(self):
        """Initialize Selenium WebDriver if not already done and if available."""
        if not self.selenium_available:
            logger.warning("Selenium is not available - cannot render JavaScript")
            return False
            
        if not self.selenium_driver:
            logger.info("Initializing Selenium WebDriver")
            try:
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--window-size=1920,1080")
                
                # Add a custom user agent
                chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
                
                try:
                    # Try to use Service for newer selenium versions
                    self.selenium_driver = webdriver.Chrome(options=chrome_options)
                except Exception as e:
                    logger.error(f"Error initializing Chrome with Service: {e}")
                    # Fallback to older selenium versions
                    try:
                        self.selenium_driver = webdriver.Chrome(chrome_options=chrome_options)
                    except Exception as e2:
                        logger.error(f"Error initializing Chrome (fallback): {e2}")
                        return False
                        
                logger.info("Selenium WebDriver initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Error initializing Selenium WebDriver: {e}")
                self.selenium_driver = None
                return False
        return True
    
    def process_request(self, request, spider):
        """Process request to handle JavaScript-heavy sites."""
        # Skip non-HTML resources
        if request.meta.get('skip_non_html', False):
            return None
            
        # Only use Selenium for requests that need JS rendering
        if request.meta.get('js_render', False):
            # Initialize Selenium if not done already
            if not self._initialize_selenium():
                logger.error("Failed to initialize Selenium, skipping JS rendering")
                return None
            
            url = request.url
            logger.info(f"Using Selenium to render JavaScript for {url}")
            
            try:
                # Load the page with Selenium
                self.selenium_driver.get(url)
                
                # Add this URL to the set of JS-processed URLs
                self.js_urls.add(url)
                
                # Wait for JavaScript to load (wait for body to have content)
                try:
                    WebDriverWait(self.selenium_driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, 'body'))
                    )
                    # Also wait for common content containers if possible
                    for selector in ['main', 'article', '#content', '.content', '.main-content']:
                        try:
                            WebDriverWait(self.selenium_driver, 2).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                            )
                            logger.info(f"Found content container: {selector}")
                            break
                        except TimeoutException:
                            pass
                except TimeoutException:
                    logger.warning(f"Timeout waiting for body element on {url}")
                
                # Additional wait for dynamic content
                time.sleep(3)
                
                # Scroll to load lazy content
                self.selenium_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
                time.sleep(1)
                self.selenium_driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                
                # Try clicking on "Accept cookies" buttons if present
                try:
                    cookie_selectors = [
                        'button[contains(text(), "Accept")', 
                        'button[contains(text(), "Agree")',
                        'button[contains(text(), "Accept cookies")',
                        'button[contains(text(), "Accept all")',
                        '.cookie-consent-accept',
                        '#cookie-accept'
                    ]
                    for selector in cookie_selectors:
                        try:
                            self.selenium_driver.find_element(By.XPATH, f"//{selector}").click()
                            logger.info(f"Clicked cookie acceptance button: {selector}")
                            time.sleep(1)
                            break
                        except:
                            pass
                except Exception as e:
                    # Not critical, just log it
                    logger.debug(f"Could not handle cookie dialog: {e}")
                
                # IMPROVED: Check for and handle redirects
                try:
                    current_url = self.selenium_driver.current_url
                    if current_url != url:
                        logger.info(f"Selenium detected redirect from {url} to {current_url}")
                        # Check if the redirect is to a different domain
                        original_domain = urlparse(url).netloc
                        new_domain = urlparse(current_url).netloc
                        if original_domain != new_domain:
                            logger.info(f"Redirect changed domain from {original_domain} to {new_domain}")
                except Exception as e:
                    logger.warning(f"Error checking for Selenium redirects: {e}")
                
                # Get the rendered HTML
                body = self.selenium_driver.page_source
                
                # Return the HTML response
                return HtmlResponse(
                    url=self.selenium_driver.current_url,  # Use CURRENT URL to handle redirects
                    body=body.encode('utf-8'),
                    encoding='utf-8',
                    request=request
                )
                
            except Exception as e:
                logger.error(f"Error using Selenium for {url}: {e}")
                logger.error(traceback.format_exc())
                # Continue with standard processing
        
        # For all other requests, use normal processing
        return None


class EnhancedScrapySpider(scrapy.Spider):
    """Enhanced Spider for domain classification with improved content extraction."""
    
    name = "enhanced_domain_spider"
    
    # IMPROVED SETTINGS for better reliability and redirect handling
    custom_settings = {
        'DOWNLOAD_TIMEOUT': 60,  # INCREASED from 40
        'RETRY_TIMES': 4,        # INCREASED from 3
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429, 403],  # Added 403
        'COOKIES_ENABLED': True,  # Enable cookies for session-based sites
        'REDIRECT_MAX_TIMES': 15, # INCREASED from 10 to handle more redirects
        'REDIRECT_ENABLED': True, # Explicitly enable redirects
        'DOWNLOAD_DELAY': 0.5,  # Small delay to reduce blocking
        'HTTPERROR_ALLOW_ALL': True,  # Process pages that return errors
        'ROBOTSTXT_OBEY': False,  # Skip robots.txt check for better success rate
        'DOWNLOADER_MIDDLEWARES': {
            'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
            'scrapy.downloadermiddlewares.retry.RetryMiddleware': None,
            'domain_classifier.crawlers.scrapy_crawler.RotatingUserAgentMiddleware': 400,
            'domain_classifier.crawlers.scrapy_crawler.SmartRetryMiddleware': 550,
            'domain_classifier.crawlers.scrapy_crawler.JavaScriptMiddleware': 600,
            'scrapy.downloadermiddlewares.redirect.RedirectMiddleware': 610,  # Override to enable redirects
        },
        # NEW SETTINGS for better stability
        'CONCURRENT_REQUESTS': 4,  # Limit concurrent requests 
        'DOWNLOAD_MAXSIZE': 10485760,  # 10MB max download
        'REACTOR_THREADPOOL_MAXSIZE': 20,  # More threads
        'AJAXCRAWL_ENABLED': True,  # Better AJAX handling
        'LOG_LEVEL': 'INFO',  # More detailed logging
        # Enhanced redirect handling
        'REDIRECT_PRIORITY_ADJUST': 2,  # Give redirects a higher priority
        'METAREFRESH_ENABLED': True,   # Enable META refresh redirects
    }
    
    def __init__(self, url):
        """Initialize spider with URL."""
        self.start_urls = [url]
        self.original_url = url
        self.domain = urlparse(url).netloc
        if self.domain.startswith('www.'):
            self.domain = self.domain[4:]
        self.content_fragments = []
        self.js_required = self._check_if_js_required(url)
        self.found_content = False
        # Track redirects
        self.redirect_history = []
    
    def _check_if_js_required(self, url):
        """Check if the domain likely requires JavaScript."""
        domain = urlparse(url).netloc.lower()
        
        # Expanded list of patterns that indicate JS-heavy sites
        js_patterns = [
            # Website builders
            'wix.com', 'squarespace.com', 'webflow.com', 'shopify.com',
            'duda.co', 'weebly.com', 'godaddy.com/websites', 'wordpress.com',
            # JS frameworks
            'react', 'angular', 'vue', 'spa', 'jquery', 'gsap', 'ajax',
            # CDNs and security
            'cloudflare', 'cdn', 'akamai', 'fastly',
            # Educational and government sites
            'university', 'college', 'edu', 'school', '.gov', 
            # Common complex sites
            'dashboard', 'portal', 'app', 'login',
            # Added more domains that typically use JS
            'hubspot', 'salesforce', 'zendesk', 'freshdesk', 'clickfunnels',
            'woocommerce', 'prestashop', 'magento', 'shopify', 'bigcommerce',
            # Common site sections that often use JS
            'store', 'shop', 'showcase', 'portfolio', 'gallery'
        ]
        
        # Check if domain contains any JS-heavy patterns
        for pattern in js_patterns:
            if pattern in domain:
                return True
                
        # Also check TLDs that often host js-heavy sites
        tlds = ['.io', '.app', '.dev', '.tech', '.ai']
        for tld in tlds:
            if domain.endswith(tld):
                return True
                
        return False
    
    def start_requests(self):
        """Generate initial requests with appropriate metadata."""
        for url in self.start_urls:
            domain = urlparse(url).netloc
            
            # Special handling for known JS-heavy sites or platforms
            if self.js_required and selenium_available:
                logger.info(f"Detected JS-heavy site: {domain}. Using Selenium if available.")
                yield scrapy.Request(
                    url, 
                    callback=self.parse,
                    meta={
                        'js_render': True,
                        'domain_type': 'js_heavy',
                        'dont_redirect': False,
                        'handle_httpstatus_list': [403, 404, 500],
                        'redirect_times': 0  # Initialize redirect counter
                    }
                )
            else:
                # Standard request for most domains with enhanced redirect handling
                yield scrapy.Request(
                    url, 
                    callback=self.parse,
                    meta={
                        'dont_redirect': False,  # Allow redirects
                        'handle_httpstatus_list': [403, 404, 500],
                        'redirect_times': 0,     # Initialize redirect counter
                        'download_timeout': 60   # Longer timeout for redirects
                    }
                )
    
    def parse(self, response):
        """Parse response with improved content extraction."""
        # Track redirects
        if response.request.meta.get('redirect_urls'):
            self.redirect_history.extend(response.request.meta['redirect_urls'])
            logger.info(f"Followed redirects: {' -> '.join(response.request.meta['redirect_urls'])} -> {response.url}")
            
        # Skip non-HTML responses
        if not hasattr(response, 'text'):
            return 
        
        # Check for empty responses
        if not response.body:
            logger.warning(f"Empty response body for {response.url}")
            return {'url': response.url, 'content': '', 'is_empty': True}
        
        # Log the raw HTML length for diagnostic purposes 
        logger.info(f"Raw HTML size for {response.url}: {len(response.text)} bytes")
        
        # Extract all text content with improved methods
        content = self._extract_content(response)
        
        # Log the extracted content length
        logger.info(f"Extracted content size: {len(content)} characters")
        if len(content) > 0:
            logger.info(f"Content sample: {content[:200]}...")
        
        if len(content) > 30:  # More than minimum threshold
            self.found_content = True
        
        # Store content for this URL
        url_info = {
            'url': response.url,
            'content': content,
            'is_homepage': response.url == self.original_url or response.url in self.redirect_history,
            'raw_html': response.text[:50000] if len(content) < 30 else None,  # Store raw HTML if extraction failed
            'redirected_from': self.redirect_history if self.redirect_history else None  # Add redirect information
        }
        
        # Add content to our collection
        self.content_fragments.append(url_info)
        
        # Only follow links from homepage to avoid crawling too much
        if response.url == self.original_url or response.url in self.redirect_history:
            # Extract and follow important links
            yield from self._follow_important_links(response)
    
    def _extract_content(self, response):
        """Extract content with enhanced hierarchical approach."""
        extracted_text = []
        
        # 1. Try to get content from common container elements first
        containers_found = False
        for selector in [
            'main', 'article', '#content', '#main', '.content', '.main-content', 
            '.page-content', '.entry-content', '.post-content', '.article-content',
            'section', '.container', '.wrapper'
        ]:
            container_texts = response.css(f'{selector}::text, {selector} *::text').getall()
            clean_container_texts = [t.strip() for t in container_texts if t.strip()]
            if clean_container_texts:
                containers_found = True
                extracted_text.extend(clean_container_texts)
                logger.info(f"Found content in container: {selector} ({len(clean_container_texts)} text fragments)")
        
        # 2. If no containers found, extract structure based on common elements
        if not containers_found or len(' '.join(extracted_text)) < 100:
            # a) Extract paragraph text (most important content)
            paragraphs = response.css('p::text, p *::text').getall()
            clean_paragraphs = [p.strip() for p in paragraphs if p.strip()]
            extracted_text.extend(clean_paragraphs)
            
            # b) Extract headings
            headings = response.css('h1::text, h2::text, h3::text, h4::text, h5::text, h6::text, h1 *::text, h2 *::text, h3 *::text, h4 *::text, h5 *::text, h6 *::text').getall()
            clean_headings = [h.strip() for h in headings if h.strip()]
            extracted_text.extend(clean_headings)
            
            # c) Extract span elements (often contain important text)
            span_texts = response.css('span::text').getall()
            clean_spans = [s.strip() for s in span_texts if len(s.strip()) > 5]
            extracted_text.extend(clean_spans)
            
            # d) Extract button text (often contains action descriptions)
            button_texts = response.css('button::text, a.button::text, .btn::text').getall()
            clean_buttons = [b.strip() for b in button_texts if b.strip()]
            extracted_text.extend(clean_buttons)
            
            # e) Extract alt text from images (can contain descriptive content)
            img_alts = response.css('img::attr(alt)').getall()
            clean_alts = [alt.strip() for alt in img_alts if len(alt.strip()) > 5]
            extracted_text.extend(clean_alts)
            
            # f) Extract list items (often contain important information)
            list_items = response.css('li::text, li *::text').getall()
            clean_list_items = [li.strip() for li in list_items if li.strip()]
            extracted_text.extend(clean_list_items)
        
        # 3. Always extract meta information
        # a) Title (very important - add it with extra weight by including 3 times)
        title = response.css('title::text').get()
        if title and title.strip():
            for _ in range(3):  # Add title multiple times for extra weight
                extracted_text.append(title.strip())
        
        # b) Meta description and keywords
        meta_desc = response.css('meta[name="description"]::attr(content)').get()
        if meta_desc and meta_desc.strip():
            extracted_text.append(meta_desc.strip())
            
        meta_keywords = response.css('meta[name="keywords"]::attr(content)').get()
        if meta_keywords and meta_keywords.strip():
            extracted_text.append(meta_keywords.strip())
        
        # 4. Extract text from common content areas by class/id
        for selector in [
            '.about-us', '.mission', '.vision', '.services', '.products', 
            '.team', '.contact', '.footer', '.header', '.about', 
            '#about', '#services', '#products', '#contact',
            '.home-content', '.intro', '.hero', '.banner'
        ]:
            section_texts = response.css(f'{selector} ::text').getall()
            clean_sections = [s.strip() for s in section_texts if s.strip()]
            extracted_text.extend(clean_sections)
        
        # 5. If we still have minimal content, try a more aggressive approach - grab all visible text
        if len(' '.join(extracted_text)) < 300:
            # Select all visible text except scripts and styles
            all_texts = response.css('body *:not(script):not(style)::text').getall()
            clean_all_texts = [t.strip() for t in all_texts if t.strip()]
            extracted_text.extend(clean_all_texts)
        
        # 6. Special handling for divs if content is still minimal
        if len(' '.join(extracted_text)) < 200:
            # Look for divs with substantial text but not too big 
            # (to avoid capturing entire page in one div)
            for div in response.css('div'):
                div_text = ' '.join(div.css('::text').getall()).strip()
                if 30 < len(div_text) < 1000:  # Just right size for content
                    extracted_text.append(div_text)
        
        # Clean and join the extracted text
        all_text = ' '.join(extracted_text)
        
        # Remove excessive whitespace
        all_text = re.sub(r'\s+', ' ', all_text).strip()
        
        # Final fallback: if extraction failed completely, use simplified raw HTML
        if len(all_text) < 30 and response.text:
            logger.warning(f"Content extraction methods failed, falling back to simplified HTML for {response.url}")
            # Get raw HTML and clean it
            html = response.text
            # Basic HTML cleaning
            clean_html = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', ' ', html, flags=re.DOTALL)
            clean_html = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', ' ', clean_html, flags=re.DOTALL)
            clean_html = re.sub(r'<[^>]+>', ' ', clean_html)
            clean_html = re.sub(r'\s+', ' ', clean_html).strip()
            
            if len(clean_html) > 50:
                all_text = clean_html
                logger.info(f"Extracted {len(all_text)} characters from simplified HTML")
        
        return all_text
    
    def _follow_important_links(self, response):
        """Follow important links like About, Services pages."""
        # Enhanced list of patterns for important pages
        important_patterns = [
            'about', 'services', 'solutions', 'products', 'company',
            'what-we-do', 'technology', 'capabilities', 'team', 'mission',
            'vision', 'contact', 'portfolio', 'work', 'clients', 'testimonials',
            'partners', 'industries', 'expertise', 'case-studies', 'projects',
            # Added more important patterns
            'features', 'support', 'how-it-works', 'faq', 'overview',
            'benefits', 'our-team', 'careers', 'locations', 'history'
        ]
        
        # Extract all links
        links = response.css('a[href]')
        
        # Filter to internal links on the same domain
        same_domain_links = []
        for link in links:
            href = link.attrib['href']
            
            # Skip if it's one of these types or contains file extensions
            if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
                continue
                
            # Skip media files
            if any(href.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.mp4', '.mp3', '.css', '.js']):
                continue
                
            # Handle relative URLs
            if href.startswith('/'):
                full_url = response.urljoin(href)
                same_domain_links.append(full_url)
            else:
                # Check if link is to same domain
                try:
                    url_domain = urlparse(href).netloc
                    if not url_domain or url_domain == self.domain or url_domain == 'www.' + self.domain:
                        full_url = response.urljoin(href)
                        same_domain_links.append(full_url)
                except Exception:
                    continue
        
        # Prioritize important links
        important_links = []
        for link in same_domain_links:
            link_lower = link.lower()
            if any(pattern in link_lower for pattern in important_patterns):
                important_links.append(link)
        
        # Increase the maximum number of links to follow
        important_links = list(set(important_links))[:10]  # Increased from 5 to 10
        
        logger.info(f"Following {len(important_links)} important links")
        
        # Follow important links
        for link in important_links:
            if self.js_required and selenium_available:
                yield scrapy.Request(
                    link, 
                    callback=self.parse,
                    meta={
                        'js_render': True,
                        'domain_type': 'js_heavy',
                        'dont_redirect': False,
                        'redirect_times': 0  # Start fresh redirect counter
                    }
                )
            else:
                yield scrapy.Request(
                    link, 
                    callback=self.parse,
                    meta={
                        'dont_redirect': False,  # Allow redirects
                        'redirect_times': 0      # Start fresh redirect counter
                    }
                )
    
    def closed(self, reason):
        """Called when spider closes."""
        # Transfer content fragments to crawler results
        if self.content_fragments:
            logger.info(f"Spider closing with {len(self.content_fragments)} content fragments")
            # Signal each content fragment as a scraped item
            for fragment in self.content_fragments:
                if hasattr(self.crawler, 'signals'):
                    self.crawler.signals.send_catch_log(
                        signal=signals.item_scraped,
                        item=fragment,
                        spider=self
                    )
        else:
            logger.warning(f"Spider closing with no content fragments")


class EnhancedScrapyCrawler:
    """Enhanced Scrapy crawler for domain classification."""
    
    def __init__(self):
        """Initialize the crawler."""
        self.results = []
        self.runner = CrawlerRunner({
            # Add settings to increase reliability
            'HTTPERROR_ALLOW_ALL': True,
            'DOWNLOAD_FAIL_ON_DATALOSS': False,
            'COOKIES_ENABLED': True,
            'RETRY_ENABLED': True,
            'RETRY_TIMES': 4,  # INCREASED from 3
            'DOWNLOAD_TIMEOUT': 60,  # INCREASED from 40
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'DOWNLOADER_CLIENTCONTEXTFACTORY': 'scrapy.core.downloader.contextfactory.BrowserLikeContextFactory',
            # Improved redirect handling
            'REDIRECT_ENABLED': True,  # Explicitly enable redirects
            'REDIRECT_MAX_TIMES': 15,  # INCREASED from 10
            'METAREFRESH_ENABLED': True, # Enable meta refresh redirects
            # NEW SETTINGS for better stability
            'CONCURRENT_REQUESTS': 4,
            'DOWNLOAD_MAXSIZE': 10485760,  # 10MB
            'LOG_LEVEL': 'INFO'
        })
        dispatcher.connect(self._crawler_results, signal=signals.item_scraped)

    def _crawler_results(self, item, response=None, spider=None):
        """Collect scraped items with better handling of content fragments."""
        # Add some debugging
        logger.info(f"Received result item with keys: {list(item.keys())}")
        # Ensure we're storing content in the results list
        self.results.append(item)
        # Log the content size for debugging
        content = item.get('content', '')
        if content:
            logger.info(f"Added content to results: {len(content)} characters")
        # Log redirect info if available
        if item.get('redirected_from'):
            logger.info(f"Item was redirected from: {' -> '.join(item['redirected_from'])}")

    @crochet.wait_for(timeout=45.0)  # INCREASED from 30 to allow more time for JS rendering
    def _run_spider(self, url):
        """Run the enhanced spider."""
        return self.runner.crawl(EnhancedScrapySpider, url=url)

    def scrape(self, url):
        """Scrape a website with enhanced content extraction."""
        self.results.clear()
        try:
            logger.info(f"Starting enhanced Scrapy crawl for {url}")
            
            # Parse the domain for parked domain checking later
            domain = urlparse(url).netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Create crawler instance and scrape
            try:
                self._run_spider(url)
            except Exception as e:
                logger.error(f"Error running spider: {e}")
                logger.error(traceback.format_exc())
                return None, ("spider_error", str(e))
            
            # Parse and combine results
            combined_content = ""
            
            # Look for homepage content first
            homepage_fragments = [r for r in self.results if r.get('is_homepage', False)]
            
            if homepage_fragments:
                # Sort by content length (we want the longest content)
                homepage_fragments.sort(key=lambda x: len(x.get('content', "")), reverse=True)
                best_fragment = homepage_fragments[0]
                combined_content = best_fragment.get('content', "")
                logger.info(f"Using homepage content: {len(combined_content)} characters")
                
                # If it's empty, check if there's raw HTML we can use
                if not combined_content and best_fragment.get('raw_html'):
                    # Clean HTML
                    html = best_fragment.get('raw_html', "")
                    clean_html = re.sub(r'<script.*?>.*?</script>', ' ', html, flags=re.DOTALL)
                    clean_html = re.sub(r'<style.*?>.*?</style>', ' ', clean_html, flags=re.DOTALL)
                    clean_html = re.sub(r'<[^>]+>', ' ', clean_html)
                    clean_html = re.sub(r'\s+', ' ', clean_html).strip()
                    
                    combined_content = clean_html
                    logger.info(f"Using cleaned homepage raw HTML: {len(combined_content)} characters")
            else:
                # If no homepage fragments, combine all fragments sorted by length
                if self.results:
                    self.results.sort(key=lambda x: len(x.get('content', "")), reverse=True)
                    # Take content from top 3 fragments
                    for fragment in self.results[:3]:
                        fragment_content = fragment.get('content', "")
                        if fragment_content:
                            if combined_content:
                                combined_content += "\n\n"
                            combined_content += fragment_content
                    
                    logger.info(f"Combined content from {min(3, len(self.results))} fragments: {len(combined_content)} characters")
                
                # If no content in fragments, check raw HTML
                if not combined_content and self.results and self.results[0].get('raw_html'):
                    # Clean HTML from first result
                    html = self.results[0].get('raw_html', "")
                    clean_html = re.sub(r'<script.*?>.*?</script>', ' ', html, flags=re.DOTALL)
                    clean_html = re.sub(r'<style.*?>.*?</style>', ' ', clean_html, flags=re.DOTALL)
                    clean_html = re.sub(r'<[^>]+>', ' ', clean_html)
                    clean_html = re.sub(r'\s+', ' ', clean_html).strip()
                    
                    combined_content = clean_html
                    logger.info(f"Using cleaned raw HTML: {len(combined_content)} characters")
            
            # If we got content, check if it's a parked domain
            if combined_content:
                # Ensure combined_content is a string, not a tuple
                # FIX: This is the main issue - if combined_content is a tuple, convert to string
                if isinstance(combined_content, tuple):
                    logger.warning(f"Content is a tuple, converting to string")
                    combined_content = combined_content[0] if combined_content else ""
                
                # Now safely check if it's a parked domain
                from domain_classifier.classifiers.decision_tree import is_parked_domain
                # FIX: Ensure we're passing a string, not a tuple
                content_str = combined_content
                if is_parked_domain(content_str, domain):
                    logger.info(f"Detected parked domain for {domain}")
                    return None, ("is_parked", "Domain appears to be parked based on content analysis")
            
            return combined_content, (None, None)
            
        except Exception as e:
            from domain_classifier.crawlers.apify_crawler import detect_error_type
            error_type, error_detail = detect_error_type(str(e))
            logger.error(f"Error in Enhanced Scrapy crawler: {e}")
            logger.error(traceback.format_exc())
            return None, (error_type, error_detail)


def scrapy_crawl(url: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Crawl a website using enhanced Scrapy with better error handling.
    
    Args:
        url (str): The URL to crawl
        
    Returns:
        tuple: (content, (error_type, error_detail))
            - content: The crawled content or None if failed
            - error_type: Type of error if failed, None if successful
            - error_detail: Detailed error message if failed, None if successful
    """
    try:
        logger.info(f"Starting enhanced Scrapy crawl for {url}")
        
        # Parse the domain for parked domain checking later
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Create crawler instance and scrape
        crawler = EnhancedScrapyCrawler()
        content = crawler.scrape(url)
        
        # FIX: Handle the result properly based on what scrape() returns
        # The scrape() function returns a tuple of (content, (error_type, error_detail))
        if isinstance(content, tuple) and len(content) == 2:
            content_text, error_info = content
            
            # Log the content length for better diagnostics
            content_length = len(content_text) if content_text else 0
            logger.info(f"Scrapy crawl for {domain} returned {content_length} characters")
            
            # Check for parked domain indicators in content before proceeding
            if content_text:
                from domain_classifier.classifiers.decision_tree import is_parked_domain
                # FIX: Ensure we're passing a string, not a tuple
                content_str = content_text
                if is_parked_domain(content_str, domain):
                    logger.info(f"Detected parked domain from enhanced Scrapy content: {domain}")
                    return None, ("is_parked", "Domain appears to be parked based on content analysis")
                    
                # Check for proxy errors or hosting provider mentions that indicate parked domains
                if len(content_str.strip()) < 300 and any(phrase in content_str.lower() for phrase in 
                                                   ["proxy error", "error connecting", "godaddy", 
                                                    "domain registration", "hosting provider", "buy this domain"]):
                    logger.info(f"Domain {domain} appears to be parked based on proxy errors or hosting mentions")
                    return None, ("is_parked", "Domain appears to be parked with a domain registrar")
            
            # CRITICAL FIX: Return content even if it's minimal - Apify fallback will handle cases where truly needed
            if content_text:
                # Any content is better than falling back to Apify which takes a very long time
                logger.info(f"Enhanced Scrapy crawl successful, got {len(content_text)} characters")
                return content_text, (None, None)
            else:
                # No content at all, use the error info
                logger.warning(f"Enhanced Scrapy crawl returned no content for {domain}")
                return None, error_info
        else:
            # For backwards compatibility, handle the case where content isn't a tuple
            logger.warning(f"Unexpected return type from scrape(): {type(content)}")
            if content:
                return content, (None, None)
            else:
                return None, ("unexpected_error", "Unexpected result format from scraper")
                
    except Exception as e:
        from domain_classifier.crawlers.apify_crawler import detect_error_type
        error_type, error_detail = detect_error_type(str(e))
        logger.error(f"Error in Enhanced Scrapy crawler: {e} (Type: {error_type})")
        return None, (error_type, error_detail)


def enhanced_scrapy_crawl(url: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Alias for scrapy_crawl to be used by the apify_crawler module.
    This allows for a seamless replacement.
    
    Args:
        url (str): The URL to crawl
        
    Returns:
        tuple: Same as scrapy_crawl
    """
    return scrapy_crawl(url)


def test_scrapy_crawler(url: str) -> bool:
    """
    Test function to verify the Scrapy crawler can be called correctly.
    
    Args:
        url (str): The URL to test
        
    Returns:
        bool: True if crawling succeeded, False otherwise
    """
    try:
        logger.info(f"Testing Scrapy crawler with {url}")
        start_time = time.time()
        content, (error_type, error_detail) = scrapy_crawl(url)
        elapsed = time.time() - start_time
        
        if content:
            logger.info(f"Scrapy test succeeded - got {len(content)} characters of content in {elapsed:.2f} seconds")
            return True
        else:
            logger.warning(f"Scrapy test failed - no content. Error: {error_type} - {error_detail} (took {elapsed:.2f} seconds)")
            return False
            
    except Exception as e:
        logger.error(f"Error in Scrapy test: {e}")
        return False
