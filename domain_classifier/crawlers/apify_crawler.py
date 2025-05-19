"""Multi-crawler module for domain classification that tries Direct crawler first, then Scrapy, then Apify."""
import requests
import logging
import time
import re
import socket
from urllib.parse import urlparse
from typing import Tuple, Optional, Any

# Explicitly define what should be exported from this module
__all__ = ['crawl_website', 'detect_error_type', 'apify_crawl']

# Import settings for API keys
from domain_classifier.config.settings import APIFY_TASK_ID, APIFY_API_TOKEN

# Import Scrapy crawler
from domain_classifier.crawlers.scrapy_crawler import scrapy_crawl

# Set up logging
logger = logging.getLogger(__name__)

# Global stats for tracking crawler usage
CRAWLER_STATS = {
    "direct": 0,
    "scrapy": 0,
    "apify": 0,
    "total": 0,
    "last_reset": time.time()
}

def track_crawler_usage(crawler_type):
    """Track crawler usage statistics."""
    global CRAWLER_STATS
    
    # Update stats
    CRAWLER_STATS["total"] += 1
    
    if "direct" in crawler_type.lower():
        CRAWLER_STATS["direct"] += 1
    elif "scrapy" in crawler_type.lower():
        CRAWLER_STATS["scrapy"] += 1
    elif "apify" in crawler_type.lower():
        CRAWLER_STATS["apify"] += 1
        
    # Reset every 24 hours
    if time.time() - CRAWLER_STATS["last_reset"] > 86400:
        logger.info(f"Resetting crawler stats. Previous: {CRAWLER_STATS}")
        old_stats = CRAWLER_STATS.copy()  # Keep old stats for logging
        CRAWLER_STATS = {
            "direct": 0,
            "scrapy": 0,
            "apify": 0,
            "total": 0,
            "last_reset": time.time()
        }
        
        # Log a detailed summary of the previous stats
        direct_pct = old_stats["direct"] / max(1, old_stats["total"]) * 100
        scrapy_pct = old_stats["scrapy"] / max(1, old_stats["total"]) * 100
        apify_pct = old_stats["apify"] / max(1, old_stats["total"]) * 100
        
        logger.info(f"Crawler usage summary (past 24h): Total={old_stats['total']}")
        logger.info(f"Direct: {old_stats['direct']} ({direct_pct:.1f}%)")
        logger.info(f"Scrapy: {old_stats['scrapy']} ({scrapy_pct:.1f}%)")
        logger.info(f"Apify: {old_stats['apify']} ({apify_pct:.1f}%)")
        
    # Log current stats every 10 crawls
    if CRAWLER_STATS["total"] % 10 == 0:
        total = max(1, CRAWLER_STATS["total"])
        logger.info(f"Crawler usage stats: direct={CRAWLER_STATS['direct']} ({CRAWLER_STATS['direct']/total*100:.1f}%), "
                    f"scrapy={CRAWLER_STATS['scrapy']} ({CRAWLER_STATS['scrapy']/total*100:.1f}%), "
                    f"apify={CRAWLER_STATS['apify']} ({CRAWLER_STATS['apify']/total*100:.1f}%)")

def detect_error_type(error_message: str) -> Tuple[str, str]:
    """
    Analyze error message to determine the specific type of error.
    
    Args:
        error_message (str): The error message string
        
    Returns:
        tuple: (error_type, detailed_message)
    """
    error_message = str(error_message).lower()
    
    # SSL Certificate errors
    if any(phrase in error_message for phrase in ['certificate has expired', 'certificate verify failed', 'ssl', 'cert']):
        if 'expired' in error_message:
            return "ssl_expired", "The website's SSL certificate has expired."
        elif 'verify failed' in error_message:
            return "ssl_invalid", "The website has an invalid SSL certificate."
        else:
            return "ssl_error", "The website has SSL certificate issues."
    
    # DNS resolution errors
    elif any(phrase in error_message for phrase in ['getaddrinfo failed', 'name or service not known', 'no such host']):
        return "dns_error", "The domain could not be resolved. It may not exist or DNS records may be misconfigured."
    
    # Connection errors
    elif any(phrase in error_message for phrase in ['connection refused', 'connection timed out', 'connection error']):
        return "connection_error", "Could not establish a connection to the website. It may be down or blocking our requests."
    
    # 4XX HTTP errors
    elif any(phrase in error_message for phrase in ['403', 'forbidden', '401', 'unauthorized']):
        return "access_denied", "Access to the website was denied. The site may be blocking automated access."
    elif '404' in error_message or 'not found' in error_message:
        return "not_found", "The requested page was not found on this website."
    
    # 5XX HTTP errors
    elif any(phrase in error_message for phrase in ['500', '502', '503', '504', 'server error']):
        return "server_error", "The website is experiencing server errors."
    
    # Robots.txt or crawling restrictions
    elif any(phrase in error_message for phrase in ['robots.txt', 'disallowed', 'blocked by robots']):
        return "robots_restricted", "The website has restricted automated access in its robots.txt file."
    
    # Default fallback
    return "unknown_error", "An unknown error occurred while trying to access the website."

def quick_parked_check(url: str) -> Tuple[bool, Optional[str]]:
    """
    Perform a quick check to see if a domain is likely parked.
    
    Args:
        url (str): The URL to check
        
    Returns:
        tuple: (is_parked, content)
            - is_parked: Whether domain is parked
            - content: Any content retrieved during check
    """
    try:
        # Parse domain for checking
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
            
        # First try HTTPS request
        try:
            # Try quick direct request with HTTPS
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache'
            }
            
            # Set a low timeout to fail quickly
            response = requests.get(url, headers=headers, timeout=5.0, stream=True)
            
            # Get a chunk of content
            content = next(response.iter_content(2048), None)
            if content:
                content_str = content.decode('utf-8', errors='ignore')
                
                # Check for common parked domain indicators in this small chunk
                parked_indicators = [
                    "domain is for sale", "buy this domain", "domain parking", 
                    "parked by", "godaddy", "domain registration", "hosting provider"
                ]
                
                if any(indicator in content_str.lower() for indicator in parked_indicators):
                    logger.info(f"Quick check found parked domain indicators for {domain}")
                    return True, content_str
                    
                # If not immediately obvious from first chunk, get more content
                full_content = content_str
                try:
                    # Get up to 10KB more
                    for _ in range(5):
                        chunk = next(response.iter_content(2048), None)
                        if not chunk:
                            break
                        full_content += chunk.decode('utf-8', errors='ignore')
                    
                    from domain_classifier.classifiers.decision_tree import is_parked_domain
                    if is_parked_domain(full_content, domain):
                        logger.info(f"Quick check determined {domain} is a parked domain")
                        return True, full_content
                except Exception as e:
                    logger.warning(f"Error in additional content check: {e}")
                    
                return False, full_content
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            logger.warning(f"HTTPS failed in quick parked check: {e}, trying HTTP")
            
            # Fall back to HTTP if HTTPS fails
            try:
                http_url = url.replace("https://", "http://")
                if not http_url.startswith("http"):
                    http_url = "http://" + url
                    
                response = requests.get(http_url, headers=headers, timeout=5.0, stream=True)
                
                # Get a chunk of content
                content = next(response.iter_content(2048), None)
                if content:
                    content_str = content.decode('utf-8', errors='ignore')
                    
                    # Check for common parked domain indicators
                    parked_indicators = [
                        "domain is for sale", "buy this domain", "domain parking", 
                        "parked by", "godaddy", "domain registration", "hosting provider"
                    ]
                    
                    if any(indicator in content_str.lower() for indicator in parked_indicators):
                        logger.info(f"HTTP quick check found parked domain indicators for {domain}")
                        return True, content_str
                        
                    # If not immediately obvious, get more content
                    full_content = content_str
                    try:
                        # Get up to 10KB more
                        for _ in range(5):
                            chunk = next(response.iter_content(2048), None)
                            if not chunk:
                                break
                            full_content += chunk.decode('utf-8', errors='ignore')
                        
                        from domain_classifier.classifiers.decision_tree import is_parked_domain
                        if is_parked_domain(full_content, domain):
                            logger.info(f"HTTP quick check determined {domain} is a parked domain")
                            return True, full_content
                    except Exception as e:
                        logger.warning(f"Error in HTTP additional content check: {e}")
                        
                    return False, full_content
            except Exception as http_e:
                logger.warning(f"HTTP also failed in quick parked check: {http_e}")
        
        return False, None
        
    except Exception as e:
        logger.warning(f"Quick parked domain check failed: {e}")
        return False, None

def crawl_website(url: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]], Optional[str]]:
    """
    IMPROVED: Crawl a website using Direct Crawler first, then ALWAYS try Scrapy before falling back to Apify.
    This version ensures proper crawler priority and content selection.
    
    Args:
        url (str): The URL to crawl
        
    Returns:
        tuple: (content, (error_type, error_detail), crawler_type)
            - content: The crawled content or None if failed
            - error_type: Type of error if failed, None if successful
            - error_detail: Detailed error message if failed, None if successful
            - crawler_type: The type of crawler used ("direct", "scrapy", "apify", etc.)
    """
    try:
        logger.info(f"Starting crawl for {url}")
        
        # Parse domain for later use
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Quick DNS check before attempting full crawl
        try:
            socket.gethostbyname(domain)
        except socket.gaierror:
            logger.warning(f"Domain {domain} does not resolve - DNS error")
            return None, ("dns_error", "This domain does not exist or cannot be resolved"), None
        
        # Perform a quick check for parked domains before full crawl
        is_parked, quick_content = quick_parked_check(url)
        if is_parked:
            logger.info(f"Quick check identified {domain} as a parked domain")
            return None, ("is_parked", "Domain appears to be parked based on quick check"), "quick_check_parked"
        
        # Variables to track direct crawl results
        direct_content = None
        direct_error_type = None
        direct_error_detail = None
        direct_crawler_type = None
        
        # Try both HTTP and HTTPS with direct crawler first
        from domain_classifier.crawlers.direct_crawler import try_multiple_protocols
        logger.info(f"Trying multiple protocols for {domain}")
        
        direct_content, (direct_error_type, direct_error_detail), direct_crawler_type = try_multiple_protocols(domain)
        
        # For minimal content with direct crawler, check if it's a parked domain
        if direct_content and len(direct_content.strip()) <= 300:
            from domain_classifier.classifiers.decision_tree import is_parked_domain
            if is_parked_domain(direct_content, domain):
                logger.info(f"Detected parked domain from direct crawl content: {domain}")
                return None, ("is_parked", "Domain appears to be parked based on content analysis"), "direct_parked_domain"
        
        # Save direct crawl results for later comparison
        direct_content_length = len(direct_content.strip()) if direct_content else 0
        
        # Track direct crawler usage
        if direct_content:
            track_crawler_usage(f"direct_{direct_crawler_type}")
        
        # CRITICAL FIX: ALWAYS try Scrapy next, regardless of direct crawl results
        logger.info(f"ALWAYS trying Scrapy for {url}, direct crawl got {direct_content_length} chars")
        scrapy_content, (scrapy_error_type, scrapy_error_detail) = scrapy_crawl(url)
        scrapy_content_length = len(scrapy_content.strip()) if scrapy_content else 0
        
        # Log detailed debug info
        logger.info(f"Crawler comparison for {domain}: direct={direct_content_length} chars, scrapy={scrapy_content_length} chars")
        
        # Check for parked domain indicators in Scrapy content
        if scrapy_content:
            from domain_classifier.classifiers.decision_tree import is_parked_domain
            if is_parked_domain(scrapy_content, domain):
                logger.info(f"Detected parked domain from Scrapy content: {domain}")
                return None, ("is_parked", "Domain appears to be parked based on content analysis"), "scrapy_parked_domain"
        
        # IMPROVED DECISION LOGIC:
        # 1. If Scrapy got good content (>150 chars), use it
        if scrapy_content and scrapy_content_length > 150:
            logger.info(f"Using Scrapy content: {scrapy_content_length} chars")
            track_crawler_usage("scrapy_primary")
            return scrapy_content, (None, None), "scrapy"
        
        # 2. If direct got very good content (>500 chars), use it
        if direct_content and direct_content_length > 500:
            logger.info(f"Using direct content: {direct_content_length} chars")
            return direct_content, (None, None), direct_crawler_type
        
        # 3. If both got some content, use the longer one
        if scrapy_content and direct_content:
            if scrapy_content_length >= direct_content_length:
                logger.info(f"Using Scrapy content ({scrapy_content_length} chars) over Direct content ({direct_content_length} chars)")
                track_crawler_usage("scrapy_preferred")
                return scrapy_content, (None, None), "scrapy_preferred"
            else:
                logger.info(f"Using Direct content ({direct_content_length} chars) over Scrapy content ({scrapy_content_length} chars)")
                return direct_content, (None, None), direct_crawler_type
        
        # 4. If only one got any content at all, use it
        if scrapy_content and scrapy_content_length > 0:
            logger.info(f"Using minimal Scrapy content ({scrapy_content_length} chars) as only option")
            track_crawler_usage("scrapy_minimal")
            return scrapy_content, (None, None), "scrapy_minimal"
            
        if direct_content and direct_content_length > 0:
            logger.info(f"Using minimal Direct content ({direct_content_length} chars) as only option")
            return direct_content, (None, None), direct_crawler_type
        
        # 5. ONLY if both direct and Scrapy completely failed, try Apify as absolute last resort
        logger.info(f"Both Direct and Scrapy failed for {url}, trying Apify fallback")
        content, (error_type, error_detail) = apify_crawl(url, timeout=40)  # Reduced timeout to speed up processing
        
        # Final check for parked domain with Apify content
        if content:
            from domain_classifier.classifiers.decision_tree import is_parked_domain
            if is_parked_domain(content, domain):
                logger.info(f"Detected parked domain from Apify content: {domain}")
                return None, ("is_parked", "Domain appears to be parked based on content analysis"), "apify_parked_domain"
                
        if content and len(content.strip()) > 100:
            logger.info(f"Apify crawl successful for {domain}, got {len(content)} characters")
            track_crawler_usage("apify_last_resort")  # Mark clearly as last resort
            return content, (None, None), "apify_last_resort"
        elif content:
            logger.info(f"Apify got minimal content for {domain}, using it as last resort")
            track_crawler_usage("apify_minimal")
            return content, (None, None), "apify_minimal"
        else:
            # Last attempt - try direct HTTP crawl with higher timeout
            try:
                http_url = "http://" + domain
                logger.info(f"Final attempt: HTTP direct crawl with higher timeout for {http_url}")
                from domain_classifier.crawlers.direct_crawler import direct_crawl
                content, (final_error_type, final_error_detail), final_crawler_type = direct_crawl(http_url, timeout=15.0)
                
                if content and len(content.strip()) > 100:
                    logger.info(f"Final HTTP attempt successful for {domain}")
                    track_crawler_usage("direct_http_final")
                    return content, (None, None), "direct_http_final"
            except Exception as final_e:
                logger.warning(f"Final HTTP attempt failed: {final_e}")
                
            return None, (error_type or direct_error_type, error_detail or direct_error_detail), None
            
    except Exception as e:
        error_type, error_detail = detect_error_type(str(e))
        logger.error(f"Error crawling website: {e} (Type: {error_type})")
        return None, (error_type, error_detail), None

def apify_crawl(url: str, timeout: int = 60) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Crawl a website using Apify with improved multi-stage approach for JavaScript-heavy sites.
    
    Args:
        url: The URL to crawl
        timeout: Maximum time to wait for crawl to complete (seconds)
        
    Returns:
        tuple: (content, (error_type, error_detail))
            - content: The crawled content or None if failed
            - error_type: Type of error if failed, None if successful
            - error_detail: Detailed error message if failed, None if successful
    """
    try:
        logger.info(f"Starting Apify crawl for {url}")
        track_crawler_usage("apify_started")
        
        # Extract domain for parked domain checks
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Start the crawl with standard settings
        endpoint = f"https://api.apify.com/v2/actor-tasks/{APIFY_TASK_ID}/runs?token={APIFY_API_TOKEN}"
        payload = {
            "startUrls": [{"url": url}],
            "maxCrawlingDepth": 1,
            "maxCrawlPages": 5,
            "timeoutSecs": 30  # Reduced timeout to avoid lengthy processing
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=15)  # Reduced timeout
            response.raise_for_status()
            run_id = response.json()['data']['id']
            logger.info(f"Successfully started Apify run with ID: {run_id}")
        except Exception as e:
            logger.error(f"Error starting Apify crawl: {e}")
            return None, detect_error_type(str(e))
            
        # Wait for crawl to complete
        endpoint = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={APIFY_API_TOKEN}"
        
        max_attempts = 4  # Reduced attempts to avoid long waits (about 40 seconds max)
        for attempt in range(max_attempts):
            logger.info(f"Checking Apify crawl results, attempt {attempt+1}/{max_attempts}")
            
            try:
                response = requests.get(endpoint, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data:
                        combined_text = ' '.join(item.get('text', '') for item in data if item.get('text'))
                        
                        # Check if this is a parked domain before continuing
                        if combined_text:
                            from domain_classifier.classifiers.decision_tree import is_parked_domain
                            if is_parked_domain(combined_text, domain):
                                logger.info(f"Detected parked domain during Apify crawl: {domain}")
                                return None, ("is_parked", "Domain appears to be parked based on content analysis")
                        
                        if combined_text and len(combined_text.strip()) > 100:
                            logger.info(f"Apify crawl completed, got {len(combined_text)} characters")
                            return combined_text, (None, None)
                        elif combined_text:
                            logger.warning(f"Apify crawl returned minimal content: {len(combined_text)} characters")
                            # Continue trying, might get better results on next attempt
                else:
                    logger.warning(f"Received status code {response.status_code} when checking Apify crawl results")
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout when checking Apify status (attempt {attempt+1})")
            except Exception as e:
                logger.warning(f"Error checking Apify status: {e}")
            
            # If we're on the early checks, try shorter waits
            if attempt < 3:
                time.sleep(5)
            else:
                time.sleep(10)
                
            # After 3 attempts, try a different approach (around 30 seconds in)
            if attempt == 3:
                logger.info(f"Trying Puppeteer-based approach for JavaScript-heavy site...")
                try:
                    # Try puppeteer-based approach
                    puppeteer_url = f"https://api.apify.com/v2/actor-runs/{run_id}/?token={APIFY_API_TOKEN}"
                    response = requests.post(puppeteer_url, timeout=10)
                    
                    if response.status_code == 200:
                        logger.info("Puppeteer request successful, continuing with checks")
                    else:
                        logger.error(f"Error with Puppeteer approach: {response.status_code} {response.reason} for url: {puppeteer_url}")
                except Exception as puppeteer_error:
                    logger.error(f"Error with Puppeteer approach: {puppeteer_error}")
        
        # If we've reached half our attempts, try direct request
        # This happens around 30-40 seconds in
        logger.info(f"Trying direct request fallback...")
        from domain_classifier.crawlers.direct_crawler import direct_crawl
        direct_content, (error_type, error_detail), crawler_type = direct_crawl(url, timeout=15.0)
        
        # If we got direct content, use it
        if direct_content and len(direct_content) > 100:
            logger.info(f"Direct request fallback got {len(direct_content)} characters")
            track_crawler_usage("direct_fallback_from_apify")
            return direct_content, (None, None)
                
        # Critical Fix: Try HTTP version explicitly
        http_url = url.replace('https://', 'http://')
        if not http_url.startswith('http'):
            http_url = 'http://' + url
            
        logger.info(f"Trying final HTTP direct crawl for {http_url}")
        direct_content, (error_type, error_detail), crawler_type = direct_crawl(http_url, timeout=15.0)
        
        # If direct HTTP request worked, use it
        if direct_content and len(direct_content) > 100:
            logger.info(f"HTTP direct request got {len(direct_content)} characters")
            track_crawler_usage("http_direct_fallback_from_apify")
            return direct_content, (None, None)
        
        # If we get to this point, we've tried everything
        logger.warning(f"Crawl timed out after all attempts")
        return None, ("timeout", "The website took too long to respond or has minimal crawlable content.")
    except Exception as e:
        error_type, error_detail = detect_error_type(str(e))
        logger.error(f"Error crawling with Apify: {e} (Type: {error_type})")
        return None, (error_type, error_detail)
