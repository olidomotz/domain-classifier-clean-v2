"""
Go Crawler for domain classification.

This module replaces all existing crawlers with a Go-based crawler
while maintaining compatibility with the rest of the system.
"""

import requests
import logging
import time
import json
import os
import socket
import traceback
import re
from urllib.parse import urlparse
from typing import Tuple, Optional, Dict, Any, List

# Set up logging
logger = logging.getLogger(__name__)

# Go crawler API URL (can be overridden with environment variable)
CRAWLER_API_URL = os.environ.get("CRAWLER_API_URL", "http://157.245.84.110:8080/scrape")

# Global stats for tracking crawler usage (maintaining compatibility with original code)
CRAWLER_STATS = {
    "direct": 0,
    "scrapy": 0,
    "apify": 0,
    "go": 0,  # Add Go crawler to statistics
    "total": 0,
    "last_reset": time.time()
}

def track_crawler_usage(crawler_type):
    """Track crawler usage statistics."""
    global CRAWLER_STATS
    
    # Update stats
    CRAWLER_STATS["total"] += 1
    
    if "go" in crawler_type.lower():
        CRAWLER_STATS["go"] += 1
    elif "direct" in crawler_type.lower():
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
            "go": 0,
            "total": 0,
            "last_reset": time.time()
        }
        
        # Log a detailed summary of the previous stats
        direct_pct = old_stats["direct"] / max(1, old_stats["total"]) * 100
        scrapy_pct = old_stats["scrapy"] / max(1, old_stats["total"]) * 100
        apify_pct = old_stats["apify"] / max(1, old_stats["total"]) * 100
        go_pct = old_stats["go"] / max(1, old_stats["total"]) * 100
        
        logger.info(f"Crawler usage summary (past 24h): Total={old_stats['total']}")
        logger.info(f"Direct: {old_stats['direct']} ({direct_pct:.1f}%)")
        logger.info(f"Scrapy: {old_stats['scrapy']} ({scrapy_pct:.1f}%)")
        logger.info(f"Apify: {old_stats['apify']} ({apify_pct:.1f}%)")
        logger.info(f"Go: {old_stats['go']} ({go_pct:.1f}%)")

    # Log current stats every 10 crawls
    if CRAWLER_STATS["total"] % 10 == 0:
        total = max(1, CRAWLER_STATS["total"])
        logger.info(f"Crawler usage stats: "
                   f"go={CRAWLER_STATS['go']} ({CRAWLER_STATS['go']/total*100:.1f}%), "
                   f"direct={CRAWLER_STATS['direct']} ({CRAWLER_STATS['direct']/total*100:.1f}%), "
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
    
    # Remote disconnect detection (anti-scraping)
    if any(phrase in error_message for phrase in ['remotedisconnected', 'remote end closed', 'connection reset', 'connection aborted']):
        return "anti_scraping", "The website appears to be using anti-scraping protection."
        
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

def go_crawl(url: str, timeout: int = 30) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Crawl a website using the Go-based crawler.
    
    Args:
        url (str): The URL to crawl
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
    
    Returns:
        tuple: (content, (error_type, error_detail))
        - content: The crawled content or None if failed
        - error_type: Type of error if failed, None if successful
        - error_detail: Detailed error message if failed, None if successful
    """
    try:
        logger.info(f"Starting Go crawler for {url}")
        track_crawler_usage("go_started")
        
        # Extract domain for API call
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Create a session with retry capabilities
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=requests.adapters.Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504]
            )
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        # Prepare the request
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "domain": domain
        }
        
        # Make the request to Go crawler API
        logger.info(f"Sending request to Go crawler API for domain: {domain}")
        start_time = time.time()
        
        response = session.post(
            CRAWLER_API_URL, 
            json=payload, 
            headers=headers, 
            timeout=timeout
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Go crawler API request completed in {elapsed_time:.2f} seconds")
        
        # Handle response
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Check for success indicator
                if data.get('success', False) is False:
                    error_message = data.get('error', 'Unknown error')
                    logger.warning(f"Go crawler returned error for {domain}: {error_message}")
                    
                    # Map error type from Go crawler to what the system expects
                    error_type = data.get('error_type', "go_crawler_error")
                    
                    # Standardize error types to match what the system expects
                    if "dns" in error_type.lower() or "no such host" in error_message.lower():
                        error_type = "dns_error"
                    elif "ssl" in error_type.lower() or "certificate" in error_message.lower():
                        error_type = "ssl_error"
                    elif "timeout" in error_type.lower():
                        error_type = "timeout"
                    elif "forbidden" in error_message.lower() or "403" in error_message:
                        error_type = "access_denied"
                    elif "not found" in error_message.lower() or "404" in error_message:
                        error_type = "not_found"
                    
                    return None, (error_type, error_message)
                
                # Extract content
                content = data.get('content', '')
                
                # Additional metadata
                pages_crawled = data.get('pages_crawled', 0)
                word_count = data.get('word_count', 0)
                
                # Handle empty content
                if not content or (isinstance(content, str) and len(content.strip()) == 0):
                    logger.warning(f"Go crawler returned empty content for {domain}")
                    return None, ("empty_content", "The website returned no content")
                
                # Check for parked domains in the crawler result
                is_parked = data.get('is_parked', False)
                if is_parked:
                    logger.info(f"Go crawler detected parked domain: {domain}")
                    return None, ("is_parked", "Domain appears to be parked based on Go crawler analysis")
                
                # Double-check for parked domain with local detection
                from domain_classifier.classifiers.decision_tree import is_parked_domain
                if content and isinstance(content, str) and is_parked_domain(content, domain):
                    logger.info(f"Local check detected parked domain for {domain}")
                    return None, ("is_parked", "Domain appears to be parked based on content analysis")
                
                # Log success
                logger.info(f"Go crawler successful for {domain}: {pages_crawled} pages, {word_count} words, {len(content)} chars")
                return content, (None, None)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Go crawler response: {e}")
                return None, ("response_error", f"Invalid JSON response from Go crawler: {e}")
                
            except Exception as e:
                logger.error(f"Error processing Go crawler response: {e}")
                return None, ("processing_error", f"Error processing Go crawler response: {e}")
                
        else:
            logger.error(f"Go crawler API returned status code {response.status_code}")
            return None, ("http_error", f"Go crawler API returned status code {response.status_code}")
            
    except requests.exceptions.Timeout:
        logger.error(f"Timeout connecting to Go crawler API after {timeout} seconds")
        return None, ("timeout", f"The Go crawler API took too long to respond (> {timeout} seconds)")
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error with Go crawler API: {e}")
        return None, ("connection_error", f"Could not connect to Go crawler API: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error with Go crawler: {e}")
        error_type, error_detail = detect_error_type(str(e))
        return None, (error_type, error_detail)

def crawl_website(url: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]], Optional[str]]:
    """
    Main function to crawl a website. This replaces the original crawler chain with
    the Go crawler while maintaining compatibility with the rest of the system.
    
    Args:
        url: The URL to crawl
    
    Returns:
        tuple: (content, (error_type, error_detail), crawler_type)
        - content: The crawled content or None if failed
        - error_type: Type of error if failed, None if successful
        - error_detail: Detailed error message if failed, None if successful
        - crawler_type: The type of crawler used (always "go" in this case)
    """
    try:
        logger.info(f"Starting crawl for {url}")
        
        # Parse domain for later use
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # CRITICAL: Comprehensive DNS check before attempting to crawl
        # This is essential for proper classification of non-existent domains
        try:
            dns_start = time.time()
            socket.setdefaulttimeout(3.0)  # 3 seconds max for DNS
            
            # Try both with and without www
            dns_errors = []
            domains_to_check = [domain]
            if not domain.startswith('www.'):
                domains_to_check.append('www.' + domain)
            
            for d in domains_to_check:
                try:
                    socket.gethostbyname(d)
                    logger.info(f"DNS resolution successful for {d}")
                    break
                except socket.gaierror as e:
                    error_code = getattr(e, 'errno', None)
                    error_str = str(e).lower()
                    dns_errors.append((error_code, error_str))
                    continue
            
            # If all checks failed, report DNS error
            if len(dns_errors) == len(domains_to_check):
                logger.warning(f"DNS resolution failed for {domain} and www.{domain}")
                
                # Get the first error for reporting
                error_code, error_str = dns_errors[0]
                
                if error_code == -2 or "name or service not known" in error_str:
                    logger.warning(f"Domain {domain} does not exist (Name not known)")
                    return None, ("dns_error", f"The domain {domain} could not be resolved. It may not exist."), "dns_check"
                elif error_code == -3 or "temporary failure in name resolution" in error_str:
                    logger.warning(f"Domain {domain} has temporary DNS issues")
                    return None, ("dns_error", f"Temporary DNS resolution failure for {domain}."), "dns_check"
                else:
                    logger.warning(f"DNS error for {domain}: {error_str}")
                    return None, ("dns_error", f"The domain {domain} could not be resolved."), "dns_check"
            
            logger.info(f"DNS check completed in {time.time() - dns_start:.2f}s")
            
        except socket.timeout:
            logger.warning(f"DNS resolution timed out for {domain}")
            return None, ("dns_error", f"DNS resolution timed out for {domain}"), "dns_check"
        except Exception as dns_err:
            logger.error(f"Error during DNS check: {dns_err}")
            return None, ("dns_error", f"Error checking DNS for {domain}: {dns_err}"), "dns_check"
        
        # CRITICAL: Quick check for parked domains before full crawl
        is_parked, quick_content = quick_parked_check(url)
        if is_parked:
            logger.info(f"Quick check identified {domain} as a parked domain")
            return None, ("is_parked", "Domain appears to be parked based on quick check"), "parked_check"
        
        # Call the Go crawler for the main content
        content, (error_type, error_detail) = go_crawl(url)
        
        # Track the crawler usage
        track_crawler_usage("go_crawler")
        
        # Process the result
        if content:
            # Check content length
            content_length = len(content.strip())
            logger.info(f"Go crawler got {content_length} chars for {domain}")
            
            # Make a final check for parked domain with the complete content
            if content_length < 5000:  # Only check smaller content to avoid false positives
                from domain_classifier.classifiers.decision_tree import is_parked_domain
                if is_parked_domain(content, domain):
                    logger.info(f"Content analysis detected parked domain for {domain}")
                    return None, ("is_parked", "Domain appears to be parked based on content analysis"), "go_parked_detection"
            
            return content, (None, None), "go"
        else:
            # Handle specific error types
            logger.warning(f"Go crawler failed for {domain}: {error_type} - {error_detail}")
            
            # Make sure error type is one the system recognizes
            standardized_error_type = error_type
            
            # The following error types are specifically checked by the system
            if error_type not in ["dns_error", "ssl_error", "ssl_expired", "ssl_invalid", 
                                "connection_error", "timeout", "access_denied", "not_found", 
                                "server_error", "robots_restricted", "is_parked"]:
                # Map unknown error types to something the system understands
                if "dns" in error_type.lower():
                    standardized_error_type = "dns_error"
                elif "ssl" in error_type.lower() or "certificate" in error_type.lower():
                    standardized_error_type = "ssl_error"
                elif "timeout" in error_type.lower():
                    standardized_error_type = "timeout"
                elif "403" in error_type.lower() or "forbidden" in error_type.lower():
                    standardized_error_type = "access_denied"
                elif "404" in error_type.lower() or "not found" in error_type.lower():
                    standardized_error_type = "not_found"
                elif "parked" in error_type.lower():
                    standardized_error_type = "is_parked"
                else:
                    standardized_error_type = "unknown_error"
            
            return None, (standardized_error_type, error_detail), "go_error"
            
    except Exception as e:
        logger.error(f"Error in crawl_website: {e}")
        logger.error(traceback.format_exc())
        error_type, error_detail = detect_error_type(str(e))
        return None, (error_type, error_detail), "go_exception"

def quick_parked_check(url: str) -> Tuple[bool, Optional[str]]:
    """
    Perform a quick check to see if a domain is likely parked.
    This implementation checks using both Go crawler's is_parked flag
    and our local is_parked_domain function if content is available.
    
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
        
        # First try a quick crawl with the Go crawler
        logger.info(f"Performing quick parked domain check for {domain}")
        
        # Call Go crawler with a short timeout for quick check
        content, (error_type, error_detail) = go_crawl(url, timeout=10)
        
        # If Go crawler explicitly reports it as parked
        if error_type == "is_parked":
            logger.info(f"Go crawler detected {domain} as parked domain")
            return True, None
        
        # If we got content, do a local check
        if content:
            # Use the decision tree's is_parked_domain function
            from domain_classifier.classifiers.decision_tree import is_parked_domain
            if is_parked_domain(content, domain):
                logger.info(f"Local check detected {domain} as parked domain")
                return True, content
            
            # If not parked, return the content for potential reuse
            return False, content
        
        # No content but also not explicitly marked as parked
        return False, None
        
    except Exception as e:
        logger.warning(f"Error in quick_parked_check for {url}: {e}")
        return False, None

# Compatibility functions - required by the rest of the system
def apify_crawl(url: str, timeout: int = 30) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Legacy function for backward compatibility.
    Redirects to the Go crawler implementation.
    """
    logger.info(f"Redirecting apify_crawl to go_crawl for {url}")
    return go_crawl(url, timeout)

def scrapy_crawl(url: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Legacy function for backward compatibility.
    Redirects to the Go crawler implementation.
    """
    logger.info(f"Redirecting scrapy_crawl to go_crawl for {url}")
    return go_crawl(url)

def try_multiple_protocols(domain: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]], Optional[str]]:
    """
    Legacy function for backward compatibility.
    Redirects to the Go crawler implementation.
    """
    url = f"https://{domain}"
    logger.info(f"Redirecting try_multiple_protocols to go_crawl for {domain}")
    content, error_info = go_crawl(url)
    return content, error_info, "go_multiple_protocols"
