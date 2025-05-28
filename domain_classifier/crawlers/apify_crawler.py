"""
Go Crawler for domain classification.

This module completely replaces all existing crawlers with a Go-based crawler
and removes duplicate parked domain and SSL checks.
"""

import requests
import logging
import time
import json
import os
import socket
import traceback
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
    This is kept from the original code to maintain compatibility.
    
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
        
        # Extract domain for API call
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
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
        
        response = requests.post(
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
                    
                    # Map error type if provided
                    error_type = data.get('error_type', "go_crawler_error")
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
                    return None, ("is_parked", "Domain appears to be parked based on Go crawler analysis"), 
                
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
        return None, ("unknown_error", f"Unexpected error: {e}")

def crawl_website(url: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]], Optional[str]]:
    """
    Crawl a website using the Go-based crawler.
    This function replaces the original crawl_website in apify_crawler.py.
    
    Args:
        url: The URL to crawl
    
    Returns:
        tuple: (content, (error_type, error_detail), crawler_type)
        - content: The crawled content or None if failed
        - error_type: Type of error if failed, None if successful
        - error_detail: Detailed error message if failed, None if successful
        - crawler_type: The type of crawler used ("go")
    """
    try:
        logger.info(f"Starting crawl for {url} with Go crawler")
        
        # Parse domain for DNS check
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Minimal DNS check - we still need this to avoid wasting API calls on non-existent domains
        try:
            socket.gethostbyname(domain)
        except socket.gaierror:
            logger.warning(f"Domain {domain} does not resolve - DNS error")
            return None, ("dns_error", "This domain does not exist or cannot be resolved"), "go_dns_check"
        
        # Call the Go crawler - all other checks (SSL, parked domains, etc.) are handled by the Go crawler
        content, (error_type, error_detail) = go_crawl(url)
        
        # Track the usage
        track_crawler_usage("go")
        
        # Check the result
        if content:
            logger.info(f"Go crawler successful for {domain}, got {len(content)} characters")
            return content, (None, None), "go"
        else:
            logger.warning(f"Go crawler failed for {domain}: {error_type} - {error_detail}")
            return None, (error_type, error_detail), "go_error"
            
    except Exception as e:
        logger.error(f"Error in crawl_website: {e}")
        logger.error(traceback.format_exc())
        error_type, error_detail = detect_error_type(str(e))
        return None, (error_type, error_detail), "go_exception"

# Required for compatibility with the original module
def quick_parked_check(url: str) -> Tuple[bool, Optional[str]]:
    """
    Perform a quick check to see if a domain is likely parked.
    This is a stub that delegates to the Go crawler and is kept for compatibility.
    """
    # Extract domain for API call
    domain = urlparse(url).netloc
    if domain.startswith('www.'):
        domain = domain[4:]
        
    logger.info(f"Delegating parked domain check for {domain} to Go crawler")
    
    # Call the Go crawler with a quick timeout
    try:
        content, (error_type, error_detail) = go_crawl(url, timeout=10)
        
        # If the Go crawler identifies it as a parked domain
        if error_type == "is_parked":
            return True, error_detail
            
        # If we got content, it's not parked
        if content:
            return False, content
            
        # If we got an error but not a parked domain error, it's not confirmed parked
        return False, None
        
    except Exception as e:
        logger.warning(f"Error in parked domain check: {e}")
        return False, None

# Compatibility functions - stubs that just use Go crawler
def apify_crawl(url: str, timeout: int = 30) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Compatibility function for apify_crawl. Uses Go crawler instead.
    """
    logger.info(f"Using Go crawler instead of Apify for {url}")
    return go_crawl(url, timeout)

# Required to maintain compatibility with original code
def scrapy_crawl(url: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Compatibility function that redirects to Go crawler.
    This allows importing this function from other modules.
    """
    logger.info(f"Using Go crawler instead of Scrapy for {url}")
    return go_crawl(url)
