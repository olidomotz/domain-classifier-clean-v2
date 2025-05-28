"""
Go Crawler for domain classification.

This module completely replaces all existing crawlers with a Go-based crawler.
"""

import requests
import logging
import time
import json
import os
import socket
from urllib.parse import urlparse
from typing import Tuple, Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

# Go crawler API URL (can be overridden with environment variable)
CRAWLER_API_URL = os.environ.get("CRAWLER_API_URL", "http://157.245.84.110:8080/scrape")

def detect_error_type(error_message: str) -> Tuple[str, str]:
    """
    Analyze error message to determine the specific type of error.
    This is copied from the original code to maintain compatibility.
    
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

def check_domain_dns(domain: str) -> Tuple[bool, Optional[str], bool]:
    """
    Check if a domain has valid DNS resolution.
    This function helps maintain compatibility with the original code.
    
    Args:
        domain (str): The domain to check
    
    Returns:
        tuple: (has_dns, error_message, potentially_flaky)
    """
    try:
        # Remove protocol if present
        clean_domain = domain.replace('https://', '').replace('http://', '')
        
        # Remove path if present
        if '/' in clean_domain:
            clean_domain = clean_domain.split('/', 1)[0]
        
        logger.info(f"Checking DNS resolution for domain: {clean_domain}")
        socket.setdefaulttimeout(3.0)  # 3 seconds max
        
        # Try to resolve the domain
        socket.gethostbyname(clean_domain)
        
        logger.info(f"DNS resolution successful for domain: {clean_domain}")
        return True, None, False
        
    except socket.gaierror as e:
        error_code = getattr(e, 'errno', None)
        error_str = str(e).lower()
        
        if error_code == -2 or "name or service not known" in error_str:
            logger.warning(f"Domain {clean_domain} does not exist (Name not known)")
            return False, f"The domain {domain} could not be resolved. It may not exist or DNS records may be misconfigured.", False
        elif error_code == -3 or "temporary failure in name resolution" in error_str:
            logger.warning(f"Domain {clean_domain} has temporary DNS issues")
            return False, f"Temporary DNS resolution failure for {domain}. Please try again later.", True
        else:
            logger.warning(f"DNS resolution failed for {domain}: {e}")
            return False, f"The domain {domain} could not be resolved. It may not exist or DNS records may be misconfigured.", False
            
    except socket.timeout:
        logger.warning(f"DNS resolution timed out for {domain}")
        return False, f"Timed out while checking {domain}. DNS resolution timed out.", False
        
    except Exception as e:
        logger.error(f"Unexpected error checking domain {domain}: {e}")
        return False, f"Error checking {domain}: {e}", False

def is_domain_worth_crawling(domain: str) -> tuple:
    """
    Determines if a domain is worth attempting a full crawl based on preliminary checks.
    This function helps maintain compatibility with the original code.
    
    Args:
        domain (str): The domain to check
    
    Returns:
        tuple: (worth_crawling, has_dns, error_msg, potentially_flaky)
    """
    has_dns, error_msg, potentially_flaky = check_domain_dns(domain)
    
    # Special handling for DNS errors
    if not has_dns:
        logger.info(f"Domain {domain} has DNS resolution issues: {error_msg}")
        return False, False, "dns_error", False
    
    return True, has_dns, error_msg, potentially_flaky

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
                    return None, ("go_crawler_error", error_message)
                
                # Extract content
                content = data.get('content', '')
                
                # Additional metadata
                pages_crawled = data.get('pages_crawled', 0)
                word_count = data.get('word_count', 0)
                
                # Check for empty content
                if not content or (isinstance(content, str) and len(content.strip()) == 0):
                    logger.warning(f"Go crawler returned empty content for {domain}")
                    return None, ("empty_content", "The website returned no content")
                
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
        
        # Parse domain for later use
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Quick DNS check before attempting crawl
        try:
            socket.gethostbyname(domain)
        except socket.gaierror:
            logger.warning(f"Domain {domain} does not resolve - DNS error")
            return None, ("dns_error", "This domain does not exist or cannot be resolved"), "go_dns_check"
        
        # Call the Go crawler
        content, (error_type, error_detail) = go_crawl(url)
        
        # Check the result
        if content:
            logger.info(f"Go crawler successful for {domain}, got {len(content)} characters")
            return content, (None, None), "go"
        else:
            logger.warning(f"Go crawler failed for {domain}: {error_type} - {error_detail}")
            return None, (error_type, error_detail), "go_error"
            
    except Exception as e:
        logger.error(f"Error in crawl_website: {e}")
        error_type, error_detail = detect_error_type(str(e))
        return None, (error_type, error_detail), "go_exception"
