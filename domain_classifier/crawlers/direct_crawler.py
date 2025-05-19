"""Direct crawler for domain classification as a fallback."""
import requests
import logging
import re
from urllib.parse import urlparse
from typing import Tuple, Optional

# Set up logging
logger = logging.getLogger(__name__)

def direct_crawl(url: str, timeout: float = 15.0) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]], Optional[str]]:
    """
    Directly crawl a website using a simple GET request as a fallback method.
    
    Args:
        url (str): The URL to crawl
        timeout (float): Timeout for the request in seconds
        
    Returns:
        tuple: (content, (error_type, error_detail), crawler_type)
            - content: The crawled content or None if failed
            - error_type: Type of error if failed, None if successful
            - error_detail: Detailed error message if failed, None if successful
            - crawler_type: Always "direct" if successful
    """
    try:
        logger.info(f"Attempting direct crawl for {url} with timeout {timeout}s")
        
        # Ensure URL is properly formatted
        if not url.startswith('http'):
            url = 'https://' + url
            
        # Set up headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        # Parse the domain for parked domain checking later
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # First try HTTPS version
        https_success = False
        http_success = False
        https_content = None
        http_content = None
        
        try:
            logger.info(f"Trying HTTPS for {domain}")
            https_response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            https_response.raise_for_status()
            
            # Extract content type
            content_type = https_response.headers.get('Content-Type', '').lower()
            
            # For quick parked domain checks, just get the first chunk
            if timeout < 10:
                content = next(https_response.iter_content(4096), None)
                if content:
                    text = content.decode('utf-8', errors='ignore')
                    
                    # Check immediately for parked domain indicators
                    from domain_classifier.classifiers.decision_tree import is_parked_domain
                    if is_parked_domain(text, domain):
                        logger.info(f"Direct crawl detected a parked domain: {domain}")
                        return text, ("is_parked", "Domain appears to be parked"), "direct_parked_detection"
                    
                    https_content = text
                    https_success = True
                    return https_content, (None, None), "direct_https_quick"
            
            # Handle different content types for full crawling
            if 'text/html' in content_type or 'text/plain' in content_type or 'application/xhtml' in content_type:
                # Extract readable text by removing HTML tags
                html_content = https_response.text
                clean_text = re.sub(r'<script.*?>.*?</script>', ' ', html_content, flags=re.DOTALL)
                clean_text = re.sub(r'<style.*?>.*?</style>', ' ', clean_text, flags=re.DOTALL)
                clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                
                # Check for parked domain indicators
                from domain_classifier.classifiers.decision_tree import is_parked_domain
                if is_parked_domain(html_content, domain):
                    logger.info(f"Direct crawl identified {domain} as a parked domain")
                    return html_content, ("is_parked", "Domain appears to be parked"), "direct_parked_detection"
                
                https_content = clean_text
                https_success = True
                if clean_text and len(clean_text) > 100:
                    logger.info(f"HTTPS direct crawl successful, got {len(clean_text)} characters")
                    return clean_text, (None, None), "direct_https"
                else:
                    logger.warning(f"HTTPS direct crawl returned minimal content: {len(clean_text)} characters")
                    # Don't return here, try HTTP as well
            else:
                logger.warning(f"HTTPS direct crawl returned non-text content: {content_type}")
        
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            logger.warning(f"HTTPS failed for {domain}, trying HTTP: {e}")

        # If HTTPS failed or returned minimal content, try HTTP
        try:
            http_url = url.replace('https://', 'http://')
            logger.info(f"Trying HTTP for {domain}: {http_url}")
            http_response = requests.get(http_url, headers=headers, timeout=timeout, stream=True)
            http_response.raise_for_status()
            
            # Extract content type
            content_type = http_response.headers.get('Content-Type', '').lower()
            
            # For quick parked domain checks, just get the first chunk
            if timeout < 10:
                content = next(http_response.iter_content(4096), None)
                if content:
                    text = content.decode('utf-8', errors='ignore')
                    
                    # Check immediately for parked domain indicators
                    from domain_classifier.classifiers.decision_tree import is_parked_domain
                    if is_parked_domain(text, domain):
                        logger.info(f"HTTP direct crawl detected a parked domain: {domain}")
                        return text, ("is_parked", "Domain appears to be parked"), "direct_http_parked_detection"
                    
                    http_content = text
                    http_success = True
                    return http_content, (None, None), "direct_http_quick"
            
            # Handle different content types for full crawling
            if 'text/html' in content_type or 'text/plain' in content_type or 'application/xhtml' in content_type:
                # Extract readable text by removing HTML tags
                html_content = http_response.text
                clean_text = re.sub(r'<script.*?>.*?</script>', ' ', html_content, flags=re.DOTALL)
                clean_text = re.sub(r'<style.*?>.*?</style>', ' ', clean_text, flags=re.DOTALL)
                clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                
                # Check for parked domain indicators
                from domain_classifier.classifiers.decision_tree import is_parked_domain
                if is_parked_domain(html_content, domain):
                    logger.info(f"HTTP direct crawl identified {domain} as a parked domain")
                    return html_content, ("is_parked", "Domain appears to be parked"), "direct_http_parked_detection"
                
                http_content = clean_text
                http_success = True
                if clean_text and len(clean_text) > 100:
                    logger.info(f"HTTP direct crawl successful, got {len(clean_text)} characters")
                    return clean_text, (None, None), "direct_http"
                else:
                    logger.warning(f"HTTP direct crawl returned minimal content: {len(clean_text)} characters")
                    # Continue with fallback logic
            else:
                logger.warning(f"HTTP direct crawl returned non-text content: {content_type}")
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            logger.warning(f"HTTP also failed for {domain}: {e}")
        
        # If we got here, decide what to return based on what we got
        if https_success and https_content and len(https_content) > 100:
            return https_content, (None, None), "direct_https_fallback"
        elif http_success and http_content and len(http_content) > 100:
            return http_content, (None, None), "direct_http_fallback"
        elif https_success and https_content:
            return https_content, (None, None), "direct_https_minimal"
        elif http_success and http_content:
            return http_content, (None, None), "direct_http_minimal"
        else:
            return None, ("empty_content", "Website returned empty or insufficient content"), None
    
    except requests.exceptions.SSLError as e:
        logger.error(f"SSL error during direct crawl: {e}")
        # Try HTTP as a last resort
        try:
            http_url = url.replace('https://', 'http://')
            logger.info(f"Final HTTP attempt for {domain}: {http_url}")
            http_response = requests.get(http_url, headers=headers, timeout=timeout)
            html_content = http_response.text
            if html_content and len(html_content.strip()) > 100:
                # Simple cleanup
                clean_text = re.sub(r'<script.*?>.*?</script>', ' ', html_content, flags=re.DOTALL)
                clean_text = re.sub(r'<style.*?>.*?</style>', ' ', clean_text, flags=re.DOTALL)
                clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                logger.info(f"Final HTTP attempt successful, got {len(clean_text)} characters")
                return clean_text, (None, None), "direct_http_final"
        except Exception:
            pass
            
        if "certificate verify failed" in str(e):
            return None, ("ssl_invalid", "The website has an invalid SSL certificate"), None
        elif "certificate has expired" in str(e):
            return None, ("ssl_expired", "The website's SSL certificate has expired"), None
        else:
            return None, ("ssl_error", "The website has SSL certificate issues"), None
    
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error during direct crawl: {e}")
        return None, ("connection_error", "Could not establish a connection to the website"), None
    
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout during direct crawl: {e}")
        return None, ("timeout", "The website took too long to respond"), None
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error during direct crawl: {e}")
        status_code = e.response.status_code if hasattr(e, 'response') and hasattr(e.response, 'status_code') else None
        
        if status_code == 403 or status_code == 401:
            return None, ("access_denied", "Access to the website was denied"), None
        elif status_code == 404:
            return None, ("not_found", "The requested page was not found"), None
        elif status_code and 500 <= status_code < 600:
            return None, ("server_error", "The website is experiencing server errors"), None
        else:
            return None, ("http_error", f"HTTP error {status_code}"), None
    
    except Exception as e:
        logger.error(f"Unexpected error during direct crawl: {e}")
        return None, ("unknown_error", f"An unexpected error occurred: {str(e)}"), None

def try_multiple_protocols(domain: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]], Optional[str]]:
    """
    Try crawling a domain with different protocols (https, http) in case one fails.
    
    Args:
        domain (str): The domain to crawl (without protocol)
        
    Returns:
        tuple: Same as direct_crawl
    """
    # First try https
    clean_domain = domain.replace('https://', '').replace('http://', '')
    https_url = f"https://{clean_domain}"
    
    logger.info(f"Trying HTTPS protocol for {clean_domain}")
    content, (error_type, error_detail), crawler_type = direct_crawl(https_url)
    
    # If https failed due to SSL or connection issues, try http
    if not content and error_type in ['ssl_invalid', 'ssl_expired', 'ssl_error', 'connection_error']:
        logger.info(f"HTTPS failed, trying HTTP protocol for {clean_domain}")
        http_url = f"http://{clean_domain}"
        return direct_crawl(http_url)
    
    return content, (error_type, error_detail), crawler_type
