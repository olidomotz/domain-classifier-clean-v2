"""Error handling utilities for domain classification."""
import logging
import socket
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
from typing import Dict, Any, Tuple, Optional

# Import final classification utility (if we can - use try/except to avoid circular imports)
try:
    from domain_classifier.utils.final_classification import determine_final_classification
    HAS_FINAL_CLASSIFICATION = True
except ImportError:
    HAS_FINAL_CLASSIFICATION = False

# Import text processing functions
try:
    from domain_classifier.utils.text_processing import generate_one_line_description
    HAS_TEXT_PROCESSING = True
except ImportError:
    HAS_TEXT_PROCESSING = False

# Import JSON utilities if available
try:
    from domain_classifier.utils.json_utils import ensure_dict, safe_get
    HAS_JSON_UTILS = True
except ImportError:
    HAS_JSON_UTILS = False
    logger = logging.getLogger(__name__)
    logger.warning("JSON utilities not available in error_handling.py")

# Set up logging
logger = logging.getLogger(__name__)

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

def check_domain_dns(domain: str) -> Tuple[bool, Optional[str], bool]:
    """
    Check if a domain has valid DNS resolution AND can respond to a basic HTTP request.
    Also detects potentially flaky sites that may fail during crawling.
    
    Args:
        domain (str): The domain to check
        
    Returns:
        tuple: (has_dns, error_message, potentially_flaky)
            - has_dns: Whether the domain has DNS resolution
            - error_message: Error message if DNS resolution failed
            - potentially_flaky: Whether the site shows signs of being flaky
    """
    potentially_flaky = False
    
    try:
        # Remove protocol if present
        clean_domain = domain.replace('https://', '').replace('http://', '')
        
        # Remove path if present
        if '/' in clean_domain:
            clean_domain = clean_domain.split('/', 1)[0]
        
        # Step 1: Try to resolve the domain using socket
        try:
            logger.info(f"Checking DNS resolution for domain: {clean_domain}")
            socket.setdefaulttimeout(3.0)  # 3 seconds max
            ip_address = socket.gethostbyname(clean_domain)
            logger.info(f"DNS resolution successful for domain: {clean_domain} (IP: {ip_address})")
            
            # Step 2: Try to establish a reliable HTTP connection
            try:
                logger.info(f"Attempting HTTP connection check for {clean_domain}")
                session = requests.Session()
                
                # Try HTTPS first
                success = False
                remote_disconnect_https = False
                https_ssl_error = False
                
                try:
                    url = f"https://{clean_domain}"
                    response = session.get(
                        url, 
                        timeout=5.0,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                            'Accept-Language': 'en-US,en;q=0.9',
                            'Accept-Encoding': 'gzip, deflate, br',
                            'Connection': 'keep-alive',
                            'Cache-Control': 'max-age=0',
                            'Upgrade-Insecure-Requests': '1'
                        },
                        stream=True
                    )
                    
                    # CRITICAL: Actually try to read a chunk of content
                    # This is what will detect connection issues with problematic sites
                    try:
                        chunk = next(response.iter_content(1024), None)
                        if chunk:
                            # Quick check for parked domain in this first chunk
                            from domain_classifier.classifiers.decision_tree import is_parked_domain
                            chunk_text = chunk.decode('utf-8', errors='ignore')
                            if is_parked_domain(chunk_text, clean_domain):
                                logger.info(f"Domain {clean_domain} appears to be a parked domain based on initial content")
                                return True, "parked_domain", False
                                
                            success = True
                            logger.info(f"Successfully read content chunk from {clean_domain}")
                        else:
                            potentially_flaky = True
                            logger.warning(f"No content received from {clean_domain}")
                    except Exception as read_error:
                        logger.warning(f"Error reading content from {clean_domain}: {read_error}")
                        if "RemoteDisconnected" in str(read_error) or "ConnectionResetError" in str(read_error):
                            remote_disconnect_https = True
                            logger.info(f"Detected remote disconnect during content read for {clean_domain}")
                        potentially_flaky = True
                    
                    response.close()
                    
                    if success:
                        return True, None, False
                    
                except requests.exceptions.SSLError as https_e:
                    logger.warning(f"HTTPS failed for {clean_domain}, trying HTTP: {https_e}")
                    
                    # Mark SSL error for special handling
                    https_ssl_error = True
                    
                    # Look for reset indicators
                    if "RemoteDisconnected" in str(https_e) or "ConnectionResetError" in str(https_e) or "reset by peer" in str(https_e) or "connection aborted" in str(https_e).lower():
                        potentially_flaky = True
                        remote_disconnect_https = True
                        logger.info(f"Detected potential anti-scraping protection on HTTPS for {clean_domain}")
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as https_e:
                    logger.warning(f"HTTPS connection failed for {clean_domain}: {https_e}")
                    potentially_flaky = True
                
                # Try HTTP as fallback
                remote_disconnect_http = False
                http_success = False
                
                try:
                    url = f"http://{clean_domain}"
                    response = session.get(
                        url, 
                        timeout=5.0,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                            'Accept-Language': 'en-US,en;q=0.9',
                            'Accept-Encoding': 'gzip, deflate, br',
                            'Connection': 'keep-alive',
                            'Cache-Control': 'max-age=0',
                            'Upgrade-Insecure-Requests': '1'
                        },
                        stream=True
                    )
                    
                    # CRITICAL: Actually try to read a chunk of content
                    try:
                        chunk = next(response.iter_content(1024), None)
                        if chunk:
                            # Quick check for parked domain in this first chunk
                            from domain_classifier.classifiers.decision_tree import is_parked_domain
                            chunk_text = chunk.decode('utf-8', errors='ignore')
                            if is_parked_domain(chunk_text, clean_domain):
                                logger.info(f"Domain {clean_domain} appears to be a parked domain based on HTTP content")
                                return True, "parked_domain", False
                                
                            http_success = True
                            logger.info(f"Successfully read content chunk from {clean_domain} (HTTP)")
                            
                            # CRITICAL CHANGE: If HTTPS failed with SSL but HTTP works, signal it
                            if https_ssl_error:
                                return True, "http_success_https_failed", False
                            
                            return True, None, False
                        else:
                            potentially_flaky = True
                            logger.warning(f"No content received from {clean_domain} (HTTP)")
                    except Exception as read_error:
                        logger.warning(f"Error reading content from {clean_domain} (HTTP): {read_error}")
                        if "RemoteDisconnected" in str(read_error) or "ConnectionResetError" in str(read_error):
                            remote_disconnect_http = True
                            logger.info(f"Detected remote disconnect during content read for {clean_domain} (HTTP)")
                        potentially_flaky = True
                    
                    response.close()
                    
                    if http_success:
                        # CRITICAL CHANGE: If HTTPS failed with SSL but HTTP works, signal it
                        if https_ssl_error:
                            return True, "http_success_https_failed", False
                            
                        return True, None, False
                        
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as http_e:
                    logger.warning(f"HTTP also failed for {clean_domain}: {http_e}")
                    
                    # Look for reset indicators
                    if "RemoteDisconnected" in str(http_e) or "ConnectionResetError" in str(http_e) or "reset by peer" in str(http_e) or "connection aborted" in str(http_e).lower():
                        potentially_flaky = True
                        remote_disconnect_http = True
                        logger.info(f"Detected potential anti-scraping protection on HTTP for {clean_domain}")
                        
                    # CRITICAL CHANGE: If either HTTP or HTTPS were closed by remote end, this is likely 
                    # anti-scraping protection - mark it as needing advanced crawling
                    if remote_disconnect_https or remote_disconnect_http:
                        logger.warning(f"Domain {clean_domain} appears to be using anti-scraping protection - will attempt advanced crawlers")
                        return True, "anti_scraping_protection", True
                        
                    # If it failed with both HTTPS and HTTP, it's not usable
                    error_message = f"The domain {domain} resolves but the web server is not responding properly. The server might be misconfigured or blocking requests."
                    
                    return False, error_message, potentially_flaky
                
                # If we got here, we tried both protocols but couldn't read content properly
                if potentially_flaky:
                    # If we detected anti-scraping protection in either request, pass the domain to advanced crawlers
                    if remote_disconnect_https or remote_disconnect_http:
                        logger.warning(f"Domain {clean_domain} has shown signs of anti-scraping protection - will attempt advanced crawlers")
                        return True, "anti_scraping_protection", True
                        
                    return False, f"The domain {domain} connects but fails during content transfer.", True
                
                return False, f"Could not establish a proper connection to {domain}.", False
                
            except Exception as conn_e:
                logger.warning(f"Connection error for {clean_domain}: {conn_e}")
                
                # Check for specific flaky indicators
                if "RemoteDisconnected" in str(conn_e) or "ConnectionResetError" in str(conn_e) or "reset by peer" in str(conn_e) or "connection aborted" in str(conn_e).lower():
                    potentially_flaky = True
                    logger.warning(f"Domain {clean_domain} appears to be using anti-scraping protection - will attempt advanced crawlers")
                    return True, "anti_scraping_protection", True
                    
                return False, f"The domain {domain} resolves but cannot be connected to. The server might be down or blocking connections.", potentially_flaky
                
        except socket.gaierror as e:
            logger.warning(f"DNS resolution failed for {domain}: {e}")
            return False, f"The domain {domain} could not be resolved. It may not exist or DNS records may be misconfigured.", False
            
    except socket.timeout as e:
        logger.warning(f"DNS resolution timed out for {domain}: {e}")
        return False, f"Timed out while checking {domain}. Domain may not exist or the server is not responding.", False
    except Exception as e:
        logger.error(f"Unexpected error checking domain {domain}: {e}")
        return False, f"Error checking {domain}: {e}", False

def is_domain_worth_crawling(domain: str) -> tuple:
    """
    Determines if a domain is worth attempting a full crawl based on preliminary checks.
    
    Args:
        domain (str): The domain to check
        
    Returns:
        tuple: (worth_crawling, has_dns, error_msg, potentially_flaky)
    """
    has_dns, error_msg, potentially_flaky = check_domain_dns(domain)
    
    # CHANGE: Special handling for anti-scraping protection
    if error_msg == "anti_scraping_protection":
        logger.info(f"Domain {domain} has anti-scraping protection, will proceed with advanced crawlers")
        return True, has_dns, error_msg, potentially_flaky
    
    # Store HTTP success for later use
    http_success = error_msg == "http_success_https_failed"
    
    # Don't crawl if DNS resolution fails or if it's a parked domain
    if not has_dns or error_msg == "parked_domain":
        logger.info(f"Domain {domain} failed check: {error_msg}")
        return False, has_dns, error_msg, potentially_flaky
        
    # Be cautious with potentially flaky domains but still allow crawling
    if potentially_flaky:
        logger.warning(f"Domain {domain} may be flaky, proceeding with caution")
        
    # If HTTP worked but HTTPS failed, return a special signal
    if http_success:
        logger.info(f"Domain {domain} works with HTTP but not HTTPS, setting special flag")
        return True, has_dns, "http_success_https_failed", potentially_flaky
        
    return True, has_dns, error_msg, potentially_flaky

def create_error_result(domain: str, error_type: Optional[str] = None, 
                        error_detail: Optional[str] = None, email: Optional[str] = None,
                        crawler_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a standardized error response based on the error type.
    
    Args:
        domain (str): The domain being processed
        error_type (str, optional): The type of error detected
        error_detail (str, optional): Detailed explanation of the error
        email (str, optional): Email address if processing an email
        crawler_type (str, optional): The type of crawler used/attempted
        
    Returns:
        dict: Standardized error response
    """
    # Default error response
    error_result = {
        "domain": domain,
        "error": "Failed to crawl website",
        "predicted_class": "Unknown",
        "confidence_score": 0,
        "confidence_scores": {
            "Managed Service Provider": 0,
            "Integrator - Commercial A/V": 0,
            "Integrator - Residential A/V": 0,
            "Internal IT Department": 0
        },
        "low_confidence": True,
        "is_crawl_error": True
    }
    
    # Add email if provided
    if email:
        error_result["email"] = email
    
    # Add error_type if provided
    if error_type:
        error_result["error_type"] = error_type
    
    # Default explanation
    explanation = f"We were unable to retrieve content from {domain}. This could be due to a server timeout or the website being unavailable. Without analyzing the website content, we cannot determine the company type with confidence, but will still attempt enrichment."
    
    # Enhanced error handling based on error type
    if error_type:
        if error_type == 'anti_scraping':
            explanation = f"We couldn't analyze {domain} because the website appears to be using anti-scraping protection that prevents automated access. Our system will try more advanced techniques to extract the content and will attempt enrichment from external sources regardless."
            error_result["is_anti_scraping"] = True
        elif error_type.startswith('ssl_'):
            explanation = f"We couldn't analyze {domain} because of SSL certificate issues. "
            if error_type == 'ssl_expired':
                explanation += f"The website's SSL certificate has expired. This is a security issue with the target website, not our classification service."
            elif error_type == 'ssl_invalid':
                explanation += f"The website has an invalid SSL certificate. This is a security issue with the target website, not our classification service."
            else:
                explanation += f"This is a security issue with the target website, not our classification service."
            
            error_result["is_ssl_error"] = True
            
        elif error_type == 'dns_error':
            explanation = f"We couldn't analyze {domain} because the domain could not be resolved. This typically means the domain doesn't exist or its DNS records are misconfigured."
            error_result["is_dns_error"] = True
            
        elif error_type == 'connection_error':
            explanation = f"We couldn't analyze {domain} because a connection couldn't be established. The website may be down, temporarily unavailable, or blocking our requests. We will still attempt to enrich the domain from external sources."
            error_result["is_connection_error"] = True
            
        elif error_type == 'access_denied':
            explanation = f"We couldn't analyze {domain} because access was denied (403 Forbidden). The website may be blocking automated access or requiring authentication. We will still attempt to enrich the domain from external sources."
            error_result["is_access_denied"] = True
            
        elif error_type == 'not_found':
            explanation = f"We couldn't analyze {domain} because the main page was not found. The website may be under construction or have moved to a different URL. We will still attempt to enrich the domain from external sources."
            error_result["is_not_found"] = True
            
        elif error_type == 'server_error':
            explanation = f"We couldn't analyze {domain} because the website is experiencing server errors. This is an issue with the target website, not our classification service. We will still attempt to enrich the domain from external sources."
            error_result["is_server_error"] = True
            
        elif error_type == 'robots_restricted':
            explanation = f"We couldn't analyze {domain} because the website restricts automated access. This is a policy set by the website owner. We will still attempt to enrich the domain from external sources."
            error_result["is_robots_restricted"] = True
            
        elif error_type == 'timeout':
            explanation = f"We couldn't analyze {domain} because the website took too long to respond. The website may be experiencing performance issues or temporarily unavailable. We will still attempt to enrich the domain from external sources."
            error_result["is_timeout"] = True
            
        elif error_type == 'is_parked' or error_type == 'parked_domain':
            explanation = f"The domain {domain} appears to be parked or inactive. This domain may be registered but not actively in use for a business."
            error_result["is_parked"] = True
            error_result["predicted_class"] = "Parked Domain"
            
        # If we have a specific error detail, use it to enhance the explanation
        if error_detail:
            explanation += f" {error_detail}"
    
    error_result["explanation"] = explanation
    
    # Add one-line company description based on error type
    if HAS_TEXT_PROCESSING:
        if error_type == "is_parked" or error_type == "parked_domain":
            error_result["company_one_line"] = f"{domain} is a parked domain with no active business."
        else:
            error_result["company_one_line"] = f"Unable to determine what {domain} does due to access issues. Will attempt enrichment from external sources."
    else:
        # Fallback if text processing not available
        if error_type == "is_parked" or error_type == "parked_domain":
            error_result["company_one_line"] = f"{domain} is a parked domain with no active business."
        else:
            error_result["company_one_line"] = f"Unable to determine what {domain} does due to technical issues. Will attempt enrichment."
    
    # Add crawler_type if provided
    error_result["crawler_type"] = crawler_type or "error_handler"  # Set a default
    
    # Add classifier_type
    error_result["classifier_type"] = "early_detection" if error_type in ["is_parked", "parked_domain"] else "error_handler"
    
    # Add final classification if possible
    if HAS_FINAL_CLASSIFICATION:
        # Import here to avoid circular imports
        from domain_classifier.utils.final_classification import determine_final_classification
        error_result["final_classification"] = determine_final_classification(error_result)
    else:
        # Default for DNS errors or connection errors
        if error_type in ["dns_error", "connection_error"]:
            error_result["final_classification"] = "7-No Website available"
        elif error_type in ["is_parked", "parked_domain"]:
            error_result["final_classification"] = "6-Parked Domain - no enrichment"
        else:
            error_result["final_classification"] = "2-Internal IT"  # Default fallback
    
    return error_result
