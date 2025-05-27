"""
Comprehensive fixes for domain classifier.

Addresses DNS error detection, parked domain detection, Apollo data usage,
and timeout issues.
"""

import logging
import re
import json
import os
import requests
import importlib.util
import sys
import socket
import time
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import importlib
import inspect
from functools import wraps

# Set up logging
logger = logging.getLogger(__name__)

def apply_patches():
    """Apply all necessary patches for domain classifier."""
    logger.info("Applying comprehensive fixes to domain classifier...")

    # Step 1: Enhance DNS error handling
    fix_dns_error_detection()

    # Step 2: Improve parked domain detection
    fix_parked_domain_detection()

    # Step 3: Fix Apollo data usage for classification
    fix_apollo_data_classification()

    # Step 4: Fix timeout issues
    fix_timeout_issues()

    # Step 5: Fix Process Did Not Complete classification
    fix_process_did_not_complete()

    # Step 6: Set up monitoring
    setup_enhanced_monitoring()

    # Step 7: Add request interceptor for DNS errors
    add_dns_error_interceptor()

    logger.info("All patches successfully applied")

def fix_dns_error_detection():
    """Fix DNS error detection to properly identify non-existent domains."""
    try:
        # Patch error_handling.py
        from domain_classifier.utils import error_handling

        # Store original functions
        original_check_domain_dns = error_handling.check_domain_dns
        original_is_domain_worth_crawling = error_handling.is_domain_worth_crawling

        def enhanced_check_domain_dns(domain: str) -> Tuple[bool, Optional[str], bool]:
            """
            Enhanced check_domain_dns with better non-existent domain detection.
            Properly identifies invalid TLDs and handles DNS errors.
            """
            potentially_flaky = False

            try:
                # Remove protocol if present
                clean_domain = domain.replace('https://', '').replace('http://', '')

                # Remove path if present
                if '/' in clean_domain:
                    clean_domain = clean_domain.split('/', 1)[0]

                # Check if TLD is valid - very basic check for common TLDs
                valid_tlds = ['.com', '.net', '.org', '.io', '.co', '.edu', '.gov', '.info', '.biz',
                             '.ai', '.app', '.dev', '.me', '.tech', '.us', '.uk', '.ca', '.au', '.de',
                             '.fr', '.jp', '.cn', '.ru', '.br', '.it', '.nl', '.es', '.sg', '.in']

                domain_parts = clean_domain.split('.')
                if len(domain_parts) > 1:
                    tld = f".{domain_parts[-1].lower()}"
                    if tld not in valid_tlds:
                        logger.warning(f"Domain {clean_domain} has an unusual TLD: {tld}")
                        # Non-standard TLD - likely invalid
                        return False, f"The domain {domain} has an unusual TLD ({tld}) which may not be valid.", False

                # Try a more definitive DNS check
                try:
                    socket.setdefaulttimeout(3.0)  # 3 seconds max
                    
                    # Try getaddrinfo for a thorough check
                    addr_info = socket.getaddrinfo(clean_domain, None)
                    ip_address = addr_info[0][4][0] if addr_info else None
                    
                    logger.info(f"DNS resolution successful for domain: {clean_domain} (IP: {ip_address})")
                    
                    # Continue with original function for HTTP connection check
                    original_has_dns, original_error, original_flaky = original_check_domain_dns(domain)
                    return original_has_dns, original_error, original_flaky
                    
                except socket.gaierror as e:
                    # Detailed error handling for different DNS error codes
                    error_code = getattr(e, 'errno', None)
                    error_str = str(e).lower()
                    
                    # Common DNS error codes:
                    # -2: Name or service not known (domain doesn't exist)
                    # -3: Temporary name resolution failure
                    # -5: No address associated with hostname
                    
                    if error_code == -2 or "name or service not known" in error_str:
                        logger.warning(f"Domain {clean_domain} does not exist (Name not known)")
                        return False, f"The domain {domain} could not be resolved. It may not exist or DNS records may be misconfigured.", False
                    elif error_code == -3 or "temporary failure in name resolution" in error_str:
                        logger.warning(f"Domain {clean_domain} has temporary DNS issues")
                        return False, f"Temporary DNS resolution failure for {domain}. Please try again later.", True
                    elif error_code == -5 or "no address associated" in error_str:
                        logger.warning(f"Domain {clean_domain} has no DNS records")
                        return False, f"The domain {domain} has no DNS A records.", False
                    else:
                        logger.warning(f"DNS resolution failed for {domain}: {e}")
                        return False, f"The domain {domain} could not be resolved. It may not exist or DNS records may be misconfigured.", False
                        
                except socket.timeout:
                    logger.warning(f"DNS resolution timed out for {domain}")
                    return False, f"Timed out while checking {domain}. DNS resolution timed out.", False
                    
            except Exception as e:
                logger.error(f"Unexpected error checking domain {domain}: {e}")
                return False, f"Error checking {domain}: {e}", False

        # Patch is_domain_worth_crawling to ensure DNS errors are properly propagated
        def enhanced_is_domain_worth_crawling(domain: str) -> tuple:
            """Enhanced is_domain_worth_crawling that properly handles DNS errors."""
            
            # Use enhanced check_domain_dns
            has_dns, error_msg, potentially_flaky = enhanced_check_domain_dns(domain)
            
            # Special handling for DNS errors
            if not has_dns and error_msg and ("DNS" in error_msg or "domain" in error_msg.lower() or "resolve" in error_msg.lower()):
                logger.info(f"Domain {domain} has DNS resolution issues: {error_msg}")
                return False, False, "dns_error", False
            
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

        # Apply the patches
        error_handling.check_domain_dns = enhanced_check_domain_dns
        error_handling.is_domain_worth_crawling = enhanced_is_domain_worth_crawling

        # Patch create_error_result to ensure DNS errors get the right classification
        original_create_error_result = error_handling.create_error_result

        def enhanced_create_error_result(domain: str, error_type: Optional[str] = None,
                                
                    except Exception as e:
                        logger.error(f"Error in DNS error handler: {e}")
                    
                    # For all other cases, proceed with normal request handling
                    return None
        
        # Try different methods to patch the app
        try:
            # Method 1: Direct import of app
            try:
                from domain_classifier.api.app import app
                dns_error_handler(app)
                logger.info("✅ Applied DNS error interceptor to app directly")
                return
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not apply DNS interceptor directly to app: {e}")
            
            # Method 2: Patch the app factory
            try:
                from domain_classifier.api.app import create_app as original_create_app
                
                @wraps(original_create_app)
                def patched_create_app(*args, **kwargs):
                    app = original_create_app(*args, **kwargs)
                    dns_error_handler(app)
                    logger.info("✅ Applied DNS error interceptor via app factory")
                    return app
                
                # Replace the create_app function
                sys.modules['domain_classifier.api.app'].create_app = patched_create_app
                
                logger.info("✅ Patched create_app to add DNS error interceptor")
                return
                
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not patch create_app: {e}")
                
            # Method 3: Monkey patch Flask itself
            try:
                original_full_dispatch_request = flask.Flask.full_dispatch_request
                
                def patched_full_dispatch_request(self):
                    if request.method == 'POST' and request.path == '/classify-and-enrich':
                        try:
                            # Extract data from request
                            data = request.json
                            if data:
                                input_value = data.get('url', '').strip()
                                if input_value:
                                    # Determine if input is an email
                                    is_email = '@' in input_value
                                    email = input_value if is_email else None
                                    
                                    # Extract domain
                                    domain = None
                                    if is_email:
                                        from domain_classifier.utils.domain_utils import extract_domain_from_email
                                        domain = extract_domain_from_email(input_value)
                                    else:
                                        from domain_classifier.utils.domain_utils import normalize_domain
                                        domain = normalize_domain(input_value)
                                        
                                    if domain:
                                        # Create URL for checking
                                        url = f"https://{domain}"
                                        
                                        # Check for DNS resolution
                                        from domain_classifier.utils.error_handling import is_domain_worth_crawling
                                        worth_crawling, has_dns, error_msg, potentially_flaky = is_domain_worth_crawling(domain)
                                        
                                        # For DNS errors, create a response and return immediately
                                        if not has_dns and (error_msg == "dns_error" or "dns" in str(error_msg).lower()):
                                            logger.info(f"DNS error detected for {domain}, returning DNS error response")
                                            
                                            from domain_classifier.utils.error_handling import create_error_result
                                            
                                            error_result = create_error_result(
                                                domain,
                                                "dns_error",
                                                error_msg if error_msg != "dns_error" else None,
                                                email,
                                                "early_check"
                                            )
                                            error_result["website_url"] = url
                                            error_result["final_classification"] = "7-No Website available"
                                            error_result["predicted_class"] = "DNS Error"
                                            
                                            # Format the response
                                            try:
                                                from domain_classifier.utils.api_formatter import format_api_response
                                                formatted_result = format_api_response(error_result)
                                                return jsonify(formatted_result)
                                            except Exception as format_error:
                                                logger.error(f"Error formatting DNS error response: {format_error}")
                                                return jsonify(error_result)
                        except Exception as e:
                            logger.error(f"Error in patched Flask dispatch: {e}")
                    
                    # For all other cases, proceed with normal request handling
                    return original_full_dispatch_request(self)
                
                # Apply the patch
                flask.Flask.full_dispatch_request = patched_full_dispatch_request
                
                logger.info("✅ Applied global Flask request handler patch for DNS errors")
                return
                
            except Exception as flask_error:
                logger.warning(f"Could not patch Flask request dispatch: {flask_error}")
                
            # Method 4: Try to find and patch specific routes
            try:
                # Find all route modules
                from domain_classifier.api.routes import enrich
                
                # Try to find the classify_and_enrich function
                for name, obj in inspect.getmembers(enrich):
                    if inspect.isfunction(obj) and ('classify' in name.lower() or 'enrich' in name.lower()):
                        original_func = obj
                        
                        @wraps(original_func)
                        def patched_function(*args, **kwargs):
                            # Check for DNS issues first
                            if request.method == 'POST':
                                try:
                                    # Extract data from request
                                    data = request.json
                                    if data:
                                        input_value = data.get('url', '').strip()
                                        if input_value:
                                            # Process as before...
                                            is_email = '@' in input_value
                                            email = input_value if is_email else None
                                            
                                            # Extract domain
                                            domain = None
                                            if is_email:
                                                from domain_classifier.utils.domain_utils import extract_domain_from_email
                                                domain = extract_domain_from_email(input_value)
                                            else:
                                                from domain_classifier.utils.domain_utils import normalize_domain
                                                domain = normalize_domain(input_value)
                                                
                                            if domain:
                                                # Check DNS
                                                from domain_classifier.utils.error_handling import is_domain_worth_crawling
                                                worth_crawling, has_dns, error_msg, potentially_flaky = is_domain_worth_crawling(domain)
                                                
                                                if not has_dns and (error_msg == "dns_error" or "dns" in str(error_msg).lower()):
                                                    # Return error response
                                                    logger.info(f"DNS error detected in patched route function for {domain}")
                                                    
                                                    from domain_classifier.utils.error_handling import create_error_result
                                                    
                                                    error_result = create_error_result(
                                                        domain,
                                                        "dns_error",
                                                        error_msg if error_msg != "dns_error" else None,
                                                        email,
                                                        "early_check"
                                                    )
                                                    error_result["website_url"] = f"https://{domain}"
                                                    error_result["final_classification"] = "7-No Website available"
                                                    error_result["predicted_class"] = "DNS Error"
                                                    
                                                    # Format the response
                                                    try:
                                                        from domain_classifier.utils.api_formatter import format_api_response
                                                        formatted_result = format_api_response(error_result)
                                                        return jsonify(formatted_result), 200
                                                    except Exception as format_error:
                                                        logger.error(f"Error formatting DNS error response: {format_error}")
                                                        return jsonify(error_result), 200
                                except Exception as e:
                                    logger.error(f"Error in patched route function: {e}")
                                    
                            # Otherwise, proceed with original function
                            return original_func(*args, **kwargs)
                            
                        # Apply the patch
                        setattr(enrich, name, patched_function)
                        
                        logger.info(f"✅ Applied DNS error handling to route function: {name}")
                        return
                
                logger.warning("Could not find appropriate route function to patch")
                
            except Exception as route_error:
                logger.warning(f"Could not patch route functions: {route_error}")
                
        except Exception as e:
            logger.error(f"Failed to apply any DNS error interceptor method: {e}")
            
    except Exception as e:
        logger.error(f"❌ Failed to add DNS error interceptor: {e}")

# Run the patches if this module is executed directly
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Apply all patches
    apply_patches()       error_detail: Optional[str] = None, email: Optional[str] = None,
                                       crawler_type: Optional[str] = None) -> Dict[str, Any]:
            """Enhanced create_error_result that properly handles DNS errors."""
            
            # Get the basic error result from the original function
            error_result = original_create_error_result(domain, error_type, error_detail, email, crawler_type)
            
            # Ensure DNS errors get the correct treatment
            if error_type == "dns_error":
                # CRITICAL: Set the predicted_class to "DNS Error" for better visibility
                error_result["predicted_class"] = "DNS Error"
                error_result["is_dns_error"] = True
                error_result["final_classification"] = "7-No Website available"
                
                # Set a proper explanation
                error_result["explanation"] = f"The domain {domain} could not be resolved. It may not exist or DNS records may be misconfigured."
                
                # Set company_description and one_line
                error_result["company_description"] = f"The domain {domain} could not be resolved. It may not exist or DNS records may be misconfigured."
                error_result["company_one_line"] = f"Domain cannot be reached - DNS error."
                
            return error_result

        # Apply the patch
        error_handling.create_error_result = enhanced_create_error_result

        logger.info("✅ Applied DNS error detection fixes to error_handling.py")

    except Exception as e:
        logger.error(f"❌ Failed to fix DNS error detection: {e}")

def fix_parked_domain_detection():
    """Fix parked domain detection to catch actual parked domains while avoiding false positives."""
    try:
        from domain_classifier.classifiers import decision_tree

        # Define a completely new implementation
        def enhanced_is_parked_domain(content: str, domain: str = None) -> bool:
            """
            Fixed parked domain detection with better balance between detection and false positives.
            Args:
                content: The website content
                domain: Optional domain name for additional checks
            Returns:
                bool: True if the domain is parked/inactive
            """
            if not content:
                logger.info("Domain has no content at all, considering as parked")
                return True
                
            content_lower = content.lower()
            
            # 1. Check for DEFINITIVE parking indicators (highest confidence)
            definitive_indicators = [
                "domain is for sale", "buy this domain", "domain parking",
                "this domain is for sale", "parked by", "domain broker",
                "inquire about this domain", "this domain is available for purchase",
                "bid on this domain"
            ]
            
            for indicator in definitive_indicators:
                if indicator in content_lower:
                    logger.info(f"Domain contains definitive parking indicator: '{indicator}'")
                    return True
            
            # 2. Check for combinations of parking-related indicators
            parking_phrases = [
                "domain may be for sale", "domain for sale", "domain name parking",
                "purchase this domain", "domain has expired", "domain available"
            ]
            
            hosting_mentions = [
                "godaddy", "namecheap", "domain.com", "namesilo", "porkbun",
                "domain registration", "web hosting service", "hosting provider",
                "register this domain", "parkingcrew", "sedo", "bodis", "parked.com"
            ]
            
            parking_count = sum(1 for phrase in parking_phrases if phrase in content_lower)
            hosting_count = sum(1 for phrase in hosting_mentions if phrase in content_lower)
            
            # Require stronger evidence - multiple indicators or specific combinations
            if parking_count >= 2 or (parking_count >= 1 and hosting_count >= 1):
                logger.info(f"Domain has multiple parking indicators: {parking_count} parking phrases, {hosting_count} hosting mentions")
                return True
            
            # 3. Special case for GoDaddy proxy errors - but with stricter requirements
            if "proxy error" in content_lower and "godaddy" in content_lower and len(content) < 400:
                logger.info("Found GoDaddy proxy error specifically, likely parked")
                return True
            
            # 4. For very minimal content, check for specific patterns but require more evidence
            if len(content.strip()) < 200:
                technical_issues = [
                    "domain not configured", "website coming soon", "under construction",
                    "site temporarily unavailable", "default web page"
                ]
                
                tech_count = sum(1 for issue in technical_issues if issue in content_lower)
                
                # For minimal content, require at least 2 technical indicators
                if tech_count >= 2:
                    logger.info(f"Minimal content with multiple technical issues ({tech_count}), likely parked")
                    return True
                    
                # Check for few unique words - a sign of placeholder content
                words = re.findall(r'\b\w+\b', content_lower)
                unique_words = set(words)
                
                if len(words) < 15 or len(unique_words) < 10:
                    # ADDITIONAL CHECK - make sure it doesn't look like a legitimate minimal page
                    # Don't classify as parked if it has real business terms
                    business_terms = ["service", "contact", "about", "product", "company", "solution"]
                    has_business_content = any(term in content_lower for term in business_terms)
                    
                    if not has_business_content:
                        logger.info(f"Very minimal content with few unique words ({len(unique_words)}), likely parked")
                        return True
            
            # Default - not parked
            return False

        # Apply the patch
        decision_tree.is_parked_domain = enhanced_is_parked_domain
        
        logger.info("✅ Applied fixed parked domain detection to decision_tree.py")
        
        # Also patch quick_parked_check in apify_crawler.py
        try:
            from domain_classifier.crawlers import apify_crawler
            
            # Store original function
            original_quick_parked_check = apify_crawler.quick_parked_check
            
            def quick_parked_check_patch(url: str):
                """
                Patched version of quick_parked_check that's less aggressive.
                Args:
                    url: The URL to check
                Returns:
                    tuple: (is_parked, content) where is_parked is a boolean and content is the content if available
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
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                            'Accept-Language': 'en-US,en;q=0.5',
                            'Connection': 'keep-alive',
                            'Cache-Control': 'no-cache'
                        }
                        
                        # Set a low timeout to fail quickly
                        # IMPORTANT: Set allow_redirects=True to follow redirects
                        response = requests.get(url, headers=headers, timeout=5.0, stream=True, allow_redirects=True)

                        # Log redirect if it happened
                        if response.history:
                            redirect_chain = " -> ".join([r.url for r in response.history])
                            final_url = response.url
                            logger.info(f"Followed HTTPS redirect chain: {redirect_chain} -> {final_url}")
                            
                            # If redirected to a different domain, update for parked check
                            final_domain = urlparse(final_url).netloc
                            if final_domain != domain and final_domain.startswith('www.'):
                                final_domain = final_domain[4:]
                                
                            if final_domain != domain:
                                logger.info(f"Domain redirected from {domain} to {final_domain}")
                                domain = final_domain

                        # Get a chunk of content
                        content = next(response.iter_content(2048), None)
                        if content:
                            content_str = content.decode('utf-8', errors='ignore')
                            
                            # Check for DEFINITIVE parked domain indicators only in quick check
                            parked_indicators = [
                                "domain is for sale", "buy this domain", "domain parking",
                                "this domain is for sale", "parked by"
                            ]
                            
                            if any(indicator in content_str.lower() for indicator in parked_indicators):
                                logger.info(f"Quick check found definitive parked domain indicators for {domain}")
                                return True, content_str
                                
                            # If not immediately obvious from first chunk, get more content
                            full_content = content_str
                            try:
                                # Get up to 10KB more
                                for _ in range(5):
                                    chunk = next(response.iter_content(2048), None)
                                    if not chunk:
                                        break
                                    chunk_str = chunk.decode('utf-8', errors='ignore')
                                    full_content += chunk_str
                                    
                                # Use the fixed parked domain detection function to determine if it's parked
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
                                
                            # IMPORTANT: Set allow_redirects=True to follow redirects
                            response = requests.get(http_url, headers=headers, timeout=5.0, stream=True, allow_redirects=True)
                            
                            # Log redirect if it happened
                            if response.history:
                                redirect_chain = " -> ".join([r.url for r in response.history])
                                final_url = response.url
                                logger.info(f"Followed HTTP redirect chain: {redirect_chain} -> {final_url}")
                                
                                # If redirected to a different domain, update for parked check
                                final_domain = urlparse(final_url).netloc
                                if final_domain != domain and final_domain.startswith('www.'):
                                    final_domain = final_domain[4:]
                                    
                                if final_domain != domain:
                                    logger.info(f"Domain redirected from {domain} to {final_domain}")
                                    domain = final_domain
                            
                            # Get a chunk of content
                            content = next(response.iter_content(2048), None)
                            if content:
                                content_str = content.decode('utf-8', errors='ignore')
                                
                                # Check for DEFINITIVE parked domain indicators only in quick check
                                parked_indicators = [
                                    "domain is for sale", "buy this domain", "domain parking",
                                    "this domain is for sale", "parked by"
                                ]
                                
                                if any(indicator in content_str.lower() for indicator in parked_indicators):
                                    logger.info(f"HTTP quick check found definitive parked domain indicators for {domain}")
                                    return True, content_str
                                    
                                # If not immediately obvious, get more content
                                full_content = content_str
                                try:
                                    # Get up to 10KB more
                                    for _ in range(5):
                                        chunk = next(response.iter_content(2048), None)
                                        if not chunk:
                                            break
                                        chunk_str = chunk.decode('utf-8', errors='ignore')
                                        full_content += chunk_str
                                        
                                    # Use the fixed parked domain detection function to determine if it's parked
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

            # Apply the patch to apify_crawler
            apify_crawler.quick_parked_check = quick_parked_check_patch
            logger.info("✅ Applied fixed quick_parked_check to apify_crawler.py")
            
        except Exception as crawler_e:
            logger.error(f"❌ Failed to patch apify_crawler.py: {crawler_e}")
        
    except Exception as e:
        logger.error(f"❌ Failed to fix parked domain detection: {e}")

def fix_apollo_data_classification():
    """Fix Apollo data usage for classification when web content is unavailable."""
    try:
        # Patch description_enhancer.py to use Apollo data for classification
        from domain_classifier.enrichment import description_enhancer

        original_generate_detailed_description = description_enhancer.generate_detailed_description

        def enhanced_generate_detailed_description(classification: Dict[str, Any],
                                                apollo_data: Optional[Dict] = None,
                                                apollo_person_data: Optional[Dict] = None) -> str:
            """Enhanced description generation that uses Apollo data for classification when web content is unavailable."""
            try:
                # Domain name for logging
                domain = classification.get("domain", "unknown")

                # Handle DNS error cases
                if classification.get("error_type") == "dns_error" or classification.get("is_dns_error") == True:
                    logger.info(f"DNS error detected in generate_detailed_description for {domain}")
                    
                    # Set proper final classification
                    classification["final_classification"] = "7-No Website available"
                    classification["predicted_class"] = "DNS Error"
                    
                    return f"The domain {domain} could not be resolved. It may not exist or its DNS records may be misconfigured."

                # Handle Process Did Not Complete but with Apollo data
                if classification.get("predicted_class") == "Process Did Not Complete" and apollo_data:
                    # Get description from Apollo
                    description = None
                    for field in ["short_description", "description", "long_description"]:
                        if apollo_data.get(field):
                            description = apollo_data.get(field)
                            logger.info(f"Found Apollo {field} for {domain}")
                            break
                            
                    if description:
                        # Try to classify using the description
                        try:
                            # Import LLM classifier
                            from domain_classifier.classifiers.llm_classifier import LLMClassifier
                            
                            # Get API key
                            import os
                            api_key = os.environ.get("ANTHROPIC_API_KEY")
                            
                            if api_key:
                                # Create classifier
                                try:
                                    # Use the haiku model to improve performance
                                    classifier = LLMClassifier(api_key=api_key, model="claude-3-haiku-20240307")
                                except Exception:
                                    # Fall back to default model if haiku not available
                                    classifier = LLMClassifier(api_key=api_key)
                                
                                # Create classification text
                                classification_text = f"Domain: {domain}\n\n"
                                classification_text += f"Company Description from Apollo: {description}\n\n"
                                
                                if apollo_data.get("industry"):
                                    classification_text += f"Industry: {apollo_data['industry']}\n"
                                    
                                if apollo_data.get("employee_count"):
                                    classification_text += f"Employee Count: {apollo_data['employee_count']}\n"
                                
                                # Run classification on Apollo description
                                logger.info(f"Attempting to classify {domain} using Apollo data")
                                classification_attempt = classifier.classify(
                                    content=classification_text,
                                    domain=domain
                                )
                                
                                if classification_attempt and classification_attempt.get("predicted_class"):
                                    logger.info(f"Successfully classified {domain} as {classification_attempt.get('predicted_class')} using Apollo data")
                                    
                                    # Update the classification with Apollo-based classification
                                    classification["predicted_class"] = classification_attempt.get("predicted_class")
                                    classification["detection_method"] = "apollo_data_classification"
                                    classification["source"] = "apollo_data"
                                    
                                    # Update confidence scores if available
                                    if "confidence_scores" in classification_attempt:
                                        classification["confidence_scores"] = classification_attempt.get("confidence_scores")
                                    else:
                                        # Add default confidence scores if none provided by classifier
                                        classification["confidence_scores"] = {
                                            "Managed Service Provider": 10,
                                            "Integrator - Commercial A/V": 10,
                                            "Integrator - Residential A/V": 10,
                                            "Internal IT Department": 70
                                        }
                                        
                                    # Also set a default confidence score
                                    classification["confidence_score"] = 50
                                    
                                    # Set is_service_business based on the classification
                                    classification["is_service_business"] = classification["predicted_class"] in [
                                        "Managed Service Provider",
                                        "Integrator - Commercial A/V",
                                        "Integrator - Residential A/V"
                                    ]
                                    
                                    # Set final_classification based on the new predicted_class
                                    from domain_classifier.utils.final_classification import determine_final_classification
                                    classification["final_classification"] = determine_final_classification(classification)
                                    
                                    # Use the Apollo description as the company description
                                    classification["company_description"] = description
                                    
                                    # Set a company name if not present
                                    if not classification.get("company_name") and apollo_data.get("name"):
                                        classification["company_name"] = apollo_data.get("name")
                                    
                                    # Add a note about using Apollo data
                                    return f"{description}\n\nNote: This description is based on Apollo data as the website could not be analyzed."
                                else:
                                    # If LLM classification failed but we have Apollo data,
                                    # use "Internal IT Department" as a fallback with Apollo description
                                    logger.info(f"LLM classification with Apollo data failed, using fallback for {domain}")
                                    
                                    classification["predicted_class"] = "Internal IT Department"
                                    classification["detection_method"] = "apollo_data_classification"
                                    classification["source"] = "apollo_data"
                                    classification["confidence_scores"] = {
                                        "Managed Service Provider": 10,
                                        "Integrator - Commercial A/V": 10,
                                        "Integrator - Residential A/V": 10,
                                        "Internal IT Department": 70
                                    }
                                    classification["confidence_score"] = 50
                                    classification["is_service_business"] = False
                                    classification["final_classification"] = "2-Internal IT"
                                    classification["company_description"] = description
                                    
                                    # Set a company name if not present
                                    if not classification.get("company_name") and apollo_data.get("name"):
                                        classification["company_name"] = apollo_data.get("name")
                                    
                                    return f"{description}\n\nNote: This description is based on Apollo data as the website could not be analyzed."
                                    
                        except Exception as e:
                            logger.error(f"Error classifying with Apollo data: {e}")
                            
                        # Provide a fallback classification when LLM fails but we have Apollo data
                        if description:
                            logger.info(f"Using fallback classification with Apollo data for {domain}")
                            
                            classification["predicted_class"] = "Internal IT Department"
                            classification["detection_method"] = "apollo_data_classification"
                            classification["source"] = "apollo_data"
                            classification["confidence_scores"] = {
                                "Managed Service Provider": 10,
                                "Integrator - Commercial A/V": 10,
                                "Integrator - Residential A/V": 10,
                                "Internal IT Department": 70
                            }
                            classification["confidence_score"] = 50
                            classification["is_service_business"] = False
                            classification["final_classification"] = "2-Internal IT"
                            classification["company_description"] = description
                            
                            # Set a company name if not present
                            if not classification.get("company_name") and apollo_data.get("name"):
                                classification["company_name"] = apollo_data.get("name")
                            
                            return f"{description}\n\nNote: This description is based on Apollo data as the website could not be analyzed."

                # For all other cases, use the original function
                return original_generate_detailed_description(classification, apollo_data, apollo_person_data)
                
            except Exception as e:
                logger.error(f"Error in enhanced_generate_detailed_description: {e}")
                return original_generate_detailed_description(classification, apollo_data, apollo_person_data)

        # Apply the patch
        description_enhancer.generate_detailed_description = enhanced_generate_detailed_description

        logger.info("✅ Applied Apollo data classification fixes to description_enhancer.py")

    except Exception as e:
        logger.error(f"❌ Failed to fix Apollo data classification: {e}")

def fix_timeout_issues():
    """Fix timeout issues in crawler modules to prevent server timeouts."""
    try:
        # 1. Patch apify_crawler.py
        from domain_classifier.crawlers import apify_crawler
        
        # Store original functions
        original_crawl_website = apify_crawler.crawl_website
        original_apify_crawl = apify_crawler.apify_crawl
        
        def patched_crawl_website(url: str):
            """Patched version with reduced timeouts."""
            try:
                logger.info(f"Starting crawl for {url} with reduced timeouts")
                
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
                is_parked, quick_content = apify_crawler.quick_parked_check(url)
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
                
                # Save direct crawl results for later comparison
                direct_content_length = len(direct_content.strip()) if direct_content else 0
                
                # Track direct crawler usage
                if direct_content:
                    apify_crawler.track_crawler_usage(f"direct_{direct_crawler_type}")
                
                # REDUCED TIMEOUT: Only try Scrapy if direct crawl was unsuccessful or returned minimal content
                if not direct_content or direct_content_length < 300:
                    logger.info(f"Direct crawl got insufficient content ({direct_content_length} chars), trying Scrapy for {url}")
                    
                    from domain_classifier.crawlers.scrapy_crawler import scrapy_crawl
                    
                    # REDUCED TIMEOUT: Only wait 15 seconds instead of 30+
                    scrapy_content, (scrapy_error_type, scrapy_error_detail) = scrapy_crawl(url, timeout=15)
                    
                    scrapy_content_length = len(scrapy_content.strip()) if scrapy_content else 0
                    
                    # Log detailed debug info
                    logger.info(f"Crawler comparison for {domain}: direct={direct_content_length} chars, scrapy={scrapy_content_length} chars")
                    
                    # If Scrapy got good content (>150 chars), use it
                    if scrapy_content and scrapy_content_length > 150:
                        logger.info(f"Using Scrapy content: {scrapy_content_length} chars")
                        apify_crawler.track_crawler_usage("scrapy_primary")
                        return scrapy_content, (None, None), "scrapy"
                    
                    # If direct content is better than Scrapy, use direct content
                    if direct_content and direct_content_length > scrapy_content_length:
                        logger.info(f"Using direct content: {direct_content_length} chars")
                        return direct_content, (None, None), direct_crawler_type
                    
                    # If Scrapy got any content at all, use it
                    if scrapy_content and scrapy_content_length > 0:
                        logger.info(f"Using minimal Scrapy content: {scrapy_content_length} chars")
                        apify_crawler.track_crawler_usage("scrapy_minimal")
                        return scrapy_content, (None, None), "scrapy_minimal"
                    
                else:
                    # Direct crawl got good content, use it directly
                    logger.info(f"Direct crawl successful with {direct_content_length} chars, using it without trying Scrapy")
                    return direct_content, (None, None), direct_crawler_type
                
                # REDUCED TIMEOUT: Skip Apify entirely as it's the main source of timeouts
                # Just return the best content we have or an error
                if direct_content and direct_content_length > 0:
                    logger.info(f"Using minimal direct content instead of trying Apify: {direct_content_length} chars")
                    return direct_content, (None, None), "direct_minimal_fallback"
                
                # If we have no content at all, try direct HTTP one last time with a short timeout
                try:
                    logger.info(f"Final attempt: quick HTTP direct crawl for http://{domain}")
                    
                    from domain_classifier.crawlers.direct_crawler import direct_crawl
                    
                    # IMPORTANT: Use a short timeout and ensure redirects are followed
                    content, (final_error_type, final_error_detail), final_crawler_type = direct_crawl(
                        f"http://{domain}",
                        timeout=8.0  # Very short timeout
                    )
                    
                    if content and len(content.strip()) > 0:
                        logger.info(f"Final HTTP attempt successful for {domain}")
                        apify_crawler.track_crawler_usage("direct_http_final")
                        return content, (None, None), "direct_http_final"
                        
                except Exception as final_e:
                    logger.warning(f"Final HTTP attempt failed: {final_e}")
                
                # If everything failed, return error info from the best source
                error_info = (direct_error_type, direct_error_detail)
                if 'scrapy_error_type' in locals() and scrapy_error_type:
                    error_info = (scrapy_error_type, scrapy_error_detail)
                    
                return None, error_info, None
                
            except Exception as e:
                error_type, error_detail = apify_crawler.detect_error_type(str(e))
                logger.error(f"Error crawling website: {e} (Type: {error_type})")
                return None, (error_type, error_detail), None
        
        def patched_apify_crawl(url: str, timeout: int = 15):
            """Patched version with significantly reduced timeout."""
            try:
                logger.info(f"Starting fast Apify crawl for {url}")
                apify_crawler.track_crawler_usage("apify_started")
                
                # Extract domain for parked domain checks
                domain = urlparse(url).netloc
                if domain.startswith('www.'):
                    domain = domain[4:]
                
                # Start the crawl with standard settings
                endpoint = f"https://api.apify.com/v2/actor-tasks/{apify_crawler.APIFY_TASK_ID}/runs?token={apify_crawler.APIFY_API_TOKEN}"
                
                payload = {
                    "startUrls": [{"url": url}],
                    "maxCrawlingDepth": 1,
                    "maxCrawlPages": 3,  # REDUCED from 5
                    "timeoutSecs": 20,  # REDUCED from 30
                    "maxRequestRetries": 2,  # REDUCED from 3
                    "maxRedirects": 5,  # REDUCED from 10
                    "forceResponseEncoding": "utf-8"
                }
                
                headers = {"Content-Type": "application/json"}
                
                try:
                    response = requests.post(endpoint, json=payload, headers=headers, timeout=10)  # REDUCED from 15
                    response.raise_for_status()
                    run_id = response.json()['data']['id']
                    logger.info(f"Successfully started Apify run with ID: {run_id}")
                except Exception as e:
                    logger.error(f"Error starting Apify crawl: {e}")
                    return None, apify_crawler.detect_error_type(str(e))
                
                # Wait for crawl to complete
                endpoint = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={apify_crawler.APIFY_API_TOKEN}"
                
                max_attempts = 2  # REDUCED from 4
                
                for attempt in range(max_attempts):
                    logger.info(f"Checking Apify crawl results, attempt {attempt+1}/{max_attempts}")
                    
                    try:
                        response = requests.get(endpoint, timeout=8)  # REDUCED from 10
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            if data:
                                combined_text = ' '.join(item.get('text', '') for item in data if item.get('text'))
                                
                                if combined_text and len(combined_text.strip()) > 100:
                                    logger.info(f"Apify crawl completed, got {len(combined_text)} characters")
                                    return combined_text, (None, None)
                                    
                    except Exception as e:
                        logger.warning(f"Error checking Apify status: {e}")
                    
                    # REDUCED sleep times
                    if attempt < max_attempts - 1:
                        time.sleep(3)  # REDUCED from 5-10
                
                # Try direct request as fallback MUCH sooner
                logger.info(f"Trying direct request fallback...")
                
                from domain_classifier.crawlers.direct_crawler import direct_crawl
                
                # Use a shorter timeout
                direct_content, (error_type, error_detail), crawler_type = direct_crawl(url, timeout=8.0)  # REDUCED
                
                if direct_content and len(direct_content) > 0:
                    logger.info(f"Direct request fallback got content")
                    apify_crawler.track_crawler_usage("direct_fallback_from_apify")
                    return direct_content, (None, None)
                
                # Skip further Apify attempts and just return an error
                logger.warning(f"Crawl timed out after attempts")
                return None, ("timeout", "The website took too long to respond or has minimal crawlable content.")
                
            except Exception as e:
                error_type, error_detail = apify_crawler.detect_error_type(str(e))
                logger.error(f"Error crawling with Apify: {e} (Type: {error_type})")
                return None, (error_type, error_detail)
        
        # Apply the patches
        apify_crawler.crawl_website = patched_crawl_website
        apify_crawler.apify_crawl = patched_apify_crawl
        
        # 2. Patch scrapy_crawler.py
        try:
            from domain_classifier.crawlers import scrapy_crawler
            
            # Patch the scrapy_crawl function to use shorter timeouts
            original_scrapy_crawl = scrapy_crawler.scrapy_crawl
            
            def patched_scrapy_crawl(url: str, timeout: int = 15):
                """Patched scrapy_crawl with shorter timeout."""
                try:
                    logger.info(f"Starting enhanced Scrapy crawl for {url} (reduced timeout: {timeout}s)")
                    
                    # Create crawler instance with reduced timeout
                    crawler = scrapy_crawler.EnhancedScrapyCrawler()
                    
                    # Override the _run_spider method's timeout
                    original_run_spider = crawler._run_spider
                    
                    # Create a wrapper that uses the provided timeout
                    def timeout_wrapper(url):
                        import crochet
                        
                        # Use the crochet.run_in_reactor decorator with our custom timeout
                        @crochet.wait_for(timeout=timeout)
                        def _run_with_timeout():
                            return crawler.runner.crawl(scrapy_crawler.EnhancedScrapySpider, url=url)
                            
                        return _run_with_timeout()
                    
                    # Replace the method
                    crawler._run_spider = timeout_wrapper
                    
                    # Call the scrape method
                    content, error_info = crawler.scrape(url)
                    
                    # Handle the result the same way as the original
                    if isinstance(content, tuple) and len(content) == 2:
                        content_text, error_info = content
                        
                        # Log the content length for better diagnostics
                        content_length = len(content_text) if content_text else 0
                        logger.info(f"Scrapy crawl for {url} returned {content_length} characters")
                        
                        if content_text:
                            return content_text, (None, None)
                        else:
                            return None, error_info
                    else:
                        # For backwards compatibility, handle the case where content isn't a tuple
                        if content:
                            return content, (None, None)
                        else:
                            return None, ("unexpected_error", "Unexpected result format from scraper")
                            
                except Exception as e:
                    error_type, error_detail = scrapy_crawler.detect_error_type(str(e))
                    logger.error(f"Error in Enhanced Scrapy crawler: {e} (Type: {error_type})")
                    return None, (error_type, error_detail)
            
            # Apply the patch
            scrapy_crawler.scrapy_crawl = patched_scrapy_crawl
            
            # Also modify the _run_spider method timeout in the original EnhancedScrapyCrawler class
            # This is a bit hacky but ensures the timeout is properly set
            original_crochet_decorator = None
            
            for attr_name in dir(scrapy_crawler.EnhancedScrapyCrawler):
                attr = getattr(scrapy_crawler.EnhancedScrapyCrawler, attr_name)
                if attr_name == '_run_spider' and callable(attr):
                    # Get the original decorator
                    if hasattr(attr, '__wrapped__'):
                        original_crochet_decorator = attr.__wrapped__
                    
                    # Create a new decorator with reduced timeout
                    import crochet
                    new_crochet_decorator = crochet.wait_for(timeout=15.0)  # REDUCED from 45.0
                    
                    # Create a new function with the new decorator
                    @new_crochet_decorator
                    def new_run_spider(self, url):
                        return self.runner.crawl(scrapy_crawler.EnhancedScrapySpider, url=url)
                    
                    # Replace the method
                    scrapy_crawler.EnhancedScrapyCrawler._run_spider = new_run_spider
                    
                    logger.info("✅ Reduced timeout for _run_spider in EnhancedScrapyCrawler")
                    break
            
            # Also update custom_settings in the EnhancedScrapySpider class
            if hasattr(scrapy_crawler, 'EnhancedScrapySpider') and hasattr(scrapy_crawler.EnhancedScrapySpider, 'custom_settings'):
                reduced_settings = {
                    'DOWNLOAD_TIMEOUT': 15,  # REDUCED from 60
                    'RETRY_TIMES': 2,  # REDUCED from 4
                    'CONCURRENT_REQUESTS': 2,  # REDUCED from 4
                    'DOWNLOAD_MAXSIZE': 1048576,  # 1MB (REDUCED from 10MB)
                    'REDIRECT_MAX_TIMES': 5,  # REDUCED from 15
                }
                
                # Update the existing settings
                for key, value in reduced_settings.items():
                    scrapy_crawler.EnhancedScrapySpider.custom_settings[key] = value
                    
                logger.info("✅ Reduced timeouts in EnhancedScrapySpider custom_settings")
                
            logger.info("✅ Applied timeout fixes to scrapy_crawler.py")
            
        except Exception as e:
            logger.error(f"❌ Failed to patch scrapy_crawler.py for timeouts: {e}")
            
        logger.info("✅ Applied timeout fixes to crawler modules")
        
    except Exception as e:
        logger.error(f"❌ Failed to fix timeout issues: {e}")

def fix_process_did_not_complete():
    """Fix Process Did Not Complete classification to have a proper final classification."""
    try:
        from domain_classifier.utils import final_classification
        
        original_determine = final_classification.determine_final_classification
        
        def patched_determine_final_classification(result: Dict[str, Any]) -> str:
            """Directly patched version that handles Process Did Not Complete properly."""
            
            # Get the domain for logging
            domain = result.get("domain", "unknown")
            
            # Handle DNS errors first
            if (result.get("error_type") == "dns_error" or
                result.get("is_dns_error") == True or
                (isinstance(result.get("explanation", ""), str) and "DNS" in result.get("explanation", ""))):
                    
                logger.info(f"DNS error detected for {domain}, classifying as No Website available")
                return "7-No Website available"
            
            # Then handle Process Did Not Complete
            if result.get("predicted_class") == "Process Did Not Complete":
                # Check if we have Apollo data to potentially override this
                apollo_data = result.get("apollo_data", {})
                if apollo_data and isinstance(apollo_data, dict) and any(apollo_data.values()):
                    # If already reclassified using Apollo data
                    if result.get("detection_method") == "apollo_data_classification":
                        logger.info(f"Using Apollo-based classification for {domain} with Process Did Not Complete")
                        
                        # Determine based on the new predicted_class
                        predicted_class = result.get("predicted_class", "")
                        if predicted_class == "Managed Service Provider":
                            return "1-MSP"
                        elif predicted_class == "Integrator - Commercial A/V":
                            return "3-Commercial Integrator"
                        elif predicted_class == "Integrator - Residential A/V":
                            return "4-Residential Integrator"
                        elif predicted_class == "Internal IT Department":
                            return "2-Internal IT"
                        else:
                            # If we can't determine specifically, default to Internal IT
                            return "2-Internal IT"
                
                # If we couldn't override with Apollo data
                logger.info(f"No data available for {domain} with Process Did Not Complete status")
                return "8-Unknown/No Data"
            
            # Handle specific classifications
            predicted_class = result.get("predicted_class", "")
            if predicted_class == "Managed Service Provider":
                return "1-MSP"
            elif predicted_class == "Integrator - Commercial A/V":
                return "3-Commercial Integrator"
            elif predicted_class == "Integrator - Residential A/V":
                return "4-Residential Integrator"
            elif predicted_class == "Internal IT Department":
                return "2-Internal IT"
            
            # For all other cases, use the original function
            return original_determine(result)
        
        # Apply the patch
        final_classification.determine_final_classification = patched_determine_final_classification
        
        logger.info("✅ Applied Process Did Not Complete classification fix")
        
    except Exception as e:
        logger.error(f"❌ Failed to fix Process Did Not Complete classification: {e}")

def setup_enhanced_monitoring():
    """Set up enhanced monitoring for tracking classification behavior."""
    try:
        # Create a counter to track classifications
        global classification_counter
        classification_counter = {
            "total": 0,
            "dns_error": 0,
            "parked_domain": 0,
            "process_did_not_complete": 0,
            "msp": 0,
            "internal_it": 0,
            "commercial_av": 0,
            "residential_av": 0,
            "apollo_data_classification": 0
        }
        
        # Patch classify endpoint to count classifications
        try:
            from domain_classifier.api.routes import classify
            
            if hasattr(classify, 'classify_domain'):
                original_function = classify.classify_domain
                
                def monitoring_wrapper():
                    result, status_code = original_function()
                    try:
                        json_data = result.get_json()
                        if json_data:
                            # Count this classification
                            classification_counter["total"] += 1
                            
                            # Track by type
                            predicted_class = json_data.get("predicted_class", "")
                            if predicted_class == "Managed Service Provider":
                                classification_counter["msp"] += 1
                            elif predicted_class == "Internal IT Department":
                                classification_counter["internal_it"] += 1
                            elif predicted_class == "Integrator - Commercial A/V":
                                classification_counter["commercial_av"] += 1
                            elif predicted_class == "Integrator - Residential A/V":
                                classification_counter["residential_av"] += 1
                            elif predicted_class == "Process Did Not Complete":
                                classification_counter["process_did_not_complete"] += 1
                                
                            # Track by condition
                            if json_data.get("is_parked", False):
                                classification_counter["parked_domain"] += 1
                                
                            if json_data.get("error_type") == "dns_error" or json_data.get("is_dns_error", False):
                                classification_counter["dns_error"] += 1
                                
                            if json_data.get("detection_method") == "apollo_data_classification":
                                classification_counter["apollo_data_classification"] += 1
                            
                            # Periodically log stats
                            if classification_counter["total"] % 10 == 0:
                                logger.info(f"Classification stats: {classification_counter}")
                                
                    except Exception as e:
                        logger.warning(f"Monitoring error (non-critical): {e}")
                        
                    return result, status_code
                
                # Apply monitoring wrapper
                classify.classify_domain = monitoring_wrapper
                
                logger.info("✅ Enhanced monitoring applied to classify_domain endpoint")
                
        except Exception as e:
            logger.warning(f"Non-critical: Could not apply monitoring patch: {e}")
            
        # Also add monitoring to the API formatter
        try:
            from domain_classifier.utils import api_formatter
            
            original_format = api_formatter.format_api_response
            
            def monitoring_format_wrapper(result: Dict[str, Any]) -> Dict[str, Any]:
                """Wrapper to monitor API formatting and ensure fields are set correctly."""
                
                # Handle special cases before formatting
                
                # For DNS errors, ensure proper fields
                if result.get("error_type") == "dns_error" or result.get("is_dns_error") == True:
                    domain = result.get("domain", "unknown")
                    result["final_classification"] = "7-No Website available"
                    result["predicted_class"] = "DNS Error"
                    result["company_description"] = f"The domain {domain} could not be resolved. It may not exist or DNS records may be misconfigured."
                    result["company_one_line"] = f"Domain cannot be reached - DNS error."
                    
                # For Process Did Not Complete, ensure proper classification
                elif result.get("predicted_class") == "Process Did Not Complete":
                    # Check if Apollo data allowed classification override
                    if result.get("detection_method") == "apollo_data_classification":
                        logger.info(f"Process Did Not Complete but classified with Apollo data for {result.get('domain', 'unknown')}")
                    else:
                        result["final_classification"] = "8-Unknown/No Data"
                
                # Add default confidence score if missing
                if not result.get("confidence_score"):
                    result["confidence_score"] = 50
                    logger.warning(f"Added missing confidence_score for {result.get('domain', 'unknown')}")
                    
                # Add default confidence scores if missing
                if not result.get("confidence_scores"):
                    result["confidence_scores"] = {
                        "Managed Service Provider": 10,
                        "Integrator - Commercial A/V": 10,
                        "Integrator - Residential A/V": 10,
                        "Internal IT Department": 70
                    }
                    logger.warning(f"Added missing confidence_scores for {result.get('domain', 'unknown')}")
                
                # Call original formatter
                formatted = original_format(result)
                
                # Ensure critical fields are in the output
                domain = result.get("domain", "unknown")
                
                # Force final_classification based on special cases
                if result.get("predicted_class") == "Process Did Not Complete" and result.get("detection_method") != "apollo_data_classification":
                    if "02_final_classification" in formatted:
                        formatted["02_final_classification"] = "8-Unknown/No Data"
                
                # For DNS errors, ensure critical fields
                if result.get("error_type") == "dns_error" or result.get("is_dns_error") == True:
                    if "02_final_classification" in formatted:
                        formatted["02_final_classification"] = "7-No Website available"
                    if "02_classification" in formatted:
                        formatted["02_classification"] = "DNS Error"
                    if "03_description" in formatted:
                        formatted["03_description"] = f"The domain {domain} could not be resolved. It may not exist or DNS records may be misconfigured."
                
                # Ensure confidence score is in the output
                if "01_confidence_score" not in formatted and "confidence_score" in result:
                    formatted["01_confidence_score"] = result["confidence_score"]
                    
                return formatted
            
            # Apply the patch
            api_formatter.format_api_response = monitoring_format_wrapper
            
            logger.info("✅ Enhanced monitoring and corrections applied to API formatter")
            
        except Exception as e:
            logger.warning(f"Non-critical: Could not apply API formatter monitoring: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error setting up enhanced monitoring: {e}")

def add_dns_error_interceptor():
    """Add a DNS error interceptor at the Flask level to catch DNS errors early."""
    try:
        # Use Flask request processing to catch DNS errors
        import flask
        from flask import request, jsonify
        
        # Create a decorator that can be applied to route handlers
        def dns_error_handler(app):
            @app.before_request
            def check_dns_before_request():
                # Only intercept POST requests to /classify-and-enrich
                if request.method == 'POST' and request.path == '/classify-and-enrich':
                    try:
                        # Extract data from request
                        data = request.json
                        if not data:
                            return None
                            
                        input_value = data.get('url', '').strip()
                        if not input_value:
                            return None
                            
                        # Determine if input is an email
                        is_email = '@' in input_value
                        email = input_value if is_email else None
                        
                        # Extract domain
                        domain = None
                        if is_email:
                            from domain_classifier.utils.domain_utils import extract_domain_from_email
                            domain = extract_domain_from_email(input_value)
                        else:
                            from domain_classifier.utils.domain_utils import normalize_domain
                            domain = normalize_domain(input_value)
                            
                        if not domain:
                            return None
                            
                        # Create URL for checking
                        url = f"https://{domain}"
                        
                        # Check for DNS resolution
                        from domain_classifier.utils.error_handling import is_domain_worth_crawling
                        worth_crawling, has_dns, error_msg, potentially_flaky = is_domain_worth_crawling(domain)
                        
                        # For DNS errors, create a response and return immediately
                        if not has_dns and (error_msg == "dns_error" or "dns" in str(error_msg).lower()):
                            logger.info(f"DNS error detected for {domain}, returning DNS error response")
                            
                            from domain_classifier.utils.error_handling import create_error_result
                            
                            error_result = create_error_result(
                                domain,
                                "dns_error",
                                error_msg if error_msg != "dns_error" else None,
                                email,
                                "early_check"
                            )
                            error_result["website_url"] = url
                            error_result["final_classification"] = "7-No Website available"
                            error_result["predicted_class"] = "DNS Error"
                            
                            # Format the response
                            try:
                                from domain_classifier.utils.api_formatter import format_api_response
                                formatted_result = format_api_response(error_result)
                                return jsonify(formatted_result), 200
                            except Exception as format_error:
                                logger.error(f"Error formatting DNS error response: {format_error}")
                                return jsonify(error_result), 200
