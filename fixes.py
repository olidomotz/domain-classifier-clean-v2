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
                                       error_detail: Optional[str] = None, email: Optional[str] = None,
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
        
        # Patch enrich.py to stop processing for DNS errors
        try:
            from domain_classifier.api.routes import enrich
            
            # Store original function
            original_classify_and_enrich = enrich.classify_and_enrich
            
            def patched_classify_and_enrich():
                """Patched version that properly handles DNS errors."""
                # Check if this is a POST request
                from flask import request
                if request.method != 'POST':
                    return original_classify_and_enrich()
                    
                try:
                    # Extract data from request
                    data = request.json
                    input_value = data.get('url', '').strip()
                    
                    if not input_value:
                        return original_classify_and_enrich()
                        
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
                        return original_classify_and_enrich()
                        
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
                            from flask import jsonify
                            formatted_result = format_api_response(error_result)
                            return jsonify(formatted_result), 200
                        except Exception as format_error:
                            logger.error(f"Error formatting DNS error response: {format_error}")
                            from flask import jsonify
                            return jsonify(error_result), 200
                    
                    # For all other cases, use the original function
                    return original_classify_and_enrich()
                    
                except Exception as e:
                    logger.error(f"Error in patched classify_and_enrich: {e}")
                    return original_classify_and_enrich()
            
            # Apply the patch
            enrich.classify_and_enrich = patched_classify_and_enrich
            logger.info("✅ Applied DNS error handling fix to classify_and_enrich function")
            
        except Exception as e:
            logger.error(f"❌ Failed to patch enrich.py for DNS error handling: {e}")
        
        logger.info("✅ Applied DNS error detection fixes to error_handling.py")
        
    except Exception as e:
        logger.error(f"❌ Failed to fix DNS error detection: {e}")

def fix_parked_domain_detection():
    """Fix parked domain detection to catch more cases like crapanzano.net."""
    try:
        from domain_classifier.classifiers import decision_tree
        
        original_is_parked = decision_tree.is_parked_domain
        
        def enhanced_is_parked_domain(content: str, domain: str = None) -> bool:
            """Enhanced detection of truly parked domains vs. just having minimal content."""
            if not content:
                logger.info("Domain has no content at all, considering as parked")
                return True
                
            content_lower = content.lower()
            
            # Special case for GoDaddy proxy errors (for cases like crapanzano.net)
            # This has to be checked first to catch proxy errors
            if ("proxy error" in content_lower or "connection refused" in content_lower) and len(content) < 500:
                logger.info("Found proxy error message in minimal content, likely parked or inactive")
                return True
            
            # 1. Direct explicit parking phrases
            explicit_parking_phrases = [
                "domain is for sale", "buy this domain", "purchasing this domain",
                "domain may be for sale", "this domain is for sale", "parked by",
                "domain parking", "this web page is parked", "domain for sale",
                "this website is for sale", "domain name parking",
                "purchase this domain", "domain has expired", "domain available",
                "domain not configured", "inquire about this domain", 
                "this domain is available for purchase", "domain has been registered",
                "domain has expired", "reserve this domain name", "bid on this domain"
            ]
            
            # 2. Expanded list of hosting providers and registrars
            hosting_providers = [
                "godaddy", "hostgator", "bluehost", "namecheap", "dreamhost",
                "domain registration", "web hosting service", "hosting provider",
                "register this domain", "domain broker", "proxy error", "error connecting",
                "domain has expired", "domain has been registered", "courtesy page",
                "ionos", "domain.com", "hover", "namesilo", "porkbun", 
                "network solutions", "register.com", "name.com", "enom", 
                "dynadot", "hover", "domainking", "domainmonster", "1and1", 
                "1&1", "ionos", "registrar", "dominio", "parkingcrew"
            ]
            
            # 3. Technical issues phrases
            technical_phrases = [
                "proxy error", "error connecting", "connection error", "courtesy page",
                "site not found", "domain not configured", "default web page",
                "website coming soon", "under construction", "future home of",
                "site temporarily unavailable", "domain has been registered",
                "refused to connect", "connection refused", "this page isn't working"
            ]
            
            # Count explicit parking phrases
            explicit_matches = sum(1 for phrase in explicit_parking_phrases if phrase in content_lower)
            
            # If we have 1 or more explicit parking indicators, check for additional evidence
            if explicit_matches >= 1:
                logger.info(f"Domain contains explicit parking phrases, checking for additional evidence")
                
                # Check for hosting providers
                hosting_matches = sum(1 for phrase in hosting_providers if phrase in content_lower)
                if hosting_matches >= 1:
                    logger.info(f"Domain contains explicit parking phrase and hosting provider reference, considering as parked")
                    return True
                    
                # Check for technical issues phrases
                tech_matches = sum(1 for phrase in technical_phrases if phrase in content_lower)
                if tech_matches >= 1:
                    logger.info(f"Domain contains explicit parking phrase and technical issues, considering as parked")
                    return True
            
            # 4. Common hosting/registrar parking indicators with technical issues
            hosting_matches = sum(1 for phrase in hosting_providers if phrase in content_lower)
            tech_matches = sum(1 for phrase in technical_phrases if phrase in content_lower)
            
            # If multiple indicators, likely parked
            if (hosting_matches >= 2) or (hosting_matches >= 1 and tech_matches >= 1):
                logger.info(f"Domain contains multiple hosting/registrar/technical indicators, considering as parked")
                return True
            
            # If minimal content and contains technical issues phrases, likely parked
            if len(content.strip()) < 300 and tech_matches >= 1:
                logger.info(f"Domain contains minimal content with technical issues, considering as parked")
                return True
            
            # Fall back to original implementation for other cases
            return original_is_parked(content, domain)
        
        # Apply the patch
        decision_tree.is_parked_domain = enhanced_is_parked_domain
        logger.info("✅ Applied parked domain detection fixes to decision_tree.py")
        
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
                                
                                # Add a note about using Apollo data
                                return f"{description}\n\nNote: This description is based on Apollo data as the website could not be analyzed."
                    except Exception as e:
                        logger.error(f"Error classifying with Apollo data: {e}")
            
            # For all other cases, use the original function
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
                
                # Variables to track direct crawl results
                direct_content = None
                direct_error_type = None
                direct_error_detail = None
                direct_crawler_type = None
                
                # Try direct crawler first
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
                    "timeoutSecs": 20,   # REDUCED from 30
                    "maxRequestRetries": 2,  # REDUCED from 3
                    "maxRedirects": 5,   # REDUCED from 10
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
                    'RETRY_TIMES': 2,        # REDUCED from 4
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
                        else:
                            return "2-Internal IT"
                
                # If we couldn't override with Apollo data
                logger.info(f"No data available for {domain} with Process Did Not Complete status")
                return "8-Unknown/No Data"
            
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
                    result["company_description"] = f"The domain {domain} could not be resolved. It may not exist or its DNS records may be misconfigured."
                    result["company_one_line"] = f"Domain cannot be reached - DNS error."
                
                # For Process Did Not Complete, ensure proper classification
                elif result.get("predicted_class") == "Process Did Not Complete":
                    # Check if Apollo data allowed classification override
                    if result.get("detection_method") == "apollo_data_classification":
                        logger.info(f"Process Did Not Complete but classified with Apollo data for {result.get('domain', 'unknown')}")
                    else:
                        result["final_classification"] = "8-Unknown/No Data"
                
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
                        formatted["03_description"] = f"The domain {domain} could not be resolved. It may not exist or its DNS records may be misconfigured."
                
                return formatted
            
            # Apply the patch
            api_formatter.format_api_response = monitoring_format_wrapper
            logger.info("✅ Enhanced monitoring and corrections applied to API formatter")
            
        except Exception as e:
            logger.warning(f"Non-critical: Could not apply API formatter monitoring: {e}")
            
    except Exception as e:
        logger.error(f"❌ Error setting up enhanced monitoring: {e}")
