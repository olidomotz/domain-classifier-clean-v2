"""Fixed classify_routes.py to properly return app and use LLM."""

import logging
import traceback
from flask import request, jsonify
from urllib.parse import urlparse
from typing import Dict, Any, Tuple, Optional

# Import domain utilities
from domain_classifier.utils.domain_utils import extract_domain_from_email
from domain_classifier.utils.error_handling import detect_error_type, create_error_result, check_domain_dns, is_domain_worth_crawling
from domain_classifier.utils.text_processing import extract_company_description, ensure_classifier_type

# Import configuration
from domain_classifier.config.overrides import check_domain_override

# Import services
from domain_classifier.crawlers.apify_crawler import crawl_website
from domain_classifier.storage.operations import save_to_snowflake
from domain_classifier.classifiers.result_validator import validate_result_consistency
from domain_classifier.storage.cache_manager import process_cached_result
from domain_classifier.storage.result_processor import process_fresh_result

# Import the API formatter if available
try:
    from domain_classifier.utils.api_formatter import format_api_response
    has_api_formatter = True
except ImportError:
    has_api_formatter = False

# Set up logging
logger = logging.getLogger(__name__)

def register_classify_routes(app, llm_classifier, snowflake_conn):
    """Register domain/email classification related routes."""
    if app is None:
        logger.error("App object is None in register_classify_routes")
        from flask import Flask
        app = Flask(__name__)
        logger.info("Created new Flask app as fallback")

    @app.route('/classify-domain', methods=['POST', 'OPTIONS'])
    def classify_domain():
        """Direct API that classifies a domain or email and returns the result"""
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return '', 204

        try:
            data = request.json
            input_value = data.get('url', '').strip()
            
            # Change default for force_reclassify to False
            force_reclassify = data.get('force_reclassify', False)
            use_existing_content = data.get('use_existing_content', True)

            # New option to control vector classification
            use_vector_classification = data.get('use_vector_classification', True)
            
            # New option to control response format
            use_new_format = data.get('use_new_format', True) if has_api_formatter else False

            if not input_value:
                return jsonify({"error": "URL or email is required"}), 400

            # Check if input is an email
            is_email = '@' in input_value
            email = None
            
            if is_email:
                # Extract domain from email
                email = input_value
                domain = extract_domain_from_email(email)
                
                if not domain:
                    return jsonify({"error": "Invalid email format"}), 400
                    
                logger.info(f"Extracted domain '{domain}' from email '{email}'")
            else:
                # Process as domain/URL
                # Format URL properly
                if not input_value.startswith('http'):
                    input_value = 'https://' + input_value
                    
                # Extract domain
                parsed_url = urlparse(input_value)
                domain = parsed_url.netloc
                
                if not domain:
                    domain = parsed_url.path
                    
                # Remove www. if present
                if domain.startswith('www.'):
                    domain = domain[4:]
                    
                if not domain:
                    return jsonify({"error": "Invalid URL or email"}), 400
                    
            url = f"https://{domain}"
            
            logger.info(f"Processing classification request for {domain}")
            
            # Check for domain override before any other processing
            domain_override = check_domain_override(domain)
            
            if domain_override:
                # Add email to response if input was an email
                if email:
                    domain_override["email"] = email
                    
                # Add website URL for clickable link
                domain_override["website_url"] = url
                
                # Add final classification based on predicted class
                if domain_override.get("predicted_class") == "Managed Service Provider":
                    domain_override["final_classification"] = "1-MSP"
                elif domain_override.get("predicted_class") == "Integrator - Commercial A/V":
                    domain_override["final_classification"] = "3-Commercial Integrator"
                elif domain_override.get("predicted_class") == "Integrator - Residential A/V":
                    domain_override["final_classification"] = "4-Residential Integrator"
                else:
                    domain_override["final_classification"] = "2-Internal IT"
                    
                # Add crawler_type and classifier_type to ensure they appear at the bottom
                domain_override["crawler_type"] = "override"
                domain_override["classifier_type"] = "override"
                
                # Return the override directly
                logger.info(f"Sending override response to client: {domain_override}")
                
                # Format the response if requested
                if use_new_format and has_api_formatter:
                    return jsonify(format_api_response(domain_override)), 200
                else:
                    return jsonify(domain_override), 200
                    
            # Enhanced domain screening before attempting crawl
            worth_crawling, has_dns, dns_error, potentially_flaky = is_domain_worth_crawling(domain)
            
            # Check for parked domain early
            if dns_error == "parked_domain":
                logger.info(f"Domain {domain} detected as parked domain during initial DNS check")
                
                from domain_classifier.classifiers.decision_tree import create_parked_domain_result
                parked_result = create_parked_domain_result(domain, crawler_type="dns_check_parked")
                result = process_fresh_result(parked_result, domain, email, url)
                result["crawler_type"] = "dns_check_parked"
                result["classifier_type"] = "early_detection"
                
                # Format the response if requested
                if use_new_format and has_api_formatter:
                    return jsonify(format_api_response(result)), 200
                else:
                    return jsonify(result), 200
                    
            if not worth_crawling:
                logger.warning(f"Domain {domain} is not worth crawling: {dns_error}")
                
                error_result = create_error_result(domain, "dns_error" if "DNS" in dns_error else "connection_error", 
                                               dns_error, email, crawler_type)
                error_result["website_url"] = url
                error_result["final_classification"] = "7-No Website available"
                
                # Format the response if requested
                if use_new_format and has_api_formatter:
                    return jsonify(format_api_response(error_result)), 503  # Service Unavailable
                else:
                    return jsonify(error_result), 503  # Service Unavailable
                    
            # If domain is worth crawling, do an additional early check for parked domains
            if worth_crawling:
                # Add early parked domain detection
                is_parked, parked_result = check_for_parked_domain(domain, url)
                
                if is_parked:
                    # Create result for the API
                    from domain_classifier.classifiers.decision_tree import create_parked_domain_result
                    
                    parked_domain_data = create_parked_domain_result(domain, crawler_type="quick_check_parked")
                    
                    # Process the result
                    result = process_fresh_result(parked_domain_data, domain, email, url)
                    
                    # Add to final result
                    result["crawler_type"] = "quick_check_parked"
                    result["classifier_type"] = "early_detection"
                    
                    # Format the response if requested
                    if use_new_format and has_api_formatter:
                        return jsonify(format_api_response(result)), 200
                    else:
                        return jsonify(result), 200

            if potentially_flaky:
                logger.warning(f"Domain {domain} passed basic checks but shows signs of being flaky (resetting connections)")
                # We'll still try to crawl, but warn the user that it might be unreliable
            else:
                logger.info(f"DNS check passed for domain: {domain}")
                
            # Check for existing classification if not forcing reclassification
            if not force_reclassify:
                existing_record = snowflake_conn.check_existing_classification(domain)
                
                if existing_record:
                    logger.info(f"Found existing classification for {domain}")
                    
                    # Process and return the cached result
                    result = process_cached_result(existing_record, domain, email, url)
                    
                    # Ensure result consistency
                    result = validate_result_consistency(result, domain)
                    
                    # Log the response for debugging
                    logger.info(f"Sending cached response to client")
                    
                    # Format the response if requested
                    if use_new_format and has_api_formatter:
                        return jsonify(format_api_response(result)), 200
                    else:
                        return jsonify(result), 200
                        
            # Try to get content (either from DB or by crawling)
            content = None
            
            # If reclassifying AND using existing content is explicitly true, try to get existing content first
            # CRITICAL FIX: Only check existing content if use_existing_content is True
            if use_existing_content:
                try:
                    content = snowflake_conn.get_domain_content(domain)
                    
                    if content:
                        logger.info(f"Using existing content for {domain}")
                        
                        # Set crawler_type for existing content
                        crawler_type = "existing_content"
                        
                        # Add check for parked domain in cached content
                        from domain_classifier.classifiers.decision_tree import is_parked_domain
                        
                        if is_parked_domain(content, domain):
                            logger.info(f"Detected parked domain from cached content: {domain}")
                            
                            from domain_classifier.classifiers.decision_tree import create_parked_domain_result
                            parked_result = create_parked_domain_result(domain, crawler_type="cached_content_parked")
                            
                            # Process the parked domain result
                            result = process_fresh_result(parked_result, domain, email, url)
                            
                            # Add crawler_type and classifier_type to ensure they appear at the bottom
                            result["crawler_type"] = "cached_content_parked"
                            result["classifier_type"] = "early_detection"
                            
                            # Format the response if requested
                            if use_new_format and has_api_formatter:
                                return jsonify(format_api_response(result)), 200
                            else:
                                return jsonify(result), 200
                except Exception as e:
                    logger.warning(f"Could not get existing content, will crawl instead: {e}")
                    content = None
                    
            # If we specifically requested to use existing content but none was found
            if use_existing_content and not content:
                # CRITICAL FIX: Don't return a 404 error, continue with crawling
                logger.warning(f"No existing content found for {domain}, will attempt to crawl")
            
            # If no content yet, crawl the website
            error_type = None
            error_detail = None
            
            if not content:
                # CRITICAL CHANGE: Make sure we log this clearly
                logger.info(f"Crawling website for {domain} because no content was found or use_existing_content=False")
                
                content, (error_type, error_detail), crawler_type = crawl_website(url)
                
                if not content:
                    # CRITICAL: Log this explicitly 
                    logger.warning(f"Crawl returned no content for {domain} - error: {error_type}, {error_detail}")
                    
                    # Still try to classify based on domain name only
                    logger.info(f"Attempting LLM classification with domain name only for {domain}")
                    
                    # Check for special domain cases to enhance LLM classification
                    from domain_classifier.classifiers.decision_tree import check_special_domain_cases
                    
                    special_case_enhancements = check_special_domain_cases(domain, content)

            # Whether content succeeded or failed, always use the LLM classifier
            if not llm_classifier:
                error_result = {
                    "domain": domain,
                    "error": "LLM classifier is not available",
                    "predicted_class": "Unknown",
                    "confidence_score": 0,
                    "confidence_scores": {
                        "Managed Service Provider": 0,
                        "Integrator - Commercial A/V": 0,
                        "Integrator - Residential A/V": 0,
                        "Internal IT Department": 0
                    },
                    "explanation": "Our classification system is temporarily unavailable. Please try again later.",
                    "low_confidence": True,
                    "website_url": url,
                    "final_classification": "2-Internal IT",
                    "crawler_type": crawler_type or "error_handler",
                    "classifier_type": "error_handler"
                }
                
                # Add email to error response if input was an email
                if email:
                    error_result["email"] = email
                    
                # Format the response if requested
                if use_new_format and has_api_formatter:
                    return jsonify(format_api_response(error_result)), 500
                else:
                    return jsonify(error_result), 500
                    
            # CRITICAL FIX: Always use the LLM classifier, even if content is minimal or missing
            logger.info(f"Classifying content for {domain}")
            
            classification = llm_classifier.classify(
                content=content,  # This might be None, but the LLM can still classify based on domain
                domain=domain,
                use_vector_classification=use_vector_classification
            )
            
            # If classification fails, return an error
            if not classification:
                error_result = {
                    "domain": domain,
                    "error": "Classification failed",
                    "predicted_class": "Unknown",
                    "confidence_score": 0,
                    "confidence_scores": {
                        "Managed Service Provider": 0,
                        "Integrator - Commercial A/V": 0,
                        "Integrator - Residential A/V": 0,
                        "Internal IT Department": 0
                    },
                    "explanation": f"We encountered an issue while analyzing {domain}.",
                    "low_confidence": True,
                    "website_url": url,
                    "final_classification": "2-Internal IT",
                    "crawler_type": crawler_type or "unknown",
                    "classifier_type": "error_handler"
                }
                
                # Add email to error response if input was an email
                if email:
                    error_result["email"] = email
                    
                # Format the response if requested
                if use_new_format and has_api_formatter:
                    return jsonify(format_api_response(error_result)), 500
                else:
                    return jsonify(error_result), 500
                    
            # Add crawler_type to the classification if available
            if crawler_type:
                classification["crawler_type"] = crawler_type
                
            # UPDATED: Use ensure_classifier_type helper function instead of direct assignment
            # This avoids warnings about classifier_type not being found
            classification = ensure_classifier_type(classification, domain, default_type="claude-llm")
            
            # CRITICAL FIX: If we have special case enhancements, apply them to the LLM classification
            if special_case_enhancements:
                logger.info(f"Enhancing LLM classification with special case data for {domain}")
                
                # Add special domain info to detection method
                classification["detection_method"] = f"llm_with_{special_case_enhancements.get('detection_method', 'special_knowledge')}"
                
                # If special case and LLM classification agree, boost confidence
                if (special_case_enhancements.get("suggested_class") == classification["predicted_class"] and
                    "confidence_boost" in special_case_enhancements):
                    
                    boost = special_case_enhancements["confidence_boost"]
                    logger.info(f"Boosting confidence by {boost} due to special case match")
                    
                    # Apply to max_confidence
                    if "max_confidence" in classification:
                        classification["max_confidence"] = min(0.95, classification["max_confidence"] + boost)
                        
                    # Apply to confidence_scores for the specific class
                    if "confidence_scores" in classification:
                        pred_class = classification["predicted_class"]
                        
                        if pred_class in classification["confidence_scores"]:
                            current_score = classification["confidence_scores"][pred_class]
                            
                            # If score is 0-1 range
                            if isinstance(current_score, float) and current_score <= 1.0:
                                classification["confidence_scores"][pred_class] = min(0.95, current_score + boost)
                            # If score is 0-100 range
                            else:
                                classification["confidence_scores"][pred_class] = min(95, current_score + int(boost * 100))
                                
            # CRITICAL FIX: Ensure detection_method is not too long for Snowflake
            if "detection_method" in classification:
                if len(classification["detection_method"]) > 40:
                    logger.warning(f"Truncating detection_method for {domain} to fit in Snowflake")
                    classification["detection_method"] = classification["detection_method"][:40]
                    
            # CRITICAL FIX: Ensure domain and URL is included in classification
            if "domain" not in classification:
                classification["domain"] = domain
                
            if "website_url" not in classification:
                classification["website_url"] = url
                
            if email and "email" not in classification:
                classification["email"] = email
                
            # Log the successful classification
            logger.info(f"✅ LLM classification completed for {domain} - Result: {classification.get('predicted_class', 'Unknown')}")
            
            # Save to Snowflake and Vector DB (always save, even for reclassifications)
            save_to_snowflake(
                domain=domain,
                url=url,
                content=content,
                classification=classification,
                snowflake_conn=snowflake_conn,
                crawler_type=crawler_type,
                classifier_type=classification.get("classifier_type")  # Use the set classifier_type
            )
            
            # Process the fresh classification result
            result = process_fresh_result(classification, domain, email, url)
            
            # Add crawler_type to the result if not already included
            if crawler_type and "crawler_type" not in result:
                result["crawler_type"] = crawler_type
                
            # Ensure classifier_type is set properly in the result
            result = ensure_classifier_type(result, domain)
            
            # Ensure result consistency
            result = validate_result_consistency(result, domain)
            
            # Add company description if not already present
            if "company_description" not in result:
                result["company_description"] = extract_company_description(content, result.get("explanation", ""), domain)
                
            # Log the response for debugging
            logger.info(f"Sending fresh response to client")
            
            # Format the response if requested
            if use_new_format and has_api_formatter:
                return jsonify(format_api_response(result)), 200
            else:
                return jsonify(result), 200
                
        except Exception as e:
            logger.error(f"Error processing request: {e}\n{traceback.format_exc()}")
            
            # Try to identify the error type if possible
            error_type, error_detail = detect_error_type(str(e))
            
            error_result = create_error_result(
                domain if 'domain' in locals() else "unknown",
                error_type,
                error_detail,
                email if 'email' in locals() else None,
                crawler_type if 'crawler_type' in locals() else "exception_handler"
            )
            
            error_result["error"] = str(e)  # Add the actual error message
            
            if 'url' in locals():
                error_result["website_url"] = url
                
            # Format the response if requested
            if use_new_format and has_api_formatter:
                return jsonify(format_api_response(error_result)), 500
            else:
                return jsonify(error_result), 500
                
    # Helper function for parked domain detection
    def check_for_parked_domain(domain: str, url: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform a quick check to determine if a domain is parked before full crawling.
        
        Args:
            domain (str): The domain name
            url (str): The full URL
            
        Returns:
            tuple: (is_parked, result)
                - is_parked: Whether the domain is parked
                - result: The result dict if parked, None otherwise
        """
        try:
            from domain_classifier.crawlers.direct_crawler import direct_crawl
            from domain_classifier.classifiers.decision_tree import is_parked_domain, create_parked_domain_result
            
            logger.info(f"Performing quick check for parked domain: {domain}")
            
            quick_check_content, (error_type, error_detail), quick_crawler_type = direct_crawl(url, timeout=5.0)
            
            # Check if this is a parked domain
            if error_type == "is_parked" or (quick_check_content and is_parked_domain(quick_check_content, domain)):
                logger.info(f"Quick check detected parked domain: {domain}")
                
                parked_result = create_parked_domain_result(domain, crawler_type="quick_check_parked")
                
                # Process the result from the decision tree
                result = {
                    "domain": domain,
                    "predicted_class": "Parked Domain",
                    "confidence_score": 0,
                    "confidence_scores": {
                        "Managed Service Provider": 0,
                        "Integrator - Commercial A/V": 0,
                        "Integrator - Residential A/V": 0,
                        "Internal IT Department": 0
                    },
                    "explanation": parked_result.get("llm_explanation", f"The domain {domain} appears to be parked or inactive. This domain may be registered but not actively in use for a business."),
                    "low_confidence": True,
                    "is_parked": True,
                    "final_classification": "6-Parked Domain - no enrichment",
                    "crawler_type": "quick_check_parked",
                    "classifier_type": "early_detection",
                    "detection_method": "parked_domain_detection",
                    "source": "fresh"
                }
                
                return True, result
                
            # If we get here, domain is not parked
            return False, None
            
        except Exception as e:
            logger.warning(f"Early parked domain check failed: {e}")
            return False, None
            
    # CRITICAL: Return the app object
    return app
