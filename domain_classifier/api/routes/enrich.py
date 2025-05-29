"""Complete fixed enrich.py to properly handle formatted classification responses and DNS errors."""

import logging
import traceback
import re
import json
from flask import request, jsonify, current_app

# Import utilities
from domain_classifier.utils.error_handling import detect_error_type, create_error_result, is_domain_worth_crawling
from domain_classifier.storage.operations import save_to_snowflake
from domain_classifier.utils.final_classification import determine_final_classification
from domain_classifier.classifiers.decision_tree import create_parked_domain_result, is_parked_domain, check_industry_context
from domain_classifier.storage.result_processor import process_fresh_result
from domain_classifier.enrichment.ai_data_extractor import extract_company_data_from_content
from domain_classifier.utils.domain_analysis import analyze_domain_words
from domain_classifier.utils.cross_validator import reconcile_classification
from domain_classifier.utils.json_parser import ensure_dict, safe_get, clean_json_string
from domain_classifier.utils.domain_utils import extract_domain_from_email, normalize_domain
from domain_classifier.config.overrides import check_domain_override
from domain_classifier.utils.text_processing import ensure_classifier_type

# Import the API formatter if available
try:
    from domain_classifier.utils.api_formatter import format_api_response
    has_api_formatter = True
except ImportError:
    has_api_formatter = False

# Set up logging
logger = logging.getLogger(__name__)

def register_enrich_routes(app, snowflake_conn):
    """Register enrichment related routes."""
    logger.info("Registering enrichment routes")
    
    if app is None:
        logger.error("App object is None in register_enrich_routes")
        from flask import Flask
        app = Flask(__name__)
        logger.info("Created new Flask app as fallback in enrich.py")

    @app.route('/store-classification-data', methods=['POST'])
    def store_classification_data():
        """
        Endpoint for storing classification data from n8n workflow.
        Stores both content and classification in Snowflake.
        """
        from domain_classifier.storage.snowflake_connector import SnowflakeConnector
        
        logger = logging.getLogger(__name__)
        data = request.json
        
        if not data or not data.get('domain'):
            return jsonify({
                "success": False,
                "message": "Invalid request: domain is required"
            }), 400
        
        domain = data.get('domain')
        classification = data.get('classification', {})
        company_info = data.get('company_info', {})
        apollo_data = data.get('apollo_data', {})
        content = data.get('content')
        crawl_stats = data.get('crawl_stats', {})
        
        logger.info(f"Received n8n store request for domain: {domain}")
        
        # Initialize Snowflake connector
        snowflake_conn = SnowflakeConnector()
        
        # Process and store the content if provided
        content_stored = False
        if content:
            try:
                url = company_info.get('url', f"https://{domain}")
                success, error = snowflake_conn.save_domain_content(
                    domain=domain,
                    url=url,
                    content=content
                )
                content_stored = success
                if not success:
                    logger.error(f"Failed to store content for {domain}: {error}")
            except Exception as e:
                logger.error(f"Error storing content for {domain}: {str(e)}")
        
        # Process and store the classification
        classification_stored = False
        try:
            # Prepare classification data
            predicted_class = classification.get('predicted_class', 'Unknown')
            confidence_score = classification.get('confidence_score', 0)
            confidence_scores = json.dumps(classification.get('confidence_scores', {}))
            is_service_business = classification.get('is_service_business', False)
            explanation = classification.get('explanation', '')
            internal_it_potential = classification.get('internal_it_potential', 0)
            
            # Prepare Apollo data
            apollo_json = json.dumps(apollo_data) if apollo_data else None
            
            # Store in Snowflake
            success, error = snowflake_conn.save_classification(
                domain=domain,
                company_type=predicted_class,
                confidence_score=confidence_score,
                all_scores=confidence_scores,
                model_metadata=json.dumps({"source": "n8n_workflow", "crawler": crawl_stats}),
                low_confidence=confidence_score < 30,
                detection_method="n8n_claude_agent",
                llm_explanation=explanation,
                apollo_company_data=apollo_json,
                crawler_type="go_crawler",
                classifier_type="claude-ai-agent"
            )
            classification_stored = success
            if not success:
                logger.error(f"Failed to store classification for {domain}: {error}")
        except Exception as e:
            logger.error(f"Error storing classification for {domain}: {str(e)}")
        
        return jsonify({
            "success": content_stored and classification_stored,
            "message": "Data processed for Snowflake storage",
            "domain": domain,
            "content_stored": content_stored,
            "classification_stored": classification_stored
        })

    @app.route('/classify-and-enrich', methods=['POST', 'OPTIONS'])
    def classify_and_enrich():
        """Classify a domain and enrich it with Apollo data"""
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return '', 204
        
        try:
            data = request.json
            input_value = data.get('url', '').strip()
            force_reclassify = data.get('force_reclassify', True)  # CHANGED default to True to force classification
            
            # Add parameter to control response format
            use_new_format = data.get('use_new_format', True) if has_api_formatter else False
            
            if not input_value:
                return jsonify({"error": "URL or email is required"}), 400
            
            # CRITICAL: Add detailed logging
            logger.info("=" * 80)
            logger.info(f"CLASSIFY AND ENRICH REQUEST for: {input_value}")
            logger.info(f"Force reclassify: {force_reclassify}")
            logger.info("=" * 80)
            
            # Determine if input is an email
            is_email = '@' in input_value
            email = input_value if is_email else None
            
            # Extract domain for checking
            domain = None
            if is_email and '@' in input_value:
                domain = extract_domain_from_email(input_value)
                if not domain:
                    return jsonify({"error": "Invalid email format"}), 400
            else:
                # Basic domain extraction for URL
                domain = normalize_domain(input_value)
            
            # Create URL for checks and displaying
            url = f"https://{domain}"
            
            # Cache for storing domain content to avoid multiple queries
            domain_content_cache = None
            
            # Check for domain override before any other processing
            domain_override = check_domain_override(domain) if 'check_domain_override' in globals() else None
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
                
                # Return the override directly with appropriate formatting
                logger.info(f"Sending override response to client: {domain_override}")
                
                if use_new_format and has_api_formatter:
                    return jsonify(format_api_response(domain_override)), 200
                else:
                    return jsonify(domain_override), 200
            
            # Direct check if domain is worth crawling or is parked
            worth_crawling, has_dns, dns_error, potentially_flaky = is_domain_worth_crawling(domain)
            
            # Check for DNS resolution failure
            if not has_dns:
                logger.info(f"Domain {domain} has DNS resolution issues: {dns_error}")
                
                # Create an error result
                error_result = create_error_result(
                    domain,
                    "dns_error",
                    dns_error,
                    email,
                    "early_check"
                )
                
                error_result["website_url"] = url
                error_result["final_classification"] = "7-No Website available"
                
                # Format and return immediately - DO NOT CONTINUE WITH ENRICHMENT
                if use_new_format and has_api_formatter:
                    formatted_result = format_api_response(error_result)
                    return jsonify(formatted_result), 200
                else:
                    return jsonify(error_result), 200
            
            # Check for parked domain
            if dns_error == "parked_domain" or not worth_crawling:
                logger.info(f"Domain {domain} detected as parked domain during initial check")
                
                # Create a proper parked domain result
                parked_result = create_parked_domain_result(domain, crawler_type="early_check_parked")
                
                # Process it through the normal result processor
                result = process_fresh_result(parked_result, domain, email, url)
                
                # Ensure proper classification and fields
                result["final_classification"] = "6-Parked Domain - no enrichment"
                result["crawler_type"] = "early_check_parked"
                result["classifier_type"] = "early_detection"
                result["is_parked"] = True
                
                # Add email and URL if provided
                if email:
                    result["email"] = email
                
                result["website_url"] = url
                
                # Try to enrich with Apollo data for parked domains
                try:
                    # Import Apollo connector
                    from domain_classifier.enrichment.apollo_connector import ApolloConnector
                    
                    # Initialize Apollo connector
                    apollo = ApolloConnector()
                    
                    # Get Apollo data
                    apollo_data = apollo.enrich_company(domain)
                    
                    if apollo_data and isinstance(apollo_data, dict) and any(apollo_data.values()):
                        # Store Apollo data in result
                        result["apollo_data"] = apollo_data
                        
                        # Try to classify based on Apollo data
                        if "short_description" in apollo_data or "description" in apollo_data:
                            # Get description from Apollo
                            description = None
                            for field in ["short_description", "description", "long_description"]:
                                if field in apollo_data:
                                    description = apollo_data.get(field)
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
                                        
                                        # Run classification on Apollo description
                                        classification = classifier.classify(
                                            content=f"Company description from Apollo: {description}",
                                            domain=domain
                                        )
                                        
                                        if classification:
                                            # Update result with classification
                                            result.update(classification)
                                            result["detection_method"] = "apollo_data_classification"
                                            result["source"] = "apollo_data"
                                            
                                            # Update final_classification based on the new predicted_class
                                            result["final_classification"] = determine_final_classification(result)
                                except Exception as e:
                                    logger.error(f"Error classifying parked domain with Apollo data: {e}")
                        
                        # Update final classification if we have Apollo data but didn't reclassify
                        if result.get("predicted_class") == "Parked Domain":
                            result["final_classification"] = "5-Parked Domain with partial enrichment"
                except Exception as e:
                    logger.error(f"Error enriching parked domain with Apollo data: {e}")
                
                # Format and return
                if use_new_format and has_api_formatter:
                    return jsonify(format_api_response(result)), 200
                else:
                    return jsonify(result), 200
            
            # Check for early parked domain detection using direct crawl
            try:
                from domain_classifier.crawlers.direct_crawler import direct_crawl
                
                logger.info(f"Performing quick check for parked domain before enriching: {domain}")
                
                quick_check_content, (error_type, error_detail), quick_crawler_type = direct_crawl(url, timeout=5.0)
                
                # Check if this is a parked domain
                if error_type == "is_parked" or (quick_check_content and is_parked_domain(quick_check_content, domain)):
                    logger.info(f"Quick check detected parked domain: {domain}")
                    
                    parked_result = create_parked_domain_result(domain, crawler_type="quick_check_parked")
                    
                    # Process the result through the normal result processor
                    result = process_fresh_result(parked_result, domain, email, url)
                    
                    # Ensure proper classification and fields
                    result["final_classification"] = "6-Parked Domain - no enrichment"
                    result["crawler_type"] = "quick_check_parked"
                    result["classifier_type"] = "early_detection"
                    result["is_parked"] = True
                    
                    # Add email and URL if provided
                    if email:
                        result["email"] = email
                    
                    result["website_url"] = url
                    
                    # Try to enrich with Apollo data for parked domains
                    try:
                        # Import Apollo connector
                        from domain_classifier.enrichment.apollo_connector import ApolloConnector
                        
                        # Initialize Apollo connector
                        apollo = ApolloConnector()
                        
                        # Get Apollo data
                        apollo_data = apollo.enrich_company(domain)
                        
                        if apollo_data and isinstance(apollo_data, dict) and any(apollo_data.values()):
                            # Store Apollo data in result
                            result["apollo_data"] = apollo_data
                            
                            # Try to classify based on Apollo data
                            if "short_description" in apollo_data or "description" in apollo_data:
                                # Get description from Apollo
                                description = None
                                for field in ["short_description", "description", "long_description"]:
                                    if field in apollo_data:
                                        description = apollo_data.get(field)
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
                                            
                                            # Run classification on Apollo description
                                            classification = classifier.classify(
                                                content=f"Company description from Apollo: {description}",
                                                domain=domain
                                            )
                                            
                                            if classification:
                                                # Update result with classification
                                                result.update(classification)
                                                result["detection_method"] = "apollo_data_classification"
                                                result["source"] = "apollo_data"
                                                
                                                # Update final_classification based on the new predicted_class
                                                result["final_classification"] = determine_final_classification(result)
                                    except Exception as e:
                                        logger.error(f"Error classifying parked domain with Apollo data: {e}")
                            
                            # Update final classification if we have Apollo data but didn't reclassify
                            if result.get("predicted_class") == "Parked Domain":
                                result["final_classification"] = "5-Parked Domain with partial enrichment"
                    except Exception as e:
                        logger.error(f"Error enriching parked domain with Apollo data: {e}")
                    
                    # Format and return
                    if use_new_format and has_api_formatter:
                        return jsonify(format_api_response(result)), 200
                    else:
                        return jsonify(result), 200
            except Exception as e:
                logger.warning(f"Early parked domain check failed in enrichment route: {e}")
            
            # First perform standard classification by making an internal request
            # We'll use the routes directly from the app, rather than importing functions
            
            # Fix: Create a properly structured classification_result to start with
            classification_result = {
                "domain": domain,
                "website_url": url,
                "predicted_class": "Internal IT Department",  # Default, will be updated
                "confidence_score": 50,
                "confidence_scores": {
                    "Managed Service Provider": 10,
                    "Integrator - Commercial A/V": 5,
                    "Integrator - Residential A/V": 5,
                    "Internal IT Department": 50
                },
                "detection_method": "initial_default",
                "classifier_type": "pending_classification",
                "crawler_type": "pending",
                "source": "initial"
            }
            
            # If email was provided, add it
            if email:
                classification_result["email"] = email
            
            # Call the registered classify-domain route directly using the Flask test client
            with app.test_client() as client:
                # Create the request payload
                # CRITICAL CHANGE: Always force reclassify and add force_llm to ensure LLM classification
                request_data = {
                    'url': input_value,
                    'force_reclassify': True,  # Always force reclassify
                    'force_llm': True  # Ensure LLM is used for classification
                }
                
                logger.info(f"Sending internal classification request for {domain} with force_reclassify=True and force_llm=True")
                
                response = client.post('/classify-domain', json=request_data)
                
                # IMPROVED: Better handling of the response
                status_code = response.status_code
                logger.info(f"Classification result status code: {status_code}")
                
                if status_code == 200:
                    try:
                        # Get JSON response and log its structure
                        response_data = response.get_json()
                        
                        if response_data:
                            logger.info(f"Classification response keys: {list(response_data.keys())}")
                            
                            # FIXED: Handle both original and formatted field names
                            # Map formatted field names to original field names
                            field_mapping = {
                                "predicted_class": ["02_classification", "classification"],
                                "confidence_score": ["02_confidence_score", "confidence_score"],
                                "confidence_scores": ["02_confidence_scores", "confidence_scores"],
                                "detection_method": ["02_detection_method", "detection_method"],
                                "explanation": ["02_explanation", "explanation"],
                                "classifier_type": ["02_classifier_type", "classifier_type"],
                                "crawler_type": ["02_crawler_type", "crawler_type"]
                            }
                            
                            # Process each field with its possible formatted alternatives
                            for original_field, formatted_fields in field_mapping.items():
                                # First try the original field name
                                if original_field in response_data:
                                    classification_result[original_field] = response_data[original_field]
                                else:
                                    # Try each of the formatted field names
                                    for formatted_field in formatted_fields:
                                        if formatted_field in response_data:
                                            classification_result[original_field] = response_data[formatted_field]
                                            break
                            
                            # Log important fields
                            if "predicted_class" in classification_result:
                                logger.info(f"Classification result predicted_class: {classification_result['predicted_class']}")
                            
                            if "classifier_type" in classification_result:
                                logger.info(f"Classifier type used: {classification_result['classifier_type']}")
                            else:
                                # Don't log a warning if classifier_type isn't found at this stage
                                # It will be set later in the process
                                logger.info(f"Note: classifier_type not found in initial result for {domain} - will be set later")
                            
                            if "detection_method" in classification_result:
                                logger.info(f"Detection method: {classification_result['detection_method']}")
                            
                            # Verify we have required fields
                            if "predicted_class" not in classification_result or not classification_result.get("predicted_class"):
                                logger.warning("predicted_class not found in response data")
                            
                            # CRITICAL VERIFICATION: Check if LLM was used for classification
                            if "classifier_type" in classification_result:
                                if "claude-llm" in classification_result['classifier_type']:
                                    logger.info(f"✅ LLM classification successful for {domain}")
                                else:
                                    logger.warning(f"⚠️ LLM classifier was not used for {domain}. Using {classification_result.get('classifier_type', 'unknown')} instead.")
                        else:
                            logger.error("Response JSON is None or empty")
                    
                    except Exception as parse_error:
                        logger.error(f"Error parsing response JSON: {parse_error}")
                        logger.error(f"Response content: {response.data}")
                
                # If predicted_class is empty, that's a critical error
                if not classification_result.get("predicted_class"):
                    logger.error(f"❌ CRITICAL ERROR: predicted_class missing or empty for {domain}")
                    
                    # Try to fix the response by adding a default classification
                    classification_result["predicted_class"] = "Internal IT Department"
                    classification_result["confidence_score"] = 30
                    classification_result["confidence_scores"] = {
                        "Managed Service Provider": 5,
                        "Integrator - Commercial A/V": 3,
                        "Integrator - Residential A/V": 2,
                        "Internal IT Department": 30
                    }
                    classification_result["detection_method"] = "emergency_fallback"
                    classification_result["explanation"] = f"Unable to determine classification for {domain} due to technical issues. Defaulting to Internal IT."
                
                # Log complete classification result
                logger.info(f"Classification result after test client: {classification_result.get('predicted_class', 'unknown')}")
                
                # We'll proceed with enrichment regardless of classification status
                # But we'll log a warning if the status code indicates an error
                if status_code >= 400:
                    logger.warning(f"Classification returned status {status_code}, but continuing with enrichment anyway")
            
            # Make sure we have the minimum required fields for enrichment
            if "domain" not in classification_result:
                # We should already have the domain from earlier
                classification_result["domain"] = domain
            
            # Ensure final_classification is set for error results
            if "final_classification" not in classification_result:
                classification_result["final_classification"] = determine_final_classification(classification_result)
            
            # Add website URL for clickable link if not present
            if "website_url" not in classification_result:
                classification_result["website_url"] = url
            
            # Add email to response if input was an email and not already present
            if email and "email" not in classification_result:
                classification_result["email"] = email
            
            # Extract domain, email from classification result
            domain = classification_result.get('domain', domain)  # Use the extracted domain if not in result
            email = classification_result.get('email', email)  # Use the extracted email if not in result
            crawler_type = classification_result.get('crawler_type', "not_available")  # Get crawler type from classification result
            
            if not domain:
                logger.error("No domain found in classification result and couldn't extract from input")
                return jsonify({"error": "Failed to extract domain for enrichment"}), 400
            
            # Initialize variables for Apollo data
            apollo_company_data = None
            
            # Check if there's already Apollo data in the cached result
            if "apollo_data" in classification_result and classification_result["apollo_data"]:
                logger.info(f"Using cached Apollo data for {domain}")
                apollo_company_data = classification_result["apollo_data"]
            else:
                # Import Apollo connector here to avoid circular imports
                from domain_classifier.enrichment.apollo_connector import ApolloConnector
                
                # Initialize Apollo connector
                apollo = ApolloConnector()
                
                # Enrich with Apollo company data only if we don't already have it
                apollo_company_data = apollo.enrich_company(domain)
                logger.info(f"Retrieved fresh Apollo data for {domain}")
            
            from domain_classifier.enrichment.description_enhancer import enhance_company_description, generate_detailed_description
            
            # Don't look up person data to save Apollo credits
            person_data = None
            
            # Get the website content for AI extraction - do this ONCE and cache it
            if domain_content_cache is None:
                domain_content_cache = snowflake_conn.get_domain_content(domain)
                logger.info(f"Retrieved and cached content for {domain} (length: {len(domain_content_cache) if domain_content_cache else 0})")
            
            # Always attempt AI extraction, regardless of Apollo data
            logger.info(f"Attempting AI extraction for {domain}")
            
            # Extract company data using AI from the website content
            ai_company_data = None
            if domain_content_cache:
                ai_company_data = extract_company_data_from_content(
                    domain_content_cache,
                    domain,
                    classification_result
                )
                
                # Add the AI-extracted data to the result
                if ai_company_data:
                    logger.info(f"Successfully extracted AI company data for {domain}")
                    classification_result["ai_company_data"] = ai_company_data
            else:
                logger.warning(f"No website content available for AI extraction for {domain}")
            
            # Run domain word analysis
            domain_word_scores = analyze_domain_words(domain)
            
            # Check industry context
            is_service, industry_confidence = check_industry_context(
                domain_content_cache,
                apollo_company_data,
                ai_company_data
            )
            
            # Log results of enhanced analysis
            logger.info(f"Domain word analysis for {domain}: {domain_word_scores}")
            logger.info(f"Industry context for {domain}: is_service={is_service}, confidence={industry_confidence}")
            
            # Import recommendation engine
            from domain_classifier.enrichment.recommendation_engine import DomotzRecommendationEngine
            
            # Create an instance of the recommendation engine
            recommendation_engine = DomotzRecommendationEngine()
            
            # Store original class for comparison after cross-validation
            original_class = classification_result.get('predicted_class', '')
            original_description = classification_result.get('company_description', '')
            
            # CRITICAL CHANGE: Skip cross-validation completely
            # This prevents the original LLM classification from being overridden
            # classification_result = reconcile_classification(classification_result, apollo_company_data, ai_company_data)
            
            # Generate recommendations based on current classification
            recommendations = recommendation_engine.generate_recommendations(
                classification_result.get('predicted_class'),
                apollo_company_data
            )
            
            # Step 1: First enhance with Apollo data
            if apollo_company_data:
                logger.info(f"Enhancing description with Apollo data for {domain}")
                
                basic_enhanced_description = enhance_company_description(
                    classification_result.get("company_description", ""),
                    apollo_company_data,
                    classification_result
                )
                
                classification_result["company_description"] = basic_enhanced_description
            
            # Step 2: Then use Claude to generate a more detailed description
            try:
                detailed_description = generate_detailed_description(
                    classification_result,
                    apollo_company_data,
                    None  # No person data passed
                )
                
                if detailed_description and len(detailed_description) > 50:
                    # Check for potential industry mismatches
                    industry_mismatch = False
                    
                    # Check if the company is claimed to be in the maritime industry
                    if 'maritime' in detailed_description.lower() and 'audio' in detailed_description.lower() and 'visual' in detailed_description.lower():
                        logger.warning(f"Detected potential fabrication: maritime company with AV services for {domain}")
                        industry_mismatch = True
                    
                    # Check for other suspicious fabrications - A/V services for non-A/V companies
                    if classification_result.get('predicted_class') == "Internal IT Department" and ('audio-visual' in detailed_description.lower() or 'av integration' in detailed_description.lower()):
                        logger.warning(f"Detected potential fabrication: Internal IT with AV services for {domain}")
                        industry_mismatch = True
                    
                    if not industry_mismatch:
                        classification_result["company_description"] = detailed_description
                        logger.info(f"Updated description with detailed Claude-generated version for {domain}")
                else:
                    logger.warning(f"Generated description was too short or empty for {domain}")
            except Exception as desc_error:
                logger.error(f"Error generating detailed description: {desc_error}")
                # Keep the basic enhanced description if the detailed one fails
            
            # Add enrichment data to classification result
            classification_result['apollo_data'] = apollo_company_data or {}
            
            # Add recommendations
            classification_result['domotz_recommendations'] = recommendations
            
            # Make sure the crawler_type is preserved from the original classification
            if not classification_result.get('crawler_type') and crawler_type:
                classification_result['crawler_type'] = crawler_type
            
            # UPDATED: Use the ensure_classifier_type helper to avoid warnings
            classification_result = ensure_classifier_type(classification_result, domain)
            
            # CRITICAL FIX: Truncate detection_method if it's too long for Snowflake
            if "detection_method" in classification_result and len(classification_result["detection_method"]) > 40:
                classification_result["detection_method"] = classification_result["detection_method"][:40]
                logger.warning(f"Truncated detection_method to 40 chars for Snowflake compatibility")
            
            # Prioritize Apollo data for company name and other fields
            if apollo_company_data and ai_company_data:
                # Validate and prioritize Apollo data for core fields
                if "name" in ai_company_data:
                    # Log warning if there's a suspicious navigation element
                    if ai_company_data["name"] and re.search(r'navigation|menu|open|close|header|footer',
                                                          ai_company_data.get('name', ''), re.IGNORECASE):
                        logger.warning(f"Suspicious AI-extracted company name detected: {ai_company_data.get('name')}. Using Apollo data instead.")
                        ai_company_data["name"] = apollo_company_data.get("name", ai_company_data["name"])
                
                # Always prefer Apollo's key data where available
                for key in ['name', 'industry', 'employee_count', 'founded_year']:
                    if key in ai_company_data and apollo_company_data.get(key) is not None and apollo_company_data.get(key) != "":
                        logger.info(f"Prioritizing Apollo data for field '{key}' over AI extraction")
                        ai_company_data[key] = apollo_company_data.get(key)
            
            # Add explicit company_name field to the response
            if apollo_company_data and apollo_company_data.get("name"):
                classification_result['company_name'] = apollo_company_data.get("name")
            elif ai_company_data and ai_company_data.get("name") and not re.search(r'navigation|menu|open|close|header|footer',
                                                                               ai_company_data["name"], re.IGNORECASE):
                classification_result['company_name'] = ai_company_data.get("name")
            else:
                # Fallback to domain-derived name
                classification_result['company_name'] = domain.split('.')[0].capitalize()
            
            # Update final_classification based on predicted class
            if "final_classification" not in classification_result:
                # Ensure final_classification is set
                classification_result["final_classification"] = determine_final_classification(classification_result)
                logger.info(f"Added final classification: {classification_result['final_classification']} for {domain}")
            
            # Save the enhanced data to Snowflake (with Apollo data) - use cached content
            save_to_snowflake(
                domain=domain,
                url=url,
                content=domain_content_cache,  # Use cached content
                classification=classification_result,
                snowflake_conn=snowflake_conn,
                apollo_company_data=apollo_company_data,
                crawler_type=crawler_type,  # Explicitly pass the crawler_type from the original classification
                classifier_type="claude-llm-enriched"  # CRITICAL: Always mark as LLM enriched
            )
            
            # Return the enriched result
            logger.info(f"Successfully enriched and generated recommendations for {domain}")
            
            # Format the response if requested
            if use_new_format and has_api_formatter:
                return jsonify(format_api_response(classification_result)), 200
            else:
                return jsonify(classification_result), 200
        
        except Exception as e:
            logger.error(f"Error in classify-and-enrich: {e}\n{traceback.format_exc()}")
            
            # Try to identify the error type
            error_type, error_detail = detect_error_type(str(e))
            
            # Create an error response
            error_result = create_error_result(
                domain if 'domain' in locals() else "unknown",
                error_type,
                error_detail,
                email if 'email' in locals() else None,
                "enrich_error_handler"  # Set a crawler_type for enrichment errors
            )
            
            error_result["error"] = str(e)  # Add the actual error message
            
            # Ensure final_classification is set for error results
            if "final_classification" not in error_result:
                error_result["final_classification"] = determine_final_classification(error_result)
            
            # Format the response if requested
            use_new_format = data.get('use_new_format', True) if 'data' in locals() else True
            if use_new_format and has_api_formatter:
                return jsonify(format_api_response(error_result)), 200  # Return 200 instead of 500
            else:
                return jsonify(error_result), 200  # Return 200 instead of 500

    def _is_minimal_apollo_data(apollo_data):
        """Check if Apollo data is minimal and needs enhancement."""
        # Define the essential fields we want to check
        essential_fields = ["name", "address", "industry", "employee_count", "phone"]
        
        # Count how many essential fields are missing
        missing_fields = sum(1 for field in essential_fields if not apollo_data.get(field))
        
        # If most essential fields are missing, consider it minimal
        return missing_fields >= 3

    # CRITICAL FIX: Return the app object
    logger.info("Enrich routes registered successfully")
    return app
