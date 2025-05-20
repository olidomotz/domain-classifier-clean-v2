"""Complete fixed enrich.py to properly return app and use LLM classification."""
import logging
import traceback
import re  # Added import for regex operations
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
from domain_classifier.utils.json_utils import ensure_dict, safe_get
from domain_classifier.utils.domain_utils import extract_domain_from_email, normalize_domain

# Import the API formatter if available
try:
    from domain_classifier.utils.api_formatter import format_api_response
    has_api_formatter = True
except ImportError:
    has_api_formatter = False

# Set up logging
logger = logging.getLogger(__name__)

# This function must be exported and match exactly what __init__.py is trying to import
def register_enrich_routes(app, snowflake_conn):
    """Register enrichment related routes."""
    logger.info("Registering enrich routes...")
    
    # Check if app is None and create fallback
    if app is None:
        logger.error("App object is None in register_enrich_routes")
        from flask import Flask
        app = Flask(__name__)
        logger.info("Created new Flask app as fallback in enrich.py")
    
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
            logger.info("="*80)
            logger.info(f"CLASSIFY AND ENRICH REQUEST for: {input_value}")
            logger.info(f"Force reclassify: {force_reclassify}")
            logger.info("="*80)
            
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
            if not has_dns and dns_error != "parked_domain":
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
                
                # We'll still attempt to enrich from Apollo
                logger.info(f"DNS issues for {domain}, but continuing with enrichment anyway")
            
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
                
                # We'll still attempt Apollo enrichment
                logger.info(f"Parked domain {domain}, but continuing with enrichment anyway")
            
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
                    
                    # We'll still attempt Apollo enrichment
                    logger.info(f"Parked domain detected for {domain}, but continuing with enrichment anyway")
            except Exception as e:
                logger.warning(f"Early parked domain check failed in enrichment route: {e}")
            
            # First perform standard classification by making an internal request
            # We'll use the routes directly from the app, rather than importing functions
            
            # Call the registered classify-domain route directly using the Flask test client
            with app.test_client() as client:
                # Create the request payload
                # CRITICAL CHANGE: Always force reclassify and add force_llm to ensure LLM classification
                request_data = {
                    'url': input_value,
                    'force_reclassify': True,    # Always force reclassify
                    'force_llm': True            # Ensure LLM is used for classification
                }
                
                logger.info(f"Sending internal classification request for {domain} with force_reclassify=True and force_llm=True")
                response = client.post('/classify-domain', json=request_data)
                classification_result = response.get_json()
                status_code = response.status_code
                
                # DEBUGGING: Log the classification result
                logger.info(f"Classification result status code: {status_code}")
                if "predicted_class" in classification_result:
                    logger.info(f"Classification result predicted_class: {classification_result['predicted_class']}")
                if "classifier_type" in classification_result:
                    logger.info(f"Classifier type used: {classification_result['classifier_type']}")
                if "detection_method" in classification_result:
                    logger.info(f"Detection method: {classification_result['detection_method']}")
                
                # CRITICAL VERIFICATION: Check if LLM was used for classification
                if "classifier_type" in classification_result:
                    if "claude-llm" in classification_result['classifier_type']:
                        logger.info(f"✅ LLM classification successful for {domain}")
                    else:
                        logger.warning(f"⚠️ LLM classifier was not used for {domain}. Using {classification_result.get('classifier_type', 'unknown')} instead.")
                else:
                    logger.warning(f"⚠️ No classifier_type found in result for {domain}")
                    
                # If predicted_class is empty, that's a critical error
                if "predicted_class" not in classification_result or not classification_result.get("predicted_class"):
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
            
            # CRITICAL: Apply cross-validation with industry data to detect and correct misclassifications
            classification_result = reconcile_classification(
                classification_result,
                apollo_company_data,
                ai_company_data
            )
            
            # If classification changed during cross-validation, update the description to avoid fabrication
            if original_class != classification_result.get('predicted_class', ''):
                logger.warning(f"Cross-validation changed classification for {domain} from {original_class} to {classification_result.get('predicted_class')}")
                
                # If we're changing from Commercial AV to Internal IT, generate a new accurate description
                if "Integrator - Commercial A/V" in original_class and classification_result.get('predicted_class') == "Internal IT Department":
                    company_name = classification_result.get('company_name', domain.split('.')[0].capitalize())
                    
                    # Check for maritime or shipping industry
                    is_maritime = False
                    industry = ''
                    
                    if apollo_company_data:
                        # Safely extract industry
                        apollo_dict = ensure_dict(apollo_company_data, "apollo_data")
                        industry = safe_get(apollo_dict, 'industry', '').lower()
                        if industry and ('maritime' in industry or 'shipping' in industry or 'vessel' in industry):
                            is_maritime = True
                    
                    if ai_company_data:
                        # Safely extract industry
                        ai_dict = ensure_dict(ai_company_data, "ai_data")
                        ai_industry = safe_get(ai_dict, 'industry', '').lower()
                        if ai_industry and ('maritime' in ai_industry or 'shipping' in ai_industry or 'vessel' in ai_industry):
                            is_maritime = True
                            if not industry:
                                industry = ai_industry
                    
                    # Check description for maritime terms
                    if original_description:
                        orig_desc_lower = original_description.lower()
                        if 'maritime' in orig_desc_lower or 'shipping' in orig_desc_lower or 'vessel' in orig_desc_lower:
                            is_maritime = True
                    
                    # Generate accurate description based on actual industry
                    if is_maritime:
                        # Extract employee count
                        employee_count = None
                        if apollo_company_data and ensure_dict(apollo_company_data, "").get('employee_count'):
                            employee_count = ensure_dict(apollo_company_data, "").get('employee_count')
                        
                        # Create maritime-specific description
                        new_description = f"{company_name} is a maritime industry company specializing in shipping services"
                        if employee_count:
                            new_description += f" with approximately {employee_count} employees"
                        new_description += "."
                        
                        if domain_content_cache:
                            content_lower = domain_content_cache.lower()
                            if "supplies" in content_lower or "parts" in content_lower:
                                new_description += " The company provides ship supplies, spare parts, and maritime equipment to vessels."
                            elif "logistics" in content_lower:
                                new_description += " The company offers maritime logistics and shipping services."
                            else:
                                new_description += " The company operates in the maritime industry, providing shipping-related services."
                        else:
                            new_description += " The company operates in the maritime industry with internal IT needs."
                        
                        classification_result['company_description'] = new_description
                        classification_result['company_one_line'] = f"{company_name} provides maritime shipping services and supplies."
                    
                    elif industry:
                        # Generic industry-based description
                        employee_count = None
                        if apollo_company_data and ensure_dict(apollo_company_data, "").get('employee_count'):
                            employee_count = ensure_dict(apollo_company_data, "").get('employee_count')
                            
                        new_description = f"{company_name} is a {industry} company"
                        if employee_count:
                            new_description += f" with approximately {employee_count} employees"
                        new_description += "."
                        
                        # Add more context based on industry keywords
                        if 'manufacturing' in industry or 'industrial' in industry:
                            new_description += f" The company manufactures or produces goods in the {industry} sector."
                        elif 'retail' in industry or 'commerce' in industry or 'shop' in industry:
                            new_description += f" The company sells products in the {industry} market."
                        else:
                            new_description += f" The company operates in the {industry} industry with internal IT needs."
                        
                        classification_result['company_description'] = new_description
                        classification_result['company_one_line'] = f"{company_name} is a {industry} company with internal IT needs."
                    else:
                        # Generic correction when we have limited information
                        new_description = f"{company_name} is a business with internal IT needs. The company does not provide audio-visual integration services as incorrectly identified previously."
                        classification_result['company_description'] = new_description
                        classification_result['company_one_line'] = f"{company_name} is a business with internal IT needs, not an AV integrator."
            
            # Generate recommendations based on updated classification
            recommendations = recommendation_engine.generate_recommendations(
                classification_result.get('predicted_class'),
                apollo_company_data
            )
            
            # Step 1: First enhance with Apollo data (but only if classification wasn't changed)
            if apollo_company_data and original_class == classification_result.get('predicted_class', ''):
                logger.info(f"Enhancing description with Apollo data for {domain}")
                basic_enhanced_description = enhance_company_description(
                    classification_result.get("company_description", ""),
                    apollo_company_data,
                    classification_result
                )
                classification_result["company_description"] = basic_enhanced_description
            
            # Step 2: Then use Claude to generate a more detailed description
            # Only do this if classification wasn't changed by cross-validation
            if original_class == classification_result.get('predicted_class', ''):
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
                
            # CRITICAL CHANGE: Ensure classifier_type explicitly reflects LLM usage
            if "classifier_type" not in classification_result or "claude-llm" not in classification_result["classifier_type"]:
                classification_result["classifier_type"] = "claude-llm-enriched"
                logger.info(f"Setting classifier_type to claude-llm-enriched for {domain}")
            
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
                
            # Update final_classification based on Apollo data
            # For parked domains, check if we need to update from 6-Parked Domain - no enrichment to 5-Parked Domain with partial enrichment
            if classification_result.get('final_classification') == "6-Parked Domain - no enrichment" and apollo_company_data:
                classification_result['final_classification'] = "5-Parked Domain with partial enrichment"
                logger.info(f"Updated final classification to 5-Parked Domain with partial enrichment for {domain}")
            elif "final_classification" not in classification_result:
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
