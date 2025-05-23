"""
Simplified fixes for domain classifier.
Since most fixes have been directly implemented in the respective files,
this now contains only essential runtime patches and monitoring.
"""

import logging
import sys
import importlib

# Set up logging
logger = logging.getLogger(__name__)

def apply_patches():
    """Apply minimal runtime patches and verify critical components."""
    logger.info("Applying minimal runtime patches and verifying components...")
    
    # Step 1: Verify critical imports to ensure updated files are being used
    verify_critical_imports()
    
    # Step 2: Add any necessary runtime compatibility patches
    apply_compatibility_patches()
    
    # Step 3: Set up enhanced monitoring
    setup_enhanced_monitoring()
    
    logger.info("All patches applied successfully")

def verify_critical_imports():
    """Verify that all critical updated components are being imported correctly."""
    try:
        # Import and check key components
        from domain_classifier.classifiers import decision_tree
        logger.info(f"✅ decision_tree.py loaded, is_parked_domain present: {hasattr(decision_tree, 'is_parked_domain')}")
        
        from domain_classifier.utils import error_handling
        logger.info(f"✅ error_handling.py loaded, check_domain_dns present: {hasattr(error_handling, 'check_domain_dns')}")
        
        from domain_classifier.utils import final_classification
        logger.info(f"✅ final_classification.py loaded, determine_final_classification present: {hasattr(final_classification, 'determine_final_classification')}")
        
        from domain_classifier.enrichment import description_enhancer
        logger.info(f"✅ description_enhancer.py loaded, generate_detailed_description present: {hasattr(description_enhancer, 'generate_detailed_description')}")
        
        from domain_classifier.utils import api_formatter
        logger.info(f"✅ api_formatter.py loaded, format_api_response present: {hasattr(api_formatter, 'format_api_response')}")
        
        from domain_classifier.utils import domain_analysis
        logger.info(f"✅ domain_analysis.py loaded, analyze_domain_words present: {hasattr(domain_analysis, 'analyze_domain_words')}")
        
        from domain_classifier.utils import cross_validator
        logger.info(f"✅ cross_validator.py loaded, reconcile_classification present: {hasattr(cross_validator, 'reconcile_classification')}")
        
        from domain_classifier.crawlers import direct_crawler
        logger.info(f"✅ direct_crawler.py loaded, direct_crawl present: {hasattr(direct_crawler, 'direct_crawl')}")
        
        from domain_classifier.crawlers import apify_crawler
        logger.info(f"✅ apify_crawler.py loaded, crawl_website present: {hasattr(apify_crawler, 'crawl_website')}")
        
    except ImportError as e:
        logger.error(f"❌ Critical component failed to import: {e}")
        raise

def apply_compatibility_patches():
    """Apply any necessary compatibility patches for runtime."""
    try:
        # Patch 1: Ensure Apollo API key is correctly loaded from environment
        import os
        if not os.environ.get("APOLLO_API_KEY") and os.environ.get("APOLLO_KEY"):
            os.environ["APOLLO_API_KEY"] = os.environ["APOLLO_KEY"]
            logger.info("Fixed Apollo API key environment variable")
        
        # Patch 2: Ensure proper classification of Process Did Not Complete
        try:
            from domain_classifier.api.routes import classify
            if hasattr(classify, 'classify_domain'):
                # Store original function for safety
                original_classify = classify.classify_domain
                
                # Check if we need to patch classify_domain with enhanced process_did_not_complete handling
                def enhanced_classify_domain_wrapper():
                    """Ensure Process Did Not Complete gets proper final classification."""
                    result, status_code = original_classify()
                    
                    # Get the JSON from result
                    try:
                        json_result = result.get_json()
                        
                        # Check if it's a Process Did Not Complete
                        if json_result and json_result.get('predicted_class') == 'Process Did Not Complete':
                            json_result['final_classification'] = '8-Unknown/No Data'
                            logger.info(f"Runtime patch: Set Process Did Not Complete final_classification to 8-Unknown/No Data")
                            
                            # Create a new response with modified JSON
                            from flask import jsonify
                            result = jsonify(json_result)
                            
                        return result, status_code
                    except Exception:
                        # If any error, return original result
                        return result, status_code
                
                # Only patch if needed (this is a safety measure)
                if 'Process Did Not Complete' not in str(original_classify):
                    classify.classify_domain = enhanced_classify_domain_wrapper
                    logger.info("Applied runtime patch for Process Did Not Complete classification")
        except Exception as e:
            logger.warning(f"Non-critical: Could not apply Process Did Not Complete patch: {e}")
            
    except Exception as e:
        logger.error(f"❌ Error applying compatibility patches: {e}")

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
        
        # Attempt to patch classify endpoint to count classifications
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
                logger.info("Enhanced monitoring applied to classify_domain endpoint")
        
        except Exception as e:
            logger.warning(f"Non-critical: Could not apply monitoring patch: {e}")
            
    except Exception as e:
        logger.error(f"❌ Error setting up enhanced monitoring: {e}")
