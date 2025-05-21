"""
Domain Classifier Fixes

This script applies fixes to various issues in the domain classifier system, including:
1. Missing company descriptions
2. Inconsistent confidence scores
3. Adding Apollo data presence indicator
4. Adding content quality indicator
5. Fixing crawler_type "pending" issue
"""

import logging
import json
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def apply_patches():
    """Monkey-patch the necessary functions to apply fixes."""
    logger.info("Applying domain classifier fixes...")
    
    # Import the modules we need to patch
    from domain_classifier.api.routes import enrich
    from domain_classifier.storage import result_processor
    from domain_classifier.utils import api_formatter
    from domain_classifier.crawlers import apify_crawler
    from domain_classifier.utils import cross_validator
    
    # 1. Fix missing company description in result_processor
    original_process_fresh = result_processor.process_fresh_result
    
    def patched_process_fresh(classification, domain, email=None, url=None):
        # Ensure company_description is present and meaningful
        if not classification.get('company_description') or len(classification.get('company_description', '')) < 10:
            # Generate a fallback description
            company_type = classification.get('predicted_class', 'Unknown')
            if company_type == "Managed Service Provider":
                fallback_description = f"{domain} is a Managed Service Provider offering IT services and solutions for businesses."
            elif company_type == "Integrator - Commercial A/V":
                fallback_description = f"{domain} is a Commercial A/V Integrator providing audiovisual systems for businesses."
            elif company_type == "Integrator - Residential A/V":
                fallback_description = f"{domain} is a Residential A/V Integrator specializing in home automation and entertainment systems."
            else:  # Internal IT or Unknown
                fallback_description = f"{domain} is a business with internal IT needs."
            
            # Add the fallback description
            classification['company_description'] = fallback_description
            logger.info(f"Added fallback company description for {domain}: {fallback_description}")
        
        # Call original function
        result = original_process_fresh(classification, domain, email, url)
        
        # Add Apollo data indicator
        apollo_data = classification.get("apollo_data", {})
        if isinstance(apollo_data, str):
            try:
                apollo_data = json.loads(apollo_data)
            except:
                apollo_data = {}
        
        # Check if Apollo data has meaningful content
        has_apollo = False
        if apollo_data and isinstance(apollo_data, dict):
            # Check for key fields that indicate useful data
            key_fields = ["name", "industry", "employee_count", "founded_year"]
            has_meaningful_fields = any(apollo_data.get(field) for field in key_fields)
            has_apollo = has_meaningful_fields
        
        result["has_apollo_data"] = has_apollo
        
        # Set content quality if not present
        if "content_quality" not in result:
            result["content_quality"] = 0  # Default: did not connect
            
            # Try to determine content quality from other indicators
            if result.get("crawler_type") == "existing_content" or result.get("source") == "cached":
                # We have cached content
                result["content_quality"] = 2  # Assume substantial content for cached
            elif result.get("is_parked", False):
                result["content_quality"] = 1  # Minimal/parked
                
        return result
    
    # Patch the function
    result_processor.process_fresh_result = patched_process_fresh
    
    # 2. Fix api_formatter to handle all our new fields
    original_format_api = api_formatter.format_api_response
    
    def patched_format_api(result):
        # Fix for missing company name by trying to extract from different fields
        if "company_name" not in result and "apollo_data" in result and isinstance(result["apollo_data"], dict):
            result["company_name"] = result["apollo_data"].get("name")
        
        # Add content quality label
        content_quality = result.get("content_quality", 0)
        result["content_quality_label"] = "Did not connect"
        if content_quality == 1:
            result["content_quality_label"] = "Minimal content (possibly parked)"
        elif content_quality == 2:
            result["content_quality_label"] = "Substantial content"
        
        # Fix missing company description with a fallback
        if not result.get("company_description"):
            domain_name = result.get("domain", "This company")
            predicted_class = result.get("predicted_class", "business")
            result["company_description"] = f"{domain_name} is a {predicted_class} providing technology services."
            logger.info(f"Added emergency fallback description in formatter for {domain_name}")
        
        # Call the original formatter
        formatted = original_format_api(result)
        
        # Add the content quality and Apollo data flags to the domain info section
        formatted["01_content_quality"] = result.get("content_quality", 0)
        formatted["01_content_quality_label"] = result.get("content_quality_label", "Unknown")
        formatted["01_has_apollo_data"] = result.get("has_apollo_data", False)
        
        return formatted
    
    # Patch the function
    api_formatter.format_api_response = patched_format_api
    
    # 3. Fix crawler_type in the enrichment route
    original_enrich = enrich.classify_and_enrich
    
    def patched_enrich():
        # Call the original function
        result = original_enrich()
        
        # Check if the response contains JSON data
        if hasattr(result, 'get_json'):
            data = result.get_json()
            
            # Fix pending crawler_type if present
            if data and isinstance(data, dict) and data.get('crawler_type') == "pending":
                data['crawler_type'] = "direct_fallback"
                
                # Apply the fix for confidence scores here too
                if data.get('detection_method') == 'cross_validation_it_services':
                    if data.get('predicted_class') == 'Managed Service Provider':
                        data['confidence_scores'] = {
                            "Managed Service Provider": 90,
                            "Integrator - Commercial A/V": 5,
                            "Integrator - Residential A/V": 5,
                            "Internal IT Department": 0
                        }
                        data['confidence_score'] = 90
                        data['max_confidence'] = 0.9
                
                # Log and return the modified response
                logger.info(f"Fixed pending crawler_type for {data.get('domain', 'unknown')}")
                import flask
                return flask.jsonify(data)
        
        return result
    
    # Patch the function
    enrich.classify_and_enrich = patched_enrich
    
    # 4. Fix cross_validator to ensure confidence scores are updated
    original_reconcile = cross_validator.reconcile_classification
    
    def patched_reconcile(classification, apollo_data=None, ai_data=None):
        result = original_reconcile(classification, apollo_data, ai_data)
        
        # If classification was changed to MSP, update confidence scores
        if 'predicted_class' in classification and 'predicted_class' in result:
            if classification.get('predicted_class') != result.get('predicted_class') and result.get('predicted_class') == 'Managed Service Provider':
                # Update confidence scores
                result['confidence_scores'] = {
                    "Managed Service Provider": 90,
                    "Integrator - Commercial A/V": 5,
                    "Integrator - Residential A/V": 5,
                    "Internal IT Department": 0
                }
                result['confidence_score'] = 90
                result['max_confidence'] = 0.9
                # Set a more specific detection method
                result['detection_method'] = "cross_validation_it_services"
                logger.info(f"Updated confidence scores after reclassification to MSP")
        
        return result
    
    # Patch the function
    cross_validator.reconcile_classification = patched_reconcile
    
    # 5. Update crawl_website to include content quality
    original_crawl_website = apify_crawler.crawl_website
    
    def patched_crawl_website(url):
        content, error_info, crawler_type = original_crawl_website(url)
        
        # Determine content quality
        content_quality = 0  # Default: did not connect
        if content:
            word_count = len(content.split())
            if word_count < 100:
                content_quality = 1  # Minimal content/parked domain
            else:
                content_quality = 2  # Substantial content
        
        return content, error_info, crawler_type, content_quality
    
    # Monkeypatch the method with one that has the right return tuple shape
    def compatible_crawl_website(url):
        content, error_info, crawler_type, content_quality = patched_crawl_website(url)
        # We need to maintain backwards compatibility with code that expects a 3-tuple return
        setattr(apify_crawler, 'last_content_quality', content_quality)
        return content, error_info, crawler_type
    
    # Patch the function
    apify_crawler.crawl_website = compatible_crawl_website
    apify_crawler.last_content_quality = 0  # Initialize the attribute
    
    # Add a helper to get content quality
    def get_content_quality():
        return getattr(apify_crawler, 'last_content_quality', 0)
    
    apify_crawler.get_content_quality = get_content_quality
    
    logger.info("All domain classifier fixes applied successfully")
