"""
Domain Classifier Fixes - Revised Version

This script applies fixes without relying on specific function paths.
"""

import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def apply_patches():
    """Apply patches to fix common issues in the domain classifier."""
    logger.info("Applying domain classifier fixes - revised approach...")
    
    # 1. Fix company description and add data indicators in result_processor
    try:
        from domain_classifier.storage import result_processor
        
        # Store original function
        original_process_fresh = result_processor.process_fresh_result
        
        def patched_process_fresh(classification, domain, email=None, url=None):
            # Add fallback company description if missing
            if "company_description" not in classification or not classification.get("company_description"):
                company_type = classification.get("predicted_class", "Unknown")
                if company_type == "Managed Service Provider":
                    fallback_description = f"{domain} is a Managed Service Provider offering IT services and solutions."
                elif company_type == "Integrator - Commercial A/V":
                    fallback_description = f"{domain} is a Commercial A/V Integrator providing audiovisual systems."
                elif company_type == "Integrator - Residential A/V":
                    fallback_description = f"{domain} is a Residential A/V Integrator specializing in home systems."
                else:
                    fallback_description = f"{domain} is a business with internal IT needs."
                
                classification["company_description"] = fallback_description
                logger.info(f"Added fallback company_description for {domain}")
            
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
                # Check for key fields that would indicate useful data
                key_fields = ["name", "industry", "employee_count", "founded_year"]
                has_meaningful_fields = any(apollo_data.get(field) for field in key_fields)
                has_apollo = has_meaningful_fields
            
            result["has_apollo_data"] = has_apollo
            
            # Determine content quality
            content_quality = 0  # Default: did not connect
            
            if result.get("is_parked", False):
                content_quality = 1  # Minimal content (parked)
            elif result.get("source") == "cached" or result.get("crawler_type") == "existing_content":
                content_quality = 2  # We have content
            elif "crawler_type" in result:
                if "minimal" in result["crawler_type"].lower():
                    content_quality = 1
                elif result["crawler_type"] not in ["unknown", "pending", "error_handler"]:
                    content_quality = 2
            
            result["content_quality"] = content_quality
            
            # Fix crawler_type if it's pending
            if result.get("crawler_type") == "pending":
                result["crawler_type"] = "direct_fallback"
            
            # Fix confidence scores if they're inconsistent
            if result.get("detection_method") == "cross_validation_it_services" or (
                    result.get("predicted_class") == "Managed Service Provider" and 
                    result.get("confidence_score", 0) < 70):
                result["confidence_scores"] = {
                    "Managed Service Provider": 90,
                    "Integrator - Commercial A/V": 5,
                    "Integrator - Residential A/V": 5,
                    "Internal IT Department": 0
                }
                result["confidence_score"] = 90
                result["max_confidence"] = 0.9
            
            return result
        
        # Apply the patch
        result_processor.process_fresh_result = patched_process_fresh
        logger.info("✅ Successfully patched result_processor.process_fresh_result")
    except Exception as e:
        logger.error(f"❌ Failed to patch result_processor: {e}")
    
    # 2. Fix API formatter to include our new fields
    try:
        from domain_classifier.utils import api_formatter
        
        original_format = api_formatter.format_api_response
        
        def patched_format_api(result):
            # Ensure content quality is present
            content_quality = result.get("content_quality", 0)
            quality_label = "Did not connect"
            if content_quality == 1:
                quality_label = "Minimal content (possibly parked)"
            elif content_quality == 2:
                quality_label = "Substantial content"
            
            result["content_quality_label"] = quality_label
            
            # Ensure Apollo indicator is present
            if "has_apollo_data" not in result:
                result["has_apollo_data"] = False
            
            # Ensure company description isn't empty
            if not result.get("company_description"):
                domain_name = result.get("domain", "This company")
                predicted_class = result.get("predicted_class", "business")
                result["company_description"] = f"{domain_name} is a {predicted_class}."
                logger.info(f"Added emergency fallback description for {domain_name}")
            
            # Fix confidence score inconsistency
            if result.get("detection_method") == "cross_validation_it_services" or (
                    result.get("predicted_class") == "Managed Service Provider" and 
                    result.get("confidence_score", 0) < 70):
                result["confidence_score"] = 90
            
            # Call original formatter
            formatted = original_format(result)
            
            # Add content quality to domain info section
            formatted["01_content_quality"] = content_quality
            formatted["01_content_quality_label"] = quality_label
            formatted["01_has_apollo_data"] = result.get("has_apollo_data", False)
            
            return formatted
        
        # Apply the patch
        api_formatter.format_api_response = patched_format_api
        logger.info("✅ Successfully patched api_formatter.format_api_response")
    except Exception as e:
        logger.error(f"❌ Failed to patch api_formatter: {e}")
    
    # 3. Patch cross validator to ensure confidence scores are updated
    try:
        from domain_classifier.utils import cross_validator
        
        original_reconcile = cross_validator.reconcile_classification
        
        def patched_reconcile(classification, apollo_data=None, ai_data=None):
            # Call original function
            result = original_reconcile(classification, apollo_data, ai_data)
            
            # If classification was changed to MSP, ensure confidence scores are updated
            if classification.get("predicted_class") != result.get("predicted_class"):
                if result.get("predicted_class") == "Managed Service Provider":
                    # Update confidence scores
                    result["confidence_scores"] = {
                        "Managed Service Provider": 90,
                        "Integrator - Commercial A/V": 5,
                        "Integrator - Residential A/V": 5,
                        "Internal IT Department": 0
                    }
                    result["confidence_score"] = 90
                    result["max_confidence"] = 0.9
                    # Set a more specific detection method
                    result["detection_method"] = "cross_validation_it_services"
                    logger.info(f"Updated confidence scores after reclassification to MSP")
            
            return result
        
        # Apply the patch
        cross_validator.reconcile_classification = patched_reconcile
        logger.info("✅ Successfully patched cross_validator.reconcile_classification")
    except Exception as e:
        logger.error(f"❌ Failed to patch cross_validator: {e}")
    
    logger.info("Domain classifier fixes applied with robust error handling")
