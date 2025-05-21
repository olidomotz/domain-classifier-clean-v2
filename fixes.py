"""
Domain Classifier Fixes - Improved Version

This script applies fixes focused on company descriptions and Apollo data flags.
"""

import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def apply_patches():
    """Apply improved patches to fix company description and Apollo data flags."""
    logger.info("Applying domain classifier fixes - improved version...")
    
    # 1. Fix api_formatter.format_api_response for better descriptions and Apollo flag
    try:
        from domain_classifier.utils import api_formatter
        
        original_format = api_formatter.format_api_response
        
        def patched_format_api(result):
            # More aggressively ensure company description is present and substantial
            domain = result.get("domain", "Unknown")
            
            # Check if we have Apollo data (even if not checking for "has_apollo_data" field)
            has_apollo = False
            if "apollo_data" in result and result.get("apollo_data"):
                apollo_data = result["apollo_data"]
                if isinstance(apollo_data, str):
                    try:
                        apollo_data = json.loads(apollo_data)
                    except:
                        apollo_data = {}
                
                # Check for meaningful Apollo data
                if apollo_data and any(apollo_data.get(field) for field in 
                                      ["name", "employee_count", "industry", "phone"]):
                    has_apollo = True
                    logger.info(f"Apollo data detected for {domain} in formatter")
            
            # Always set the flag whether true or false
            result["has_apollo_data"] = has_apollo
            
            # Create a robust description
            if not result.get("company_description") or len(result.get("company_description", "")) < 30:
                company_class = result.get("predicted_class", "business")
                company_name = result.get("company_name", domain.split('.')[0].capitalize())
                
                if has_apollo and isinstance(result.get("apollo_data", {}), dict):
                    apollo_data = result.get("apollo_data", {})
                    industry = apollo_data.get("industry", "")
                    founded_year = apollo_data.get("founded_year", "")
                    employees = apollo_data.get("employee_count", "")
                    
                    # Create comprehensive description with Apollo data
                    description = f"{company_name} is a {company_class}"
                    if industry:
                        description += f" in the {industry} industry"
                    if founded_year:
                        description += f", founded in {founded_year}"
                    if employees:
                        description += f" with approximately {employees} employees"
                    description += "."
                    
                    if company_class == "Managed Service Provider":
                        description += " The company provides IT services, support, and solutions for businesses."
                    elif "Integrator - Commercial A/V" in company_class:
                        description += " The company provides audio-visual solutions for commercial clients."
                    elif "Integrator - Residential A/V" in company_class:
                        description += " The company specializes in home automation and entertainment systems."
                    
                    result["company_description"] = description
                else:
                    # Fallback description
                    if company_class == "Managed Service Provider":
                        result["company_description"] = f"{company_name} is a Managed Service Provider offering IT services, infrastructure support, and technology solutions to businesses."
                    elif "Integrator - Commercial A/V" in company_class:
                        result["company_description"] = f"{company_name} is a Commercial A/V Integrator providing audio-visual systems for corporate environments, including conference rooms and presentation solutions."
                    elif "Integrator - Residential A/V" in company_class:
                        result["company_description"] = f"{company_name} is a Residential A/V Integrator specializing in home automation, entertainment systems, and smart home technology for residential clients."
                    else:
                        result["company_description"] = f"{company_name} is a business with internal IT needs rather than a provider of technology services to external clients."
                
                logger.info(f"Added detailed company description for {domain}: {result['company_description'][:50]}...")
            
            # Add content quality (default to 0 if not present)
            content_quality = result.get("content_quality", 0)
            if "crawler_type" in result and result["crawler_type"] != "pending":
                # Set content quality based on crawler_type
                if result["crawler_type"] in ["existing_content", "direct_https", "direct_http", "scrapy"]:
                    content_quality = 2  # Substantial content
                elif "minimal" in result["crawler_type"]:
                    content_quality = 1  # Minimal content
            
            # Set the content quality label
            quality_label = "Did not connect"
            if content_quality == 1:
                quality_label = "Minimal content (possibly parked)"
            elif content_quality == 2:
                quality_label = "Substantial content"
            
            result["content_quality"] = content_quality
            result["content_quality_label"] = quality_label
            
            # Fix crawler_type if it's pending
            if result.get("crawler_type") == "pending":
                if has_apollo:
                    result["crawler_type"] = "apollo_data"
                else:
                    result["crawler_type"] = "direct_fallback"
            
            # Ensure confidence scores are consistent for MSP classification
            if result.get("predicted_class") == "Managed Service Provider":
                result["confidence_score"] = 90
                result["confidence_scores"] = {
                    "Managed Service Provider": 90,
                    "Integrator - Commercial A/V": 5,
                    "Integrator - Residential A/V": 5,
                    "Internal IT Department": 0
                }
            
            # Call original formatter
            formatted = original_format(result)
            
            # Ensure our keys are in the formatted result
            formatted["01_content_quality"] = result.get("content_quality", 0)
            formatted["01_content_quality_label"] = result.get("content_quality_label", "Unknown")
            formatted["01_has_apollo_data"] = result.get("has_apollo_data", False)
            
            # Force the description in the final output
            formatted["03_description"] = result.get("company_description", "")
            
            return formatted
        
        # Apply the patch
        api_formatter.format_api_response = patched_format_api
        logger.info("✅ Successfully patched api_formatter.format_api_response with improved company descriptions")
    except Exception as e:
        logger.error(f"❌ Failed to patch api_formatter: {e}")
    
    # 2. Patch cross validator to update confidence scores
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
                    
                    # Add Apollo data indicator if Apollo data exists
                    if apollo_data:
                        result["has_apollo_data"] = True
                    
                    # Set content quality if missing
                    if "content_quality" not in result:
                        result["content_quality"] = 2  # Assume substantial if we have Apollo data
                    
                    logger.info(f"Updated confidence scores and enriched data after reclassification to MSP")
            
            return result
        
        # Apply the patch
        cross_validator.reconcile_classification = patched_reconcile
        logger.info("✅ Successfully patched cross_validator.reconcile_classification with enhanced fields")
    except Exception as e:
        logger.error(f"❌ Failed to patch cross_validator: {e}")
    
    # 3. Patch process_fresh_result to set has_apollo_data correctly
    try:
        from domain_classifier.storage import result_processor
        
        original_process_fresh = result_processor.process_fresh_result
        
        def patched_process_fresh(classification, domain, email=None, url=None):
            # First, determine Apollo data availability
            has_apollo = False
            apollo_data = classification.get("apollo_data", {})
            
            if apollo_data:
                if isinstance(apollo_data, str):
                    try:
                        apollo_data = json.loads(apollo_data)
                    except:
                        apollo_data = {}
                
                # Check if Apollo data has meaningful content
                if isinstance(apollo_data, dict):
                    key_fields = ["name", "industry", "employee_count", "founded_year", "phone"]
                    has_apollo = any(apollo_data.get(field) for field in key_fields)
            
            classification["has_apollo_data"] = has_apollo
            
            # Ensure company_description is substantial
            if not classification.get("company_description") or len(classification.get("company_description", "")) < 30:
                company_class = classification.get("predicted_class", "business")
                company_name = classification.get("company_name", domain.split('.')[0].capitalize())
                
                # Create detailed description
                if has_apollo and isinstance(apollo_data, dict):
                    industry = apollo_data.get("industry", "")
                    founded_year = apollo_data.get("founded_year", "")
                    employees = apollo_data.get("employee_count", "")
                    
                    # Create comprehensive description with Apollo data
                    description = f"{company_name} is a {company_class}"
                    if industry:
                        description += f" in the {industry} industry"
                    if founded_year:
                        description += f", founded in {founded_year}"
                    if employees:
                        description += f" with approximately {employees} employees"
                    description += "."
                    
                    if company_class == "Managed Service Provider":
                        description += " The company provides IT services, support, and solutions for businesses."
                    elif "Integrator - Commercial A/V" in company_class:
                        description += " The company provides audio-visual solutions for commercial clients."
                    elif "Integrator - Residential A/V" in company_class:
                        description += " The company specializes in home automation and entertainment systems."
                    
                    classification["company_description"] = description
                else:
                    # Fallback description
                    if company_class == "Managed Service Provider":
                        classification["company_description"] = f"{company_name} is a Managed Service Provider offering IT services, infrastructure support, and technology solutions to businesses."
                    elif "Integrator - Commercial A/V" in company_class:
                        classification["company_description"] = f"{company_name} is a Commercial A/V Integrator providing audio-visual systems for corporate environments, including conference rooms and presentation solutions."
                    elif "Integrator - Residential A/V" in company_class:
                        classification["company_description"] = f"{company_name} is a Residential A/V Integrator specializing in home automation, entertainment systems, and smart home technology for residential clients."
                    else:
                        classification["company_description"] = f"{company_name} is a business with internal IT needs rather than a provider of technology services to external clients."
                
                logger.info(f"Added detailed company description for {domain}")
            
            # Set crawler_type if it's missing or "pending"
            if "crawler_type" not in classification or classification.get("crawler_type") == "pending":
                classification["crawler_type"] = "direct_fallback"
            
            # Call original function
            result = original_process_fresh(classification, domain, email, url)
            
            # Ensure our fields are present in the result
            result["has_apollo_data"] = has_apollo
            
            # Set content quality if missing
            if "content_quality" not in result:
                result["content_quality"] = 0
                if has_apollo:  # If we have Apollo data, assume we have content
                    result["content_quality"] = 2
            
            return result
        
        # Apply the patch
        result_processor.process_fresh_result = patched_process_fresh
        logger.info("✅ Successfully patched result_processor.process_fresh_result with enhanced description handling")
    except Exception as e:
        logger.error(f"❌ Failed to patch result_processor: {e}")
    
    logger.info("Domain classifier fixes applied with improved description handling")
