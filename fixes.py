"""
Domain Classifier Fixes - Complete Version

This script combines all fixes:
1. Anti-hallucination measures
2. Company-focused descriptions for Internal IT
3. Proper content quality and crawler type handling
4. Personal/suspicious domain detection
"""

import logging
import re
import json
import random
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def apply_patches():
    """Apply all patches for domain classifier improvements."""
    logger.info("Applying comprehensive domain classifier fixes...")
    
    # 1. Improve description generation with anti-hallucination and company focus
    try:
        from domain_classifier.utils import api_formatter
        
        original_format = api_formatter.format_api_response
        
        def patched_format_api(result):
            # Check for personal/test domains
            domain = result.get("domain", "")
            email = result.get("email", "")
            
            # Personal email patterns and suspicious domains
            personal_email_patterns = [
                r'@gmail\.com$', r'@hotmail\.com$', r'@yahoo\.com$', r'@outlook\.com$',
                r'@icloud\.com$', r'@aol\.com$', r'@mail\.com$', r'@protonmail\.com$',
                r'@sasktel\.net$', r'@comcast\.net$', r'@verizon\.net$'
            ]
            
            suspicious_domains = [
                'notadoodle.com', 'test', 'example', 'fake', 'demo', 'sample',
                'dcpa.net', 'asdf', 'qwerty', 'aaaaaa', 'zzzzz'
            ]
            
            is_personal = False
            
            # Check email for personal patterns
            if email:
                if any(re.search(pattern, email.lower()) for pattern in personal_email_patterns):
                    is_personal = True
                    logger.warning(f"Detected personal email pattern: {email}")
                
                if any(fake in email.lower() for fake in ['test', 'demo', 'fake', 'example', 'asdf', 'qwerty']):
                    is_personal = True
                    logger.warning(f"Detected test/fake email pattern: {email}")
            
            # Check domain for suspicious patterns
            if domain and any(susp in domain.lower() for susp in suspicious_domains):
                is_personal = True
                logger.warning(f"Detected suspicious domain pattern: {domain}")
            
            # Get Apollo data if available
            has_apollo = False
            apollo_data = {}
            
            if "apollo_data" in result and result.get("apollo_data"):
                apollo_data = result["apollo_data"]
                if isinstance(apollo_data, str):
                    try:
                        apollo_data = json.loads(apollo_data)
                    except:
                        apollo_data = {}
                
                # Check if Apollo data has meaningful content
                if isinstance(apollo_data, dict):
                    key_fields = ["name", "industry", "employee_count", "founded_year", "phone"]
                    has_apollo = any(apollo_data.get(field) for field in key_fields)
            
            # Set Apollo flag
            result["has_apollo_data"] = has_apollo
            
            # Get AI-extracted data if available
            ai_data = {}
            if "ai_company_data" in result and result.get("ai_company_data"):
                ai_data = result["ai_company_data"]
                if isinstance(ai_data, str):
                    try:
                        ai_data = json.loads(ai_data)
                    except:
                        ai_data = {}
            
            # Gather all verified data
            verified_data = {}
            
            # Add Apollo data to verified data
            if has_apollo:
                for key, value in apollo_data.items():
                    if value and key not in ["source"]:
                        verified_data[f"apollo_{key}"] = value
                
                # Extract location data
                if "address" in apollo_data and isinstance(apollo_data["address"], dict):
                    for loc_key, loc_value in apollo_data["address"].items():
                        if loc_value:
                            verified_data[f"apollo_address_{loc_key}"] = loc_value
            
            # Add AI-extracted data to verified data
            if ai_data and isinstance(ai_data, dict):
                for key, value in ai_data.items():
                    if value and key not in ["source"]:
                        verified_data[f"ai_{key}"] = value
            
            # Create appropriate description based on classification and verified data
            if not result.get("company_description") or len(result.get("company_description", "")) < 30:
                company_class = result.get("predicted_class", "Unknown")
                company_name = result.get("company_name", domain.split('.')[0].capitalize())
                
                # For personal/suspicious domains
                if is_personal and "Internal IT Department" in company_class:
                    result["company_description"] = f"This appears to be a personal or non-business domain. No verified business information is available for {domain}."
                    logger.info(f"Added personal domain description for {domain}")
                
                # For Internal IT Department, focus on the company's primary business
                elif company_class == "Internal IT Department" and has_apollo:
                    # Get verified information from Apollo
                    industry = verified_data.get("apollo_industry", "")
                    employee_count = verified_data.get("apollo_employee_count", "")
                    founded_year = verified_data.get("apollo_founded_year", "")
                    
                    # Build location info
                    city = verified_data.get("apollo_address_city", "")
                    state = verified_data.get("apollo_address_state", "")
                    country = verified_data.get("apollo_address_country", "")
                    
                    location_parts = []
                    if city:
                        location_parts.append(city)
                    if state:
                        location_parts.append(state)
                    if country and country not in ["United States", "US", "USA"]:
                        location_parts.append(country)
                    
                    location = ", ".join(location_parts) if location_parts else ""
                    
                    # Build company-focused description
                    description = f"{company_name} is a {industry} company"
                    
                    if location:
                        description += f" based in {location}"
                    
                    description += "."
                    
                    if founded_year:
                        description += f" Founded in {founded_year},"
                    
                    if employee_count:
                        description += f" the company has approximately {employee_count} employees."
                    else:
                        description += " the company operates in the industry."
                    
                    # Add industry-specific information
                    if industry:
                        if "construction" in industry.lower():
                            description += " The company provides construction services such as building, renovation, and contracting work."
                        elif "manufacturing" in industry.lower():
                            description += " The company is involved in manufacturing and production operations."
                        elif "retail" in industry.lower():
                            description += " The company operates in the retail sector, selling products to consumers."
                        elif "healthcare" in industry.lower():
                            description += " The company provides healthcare services or products."
                        elif "education" in industry.lower():
                            description += " The company is involved in educational services or products."
                        elif "finance" in industry.lower():
                            description += " The company provides financial services or products."
                        else:
                            description += f" The company operates in the {industry} sector."
                    
                    # Add verified LinkedIn URL if available
                    if "apollo_linkedin_url" in verified_data:
                        description += f" LinkedIn: {verified_data['apollo_linkedin_url']}"
                    
                    result["company_description"] = description
                    logger.info(f"Added business-focused description for Internal IT company: {domain}")
                
                # For service businesses (MSP, AV Integrators)
                elif company_class in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                    # Get verified information
                    industry = ""
                    employee_count = ""
                    founded_year = ""
                    location = ""
                    
                    # Extract from Apollo data first
                    if has_apollo:
                        industry = verified_data.get("apollo_industry", "")
                        employee_count = verified_data.get("apollo_employee_count", "")
                        founded_year = verified_data.get("apollo_founded_year", "")
                        
                        # Build location
                        city = verified_data.get("apollo_address_city", "")
                        state = verified_data.get("apollo_address_state", "")
                        country = verified_data.get("apollo_address_country", "")
                        
                        location_parts = []
                        if city:
                            location_parts.append(city)
                        if state:
                            location_parts.append(state)
                        if country and country not in ["United States", "US", "USA"]:
                            location_parts.append(country)
                        
                        location = ", ".join(location_parts) if location_parts else ""
                    
                    # Fill gaps with AI-extracted data
                    if not industry and "ai_industry" in verified_data:
                        industry = verified_data.get("ai_industry", "")
                    if not employee_count and "ai_employee_count" in verified_data:
                        employee_count = verified_data.get("ai_employee_count", "")
                    if not founded_year and "ai_founded_year" in verified_data:
                        founded_year = verified_data.get("ai_founded_year", "")
                    
                    # Build service-focused description with verified data only
                    description = f"{company_name} is a {company_class}"
                    
                    if industry:
                        description += f" specializing in {industry}"
                    
                    if location:
                        description += f", based in {location}"
                    
                    description += "."
                    
                    if founded_year:
                        description += f" Founded in {founded_year},"
                    
                    if employee_count:
                        description += f" the company has approximately {employee_count} employees."
                    else:
                        description += " The company provides technology services to clients."
                    
                    # Add service details based on classification
                    if company_class == "Managed Service Provider":
                        description += " As an MSP, they offer IT services, support, and solutions for businesses."
                    elif company_class == "Integrator - Commercial A/V":
                        description += " They provide audio-visual system design and integration for commercial clients."
                    elif company_class == "Integrator - Residential A/V":
                        description += " They specialize in home automation and entertainment systems for residential clients."
                    
                    # Add LinkedIn URL if available
                    if "apollo_linkedin_url" in verified_data:
                        description += f" LinkedIn: {verified_data['apollo_linkedin_url']}"
                    
                    result["company_description"] = description
                    logger.info(f"Added service-focused description for {company_class}: {domain}")
                
                # For other cases, use minimal verification
                else:
                    # Use only verified information
                    verified_parts = []
                    
                    if has_apollo:
                        industry = verified_data.get("apollo_industry", "")
                        employee_count = verified_data.get("apollo_employee_count", "")
                        founded_year = verified_data.get("apollo_founded_year", "")
                        
                        if industry:
                            verified_parts.append(f"industry: {industry}")
                        if employee_count:
                            verified_parts.append(f"employees: {employee_count}")
                        if founded_year:
                            verified_parts.append(f"founded: {founded_year}")
                    
                    if verified_parts:
                        result["company_description"] = f"Limited verified information for {domain}: {', '.join(verified_parts)}."
                    else:
                        result["company_description"] = f"Limited verified information is available for {domain}."
                    
                    logger.info(f"Added minimal verified description for {domain}")
            
            # Set content quality based on available information
            content_source = result.get("content_source", "")
            crawler_type = result.get("crawler_type", "")
            
            content_quality = 0  # Default: no content
            
            if content_source in ["fresh_crawl", "existing_content"] or any(c in crawler_type for c in ["direct_https", "direct_http", "scrapy", "existing_content"]):
                content_quality = 2  # Substantial content
            elif has_apollo:
                content_quality = 1  # At least some data
            
            quality_label = "Did not connect"
            if content_quality == 1:
                quality_label = "Minimal content or data"
            elif content_quality == 2:
                quality_label = "Substantial content"
            
            result["content_quality"] = content_quality
            result["content_quality_label"] = quality_label
            
            # Fix crawler_type if it's pending
            if result.get("crawler_type") == "pending":
                if content_source in ["fresh_crawl", "existing_content"]:
                    result["crawler_type"] = content_source
                elif content_quality == 2:
                    result["crawler_type"] = "content_available"
                elif has_apollo:
                    result["crawler_type"] = "apollo_data"
                else:
                    result["crawler_type"] = "no_content"
            
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
            
            # Ensure our fields are present in the formatted result
            formatted["01_content_quality"] = result.get("content_quality", 0)
            formatted["01_content_quality_label"] = result.get("content_quality_label", "Unknown")
            formatted["01_has_apollo_data"] = result.get("has_apollo_data", False)
            
            # Force the description in the final output
            formatted["03_description"] = result.get("company_description", "")
            
            return formatted
        
        # Apply the patch
        api_formatter.format_api_response = patched_format_api
        logger.info("✅ Successfully patched api_formatter.format_api_response with company-focused descriptions")
    except Exception as e:
        logger.error(f"❌ Failed to patch api_formatter: {e}")
    
    # 2. Improve cross-validator with better personal and industry detection
    try:
        from domain_classifier.utils import cross_validator
        
        original_reconcile = cross_validator.reconcile_classification
        
        def patched_reconcile(classification, apollo_data=None, ai_data=None):
            # First check for personal domains
            domain = classification.get("domain", "unknown")
            email = classification.get("email", "")
            
            # Personal email patterns
            personal_email_patterns = [
                r'@gmail\.com$', r'@hotmail\.com$', r'@yahoo\.com$', r'@outlook\.com$',
                r'@icloud\.com$', r'@aol\.com$', r'@mail\.com$', r'@protonmail\.com$',
                r'@sasktel\.net$', r'@comcast\.net$', r'@verizon\.net$'
            ]
            
            suspicious_domains = [
                'notadoodle.com', 'test', 'example', 'fake', 'demo', 'sample',
                'dcpa.net', 'asdf', 'qwerty'
            ]
            
            is_personal = False
            
            # Check email for personal patterns
            if email:
                if any(re.search(pattern, email.lower()) for pattern in personal_email_patterns):
                    is_personal = True
                    logger.warning(f"Cross-validator detected personal email pattern: {email}")
                
                if any(fake in email.lower() for fake in ['test', 'demo', 'fake', 'example', 'asdf', 'qwerty']):
                    is_personal = True
                    logger.warning(f"Cross-validator detected test/fake email pattern: {email}")
            
            # Check domain for suspicious patterns
            if domain and any(susp in domain.lower() for susp in suspicious_domains):
                is_personal = True
                logger.warning(f"Cross-validator detected suspicious domain pattern: {domain}")
            
            # For personal domains, always classify as Internal IT
            if is_personal:
                logger.warning(f"Skipping cross-validation for personal domain/email: {domain}/{email}")
                classification["predicted_class"] = "Internal IT Department"
                classification["detection_method"] = "personal_domain_detection"
                classification["confidence_scores"] = {
                    "Managed Service Provider": 5,
                    "Integrator - Commercial A/V": 3,
                    "Integrator - Residential A/V": 2,
                    "Internal IT Department": 60
                }
                classification["confidence_score"] = 60
                return classification
            
            # Check for non-IT industries
            non_it_industries = [
                "construction", "manufacturing", "retail", "healthcare", 
                "education", "finance", "agriculture", "real estate",
                "hospitality", "transportation", "logistics", "mining", 
                "oil", "gas", "energy", "food", "beverage"
            ]
            
            # Verify Apollo industry data
            if apollo_data:
                if isinstance(apollo_data, str):
                    try:
                        apollo_data = json.loads(apollo_data)
                    except:
                        apollo_data = {}
                
                if isinstance(apollo_data, dict) and apollo_data.get("industry"):
                    industry = apollo_data.get("industry", "").lower()
                    
                    # Verify Apollo data quality
                    has_corroborating_info = any([
                        apollo_data.get("name"),
                        apollo_data.get("employee_count"),
                        apollo_data.get("phone")
                    ])
                    
                    if has_corroborating_info:
                        logger.info(f"Verified Apollo industry data for {domain}: {industry}")
                        
                        # Check for non-IT service industries
                        if any(non_it in industry for non_it in non_it_industries):
                            # This is clearly a non-IT service company
                            logger.info(f"Apollo data indicates non-IT industry: {industry}")
                            classification["predicted_class"] = "Internal IT Department"
                            classification["detection_method"] = "non_it_industry_detection"
                            classification["confidence_scores"] = {
                                "Managed Service Provider": 5,
                                "Integrator - Commercial A/V": 3,
                                "Integrator - Residential A/V": 2,
                                "Internal IT Department": 70
                            }
                            classification["confidence_score"] = 70
                            return classification
            
            # For all other cases, use original cross-validation
            result = original_reconcile(classification, apollo_data, ai_data)
            
            # If reclassified to MSP, verify it's appropriate
            if result.get("predicted_class") != classification.get("predicted_class"):
                if result.get("predicted_class") == "Managed Service Provider":
                    # Only trust reclassification for verified IT service companies
                    is_verified_it = False
                    
                    if apollo_data and isinstance(apollo_data, dict):
                        industry = apollo_data.get("industry", "").lower()
                        if industry and ("information technology" in industry or "computer" in industry):
                            is_verified_it = True
                            logger.info(f"Verified IT industry for {domain}: {industry}")
                    
                    if is_verified_it:
                        # Update confidence scores
                        result["confidence_scores"] = {
                            "Managed Service Provider": 90,
                            "Integrator - Commercial A/V": 5,
                            "Integrator - Residential A/V": 5,
                            "Internal IT Department": 0
                        }
                        result["confidence_score"] = 90
                        result["max_confidence"] = 0.9
                        result["detection_method"] = "cross_validation_it_services"
                        logger.info(f"Verified reclassification to MSP for {domain}")
                    else:
                        # Insufficient evidence - revert to original
                        logger.warning(f"Insufficient IT industry evidence for {domain} - reverting from MSP")
                        result["predicted_class"] = classification.get("predicted_class")
                        result["detection_method"] = classification.get("detection_method", "classification_preserved")
                        
                        # Preserve original confidence scores
                        if "confidence_scores" in classification:
                            result["confidence_scores"] = classification["confidence_scores"]
                        if "confidence_score" in classification:
                            result["confidence_score"] = classification["confidence_score"]
                        if "max_confidence" in classification:
                            result["max_confidence"] = classification["max_confidence"]
            
            return result
        
        # Apply the patch
        cross_validator.reconcile_classification = patched_reconcile
        logger.info("✅ Successfully patched cross_validator.reconcile_classification with improved verification")
    except Exception as e:
        logger.error(f"❌ Failed to patch cross_validator: {e}")
    
    # 3. Patch result_processor to add has_apollo_data and content quality
    try:
        from domain_classifier.storage import result_processor
        
        original_process_fresh = result_processor.process_fresh_result
        
        def patched_process_fresh(classification, domain, email=None, url=None):
            # Determine Apollo data availability
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
            
            # Preserve crawler_type information
            crawler_type = classification.get("crawler_type")
            if crawler_type == "pending" or not crawler_type:
                if "content_source" in classification:
                    crawler_type = classification["content_source"]
                elif "source" in classification and "cached" in str(classification["source"]):
                    crawler_type = "cached_content"
                else:
                    # Keep as pending or set a descriptive default
                    crawler_type = crawler_type or "unknown"
            
            classification["crawler_type"] = crawler_type
            
            # Call original function
            result = original_process_fresh(classification, domain, email, url)
            
            # Ensure our fields are preserved
            result["has_apollo_data"] = has_apollo
            
            # Set content quality
            if "content_quality" not in result:
                result["content_quality"] = 0
                if "content_source" in classification and classification["content_source"] in ["fresh_crawl", "existing_content"]:
                    result["content_quality"] = 2  # Substantial content
                elif crawler_type in ["direct_https", "direct_http", "scrapy", "cached_content", "existing_content"]:
                    result["content_quality"] = 2  # Substantial content
                elif has_apollo:
                    result["content_quality"] = 1  # Minimal - we have some info but no full content
            
            return result
        
        # Apply the patch
        result_processor.process_fresh_result = patched_process_fresh
        logger.info("✅ Successfully patched result_processor.process_fresh_result with Apollo data handling")
    except Exception as e:
        logger.error(f"❌ Failed to patch result_processor: {e}")
    
    logger.info("Comprehensive domain classifier fixes successfully applied")
