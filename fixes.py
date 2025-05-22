"""
Domain Classifier Anti-Hallucination Fixes with Complete Verified Details

This script applies strict verification while ensuring all available verified details are included.
"""

import logging
import re
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def apply_patches():
    """Apply anti-hallucination patches while preserving all verified details."""
    logger.info("Applying domain classifier anti-hallucination fixes with complete verified details...")
    
    # 1. Improve description generation to include all verified details
    try:
        from domain_classifier.utils import api_formatter
        
        original_format = api_formatter.format_api_response
        
        def patched_format_api(result):
            # Check for personal/test domains
            domain = result.get("domain", "")
            email = result.get("email", "")
            
            # Personal email patterns and suspicious domains (same as before)
            personal_email_patterns = [
                r'@gmail\.com$', r'@hotmail\.com$', r'@yahoo\.com$', r'@outlook\.com$',
                r'@icloud\.com$', r'@aol\.com$', r'@mail\.com$', r'@protonmail\.com$',
                r'@sasktel\.net$', r'@comcast\.net$', r'@verizon\.net$'
            ]
            
            suspicious_domains = [
                'notadoodle.com', 'test', 'example', 'fake', 'demo', 'sample',
                'dcpa.net', 'asdf', 'qwerty', 'aaaaaa', 'zzzzz'
            ]
            
            # Check if domain/email is suspicious
            is_suspicious = False
            if email:
                if any(re.search(pattern, email.lower()) for pattern in personal_email_patterns):
                    is_suspicious = True
                    logger.warning(f"Detected personal email pattern: {email}")
                if any(fake in email.lower() for fake in ['test', 'demo', 'fake', 'example', 'asdf', 'qwerty']):
                    is_suspicious = True
                    logger.warning(f"Detected test/fake email pattern: {email}")
            
            if domain and any(susp in domain.lower() for susp in suspicious_domains):
                is_suspicious = True
                logger.warning(f"Detected suspicious domain pattern: {domain}")
            
            # Gather all available verified data
            verified_data = {}
            
            # Extract data from AI extraction if available
            ai_data = {}
            if "ai_company_data" in result and result["ai_company_data"]:
                if isinstance(result["ai_company_data"], str):
                    try:
                        ai_data = json.loads(result["ai_company_data"])
                    except:
                        ai_data = {}
                else:
                    ai_data = result["ai_company_data"]
            
            if ai_data and isinstance(ai_data, dict):
                # Add all AI-extracted data to verified data
                for key, value in ai_data.items():
                    if value and key not in ["source"]:  # Skip source field
                        verified_data[f"ai_{key}"] = value
            
            # Extract data from Apollo
            apollo_data = {}
            has_apollo = False
            if "apollo_data" in result and result["apollo_data"]:
                if isinstance(result["apollo_data"], str):
                    try:
                        apollo_data = json.loads(result["apollo_data"])
                    except:
                        apollo_data = {}
                else:
                    apollo_data = result["apollo_data"]
                
                if apollo_data and isinstance(apollo_data, dict):
                    # Check for meaningful Apollo data
                    key_fields = ["name", "industry", "employee_count", "phone"]
                    has_apollo = any(apollo_data.get(field) for field in key_fields)
                    
                    if has_apollo:
                        # Add all Apollo data to verified data
                        for key, value in apollo_data.items():
                            if value and key not in ["source"]:  # Skip source field
                                verified_data[f"apollo_{key}"] = value
                        
                        # Extract location data from Apollo
                        if "address" in apollo_data and isinstance(apollo_data["address"], dict):
                            address = apollo_data["address"]
                            for addr_key, addr_value in address.items():
                                if addr_value:
                                    verified_data[f"apollo_address_{addr_key}"] = addr_value
            
            # Set the Apollo flag
            result["has_apollo_data"] = has_apollo
            
            # Generate description based on all verified data
            if is_suspicious:
                # For suspicious domains, provide a clear but non-fabricated description
                result["company_description"] = f"Limited verified information is available for {domain}. This may be a personal or test domain rather than a business entity."
                logger.info(f"Added minimal non-fabricated description for suspicious domain: {domain}")
            else:
                # For all other domains, include ALL verified data without fabrication
                company_class = result.get("predicted_class", "Unknown")
                company_name = result.get("company_name", domain.split('.')[0].capitalize())
                
                # Start with basic classification
                description = f"{company_name} is classified as a {company_class}"
                
                # Add industry if available
                if "apollo_industry" in verified_data:
                    description += f" in the {verified_data['apollo_industry']} industry"
                elif "ai_industry" in verified_data:
                    description += f" in the {verified_data['ai_industry']} industry"
                
                # Add location if available
                location_parts = []
                
                # Check for city
                if "apollo_address_city" in verified_data:
                    location_parts.append(verified_data["apollo_address_city"])
                elif "ai_city" in verified_data:
                    location_parts.append(verified_data["ai_city"])
                
                # Check for state
                if "apollo_address_state" in verified_data:
                    location_parts.append(verified_data["apollo_address_state"])
                elif "ai_state" in verified_data:
                    location_parts.append(verified_data["ai_state"])
                
                # Check for country (only if not US)
                country = None
                if "apollo_address_country" in verified_data:
                    country = verified_data["apollo_address_country"]
                elif "ai_country" in verified_data:
                    country = verified_data["ai_country"]
                
                if country and country not in ["United States", "US", "USA"]:
                    location_parts.append(country)
                
                # Add location to description
                if location_parts:
                    description += f", based in {', '.join(location_parts)}"
                
                # Close the first sentence
                description += "."
                
                # Add founding year if available
                if "apollo_founded_year" in verified_data:
                    description += f" Founded in {verified_data['apollo_founded_year']}."
                elif "ai_founded_year" in verified_data:
                    description += f" Founded in {verified_data['ai_founded_year']}."
                
                # Add employee count if available
                if "apollo_employee_count" in verified_data:
                    description += f" The company has approximately {verified_data['apollo_employee_count']} employees."
                elif "ai_employee_count" in verified_data:
                    description += f" The company has approximately {verified_data['ai_employee_count']} employees."
                
                # Add phone if available
                if "apollo_phone" in verified_data:
                    description += f" Contact phone: {verified_data['apollo_phone']}."
                elif "ai_phone" in verified_data:
                    description += f" Contact phone: {verified_data['ai_phone']}."
                
                # Add factual statements based on classification without fabricating specific services
                if company_class == "Managed Service Provider":
                    description += f" As an MSP, {company_name} provides IT services and technology solutions to client businesses."
                elif company_class == "Integrator - Commercial A/V":
                    description += f" As a Commercial A/V Integrator, {company_name} provides audio-visual solutions for business environments."
                elif company_class == "Integrator - Residential A/V":
                    description += f" As a Residential A/V Integrator, {company_name} specializes in home automation and entertainment systems."
                elif company_class == "Internal IT Department":
                    description += f" This classification indicates {company_name} is a business with internal IT needs rather than a provider of technology services."
                
                # Add LinkedIn if available
                if "apollo_linkedin_url" in verified_data:
                    description += f" LinkedIn: {verified_data['apollo_linkedin_url']}"
                
                # Clean up formatting
                description = description.replace("  ", " ")
                description = description.replace("..", ".")
                
                result["company_description"] = description
                logger.info(f"Added comprehensive verified description for {domain}")
            
            # Improve content quality determination
            content_source = result.get("content_source", "")
            crawler_type = result.get("crawler_type", "")
            
            content_quality = 0  # Default: no content
            
            # Determine content quality based on actual content and crawling
            if content_source in ["fresh_crawl", "existing_content"] or crawler_type in ["direct_https", "direct_http", "scrapy"]:
                content_quality = 2  # Substantial content
            elif has_apollo or "ai_company_data" in result:
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
                    result["crawler_type"] = "unknown_source"
            
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
        logger.info("✅ Successfully patched api_formatter.format_api_response with comprehensive verified details")
    except Exception as e:
        logger.error(f"❌ Failed to patch api_formatter: {e}")
    
    # 2. Fix cross-validation to avoid incorrect reclassifications
    try:
        from domain_classifier.utils import cross_validator
        
        original_reconcile = cross_validator.reconcile_classification
        
        def patched_reconcile(classification, apollo_data=None, ai_data=None):
            # First, check if this is a likely personal/test domain
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
            
            is_suspicious = False
            
            # Check email for personal patterns
            if email:
                if any(re.search(pattern, email.lower()) for pattern in personal_email_patterns):
                    is_suspicious = True
                    logger.warning(f"Cross-validator detected personal email pattern: {email}")
                
                if any(fake in email.lower() for fake in ['test', 'demo', 'fake', 'example', 'asdf', 'qwerty']):
                    is_suspicious = True
                    logger.warning(f"Cross-validator detected test/fake email pattern: {email}")
            
            # Check domain for suspicious patterns
            if domain and any(susp in domain.lower() for susp in suspicious_domains):
                is_suspicious = True
                logger.warning(f"Cross-validator detected suspicious domain pattern: {domain}")
            
            # For suspicious domains, always classify as Internal IT without any cross-validation
            if is_suspicious:
                logger.warning(f"Skipping cross-validation for suspicious domain/email: {domain}/{email}")
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
            
            # For non-suspicious domains, verify Apollo industry before using it
            verified_apollo_industry = None
            
            if apollo_data:
                if isinstance(apollo_data, str):
                    try:
                        apollo_data = json.loads(apollo_data)
                    except:
                        apollo_data = {}
                
                if isinstance(apollo_data, dict) and apollo_data.get("industry"):
                    # Only trust Apollo industry if we have sufficient corroborating info
                    has_corroborating_info = any([
                        apollo_data.get("name"),
                        apollo_data.get("employee_count"),
                        apollo_data.get("phone"),
                        apollo_data.get("website")
                    ])
                    
                    if has_corroborating_info:
                        verified_apollo_industry = apollo_data.get("industry", "").lower()
                        logger.info(f"Verified Apollo industry data for {domain}: {verified_apollo_industry}")
                    else:
                        logger.warning(f"Insufficient Apollo data verification for {domain} - not using for cross-validation")
            
            # Check if domain name contains strong IT service indicators
            domain_name_only = domain.split('.')[0].lower() if domain else ""
            has_strong_msp_indicators = any(term in domain_name_only for term in ["tech", "it", "compute", "data", "net", "host", "cloud", "cyber"])
            
            # Now call original reconciliation
            result = original_reconcile(classification, apollo_data, ai_data)
            
            # Apply special logic for MSP reclassification
            if result.get("predicted_class") != classification.get("predicted_class"):
                # If classification changed to MSP, verify it's appropriate
                if result.get("predicted_class") == "Managed Service Provider":
                    # Only accept the change with verified evidence
                    if verified_apollo_industry and "information technology" in verified_apollo_industry:
                        # This looks legitimate - update confidence scores
                        result["confidence_scores"] = {
                            "Managed Service Provider": 90,
                            "Integrator - Commercial A/V": 5,
                            "Integrator - Residential A/V": 5,
                            "Internal IT Department": 0
                        }
                        result["confidence_score"] = 90
                        result["max_confidence"] = 0.9
                        result["detection_method"] = "cross_validation_verified_it_services"
                        logger.info(f"Verified reclassification to MSP based on IT industry data for {domain}")
                    elif has_strong_msp_indicators:
                        # Domain name strongly suggests MSP
                        result["confidence_scores"] = {
                            "Managed Service Provider": 80,
                            "Integrator - Commercial A/V": 10,
                            "Integrator - Residential A/V": 5,
                            "Internal IT Department": 0
                        }
                        result["confidence_score"] = 80
                        result["max_confidence"] = 0.8
                        result["detection_method"] = "cross_validation_domain_indicators"
                        logger.info(f"Reclassification to MSP supported by domain name indicators for {domain}")
                    else:
                        # Insufficient evidence - revert to original classification
                        logger.warning(f"Insufficient evidence for MSP reclassification for {domain} - reverting")
                        result["predicted_class"] = classification.get("predicted_class")
                        result["detection_method"] = classification.get("detection_method", "classification_preserved")
                        
                        # Preserve original confidence scores
                        for key in ["confidence_scores", "confidence_score", "max_confidence"]:
                            if key in classification:
                                result[key] = classification[key]
            
            return result
        
        # Apply the patch
        cross_validator.reconcile_classification = patched_reconcile
        logger.info("✅ Successfully patched cross_validator.reconcile_classification with improved verification")
    except Exception as e:
        logger.error(f"❌ Failed to patch cross_validator: {e}")
    
    logger.info("Domain classifier fixes applied with comprehensive verified details and anti-hallucination measures")
