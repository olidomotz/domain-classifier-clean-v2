"""
Domain Classifier Fixes - Complete Classification Hierarchy Fix

This script fixes all classification inconsistencies by implementing a clear priority:
1. Domain name evidence (highest)
2. Website content evidence (second)
3. Apollo company description field (lowest)
"""

import logging
import re
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def apply_patches():
    """Apply comprehensive classification hierarchy fixes."""
    logger.info("Applying classification hierarchy fixes...")
    
    # 1. Fix domain analysis to properly detect IT-related domains
    try:
        from domain_classifier.utils import domain_analysis
        
        original_analyze = domain_analysis.analyze_domain_words
        
        def patched_analyze_domain_words(domain):
            """Improved domain word analysis that properly detects IT solution patterns."""
            # First get original scores
            scores = original_analyze(domain)
            
            # Check for IT solution patterns that original analysis misses
            domain_lower = domain.lower()
            
            # Strong MSP indicators in domain
            msp_patterns = [
                r'it\s*solutions?',
                r'tech\s*solutions?', 
                r'managed\s*services?',
                r'tech\s*services?',
                r'it\s*services?',
                r'systems?\s*solutions?',
                r'security\s*solutions?',
                r'cloud\s*solutions?',
                r'computing\s*solutions?',
                r'consulting',
                r'tech\s*support',
                r'it\s*support',
                r'cyber',
                r'host',
            ]
            
            # Check if any pattern matches
            for pattern in msp_patterns:
                if re.search(pattern, domain_lower):
                    logger.info(f"Domain {domain} contains MSP indicator pattern: {pattern}")
                    # Override the MSP score to 1.0 (100%)
                    scores['msp_score'] = 1.0
                    break
            
            # If the domain has "it" or "tech" in it but no other score, boost MSP score
            if ('it' in domain_lower or 'tech' in domain_lower) and scores['msp_score'] == 0:
                logger.info(f"Domain {domain} contains 'it' or 'tech' - setting MSP score")
                scores['msp_score'] = 0.8
            
            return scores
        
        # Apply the patch
        domain_analysis.analyze_domain_words = patched_analyze_domain_words
        logger.info("✅ Successfully patched domain_analysis.analyze_domain_words to detect IT solutions")
    except Exception as e:
        logger.error(f"❌ Failed to patch domain_analysis: {e}")
    
    # 2. Fix cross_validator to implement strict classification hierarchy
    try:
        from domain_classifier.utils import cross_validator
        
        # Complete replacement of the reconcile function
        def patched_reconcile(classification, apollo_data=None, ai_data=None):
            """Completely rebuilt reconciliation with proper classification hierarchy."""
            domain = classification.get("domain", "unknown")
            original_class = classification.get("predicted_class", "")
            detection_method = classification.get("detection_method", "")
            
            logger.info(f"CROSS-VALIDATION START for {domain} with original class: {original_class}")
            
            # =========== LEVEL 1: DOMAIN NAME EVIDENCE (HIGHEST PRIORITY) ===========
            # Check domain name for strong MSP indicators
            domain_lower = domain.lower()
            
            # Strong MSP indicators in domain name
            msp_patterns = [
                r'it\s*solutions?',
                r'tech\s*solutions?', 
                r'managed\s*services?',
                r'tech\s*services?',
                r'it\s*services?',
                r'systems?\s*solutions?',
                r'security\s*solutions?',
                r'cloud\s*solutions?',
                r'computing\s*solutions?',
                r'consulting',
                r'tech\s*support',
                r'it\s*support',
                r'cyber',
                r'host',
            ]
            
            # Check if domain contains any MSP pattern
            is_msp_by_domain = False
            matched_pattern = None
            
            for pattern in msp_patterns:
                if re.search(pattern, domain_lower):
                    is_msp_by_domain = True
                    matched_pattern = pattern
                    break
            
            if is_msp_by_domain:
                logger.info(f"LEVEL 1 DECISION: Domain {domain} contains MSP indicator '{matched_pattern}' - classifying as MSP")
                classification["predicted_class"] = "Managed Service Provider"
                classification["detection_method"] = "domain_pattern_msp"
                classification["confidence_scores"] = {
                    "Managed Service Provider": 90,
                    "Integrator - Commercial A/V": 5,
                    "Integrator - Residential A/V": 5,
                    "Internal IT Department": 0
                }
                classification["confidence_score"] = 90
                classification["max_confidence"] = 0.9
                return classification
            
            # =========== LEVEL 2: WEBSITE CONTENT EVIDENCE (SECOND PRIORITY) ===========
            # If original classification is from LLM/content analysis, respect it
            if original_class and detection_method and (
                detection_method.startswith("llm_") or 
                detection_method == "text_parsing" or
                detection_method == "vector_similarity"
            ):
                logger.info(f"LEVEL 2 DECISION: Preserving content-based classification '{original_class}' for {domain}")
                
                # Ensure confidence scores are appropriate
                if original_class == "Managed Service Provider":
                    classification["confidence_scores"] = {
                        "Managed Service Provider": 90,
                        "Integrator - Commercial A/V": 5,
                        "Integrator - Residential A/V": 5,
                        "Internal IT Department": 0
                    }
                    classification["confidence_score"] = 90
                    classification["max_confidence"] = 0.9
                
                # Keep original classification
                return classification
            
            # =========== LEVEL 3: APOLLO COMPANY DESCRIPTION FIELD (LOWEST PRIORITY) ===========
            # Use Apollo description field if we have no stronger signals
            if apollo_data:
                if isinstance(apollo_data, str):
                    try:
                        apollo_data = json.loads(apollo_data)
                    except:
                        apollo_data = {}
                
                # Check for description field in Apollo data
                description = None
                if isinstance(apollo_data, dict) and apollo_data.get("description"):
                    description = apollo_data.get("description", "").lower()
                    logger.info(f"LEVEL 3: Found Apollo description field for {domain}")
                
                # If description exists, check for service indicators
                if description:
                    # Check for MSP indicators in description
                    msp_indicators = [
                        "managed service",
                        "it service", 
                        "it support",
                        "tech support",
                        "it consulting",
                        "network management",
                        "cloud service",
                        "security service"
                    ]
                    
                    if any(indicator in description for indicator in msp_indicators):
                        logger.info(f"LEVEL 3 DECISION: Apollo description indicates MSP - classifying as MSP")
                        classification["predicted_class"] = "Managed Service Provider"
                        classification["detection_method"] = "apollo_description_msp"
                        classification["confidence_scores"] = {
                            "Managed Service Provider": 80,
                            "Integrator - Commercial A/V": 5,
                            "Integrator - Residential A/V": 5,
                            "Internal IT Department": 0
                        }
                        classification["confidence_score"] = 80
                        classification["max_confidence"] = 0.8
                        return classification
                    
                    # Check for AV integrator indicators
                    av_indicators = [
                        "audio visual", 
                        "av integration",
                        "conference room",
                        "sound system",
                        "video system",
                        "home theater",
                        "home automation"
                    ]
                    
                    if any(indicator in description for indicator in av_indicators):
                        # Determine if commercial or residential
                        if any(term in description for term in ["commercial", "business", "corporate"]):
                            logger.info(f"LEVEL 3 DECISION: Apollo description indicates Commercial AV - classifying as Commercial AV")
                            classification["predicted_class"] = "Integrator - Commercial A/V"
                            classification["detection_method"] = "apollo_description_commercial_av"
                            classification["confidence_scores"] = {
                                "Managed Service Provider": 5,
                                "Integrator - Commercial A/V": 80,
                                "Integrator - Residential A/V": 10,
                                "Internal IT Department": 0
                            }
                            classification["confidence_score"] = 80
                            classification["max_confidence"] = 0.8
                            return classification
                        elif any(term in description for term in ["home", "residential"]):
                            logger.info(f"LEVEL 3 DECISION: Apollo description indicates Residential AV - classifying as Residential AV")
                            classification["predicted_class"] = "Integrator - Residential A/V"
                            classification["detection_method"] = "apollo_description_residential_av"
                            classification["confidence_scores"] = {
                                "Managed Service Provider": 5,
                                "Integrator - Commercial A/V": 10,
                                "Integrator - Residential A/V": 80,
                                "Internal IT Department": 0
                            }
                            classification["confidence_score"] = 80
                            classification["max_confidence"] = 0.8
                            return classification
            
            # If we get here, we don't have any clear signals - return classification unchanged
            logger.info(f"NO STRONG SIGNALS: Returning original classification for {domain}")
            return classification
        
        # Apply the patch - complete replacement
        cross_validator.reconcile_classification = patched_reconcile
        logger.info("✅ Successfully replaced cross_validator.reconcile_classification with hierarchy-based approach")
    except Exception as e:
        logger.error(f"❌ Failed to patch cross_validator: {e}")
    
    # 3. Add a check in api_formatter to warn about possible misclassifications
    try:
        from domain_classifier.utils import api_formatter
        
        original_format = api_formatter.format_api_response
        
        def patched_format_api(result):
            # Make sure company_description exists
            if not result.get("company_description"):
                domain = result.get("domain", "")
                predicted_class = result.get("predicted_class", "Unknown")
                company_name = result.get("company_name", domain.split('.')[0].capitalize())
                
                # Create a basic description
                description = f"{company_name} is classified as a {predicted_class}."
                
                # Add industry if available from Apollo
                apollo_data = result.get("apollo_data", {})
                if isinstance(apollo_data, dict) and apollo_data.get("industry"):
                    description += f" The company operates in the {apollo_data.get('industry')} industry."
                
                # Add type-specific information
                if predicted_class == "Managed Service Provider":
                    description += " They provide IT services and solutions to other businesses."
                elif "Integrator" in predicted_class:
                    description += " They provide technology integration services to clients."
                else:  # Internal IT
                    description += " The company has internal IT needs rather than providing technology services to others."
                
                result["company_description"] = description
                logger.info(f"Added emergency fallback description for {domain}")
            
            # Check for potential misclassifications
            domain = result.get("domain", "")
            predicted_class = result.get("predicted_class", "")
            
            # Look for IT Solutions domains classified as Internal IT
            if domain and ("it" in domain.lower() or "tech" in domain.lower()) and "solution" in domain.lower():
                if predicted_class != "Managed Service Provider":
                    # Add warning to description
                    warning = f" WARNING: This domain contains 'IT Solutions' and may be an MSP despite being classified as {predicted_class}."
                    
                    if "company_description" in result:
                        result["company_description"] = result["company_description"] + warning
            
            # Call original format function
            formatted = original_format(result)
            
            # Force company description into output
            if "company_description" in result:
                formatted["03_description"] = result["company_description"]
                logger.info(f"Forced company description into output for {domain}")
            
            return formatted
        
        # Apply the patch
        api_formatter.format_api_response = patched_format_api
        logger.info("✅ Successfully patched api_formatter.format_api_response to add misclassification warnings")
    except Exception as e:
        logger.error(f"❌ Failed to patch api_formatter: {e}")
    
    # 4. Fix final_classification to ensure IT Solutions domains are MSPs
    try:
        from domain_classifier.utils import final_classification
        
        original_determine = final_classification.determine_final_classification
        
        def patched_determine_final_classification(result):
            domain = result.get("domain", "")
            
            # Check for IT Solutions domains
            if domain and ("it" in domain.lower() or "tech" in domain.lower()) and "solution" in domain.lower():
                logger.info(f"Final classification override: IT Solutions domain {domain} to MSP")
                # Force MSP classification code
                return "1-MSP"
            
            # For all other cases, use original function
            return original_determine(result)
        
        # Apply the patch
        final_classification.determine_final_classification = patched_determine_final_classification
        logger.info("✅ Successfully patched final_classification.determine_final_classification for IT Solutions domains")
    except Exception as e:
        logger.error(f"❌ Failed to patch final_classification: {e}")
    
    # 5. Fix description_enhancer to ensure company descriptions are properly generated
    try:
        from domain_classifier.enrichment import description_enhancer
        
        # Completely replace both description enhancement functions 
        # rather than patching the Apollo connector which is causing issues
        def safe_generate_description(classification, apollo_data=None, apollo_person_data=None):
            """Generate descriptions focused on verified business information."""
            domain = classification.get("domain", "Unknown")
            predicted_class = classification.get("predicted_class", "")
            
            # Get company name
            company_name = ""
            if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("name"):
                company_name = apollo_data.get("name")
            else:
                company_name = classification.get("company_name", domain.split('.')[0].capitalize())
            
            # Use Apollo description if available
            if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("description"):
                apollo_description = apollo_data.get("description")
                logger.info(f"Using Apollo description field for {domain}")
                
                # Create a description incorporating the Apollo description
                if predicted_class == "Managed Service Provider":
                    description = f"{company_name} is a Managed Service Provider. {apollo_description}"
                elif predicted_class == "Integrator - Commercial A/V":
                    description = f"{company_name} is a Commercial A/V Integrator. {apollo_description}"
                elif predicted_class == "Integrator - Residential A/V":
                    description = f"{company_name} is a Residential A/V Integrator. {apollo_description}"
                else:  # Internal IT
                    description = f"{company_name} is a business. {apollo_description}"
                
                return description
            
            # Build description with focus on verified information if no Apollo description
            # Start with standard opening
            if predicted_class == "Managed Service Provider":
                description = f"{company_name} is a Managed Service Provider"
            elif predicted_class == "Integrator - Commercial A/V":
                description = f"{company_name} is a Commercial A/V Integrator"
            elif predicted_class == "Integrator - Residential A/V":
                description = f"{company_name} is a Residential A/V Integrator"
            else:  # Internal IT
                description = f"{company_name} is a business"
            
            # Add location if available
            location_parts = []
            if apollo_data and isinstance(apollo_data, dict) and "address" in apollo_data:
                address = apollo_data.get("address", {})
                if isinstance(address, dict):
                    if address.get("city"):
                        location_parts.append(address.get("city"))
                    if address.get("state"):
                        location_parts.append(address.get("state"))
                    if address.get("country") and address.get("country") not in ["United States", "US", "USA"]:
                        location_parts.append(address.get("country"))
            
            location = ", ".join(location_parts) if location_parts else ""
            if location:
                description += f" based in {location}"
            
            # Add industry if available
            if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("industry"):
                description += f". The company operates in the {apollo_data.get('industry')} sector"
            
            description += "."
            
            # Add founded info if available
            if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("founded_year"):
                description += f" Founded in {apollo_data.get('founded_year')}."
            
            # Add employee count if available
            if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("employee_count"):
                description += f" The company has approximately {apollo_data.get('employee_count')} employees."
            
            # Add standard type-specific descriptions
            if predicted_class == "Managed Service Provider":
                description += " They provide IT services, technology support, and solutions to their clients."
            elif predicted_class == "Integrator - Commercial A/V":
                description += " They provide audio-visual systems and solutions for business environments."
            elif predicted_class == "Integrator - Residential A/V":
                description += " They provide home automation and entertainment systems for residential clients."
            else:  # Internal IT
                if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("industry"):
                    description += f" As a {apollo_data.get('industry')} company, they have internal IT needs rather than providing technology services to others."
                else:
                    description += " The company has internal IT needs rather than providing technology services to others."
            
            # Add LinkedIn if available
            if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("linkedin_url"):
                description += f" LinkedIn: {apollo_data.get('linkedin_url')}"
            
            logger.info(f"Generated verified business-focused description for {domain}")
            return description
        
        # Define a simple wrapper for the other function
        def safe_enhance_company_description(basic_description, apollo_data, classification):
            return safe_generate_description(classification, apollo_data)
        
        # Replace the functions
        description_enhancer.generate_detailed_description = safe_generate_description
        description_enhancer.enhance_company_description = safe_enhance_company_description
        
        logger.info("✅ Successfully replaced description_enhancer functions with business-focused versions")
    except Exception as e:
        logger.error(f"❌ Failed to patch description_enhancer: {e}")
    
    logger.info("Complete classification hierarchy fixes successfully applied")
