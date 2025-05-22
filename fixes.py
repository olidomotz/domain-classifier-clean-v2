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
    
    # 3. Fix result_processor to ensure company description is preserved
    try:
        from domain_classifier.storage import result_processor
        
        original_process_fresh = result_processor.process_fresh_result
        
        def patched_process_fresh(classification, domain, email=None, url=None):
            # Preserve any existing company_description
            original_description = classification.get("company_description", "")
            
            # Call original function
            result = original_process_fresh(classification, domain, email, url)
            
            # If original description existed but got lost, restore it
            if original_description and not result.get("company_description"):
                result["company_description"] = original_description
                logger.info(f"Restored original company description for {domain}")
            
            # Ensure Apollo data is marked correctly
            if result.get("apollo_data"):
                result["has_apollo_data"] = True
            
            # If classification is from domain pattern or website content, keep it
            if classification.get("detection_method") in [
                "domain_pattern_msp", 
                "llm_classification",
                "text_parsing", 
                "vector_similarity"
            ]:
                # Make sure classification wasn't changed
                if classification.get("predicted_class") != result.get("predicted_class"):
                    logger.warning(f"Restoring original classification for {domain} from {classification.get('detection_method')}")
                    result["predicted_class"] = classification.get("predicted_class")
                    result["detection_method"] = classification.get("detection_method")
            
            # Make one final check for IT Solutions domains
            if domain and ("it" in domain.lower() or "tech" in domain.lower()) and "solution" in domain.lower():
                if result.get("predicted_class") != "Managed Service Provider":
                    logger.warning(f"Final IT Solutions check: forcing domain {domain} to MSP")
                    result["predicted_class"] = "Managed Service Provider"
                    result["detection_method"] = "domain_pattern_msp"
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
        logger.info("✅ Successfully patched result_processor.process_fresh_result to preserve descriptions")
    except Exception as e:
        logger.error(f"❌ Failed to patch result_processor: {e}")
    
    # 4. Fix api_formatter to ensure company description is included in the output
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
            
            # Ensure description gets into 03_description
            formatted = original_format(result)
            
            # Force company description into output
            if "company_description" in result and "03_description" not in formatted:
                formatted["03_description"] = result["company_description"]
                logger.info(f"Forced company description into output for {domain}")
            
            # Ensure Apollo data fields are included
            if "apollo_data" in result and "04_aaa_section" in formatted:
                apollo_data = result["apollo_data"]
                if isinstance(apollo_data, dict):
                    # Make sure all Apollo fields are included
                    for key, value in apollo_data.items():
                        if value and f"04_{key}" not in formatted:
                            formatted[f"04_{key}"] = value
                            logger.info(f"Added Apollo field {key} to output")
            
            return formatted
        
        # Apply the patch
        api_formatter.format_api_response = patched_format_api
        logger.info("✅ Successfully patched api_formatter.format_api_response to include descriptions and Apollo fields")
    except Exception as e:
        logger.error(f"❌ Failed to patch api_formatter: {e}")
    
    # 5. Fix Apollo connector to include description field
    try:
        from domain_classifier.enrichment import apollo_connector
        
        # Save original method if we need to call it
        if hasattr(apollo_connector.ApolloConnector, "_format_company_data"):
            original_format = apollo_connector.ApolloConnector._format_company_data
            
            def patched_format_company_data(self, apollo_data):
                """Enhanced version that ensures description field is included."""
                # Call original method to get base formatted data
                formatted = original_format(self, apollo_data)
                
                # Ensure description is included
                if "description" not in formatted and apollo_data.get("description"):
                    formatted["description"] = apollo_data.get("description")
                    logger.info(f"Added description field to Apollo data for {apollo_data.get('name', 'unknown')}")
                
                # Add any other missing fields we might need
                for field in ["short_description", "long_description", "specialties", "keywords", "company_type"]:
                    if field not in formatted and apollo_data.get(field):
                        formatted[field] = apollo_data.get(field)
                
                return formatted
            
            # Apply the patch
            apollo_connector.ApolloConnector._format_company_data = patched_format_company_data
            logger.info("✅ Successfully patched Apollo connector to include description field")
        else:
            logger.warning("Could not find Apollo connector _format_company_data method to patch")
    except Exception as e:
        logger.error(f"❌ Failed to patch Apollo connector: {e}")
    
    # 6. Fix final_classification to ensure IT Solutions domains are MSPs
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
    
    logger.info("Complete classification hierarchy fixes successfully applied")
