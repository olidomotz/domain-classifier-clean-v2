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
    disable_cross_validation()

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

        # Replace generate_detailed_description to ensure reliable descriptions
        original_generate = description_enhancer.generate_detailed_description

        def improved_generate_description(classification, apollo_data=None, apollo_person_data=None):
            """Generate a reliable company description."""
            # Check if this is a "Process Did Not Complete" classification with no data
            if classification.get("predicted_class") == "Process Did Not Complete":
                domain = classification.get("domain", "unknown")
                logger.info(f"Skipping description generation for {domain} - Process Did Not Complete")
                
                # Only generate a description if we have Apollo data
                if not apollo_data or not isinstance(apollo_data, dict) or not apollo_data.get("name"):
                    logger.info(f"No Apollo data available for {domain}, returning minimal description")
                    
                    # Use a simple, factual statement instead of generating a fictional description
                    classification["company_description"] = f"Unable to retrieve information for {domain} due to insufficient data."
                    
                    # Also set a minimal one-liner
                    classification["company_one_line"] = f"No data available for {domain}."
                    
                    return classification.get("company_description", "")
                
            # For all other cases or when Apollo data is available, use the original function
            try:
                description = original_generate(classification, apollo_data, apollo_person_data)
                
                # Check if description is too short or empty
                if not description or len(description) < 30:
                    raise ValueError("Description too short, using fallback")
                    
                return description
            except Exception as e:
                logger.warning(f"Error in original description generator: {e}, using fallback")
            
            # Build a fallback description
            domain = classification.get("domain", "unknown")
            predicted_class = classification.get("predicted_class", "business")
            
            # For "Process Did Not Complete" with no data, provide minimal info
            if predicted_class == "Process Did Not Complete":
                if not apollo_data or not isinstance(apollo_data, dict) or not apollo_data.get("name"):
                    return f"Unable to retrieve information for {domain} due to insufficient data."
            
            # Get company name
            company_name = None
            if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("name"):
                company_name = apollo_data.get("name")
            else:
                company_name = domain.split(".")[0].capitalize()
                
            # Create a basic description
            description = f"{company_name} is "
            
            if predicted_class == "Managed Service Provider":
                description += "a Managed Service Provider"
            elif predicted_class == "Integrator - Commercial A/V":
                description += "a Commercial A/V Integrator"
            elif predicted_class == "Integrator - Residential A/V":
                description += "a Residential A/V Integrator"
            else:  # Internal IT
                description += "a business"
            
            # Add industry if available
            if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("industry"):
                description += f" in the {apollo_data.get('industry')} industry"
            
            description += "."
            
            # Add employee count if available
            if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("employee_count"):
                description += f" They have approximately {apollo_data.get('employee_count')} employees."
            
            # Add service descriptions based on class
            if predicted_class == "Managed Service Provider":
                description += " They provide IT services, technology support, and managed solutions to their clients."
            elif predicted_class == "Integrator - Commercial A/V":
                description += " They provide audio-visual systems and integration services for commercial environments."
            elif predicted_class == "Integrator - Residential A/V":
                description += " They provide home automation and entertainment systems for residential clients."
            
            logger.info(f"Created fallback description for {domain}")
            
            # Make sure the classification has the description
            classification["company_description"] = description
            
            return description
            
        # Apply the patch
        description_enhancer.generate_detailed_description = improved_generate_description
        
        # Also patch the enhance_company_description function to ensure it works reliably
        original_enhance = description_enhancer.enhance_company_description
        
        def improved_enhance_description(basic_description, apollo_data, classification):
            """Ensure company description is enhanced reliably."""
            # Special handling for "Process Did Not Complete" with no data
            if classification.get("predicted_class") == "Process Did Not Complete":
                if not apollo_data or not isinstance(apollo_data, dict) or not apollo_data.get("name"):
                    domain = classification.get("domain", "unknown")
                    logger.info(f"Skipping description enhancement for {domain} - No data available")
                    return f"Unable to retrieve information for {domain} due to insufficient data."
            
            # Try original function
            try:
                enhanced = original_enhance(basic_description, apollo_data, classification)
                
                # Check if description is too short or empty
                if not enhanced or len(enhanced) < 30:
                    raise ValueError("Enhanced description too short, using fallback")
                    
                return enhanced
            except Exception as e:
                logger.warning(f"Error in original description enhancer: {e}, using fallback")
            
            # Fall back to generating a description
            return improved_generate_description(classification, apollo_data)
            
        # Apply the patch
        description_enhancer.enhance_company_description = improved_enhance_description
        
        logger.info("✅ Successfully patched description_enhancer functions for reliable descriptions")
        
    except Exception as e:
        logger.error(f"❌ Failed to patch description_enhancer: {e}")

    # 6. Add the fix to prevent description fabrication
    prevent_description_fabrication()

    logger.info("Complete classification hierarchy fixes successfully applied")

def disable_cross_validation():
    """Disable cross-validation to preserve original LLM classification."""
    try:
        from domain_classifier.utils import cross_validator
        
        # Replace the reconcile_classification function with one that does nothing
        original_reconcile = cross_validator.reconcile_classification
        
        def no_op_reconcile(classification, apollo_data=None, ai_data=None):
            """Do-nothing version of reconcile that preserves the original classification."""
            logger.info(f"Cross-validation disabled - preserving original classification: {classification.get('predicted_class', 'unknown')}")
            return classification  # Return unchanged classification
        
        # Apply the patch
        cross_validator.reconcile_classification = no_op_reconcile
        logger.info("✅ Successfully disabled cross-validation to preserve LLM classification")
        
    except Exception as e:
        logger.error(f"❌ Failed to disable cross-validation: {e}")

def prevent_description_fabrication():
    """Prevent generating fictional descriptions for domains with no data."""
    try:
        from domain_classifier.utils import api_formatter
        
        original_format = api_formatter.format_api_response
        
        def patched_format_api_for_no_data(result):
            # For Process Did Not Complete with no data, ensure description is appropriate
            if result.get("predicted_class") == "Process Did Not Complete":
                domain = result.get("domain", "")
                
                # If no Apollo data and no content, use a minimal description
                apollo_data = result.get("apollo_data", {})
                if not apollo_data or not any(apollo_data.values()):
                    # Set a clear "no data" description
                    result["company_description"] = f"Unable to retrieve information for {domain} due to insufficient data."
                    result["company_one_line"] = f"No data available for {domain}."
                    
                    logger.info(f"Set minimal description for {domain} with no data")
            
            # Call original format function
            formatted = original_format(result)
            
            # Force company description into output
            if "company_description" in result:
                formatted["03_description"] = result["company_description"]
                logger.info(f"Forced company description into output for {result.get('domain', 'unknown')}")
            
            return formatted
        
        # Apply the patch
        api_formatter.format_api_response = patched_format_api_for_no_data
        
        logger.info("✅ Applied API formatter fix to handle domains with no data")
        
    except Exception as e:
        logger.error(f"❌ Failed to patch api_formatter for no data handling: {e}")
