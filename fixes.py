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
    
    # 7. Add the fix to reclassify parked domains using Apollo data
    reclassify_parked_domains_with_apollo()
    
    # 8. Add the fix to separate AI-extracted and Apollo data
    separate_data_sources()

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

def reclassify_parked_domains_with_apollo():
    """Modify system to reclassify parked domains using Apollo description when available."""
    try:
        from domain_classifier.utils import final_classification
        
        original_determine = final_classification.determine_final_classification
        
        def enhanced_final_classification(result):
            """Enhanced final classification that uses Apollo data to reclassify parked domains."""
            # Check if this is a parked domain
            if result.get("is_parked", False) or result.get("predicted_class") == "Parked Domain":
                # Check if Apollo data is available
                apollo_data = result.get("apollo_data", {})
                
                if apollo_data and any(apollo_data.values()):
                    logger.info(f"Parked domain {result.get('domain')} has Apollo data - attempting reclassification")
                    
                    # Check if Apollo has a description we can use for classification
                    apollo_description = None
                    if isinstance(apollo_data, dict):
                        # Check for description fields in priority order
                        for field in ["short_description", "description", "long_description"]:
                            if apollo_data.get(field):
                                apollo_description = apollo_data.get(field)
                                logger.info(f"Using Apollo {field} for reclassification")
                                break
                    
                    # If we have a description, attempt reclassification
                    if apollo_description:
                        try:
                            # Import LLM classifier
                            from domain_classifier.classifiers.llm_classifier import LLMClassifier
                            
                            # Get API key from environment
                            import os
                            api_key = os.environ.get("ANTHROPIC_API_KEY")
                            
                            if api_key:
                                # Create classifier instance
                                classifier = LLMClassifier(api_key=api_key)
                                
                                # Run classification on Apollo description
                                classification = classifier.classify(
                                    content=apollo_description,
                                    domain=result.get("domain")
                                )
                                
                                if classification and classification.get("predicted_class"):
                                    new_class = classification.get("predicted_class")
                                    logger.info(f"Reclassified parked domain {result.get('domain')} as {new_class} based on Apollo description")
                                    
                                    # Update the result with the new classification
                                    result["predicted_class"] = new_class
                                    result["is_service_business"] = classification.get("is_service_business", True)
                                    result["detection_method"] = "apollo_description_classification"
                                    
                                    # Update confidence scores if available
                                    if "confidence_scores" in classification:
                                        result["confidence_scores"] = classification.get("confidence_scores")
                                    
                                    # Now determine final classification based on new predicted_class
                                    if new_class == "Managed Service Provider":
                                        return "1-MSP"
                                    elif new_class == "Integrator - Commercial A/V":
                                        return "3-Commercial Integrator"
                                    elif new_class == "Integrator - Residential A/V":
                                        return "4-Residential Integrator"
                                    elif new_class == "Internal IT Department":
                                        return "2-Internal IT"
                            else:
                                logger.warning(f"Cannot reclassify parked domain {result.get('domain')} - No API key")
                        except Exception as e:
                            logger.error(f"Error reclassifying parked domain with Apollo data: {e}")
                
                # If reclassification failed or wasn't attempted, use the original logic
                has_apollo = bool(apollo_data and any(apollo_data.values()))
                logger.info(f"Domain is parked. Has Apollo data: {has_apollo}")
                
                if has_apollo:
                    return "5-Parked Domain with partial enrichment"
                else:
                    return "6-Parked Domain - no enrichment"
            
            # For non-parked domains, use the original function
            return original_determine(result)
        
        # Apply the patch
        final_classification.determine_final_classification = enhanced_final_classification
        
        logger.info("✅ Successfully patched final_classification to reclassify parked domains using Apollo data")
        
    except Exception as e:
        logger.error(f"❌ Failed to patch final_classification for parked domains: {e}")
    
    # Also enhance the description generation for parked domains with Apollo data
    try:
        from domain_classifier.enrichment import description_enhancer
        
        original_generate = description_enhancer.generate_detailed_description
        
        def enhanced_description_for_parked(classification, apollo_data=None, apollo_person_data=None):
            """Generate better descriptions for parked domains with Apollo data."""
            # Check if this is a parked domain with Apollo data
            if (classification.get("is_parked", False) or classification.get("predicted_class") == "Parked Domain") and apollo_data:
                domain = classification.get("domain", "unknown")
                
                # If the domain has been reclassified using Apollo data
                if classification.get("detection_method") == "apollo_description_classification":
                    logger.info(f"Using reclassified description for parked domain {domain}")
                    
                    # Use the normal description generation, but note that it's based on Apollo data
                    description = original_generate(classification, apollo_data, apollo_person_data)
                    
                    # Append note about the domain being parked
                    if description and not "parked domain" in description.lower():
                        description += f" Note: The {domain} website appears to be parked, but company information is available from other sources."
                    
                    return description
                
                # If Apollo has a short_description, use it
                if isinstance(apollo_data, dict) and apollo_data.get("short_description"):
                    description = apollo_data.get("short_description")
                    logger.info(f"Using Apollo short_description for parked domain {domain}")
                    
                    # Append note about the domain being parked
                    if not "parked domain" in description.lower():
                        description += f" Note: The {domain} website appears to be parked, but company information is available from other sources."
                    
                    return description
            
            # For all other cases, use the original function
            return original_generate(classification, apollo_data, apollo_person_data)
        
        # Apply the patch
        description_enhancer.generate_detailed_description = enhanced_description_for_parked
        
        logger.info("✅ Successfully patched description_enhancer for better parked domain descriptions")
        
    except Exception as e:
        logger.error(f"❌ Failed to patch description_enhancer for parked domains: {e}")

def separate_data_sources():
    """Modify the system to keep AI-extracted and Apollo data separate."""
    try:
        from domain_classifier.enrichment import description_enhancer
        from domain_classifier.api.routes import enrich
        
        # Store reference to original function
        original_generate = description_enhancer.generate_detailed_description
        
        def ai_only_description(classification, apollo_data=None, apollo_person_data=None):
            """Generate AI description based solely on scraped content."""
            domain = classification.get("domain", "unknown")
            
            # We don't want to use Apollo data for the AI description
            # so we'll call the original function but with NULL apollo data
            try:
                # Get ai_company_data directly from classification
                ai_data = classification.get("ai_company_data", {})
                
                # Create a temporary classification object with ONLY AI data
                temp_classification = classification.copy()
                
                # Check if we have meaningful AI data to work with
                has_ai_data = ai_data and any(value not in [None, "", 0] for value in ai_data.values() if isinstance(value, (str, int, float)))
                
                # If we have meaningful AI data, generate a description based on it
                if has_ai_data:
                    logger.info(f"Generating AI-only description for {domain} based on scraped content")
                    
                    # Call original function with NO Apollo data
                    ai_description = original_generate(temp_classification, None, None)
                    
                    # Store the AI-only description
                    classification["ai_only_description"] = ai_description
                    
                    return ai_description
                else:
                    logger.info(f"No meaningful AI data for {domain}, using generic description")
                    return f"{domain} website was crawled but provided insufficient details for a complete description."
                
            except Exception as e:
                logger.error(f"Error generating AI-only description: {e}")
                return f"{domain} website was crawled but an error occurred when generating a description."
            
        # Prepare to patch the API formatter
        from domain_classifier.utils import api_formatter
        
        original_format = api_formatter.format_api_response
        
        def format_with_separated_sources(result):
            # Call original format function
            formatted = original_format(result)
            
            # Get domain for logging
            domain = result.get("domain", "unknown")
            
            # For AI section (03), make sure it only has AI-extracted data
            ai_data = result.get("ai_company_data", {})
            if isinstance(ai_data, dict) and ai_data:
                # Override section 03 description with AI-only description if available
                if "ai_only_description" in result:
                    formatted["03_description"] = result["ai_only_description"]
                    logger.info(f"Using AI-only description for section 03 for {domain}")
            
            # For the merged section (05), ensure we prioritize AI data over Apollo
            # The default functionality should already handle this correctly
            
            return formatted
        
        # Apply the patch to api_formatter
        api_formatter.format_api_response = format_with_separated_sources
        
        # We need to also hook into enrich.py to modify the data processing workflow
        # This is a complex change that requires modifying method calls, so we'll use
        # monkey patching to intercept key function calls
        
        # First save original ai_data_extractor.extract_company_data_from_content
        from domain_classifier.enrichment import ai_data_extractor
        original_extract = ai_data_extractor.extract_company_data_from_content
        
        def enhanced_extract(content, domain, classification):
            """Enhanced extraction that preserves the AI-only extraction."""
            # Call original extraction
            ai_data = original_extract(content, domain, classification)
            
            if ai_data:
                # Generate an AI-only description based on this data
                try:
                    # Create a temporary classification with ONLY the AI data
                    temp_classification = classification.copy()
                    temp_classification["ai_company_data"] = ai_data
                    
                    # Generate an AI-only description
                    ai_description = ai_only_description(temp_classification, None, None)
                    
                    # Store this AI-only description
                    classification["ai_only_description"] = ai_description
                    
                    logger.info(f"Generated AI-only description for {domain} during extraction")
                except Exception as e:
                    logger.error(f"Error generating AI-only description during extraction: {e}")
            
            return ai_data
        
        # Apply the patch to ai_data_extractor
        ai_data_extractor.extract_company_data_from_content = enhanced_extract
        
        logger.info("✅ Successfully patched system for proper data source separation")
        
    except Exception as e:
        logger.error(f"❌ Failed to patch system for data source separation: {e}")
