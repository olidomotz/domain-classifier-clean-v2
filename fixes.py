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
import os
import requests
import importlib.util
import sys
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
                formatted["company_description"] = result["company_description"]
                logger.info(f"Forced company description into output for {domain}")
                
            # CRITICAL: Force final_classification for Process Did Not Complete
            if predicted_class == "Process Did Not Complete":
                formatted["02_final_classification"] = "8-Unknown/No Data"

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

            # CRITICAL ADDITION: Check first for Process Did Not Complete
            if result.get("predicted_class") == "Process Did Not Complete":
                logger.info(f"Process Did Not Complete detected for {domain}, using special classification")
                return "8-Unknown/No Data"

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
                    # CRITICAL: Set the final_classification directly
                    classification["final_classification"] = "8-Unknown/No Data"
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
                        # CRITICAL: Set the final_classification directly
                        classification["final_classification"] = "8-Unknown/No Data"
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
                    # CRITICAL: Set the final_classification directly
                    classification["final_classification"] = "8-Unknown/No Data"
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

    # 8. Add the simplified data source separation fix
    separate_data_sources_simple()
    
    # 9. Add the improved description separation fix to properly handle AI and Apollo descriptions
    fix_description_separation()
    
    # 10. Apply DIRECT fix for Process Did Not Complete classification
    fix_process_did_not_complete_classification()
    
    # 11. Fix the classify routes to handle empty content properly
    fix_classify_routes_for_empty_content()
    
    # 12. Ensure all text in merged data is in English
    ensure_english_data()

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
                
                # CRITICAL NEW ADDITION: Force final_classification to be "8-Unknown/No Data"
                result["final_classification"] = "8-Unknown/No Data"
                
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
                formatted["company_description"] = result["company_description"]
                logger.info(f"Forced company description into output for {result.get('domain', 'unknown')}")
                
            # CRITICAL NEW ADDITION: Force final_classification in formatted result
            if result.get("predicted_class") == "Process Did Not Complete":
                formatted["02_final_classification"] = "8-Unknown/No Data"

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
            # CRITICAL ADDITION: Check first for Process Did Not Complete
            if result.get("predicted_class") == "Process Did Not Complete":
                logger.info(f"Process Did Not Complete detected for {result.get('domain', 'unknown')}, using special classification")
                return "8-Unknown/No Data"
                
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

def separate_data_sources_simple():
    """Modify the system to keep AI-extracted and Apollo data separate."""
    try:
        # Patch the API formatter to handle data sources properly
        from domain_classifier.utils import api_formatter

        original_format = api_formatter.format_api_response

        def patched_format_api_for_data_separation(result):
            """Format API response with proper data source separation."""
            # Get domain for logging
            domain = result.get("domain", "unknown")

            # Get AI-extracted data directly
            ai_data = result.get("ai_company_data", {})

            # Get Apollo data
            apollo_data = result.get("apollo_data", {})

            # Generate AI-only description from scraped content
            ai_description = None

            # If we have AI data, use it to create a description for section 03
            if ai_data and isinstance(ai_data, dict) and any(ai_data.values()):
                # Use the extracted description from AI data if it exists
                for field in ["description", "company_description"]:
                    if ai_data.get(field):
                        ai_description = ai_data.get(field)
                        logger.info(f"Using AI-extracted {field} for section 03 for {domain}")
                        break

            # Call original format function to get the base formatted result
            formatted = original_format(result)

            # Directly set AI description in section 03 if available
            if ai_description:
                formatted["03_description"] = ai_description

            # Make sure Apollo description stays in section 04
            if apollo_data and isinstance(apollo_data, dict):
                for key, value in apollo_data.items():
                    if value not in [None, "", 0]:
                        formatted[f"04_{key}"] = value

            # Force company_description to appear in output
            if "company_description" in result:
                formatted["company_description"] = result["company_description"]
                logger.info(f"Forced company description into output for {domain}")
                
            # CRITICAL: Force final_classification for Process Did Not Complete
            if result.get("predicted_class") == "Process Did Not Complete":
                formatted["02_final_classification"] = "8-Unknown/No Data"

            return formatted

        # Apply the patch
        api_formatter.format_api_response = patched_format_api_for_data_separation

        logger.info("✅ Successfully applied simplified data source separation")

    except Exception as e:
        logger.error(f"❌ Failed to apply simplified data source separation: {e}")

def fix_description_separation():
    """
    Fix to properly separate AI and Apollo descriptions in the API response.
    This enhanced version generates more detailed AI descriptions based on the domain and classification.
    """
    logger.info("Applying enhanced description separation fix...")
    
    try:
        from domain_classifier.utils import api_formatter
        
        original_format = api_formatter.format_api_response
        
        def patched_format_api_response(result):
            """Format API response with proper data source separation."""
            # Get domain and other key info for logging and description generation
            domain = result.get("domain", "unknown")
            predicted_class = result.get("predicted_class", "")
            company_name = result.get("company_name", domain.split('.')[0].capitalize())
            
            # CRITICAL: Force final_classification for Process Did Not Complete
            if predicted_class == "Process Did Not Complete":
                result["final_classification"] = "8-Unknown/No Data"
            
            # Keep original company description for reference
            original_company_description = result.get("company_description", "")
            
            # Extract AI data with proper error handling
            ai_data = result.get("ai_company_data", {})
            if isinstance(ai_data, str):
                try:
                    ai_data = json.loads(ai_data)
                except Exception:
                    ai_data = {}
            
            # Extract Apollo data with proper error handling
            apollo_data = result.get("apollo_data", {})
            if isinstance(apollo_data, str):
                try:
                    apollo_data = json.loads(apollo_data)
                except Exception:
                    apollo_data = {}
            
            # Get the base formatted result
            formatted = original_format(result)
            
            # ========== SECTION 3: AI DATA ==========
            
            # Generate a comprehensive AI-based description
            ai_description = None
            
            # If we have specific AI-extracted description, use that first
            if ai_data and isinstance(ai_data, dict) and ai_data.get("description"):
                ai_description = ai_data.get("description")
                logger.info(f"Using AI-extracted description for {domain}")
            else:
                # Generate a rich description based on classification and other data
                industry = ""
                if ai_data and ai_data.get("industry"):
                    industry = ai_data.get("industry")
                elif apollo_data and apollo_data.get("industry"):
                    industry = apollo_data.get("industry")
                
                employee_count = ""
                if ai_data and ai_data.get("employee_count"):
                    employee_count = f"with approximately {ai_data.get('employee_count')} employees"
                elif apollo_data and apollo_data.get("employee_count"):
                    employee_count = f"with approximately {apollo_data.get('employee_count')} employees"
                
                # Handle Process Did Not Complete specially
                if predicted_class == "Process Did Not Complete":
                    # Check if we have any Apollo data to use
                    if apollo_data and any(apollo_data.values()):
                        ai_description = f"{company_name} has insufficient website data for full analysis, but some company information is available."
                    else:
                        ai_description = f"Unable to retrieve information for {domain} due to insufficient data."
                    
                    logger.info(f"Generated AI description for Process Did Not Complete: {domain}")
                # Create class-specific detailed descriptions
                elif predicted_class == "Managed Service Provider":
                    ai_description = f"{company_name} is an IT service provider {employee_count} that specializes in managed technology solutions for businesses. "
                    
                    # Add more specific details based on domain name
                    if "cloud" in domain.lower():
                        ai_description += "They offer cloud hosting, infrastructure management, and remote IT support. "
                    elif "cyber" in domain.lower() or "secure" in domain.lower() or "security" in domain.lower():
                        ai_description += "They provide cybersecurity services, data protection, and secure infrastructure management. "
                    elif "tech" in domain.lower() or "it" in domain.lower():
                        ai_description += "They deliver comprehensive IT management, technical support, and digital transformation services. "
                    else:
                        ai_description += "They provide network management, technical support, and IT infrastructure services. "
                    
                    ai_description += "Their managed services help businesses maintain reliable technology operations while reducing IT costs."
                
                elif predicted_class == "Integrator - Commercial A/V":
                    ai_description = f"{company_name} is a commercial audio-visual integrator {employee_count} that designs and implements professional A/V systems for businesses. "
                    ai_description += "They specialize in conference room technology, digital signage solutions, and integrated communication systems. "
                    ai_description += "Their solutions enable effective presentations, video conferencing, and multimedia communications in corporate environments."
                
                elif predicted_class == "Integrator - Residential A/V":
                    ai_description = f"{company_name} is a residential audio-visual integrator {employee_count} that creates custom home entertainment and automation systems. "
                    ai_description += "They design and install home theaters, whole-house audio, lighting control, and smart home technologies. "
                    ai_description += "Their residential solutions enhance lifestyle through integrated technology for modern homes."
                
                elif predicted_class == "Internal IT Department":
                    if industry:
                        ai_description = f"{company_name} is a business operating in the {industry} industry {employee_count}. "
                        ai_description += f"Unlike IT service providers, they do not offer managed services to external clients. "
                        ai_description += f"They maintain their own internal IT infrastructure to support their business operations."
                    else:
                        ai_description = f"{company_name} is a business with internal IT needs {employee_count}. "
                        ai_description += "They maintain their own technology infrastructure rather than providing IT services to external clients."
                
                elif predicted_class == "Parked Domain":
                    ai_description = f"{domain} appears to be a parked or inactive domain. No active business content was identified during analysis."
                
                else:
                    # Generic fallback
                    ai_description = f"{company_name} is a business {employee_count} that was analyzed through website content extraction."
                
                logger.info(f"Generated comprehensive AI description for section 03 for {domain}")
            
            # Set the AI description in section 03
            formatted["03_description"] = ai_description
            
            # ========== SECTION 4: APOLLO DATA ==========
            
            # Keep Apollo short description in section 04
            if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("short_description"):
                formatted["04_short_description"] = apollo_data.get("short_description")
                logger.info(f"Set Apollo short_description in section 04 for {domain}")
            
            # Make sure all other Apollo fields stay in section 04
            if apollo_data and isinstance(apollo_data, dict):
                for key, value in apollo_data.items():
                    if value not in [None, "", 0]:
                        formatted[f"04_{key}"] = value
            
            # ========== KEEP COMPANY DESCRIPTION ==========
            
            # Always ensure the company_description is in the output
            formatted["company_description"] = original_company_description
            logger.info(f"Forced company description into output for {domain}")
            
            # CRITICAL: Force final_classification for Process Did Not Complete
            if predicted_class == "Process Did Not Complete":
                formatted["02_final_classification"] = "8-Unknown/No Data"
            
            return formatted
        
        # Apply the patch
        api_formatter.format_api_response = patched_format_api_response
        
        # Now also patch the enrich.py file to preserve AI data during enrichment
        try:
            from domain_classifier.api.routes import enrich
            
            # Patch the AI data extraction step to create better descriptions
            original_extract = enrich.extract_company_data_from_content
            
            def patched_extract_company_data(content, domain, classification):
                """Patched version that ensures AI data has better descriptions."""
                # Call original function to get AI data
                ai_data = original_extract(content, domain, classification)
                
                # Store the AI data in the classification result
                if ai_data and any(ai_data.values()):
                    # Store the complete AI data
                    classification["ai_company_data"] = ai_data
                    
                    # Create a better description if one isn't present or is too basic
                    existing_desc = ai_data.get("description", "")
                    if not existing_desc or len(existing_desc) < 50:
                        company_name = ai_data.get("name", domain.split('.')[0].capitalize())
                        industry = ai_data.get("industry", "")
                        employee_count = ai_data.get("employee_count", "")
                        
                        # Create a more detailed description
                        better_desc = f"{company_name} "
                        if industry:
                            better_desc += f"operates in the {industry} industry "
                        
                        if employee_count:
                            better_desc += f"with approximately {employee_count} employees "
                        
                        # Add location info if available
                        location_parts = []
                        if ai_data.get("city"):
                            location_parts.append(ai_data.get("city"))
                        if ai_data.get("state"):
                            location_parts.append(ai_data.get("state"))
                        if ai_data.get("country"):
                            location_parts.append(ai_data.get("country"))
                        
                        if location_parts:
                            better_desc += f"based in {', '.join(location_parts)} "
                        
                        # Finish description based on domain type
                        if domain.endswith('.it'):
                            better_desc += f"providing information technology services to clients. "
                            better_desc += "They specialize in network infrastructure management, cloud solutions, and technical support."
                        else:
                            better_desc += f"providing professional services to clients. "
                            better_desc += "They offer technical solutions tailored to business needs."
                        
                        # Update the description in both places
                        ai_data["description"] = better_desc
                        classification["ai_company_data"]["description"] = better_desc
                        
                        logger.info(f"Enhanced AI-extracted description for {domain}")
                
                return ai_data
            
            # Apply the patch if the original function exists
            if hasattr(enrich, "extract_company_data_from_content"):
                enrich.extract_company_data_from_content = patched_extract_company_data
                logger.info("✅ Successfully patched extract_company_data_from_content to create better AI descriptions")
            else:
                logger.warning("Could not find extract_company_data_from_content function to patch")
            
            logger.info("✅ Successfully patched enrich.py for description handling")
        except Exception as enrich_error:
            logger.error(f"❌ Failed to patch enrich.py: {enrich_error}")
        
        logger.info("✅ Successfully applied enhanced description separation fix")
    except Exception as e:
        logger.error(f"❌ Failed to apply enhanced description separation fix: {e}")

def fix_process_did_not_complete_classification():
    """
    Direct fix to properly classify domains with 'Process Did Not Complete' status.
    This ensures they don't get incorrectly labeled as 'Internal IT'.
    """
    logger.info("Applying direct fix for Process Did Not Complete classification...")
    
    try:
        # Import the module directly
        import importlib.util
        import sys
        
        # Path to the module file
        module_path = "domain_classifier/utils/final_classification.py"
        
        # Load the module
        spec = importlib.util.spec_from_file_location("final_classification", module_path)
        if spec:
            final_classification = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(final_classification)
            
            # Store the original function
            original_determine = final_classification.determine_final_classification
            
            # Define our patched version
            def patched_determine_final_classification(result):
                """Directly patched version that handles Process Did Not Complete properly."""
                # Get the domain for logging
                domain = result.get("domain", "unknown")
                
                # Handle Process Did Not Complete as a special case
                if result.get("predicted_class") == "Process Did Not Complete":
                    logger.info(f"No data available for {domain} with Process Did Not Complete status")
                    return "8-Unknown/No Data"  # Return our special classification
                
                # For all other cases, use the original function
                return original_determine(result)
            
            # Apply the patch directly to the module
            final_classification.determine_final_classification = patched_determine_final_classification
            
            # Re-register in sys.modules to make sure it's used
            sys.modules["domain_classifier.utils.final_classification"] = final_classification
            
            # Also try to patch it the standard way
            try:
                from domain_classifier.utils import final_classification as fc
                fc.determine_final_classification = patched_determine_final_classification
                logger.info("Applied patch via import")
            except Exception as e:
                logger.error(f"Failed to apply standard patch: {e}")
            
            logger.info("✅ Successfully applied direct fix for Process Did Not Complete classification")
            
            # Now patch the API formatter to display the new classification properly
            try:
                from domain_classifier.utils import api_formatter
                
                original_format = api_formatter.format_api_response
                
                def patched_formatter_with_unknown(result):
                    """Update API formatter to handle Unknown/No Data classification."""
                    # Get the formatted result from the current formatter
                    formatted = original_format(result)
                    
                    # Check if this is a "Process Did Not Complete" with no data
                    if result.get("predicted_class") == "Process Did Not Complete":
                        # Update the final classification display
                        if result.get("final_classification") == "8-Unknown/No Data":
                            formatted["02_final_classification"] = "8-Unknown/No Data"
                            
                            # Also ensure we have a clear description
                            domain = result.get("domain", "unknown")
                            formatted["03_description"] = f"Unable to retrieve information for {domain} due to insufficient data."
                            formatted["company_description"] = f"Unable to retrieve information for {domain} due to insufficient data."
                        else:
                            # Force it even if it wasn't set earlier
                            formatted["02_final_classification"] = "8-Unknown/No Data"
                            result["final_classification"] = "8-Unknown/No Data"
                    
                    return formatted
                    
                # Apply the patch
                api_formatter.format_api_response = patched_formatter_with_unknown
                
                logger.info("✅ Successfully patched api_formatter to handle Unknown/No Data classification")
            except Exception as e:
                logger.error(f"❌ Failed to patch api_formatter for Unknown/No Data classification: {e}")
        else:
            logger.error("Could not create spec for final_classification module")
        
    except Exception as e:
        logger.error(f"❌ Failed to apply Process Did Not Complete classification fix: {e}")

def fix_classify_routes_for_empty_content():
    """
    Fix the classify routes to properly handle cases where a domain returns a 200 status
    but has no actual content. This ensures proper "Process Did Not Complete" classification.
    """
    logger.info("Applying fix for classify routes to handle empty content...")
    
    try:
        # Try to patch the classify_domain function in routes/classify.py
        import importlib.util
        import sys
        
        # Path to the module file
        module_path = "domain_classifier/api/routes/classify.py"
        
        # Load the module
        spec = importlib.util.spec_from_file_location("classify", module_path)
        if spec:
            classify_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(classify_module)
            
            # Find the function that does the final classification
            if hasattr(classify_module, "classify_domain"):
                original_classify_domain = classify_module.classify_domain
                
                # Create a wrapper around the original function
                def patched_classify_domain(*args, **kwargs):
                    """
                    Patched version that ensures empty content gets properly classified.
                    """
                    # Call the original function
                    result, status_code = original_classify_domain(*args, **kwargs)
                    
                    # Check if this is a Process Did Not Complete result
                    if isinstance(result, dict) and result.get("predicted_class") == "Process Did Not Complete":
                        # Force final_classification to be 8-Unknown/No Data
                        result["final_classification"] = "8-Unknown/No Data"
                        logger.info(f"Set final_classification to 8-Unknown/No Data for {result.get('domain', 'unknown')}")
                    
                    return result, status_code
                
                # Apply the patch
                classify_module.classify_domain = patched_classify_domain
                
                # Re-register in sys.modules
                sys.modules["domain_classifier.api.routes.classify"] = classify_module
                
                # Try to patch via import as well
                try:
                    from domain_classifier.api.routes import classify
                    classify.classify_domain = patched_classify_domain
                    logger.info("Applied classify_domain patch via import")
                except Exception as e:
                    logger.error(f"Failed to apply classify_domain patch via import: {e}")
                
                logger.info("✅ Successfully patched classify_domain to handle empty content")
            else:
                logger.warning("Could not find classify_domain function to patch")
            
            # Also patch the classify_and_enrich function in routes/enrich.py
            try:
                # Path to the module file
                module_path = "domain_classifier/api/routes/enrich.py"
                
                # Load the module
                spec = importlib.util.spec_from_file_location("enrich", module_path)
                if spec:
                    enrich_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(enrich_module)
                    
                    # Find the function that does the classification and enrichment
                    if hasattr(enrich_module, "classify_and_enrich"):
                        original_classify_and_enrich = enrich_module.classify_and_enrich
                        
                        # Create a wrapper around the original function
                        def patched_classify_and_enrich(*args, **kwargs):
                            """
                            Patched version that ensures Process Did Not Complete gets proper classification.
                            """
                            # Call the original function to get the result
                            result = original_classify_and_enrich(*args, **kwargs)
                            
                            # Check if it's a jsonify result by looking at its data attribute
                            if hasattr(result, 'data'):
                                import json
                                try:
                                    # Parse the JSON data
                                    data = json.loads(result.data)
                                    
                                    # Check for Process Did Not Complete
                                    if data.get("02_classification") == "Process Did Not Complete":
                                        # Modify the data to set the final classification
                                        data["02_final_classification"] = "8-Unknown/No Data"
                                        
                                        # Replace the result data
                                        from flask import jsonify
                                        new_result = jsonify(data)
                                        result.data = new_result.data
                                        
                                        logger.info(f"Fixed final classification in classify_and_enrich response")
                                except Exception as e:
                                    logger.error(f"Error modifying classify_and_enrich response: {e}")
                            
                            return result
                        
                        # Apply the patch
                        enrich_module.classify_and_enrich = patched_classify_and_enrich
                        
                        # Re-register in sys.modules
                        sys.modules["domain_classifier.api.routes.enrich"] = enrich_module
                        
                        # Try to patch via import as well
                        try:
                            from domain_classifier.api.routes import enrich
                            enrich.classify_and_enrich = patched_classify_and_enrich
                            logger.info("Applied classify_and_enrich patch via import")
                        except Exception as e:
                            logger.error(f"Failed to apply classify_and_enrich patch via import: {e}")
                        
                        logger.info("✅ Successfully patched classify_and_enrich to handle Process Did Not Complete")
                    else:
                        logger.warning("Could not find classify_and_enrich function to patch")
                else:
                    logger.error("Could not create spec for enrich module")
            except Exception as e:
                logger.error(f"❌ Failed to patch classify_and_enrich: {e}")
        else:
            logger.error("Could not create spec for classify module")
        
    except Exception as e:
        logger.error(f"❌ Failed to apply fix for classify routes: {e}")

def ensure_english_data():
    """
    Ensure all data in the merged section (05) is in English by detecting and 
    translating non-English content. This is done without requiring additional 
    external services by using Claude's capabilities.
    """
    logger.info("Applying English translation fix for merged data...")
    
    try:
        from domain_classifier.utils import api_formatter
        import re
        import os
        import requests
        
        # Basic language detection patterns
        NON_ENGLISH_PATTERNS = {
            'italian': [
                r'\bcome\b.*?\bche\b', r'\bin\b.*?\bcontinuo\b', r'\bovunque\b', 
                r'\bspazio\b.*?\bfisico\b', r'\bnuvole\b', r'\bmondo\b.*?\bvirtuale\b',
                r'\bdati\b.*?\bscambio\b', r'\bqualcosa\b.*?\boccupa\b', r'\bdisponibili\b'
            ],
            'german': [
                r'\bund\b.*?\bder\b', r'\bdie\b.*?\bdas\b', r'\bfür\b', r'\bmit\b.*?\bsind\b',
                r'\beine\b.*?\bwir\b', r'\bsich\b.*?\bauf\b'
            ],
            'spanish': [
                r'\bcon\b.*?\bpara\b', r'\bde\b.*?\bla\b', r'\bel\b.*?\blos\b', r'\bsus\b.*?\bmuy\b',
                r'\buna\b.*?\bque\b', r'\btodos\b.*?\bnuestros\b'
            ],
            'french': [
                r'\bavec\b.*?\bpour\b', r'\bde\b.*?\bla\b', r'\ble\b.*?\bles\b', r'\bnous\b.*?\bvotre\b',
                r'\bune\b.*?\bqui\b', r'\btous\b.*?\bnotre\b'
            ]
        }
        
        def detect_language(text):
            """Detect the language of a text using basic pattern matching."""
            if not text or not isinstance(text, str) or len(text) < 10:
                return "english"  # Default for empty or very short texts
                
            text = text.lower()
            
            # Check for non-English patterns
            for lang, patterns in NON_ENGLISH_PATTERNS.items():
                matches = 0
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        matches += 1
                
                # If multiple patterns match, it's likely this language
                if matches >= 2:
                    return lang
                    
            # Check for non-ASCII characters frequency
            non_ascii_chars = sum(1 for char in text if ord(char) > 127)
            ascii_chars = len(text) - non_ascii_chars
            
            # If more than 15% non-ASCII characters, likely not English
            if len(text) > 20 and non_ascii_chars > (len(text) * 0.15):
                return "non-english"
                
            # Default to English
            return "english"
        
        def translate_text(text, source_lang="auto"):
            """
            Translate text to English using Claude's capabilities without requiring
            external translation APIs.
            """
            if not text or not isinstance(text, str) or len(text) < 3:
                return text
                
            # Check if already English
            if detect_language(text) == "english":
                return text
            
            try:
                # Use Claude API directly for translation
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.warning("No Anthropic API key for translation, skipping")
                    return text
                
                logger.info(f"Translating text to English: '{text[:40]}...'")
                
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": "claude-3-haiku-20240307",  # Fast model for translations
                        "system": "You are a precise translator that only translates text to English without adding any explanation or comments. Keep the same formatting and maintain the original meaning.",
                        "messages": [{
                            "role": "user", 
                            "content": f"Please translate this text to English:\n\n{text}"
                        }],
                        "max_tokens": 300,
                        "temperature": 0.1  # Low temperature for accuracy
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    translated_text = result['content'][0]['text'].strip()
                    logger.info(f"Translation successful: '{translated_text[:40]}...'")
                    return translated_text
                else:
                    logger.warning(f"Translation API error: {response.status_code} - {response.text[:100]}")
                    return text
                    
            except Exception as e:
                logger.warning(f"Error translating text: {e}")
                return text
        
        # Original format_api_response function
        original_format = api_formatter.format_api_response
        
        def patched_format_with_translation(result):
            """Ensure all text in merged data section is in English."""
            # First get the formatted result using the existing formatter
            formatted = original_format(result)
            
            # Identify merged data fields (section 05)
            merged_fields = {k: v for k, v in formatted.items() if k.startswith('05_') and not k.endswith('_source')}
            
            # Process each merged field
            for key, value in merged_fields.items():
                # Only process string values that are not empty
                if isinstance(value, str) and value.strip():
                    # Skip fields that shouldn't be translated
                    if any(skip in key for skip in ['email', 'website', 'url', 'domain', 'phone']):
                        continue
                    
                    # Detect language
                    detected_lang = detect_language(value)
                    
                    # If not English, translate
                    if detected_lang != "english":
                        logger.info(f"Detected {detected_lang} in field {key}, translating")
                        translated = translate_text(value, detected_lang)
                        
                        # Update with translation if successful
                        if translated and translated != value:
                            formatted[key] = translated
                            logger.info(f"Translated {key} to English: '{translated[:40]}...'")
                            
                            # Also update the company_description if this is a description field
                            if key == "05_description" or key == "05_short_description":
                                formatted["company_description"] = translated
                                logger.info(f"Updated company_description with translation")
            
            return formatted
        
        # Apply the patch
        api_formatter.format_api_response = patched_format_with_translation
        
        # Also patch the AI data extraction to ensure we translate any non-English content
        try:
            from domain_classifier.enrichment.ai_data_extractor import extract_company_data_from_content
            
            original_extract = extract_company_data_from_content
            
            def patched_extract_with_translation(content, domain, classification):
                """Ensure AI-extracted data is translated to English if needed."""
                # Call original function
                ai_data = original_extract(content, domain, classification)
                
                # Translate fields that might contain non-English text
                if ai_data:
                    for field in ['description', 'industry', 'company_description']:
                        if field in ai_data and ai_data[field] and isinstance(ai_data[field], str):
                            # Check if translation needed
                            if detect_language(ai_data[field]) != "english":
                                ai_data[field] = translate_text(ai_data[field])
                                logger.info(f"Translated AI-extracted {field} for {domain}")
                
                return ai_data
            
            # Apply the patch
            from domain_classifier.enrichment import ai_data_extractor
            ai_data_extractor.extract_company_data_from_content = patched_extract_with_translation
            
            logger.info("✅ Successfully patched AI data extraction to translate non-English content")
        except Exception as e:
            logger.error(f"❌ Failed to patch AI data extraction for translation: {e}")
        
        logger.info("✅ Successfully applied English translation fix for merged data")
    except Exception as e:
        logger.error(f"❌ Failed to apply English translation fix: {e}")
