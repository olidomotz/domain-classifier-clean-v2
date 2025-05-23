"""Utility functions for determining final classification."""

import logging
import re
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def determine_final_classification(result: Dict[str, Any]) -> str:
    """
    Determine the final classification based on the classification result.

    Args:
        result: The classification result

    Returns:
        str: The final classification code
    """
    logger.info(f"Determining final classification for domain: {result.get('domain')}")
    logger.info(f"Result keys: {list(result.keys())}")
    logger.info(f"Error type: {result.get('error_type')}")
    logger.info(f"Is parked: {result.get('is_parked')}")
    logger.info(f"Predicted class: {result.get('predicted_class')}")

    # Check for DNS resolution errors first (highest priority)
    if (result.get("error_type") == "dns_error" or
        result.get("is_dns_error") == True or
        (isinstance(result.get("explanation", ""), str) and "DNS" in result.get("explanation", ""))):
        
        logger.info(f"Classifying as No Website available due to error_type={result.get('error_type')}")
        return "7-No Website available"
        
    # Check for Process Did Not Complete (second priority)
    if result.get("predicted_class") == "Process Did Not Complete":
        # Check if we have Apollo data to potentially override this
        apollo_data = result.get("apollo_data", {})
        
        if apollo_data and isinstance(apollo_data, dict) and any(apollo_data.values()):
            # If already reclassified using Apollo data
            if result.get("detection_method") == "apollo_data_classification":
                logger.info(f"Using Apollo-based classification with Process Did Not Complete")
                
                # Determine based on the new predicted_class
                predicted_class = result.get("predicted_class", "")
                
                if predicted_class == "Managed Service Provider":
                    return "1-MSP"
                elif predicted_class == "Integrator - Commercial A/V":
                    return "3-Commercial Integrator"
                elif predicted_class == "Integrator - Residential A/V":
                    return "4-Residential Integrator"
                else:
                    return "2-Internal IT"
        
        # If we couldn't override with Apollo data
        logger.info(f"No data available for domain with Process Did Not Complete status")
        return "8-Unknown/No Data"

    # Check for parked domains (third priority)
    if result.get("is_parked", False) or result.get("predicted_class") == "Parked Domain":
        # Check if Apollo data is available
        has_apollo = bool(result.get("apollo_data") and any(result["apollo_data"].values()))
        
        logger.info(f"Domain is parked. Has Apollo data: {has_apollo}")
        
        # If Apollo data enabled reclassification
        if result.get("detection_method") == "apollo_data_classification":
            # Use the reclassified category
            predicted_class = result.get("predicted_class")
            logger.info(f"Using Apollo-based reclassification for parked domain: {predicted_class}")
            
            if predicted_class == "Managed Service Provider":
                return "1-MSP"
            elif predicted_class == "Integrator - Commercial A/V":
                return "3-Commercial Integrator"
            elif predicted_class == "Integrator - Residential A/V":
                return "4-Residential Integrator"
            else:
                return "2-Internal IT"
        
        # Standard parked domain handling
        if has_apollo:
            return "5-Parked Domain with partial enrichment"
        else:
            return "6-Parked Domain - no enrichment"
    
    # Check for domain-specific classifications (fourth priority)
    domain = result.get("domain", "")
    if domain:
        # Check for IT solutions pattern in domain name
        domain_lower = domain.lower()
        if ("it" in domain_lower or "tech" in domain_lower) and "solution" in domain_lower:
            logger.info(f"Domain contains IT solutions pattern, classifying as MSP regardless of content")
            return "1-MSP"
            
        # Check for managed services in domain name
        if "managed" in domain_lower and ("service" in domain_lower or "it" in domain_lower):
            logger.info(f"Domain contains managed services pattern, classifying as MSP regardless of content")
            return "1-MSP"

    # Check for service business types (fifth priority)
    predicted_class = result.get("predicted_class", "")
    
    logger.info(f"Checking service business type: {predicted_class}")
    
    if predicted_class == "Managed Service Provider":
        return "1-MSP"
    elif predicted_class == "Internal IT Department":
        return "2-Internal IT"
    elif predicted_class == "Integrator - Commercial A/V":
        return "3-Commercial Integrator"
    elif predicted_class == "Integrator - Residential A/V":
        return "4-Residential Integrator"
    
    # Default for unknown or error cases
    logger.info(f"No specific classification matched, defaulting to 2-Internal IT")
    return "2-Internal IT"  # Default to Internal IT if we can't determine
