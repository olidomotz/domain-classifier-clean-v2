"""Utility functions for determining final classification."""
import logging
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def determine_final_classification(result: Dict[str, Any]) -> str:
    """
    Determine the final classification based on the classification result,
    with better handling of Apollo industry data.
    
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
    
    # Check for DNS resolution errors first
    if result.get("error_type") == "dns_error" or (isinstance(result.get("explanation", ""), str) and "DNS" in result.get("explanation", "")):
        logger.info(f"Classifying as No Website available due to error_type={result.get('error_type')}")
        return "7-No Website available"
        
    # Check for parked domains
    if result.get("is_parked", False) or result.get("predicted_class") == "Parked Domain":
        # Check if Apollo data is available
        has_apollo = bool(result.get("apollo_data") and any(result["apollo_data"].values()))
        logger.info(f"Domain is parked. Has Apollo data: {has_apollo}")
        if has_apollo:
            return "5-Parked Domain with partial enrichment"
        else:
            return "6-Parked Domain - no enrichment"
    
    # Check for service business types
    predicted_class = result.get("predicted_class", "")
    logger.info(f"Checking service business type: {predicted_class}")
    
    # First pass - direct class mapping
    if predicted_class == "Managed Service Provider":
        return "1-MSP"
    elif predicted_class == "Internal IT Department":
        # Check Apollo data for potential reclassification
        apollo_data = result.get("apollo_data", {})
        
        # Convert apollo_data to dictionary if it's a string
        if isinstance(apollo_data, str):
            try:
                import json
                apollo_data = json.loads(apollo_data)
            except:
                apollo_data = {}
        
        # Look for security-related industries
        apollo_industry = ""
        if apollo_data and isinstance(apollo_data, dict):
            apollo_industry = apollo_data.get('industry', '').lower()
            
            # Check for cybersecurity indicators
            security_indicators = [
                'security', 'cyber', 'network security', 'information security',
                'computer security', 'it security', 'cybersecurity', 
                'cyber security', 'computer & network security'
            ]
            
            if any(indicator in apollo_industry for indicator in security_indicators):
                logger.info(f"Reclassifying to MSP based on security industry: {apollo_industry}")
                return "1-MSP"  # Cybersecurity companies should be MSPs
        
        return "2-Internal IT"
    elif predicted_class == "Integrator - Commercial A/V":
        return "3-Commercial Integrator"
    elif predicted_class == "Integrator - Residential A/V":
        return "4-Residential Integrator"
    
    # Second pass - check Apollo data for Unknown or empty classification
    apollo_data = result.get("apollo_data", {})
    
    # Convert apollo_data to dictionary if it's a string
    if isinstance(apollo_data, str):
        try:
            import json
            apollo_data = json.loads(apollo_data)
        except:
            apollo_data = {}
    
    # Look for industry information
    if apollo_data and isinstance(apollo_data, dict):
        apollo_industry = apollo_data.get('industry', '').lower()
        
        # Handle various industries
        if apollo_industry:
            logger.info(f"Checking Apollo industry for classification: {apollo_industry}")
            
            # Check for IT service providers and MSPs
            msp_indicators = [
                'it services', 'information technology', 'managed services',
                'cloud services', 'it consulting', 'computer services', 
                'technology services', 'network services', 'technical services'
            ]
            
            security_indicators = [
                'security', 'cyber', 'network security', 'information security',
                'computer security', 'it security', 'cybersecurity', 
                'cyber security', 'computer & network security'
            ]
            
            av_indicators = [
                'audio visual', 'audio/visual', 'av', 'audiovisual',
                'multimedia', 'home automation', 'smart home', 'home technology',
                'commercial integration', 'systems integration', 'technology integration'
            ]
            
            # Check for security-related industries
            if any(indicator in apollo_industry for indicator in security_indicators):
                logger.info(f"Classifying as MSP based on security industry: {apollo_industry}")
                return "1-MSP"  # Cybersecurity companies should be MSPs
                
            # Check for MSP-related industries
            elif any(indicator in apollo_industry for indicator in msp_indicators):
                logger.info(f"Classifying as MSP based on IT industry: {apollo_industry}")
                return "1-MSP"
                
            # Check for AV-related industries
            elif any(indicator in apollo_industry for indicator in av_indicators):
                # Disambiguate between commercial and residential
                if 'commercial' in apollo_industry or 'business' in apollo_industry:
                    logger.info(f"Classifying as Commercial Integrator based on industry: {apollo_industry}")
                    return "3-Commercial Integrator"
                elif 'residential' in apollo_industry or 'home' in apollo_industry:
                    logger.info(f"Classifying as Residential Integrator based on industry: {apollo_industry}")
                    return "4-Residential Integrator"
                else:
                    # Default to commercial if unclear
                    logger.info(f"Classifying as Commercial Integrator (default) based on AV industry: {apollo_industry}")
                    return "3-Commercial Integrator"
    
    # Default for unknown or error cases - still fallback to Internal IT
    logger.info(f"No specific classification matched, defaulting to 2-Internal IT")
    return "2-Internal IT"  # Default to Internal IT if we can't determine
