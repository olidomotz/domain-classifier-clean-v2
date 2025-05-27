"""Fixed decision_tree.py to use LLM for all cases."""

import logging
import re
from typing import Dict, Any, Optional, Tuple, List

# Set up logging
logger = logging.getLogger(__name__)

def is_parked_domain(content: str, domain: str = None) -> bool:
    """
    Fixed parked domain detection with better balance between detection and false positives.
    Args:
        content: The website content
        domain: Optional domain name for additional checks
    Returns:
        bool: True if the domain is parked/inactive
    """
    if not content:
        logger.info("Domain has no content at all, considering as parked")
        return True
        
    content_lower = content.lower()
    
    # 1. Check for DEFINITIVE parking indicators (highest confidence)
    definitive_indicators = [
        "domain is for sale", "buy this domain", "domain parking",
        "this domain is for sale", "parked by", "domain broker",
        "inquire about this domain", "this domain is available for purchase",
        "bid on this domain"
    ]
    
    for indicator in definitive_indicators:
        if indicator in content_lower:
            logger.info(f"Domain contains definitive parking indicator: '{indicator}'")
            return True
    
    # 2. Check for combinations of parking-related indicators
    parking_phrases = [
        "domain may be for sale", "domain for sale", "domain name parking",
        "purchase this domain", "domain has expired", "domain available"
    ]
    
    hosting_mentions = [
        "godaddy", "namecheap", "domain.com", "namesilo", "porkbun",
        "domain registration", "web hosting service", "hosting provider",
        "register this domain", "parkingcrew", "sedo", "bodis", "parked.com"
    ]
    
    parking_count = sum(1 for phrase in parking_phrases if phrase in content_lower)
    hosting_count = sum(1 for phrase in hosting_mentions if phrase in content_lower)
    
    # Require stronger evidence - multiple indicators or specific combinations
    if parking_count >= 2 or (parking_count >= 1 and hosting_count >= 1):
        logger.info(f"Domain has multiple parking indicators: {parking_count} parking phrases, {hosting_count} hosting mentions")
        return True
    
    # 3. Special case for GoDaddy proxy errors - but with stricter requirements
    if "proxy error" in content_lower and "godaddy" in content_lower and len(content) < 400:
        logger.info("Found GoDaddy proxy error specifically, likely parked")
        return True
    
    # 4. For very minimal content, check for specific patterns but require more evidence
    if len(content.strip()) < 200:
        technical_issues = [
            "domain not configured", "website coming soon", "under construction",
            "site temporarily unavailable", "default web page"
        ]
        
        tech_count = sum(1 for issue in technical_issues if issue in content_lower)
        
        # For minimal content, require at least 2 technical indicators
        if tech_count >= 2:
            logger.info(f"Minimal content with multiple technical issues ({tech_count}), likely parked")
            return True
            
        # Check for few unique words - a sign of placeholder content
        words = re.findall(r'\b\w+\b', content_lower)
        unique_words = set(words)
        
        if len(words) < 15 or len(unique_words) < 10:
            # ADDITIONAL CHECK - make sure it doesn't look like a legitimate minimal page
            # Don't classify as parked if it has real business terms
            business_terms = ["service", "contact", "about", "product", "company", "solution"]
            has_business_content = any(term in content_lower for term in business_terms)
            
            if not has_business_content:
                logger.info(f"Very minimal content with few unique words ({len(unique_words)}), likely parked")
                return True
    
    # Default - not parked
    return False

def check_special_domain_cases(domain: str, text_content: str) -> Optional[Dict[str, Any]]:
    """
    Check for special domain cases that need custom handling.
    
    CRITICAL CHANGE: Instead of returning a complete classification,
    return enhancement information that can be applied after LLM classification.

    Args:
        domain: The domain name
        text_content: The website content

    Returns:
        Optional[Dict[str, Any]]: Enhancement info if special case, None otherwise
    """
    domain_lower = domain.lower()
    
    # Check for cybersafe.sg specifically
    if domain_lower == "cybersafe.sg":
        logger.warning(f"Special handling for known cybersecurity domain: {domain}")
        return {
            "special_domain": True,
            "suggested_class": "Managed Service Provider",
            "industry": "computer & network security",
            "description_hint": "cybersecurity services provider",
            "confidence_boost": 0.2,  # How much to boost confidence if LLM agrees
            "detection_method": "special_domain_knowledge"
        }
        
    # Check for special domains with known classifications
    # HostiFi - always MSP
    if "hostifi" in domain_lower:
        logger.info(f"Special handling for known MSP domain: {domain}")
        return {
            "special_domain": True,
            "suggested_class": "Managed Service Provider",
            "industry": "cloud services",
            "description_hint": "cloud hosting platform for network controllers",
            "confidence_boost": 0.15,
            "detection_method": "special_domain_knowledge"
        }
        
    # Special handling for ciao.dk (known problematic vacation rental site)
    if domain_lower == "ciao.dk":
        logger.warning(f"Special handling for known vacation rental domain: {domain}")
        return {
            "special_domain": True,
            "suggested_class": "Internal IT Department",
            "industry": "travel and tourism",
            "description_hint": "vacation rental and travel booking",
            "confidence_boost": 0.15,
            "detection_method": "special_domain_knowledge"
        }
        
    # Check for IT Solutions patterns in domain name
    if ("it" in domain_lower or "tech" in domain_lower) and "solution" in domain_lower:
        logger.info(f"Domain {domain} contains IT Solutions pattern, likely an MSP")
        return {
            "special_domain": True,
            "suggested_class": "Managed Service Provider",
            "industry": "information technology",
            "description_hint": "IT solutions and managed services provider",
            "confidence_boost": 0.2,
            "detection_method": "domain_pattern_recognition"
        }
        
    # Check for managed services in domain name
    if "managed" in domain_lower and ("service" in domain_lower or "it" in domain_lower):
        logger.info(f"Domain {domain} contains managed services pattern, likely an MSP")
        return {
            "special_domain": True,
            "suggested_class": "Managed Service Provider",
            "industry": "managed IT services",
            "description_hint": "managed IT service provider",
            "confidence_boost": 0.2,
            "detection_method": "domain_pattern_recognition"
        }
        
    # Check for other vacation/travel-related domains
    vacation_terms = ["vacation", "holiday", "rental", "booking", "hotel", "travel", "accommodation", "ferie"]
    found_terms = [term for term in vacation_terms if term in domain_lower]
    
    if found_terms:
        logger.warning(f"Domain {domain} contains vacation/travel terms: {found_terms}")
        
        # Look for confirmation in the content if available
        travel_terms_in_content = False
        if text_content:
            content_lower = text_content.lower()
            travel_content_terms = ["booking", "accommodation", "stay", "vacation", "holiday", "rental"]
            travel_terms_in_content = any(term in content_lower for term in travel_content_terms)
            
        if travel_terms_in_content or len(found_terms) >= 2:
            logger.warning(f"Content confirms {domain} is likely a travel/vacation site")
            return {
                "special_domain": True,
                "suggested_class": "Internal IT Department",
                "industry": "travel and tourism",
                "description_hint": "vacation rental and travel booking",
                "confidence_boost": 0.15,
                "detection_method": "domain_pattern_recognition"
            }
            
    # Check for transportation/logistics companies
    transport_terms = ["trucking", "transport", "logistics", "shipping", "freight", "delivery", "carrier"]
    found_transport_terms = [term for term in transport_terms if term in domain_lower]
    
    if found_transport_terms:
        logger.warning(f"Domain {domain} contains transportation terms: {found_transport_terms}")
        
        # Look for confirmation in the content if available
        transport_terms_in_content = False
        if text_content:
            content_lower = text_content.lower()
            transport_content_terms = ["shipping", "logistics", "fleet", "trucking", "transportation", "delivery"]
            transport_terms_in_content = any(term in content_lower for term in transport_content_terms)
            
        if transport_terms_in_content or len(found_transport_terms) >= 2:
            logger.warning(f"Content confirms {domain} is likely a transportation/logistics company")
            return {
                "special_domain": True,
                "suggested_class": "Internal IT Department",
                "industry": "transportation and logistics",
                "description_hint": "transportation and logistics services",
                "confidence_boost": 0.15,
                "detection_method": "domain_pattern_recognition"
            }
            
    # Check for audio-visual or A/V in domain
    av_terms = ["av", "audio", "visual", "theater", "cinema", "sound"]
    found_av_terms = [term for term in av_terms if term in domain_lower]
    
    if found_av_terms and len(found_av_terms) >= 2:
        # Differentiate between commercial and residential
        if any(term in domain_lower for term in ["home", "residential", "smart"]):
            logger.info(f"Domain {domain} likely a Residential A/V Integrator based on domain terms")
            return {
                "special_domain": True,
                "suggested_class": "Integrator - Residential A/V",
                "industry": "home automation",
                "description_hint": "residential audio-visual integration and home automation",
                "confidence_boost": 0.15,
                "detection_method": "domain_pattern_recognition"
            }
        else:
            logger.info(f"Domain {domain} likely a Commercial A/V Integrator based on domain terms")
            return {
                "special_domain": True,
                "suggested_class": "Integrator - Commercial A/V",
                "industry": "audio visual integration",
                "description_hint": "commercial audio-visual integration services",
                "confidence_boost": 0.15,
                "detection_method": "domain_pattern_recognition"
            }
            
    return None

def create_process_did_not_complete_result(domain: str = None) -> Dict[str, Any]:
    """
    Create a standardized result for when processing couldn't complete.

    Args:
        domain: The domain name

    Returns:
        dict: Standardized process failure result
    """
    domain_name = domain or "Unknown domain"
    
    return {
        "processing_status": 0,
        "is_service_business": None,
        "predicted_class": "Process Did Not Complete",
        "internal_it_potential": 0,
        "confidence_scores": {
            "Managed Service Provider": 0,
            "Integrator - Commercial A/V": 0,
            "Integrator - Residential A/V": 0,
            "Internal IT Department": 0
        },
        "llm_explanation": f"Classification process for {domain_name} could not be completed. This may be due to connection issues, invalid domain, or other technical problems.",
        "company_description": f"Unable to determine what {domain_name} does due to processing failure.",
        "detection_method": "process_failed",
        "low_confidence": True,
        "max_confidence": 0.0
    }

def create_parked_domain_result(domain: str = None, crawler_type: str = None) -> Dict[str, Any]:
    """
    Create a standardized result for parked domains.

    Args:
        domain: The domain name
        crawler_type: The type of crawler that detected the parked domain

    Returns:
        dict: Standardized parked domain result
    """
    domain_name = domain or "This domain"
    
    # If crawler_type is None, set a default
    if crawler_type is None:
        crawler_type = "early_detection"
        
    return {
        "processing_status": 1,
        "is_service_business": None,
        "predicted_class": "Parked Domain",
        "internal_it_potential": 0,
        "confidence_scores": {
            "Managed Service Provider": 0,
            "Integrator - Commercial A/V": 0,
            "Integrator - Residential A/V": 0,
            "Internal IT Department": 0
        },
        "llm_explanation": f"{domain_name} appears to be a parked or inactive domain. No business-specific content was found to determine the company type. This may be a domain that is reserved but not yet in use, for sale, or simply under construction.",
        "company_description": f"{domain_name} appears to be a parked or inactive domain with no active business.",
        "detection_method": "parked_domain_detection",
        "low_confidence": True,
        "is_parked": True,
        "max_confidence": 0.0,
        "crawler_type": crawler_type  # Add crawler_type to the result
    }

def check_industry_context(content: str, apollo_data: Optional[Dict] = None, ai_data: Optional[Dict] = None) -> Tuple[bool, float]:
    """
    Check industry context from Apollo data and AI extraction to determine if likely service business.

    Args:
        content: Website content
        apollo_data: Apollo company data if available
        ai_data: AI extracted company data if available

    Returns:
        tuple: (is_service_business, confidence)
    """
    # Default to undetermined with low confidence
    is_service = True
    confidence = 0.5
    
    # Non-service industries list
    non_service_industries = [
        'manufacturing', 'forging', 'steel', 'metals', 'industrial',
        'construction', 'factory', 'production', 'fabrication',
        'retail', 'shop', 'store', 'ecommerce', 'e-commerce',
        'hospitality', 'hotel', 'restaurant', 'tourism', 'vacation',
        'healthcare', 'medical', 'hospital', 'clinic',
        'education', 'school', 'university', 'academic',
        'agriculture', 'farming', 'logistics', 'transportation',
        'shipping', 'international trade'
    ]
    
    # Check Apollo data first (most reliable)
    # Handle case where apollo_data might be a string (JSON)
    if apollo_data:
        # Convert string to dict if needed
        if isinstance(apollo_data, str):
            try:
                import json
                apollo_data = json.loads(apollo_data)
            except:
                # If parsing fails, treat as empty dict
                apollo_data = {}
                
        # Now safely access industry field
        industry = apollo_data.get('industry', '') if isinstance(apollo_data, dict) else ''
        
        if industry:
            industry = industry.lower()
            
            # Check for non-service industry matches
            for term in non_service_industries:
                if term in industry:
                    is_service = False
                    confidence = 0.8  # High confidence from Apollo data
                    logger.info(f"Apollo data indicates non-service industry: {industry}")
                    return is_service, confidence
    
    # Check AI-extracted data next
    if ai_data and ai_data.get('industry'):
        industry = ai_data.get('industry', '').lower()
        
        # Check for non-service industry matches
        for term in non_service_industries:
            if term in industry:
                is_service = False
                confidence = 0.7  # Good confidence from AI extraction
                logger.info(f"AI-extracted data indicates non-service industry: {industry}")
                return is_service, confidence
    
    # Check content for industry indicators if still undetermined and content is not None
    if content is not None:
        content_lower = content.lower()
        
        # Count manufacturing/industrial terms
        manufacturing_indicators = [
            'factory', 'manufacturing', 'production', 'industrial', 'machinery',
            'steel', 'metal', 'alloy', 'forging', 'fabrication', 'assembly',
            'plant', 'facility', 'raw materials', 'supply chain', 'quality control'
        ]
        
        manufacturing_count = sum(1 for term in manufacturing_indicators if term in content_lower)
        
        # If significant industrial terms are found
        if manufacturing_count >= 3:
            is_service = False
            confidence = 0.6  # Moderate confidence from content analysis
            logger.info(f"Content analysis found {manufacturing_count} manufacturing indicators")
            return is_service, confidence
    
    # Default return - couldn't determine with certainty
    return is_service, confidence
