"""Fixed decision_tree.py to use LLM for all cases."""

import logging
import re
from typing import Dict, Any, Optional, Tuple, List

# Set up logging
logger = logging.getLogger(__name__)

def is_parked_domain(content: str, domain: str = None) -> bool:
    """
    Enhanced detection of truly parked domains vs. just having minimal content.
    Better detection for GoDaddy parked domains and proxy errors.

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
    
    # 1. Direct explicit parking phrases (strongest indicators)
    explicit_parking_phrases = [
        "domain is for sale", "buy this domain", "purchasing this domain",
        "domain may be for sale", "this domain is for sale", "parked by",
        "domain parking", "this web page is parked", "domain for sale",
        "this website is for sale", "domain name parking",
        "purchase this domain", "domain has expired", "domain available",
        "domain not configured", "inquire about this domain", 
        "this domain is available for purchase", "domain has been registered",
        "domain has expired", "reserve this domain name", "bid on this domain"
    ]
    
    # 2. Expanded list of hosting providers and registrars
    hosting_providers = [
        "godaddy", "hostgator", "bluehost", "namecheap", "dreamhost",
        "domain registration", "web hosting service", "hosting provider",
        "register this domain", "domain broker", "proxy error", "error connecting",
        "domain has expired", "domain has been registered", "courtesy page",
        "ionos", "domain.com", "hover", "namesilo", "porkbun", 
        "network solutions", "register.com", "name.com", "enom", 
        "dynadot", "hover", "domainking", "domainmonster", "1and1", 
        "1&1", "ionos", "registrar", "dominio", "parkingcrew",
        "sedo", "bodis", "parked.com", "parking.com"
    ]
    
    # 3. Technical issues phrases that often appear on parked domains or error pages
    technical_phrases = [
        "proxy error", "error connecting", "connection error", "courtesy page",
        "site not found", "domain not configured", "default web page",
        "website coming soon", "under construction", "future home of",
        "site temporarily unavailable", "domain has been registered",
        "refused to connect", "connection refused", "this page isn't working",
        "error 403", "error 404", "forbidden", "access denied", 
        "default page", "server configuration", "web server at", 
        "webserver at", "website is unavailable", "site is unavailable",
        "this site can't be reached", "server IP address could not be found"
    ]
    
    # Count explicit parking phrases
    explicit_matches = 0
    matched_phrases = []
    
    for phrase in explicit_parking_phrases:
        if phrase in content_lower:
            explicit_matches += 1
            matched_phrases.append(phrase)
    
    # If we have 1 or more explicit parking indicators, check for additional evidence
    if explicit_matches >= 1:
        logger.info(f"Domain contains {explicit_matches} explicit parking phrases ({', '.join(matched_phrases)})")
        
        # Check for hosting providers
        hosting_matches = sum(1 for phrase in hosting_providers if phrase in content_lower)
        if hosting_matches >= 1:
            logger.info(f"Domain contains explicit parking phrase and hosting provider reference, considering as parked")
            return True
            
        # Check for technical issues phrases
        tech_matches = sum(1 for phrase in technical_phrases if phrase in content_lower)
        if tech_matches >= 1:
            logger.info(f"Domain contains explicit parking phrase and technical issues, considering as parked")
            return True
            
        # If only one explicit phrase but it's a strong indicator
        strong_indicators = ["domain is for sale", "buy this domain", "this domain is for sale", "parked by"]
        if any(indicator in matched_phrases for indicator in strong_indicators):
            logger.info(f"Domain contains strong parking phrase indicator, considering as parked")
            return True
    
    # 4. Check for common hosting/registrar parking indicators with additional validation
    hosting_matches = 0
    hosting_matched_phrases = []
    
    for phrase in hosting_providers:
        if phrase in content_lower:
            hosting_matches += 1
            hosting_matched_phrases.append(phrase)
    
    # Check for technical issues too
    tech_matches = sum(1 for phrase in technical_phrases if phrase in content_lower)
    
    # If we have multiple hosting phrases or hosting + technical issues, likely parked
    if (hosting_matches >= 2) or (hosting_matches >= 1 and tech_matches >= 1):
        logger.info(f"Domain contains hosting/registrar phrases and technical issues, considering as parked")
        return True
    
    # 5. Special check for proxy errors with GoDaddy mentions
    if "proxy error" in content_lower and any(provider in content_lower for provider in ["godaddy", "domain", "hosting"]):
        logger.info(f"Found proxy error with hosting service mentions, likely parked")
        return True
        
    # 6. Special check for connection refused with common error patterns
    if ("connection refused" in content_lower or "refused to connect" in content_lower) and len(content.strip()) < 600:
        logger.info(f"Found connection refused error, likely parked or inactive")
        return True
    
    # 7. Check for minimal content with specific patterns
    if len(content.strip()) < 200:
        indicators_found = 0
        
        # a) Check for domain name mentions
        if domain and domain.split('.')[0].lower() in content_lower:
            domain_root = domain.split('.')[0].lower()
            domain_mentions = content_lower.count(domain_root)
            
            if domain_mentions >= 2 and len(content.strip()) < 150:
                indicators_found += 1
                logger.info(f"Found indicator: Multiple domain name mentions in minimal content")
        
        # b) Very little content with no indicators of real site structure
        if len(content.strip()) < 100:
            # Check for JS frameworks before assuming parked
            js_indicators = ["react", "angular", "vue", "javascript", "script", "bootstrap", "jquery"]
            has_js_indicator = any(indicator in content_lower for indicator in js_indicators)
            
            # Check for HTML structure
            html_indicators = ["<!doctype", "<html", "<head", "<meta", "<title", "<body", "<div"]
            html_count = sum(1 for indicator in html_indicators if indicator in content_lower)
            
            # Check for content with useful text (to exclude parked domains)
            content_indicators = ["about", "contact", "service", "product", "company", "team", "home", "blog"]
            text_indicators = any(indicator in content_lower for indicator in content_indicators)
            
            if not has_js_indicator and html_count < 3 and not text_indicators:
                indicators_found += 1
                logger.info(f"Found indicator: Minimal HTML structure with no content indicators")
        
        # c) Check for organization-specific terms (to avoid false positives for businesses)
        org_indicators = ["company", "business", "service", "product", "about us", "contact",
                         "team", "mission", "vision", "customer", "client", "solution",
                         "technology", "industry", "platform", "app", "application"]
                         
        # If org indicators are present, this is likely a real site with minimal content, not parked
        if any(indicator in content_lower for indicator in org_indicators):
            logger.info(f"Found organization indicators in content, not considering as parked")
            return False
            
        # d) Check for few unique words
        words = re.findall(r'\b\w+\b', content_lower)
        unique_words = set(words)
        
        if len(unique_words) < 10:  # Stricter threshold
            indicators_found += 1
            logger.info(f"Found indicator: Very few unique words ({len(unique_words)})")
            
        # Require multiple indicators for minimal content
        if indicators_found >= 2:
            logger.info(f"Domain has {indicators_found} indicators of being parked with minimal content")
            return True
            
    # 8. Special case for GoDaddy proxy errors (for cases like crapanzano.net)
    if "proxy error" in content_lower and len(content) < 500:
        logger.info("Found proxy error message in minimal content, likely parked or inactive")
        return True
        
    # 9. Special case for default web server pages
    default_server_indicators = [
        "default web page", "it works", "web server at", "apache is functioning",
        "nginx is functioning", "server configuration", "default site", 
        "default server page", "welcome to nginx"
    ]
    
    if any(indicator in content_lower for indicator in default_server_indicators) and len(content) < 600:
        logger.info("Found default web server page, considering as inactive")
        return True
    
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
