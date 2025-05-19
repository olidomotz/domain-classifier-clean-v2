"""Decision tree logic for domain classification."""
import logging
import re
from typing import Dict, Any, Optional, Tuple, List

# Set up logging
logger = logging.getLogger(__name__)

def is_parked_domain(content: str, domain: str = None) -> bool:
    """
    Enhanced detection of truly parked domains vs. just having minimal content.
    
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
    
    # 1. Check for direct explicit parking phrases (strongest indicators)
    explicit_parking_phrases = [
        "domain is for sale", "buy this domain", "purchasing this domain", 
        "domain may be for sale", "this domain is for sale", "parked by",
        "domain parking", "this web page is parked", "domain for sale",
        "this website is for sale", "domain name parking",
        "purchase this domain"
    ]
    
    # Count explicit parking phrases - require at least 2 to avoid false positives
    explicit_matches = 0
    matched_phrases = []
    for phrase in explicit_parking_phrases:
        if phrase in content_lower:
            explicit_matches += 1
            matched_phrases.append(phrase)
    
    # If we have 2 or more explicit parking indicators, it's definitely parked
    if explicit_matches >= 2:
        logger.info(f"Domain contains {explicit_matches} explicit parking phrases ({', '.join(matched_phrases)}), considering as parked")
        return True
    # If just one explicit indicator, require additional evidence
    elif explicit_matches == 1:
        # Continue to other checks, but with a lower threshold
        logger.info(f"Domain contains 1 explicit parking phrase ({matched_phrases[0]}), checking for additional evidence")
    else:
        # No explicit parking phrases, require stronger evidence from other checks
        pass
    
    # 2. Check for common hosting/registrar parking indicators with additional validation
    hosting_phrases = [
        "godaddy", "hostgator", "bluehost", "namecheap", "dreamhost", 
        "domain registration", "web hosting service", "hosting provider",
        "register this domain", "domain broker", "proxy error", "error connecting",
        "domain has expired", "domain has been registered", "courtesy page"
    ]
    
    # Count hosting phrases and require more matches for positive detection
    hosting_matches = 0
    hosting_matched_phrases = []
    for phrase in hosting_phrases:
        if phrase in content_lower:
            hosting_matches += 1
            hosting_matched_phrases.append(phrase)
    
    # Require at least 2 hosting phrases (more strict than before) to reduce false positives
    if hosting_matches >= 2:
        logger.info(f"Domain contains {hosting_matches} hosting/registrar phrases ({', '.join(hosting_matched_phrases)}), considering as parked")
        return True
    
    # 3. Check for minimal content with specific patterns (require multiple indicators)
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
        
        if len(unique_words) < 10:  # Decreased from 15 to be more strict
            indicators_found += 1
            logger.info(f"Found indicator: Very few unique words ({len(unique_words)})")
        
        # Require at least 3 indicators for minimal content (increased from 2 to avoid false positives)
        if indicators_found >= 3:
            logger.info(f"Domain has {indicators_found} indicators of being parked with minimal content")
            return True
    
    # 4. Check specifically for proxy errors with hosting mentions (like crapanzano.net example)
    if len(content.strip()) < 300 and "proxy" in content_lower and any(phrase in content_lower for phrase in ["godaddy", "domain", "hosting"]):
        logger.info("Found proxy error with hosting service mentions, likely parked")
        return True
        
    return False

def check_special_domain_cases(domain: str, text_content: str) -> Optional[Dict[str, Any]]:
    """
    Check for special domain cases that need custom handling.
    
    Args:
        domain: The domain name
        text_content: The website content
        
    Returns:
        Optional[Dict[str, Any]]: Custom result if special case, None otherwise
    """
    domain_lower = domain.lower()
    content_lower = text_content.lower() if text_content else ""
    
    # Check for security-focused domains
    security_domain_indicators = ['cyber', 'security', 'secure', 'protect', 'defense', 'infosec', 'threat']
    security_content_indicators = [
        'cyber security', 'cybersecurity', 'information security', 'network security',
        'security services', 'security solutions', 'managed security', 'vulnerability',
        'penetration testing', 'security monitoring', 'threat', 'attack', 'malware',
        'ransomware', 'firewall', 'intrusion', 'data breach', 'risk assessment',
        'security operations', 'soc', 'security operations center', 'mssp',
        'managed security service provider', 'security awareness',
        'incident response', 'threat intelligence', 'endpoint protection',
        'zero trust', 'compliance', 'audit', 'protection', 'defend'
    ]
    
    # Check for security indicators in domain name
    domain_matches = [indicator for indicator in security_domain_indicators if indicator in domain_lower]
    
    # Check for security indicators in content
    content_matches = []
    if content_lower:
        content_matches = [indicator for indicator in security_content_indicators if indicator in content_lower]
    
    # If we have strong security indicators (domain name AND content matches)
    if (domain_matches and len(content_matches) >= 2) or (not domain_matches and len(content_matches) >= 4):
        logger.info(f"Detected security company from domain {domain} and content indicators: {content_matches}")
        
        return {
            "processing_status": 2,
            "is_service_business": True,
            "predicted_class": "Managed Service Provider",
            "internal_it_potential": 0,
            "confidence_scores": {
                "Managed Service Provider": 85,
                "Integrator - Commercial A/V": 8,
                "Integrator - Residential A/V": 5,
                "Internal IT Department": 0
            },
            "llm_explanation": f"{domain} appears to be a cybersecurity company providing security services and solutions to protect organizations from cyber threats. Based on the strong security focus in {'both the domain name and' if domain_matches else ''} website content, this is classified as a Managed Service Provider specializing in security services.",
            "company_description": f"{domain} is a cybersecurity company providing managed security services and solutions to protect organizations from cyber threats.",
            "detection_method": "security_domain_detection",
            "low_confidence": False,
            "max_confidence": 0.85
        }
    
    # Check for cybersafe.sg specifically
    if domain_lower == "cybersafe.sg":
        logger.warning(f"Special handling for known cybersecurity domain: {domain}")
        return {
            "processing_status": 2,
            "is_service_business": True,
            "predicted_class": "Managed Service Provider",
            "internal_it_potential": 0,
            "confidence_scores": {
                "Managed Service Provider": 90,
                "Integrator - Commercial A/V": 5,
                "Integrator - Residential A/V": 3,
                "Internal IT Department": 0
            },
            "llm_explanation": f"{domain} is a cybersecurity services provider offering managed security solutions, risk assessment, and protection against cyber threats. The company focuses on security services which classifies it as a Managed Service Provider specializing in cybersecurity.",
            "company_description": f"{domain} is a cybersecurity services provider offering managed security solutions, vulnerability assessments, and protection against cyber threats for businesses.",
            "company_one_line": f"{domain} provides managed cybersecurity services to protect organizations.",
            "detection_method": "domain_override",
            "low_confidence": False,
            "max_confidence": 0.9
        }
        
    # Check for special domains with known classifications
    # HostiFi - always MSP
    if "hostifi" in domain_lower:
        logger.info(f"Special case handling for known MSP domain: {domain}")
        return {
            "processing_status": 2,
            "is_service_business": True,
            "predicted_class": "Managed Service Provider",
            "internal_it_potential": 0,
            "confidence_scores": {
                "Managed Service Provider": 85,
                "Integrator - Commercial A/V": 8,
                "Integrator - Residential A/V": 5,
                "Internal IT Department": 0
            },
            "llm_explanation": f"{domain} is a cloud hosting platform specializing in Ubiquiti network management. They provide managed hosting services for UniFi Controller, UISP, and other network management software, which is a clear indication they are a Managed Service Provider focused on network infrastructure management.",
            "company_description": f"{domain} is a cloud hosting platform specializing in Ubiquiti network management and UniFi Controller hosting services.",
            "detection_method": "domain_override",
            "low_confidence": False,
            "max_confidence": 0.85
        }
        
    # Special handling for ciao.dk (known problematic vacation rental site)
    if domain_lower == "ciao.dk":
        logger.warning(f"Special handling for known vacation rental domain: {domain}")
        return {
            "processing_status": 2,
            "is_service_business": False,
            "predicted_class": "Internal IT Department",
            "internal_it_potential": 40,
            "confidence_scores": {
                "Managed Service Provider": 5,
                "Integrator - Commercial A/V": 3,
                "Integrator - Residential A/V": 2,
                "Internal IT Department": 40
            },
            "llm_explanation": f"{domain} appears to be a vacation rental/travel booking website offering holiday accommodations in various destinations. This is not a service business in the IT or A/V integration space. It's a travel industry business that might have a small internal IT department to maintain their booking systems and website.",
            "company_description": f"{domain} is a vacation rental and travel booking website offering holiday accommodations in various destinations.",
            "detection_method": "domain_override",
            "low_confidence": False,
            "max_confidence": 0.4
        }
        
    # Check for other vacation/travel-related domains
    vacation_terms = ["vacation", "holiday", "rental", "booking", "hotel", "travel", "accommodation", "ferie"]
    found_terms = [term for term in vacation_terms if term in domain_lower]
    if found_terms:
        logger.warning(f"Domain {domain} contains vacation/travel terms: {found_terms}")
        
        # Look for confirmation in the content
        travel_content_terms = ["booking", "accommodation", "stay", "vacation", "holiday", "rental"]
        if content_lower and any(term in content_lower for term in travel_content_terms):
            logger.warning(f"Content confirms {domain} is likely a travel/vacation site")
            return {
                "processing_status": 2,
                "is_service_business": False,
                "predicted_class": "Internal IT Department",
                "internal_it_potential": 35,
                "confidence_scores": {
                    "Managed Service Provider": 5,
                    "Integrator - Commercial A/V": 3,
                    "Integrator - Residential A/V": 2,
                    "Internal IT Department": 35
                },
                "llm_explanation": f"{domain} appears to be a travel/vacation related business, not an IT or A/V service provider. The website focuses on accommodations, bookings, or vacation rentals rather than technology services or integration. This type of business might have minimal internal IT needs depending on its size.",
                "company_description": f"{domain} appears to be a travel or vacation rental business focused on accommodations and bookings.",
                "detection_method": "domain_override",
                "low_confidence": False,
                "max_confidence": 0.35
            }
            
    # Check for transportation/logistics companies
    transport_terms = ["trucking", "transport", "logistics", "shipping", "freight", "delivery", "carrier"]
    found_transport_terms = [term for term in transport_terms if term in domain_lower]
    if found_transport_terms:
        logger.warning(f"Domain {domain} contains transportation terms: {found_transport_terms}")
        # Look for confirmation in the content
        transport_content_terms = ["shipping", "logistics", "fleet", "trucking", "transportation", "delivery"]
        if content_lower and any(term in content_lower for term in transport_content_terms):
            logger.warning(f"Content confirms {domain} is likely a transportation/logistics company")
            return {
                "processing_status": 2,
                "is_service_business": False,
                "predicted_class": "Internal IT Department",
                "internal_it_potential": 60,
                "confidence_scores": {
                    "Managed Service Provider": 5,
                    "Integrator - Commercial A/V": 3,
                    "Integrator - Residential A/V": 2,
                    "Internal IT Department": 60
                },
                "llm_explanation": f"{domain} appears to be a transportation and logistics company, not an IT or A/V service provider. The website focuses on shipping, transportation, and logistics services rather than technology services or integration. This type of company typically has moderate internal IT needs to manage their operations and fleet management systems.",
                "company_description": f"{domain} is a transportation and logistics company providing shipping and freight services.",
                "detection_method": "domain_override",
                "low_confidence": False,
                "max_confidence": 0.6
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
