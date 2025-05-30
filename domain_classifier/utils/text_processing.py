import re
import logging
from typing import Optional, Dict, Any

# Import from centralized JSON parser module
from domain_classifier.utils.json_parser import extract_json_from_text as extract_json
from domain_classifier.utils.json_parser import clean_json_string

# Set up logging
logger = logging.getLogger(__name__)

def detect_minimal_content(content: str) -> bool:
    """
    Detect if domain has minimal content.
    
    Args:
        content: The website content
        
    Returns:
        bool: True if the domain has minimal content
    """
    if not content or len(content.strip()) < 100:
        logger.info(f"Content is very short: {len(content) if content else 0} characters")
        return True
        
    # Count words in content
    words = re.findall(r'\b\w+\b', content.lower())
    unique_words = set(words)
    
    # Return true if few words or unique words
    if len(words) < 50:
        logger.info(f"Content has few words ({len(words)}), likely minimal content")
        return True
        
    if len(unique_words) < 30:
        logger.info(f"Content has few unique words ({len(unique_words)}), likely minimal content")
        return True
            
    return False

def remove_redundancy(text: str, domain: str = None) -> str:
    """
    Remove redundant phrases from a text.
    
    Args:
        text: The text to process
        domain: Optional domain name for context
        
    Returns:
        str: The processed text with redundancies removed
    """
    # Skip if text is too short
    if not text or len(text) < 20:
        return text
        
    # Convert domain to company name format if provided
    company_name = None
    if domain:
        company_name = domain.split('.')[0].capitalize()
    
    # List of redundant phrases to remove
    redundant_phrases = [
        r'allowing (?:clients|customers) to focus on (?:their|its) core business',
        r'with a focus on customer service',
        r'with a commitment to quality',
        r'with a dedication to excellence',
        r'with a focus on (?:excellence|quality|service)',
        r'for businesses of all sizes',
        r'for all your .* needs',
        r'providing quality .* services',
        r'premier provider of',
        r'trusted provider of',
        r'industry-leading',
        r'state-of-the-art',
        r'cutting-edge',
        r'one-stop-shop for'
    ]
    
    # Add domain-specific redundant phrases
    if company_name:
        domain_phrases = [
            fr'{company_name} is a (?:leading|premier|trusted) (?:provider of|company for)',
            fr'{company_name} offers a wide range of',
            fr'{company_name} specializes in providing',
        ]
        redundant_phrases.extend(domain_phrases)
    
    # Remove redundant phrases
    result = text
    for phrase in redundant_phrases:
        result = re.sub(phrase, '', result, flags=re.IGNORECASE)
    
    # Fix spacing issues from removals
    result = re.sub(r'\s+', ' ', result).strip()
    result = re.sub(r'\s+\.', '.', result)
    result = re.sub(r'\s+,', ',', result)
    
    # Add logging for debugging
    word_count_before = len(text.split())
    word_count_after = len(result.split())
    if word_count_before != word_count_after:
        logger.info(f"Description after redundancy removal has {word_count_after} words")
    
    return result

def extract_explanation(text: str) -> Optional[str]:
    """
    Extract an explanation from free-form text.
    
    Args:
        text: The text to extract from
        
    Returns:
        str: The extracted explanation, or None if not found
    """
    # Look for explicit explanation markers
    explanation_patterns = [
        r'explanation:(.*?)(?:\n\n|\Z)',
        r'llm explanation:(.*?)(?:\n\n|\Z)',
        r'reasoning:(.*?)(?:\n\n|\Z)',
        r'analysis:(.*?)(?:\n\n|\Z)',
    ]
    
    for pattern in explanation_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            explanation = match.group(1).strip()
            if len(explanation) > 50:  # Ensure it's substantial
                return explanation
    
    # If no explicit marker, look for STEP format
    step_pattern = r'STEP 1:.*?STEP [2-5]:'
    match = re.search(step_pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        explanation = match.group(0).strip()
        if len(explanation) > 50:
            return explanation + " " + text.split("STEP 5:", 1)[-1].strip()
    
    # As a last resort, just return the raw text if it's not too long
    if len(text) < 1000:
        return text
        
    return None

def generate_explanation(predicted_class: str, domain: str = None, 
                        is_service: bool = True, internal_it_potential: int = None) -> str:
    """
    Generate a standard explanation for a classification.
    
    Args:
        predicted_class: The predicted class
        domain: Optional domain name for context
        is_service: Whether the business is a service business
        internal_it_potential: Internal IT potential (0-100)
        
    Returns:
        str: The generated explanation
    """
    domain_name = domain or "This domain"
    
    # Format the explanation with step-by-step decision tree format
    explanation = f"STEP 1: The website content provides sufficient information to analyze and classify this business.\n\n"
    explanation += f"STEP 2: {domain_name} is not a parked or inactive domain.\n\n"
    
    if is_service:
        explanation += f"STEP 3: {domain_name} is a technology service/management business that provides services to external clients.\n\n"
        
        if predicted_class == "Managed Service Provider":
            explanation += f"STEP 4: The business is classified as a Managed Service Provider because it primarily offers IT infrastructure management, security services, and ongoing technical support to clients.\n\n"
        elif predicted_class == "Integrator - Commercial A/V":
            explanation += f"STEP 4: The business is classified as a Commercial A/V Integrator because it primarily designs and installs audio-visual systems for commercial environments like offices and conference rooms.\n\n"
        elif predicted_class == "Integrator - Residential A/V":
            explanation += f"STEP 4: The business is classified as a Residential A/V Integrator because it primarily focuses on home automation and entertainment systems for residential clients.\n\n"
        
        explanation += f"STEP 5: Since this is classified as a service business, the internal IT potential is set to 0/100.\n\n"
    else:
        explanation += f"STEP 3: {domain_name} is NOT a technology service/management business. It does not appear to provide IT or A/V services to clients.\n\n"
        explanation += f"STEP 4: Since this is not a service business, it is classified as Internal IT Department.\n\n"
        
        if internal_it_potential is not None:
            explanation += f"STEP 5: Based on the company size and industry, the internal IT potential is assessed at {internal_it_potential}/100.\n\n"
        else:
            explanation += f"STEP 5: The company's internal IT potential could not be accurately determined from the available information.\n\n"
    
    return explanation

def extract_company_description(content: str, explanation: str, domain: str) -> str:
    """
    Extract or generate a concise company description.
    
    Args:
        content: The website content
        explanation: The classification explanation
        domain: The domain name
        
    Returns:
        str: A concise company description
    """
    # First try to extract from LLM explanation
    description_patterns = [
        r'company description: (.*?)(?=\n|\.|$)',
        r'(?:the company|this company|the business|this business) (provides|offers|specializes in|focuses on|is) ([^.]+)',
        r'(?:appears to be|seems to be) (a|an) ([^.]+)'
    ]
    
    for pattern in description_patterns:
        match = re.search(pattern, explanation, re.IGNORECASE)
        if match and len(match.group(0)) > 20:
            # Clean up the description
            desc = match.group(0).strip()
            # Convert to third person if needed
            desc = re.sub(r'^we ', f"{domain} ", desc, flags=re.IGNORECASE)
            desc = re.sub(r'^our ', f"{domain}'s ", desc, flags=re.IGNORECASE)
            
            # Ensure it starts with the domain name
            if not desc.lower().startswith(domain.lower()):
                desc = f"{domain} {desc}"
                
            return desc
    
    # If explanation doesn't yield a good description, try to extract from website content
    if content:
        # Look for an "about us" paragraph
        about_patterns = [
            r'about\s+us[^.]*(?:[^.]*\.){1,2}',
            r'who\s+we\s+are[^.]*(?:[^.]*\.){1,2}',
            r'our\s+company[^.]*(?:[^.]*\.){1,2}'
        ]
        
        for pattern in about_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match and len(match.group(0)) > 30:
                # Clean up the description
                desc = match.group(0).strip()
                # Convert to third person if needed
                desc = re.sub(r'^we ', f"{domain} ", desc, flags=re.IGNORECASE)
                desc = re.sub(r'^our ', f"{domain}'s ", desc, flags=re.IGNORECASE)
                
                # Ensure it starts with the domain name
                if not desc.lower().startswith(domain.lower()):
                    desc = f"{domain} {desc}"
                    
                # Limit length
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                    
                return desc
    
    # Fall back to using information from the explanation
    predicted_class = ""
    if "managed service provider" in explanation.lower():
        predicted_class = "a Managed Service Provider offering IT services and solutions"
    elif "commercial a/v" in explanation.lower():
        predicted_class = "a Commercial A/V Integrator providing audiovisual solutions for businesses"
    elif "residential a/v" in explanation.lower():
        predicted_class = "a Residential A/V Integrator specializing in home automation and entertainment systems"
    elif "internal it" in explanation.lower():
        predicted_class = "a business with internal IT needs rather than an IT service provider"
    else:
        predicted_class = "a business whose specific activities couldn't be determined with high confidence"
    
    return f"{domain} appears to be {predicted_class}."

def extract_keywords_company_description(content: str, predicted_class: str, domain: str) -> str:
    """
    Generate a company description based on keywords in the content.
    
    Args:
        content: The website content
        predicted_class: The predicted class
        domain: The domain name
        
    Returns:
        str: A keyword-based company description
    """
    content_lower = content.lower()
    
    # Identify industry keywords
    industry_keywords = []
    industry_patterns = [
        (r'(healthcare|medical|health\s+care|patient)', "healthcare"),
        (r'(education|school|university|college|learning|teaching)', "education"),
        (r'(finance|banking|investment|financial|insurance)', "finance"),
        (r'(retail|ecommerce|e-commerce|online\s+store|shop)', "retail"),
        (r'(manufacturing|factory|production|industrial)', "manufacturing"),
        (r'(government|public\s+sector|federal|state|municipal)', "government"),
        (r'(hospitality|hotel|restaurant|tourism)', "hospitality"),
        (r'(technology|software|saas|cloud|application)', "technology"),
        (r'(construction|building|architecture|engineering)', "construction"),
        (r'(transportation|logistics|shipping|freight)', "transportation")
    ]
    
    for pattern, keyword in industry_patterns:
        if re.search(pattern, content_lower):
            industry_keywords.append(keyword)
    
    # Create appropriate description based on predicted class and keywords
    if predicted_class == "Managed Service Provider":
        services = []
        if "network" in content_lower or "networking" in content_lower:
            services.append("network management")
        if "security" in content_lower or "cyber" in content_lower:
            services.append("cybersecurity")
        if "cloud" in content_lower:
            services.append("cloud services")
        if "support" in content_lower or "helpdesk" in content_lower:
            services.append("technical support")
        
        service_text = ""
        if services:
            service_text = f" specializing in {', '.join(services)}"
            
        industry_text = ""
        if industry_keywords:
            industry_text = f" for the {', '.join(industry_keywords)} {'industry' if len(industry_keywords) == 1 else 'industries'}"
            
        return f"{domain} is a Managed Service Provider{service_text}{industry_text}."
        
    elif predicted_class == "Integrator - Commercial A/V":
        solutions = []
        if "conference" in content_lower or "meeting" in content_lower:
            solutions.append("conference room systems")
        if "digital signage" in content_lower or "display" in content_lower:
            solutions.append("digital signage")
        if "video" in content_lower and "wall" in content_lower:
            solutions.append("video walls")
        if "automation" in content_lower:
            solutions.append("automation systems")
            
        solution_text = ""
        if solutions:
            solution_text = f" providing {', '.join(solutions)}"
            
        industry_text = ""
        if industry_keywords:
            industry_text = f" for the {', '.join(industry_keywords)} {'sector' if len(industry_keywords) == 1 else 'sectors'}"
            
        return f"{domain} is a Commercial A/V Integrator{solution_text}{industry_text}."
        
    elif predicted_class == "Integrator - Residential A/V":
        solutions = []
        if "home theater" in content_lower or "cinema" in content_lower:
            solutions.append("home theaters")
        if "automation" in content_lower or "smart home" in content_lower:
            solutions.append("smart home automation")
        if "audio" in content_lower:
            solutions.append("audio systems")
        if "lighting" in content_lower:
            solutions.append("lighting control")
            
        solution_text = ""
        if solutions:
            solution_text = f" specializing in {', '.join(solutions)}"
            
        return f"{domain} is a Residential A/V Integrator{solution_text} for home environments."
        
    else:  # Internal IT Department / non-service business
        business_type = ""
        if industry_keywords:
            business_type = f"a {', '.join(industry_keywords)} business"
        else:
            business_type = "a business entity"
            
        return f"{domain} appears to be {business_type} that doesn't provide IT or A/V services to clients."

def generate_one_line_description(content: str, predicted_class: str, domain: str, company_description: str = "") -> str:
    """
    Generate a concise one-line description of what the company does.
    
    Args:
        content: The website content
        predicted_class: The predicted class
        domain: The domain name
        company_description: Existing longer company description
        
    Returns:
        str: A concise one-line description
    """
    # First try to extract the first relevant sentence from the longer description
    if company_description:
        # Look for sentences that describe what the company does
        sentences = re.split(r'(?<=[.!?])\s+', company_description)
        
        for sentence in sentences:
            # Skip sentences that talk about classification
            if re.search(r'(classif|previous|based on|was)', sentence, re.IGNORECASE):
                continue
                
            # Look for sentences with action verbs that describe services
            if re.search(r'(provide|offer|specialize|focus|deliver)', sentence, re.IGNORECASE):
                # Clean up and return the sentence
                return sentence.strip()
            
        # If no sentence with action verbs, try first sentence if it's descriptive enough
        if sentences and len(sentences) > 0 and len(sentences[0]) > 20:
            # Skip if it talks about classification
            if not re.search(r'(classif|previous|based on|was)', sentences[0], re.IGNORECASE):
                return sentences[0].strip()
    
    # Fall back to content extraction
    if content:
        # Look for common descriptive patterns
        patterns = [
            rf"{re.escape(domain)}(?:\s+is)?\s+(?:a|an)\s+([^.]+)",
            r"(?:we|our company)\s+(?:provide|offer|specialize in|are)\s+([^.]+)",
            r"(?:leading|premier|trusted)\s+provider\s+of\s+([^.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match and len(match.group(1)) > 10:
                desc = match.group(1).strip()
                # Convert "we provide" to "{domain} provides"
                desc = re.sub(r'^we ', f"{domain} ", desc, flags=re.IGNORECASE)
                desc = re.sub(r'^our company ', f"{domain} ", desc, flags=re.IGNORECASE)
                
                # Format as a complete sentence if needed
                if not desc.lower().startswith(domain.lower()):
                    return f"{domain} is {desc}."
                return f"{desc}."
    
    # Generate a basic description based on the class
    if predicted_class == "Managed Service Provider":
        return f"{domain} is an IT service provider offering managed technology solutions."
    elif predicted_class == "Integrator - Commercial A/V":
        return f"{domain} installs and maintains audiovisual systems for businesses."
    elif predicted_class == "Integrator - Residential A/V":
        return f"{domain} specializes in home automation and entertainment systems."
    elif predicted_class == "Parked Domain":
        return f"{domain} is a parked domain with no active business."
    else:  # Internal IT Department / non-service business
        return f"{domain} is a business with internal IT needs."

def ensure_classifier_type(result: dict, domain: str = None, 
                       default_type: str = "claude-llm-enriched",
                       log_warning: bool = False) -> dict:
    """
    Ensure that classifier_type is set in the result dictionary.
    
    Args:
        result: The result dictionary
        domain: Optional domain name for logging
        default_type: Default classifier_type to use
        log_warning: Whether to log a warning if classifier_type is missing
        
    Returns:
        dict: The result dictionary with classifier_type set
    """
    domain_str = f" for {domain}" if domain else ""
    
    # Check for API formatter output first
    if '02_classifier_type' in result:
        # If we have a formatted result, extract the original classifier_type
        result['classifier_type'] = result.get('02_classifier_type')
        logger.info(f"Found classifier_type in formatted result{domain_str}: {result['classifier_type']}")
        return result
        
    # Handle case when classifier_type is missing or empty
    if "classifier_type" not in result or not result["classifier_type"]:
        if log_warning:
            logger.warning(f"No classifier_type found in result{domain_str}")
        else:
            logger.info(f"Setting classifier_type to {default_type}{domain_str}")
        result["classifier_type"] = default_type
    elif "claude-llm" not in result["classifier_type"]:
        # Special case for 'pending_classification'
        if result["classifier_type"] == "pending_classification":
            logger.info(f"Upgrading pending_classification to {default_type}{domain_str}")
            result["classifier_type"] = default_type
        elif log_warning:
            logger.warning(f"Non-LLM classifier type found: {result['classifier_type']}{domain_str}")
    else:
        # Preserve original type info but ensure enriched is indicated if needed
        if "enriched" in default_type and "enriched" not in result["classifier_type"]:
            original_type = result["classifier_type"]
            result["classifier_type"] = f"{original_type}-enriched"
            logger.info(f"Updated classifier_type to {result['classifier_type']}{domain_str}")
            
    # Ensure classifier_type is not too long for Snowflake
    if len(result["classifier_type"]) > 40:
        original_length = len(result["classifier_type"])
        result["classifier_type"] = result["classifier_type"][:40]
        logger.info(f"Truncated classifier_type from {original_length} to 40 chars{domain_str}")
    
    return result
