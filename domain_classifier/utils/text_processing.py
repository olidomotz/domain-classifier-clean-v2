"""Text processing utilities for domain classification."""
import re
import logging
import json
from typing import Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

def clean_json_string(json_str: str) -> str:
    """
    Clean a JSON string by removing control characters and fixing common issues.
    
    Args:
        json_str: The JSON string to clean
        
    Returns:
        str: The cleaned JSON string
    """
    # Replace control characters
    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
    
    # Replace single quotes with double quotes
    cleaned = re.sub(r"'([^']*)':", r'"\1":', cleaned)
    
    # Fix trailing commas
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Fix missing quotes around property names
    cleaned = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', cleaned)
    
    # Replace unescaped newlines in strings
    cleaned = re.sub(r'(".*?)\n(.*?")', r'\1\\n\2', cleaned, flags=re.DOTALL)
    
    # Handle decimal values without leading zero
    cleaned = re.sub(r':\s*\.(\d+)', r': 0.\1', cleaned)
    
    # Try to fix quotes within quotes in explanation fields by escaping them
    if '"llm_explanation"' in cleaned:
        # Complex regex to find the explanation field and properly escape quotes
        explanation_pattern = r'"llm_explanation"\s*:\s*"(.*?)"(?=,|\s*})'
        match = re.search(explanation_pattern, cleaned, re.DOTALL)
        if match:
            explanation_text = match.group(1)
            # Escape any unescaped quotes within the explanation
            fixed_explanation = explanation_text.replace('"', '\\"')
            # Replace back in the original string
            cleaned = cleaned.replace(explanation_text, fixed_explanation)
    
    return cleaned

def extract_json(text: str) -> Optional[str]:
    """
    Extract JSON from text response.
    
    Args:
        text: The text to extract JSON from
        
    Returns:
        str: The extracted JSON string, or None if not found
    """
    # Try multiple patterns to extract JSON
    json_patterns = [
        r'({[\s\S]*"predicted_class"[\s\S]*})',  # Most general pattern
        r'```(?:json)?\s*({[\s\S]*})\s*```',     # For markdown code blocks
        r'({[\s\S]*"confidence_scores"[\s\S]*})' # Alternative key pattern
    ]
    
    for pattern in json_patterns:
        json_match = re.search(pattern, text, re.DOTALL)
        if json_match:
            return json_match.group(1)
            
    return None

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

def remove_redundancy(description: str, domain: str) -> str:
    """
    Process a company description to remove redundancy, technology mentions,
    and Internal IT department references while preserving appropriate length (~100 words).
    
    Args:
        description: The original description
        domain: The domain name
        
    Returns:
        str: Cleaned, non-redundant description
    """
    # Get company name without domain extension
    company_name = domain.split('.')[0]
    
    # Replace full domain with just the company name to avoid repetition
    description = re.sub(rf"{re.escape(domain)}", company_name, description)
    
    # Remove repeated mentions of business type
    description = re.sub(r'(managed service provider|msp).*?\b\1\b', r'\1', description, flags=re.IGNORECASE)
    description = re.sub(r'(commercial a\/v integrator).*?\b\1\b', r'\1', description, flags=re.IGNORECASE)
    description = re.sub(r'(residential a\/v integrator).*?\b\1\b', r'\1', description, flags=re.IGNORECASE)
    description = re.sub(r'(internal it department).*?\b\1\b', r'\1', description, flags=re.IGNORECASE)
    
    # Remove generic phrases that don't add information
    generic_phrases = [
        r'as a managed service provider',
        r'as an integrator',
        r'allowing (?:clients|customers) to focus on (?:their|its) core business',
        r'tailored to meet the unique needs of each client',
        r'leveraging a range of technologies',
        r'with a focus on quality',
        r'with a commitment to excellence',
        r'using the latest technologies',
        r'using cutting-edge technologies',
        r'state-of-the-art'
    ]
    
    for phrase in generic_phrases:
        description = re.sub(phrase, '', description, flags=re.IGNORECASE)
    
    # Remove technology names/specific technology mentions
    tech_patterns = [
        r'using \w+ (?:and|,) \w+ (technologies|platforms|solutions)',
        r'based on \w+ platform',
        r'cloud-based \w+ platform',
        r'proprietary \w+ platform',
        r'leverages? \w+ technology',
        r'through \w+ software',
        r'utilizing \w+ solutions',
        r'with \w+ tools',
        r'powered by \w+',
        r'with (?:a|the) \w+ platform'
    ]
    
    for pattern in tech_patterns:
        description = re.sub(pattern, '', description, flags=re.IGNORECASE)
    
    # Remove specific references to internal IT department activities
    it_patterns = [
        r'(?:with|has) (?:an|their own) (?:internal|in-house) IT (?:department|team|staff)',
        r'(?:maintains|supports) (?:an|their) (?:internal|in-house) IT infrastructure',
        r'(?:their|its) (?:internal|in-house) IT (?:department|team) (?:manages|supports|maintains)',
        r'(?:relies on|uses) (?:an|their) internal IT (?:department|team)',
        r'internal IT needs',
        r'IT infrastructure',
        r'IT operations'
    ]
    
    for pattern in it_patterns:
        description = re.sub(pattern, '', description, flags=re.IGNORECASE)
    
    # Focus on what the business does, not its IT department
    # For companies classified as "Internal IT Department", make sure we're talking about the business
    if "internal it department" in description.lower():
        # Change mentions of "Internal IT Department" to more business-focused language
        description = re.sub(
            r'is (?:an?|the) internal IT department', 
            'is a business', 
            description, 
            flags=re.IGNORECASE
        )
        
        # If the company is described primarily by its IT department, refocus
        description = re.sub(
            r'focuses on (?:managing|maintaining) (?:its|their) technology',
            'operates in its industry sector',
            description,
            flags=re.IGNORECASE
        )
    
    # Remove double spaces and clean up
    description = re.sub(r'\s+', ' ', description).strip()
    description = re.sub(r'\s*,\s*,', ',', description)
    description = re.sub(r'\s*\.\s*\.', '.', description)
    
    # Ensure the description ends with a period
    if not description.endswith('.'):
        description += '.'
        
    # Make sure we don't shorten too much - we want approximately 100 words
    word_count = len(description.split())
    
    # Log the word count
    logger.info(f"Description after redundancy removal has {word_count} words")
    
    return description

def generate_one_line_description(content: str, predicted_class: str, domain: str, company_description: str = "") -> str:
    """
    Generate a concise one-line description of what the company does without redundancy.
    
    Args:
        content: The website content
        predicted_class: The predicted class
        domain: The domain name
        company_description: Existing longer company description
        
    Returns:
        str: A concise one-line description
    """
    # First try to extract from longer description
    if company_description:
        # Look for sentences that describe what the company does
        sentences = re.split(r'(?<=[.!?])\s+', company_description)
        
        for sentence in sentences:
            # Skip sentences that talk about classification
            if re.search(r'(classif|previous|based on|was)', sentence, re.IGNORECASE):
                continue
                
            # Look for sentences with action verbs that describe services
            if re.search(r'(provide|offer|specialize|focus|deliver)', sentence, re.IGNORECASE):
                # Clean up, remove redundancy, and return
                return remove_redundancy(sentence.strip(), domain)
            
        # If no sentence with action verbs, try first sentence if it's descriptive enough
        if sentences and len(sentences) > 0 and len(sentences[0]) > 20:
            # Skip if it talks about classification
            if not re.search(r'(classif|previous|based on|was)', sentences[0], re.IGNORECASE):
                return remove_redundancy(sentences[0].strip(), domain)
    
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
                    desc = f"{domain} is {desc}."
                if not desc.endswith('.'):
                    desc += '.'
                    
                return remove_redundancy(desc, domain)
    
    # Generate a basic description based on the class
    if predicted_class == "Managed Service Provider":
        return f"{domain} offers IT services and network management for businesses."
    elif predicted_class == "Integrator - Commercial A/V":
        return f"{domain} installs audiovisual systems for commercial environments."
    elif predicted_class == "Integrator - Residential A/V":
        return f"{domain} specializes in home automation and entertainment systems."
    elif predicted_class == "Parked Domain":
        return f"{domain} is a parked domain with no active business."
    elif predicted_class == "Internal IT Department":
        return f"{domain} is a business that operates in its industry sector."
    else:
        return f"{domain} is a business with specific industry needs."

def extract_company_description(content: str, explanation: str, domain: str) -> str:
    """
    Extract or generate a concise company description without technology mentions.
    
    Args:
        content: The website content
        explanation: The classification explanation
        domain: The domain name
        
    Returns:
        str: A concise company description without redundancy
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
                
            # Process to remove redundancy and technology mentions
            return remove_redundancy(desc, domain)
    
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
                
                # Process to remove redundancy and technology mentions
                return remove_redundancy(desc, domain)
    
    # Fall back to using information from the explanation, focusing on the business, not IT
    predicted_class = ""
    if "managed service provider" in explanation.lower():
        predicted_class = "a managed service provider offering IT services"
    elif "commercial a/v" in explanation.lower():
        predicted_class = "a commercial A/V integrator providing audiovisual solutions for businesses"
    elif "residential a/v" in explanation.lower():
        predicted_class = "a residential A/V integrator specializing in home automation systems"
    elif "internal it" in explanation.lower():
        # For Internal IT, focus on the business itself, not IT department
        industry_hints = re.search(r'in the (\w+) (industry|sector|field)', explanation, re.IGNORECASE)
        if industry_hints:
            industry = industry_hints.group(1)
            predicted_class = f"a business operating in the {industry} sector"
        else:
            predicted_class = "a business entity"
    else:
        predicted_class = "a business whose specific activities couldn't be determined"
    
    description = f"{domain} is {predicted_class}."
    
    # Process to remove redundancy and technology mentions
    return remove_redundancy(description, domain)

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
            
        return remove_redundancy(f"{domain} is a Managed Service Provider{service_text}{industry_text}.", domain)
        
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
            
        return remove_redundancy(f"{domain} is a Commercial A/V Integrator{solution_text}{industry_text}.", domain)
        
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
            
        return remove_redundancy(f"{domain} is a Residential A/V Integrator{solution_text} for home environments.", domain)
        
    else:  # Internal IT Department / non-service business
        business_type = ""
        if industry_keywords:
            business_type = f"a {', '.join(industry_keywords)} business"
        else:
            business_type = "a business entity"
            
        return remove_redundancy(f"{domain} is {business_type} that operates in its market.", domain)
