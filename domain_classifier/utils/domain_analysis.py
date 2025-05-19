"""Domain name analysis utilities."""
import re
import logging
from typing import Dict, Any, List

# Set up logging
logger = logging.getLogger(__name__)

def analyze_domain_words(domain: str) -> Dict[str, float]:
    """
    Analyze domain name words for classification signals.
    
    Args:
        domain: The domain name to analyze
        
    Returns:
        dict: Scores for different business types
    """
    # Extract the part before the TLD
    domain_base = domain.split('.')[0].lower()
    
    # Remove common suffixes like "inc", "llc", "corp"
    domain_base = re.sub(r'(inc|llc|corp|ltd|spa|srl|gmbh)$', '', domain_base)
    
    # Split by common separators
    words = re.split(r'[-_]', domain_base)
    
    # For compound words, try to break them down
    expanded_words = []
    for word in words:
        # Find word boundaries in camelCase or multiple words
        parts = re.findall(r'[A-Za-z][a-z]+', word)
        if parts:
            expanded_words.extend(parts)
        else:
            # If no parts found, just use the original word
            expanded_words.append(word)
    
    # Clean up any empty strings
    expanded_words = [w for w in expanded_words if w]
    
    logger.info(f"Extracted words from domain {domain}: {expanded_words}")
    
    # Define dictionaries of words associated with different business types
    msp_words = {'tech', 'it', 'cyber', 'net', 'cloud', 'secure', 'host', 'data', 
                 'web', 'computer', 'support', 'service', 'managed', 'network',
                 'soft', 'systems', 'solutions', 'consulting', 'digital', 'group'}
    
    commercial_av_words = {'av', 'audio', 'visual', 'media', 'video', 'sound', 
                          'integration', 'communications', 'conference', 'systems',
                          'display', 'projector', 'presentation', 'control'}
    
    residential_av_words = {'home', 'smart', 'automation', 'theater', 'cinema', 
                           'residential', 'house', 'living', 'entertainment'}
    
    manufacturing_words = {'steel', 'metal', 'forge', 'machine', 'factory', 'industrial',
                          'manufacturing', 'production', 'materials', 'supply', 'products',
                          'engineering', 'fabrication', 'construction'}
                          
    retail_words = {'shop', 'store', 'market', 'retail', 'buy', 'sale', 'price',
                   'product', 'goods', 'outlet', 'mall', 'shopping', 'discount'}
    
    # Calculate scores based on presence of words
    scores = {
        'msp_score': sum(1 for w in expanded_words if w.lower() in msp_words),
        'commercial_av_score': sum(1 for w in expanded_words if w.lower() in commercial_av_words),
        'residential_av_score': sum(1 for w in expanded_words if w.lower() in residential_av_words),
        'manufacturing_score': sum(1 for w in expanded_words if w.lower() in manufacturing_words),
        'retail_score': sum(1 for w in expanded_words if w.lower() in retail_words)
    }
    
    # Check for non-English words that might indicate industries
    # For example, "acciai" (Italian for steels)
    if "acciai" in domain_base:
        scores['manufacturing_score'] += 2
        logger.info(f"Detected 'acciai' (Italian for 'steels') in domain, boosting manufacturing score")
    
    # Normalize scores to 0-1 range if any scores exist
    total_score = sum(scores.values())
    if total_score > 0:
        normalized_scores = {k: v / total_score for k, v in scores.items()}
    else:
        normalized_scores = scores
    
    # Log the results
    logger.info(f"Domain analysis scores for {domain}: {normalized_scores}")
    
    return normalized_scores
