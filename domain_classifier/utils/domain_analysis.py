"""Domain name analysis utilities."""

import re
import logging
from typing import Dict, Any, List

# Set up logging
logger = logging.getLogger(__name__)

def analyze_domain_words(domain: str) -> Dict[str, float]:
    """
    Analyze domain name words for classification signals.
    Enhanced to better detect IT solutions and MSP patterns.

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
                'soft', 'systems', 'solutions', 'consulting', 'digital', 'group',
                'security', 'protect', 'backup', 'monitor', 'remote', 'consult',
                'help', 'desk', 'services', 'provider', 'server', 'itsupport',
                'online', 'compute', 'devops', 'infra', 'telecom', 'technologies',
                'fix', 'pro', 'expert', 'wizard'}
    
    commercial_av_words = {'av', 'audio', 'visual', 'media', 'video', 'sound',
                         'integration', 'communications', 'conference', 'systems',
                         'display', 'projector', 'presentation', 'control', 
                         'boardroom', 'meeting', 'professional', 'stage',
                         'studio', 'broadcast', 'projection', 'collaborate',
                         'virtual', 'av', 'screens', 'mic', 'speakers'}
    
    residential_av_words = {'home', 'smart', 'automation', 'theater', 'cinema',
                          'residential', 'house', 'living', 'entertainment',
                          'family', 'comfort', 'lifestyle', 'room', 'lighting',
                          'security', 'control', 'connected', 'automate', 'scene',
                          'media', 'smarthome', 'homecontrol', 'homesystem'}
    
    manufacturing_words = {'steel', 'metal', 'forge', 'machine', 'factory', 'industrial',
                         'manufacturing', 'production', 'materials', 'supply', 'products',
                         'engineering', 'fabrication', 'construction', 'build', 'hardware',
                         'tool', 'equipment', 'parts', 'assemble', 'industry'}
    
    retail_words = {'shop', 'store', 'market', 'retail', 'buy', 'sale', 'price',
                  'product', 'goods', 'outlet', 'mall', 'shopping', 'discount',
                  'commerce', 'marketplace', 'purchase', 'sell', 'deals', 'bargain',
                  'ecommerce', 'order', 'cart', 'checkout', 'store'}
    
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
    
    # Check for IT solution patterns that original analysis might miss
    domain_lower = domain.lower()
    
    # IT Solutions/MSP patterns - comprehensive list
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
        r'(it|tech|computer)\s*(service|support)',
        r'(managed|remote)\s*(service|monitor)',
        r'cloud\s*(hosting|provider|service)',
        r'data\s*(backup|recovery|protection)',
        r'network\s*(admin|management|monitoring)'
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
    
    # Check for commercial AV patterns
    av_patterns = [
        r'(audio|visual|media)\s*(solution|system|integration)',
        r'(av|a/v)\s*(integration|solution|system)',
        r'(conference|meeting)\s*(room|system)',
        r'(digital)\s*(signage|display)',
        r'(presentation|projection)\s*(system|solution)'
    ]
    
    for pattern in av_patterns:
        if re.search(pattern, domain_lower):
            logger.info(f"Domain {domain} contains Commercial AV indicator pattern: {pattern}")
            
            # Set Commercial AV score high
            scores['commercial_av_score'] = 0.9
            break
    
    # Check for residential AV patterns
    home_patterns = [
        r'(home|smart)\s*(automation|theater|entertainment)',
        r'(residential|home)\s*(av|a/v|audio|visual)',
        r'(smart|connected)\s*(home|house|living)'
    ]
    
    for pattern in home_patterns:
        if re.search(pattern, domain_lower):
            logger.info(f"Domain {domain} contains Residential AV indicator pattern: {pattern}")
            
            # Set Residential AV score high
            scores['residential_av_score'] = 0.9
            break
    
    # Normalize scores to 0-1 range if any scores exist
    total_score = sum(scores.values())
    
    if total_score > 0:
        normalized_scores = {k: v / total_score for k, v in scores.items()}
    else:
        normalized_scores = scores
    
    # Log the results
    logger.info(f"Domain analysis scores for {domain}: {normalized_scores}")
    
    return normalized_scores
