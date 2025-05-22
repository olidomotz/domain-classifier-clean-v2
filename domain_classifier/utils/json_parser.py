"""Enhanced JSON parser that can handle Claude's malformed JSON responses."""

import logging
import json
import re
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def clean_json_string(json_str: str) -> str:
    """
    More aggressive cleaning of JSON strings from LLM responses.
    
    Args:
        json_str: The JSON string to clean
        
    Returns:
        str: The cleaned JSON string
    """
    logger.info("Using enhanced JSON cleaner with Claude-specific fixes")
    
    if not json_str:
        return json_str
    
    # Store original for debugging
    original = json_str
    
    # Remove any control characters
    json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
    
    # Fix escaped newlines and quotes
    json_str = re.sub(r'\\n', '\\n', json_str)
    json_str = re.sub(r'\\"', '"', json_str)
    json_str = re.sub(r'\\"', '"', json_str)
    
    # CRITICAL FIX: Handle Claude's specific format of double quotes inside quotes
    # Pattern 1: "predicted_class": " "Managed Service Provider"" 
    json_str = re.sub(r': " "([^"]+)"", ', r': "\1", ', json_str)
    json_str = re.sub(r': " "([^"]+)""}', r': "\1"}', json_str)
    
    # Pattern 2: "predicted_class": ""Managed Service Provider""
    json_str = re.sub(r': ""([^"]+)"", ', r': "\1", ', json_str)
    json_str = re.sub(r': ""([^"]+)""}', r': "\1"}', json_str)
    
    # Log if we found Claude's specific format
    if re.search(r': " "([^"]+)"', json_str) or re.search(r': ""([^"]+)"', json_str):
        logger.info("Detected and fixed Claude's double-quoted format")
        # Print the specific portion being fixed
        match = re.search(r': " "([^"]+)"', json_str) or re.search(r': ""([^"]+)"', json_str)
        if match:
            logger.info(f"Fixed pattern: {match.group(0)}")
    
    # Fix the problematic pattern: "score": X, 0
    json_str = re.sub(r'": (\d+),\s*0([,}])', r'": \1\2', json_str)
    
    # Fix numbers with commas (like "90, 0" should be "90")
    json_str = re.sub(r'(\d+),\s*(\d+)(?=,|\s|"|\]|}|$)', r'\1\2', json_str)
    
    # Fix JSON with wrapped quotation marks
    if json_str.startswith('"') and json_str.endswith('"'):
        try:
            # Try to parse as JSON-escaped string
            unescaped = json.loads(json_str)
            if isinstance(unescaped, str) and unescaped.startswith('{') and unescaped.endswith('}'):
                json_str = unescaped
                logger.info("Unwrapped JSON string wrapped in quotes")
        except Exception as e:
            logger.info(f"Failed to unwrap JSON: {e}")
            # If that fails, just strip the quotes directly
            if json_str.startswith('"') and json_str.endswith('"'):
                json_str = json_str[1:-1]
                logger.info("Manually stripped outer quotes from JSON string")
    
    # Fix missing quotes around property names
    json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
    
    # Fix trailing commas in objects/arrays
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    
    # Fix double colons
    json_str = re.sub(r'::', r':', json_str)
    
    # Fix missing quotes around string values for common fields
    for field in ["predicted_class", "detection_method", "llm_explanation", "company_description", "company_one_line"]:
        json_str = re.sub(
            r'"{0}":\s*([^"{{\[\d][^,}}]+)([,}}])'.format(field),
            r'"{0}": "\1"\2'.format(field),
            json_str
        )
    
    # Add special handling for internal_it_potential and max_confidence fields
    # Claude often misformatting these fields
    for field in ["internal_it_potential", "max_confidence", "confidence_score"]:
        # Handle float representation (0.X)
        json_str = re.sub(
            r'"{0}":\s*([0-9]\.[0-9]+)\s*,'.format(field),
            r'"{0}": \1,'.format(field),
            json_str
        )
        
        # Handle integer representation
        json_str = re.sub(
            r'"{0}":\s*([0-9]+)\s*,'.format(field),
            r'"{0}": \1,'.format(field),
            json_str
        )
    
    # Ultra-aggressive fix: If we still have Claude's double quotes pattern, do a more direct replacement
    if ' ""' in json_str or '" "' in json_str:
        # Pattern: "field": ""Value""
        json_str = re.sub(r'"([^"]+)":\s*""([^"]+)""', r'"\1": "\2"', json_str)
        # Pattern: "field": " "Value" "
        json_str = re.sub(r'"([^"]+)":\s*" "([^"]+)" "', r'"\1": "\2"', json_str)
        # Pattern: "field": " "Value""
        json_str = re.sub(r'"([^"]+)":\s*" "([^"]+)""', r'"\1": "\2"', json_str)
        logger.info("Applied ultra-aggressive double quotes fix")
    
    # Log significant changes
    if original != json_str:
        # Calculate the difference percentage
        diff_percentage = 100 - (len(json_str) / max(1, len(original)) * 100)
        logger.info(f"JSON cleaned (changed by {diff_percentage:.1f}%)")
        
        # Log more details
        if diff_percentage > 10:
            logger.info(f"Original JSON snippet: {original[:50]}...")
            logger.info(f"Cleaned JSON snippet: {json_str[:50]}...")
    
    # Final test: Try to parse it to see if our cleaning worked
    try:
        json.loads(json_str)
        logger.info("✅ JSON is now valid after cleaning")
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ JSON is still invalid after cleaning: {e}")
        logger.warning(f"Problem area: {json_str[max(0, e.pos-20):min(len(json_str), e.pos+20)]}")
    
    return json_str

def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON from text response with improved reliability.
    
    Args:
        text: The text to extract JSON from
        
    Returns:
        str: The extracted JSON string, or None if not found
    """
    if not text:
        return None
    
    # First try to find a complete, well-formed JSON object
    full_json_pattern = r'({[\s\S]*})'
    
    full_match = re.search(full_json_pattern, text)
    if full_match:
        json_str = full_match.group(1)
        
        # Try to parse it directly
        try:
            json.loads(json_str)
            return json_str  # If it parses, return it immediately
        except json.JSONDecodeError:
            # If it doesn't parse, continue to other patterns
            pass
    
    # Try multiple patterns to extract JSON
    json_patterns = [
        r'({[\s\S]*?"predicted_class"[\s\S]*?})',  # Most common pattern with predicted_class field
        r'```(?:json)?\s*({[\s\S]*?})\s*```',  # Markdown code blocks
        r'({[\s\S]*?"confidence_scores"[\s\S]*?})'  # Alternative key pattern
    ]
    
    for pattern in json_patterns:
        json_match = re.search(pattern, text, re.DOTALL)
        if json_match:
            candidate = json_match.group(1).strip()
            
            # Verify it at least starts and ends with {} and contains some internal content
            if candidate.startswith('{') and candidate.endswith('}') and len(candidate) > 10:
                return candidate
    
    # Look for partial JSON with key fields and try to complete it
    if '"predicted_class"' in text and '"confidence_scores"' in text:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx >= 0 and end_idx > start_idx:
            partial_json = text[start_idx:end_idx+1]
            return partial_json
    
    # If JSON extraction failed but the text contains JSON-like structure
    if '"predicted_class"' in text:
        logger.info("Attempting to extract classification from structured text")
        
        # Look for the predicted class using reliable pattern
        predicted_class_match = re.search(r'"predicted_class"\s*:\s*"([^"]+)"', text)
        if predicted_class_match:
            logger.info(f"Found predicted_class: {predicted_class_match.group(1)}")
            return f'{{"predicted_class": "{predicted_class_match.group(1)}"}}'
            
    return None

def extract_predicted_class_from_text(text: str) -> Optional[str]:
    """
    Extract predicted class directly from text when JSON parsing fails.
    
    Args:
        text: The raw text response from the LLM
        
    Returns:
        str: The extracted predicted class or None if not found
    """
    if not text:
        return None
    
    # First, try JSON-like pattern
    class_match = re.search(r'"predicted_class"\s*:\s*"([^"]+)"', text)
    if class_match:
        return class_match.group(1)
    
    # Try alternative JSON-like pattern without quotes
    alt_match = re.search(r'predicted_class["\\s:]*([A-Za-z\\s\\-]+)[",}]', text)
    if alt_match:
        return alt_match.group(1).strip()
    
    # Handle special case: double quoted values from Claude
    double_quote_match = re.search(r'"predicted_class"\s*:\s*"\s*"([^"]+)"\s*"', text)
    if double_quote_match:
        return double_quote_match.group(1)
    
    # Look for specific class mentions in common phrases
    class_types = [
        "Managed Service Provider",
        "Integrator - Commercial A/V",
        "Integrator - Residential A/V",
        "Internal IT Department",
        "Parked Domain"
    ]
    
    for class_type in class_types:
        patterns = [
            rf"classified as (?:a |an )?{class_type}",
            rf"is (?:a |an )?{class_type}",
            rf"appears to be (?:a |an )?{class_type}",
            rf"identified as (?:a |an )?{class_type}"
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return class_type
    
    # Check for specific MSP indicators
    msp_indicators = [
        r"provides (IT|technology) services to (other|multiple|various) (companies|businesses|organizations)",
        r"offers managed (IT|technology|service) to clients",
        r"managed service provider",
        r"provides managed (IT|tech|services)",
        r"IT service provider"
    ]
    
    for indicator in msp_indicators:
        if re.search(indicator, text, re.IGNORECASE):
            return "Managed Service Provider"
            
    return None

def parse_and_validate_json(json_str: str, context: str = "") -> Optional[Dict[str, Any]]:
    """
    Centralized function to parse and validate JSON with robust error handling.
    
    Args:
        json_str: The JSON string to parse
        context: Context information for logging (e.g., domain name)
        
    Returns:
        dict: The parsed JSON as a dictionary, or None if parsing failed
    """
    if not json_str:
        logger.warning(f"Empty JSON string provided for parsing {context}")
        return None
        
    try:
        # First attempt with standard JSON parsing
        try:
            parsed_json = json.loads(json_str)
            logger.info(f"Successfully parsed JSON {context}")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed {context}: {e}")
            
            # Try with cleaning
            cleaned_json = clean_json_string(json_str)
            try:
                parsed_json = json.loads(cleaned_json)
                logger.info(f"Successfully parsed JSON after cleaning {context}")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed after cleaning {context}: {e}")
                
                # Try ultra-aggressive cleaning as a last resort
                ultra_clean = re.sub(r'[^\x20-\x7E]', '', cleaned_json)
                try:
                    parsed_json = json.loads(ultra_clean)
                    logger.info(f"Successfully parsed JSON after ultra cleaning {context}")
                    return parsed_json
                except json.JSONDecodeError as e:
                    # Log detailed error information for debugging
                    position = e.pos if hasattr(e, 'pos') else -1
                    if position >= 0:
                        context_start = max(0, position - 30)
                        context_end = min(len(cleaned_json), position + 30)
                        error_context = cleaned_json[context_start:context_end]
                        logger.error(f"Final JSON parsing failed {context} at position {position}: {e}")
                        logger.error(f"Context near error: '{error_context}'")
                    else:
                        logger.error(f"Final JSON parsing failed {context}: {e}")
                        
                    return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON {context}: {e}")
        return None

def ensure_dict(data: Dict[str, Any], field_name: str = "unknown") -> Dict[str, Any]:
    """
    Ensure that data is a dictionary, parsing it if it's a string.
    
    Args:
        data: The data to process
        field_name: Name of the field for logging purposes
        
    Returns:
        dict: The data as a dictionary
    """
    if data is None:
        return {}
        
    if isinstance(data, dict):
        return data
        
    if isinstance(data, str):
        return parse_and_validate_json(data, context=field_name) or {}
        
    # For any other type, return an empty dict
    logger.warning(f"{field_name} is not a dict or string, but {type(data)}")
    return {}

def safe_get(data: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary-like object.
    
    Args:
        data: The dictionary-like object
        key: The key to retrieve
        default: Default value if key is not found
        
    Returns:
        Any: The value or default
    """
    # First ensure data is a dictionary
    data_dict = ensure_dict(data)
    
    # Then safely get the value
    return data_dict.get(key, default)
