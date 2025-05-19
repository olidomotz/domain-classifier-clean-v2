"""JSON utilities for data processing."""
import json
import logging
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def ensure_dict(data: Any, field_name: str = "unknown") -> Dict[str, Any]:
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
        try:
            parsed_data = json.loads(data)
            if isinstance(parsed_data, dict):
                logger.info(f"Successfully parsed {field_name} string to dictionary")
                return parsed_data
            else:
                logger.warning(f"{field_name} was parsed but is not a dictionary: {type(parsed_data)}")
                return {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse {field_name} as JSON: {data[:50]}...")
            return {}
        except Exception as e:
            logger.warning(f"Unexpected error parsing {field_name} as JSON: {e}")
            return {}
            
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
