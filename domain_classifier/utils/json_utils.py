"""JSON utilities for data processing."""
import logging
from typing import Dict, Any, Optional

# Import from the centralized json_parser module
from domain_classifier.utils.json_parser import ensure_dict, safe_get

# Set up logging
logger = logging.getLogger(__name__)

# Export the imported functions
__all__ = ['ensure_dict', 'safe_get']
