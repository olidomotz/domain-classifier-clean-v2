"""Improved API response formatting utilities."""
import logging
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def format_api_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the API response into a structured format with clear sections.
    
    Args:
        result: The original API response
        
    Returns:
        dict: The reformatted API response
    """
    # Initialize the formatted result
    formatted_result = {
        "domain_info": {
            "domain": result.get("domain", ""),
            "email": result.get("email", ""),
            "website_url": result.get("website_url", "")
        },
        "classification": {
            "predicted_class": result.get("predicted_class", ""),
            "final_classification": result.get("final_classification", ""),
            "confidence_score": result.get("confidence_score", 0),
            "confidence_scores": result.get("confidence_scores", {}),
            "is_parked": result.get("is_parked", False),
            "low_confidence": result.get("low_confidence", True),
            "detection_method": result.get("detection_method", ""),
            "crawler_type": result.get("crawler_type", ""),
            "classifier_type": result.get("classifier_type", ""),
            "source": result.get("source", ""),
            "explanation": result.get("explanation", "")
        },
        "ai_company_data": {
            "company_name": result.get("company_name", ""),
            "company_description": result.get("company_description", ""),
            "company_one_line": result.get("company_one_line", "")
        },
        "apollo_data": result.get("apollo_data", {}),
        "ai_extraction": result.get("ai_company_data", {}),
        "merged_data": {},
        "recommendations": result.get("domotz_recommendations", {})
    }
    
    # Fix for error cases - make sure error information is propagated
    if "error" in result:
        formatted_result["error"] = result["error"]
        
    if "error_type" in result:
        formatted_result["error_type"] = result["error_type"]
        
    if "error_detail" in result:
        formatted_result["error_detail"] = result["error_detail"]
    
    # Create the merged data section by combining Apollo and AI data
    merged_data = {}
    apollo_data = result.get("apollo_data", {})
    ai_data = result.get("ai_company_data", {})
    
    # Handle potential JSON strings
    if isinstance(apollo_data, str):
        try:
            import json
            apollo_data = json.loads(apollo_data)
        except:
            apollo_data = {}
    
    if isinstance(ai_data, str):
        try:
            import json
            ai_data = json.loads(ai_data)
        except:
            ai_data = {}
    
    # Prioritize fields for merging
    fields_to_merge = [
        "name", "industry", "employee_count", "founded_year", 
        "address", "city", "state", "country", "postal_code", "phone", "email"
    ]
    
    for field in fields_to_merge:
        # Start with empty value
        value = None
        source = None
        
        # Always prefer Apollo data if available
        if apollo_data and isinstance(apollo_data, dict) and field in apollo_data and apollo_data[field]:
            value = apollo_data[field]
            source = "apollo"
        # Use AI data as fallback
        elif ai_data and isinstance(ai_data, dict) and field in ai_data and ai_data[field]:
            value = ai_data[field]
            source = "ai_extraction"
            
        # Only add non-empty values
        if value not in [None, "", 0]:
            merged_data[field] = {
                "value": value,
                "source": source
            }
    
    # Add additional fields that might be only in one source
    if apollo_data and isinstance(apollo_data, dict):
        if "revenue" in apollo_data and apollo_data["revenue"]:
            merged_data["revenue"] = {
                "value": apollo_data["revenue"],
                "source": "apollo"
            }
        if "linkedin_url" in apollo_data and apollo_data["linkedin_url"]:
            merged_data["linkedin_url"] = {
                "value": apollo_data["linkedin_url"],
                "source": "apollo"
            }
        if "technologies" in apollo_data and apollo_data["technologies"]:
            merged_data["technologies"] = {
                "value": apollo_data["technologies"],
                "source": "apollo"
            }
        
        # Add website field if present
        if "website" in apollo_data and apollo_data["website"]:
            merged_data["website"] = {
                "value": apollo_data["website"],
                "source": "apollo"
            }
        
        # Add funding information if present
        if "funding" in apollo_data and apollo_data["funding"]:
            merged_data["funding"] = {
                "value": apollo_data["funding"],
                "source": "apollo"
            }
    
    # Set the merged data
    formatted_result["merged_data"] = merged_data
    
    # Handle possible misclassification warnings
    if "possible_misclassification" in result and result["possible_misclassification"]:
        formatted_result["classification"]["possible_misclassification"] = True
        
        if "indicators_found" in result:
            formatted_result["classification"]["indicators_found"] = result["indicators_found"]
            
        if "misclassification_warning" in result:
            formatted_result["classification"]["misclassification_warning"] = result["misclassification_warning"]
    
    # Ensure any other important fields from the original result are preserved
    # This handles custom fields that might be added in the future
    important_fields = [
        "is_anti_scraping", "is_ssl_error", "is_dns_error", "is_connection_error",
        "is_access_denied", "is_not_found", "is_server_error", "is_robots_restricted",
        "is_timeout", "is_crawl_error", "bulk_process_id"
    ]
    
    for field in important_fields:
        if field in result and result[field]:
            formatted_result[field] = result[field]
    
    # Move company_name to a more logical place (from ai_company_data to domain_info)
    if formatted_result["ai_company_data"]["company_name"]:
        formatted_result["domain_info"]["company_name"] = formatted_result["ai_company_data"]["company_name"]
    
    return formatted_result
