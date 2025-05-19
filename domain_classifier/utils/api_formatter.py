"""Improved API response formatting utilities."""
import logging
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def format_api_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the API response with numbered section markers to preserve order in PowerShell.
    
    Args:
        result: The original API response
        
    Returns:
        dict: The reformatted API response with properly ordered sections
    """
    # Initialize empty result
    formatted_result = {}
    
    # ========== DOMAIN INFO SECTION ==========
    formatted_result["_01_domain_info"] = "============ DOMAIN INFO ============"
    
    formatted_result["_01_domain"] = result.get("domain", "")
    formatted_result["_01_email"] = result.get("email", "")
    formatted_result["_01_website_url"] = result.get("website_url", "")
    
    # Add company name from either source
    company_name = result.get("company_name", "")
    if not company_name and "apollo_data" in result and isinstance(result["apollo_data"], dict):
        company_name = result["apollo_data"].get("name", "")
    if not company_name and "ai_company_data" in result and isinstance(result["ai_company_data"], dict):
        company_name = result["ai_company_data"].get("name", "")
    
    formatted_result["_01_company_name"] = company_name
    
    # ========== CLASSIFICATION SECTION ==========
    formatted_result["_02_classification"] = "============ CLASSIFICATION ============"
    
    formatted_result["_02_class"] = result.get("predicted_class", "")
    formatted_result["_02_final_classification"] = result.get("final_classification", "")
    formatted_result["_02_confidence_score"] = result.get("confidence_score", 0)
    formatted_result["_02_confidence_scores"] = result.get("confidence_scores", {})
    formatted_result["_02_is_parked"] = result.get("is_parked", False)
    formatted_result["_02_low_confidence"] = result.get("low_confidence", True)
    formatted_result["_02_detection_method"] = result.get("detection_method", "")
    formatted_result["_02_crawler_type"] = result.get("crawler_type", "")
    formatted_result["_02_classifier_type"] = result.get("classifier_type", "")
    formatted_result["_02_source"] = result.get("source", "")
    formatted_result["_02_explanation"] = result.get("explanation", "")
    
    # ========== AI DATA SECTION ==========
    formatted_result["_03_ai_data"] = "============ AI DATA ============"
    
    formatted_result["_03_ai_description"] = result.get("company_description", "")
    formatted_result["_03_ai_one_liner"] = result.get("company_one_line", "")
    
    # Add AI extracted structured data
    ai_data = result.get("ai_company_data", {})
    if isinstance(ai_data, str):
        try:
            import json
            ai_data = json.loads(ai_data)
        except:
            ai_data = {}
            
    if ai_data and isinstance(ai_data, dict):
        for key, value in ai_data.items():
            formatted_result[f"_03_ai_{key}"] = value
    
    # ========== APOLLO DATA SECTION ==========
    formatted_result["_04_apollo_data"] = "============ APOLLO DATA ============"
    
    apollo_data = result.get("apollo_data", {})
    if isinstance(apollo_data, str):
        try:
            import json
            apollo_data = json.loads(apollo_data)
        except:
            apollo_data = {}
            
    if apollo_data and isinstance(apollo_data, dict):
        for key, value in apollo_data.items():
            formatted_result[f"_04_apollo_{key}"] = value
    
    # ========== MERGED DATA SECTION ==========
    formatted_result["_05_merged_data"] = "============ MERGED DATA ============"
    
    # Define fields to merge with priority
    fields_to_merge = [
        "name", "industry", "employee_count", "founded_year", 
        "address", "city", "state", "country", "postal_code", "phone", "email",
        "revenue", "linkedin_url", "website", "funding"
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
            source = "ai"
            
        # Only add non-empty values
        if value not in [None, "", 0]:
            formatted_result[f"_05_merged_{field}"] = value
            formatted_result[f"_05_merged_{field}_source"] = source
    
    # ========== RECOMMENDATIONS SECTION ==========
    formatted_result["_06_recommendations"] = "============ RECOMMENDATIONS ============"
    
    recommendations = result.get("domotz_recommendations", {})
    if recommendations and isinstance(recommendations, dict):
        formatted_result["_06_rec_primary_value"] = recommendations.get("primary_value", "")
        formatted_result["_06_rec_recommended_plan"] = recommendations.get("recommended_plan", "")
        formatted_result["_06_rec_use_cases"] = recommendations.get("use_cases", [])
    
    # ========== ERROR HANDLING ==========
    if "error" in result:
        formatted_result["_07_error"] = "============ ERROR ============"
        formatted_result["_07_error_message"] = result["error"]
        
    if "error_type" in result:
        formatted_result["_07_error_type"] = result["error_type"]
        
    if "error_detail" in result:
        formatted_result["_07_error_detail"] = result["error_detail"]
    
    # Handle possible misclassification warnings
    if "possible_misclassification" in result and result["possible_misclassification"]:
        formatted_result["_07_possible_misclassification"] = True
        
        if "indicators_found" in result:
            formatted_result["_07_indicators_found"] = result["indicators_found"]
            
        if "misclassification_warning" in result:
            formatted_result["_07_misclassification_warning"] = result["misclassification_warning"]
    
    # Ensure any other important fields from the original result are preserved
    important_fields = [
        "is_anti_scraping", "is_ssl_error", "is_dns_error", "is_connection_error",
        "is_access_denied", "is_not_found", "is_server_error", "is_robots_restricted",
        "is_timeout", "is_crawl_error", "bulk_process_id"
    ]
    
    for field in important_fields:
        if field in result and result[field]:
            formatted_result[f"_99_{field}"] = result[field]
    
    return formatted_result
