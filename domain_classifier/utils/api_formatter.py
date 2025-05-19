"""Improved API response formatting utilities."""
import logging
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def format_api_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the API response for better PowerShell display and n8n access.
    
    Args:
        result: The original API response
        
    Returns:
        dict: The reformatted API response
    """
    # Debug logging for input tracking
    logger.info(f"Input to format_api_response - keys: {list(result.keys())}")
    logger.info(f"Domain: {result.get('domain')}, Email: {result.get('email')}, URL: {result.get('website_url')}")
    
    # For n8n, we want a flat structure with prefixed fields
    flat_result = {}
    
    # ========== UTILITY FUNCTION ==========
    def add_section(prefix, title, fields_dict):
        """Add a section to the flat result with the given prefix and title."""
        # Add section marker with "aaa" to ensure it appears first
        section_key = f"{prefix}_aaa_section"
        flat_result[section_key] = "=" * 20 + f" {title} " + "=" * 20
        
        # Add fields - MODIFIED: Always add critical fields regardless of value
        critical_fields = ["domain", "email", "website_url", "classification", "final_classification", 
                         "confidence_score", "crawler_type", "classifier_type"]
        
        for key, value in fields_dict.items():
            # Always include critical fields or non-empty values
            if value not in [None, "", 0] or key in critical_fields:
                flat_result[f"{prefix}_{key}"] = value
    
    # ========== DOMAIN INFO SECTION ==========
    domain_info = {
        "domain": result.get("domain", ""),
        "email": result.get("email", ""),
        "website_url": result.get("website_url", ""),
    }
    
    # Add company name from either source
    company_name = result.get("company_name", "")
    if not company_name and "apollo_data" in result and isinstance(result["apollo_data"], dict):
        company_name = result["apollo_data"].get("name", "")
    if not company_name and "ai_company_data" in result and isinstance(result["ai_company_data"], dict):
        company_name = result["ai_company_data"].get("name", "")
    
    domain_info["company_name"] = company_name
    add_section("01", "DOMAIN INFO", domain_info)
    
    # ========== CLASSIFICATION SECTION ==========
    classification_info = {
        "classification": result.get("predicted_class", ""),
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
    }
    add_section("02", "CLASSIFICATION", classification_info)
    
    # ========== AI DATA SECTION ==========
    ai_info = {
        "description": result.get("company_description", ""),
        "one_liner": result.get("company_one_line", "")
    }
    
    # Add AI extracted data
    ai_data = result.get("ai_company_data", {})
    if isinstance(ai_data, str):
        try:
            import json
            ai_data = json.loads(ai_data)
        except:
            ai_data = {}
            
    if ai_data and isinstance(ai_data, dict):
        for key, value in ai_data.items():
            ai_info[key] = value
    
    add_section("03", "AI DATA", ai_info)
    
    # ========== APOLLO DATA SECTION ==========
    apollo_info = {}
    apollo_data = result.get("apollo_data", {})
    
    if isinstance(apollo_data, str):
        try:
            import json
            apollo_data = json.loads(apollo_data)
        except:
            apollo_data = {}
            
    if apollo_data and isinstance(apollo_data, dict):
        for key, value in apollo_data.items():
            apollo_info[key] = value
    
    add_section("04", "APOLLO DATA", apollo_info)
    
    # ========== MERGED DATA SECTION ==========
    merged_info = {}
    
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
            merged_info[field] = value
            merged_info[f"{field}_source"] = source
    
    add_section("05", "MERGED DATA", merged_info)
    
    # ========== RECOMMENDATIONS SECTION ==========
    rec_info = {}
    recommendations = result.get("domotz_recommendations", {})
    if recommendations and isinstance(recommendations, dict):
        rec_info["primary_value"] = recommendations.get("primary_value", "")
        rec_info["recommended_plan"] = recommendations.get("recommended_plan", "")
        rec_info["use_cases"] = recommendations.get("use_cases", [])
    
    add_section("06", "RECOMMENDATIONS", rec_info)
    
    # ========== ERROR HANDLING ==========
    if "error" in result:
        error_info = {
            "error": result["error"]
        }
        
        if "error_type" in result:
            error_info["error_type"] = result["error_type"]
            
        if "error_detail" in result:
            error_info["error_detail"] = result["error_detail"]
        
        add_section("07", "ERROR", error_info)
    
    # Handle possible misclassification warnings
    if "possible_misclassification" in result and result["possible_misclassification"]:
        misclass_info = {
            "possible_misclassification": True
        }
        
        if "indicators_found" in result:
            misclass_info["indicators_found"] = result["indicators_found"]
            
        if "misclassification_warning" in result:
            misclass_info["misclassification_warning"] = result["misclassification_warning"]
        
        add_section("08", "MISCLASSIFICATION WARNING", misclass_info)
    
    # Ensure any other important fields from the original result are preserved
    important_fields = [
        "is_anti_scraping", "is_ssl_error", "is_dns_error", "is_connection_error",
        "is_access_denied", "is_not_found", "is_server_error", "is_robots_restricted",
        "is_timeout", "is_crawl_error", "bulk_process_id"
    ]
    
    other_info = {}
    for field in important_fields:
        if field in result and result[field]:
            other_info[field] = result[field]
    
    if other_info:
        add_section("09", "OTHER INFO", other_info)
    
    # Filter out empty fields, but ALWAYS keep critical fields
    filtered_result = {}
    critical_field_suffixes = ["domain", "email", "website_url", "classification", 
                              "final_classification", "confidence_score", 
                              "crawler_type", "classifier_type"]
    
    for key, value in flat_result.items():
        if (value not in [None, "", 0] or 
            "section" in key or 
            any(key.endswith(f"_{suffix}") for suffix in critical_field_suffixes)):
            filtered_result[key] = value
    
    # Final debug check before returning
    for suffix in critical_field_suffixes:
        key = f"01_{suffix}"
        if key not in filtered_result:
            logger.warning(f"Critical field {key} is missing from filtered result!")
    
    return filtered_result
