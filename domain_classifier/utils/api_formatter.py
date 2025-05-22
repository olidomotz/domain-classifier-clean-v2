"""API response formatting utilities."""

import logging
import json
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def format_api_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the API response for better PowerShell display and n8n access.
    Optimized for reliability and performance.
    
    Args:
        result: The original API response
        
    Returns:
        dict: The reformatted API response
    """
    logger.info(f"Input to format_api_response - keys: {list(result.keys())}")
    
    # For debugging, log core fields if available
    logger.info(f"Domain: {result.get('domain')}, Email: {result.get('email')}, URL: {result.get('website_url')}")
    
    # For n8n, we want a flat structure with prefixed fields
    flat_result = {}
    
    # Special helper function to handle potentially inconsistent key names
    def safe_get(result, key, alternative_keys=None, default=None):
        """Safely get a value from result using key or alternative_keys."""
        if key in result and result[key] not in [None, ""]:
            return result[key]
        if alternative_keys:
            for alt_key in alternative_keys:
                if alt_key in result and result[alt_key] not in [None, ""]:
                    return result[alt_key]
        return default
    
    # ========== UTILITY FUNCTION ==========
    def add_section(prefix, title, fields_dict):
        """Add a section to the flat result with the given prefix and title."""
        # Add section marker with "aaa" to ensure it appears first
        section_key = f"{prefix}_aaa_section"
        flat_result[section_key] = "=" * 20 + f" {title} " + "=" * 20
        
        # Add fields
        for key, value in fields_dict.items():
            if value not in [None, "", 0, False] or key in ["domain", "email", "website_url", "classification", 
                                                          "final_classification", "company_description"]:
                flat_result[f"{prefix}_{key}"] = value
    
    # ========== DOMAIN INFO SECTION ==========
    domain_info = {
        "domain": safe_get(result, "domain"),
        "email": safe_get(result, "email"),
        "website_url": safe_get(result, "website_url"),
    }
    
    # Add company name from either source
    company_name = safe_get(result, "company_name")
    if not company_name and "apollo_data" in result and isinstance(result["apollo_data"], dict):
        company_name = result["apollo_data"].get("name", "")
    if not company_name and "ai_company_data" in result and isinstance(result["ai_company_data"], dict):
        company_name = result["ai_company_data"].get("name", "")
        
    domain_info["company_name"] = company_name
    
    add_section("01", "DOMAIN INFO", domain_info)
    
    # ========== CLASSIFICATION SECTION ==========
    classification_info = {
        "classification": safe_get(result, "predicted_class", ["classification"], ""),
        "final_classification": safe_get(result, "final_classification", [], ""),
        "confidence_score": safe_get(result, "confidence_score", ["max_confidence"], 0),
        "confidence_scores": safe_get(result, "confidence_scores", {}, {}),
        "is_parked": safe_get(result, "is_parked", [], False),
        "low_confidence": safe_get(result, "low_confidence", [], True),
        "detection_method": safe_get(result, "detection_method", [], ""),
        "crawler_type": safe_get(result, "crawler_type", [], ""),
        "classifier_type": safe_get(result, "classifier_type", [], ""),
        "source": safe_get(result, "source", [], ""),
        "explanation": safe_get(result, "explanation", ["llm_explanation"], "")
    }
    
    # Log any missing required fields for debugging
    required_fields = ["classification", "final_classification", "confidence_score", "crawler_type", "classifier_type"]
    for field in required_fields:
        if not classification_info[field]:
            logger.warning(f"Critical field 01_{field} is missing from filtered result!")
    
    add_section("02", "CLASSIFICATION", classification_info)
    
    # ========== AI DATA SECTION ==========
    ai_info = {
        "description": safe_get(result, "company_description"),
        "one_liner": safe_get(result, "company_one_line")
    }
    
    # Add AI extracted data
    ai_data = safe_get(result, "ai_company_data", {})
    if isinstance(ai_data, str):
        try:
            ai_data = json.loads(ai_data)
        except:
            ai_data = {}
            
    if ai_data and isinstance(ai_data, dict):
        for key, value in ai_data.items():
            if value not in [None, "", 0]:
                ai_info[key] = value
    
    # CRITICAL: Make sure company_description is included even if it's coming from another source
    if "company_description" in result and result["company_description"] and "description" not in ai_info:
        ai_info["description"] = result["company_description"]
    
    add_section("03", "AI DATA", ai_info)
    
    # ========== APOLLO DATA SECTION ==========
    apollo_info = {}
    
    apollo_data = safe_get(result, "apollo_data", {})
    if isinstance(apollo_data, str):
        try:
            apollo_data = json.loads(apollo_data)
        except:
            apollo_data = {}
            
    if apollo_data and isinstance(apollo_data, dict):
        # Include ALL fields from Apollo, not just a subset
        for key, value in apollo_data.items():
            if value not in [None, "", 0]:
                apollo_info[key] = value
    
    add_section("04", "APOLLO DATA", apollo_info)
    
    # ========== MERGED DATA SECTION ==========
    merged_info = {}
    
    # Define fields to merge with priority - expand this list to include new Apollo fields
    fields_to_merge = [
        "name", "industry", "employee_count", "founded_year", 
        "address", "city", "state", "country", "postal_code", "phone", "email",
        "revenue", "linkedin_url", "website", "funding", "primary_domain",
        "description", "technologies", "specialties", "short_description", 
        "long_description", "keywords", "tags", "organization_type",
        "company_type", "social_links"
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
    
    recommendations = safe_get(result, "domotz_recommendations", {})
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
    
    # Add company_description again to ensure it shows up in output
    if "company_description" in result and result["company_description"]:
        flat_result["company_description"] = result["company_description"]
    
    return flat_result
