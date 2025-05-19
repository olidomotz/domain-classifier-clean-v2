"""Cache management for domain classification results."""
import logging
import json
import re
from typing import Dict, Any, Optional

# Import final classification utility
from domain_classifier.utils.final_classification import determine_final_classification
# Import text processing utilities
from domain_classifier.utils.text_processing import generate_one_line_description
# Import JSON utilities
from domain_classifier.utils.json_utils import ensure_dict, safe_get

# Set up logging
logger = logging.getLogger(__name__)

# In-memory cache for domain results (used when Snowflake is unavailable)
_domain_cache = {}

def cache_result(domain: str, result: Dict[str, Any]) -> None:
    """
    Cache a domain classification result in memory.
    
    Args:
        domain (str): The domain being classified
        result (Dict[str, Any]): The classification result
    """
    logger.info(f"Caching result for domain: {domain}")
    _domain_cache[domain.lower()] = result
    
def get_cached_result(domain: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a cached domain classification result.
    
    Args:
        domain (str): The domain to look up
        
    Returns:
        Optional[Dict[str, Any]]: The cached result or None if not found
    """
    domain_key = domain.lower()
    if domain_key in _domain_cache:
        logger.info(f"Found cached result for domain: {domain}")
        return _domain_cache[domain_key]
    logger.info(f"No cached result found for domain: {domain}")
    return None

def process_cached_result(record: Dict[str, Any], domain: str, email: Optional[str] = None, 
                          url: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a cached record from Snowflake to return a standardized result.
    
    Args:
        record (Dict[str, Any]): The record from Snowflake
        domain (str): The domain being classified
        email (Optional[str]): Optional email address
        url (Optional[str]): Optional website URL
        
    Returns:
        Dict[str, Any]: The processed result
    """
    logger.info(f"Processing cached record for domain: {domain}")
    
    # Extract confidence scores
    confidence_scores = {}
    try:
        confidence_scores = json.loads(record.get('ALL_SCORES', '{}'))
    except Exception as e:
        logger.warning(f"Could not parse ALL_SCORES for {domain}: {e}")
    
    # Extract LLM explanation from the LLM_EXPLANATION column
    llm_explanation = record.get('LLM_EXPLANATION', '')
    
    # If LLM_EXPLANATION is not available, try to get it from model_metadata
    if not llm_explanation:
        try:
            metadata = json.loads(record.get('MODEL_METADATA', '{}'))
            llm_explanation = metadata.get('llm_explanation', '')
        except Exception as e:
            logger.warning(f"Could not parse model_metadata for {domain}: {e}")
    
    # Extract cached fields we need to preserve for end of result
    crawler_type = record.get('CRAWLER_TYPE')  
    classifier_type = record.get('CLASSIFIER_TYPE')
            
    # Extract Apollo data if available
    apollo_company_data = None
    apollo_person_data = None
    
    try:
        if record.get('APOLLO_COMPANY_DATA'):
            apollo_data = record.get('APOLLO_COMPANY_DATA')
            # Use the ensure_dict function from json_utils
            apollo_company_data = ensure_dict(apollo_data, "apollo_data")
            logger.info(f"Found cached Apollo company data for {domain}")
    except Exception as e:
        logger.warning(f"Could not parse APOLLO_COMPANY_DATA for {domain}: {e}")
        apollo_company_data = {}
    
    try:
        if record.get('APOLLO_PERSON_DATA'):
            person_data = record.get('APOLLO_PERSON_DATA')
            # Use the ensure_dict function from json_utils
            apollo_person_data = ensure_dict(person_data, "apollo_person_data")
            logger.info(f"Found cached Apollo person data for {domain}")
    except Exception as e:
        logger.warning(f"Could not parse APOLLO_PERSON_DATA for {domain}: {e}")
        apollo_person_data = {}
    
    # Ensure we have an explanation
    if not llm_explanation:
        llm_explanation = f"The domain {domain} was previously classified as a {record.get('COMPANY_TYPE')} based on analysis of website content."
    
    # Add low_confidence flag based on confidence score
    confidence_score = record.get('CONFIDENCE_SCORE', 0.5)
    low_confidence = record.get('LOW_CONFIDENCE', confidence_score < 0.7)
    
    # Check if it's a parked domain
    is_parked = record.get('COMPANY_TYPE') == "Parked Domain"
    
    # Handle legacy "Corporate IT" and "Non-Service Business"
    company_type = record.get('COMPANY_TYPE', 'Unknown')
    if company_type == "Non-Service Business":
        company_type = "Internal IT Department"
    
    # Create the standardized result - MODIFIED: Set basic fields directly at start
    result = {
        "domain": domain,
        "email": email,
        "website_url": url,
        "predicted_class": company_type,
        "confidence_score": int(confidence_score * 100) if isinstance(confidence_score, float) and confidence_score <= 1.0 else int(confidence_score),
        "confidence_scores": confidence_scores,
        "explanation": llm_explanation,
        "low_confidence": low_confidence,
        "detection_method": record.get('DETECTION_METHOD', 'api'),
        "source": "cached",
        "is_parked": is_parked
    }
    
    # Add company description if available in the record
    if record.get('COMPANY_DESCRIPTION'):
        result["company_description"] = record.get('COMPANY_DESCRIPTION')
    
    # Add one-line company description
    # First check if there's one in the record
    if record.get('COMPANY_ONE_LINE'):
        result["company_one_line"] = record.get('COMPANY_ONE_LINE')
    else:
        # Generate a new one-line description
        result["company_one_line"] = generate_one_line_description(
            content="",  # We don't have content here
            predicted_class=company_type,
            domain=domain,
            company_description=result.get("explanation", "")
        )
    
    # Add Apollo data if available (as a dictionary)
    if apollo_company_data and isinstance(apollo_company_data, dict):
        result["apollo_data"] = apollo_company_data
        
        # Add company_name field from Apollo data if available
        if "name" in apollo_company_data and apollo_company_data["name"]:
            result["company_name"] = apollo_company_data["name"]
    
    if apollo_person_data and isinstance(apollo_person_data, dict):
        result["apollo_person_data"] = apollo_person_data
    
    # If no company_name has been set yet, use domain-derived name
    if "company_name" not in result:
        result["company_name"] = domain.split('.')[0].capitalize()
    
    # NEW: Perform cross-validation for cached results if we have Apollo or AI data
    if apollo_company_data and isinstance(apollo_company_data, dict):
        try:
            from domain_classifier.utils.cross_validator import reconcile_classification
            original_class = result.get("predicted_class")
            result = reconcile_classification(result, apollo_company_data, None)
            
            if original_class != result.get("predicted_class"):
                logger.warning(f"Cross-validation changed classification for {domain} from {original_class} to {result.get('predicted_class')}")
                
                # If classification changed, update the description to avoid fabricated details
                if "integrator" in original_class.lower() and "Internal IT Department" == result.get("predicted_class"):
                    # Get company name
                    company_name = result.get("company_name", domain.split('.')[0].capitalize())
                    
                    # Check the industry from Apollo data
                    industry = apollo_company_data.get("industry", "").lower()
                    
                    # Special handling for maritime industry
                    is_maritime = 'maritime' in industry or 'shipping' in industry or 'vessel' in industry
                    
                    if is_maritime:
                        # Create maritime-specific description
                        employee_count = apollo_company_data.get("employee_count", "")
                        new_description = f"{company_name} is a maritime industry company"
                        if employee_count:
                            new_description += f" with approximately {employee_count} employees"
                        new_description += ". The company operates in the shipping sector, providing maritime services and supplies."
                        
                        result["company_description"] = new_description
                        result["company_one_line"] = f"{company_name} provides maritime shipping services and supplies."
                    
                    elif industry:
                        # Generic industry-based description
                        employee_count = apollo_company_data.get("employee_count", "")
                        new_description = f"{company_name} is a {industry} company"
                        if employee_count:
                            new_description += f" with approximately {employee_count} employees"
                        new_description += "."
                        
                        result["company_description"] = new_description
                        result["company_one_line"] = f"{company_name} is a {industry} company with internal IT needs."
                    
                    else:
                        # Generic description with no specific industry
                        new_description = f"{company_name} is a business with internal IT needs, not an audio-visual integrator."
                        result["company_description"] = new_description
                        result["company_one_line"] = f"{company_name} is a business with internal IT needs."
        except Exception as e:
            logger.warning(f"Error during cross-validation for cached result: {e}")
    
    # Generate recommendations if Apollo data is available
    if apollo_company_data and isinstance(apollo_company_data, dict):
        try:
            from domain_classifier.enrichment.recommendation_engine import DomotzRecommendationEngine
            recommendation_engine = DomotzRecommendationEngine()
            recommendations = recommendation_engine.generate_recommendations(
                result.get("predicted_class"),  # Use potentially updated class
                apollo_company_data
            )
            result["domotz_recommendations"] = recommendations
            logger.info(f"Generated recommendations from cached Apollo data for {domain}")
        except Exception as e:
            logger.warning(f"Could not generate recommendations from cached data: {e}")
    
    # REMOVED: Don't add email and URL conditionally - they're set at the beginning
    # Add email and URL if provided
    # if email:
    #     result["email"] = email
    
    # if url:
    #     result["website_url"] = url
    
    # Add error_type if present in record
    if record.get('ERROR_TYPE'):
        result["error_type"] = record.get('ERROR_TYPE')
        
    # Determine and add the final classification
    result["final_classification"] = determine_final_classification(result)
    logger.info(f"Added final classification: {result['final_classification']} for {domain}")
    
    # Add crawler_type and classifier_type at the end to ensure they appear at the bottom
    if crawler_type:
        result["crawler_type"] = crawler_type
    else:
        result["crawler_type"] = "cached_unknown"
        
    if classifier_type:
        result["classifier_type"] = classifier_type
    else:
        result["classifier_type"] = "cached_unknown"
    
    return result

def clear_cache():
    """Clear the in-memory cache."""
    global _domain_cache
    logger.info("Clearing in-memory cache")
    _domain_cache = {}
