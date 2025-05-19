"""Process classification results for API responses."""
import logging
import json
import re  # Added import for regex operations
from typing import Dict, Any, Optional

# Import final classification utility
from domain_classifier.utils.final_classification import determine_final_classification
# Import text processing utilities
from domain_classifier.utils.text_processing import generate_one_line_description

# Set up logging
logger = logging.getLogger(__name__)

def process_fresh_result(classification: Dict[str, Any], domain: str, email: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a fresh classification result.
    
    Args:
        classification: The classification result from the classifier
        domain: The domain name
        email: Optional email address
        url: Optional URL for clickable link
        
    Returns:
        dict: The processed result ready for the client
    """
    try:
        # ADDED: Always set these critical fields at the beginning
        result = {
            "domain": domain,
            "email": email,
            "website_url": url
        }
        
        if classification.get("is_parked", False):
            # Special case for parked domains
            result.update({
                "predicted_class": "Parked Domain",
                "confidence_score": 0,
                "confidence_scores": {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0,
                    "Internal IT Department": 0
                },
                "explanation": classification.get('llm_explanation', 'This appears to be a parked or inactive domain without business-specific content.'),
                "company_description": classification.get('company_description', f"{domain} appears to be a parked or inactive domain with no active business."),
                "low_confidence": True,
                "detection_method": classification.get('detection_method', 'parked_domain_detection'),
                "source": "fresh",
                "is_parked": True
            })
            
            # Add one-line company description for parked domains
            result["company_one_line"] = classification.get('company_one_line', f"{domain} is a parked domain with no active business.")
            
            # These fields will be added later to ensure they appear at the bottom
            preserved_crawler_type = classification.get("crawler_type", None)
            preserved_classifier_type = classification.get("classifier_type", None)
            
        else:
            # Normal case with confidence scores as integers (1-100)
            # Get max confidence 
            max_confidence = 0
            if "max_confidence" in classification:
                if isinstance(classification["max_confidence"], float) and classification["max_confidence"] <= 1.0:
                    max_confidence = int(classification["max_confidence"] * 100)
                else:
                    # Already in percentage form, just use it directly
                    max_confidence = int(classification["max_confidence"])
            else:
                # If max_confidence not set, find the highest score
                confidence_scores = classification.get('confidence_scores', {})
                if confidence_scores:
                    max_score = max(confidence_scores.values())
                    if isinstance(max_score, float) and max_score <= 1.0:
                        max_confidence = int(max_score * 100)
                    else:
                        # Already in percentage form, use it directly 
                        max_confidence = int(max_score)
            
            # Get confidence scores with type handling
            processed_scores = {}
            for category, score in classification.get('confidence_scores', {}).items():
                # Convert float 0-1 to int 1-100
                if isinstance(score, float) and score <= 1.0:
                    processed_scores[category] = int(score * 100)
                # Already int in 1-100 range
                elif isinstance(score, (int, float)):
                    processed_scores[category] = int(score)
                # String (somehow)
                else:
                    try:
                        score_float = float(score)
                        if score_float <= 1.0:
                            processed_scores[category] = int(score_float * 100)
                        else:
                            processed_scores[category] = int(score_float)
                    except (ValueError, TypeError):
                        # Default if conversion fails
                        processed_scores[category] = 5
            
            # Handle legacy "Corporate IT" key
            if "Corporate IT" in processed_scores:
                score = processed_scores.pop("Corporate IT")
                processed_scores["Internal IT Department"] = score
                
            # Final validation - ensure scores are different
            if len(set(processed_scores.values())) <= 1:
                logger.warning("API response has identical confidence scores, fixing...")
                pred_class = classification.get('predicted_class')
                
                # Handle legacy "Non-Service Business" predicted class
                if pred_class == "Non-Service Business":
                    pred_class = "Internal IT Department"
                    
                if pred_class == "Managed Service Provider":
                    processed_scores = {
                        "Managed Service Provider": 90,
                        "Integrator - Commercial A/V": 10,
                        "Integrator - Residential A/V": 10,
                        "Internal IT Department": 0
                    }
                elif pred_class == "Integrator - Commercial A/V":
                    processed_scores = {
                        "Integrator - Commercial A/V": 90,
                        "Managed Service Provider": 10,
                        "Integrator - Residential A/V": 10,
                        "Internal IT Department": 0
                    }
                elif pred_class == "Integrator - Residential A/V":
                    processed_scores = {
                        "Integrator - Residential A/V": 90,
                        "Integrator - Commercial A/V": 10, 
                        "Managed Service Provider": 10,
                        "Internal IT Department": 0
                    }
                elif pred_class == "Process Did Not Complete":
                    # Set all scores to 0 for process_did_not_complete
                    processed_scores = {
                        "Managed Service Provider": 0,
                        "Integrator - Commercial A/V": 0,
                        "Integrator - Residential A/V": 0,
                        "Internal IT Department": 0
                    }
                    # Reset max_confidence to 0.0
                    max_confidence = 0
                elif pred_class == "Internal IT Department":
                    # For Internal IT Department, add Internal IT Department score
                    internal_it_potential = classification.get('internal_it_potential', 60)
                    if internal_it_potential is None:
                        internal_it_potential = 60
                        
                    processed_scores = {
                        "Managed Service Provider": 5,
                        "Integrator - Commercial A/V": 3,
                        "Integrator - Residential A/V": 2,
                        "Internal IT Department": internal_it_potential
                    }
                
                # Update max_confidence to match the new highest value if not Process Did Not Complete
                if pred_class not in ["Process Did Not Complete", "Internal IT Department"]:
                    max_confidence = 90
                    
            # Ensure explanation exists
            explanation = classification.get('llm_explanation', '')
            if not explanation:
                explanation = f"Based on analysis of website content, {domain} has been classified as a {classification.get('predicted_class')}."
                
            # Check for Non-Service Business in the explanation 
            if "non-service business" in explanation.lower() and classification.get('predicted_class') in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                if max_confidence <= 20:  # Only override if confidence is low
                    logger.info(f"Correcting classification for {domain} to Internal IT Department based on explanation")
                    classification['predicted_class'] = "Internal IT Department"

            # Handle legacy "Non-Service Business" predicted class
            if classification.get('predicted_class') == "Non-Service Business":
                classification['predicted_class'] = "Internal IT Department"
                
            # For Internal IT Department, ensure Internal IT Department score is included
            if classification.get('predicted_class') == "Internal IT Department":
                # Add Internal IT Department for Internal IT Department if not already present
                if "Internal IT Department" not in processed_scores:
                    internal_it_potential = classification.get('internal_it_potential', 60)
                    if internal_it_potential is None:
                        internal_it_potential = 60
                        
                    processed_scores["Internal IT Department"] = internal_it_potential
                    # Ensure service scores are low
                    for category in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                        processed_scores[category] = min(processed_scores.get(category, 5), 10)
                
                # Set confidence score to a consistent value for Internal IT Department
                max_confidence = 80
            
            # For service businesses, ensure Internal IT Department is 0
            elif classification.get('predicted_class') in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                processed_scores["Internal IT Department"] = 0
                
            # Create the final result (now merging with our initial result)
            result.update({
                "predicted_class": classification.get('predicted_class'),
                "confidence_score": max_confidence,
                "confidence_scores": processed_scores,
                "explanation": explanation,
                "company_description": classification.get('company_description', f"{domain} is a {classification.get('predicted_class')}."),
                "low_confidence": classification.get('low_confidence', False),
                "detection_method": classification.get('detection_method', 'api'),
                "source": "fresh",
                "is_parked": False
            })
            
            # Add one-line company description
            result["company_one_line"] = classification.get('company_one_line', generate_one_line_description(
                content="",  # We don't have content here
                predicted_class=classification.get('predicted_class', ''),
                domain=domain,
                company_description=result.get("company_description", "")
            ))
            
            # NEW: Add explicit company_name field prioritizing Apollo data
            if "apollo_data" in classification and classification["apollo_data"]:
                apollo_data = classification["apollo_data"]
                if isinstance(apollo_data, str):
                    try:
                        apollo_data = json.loads(apollo_data)
                    except:
                        apollo_data = {}
                
                if apollo_data and apollo_data.get('name'):
                    result["company_name"] = apollo_data.get('name')
                    logger.info(f"Using Apollo data for company name: {result['company_name']}")
            
            # If no Apollo company name, try AI data but check for navigation elements
            if "company_name" not in result and "ai_company_data" in classification and classification["ai_company_data"]:
                ai_data = classification["ai_company_data"]
                if isinstance(ai_data, str):
                    try:
                        ai_data = json.loads(ai_data)
                    except:
                        ai_data = {}
                
                if ai_data and ai_data.get('name'):
                    # Check for suspicious name patterns
                    if re.search(r'navigation|menu|open|close|header|footer', 
                               ai_data.get('name', ''), re.IGNORECASE):
                        logger.warning(f"Suspicious AI-extracted company name detected: {ai_data.get('name')}. Using domain instead.")
                        result["company_name"] = domain.split('.')[0].capitalize()
                    else:
                        result["company_name"] = ai_data.get('name')
                        logger.info(f"Using AI data for company name: {result['company_name']}")
            
            # Fallback to domain name if no company name found
            if "company_name" not in result:
                result["company_name"] = domain.split('.')[0].capitalize()
                logger.info(f"Using domain-derived company name: {result['company_name']}")
            
            # These fields will be added later to ensure they appear at the bottom
            preserved_crawler_type = classification.get("crawler_type", None)
            preserved_classifier_type = classification.get("classifier_type", None)

        # REMOVED: Don't add website URL conditionally - it's already set at the beginning
        # if url:
        #     result["website_url"] = url
            
        # REMOVED: Don't add email conditionally - it's already set at the beginning
        # if email:
        #     result["email"] = email
        
        # Add error_type if present in classification
        if "error_type" in classification:
            result["error_type"] = classification["error_type"]
            
        # Determine and add the final classification
        result["final_classification"] = determine_final_classification(result)
        logger.info(f"Added final classification: {result['final_classification']} for {domain}")
        
        # Add crawler_type and classifier_type at the end to ensure they appear at the bottom
        if preserved_crawler_type:
            result["crawler_type"] = preserved_crawler_type
        else:
            # Add a default if not present
            result["crawler_type"] = classification.get("crawler_type", "unknown")
            
        if preserved_classifier_type:
            result["classifier_type"] = preserved_classifier_type
        else:
            # Add a default if not present
            result["classifier_type"] = classification.get("classifier_type", "unknown")
            
        # Enhanced safety check for potential misclassifications with higher thresholds to avoid excessive alerts
        predicted_class = result.get("predicted_class", "")
        explanation_lower = result.get("explanation", "").lower()
        description_lower = result.get("company_description", "").lower()

        # Define stronger, more specific indicators with minimum thresholds
        misclassification_checks = [
            # Internal IT with strong AV integrator indicators
            {
                "check_class": "Internal IT Department",
                "min_indicators": 2,  # Require at least 2 indicators
                "indicators": [
                    "audio visual integration", "av integration", "home theater installation",
                    "commercial av systems", "audiovisual installer", "av systems integrator",
                    "media room design", "integrated control systems", "home automation installer",
                    "home automation", "smart home", "home theater", "home cinema", 
                    "media room", "lighting control", "sound system", "integrated security",
                    "entertainment system", "multi-room audio", "av installer"
                ],
                "warning": "This domain shows multiple specific audio-visual integration service indicators"
            },
            
            # Internal IT with strong MSP indicators
            {
                "check_class": "Internal IT Department",
                "min_indicators": 2,
                "indicators": [
                    "managed service provider", "it service provider", "it outsourcing",
                    "monthly support plan", "remote monitoring and management", "managed security services",
                    "24/7 it support", "it services for clients", "support desk for customers",
                    "it consulting", "managed it", "network management", "cloud services",
                    "technical support services", "providing it services"
                ],
                "warning": "This domain shows multiple specific managed IT service indicators"
            },
            
            # Clear residential/commercial confusion
            {
                "check_class": "Integrator - Commercial A/V",
                "min_indicators": 3,  # Higher threshold for this check
                "indicators": [
                    "residential installation", "home theater design", "smart home integration",
                    "whole home audio", "residential clients", "homeowners", "residential properties",
                    "home automation", "smart homes", "residential services", "home entertainment"
                ],
                "warning": "This domain strongly indicates residential A/V services"
            },
            
            # Clear commercial/residential confusion
            {
                "check_class": "Integrator - Residential A/V",
                "min_indicators": 3,  # Higher threshold for this check
                "indicators": [
                    "corporate clients", "conference room systems", "commercial installations",
                    "office av systems", "business customers", "digital signage solutions",
                    "corporate boardrooms", "meeting room technology", "commercial buildings",
                    "business conference", "commercial a/v"
                ],
                "warning": "This domain strongly indicates commercial A/V services"
            }
        ]

        # Check for misclassifications with higher thresholds
        for check in misclassification_checks:
            if result.get("predicted_class") == check["check_class"]:
                found_indicators = []
                
                # Look for indicators in both explanation and description
                for indicator in check["indicators"]:
                    if indicator in explanation_lower or indicator in description_lower:
                        found_indicators.append(indicator)
                
                # Only flag if we meet the minimum threshold of indicators
                if len(found_indicators) >= check["min_indicators"]:
                    result["possible_misclassification"] = True
                    result["indicators_found"] = found_indicators
                    result["misclassification_warning"] = f"{check['warning']} but was classified as {check['check_class']}"
                    
                    # Log the issue
                    logger.warning(f"Potential misclassification: {domain} classified as {check['check_class']} but has {len(found_indicators)} strong indicators: {', '.join(found_indicators)}")
                    break  # Only report one type of misclassification at a time
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing fresh result: {e}")
        # Return a basic result with error information
        error_result = {
            "domain": domain,
            "email": email,  # ADDED: Include email in error result
            "website_url": url,  # ADDED: Include URL in error result
            "predicted_class": classification.get('predicted_class', 'Unknown'),
            "confidence_score": 50,
            "confidence_scores": {
                "Managed Service Provider": 10,
                "Integrator - Commercial A/V": 5,
                "Integrator - Residential A/V": 5,
                "Internal IT Department": 0
            },
            "explanation": f"We encountered an error processing the classification result for {domain}.",
            "company_description": f"{domain} is a business that we attempted to classify.",
            "company_one_line": f"Unable to determine what {domain} does due to processing errors.",
            "low_confidence": True,
            "detection_method": "error_during_processing",
            "source": "error",
            "is_parked": False,
            "error": str(e),
            "final_classification": "2-Internal IT",  # Updated default for error cases
            "crawler_type": classification.get("crawler_type", "error_processor"),
            "classifier_type": classification.get("classifier_type", "error_processor")
        }
        return error_result
