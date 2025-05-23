"""Cross-validation utilities for reconciling classification with metadata."""

import logging
import re
from typing import Dict, Any, Optional

# Import utility functions
from domain_classifier.utils.json_utils import ensure_dict, safe_get

# Set up logging
logger = logging.getLogger(__name__)

def reconcile_classification(classification: Dict[str, Any],
                          apollo_data: Optional[Dict] = None,
                          ai_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Cross-validate classification results against metadata from different sources.
    Enhanced with better Apollo data usage for web crawl failures.

    Args:
        classification: The classification result
        apollo_data: Apollo company data
        ai_data: AI-extracted company data

    Returns:
        dict: The potentially modified classification
    """
    # Extract domain for logging
    domain = classification.get('domain', 'unknown')
    
    # First handle cases where web crawl failed or returned "Process Did Not Complete"
    if classification.get("error_type") or classification.get("predicted_class") == "Process Did Not Complete":
        if apollo_data and isinstance(apollo_data, dict) and any(apollo_data.values()):
            logger.info(f"Using Apollo data for classification when web content unavailable for {domain}")
            
            # Check if Apollo has description fields
            description = None
            for field in ["short_description", "description", "long_description"]:
                if apollo_data.get(field):
                    description = apollo_data.get(field)
                    break
            
            if description:
                # Check for IT/MSP specific terms in description
                if any(term in description.lower() for term in [
                    "managed service provider", "it service", "it support", "managed it",
                    "it consulting", "network management", "cloud services", "cybersecurity"
                ]):
                    logger.info(f"Apollo description indicates MSP for {domain}")
                    classification["predicted_class"] = "Managed Service Provider"
                    classification["is_service_business"] = True
                    classification["detection_method"] = "apollo_description_classification"
                    
                    # Update confidence scores
                    classification["confidence_scores"] = {
                        "Managed Service Provider": 75,
                        "Integrator - Commercial A/V": 10,
                        "Integrator - Residential A/V": 5,
                        "Internal IT Department": 0
                    }
                    
                    if "max_confidence" in classification:
                        classification["max_confidence"] = 0.75
                
                # Check for AV integration terms
                elif any(term in description.lower() for term in [
                    "audio visual", "av integration", "conference room", "commercial av",
                    "digital signage", "presentation systems", "room technology"
                ]):
                    logger.info(f"Apollo description indicates Commercial A/V for {domain}")
                    classification["predicted_class"] = "Integrator - Commercial A/V"
                    classification["is_service_business"] = True
                    classification["detection_method"] = "apollo_description_classification"
                    
                    # Update confidence scores
                    classification["confidence_scores"] = {
                        "Integrator - Commercial A/V": 75,
                        "Managed Service Provider": 10,
                        "Integrator - Residential A/V": 15,
                        "Internal IT Department": 0
                    }
                    
                    if "max_confidence" in classification:
                        classification["max_confidence"] = 0.75
                
                # Check for residential AV terms
                elif any(term in description.lower() for term in [
                    "home automation", "smart home", "home theater", "residential av",
                    "home entertainment", "home integration"
                ]):
                    logger.info(f"Apollo description indicates Residential A/V for {domain}")
                    classification["predicted_class"] = "Integrator - Residential A/V"
                    classification["is_service_business"] = True
                    classification["detection_method"] = "apollo_description_classification"
                    
                    # Update confidence scores
                    classification["confidence_scores"] = {
                        "Integrator - Residential A/V": 75,
                        "Integrator - Commercial A/V": 15,
                        "Managed Service Provider": 5,
                        "Internal IT Department": 0
                    }
                    
                    if "max_confidence" in classification:
                        classification["max_confidence"] = 0.75
                
                # Default for non-technology companies
                else:
                    # Check industry for additional clues
                    industry = apollo_data.get('industry', '').lower() if apollo_data.get('industry') else ''
                    
                    # Special handling for industries that often include IT services
                    if any(term in industry for term in [
                        "information technology", "it service", "computer", "tech", "software",
                        "network", "cyber", "cloud", "managed services"
                    ]):
                        logger.info(f"Apollo industry indicates MSP for {domain}: {industry}")
                        classification["predicted_class"] = "Managed Service Provider"
                        classification["is_service_business"] = True
                        classification["detection_method"] = "apollo_industry_classification"
                    else:
                        classification["predicted_class"] = "Internal IT Department"
                        classification["is_service_business"] = False
                        classification["detection_method"] = "apollo_description_classification"
                    
                    # Set appropriate confidence scores
                    if classification["predicted_class"] == "Managed Service Provider":
                        classification["confidence_scores"] = {
                            "Managed Service Provider": 65,
                            "Integrator - Commercial A/V": 10,
                            "Integrator - Residential A/V": 5,
                            "Internal IT Department": 0
                        }
                        
                        if "max_confidence" in classification:
                            classification["max_confidence"] = 0.65
                    else:
                        # Default Internal IT scores
                        classification["confidence_scores"] = {
                            "Managed Service Provider": 5,
                            "Integrator - Commercial A/V": 3,
                            "Integrator - Residential A/V": 2,
                            "Internal IT Department": 65
                        }
                        
                        if "max_confidence" in classification:
                            classification["max_confidence"] = 0.65
                
                # Return the Apollo-based classification
                return classification
    
    # Handle standard cases below - we should only get here for successful web crawls
    # or if no Apollo data is available to override failed crawls
    
    # Only proceed if we have Apollo or AI data to cross-validate against
    if not apollo_data and not ai_data:
        return classification

    # Fast path for cybersecurity companies
    apollo_industry = ''
    if apollo_data and isinstance(apollo_data, dict) and "industry" in apollo_data:
        apollo_industry = apollo_data["industry"].lower() if isinstance(apollo_data["industry"], str) else ''

    # Quick check for security industry - always MSP
    if 'security' in apollo_industry or 'cyber' in apollo_industry:
        logger.info(f"Security industry detected for {domain} from Apollo: {apollo_industry}")
        
        if classification.get('predicted_class') != "Managed Service Provider":
            logger.warning(f"Cybersecurity company {domain} was misclassified as {classification.get('predicted_class')}, correcting to MSP")
            
            # Override classification quickly
            classification['predicted_class'] = "Managed Service Provider"
            classification['is_service_business'] = True
            classification['confidence_scores'] = {
                'Managed Service Provider': 85,
                'Integrator - Commercial A/V': 5,
                'Integrator - Residential A/V': 5,
                'Internal IT Department': 0
            }
            classification['max_confidence'] = 0.85
            classification['detection_method'] = "cross_validation_security"
            
            # Log success
            logger.info(f"Reclassified {domain} as Managed Service Provider based on security industry")
            
            return classification

    # CRITICAL: Special check for IT Services industry - this should always be MSP
    if ('it service' in apollo_industry or
        'information technology' in apollo_industry or
        'information technology & services' in apollo_industry or
        'computer networking' in apollo_industry or
        'information technology and services' in apollo_industry or
        'managed services' in apollo_industry):
        
        logger.info(f"IT Services industry detected for {domain} from Apollo: {apollo_industry}")
        
        if classification.get('predicted_class') != "Managed Service Provider":
            logger.warning(f"IT Services company {domain} was misclassified as {classification.get('predicted_class')}, correcting to MSP")
            
            # Override classification
            classification['predicted_class'] = "Managed Service Provider"
            classification['is_service_business'] = True
            classification['confidence_scores'] = {
                'Managed Service Provider': 90,
                'Integrator - Commercial A/V': 5,
                'Integrator - Residential A/V': 5,
                'Internal IT Department': 0
            }
            classification['max_confidence'] = 0.9
            classification['detection_method'] = "cross_validation_it_services"
            
            # Log success
            logger.info(f"Reclassified {domain} as Managed Service Provider based on IT Services industry")
            
            return classification
    
    # Quick check for manufacturing/industrial companies - usually Internal IT
    industrial_terms = ['manufacturing', 'industrial', 'factory', 'production']
    
    if apollo_industry and any(term in apollo_industry for term in industrial_terms):
        if classification.get('predicted_class') in ["Integrator - Commercial A/V", "Integrator - Residential A/V"]:
            # Override classification for industrial companies misclassified as AV integrators
            logger.warning(f"Industrial company with industry '{apollo_industry}' was misclassified as {classification.get('predicted_class')}")
            
            classification['predicted_class'] = "Internal IT Department"
            classification['is_service_business'] = False
            classification['confidence_scores'] = {
                'Managed Service Provider': 5,
                'Integrator - Commercial A/V': 3,
                'Integrator - Residential A/V': 2,
                'Internal IT Department': 60
            }
            classification['detection_method'] = "cross_validation_industrial"
            
            return classification
    
    # Fast path for MSPs - we usually don't need to override these
    if classification.get('predicted_class') == "Managed Service Provider":
        # Only consider overriding if specifically in manufacturing/transportation
        if apollo_industry and ('manufacturing' in apollo_industry or 'transport' in apollo_industry):
            # Continue to detailed check
            pass
        else:
            # Fast return for MSPs with non-contradicting industry data
            return classification
    
    # Now do detailed analysis only for uncertain or contradicting cases
    
    # Convert to dictionaries if needed
    apollo_dict = ensure_dict(apollo_data, "apollo_data") if apollo_data else {}
    ai_dict = ensure_dict(ai_data, "ai_data") if ai_data else {}
    
    # Extract industry information from all sources
    ai_industry = ''
    industry = safe_get(ai_dict, 'industry', '')
    
    if industry is not None:
        ai_industry = industry.lower() if isinstance(industry, str) else ''
    
    # Extract employee count (higher counts suggest Internal IT potential)
    employee_count = 0
    apollo_count = safe_get(apollo_dict, 'employee_count', 0)
    
    if apollo_count:
        employee_count = apollo_count
    else:
        employee_count = safe_get(ai_dict, 'employee_count', 0)
    
    # List of industries that typically aren't MSPs or AV integrators
    # CRITICAL FIX: Remove 'information technology', 'it service', etc. from this list
    non_service_industries = [
        # Manufacturing & Industrial
        'manufacturing', 'forging', 'steel', 'industrial', 'factory', 'production',
        'fabrication', 'assembly', 'machining', 'engineering', 'metals', 'construction',
        
        # Retail & E-commerce
        'retail', 'shop', 'store', 'e-commerce', 'ecommerce', 'webshop',
        'marketplace', 'sales', 'dealer', 'distribution',
        
        # Hospitality & Food
        'hospitality', 'hotel', 'restaurant', 'catering', 'accommodation', 'lodging',
        'food', 'beverage', 'dining', 'cafe', 'bakery', 'culinary',
        
        # Healthcare & Medical
        'healthcare', 'medical', 'hospital', 'clinic', 'wellness', 'pharmacy',
        'dental', 'health', 'care', 'patient', 'treatment', 'therapy',
        
        # Transportation & Logistics
        'transportation', 'logistics', 'shipping', 'freight', 'delivery', 'courier',
        'trucking', 'distribution', 'supply chain', 'warehouse', 'storage',
        
        # Maritime & Shipping
        'maritime', 'shipping', 'vessel', 'boat', 'nautical', 'marine', 'cruise',
        'ferry', 'yacht', 'port', 'harbor', 'offshore', 'naval', 'dock'
    ]
    
    # List of industries that are explicitly MSP or IT service providers
    msp_industries = [
        'information technology',
        'it service',
        'information technology & services',
        'information technology and services',
        'computer networking',
        'network security',
        'computer & network security',
        'managed services',
        'it consulting',
        'cloud services',
        'cyber security',
        'cybersecurity'
    ]
    
    # Check if the industry is specifically an MSP industry
    is_msp_industry = False
    
    if apollo_industry:
        for term in msp_industries:
            if term in apollo_industry:
                is_msp_industry = True
                logger.info(f"MSP industry detected for {domain} from Apollo: {apollo_industry}")
                break
    
    # If it's an MSP industry but classified differently, override to MSP
    if is_msp_industry and classification.get('predicted_class') != "Managed Service Provider":
        logger.warning(f"MSP industry company {domain} was misclassified as {classification.get('predicted_class')}")
        
        classification['predicted_class'] = "Managed Service Provider"
        classification['is_service_business'] = True
        classification['confidence_scores'] = {
            'Managed Service Provider': 90,
            'Integrator - Commercial A/V': 5,
            'Integrator - Residential A/V': 5,
            'Internal IT Department': 0
        }
        classification['max_confidence'] = 0.9
        classification['detection_method'] = "cross_validation_msp_industry"
        
        logger.info(f"Reclassified {domain} as Managed Service Provider based on MSP industry")
        
        return classification
    
    # Industries that could be BOTH product and service businesses (dual nature)
    # AV integrators often fall into these categories
    dual_nature_industries = [
        'consumer electronics',
        'electronics',
        'audio visual',
        'audio-visual',
        'technology',
        'home automation',
        'security systems',
        'smart home',
        'integration',
        'communications',
        'audio',
        'visual',
        'media systems',
        'technical services'
    ]
    
    # Check specifically for maritime industry contradictions
    is_maritime = False
    
    if 'maritime' in apollo_industry or 'shipping' in apollo_industry or 'vessel' in apollo_industry:
        is_maritime = True
    elif ai_industry and ('maritime' in ai_industry or 'shipping' in ai_industry or 'vessel' in ai_industry):
        is_maritime = True
    elif 'company_description' in classification:
        desc = classification.get('company_description', '').lower()
        if 'maritime' in desc or 'shipping' in desc or 'vessel' in desc:
            is_maritime = True
    
    # Check if the predicted class is a service business type
    is_service = classification.get('predicted_class') in [
        'Managed Service Provider',
        'Integrator - Commercial A/V',
        'Integrator - Residential A/V'
    ]
    
    source = classification.get('source', '')
    
    # Extract confidence score
    confidence_score = classification.get('confidence_score', 0)
    max_confidence = classification.get('max_confidence', 0.5)
    high_confidence_classification = max_confidence >= 0.7 or confidence_score >= 70
    
    # Check if this is an IT department being wrongly classified as an AV integrator
    is_likely_internal_it = False
    
    # For Commercial AV or Residential AV Integrators specifically
    if classification.get('predicted_class') in ['Integrator - Commercial A/V', 'Integrator - Residential A/V']:
        # 1. Check if industry data contradicts this classification
        # BUT exclude dual-nature industries that could be both product and service
        if apollo_industry and any(term in apollo_industry for term in non_service_industries) and \
           not any(term in apollo_industry for term in dual_nature_industries):
            is_likely_internal_it = True
            logger.warning(f"Industry data from Apollo ({apollo_industry}) contradicts AV classification for {domain}")
        
        # 2. If there's no strong AV indicators in the descriptions, be more skeptical
        # But only if we don't have high confidence in the classification
        if not high_confidence_classification:
            # Check for IT department indicators in the content
            description_lower = classification.get('company_description', '').lower()
            explanation_lower = classification.get('explanation', '').lower()
            
            it_dept_indicators = ['it department', 'it team', 'internal it', 'technical staff',
                              'support team', 'helpdesk', 'it support', 'it personnel']
            
            for indicator in it_dept_indicators:
                if indicator in description_lower or indicator in explanation_lower:
                    is_likely_internal_it = True
                    logger.warning(f"Found IT department indicator: '{indicator}' for {domain} classified as AV")
                    break
        
        # Be skeptical of cached AV classifications with no dual-nature industry
        if source == 'cached' and \
           (not apollo_industry or not any(term in apollo_industry for term in dual_nature_industries)):
            # Increase likelihood of being internal IT if industry data present
            if apollo_industry or ai_industry:
                is_likely_internal_it = True
                logger.warning(f"Cached AV classification with contradicting industry data for {domain}, likely Internal IT")
    
    # For service businesses, check for contradictions with industry data
    if is_service:
        # Check if industry data suggests a non-service business
        # But be careful with dual-nature industries like consumer electronics
        apollo_non_service = apollo_industry and any(term in apollo_industry for term in non_service_industries) and \
                           not any(term in apollo_industry for term in dual_nature_industries)
        
        ai_non_service = ai_industry and any(term in ai_industry for term in non_service_industries) and \
                        not any(term in ai_industry for term in dual_nature_industries)
        
        # Potential conflict detected - industry data doesn't match classification
        should_override = False
        
        # CRITICAL FIX: Check for IT services industry - never override to Internal IT Department
        if apollo_industry and any(term in apollo_industry for term in msp_industries):
            logger.info(f"IT services industry detected for {domain}, preserving MSP classification")
            should_override = False
            return classification  # Early return to prevent further override checks
        
        # For maritime industry, always override service business classification
        if is_maritime:
            should_override = True
            logger.warning(f"Maritime industry detected for {domain} - overriding service business classification")
        
        # For AV integrators without supporting evidence, override
        elif classification.get('predicted_class') in ['Integrator - Commercial A/V', 'Integrator - Residential A/V'] and is_likely_internal_it:
            should_override = True
            logger.warning(f"Likely internal IT dept being misclassified as AV for {domain}")
        
        # For cached results with contradicting Apollo data, be more aggressive
        elif source == 'cached' and apollo_non_service:
            should_override = True
            logger.warning(f"Cached classification for {domain} contradicts Apollo industry data, overriding")
        
        # If Apollo and AI both suggest non-service
        elif apollo_non_service and ai_non_service:
            should_override = True
            logger.warning(f"Both Apollo and AI data suggest non-service for {domain}, overriding classification")
        
        # If there's a strong non-service signal from Apollo
        elif apollo_dict and apollo_non_service:
            should_override = True
            logger.warning(f"Apollo data strongly suggests non-service for {domain}, overriding classification")
        
        if should_override:
            logger.warning(f"Classification conflict for {domain}: classified as {classification.get('predicted_class')} but data suggests non-service business")
            logger.warning(f"Apollo industry: {apollo_industry}, AI industry: {ai_industry}")
            
            # Override classification
            classification['predicted_class'] = 'Internal IT Department'
            classification['is_service_business'] = False
            
            # Set internal_it_potential based on employee count
            internal_it_score = 50  # Default medium score
            
            if employee_count:
                if employee_count < 10:
                    internal_it_score = 30  # Small company
                elif employee_count < 50:
                    internal_it_score = 50  # Medium company
                else:
                    internal_it_score = 70  # Large company
            
            # For maritime companies, set a higher IT score since they need significant IT
            if is_maritime:
                internal_it_score = max(internal_it_score, 60)  # At least 60 for maritime
                
            classification['internal_it_potential'] = internal_it_score
            
            # Update confidence scores
            classification['confidence_scores'] = {
                'Managed Service Provider': 5,
                'Integrator - Commercial A/V': 3,
                'Integrator - Residential A/V': 2,
                'Internal IT Department': internal_it_score
            }
            
            # Update max_confidence
            classification['max_confidence'] = 0.8
            
            # Generate a more accurate company description based on industry data
            if is_maritime:
                # Create a maritime-specific description
                if apollo_dict and apollo_dict.get('name'):
                    company_name = apollo_dict.get('name')
                else:
                    company_name = domain.split('.')[0].capitalize()
                
                # Generate a new description that doesn't fabricate AV services
                new_description = f"{company_name} is a maritime industry company specializing in shipping services"
                
                if employee_count:
                    new_description += f" with approximately {employee_count} employees"
                
                new_description += "."
                
                company_description = classification.get('company_description', '').lower()
                
                if "supplies" in company_description or "parts" in company_description:
                    new_description += " The company provides ship supplies, spare parts, and maritime equipment to vessels."
                elif "logistics" in company_description:
                    new_description += " The company offers maritime logistics and shipping services."
                else:
                    new_description += " The company operates in the maritime industry, providing shipping-related services."
                
                # Replace the fabricated description
                classification['company_description'] = new_description
                
                # Also fix company_one_line if it mentions AV
                one_line = classification.get('company_one_line', '')
                if 'audio' in one_line.lower() or 'visual' in one_line.lower() or 'av' in one_line.lower():
                    classification['company_one_line'] = f"{company_name} provides maritime shipping services and supplies."
            
            elif apollo_industry:
                # Create an industry-specific description
                if apollo_dict and apollo_dict.get('name'):
                    company_name = apollo_dict.get('name')
                else:
                    company_name = domain.split('.')[0].capitalize()
                
                # Generate a new description
                new_description = f"{company_name} is a {apollo_industry} company"
                
                if employee_count:
                    new_description += f" with approximately {employee_count} employees"
                
                new_description += "."
                
                # Add more context if we can determine it
                company_description = classification.get('company_description', '').lower()
                
                if "products" in company_description or "manufacturing" in company_description:
                    new_description += f" The company manufactures or distributes products in the {apollo_industry} industry."
                elif "services" in company_description:
                    new_description += f" The company provides services in the {apollo_industry} sector."
                
                # Replace the fabricated description
                classification['company_description'] = new_description
                
                # Also fix company_one_line if it mentions AV or MSP
                one_line = classification.get('company_one_line', '')
                if 'audio' in one_line.lower() or 'visual' in one_line.lower() or 'av' in one_line.lower() or 'provide' in one_line.lower():
                    classification['company_one_line'] = f"{company_name} is a {apollo_industry} company with internal IT needs."
            
            # Add a note about the reconciliation
            original_explanation = classification.get('llm_explanation', '')
            reconciliation_note = f"\n\nNOTE: This classification was adjusted based on industry data from Apollo ({apollo_industry}) and/or AI extraction ({ai_industry}) which indicates this is likely a non-service business."
            
            classification['llm_explanation'] = original_explanation + reconciliation_note
            
            logger.info(f"Reclassified {domain} as Internal IT Department based on industry data")
    
    return classification
