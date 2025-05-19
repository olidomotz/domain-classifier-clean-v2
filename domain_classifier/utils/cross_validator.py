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
    
    Args:
        classification: The classification result
        apollo_data: Apollo company data
        ai_data: AI-extracted company data
        
    Returns:
        dict: The potentially modified classification
    """
    # ADDED: Save critical fields that must be preserved throughout the process
    critical_fields = ["domain", "email", "website_url", "crawler_type", "classifier_type"]
    preserved_fields = {}
    for field in critical_fields:
        if field in classification:
            preserved_fields[field] = classification[field]
            logger.info(f"Preserving critical field {field}: {preserved_fields[field]}")
    
    # Only proceed if we have Apollo or AI data to cross-validate against
    if not apollo_data and not ai_data:
        return classification
    
    # Convert to dictionaries if needed
    apollo_dict = ensure_dict(apollo_data, "apollo_data")
    ai_dict = ensure_dict(ai_data, "ai_data")
    
    # Extract industry information from all sources
    apollo_industry = ''
    industry = safe_get(apollo_dict, 'industry', '')
    if industry is not None:
        apollo_industry = industry.lower() if isinstance(industry, str) else ''
    
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
    non_service_industries = [
        # Manufacturing & Industrial
        'manufacturing', 'forging', 'steel', 'industrial', 'factory', 'production',
        'fabrication', 'assembly', 'machining', 'engineering', 'metals', 'construction',
        
        # Retail & E-commerce - EXCLUDING consumer electronics
        'retail', 'shop', 'store', 'e-commerce', 'ecommerce', 'online store', 'webshop',
        'marketplace', 'sales', 'dealer', 'distribution',
        
        # Hospitality & Food
        'hospitality', 'hotel', 'restaurant', 'catering', 'accommodation', 'lodging',
        'food', 'beverage', 'dining', 'cafe', 'bakery', 'culinary',
        
        # Healthcare & Medical
        'healthcare', 'medical', 'hospital', 'clinic', 'wellness', 'pharmacy',
        'dental', 'health', 'care', 'patient', 'treatment', 'therapy',
        
        # Education
        'education', 'school', 'university', 'college', 'academic', 'learning', 
        'training', 'teaching', 'educational', 'student', 'campus',
        
        # Transportation & Logistics
        'transportation', 'logistics', 'shipping', 'freight', 'delivery', 'courier',
        'trucking', 'distribution', 'supply chain', 'warehouse', 'storage',
        
        # Maritime & Shipping
        'maritime', 'shipping', 'vessel', 'boat', 'nautical', 'marine', 'cruise',
        'ferry', 'yacht', 'port', 'harbor', 'offshore', 'naval', 'dock',
        
        # Financial & Professional Services
        'financial', 'banking', 'insurance', 'legal', 'law', 'accounting',
        'finance', 'investment', 'wealth', 'tax', 'audit', 'advisory', 'consulting',
        
        # Agriculture
        'agriculture', 'farming', 'farm', 'crop', 'livestock', 'dairy',
        
        # Real Estate
        'real estate', 'property', 'properties', 'realty', 'estate', 'housing',
        'apartment', 'commercial property', 'residential property',
        
        # Energy & Utilities
        'energy', 'oil', 'gas', 'electricity', 'utility', 'power', 'renewable',
        'solar', 'wind', 'hydro', 'nuclear', 'electric',
        
        # Creative & Media
        'media', 'creative', 'advertising', 'publishing', 'film', 'tv', 'television',
        'radio', 'entertainment', 'arts', 'design', 'studio',
        
        # Government & Non-profit
        'government', 'public', 'non-profit', 'nonprofit', 'charity', 'foundation',
        'NGO', 'association', 'organization', 'municipal', 'federal', 'public sector'
    ]
    
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
    
    # Terms that strongly indicate AV Integrator services
    av_integrator_indicators = [
        'audio visual integration', 'av integration', 'audio visual installations',
        'conference room design', 'video conferencing installation', 'commercial av',
        'av systems integrator', 'professional av', 'video wall installation',
        'digital signage installation', 'av solutions provider', 'av contractor',
        'theatrical lighting', 'stage lighting', 'sound system installation',
        'projector installation', 'control system integration', 'classroom technology',
        'home automation', 'smart home', 'home theater', 'cinema', 'integrated security',
        'lighting control', 'multi-room audio', 'home entertainment'
    ]
    
    # Check if these strong AV indicators are present in the content description or explanation
    description_lower = classification.get('company_description', '').lower()
    explanation_lower = classification.get('explanation', '').lower()
    
    has_av_indicators = False
    av_indicators_found = []
    for indicator in av_integrator_indicators:
        if (indicator in description_lower or indicator in explanation_lower):
            has_av_indicators = True
            av_indicators_found.append(indicator)
            logger.info(f"Found AV integrator indicator: '{indicator}'")
            
            # If we find at least 2 indicators, that's strong evidence
            if len(av_indicators_found) >= 2:
                break
    
    # Check if the predicted class is a service business type
    is_service = classification.get('predicted_class') in [
        'Managed Service Provider', 
        'Integrator - Commercial A/V',
        'Integrator - Residential A/V'
    ]
    
    domain = classification.get('domain', 'unknown')
    source = classification.get('source', '')
    
    # Extract confidence score
    confidence_score = classification.get('confidence_score', 0)
    max_confidence = classification.get('max_confidence', 0.5)
    high_confidence_classification = max_confidence >= 0.7 or confidence_score >= 70
    
    # Check specifically for maritime industry contradictions
    is_maritime = False
    if 'maritime' in apollo_industry or 'shipping' in apollo_industry or 'vessel' in apollo_industry:
        is_maritime = True
    elif ai_industry and ('maritime' in ai_industry or 'shipping' in ai_industry or 'vessel' in ai_industry):
        is_maritime = True
    elif 'maritime' in description_lower or 'shipping' in description_lower or 'vessel' in description_lower:
        is_maritime = True
    
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
        if not has_av_indicators and not high_confidence_classification:
            # Check for IT department indicators in the content
            it_dept_indicators = ['it department', 'it team', 'internal it', 'technical staff', 
                                'support team', 'helpdesk', 'it support', 'it personnel']
            
            for indicator in it_dept_indicators:
                if indicator in description_lower or indicator in explanation_lower:
                    is_likely_internal_it = True
                    logger.warning(f"Found IT department indicator: '{indicator}' for {domain} classified as AV")
                    break
            
            # Be skeptical of cached AV classifications with no supporting evidence and no dual-nature industry
            if source == 'cached' and not has_av_indicators and \
               (not apollo_industry or not any(term in apollo_industry for term in dual_nature_industries)):
                # Increase likelihood of being internal IT if industry data present
                if apollo_industry or ai_industry:
                    is_likely_internal_it = True
                    logger.warning(f"Cached AV classification with no AV indicators for {domain}, likely Internal IT")
    
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
        
        # For maritime industry, always override service business classification
        if is_maritime:
            should_override = True
            logger.warning(f"Maritime industry detected for {domain} - overriding service business classification")
        
        # For AV integrators with clear service indicators, DON'T override even if industry is conflicting
        elif classification.get('predicted_class') in ['Integrator - Commercial A/V', 'Integrator - Residential A/V']:
            if has_av_indicators:
                should_override = False
                logger.info(f"Found strong AV indicators for {domain}, keeping AV classification despite industry data")
            elif is_likely_internal_it:
                should_override = True
                logger.warning(f"Likely internal IT dept being misclassified as AV for {domain}")
            # If we have high confidence in the classification, don't override
            elif high_confidence_classification:
                should_override = False
                logger.info(f"High confidence ({max_confidence:.2f}) in AV classification for {domain}, keeping despite industry data")
            # For dual-nature industries like consumer electronics, don't override
            elif apollo_industry and any(term in apollo_industry for term in dual_nature_industries):
                should_override = False
                logger.info(f"Dual-nature industry ({apollo_industry}) detected for {domain}, keeping AV classification")
        
        # For Commercial AV without supporting evidence, override
        elif classification.get('predicted_class') == 'Integrator - Commercial A/V' and is_likely_internal_it:
            should_override = True
            logger.warning(f"Likely internal IT dept being misclassified as Commercial AV for {domain}")
        
        # For cached results with contradicting Apollo data, be more aggressive
        elif source == 'cached' and apollo_non_service:
            # But still check for AV indicators before overriding
            if has_av_indicators:
                should_override = False
                logger.info(f"Found AV indicators in cached result for {domain}, keeping classification")
            else:
                should_override = True
                logger.warning(f"Cached classification for {domain} contradicts Apollo industry data, overriding")
        
        # If Apollo and AI both suggest non-service
        elif apollo_non_service and ai_non_service:
            # But still check for AV indicators before overriding
            if has_av_indicators:
                should_override = False
                logger.info(f"Found AV indicators for {domain}, keeping classification despite industry data")
            else:
                should_override = True
                logger.warning(f"Both Apollo and AI data suggest non-service for {domain}, overriding classification")
            
        # If there's a strong non-service signal from Apollo
        elif apollo_dict and apollo_non_service and not has_av_indicators:
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
                
                if "supplies" in description_lower or "parts" in description_lower:
                    new_description += " The company provides ship supplies, spare parts, and maritime equipment to vessels."
                elif "logistics" in description_lower:
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
                if "products" in description_lower or "manufacturing" in description_lower:
                    new_description += f" The company manufactures or distributes products in the {apollo_industry} industry."
                elif "services" in description_lower:
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
            
            # ADDED: Restore preserved critical fields
            for field, value in preserved_fields.items():
                if field not in classification or not classification[field]:
                    classification[field] = value
                    logger.info(f"Restored critical field {field}: {value} after override")
    
    # ADDED: Final check to ensure critical fields were preserved
    for field in critical_fields:
        if field in preserved_fields and (field not in classification or not classification[field]):
            classification[field] = preserved_fields[field]
            logger.info(f"Final restore of critical field {field}: {preserved_fields[field]}")
    
    return classification
