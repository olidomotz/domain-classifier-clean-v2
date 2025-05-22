"""Description enhancer module for company descriptions."""

import requests
import logging
import os
import json
import re
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def enhance_company_description(basic_description: str, apollo_data: Dict[str, Any], classification: Dict[str, Any]) -> str:
    """
    Create an enhanced company description using Apollo data and classification.
    
    Args:
        basic_description: The original basic description
        apollo_data: Company data from Apollo
        classification: Classification result
        
    Returns:
        str: Enhanced company description
    """
    # Handle case where apollo_data might be a string (JSON)
    if isinstance(apollo_data, str):
        try:
            apollo_data = json.loads(apollo_data)
        except:
            # If parsing fails, treat as empty dict
            apollo_data = {}
    
    # Import remove_redundancy from text_processing if available
    try:
        from domain_classifier.utils.text_processing import remove_redundancy
        has_remove_redundancy = True
    except ImportError:
        has_remove_redundancy = False
    
    enhanced_description = basic_description
    
    # Add company size and founding info if available
    if apollo_data and isinstance(apollo_data, dict) and apollo_data.get('employee_count'):
        employee_count = apollo_data.get('employee_count')
        founded_year = apollo_data.get('founded_year', '')
        founded_phrase = ""  # Initialize with empty string
        
        # Only add founding year information if it's actually available
        if founded_year and str(founded_year).strip() and str(founded_year) != "None":
            founded_phrase = f"Founded in {founded_year}, "
            
        size_description = ""
        if employee_count < 10:
            size_description = "a small"
        elif employee_count < 50:
            size_description = "a mid-sized"
        else:
            size_description = "a larger"
        
        industry = apollo_data.get('industry', '')
        if industry is not None:
            industry_phrase = f" in the {industry.lower() if isinstance(industry, str) else ''} sector" if industry else ""
        else:
            industry_phrase = ""
            
        company_name = apollo_data.get('name', '')
        domain_name = classification.get('domain', '')
        name_to_use = company_name or domain_name
        
        # Add size and founding info to the beginning
        # For Internal IT, don't mention the IT department but focus on the business
        if classification.get('predicted_class') == "Internal IT Department":
            enhanced_description = f"{founded_phrase}{name_to_use} is {size_description} {industry_phrase} company. {enhanced_description}"
        else:
            enhanced_description = f"{founded_phrase}{name_to_use} is {size_description} {classification.get('predicted_class', '').lower()}{industry_phrase}. {enhanced_description}"
            
    # Clean up the enhanced description
    if has_remove_redundancy:
        domain = classification.get('domain', '')
        enhanced_description = remove_redundancy(enhanced_description, domain)
    else:
        # Basic cleanup if remove_redundancy is not available
        enhanced_description = re.sub(r'\s+', ' ', enhanced_description).strip()
        
        # Remove common redundant phrases
        enhanced_description = re.sub(r'managed service provider.*managed service provider', 'managed service provider', enhanced_description, flags=re.IGNORECASE)
        enhanced_description = re.sub(r'allowing (?:clients|customers) to focus on (?:their|its) core business', '', enhanced_description, flags=re.IGNORECASE)
    
    # Additional cleanup to remove placeholder text
    enhanced_description = re.sub(r'\[(?:FOUNDING YEAR|founding year|YEAR|year)[^\]]*\]', '', enhanced_description)
    enhanced_description = re.sub(r'Founded in\s+,', '', enhanced_description)
    enhanced_description = re.sub(r'since\s+\.', '', enhanced_description)
    enhanced_description = re.sub(r'has been in operation since\s+\.', 'is in operation. ', enhanced_description)
    enhanced_description = re.sub(r'in operation since\s+\.', 'in operation. ', enhanced_description)
    enhanced_description = re.sub(r'established in\s+,', '', enhanced_description)
    enhanced_description = re.sub(r'\s+,', ',', enhanced_description)  # Fix spaces before commas
    enhanced_description = re.sub(r'\s+\.', '.', enhanced_description)  # Fix spaces before periods
    enhanced_description = re.sub(r'\.\s*\.', '.', enhanced_description)  # Fix multiple periods
    enhanced_description = re.sub(r'\s+', ' ', enhanced_description)  # Normalize spaces
    
    return enhanced_description

def generate_detailed_description(classification: Dict[str, Any], 
                                 apollo_data: Optional[Dict] = None,
                                 apollo_person_data: Optional[Dict] = None) -> str:
    """
    Generate a detailed company description without fabricating data when none is available.
    
    Args:
        classification: The classification result
        apollo_data: Optional company data from Apollo
        apollo_person_data: Optional person data from Apollo
        
    Returns:
        str: Detailed company description or factual statement about lack of data
    """
    try:
        # Check if this is a "Process Did Not Complete" classification with no data
        if classification.get("predicted_class") == "Process Did Not Complete":
            domain = classification.get("domain", "unknown")
            logger.info(f"Skipping description generation for {domain} - Process Did Not Complete")
            
            # Only generate a description if we have Apollo data
            if not apollo_data or not isinstance(apollo_data, dict) or not apollo_data.get("name"):
                logger.info(f"No Apollo data available for {domain}, returning minimal description")
                
                # Use a simple, factual statement instead of generating a fictional description
                classification["company_description"] = f"Unable to retrieve information for {domain} due to insufficient data."
                
                # Also set a minimal one-liner
                classification["company_one_line"] = f"No data available for {domain}."
                
                return classification.get("company_description", "")
        
        # Extract domain and prediction for direct use
        domain = classification.get('domain', '')
        predicted_class = classification.get('predicted_class', '')
        
        # Handle case where apollo_data might be a string (JSON)
        if isinstance(apollo_data, str):
            try:
                apollo_data = json.loads(apollo_data)
            except:
                # If parsing fails, treat as empty dict
                apollo_data = {}
        
        # Handle case where apollo_person_data might be a string (JSON)
        if isinstance(apollo_person_data, str):
            try:
                apollo_person_data = json.loads(apollo_person_data)
            except:
                # If parsing fails, treat as empty dict
                apollo_person_data = {}
        
        # CRITICAL: Force company_description into the classification if not present
        if not classification.get("company_description"):
            if predicted_class == "Managed Service Provider":
                classification["company_description"] = f"{domain} is a Managed Service Provider offering IT services and solutions to businesses."
            elif predicted_class == "Integrator - Commercial A/V":
                classification["company_description"] = f"{domain} is a Commercial A/V Integrator providing audio-visual solutions for businesses."
            elif predicted_class == "Integrator - Residential A/V":
                classification["company_description"] = f"{domain} is a Residential A/V Integrator providing home automation and entertainment systems."
            else:
                classification["company_description"] = f"{domain} is a business with internal IT needs."
            
            logger.info(f"Added basic company_description for {domain}")
        
        # Build company name from best available source
        company_name = ''
        if apollo_data and isinstance(apollo_data, dict) and apollo_data.get('name'):
            company_name = apollo_data.get('name')
        else:
            company_name = classification.get('company_name', domain.split('.')[0].capitalize())
        
        # Use Apollo short_description if available - this is a reliable factual source
        if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("short_description"):
            logger.info(f"Using Apollo short_description for {domain}")
            classification["company_description"] = apollo_data.get("short_description")
            return classification["company_description"]
        
        # Build prompt with available information
        prompt = f"""Based on the following information, write a factual company description for {company_name}:

Business Type: {predicted_class}

Domain: {domain}
"""
        
        # Add Apollo data if available, but only include values that are actually present
        if apollo_data and isinstance(apollo_data, dict):
            industry = apollo_data.get('industry', '')
            founded = apollo_data.get('founded_year', '')
            size = apollo_data.get('employee_count', '')
            
            if industry:
                prompt += f"Industry: {industry}\n"
                
            # Only include founding year if it's available and valid
            if founded and str(founded).strip() and str(founded) != "None":
                prompt += f"Founded: {founded}\n"
                
            if size:
                prompt += f"Size: Approximately {size} employees\n"
        
        # Add the original description
        prompt += f"""
Original Description: {classification.get('company_description', '')}

Write a detailed factual description (~100 words) that focuses on what the company does. IMPORTANT REQUIREMENTS:

1. Focus ONLY on core services/products they provide, NEVER mention technologies they use
2. Avoid all redundancy (never repeat information)
3. Avoid generic phrases like "allowing clients to focus on core business"
4. Include founding year ONLY if available, otherwise OMIT any mention of founding or establishment date
5. NEVER use placeholders like [FOUNDING YEAR], [YEAR], or similar brackets
6. Aim for approximately 100 words - not too short, not too long
7. Include specific services they offer, but NEVER mention specific technology names
8. No marketing language or quality statements
9. If they serve specific industries, mention those
10. CRITICAL: For "Internal IT Department" companies, focus ENTIRELY on what the COMPANY does as a business - NOT on their IT department
11. Do not use phrases like "as a managed service provider" or "as an integrator"
12. If founding year is not available, simply omit any sentence about when the company was founded
"""
        
        # Special system message adjustment for Internal IT companies
        system_message = "You write factual, informative company descriptions without redundancy or technology mentions. You focus on what services a company provides in approximately 100 words, without marketing language. You NEVER use placeholders in brackets like [FOUNDING YEAR]."
        
        if predicted_class == "Internal IT Department":
            system_message = "You write factual, informative business descriptions without redundancy or technology mentions. For companies with internal IT departments, you focus ONLY on what the company does as a business, NEVER on their IT department. Write approximately 100 words, without marketing language. You NEVER use placeholders in brackets like [FOUNDING YEAR]."
        
        # Call Claude
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.environ.get("ANTHROPIC_API_KEY"),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "system": system_message,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result['content'][0]['text'].strip()
                
                # Post-processing to ensure no placeholder text
                description = re.sub(r'\[(?:FOUNDING YEAR|founding year|YEAR|year)[^\]]*\]', '', description)
                description = re.sub(r'Founded in\s+,', '', description)
                description = re.sub(r'since\s+\.', '', description)
                description = re.sub(r'has been in operation since\s+\.', 'is in operation. ', description)
                description = re.sub(r'in operation since\s+\.', 'in operation. ', description)
                description = re.sub(r'established in\s+,', '', description)
                description = re.sub(r'established in \[\]', '', description)
                description = re.sub(r'founded in \[\]', '', description)
                description = re.sub(r'\s+,', ',', description)  # Fix spaces before commas
                description = re.sub(r'\s+\.', '.', description)  # Fix spaces before periods
                description = re.sub(r'\.\s*\.', '.', description)  # Fix multiple periods
                description = re.sub(r'\s+', ' ', description)  # Normalize spaces
                
                # Final cleanup to remove redundancy
                try:
                    from domain_classifier.utils.text_processing import remove_redundancy
                    description = remove_redundancy(description, domain)
                except ImportError:
                    # Basic cleanup if remove_redundancy is not available
                    description = re.sub(r'\s+', ' ', description).strip()
                    
                    # Remove common redundant phrases
                    description = re.sub(r'managed service provider.*managed service provider', 'managed service provider', description, flags=re.IGNORECASE)
                    description = re.sub(r'allowing (?:clients|customers) to focus on (?:their|its) core business', '', description, flags=re.IGNORECASE)
                
                # Remove specific technology mentions
                tech_patterns = [
                    r'using \w+ (?:and|,) \w+',
                    r'with \w+ technology',
                    r'based on \w+ platform',
                    r'cloud-based \w+ platform',
                    r'proprietary \w+ software',
                    r'implementing \w+ solutions'
                ]
                
                for pattern in tech_patterns:
                    description = re.sub(pattern, '', description, flags=re.IGNORECASE)
                
                # Remove double spaces and clean up
                description = re.sub(r'\s+', ' ', description).strip()
                
                logger.info(f"Successfully generated description")
                
                # Check word count and log it
                word_count = len(description.split())
                logger.info(f"Generated description has {word_count} words")
                
                # Additional removal of IT Department references for Internal IT companies
                if predicted_class == "Internal IT Department":
                    # Remove mentions of IT department or IT services
                    it_patterns = [
                        r'(?:with|has) (?:an|their own) (?:internal|in-house) IT (?:department|team|staff)',
                        r'(?:maintains|supports) (?:an|their) (?:internal|in-house) IT infrastructure',
                        r'(?:their|its) (?:internal|in-house) IT (?:department|team) (?:manages|supports|maintains)',
                        r'(?:relies on|uses) (?:an|their) internal IT (?:department|team)',
                        r'internal IT needs',
                        r'IT infrastructure',
                        r'IT operations'
                    ]
                    
                    for pattern in it_patterns:
                        description = re.sub(pattern, '', description, flags=re.IGNORECASE)
                    
                    # Cleanup after removals
                    description = re.sub(r'\s+', ' ', description).strip()
                    description = re.sub(r'\s*\.\s*\.', '.', description).strip()
                    
                    # Make sure the description still reads well
                    if description.endswith('.'):
                        description = description[:-1]
                        
                    # Add "a business that" if needed
                    if company_name in description and not re.search(r'is a|is an', description, flags=re.IGNORECASE):
                        description = re.sub(r'(?<=' + re.escape(company_name) + r')\s+', ' is a business that ', description, flags=re.IGNORECASE)
                        
                    # Add a period at the end if needed
                    if not description.endswith('.'):
                        description += '.'
                        
                # Save the description in the classification
                classification['company_description'] = description
                return description
            else:
                logger.error(f"Error calling Claude for description: {response.status_code} - {response.text[:200]}")
                return classification.get('company_description', '')
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            # Fall back to a constructed description
            return _construct_fallback_description(classification, apollo_data)
    except Exception as e:
        logger.error(f"Error generating description: {e}")
        # Fall back to a constructed description
        return _construct_fallback_description(classification, apollo_data)

def _construct_fallback_description(classification: Dict[str, Any], apollo_data: Optional[Dict] = None) -> str:
    """Construct a fallback description when Claude API fails."""
    try:
        domain = classification.get('domain', 'unknown')
        predicted_class = classification.get('predicted_class', 'business')
        
        # Check if this is a "Process Did Not Complete" classification with no Apollo data
        if predicted_class == "Process Did Not Complete":
            if not apollo_data or not isinstance(apollo_data, dict) or not apollo_data.get("name"):
                return f"Unable to retrieve information for {domain} due to insufficient data."
        
        # Get company name
        company_name = None
        if apollo_data and isinstance(apollo_data, dict) and apollo_data.get('name'):
            company_name = apollo_data.get('name')
        else:
            company_name = domain.split('.')[0].capitalize()
        
        # Use Apollo short_description if available
        if apollo_data and isinstance(apollo_data, dict) and apollo_data.get("short_description"):
            return apollo_data.get("short_description")
        
        # Create a basic description
        description = f"{company_name} is "
        
        if predicted_class == "Managed Service Provider":
            description += "a Managed Service Provider"
        elif predicted_class == "Integrator - Commercial A/V":
            description += "a Commercial A/V Integrator"
        elif predicted_class == "Integrator - Residential A/V":
            description += "a Residential A/V Integrator"
        else:  # Internal IT
            description += "a business"
        
        # Add industry if available
        if apollo_data and isinstance(apollo_data, dict) and apollo_data.get('industry'):
            description += f" in the {apollo_data.get('industry')} industry"
        
        description += "."
        
        # Add employee count if available
        if apollo_data and isinstance(apollo_data, dict) and apollo_data.get('employee_count'):
            description += f" They have approximately {apollo_data.get('employee_count')} employees."
        
        # Add service descriptions based on class
        if predicted_class == "Managed Service Provider":
            description += " They provide IT services, technology support, and managed solutions to their clients."
        elif predicted_class == "Integrator - Commercial A/V":
            description += " They provide audio-visual systems and integration services for commercial environments."
        elif predicted_class == "Integrator - Residential A/V":
            description += " They provide home automation and entertainment systems for residential clients."
        
        logger.info(f"Created fallback description for {domain}")
        classification['company_description'] = description
        return description
    
    except Exception as e:
        logger.error(f"Error creating fallback description: {e}")
        # Ultimate fallback
        basic_desc = f"{classification.get('domain', 'The company')} is a {classification.get('predicted_class', 'business')}."
        classification['company_description'] = basic_desc
        return basic_desc
