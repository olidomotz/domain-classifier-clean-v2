from flask import Flask, request, jsonify
from flask_cors import CORS
from llm_classifier import LLMClassifier
from snowflake_connector import SnowflakeConnector
import requests
import time
from urllib.parse import urlparse
import json
import os
import numpy as np
import logging
import traceback
import re
import socket

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Domain override system for special cases
DOMAIN_OVERRIDES = {
    # Format: 'domain': {'classification': 'type', 'confidence': score, 'explanation': 'text'}
    'nwaj.tech': {
        'classification': 'Managed Service Provider',
        'confidence': 85,
        'explanation': 'NWAJ Tech is a cybersecurity and zero trust security provider offering managed security services. They specialize in implementing zero trust security frameworks, which is a clear indication they are a Managed Service Provider focused on cybersecurity services.',
        'confidence_scores': {
            'Managed Service Provider': 85,
            'Integrator - Commercial A/V': 10,
            'Integrator - Residential A/V': 5
        }
    }
}

# Custom JSON encoder to handle problematic types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Configure Flask to use the custom encoder
app.json_encoder = CustomJSONEncoder

# Configuration
LOW_CONFIDENCE_THRESHOLD = 0.7  # Threshold below which we consider a classification "low confidence"
AUTO_RECLASSIFY_THRESHOLD = 0.6  # Threshold below which we automatically reclassify

# Get API keys and settings from environment variables
APIFY_TASK_ID = os.environ.get("APIFY_TASK_ID")
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Initialize LLM classifier directly 
try:
    llm_classifier = LLMClassifier(
        api_key=ANTHROPIC_API_KEY,
        model="claude-3-haiku-20240307"
    )
    logger.info(f"Initialized LLM classifier with model: claude-3-haiku-20240307")
except Exception as e:
    logger.error(f"Failed to initialize LLM classifier: {e}")
    llm_classifier = None

# Initialize Snowflake connector
try:
    snowflake_conn = SnowflakeConnector()
    if not getattr(snowflake_conn, 'connected', False):
        logger.warning("Snowflake connection failed, using fallback")
except Exception as e:
    logger.error(f"Error initializing Snowflake connector: {e}")
    # Define a fallback Snowflake connector for when the real one isn't available
    class FallbackSnowflakeConnector:
        def check_existing_classification(self, domain):
            logger.info(f"Fallback: No existing classification for {domain}")
            return None
            
        def save_domain_content(self, domain, url, content):
            logger.info(f"Fallback: Not saving domain content for {domain}")
            return True, None
            
        def save_classification(self, domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method, llm_explanation):
            logger.info(f"Fallback: Not saving classification for {domain}")
            return True, None
            
        def get_domain_content(self, domain):
            logger.info(f"Fallback: No content for {domain}")
            return None
    
    snowflake_conn = FallbackSnowflakeConnector()

def check_domain_override(domain):
    """
    Check if domain has a manual override classification.
    
    Args:
        domain (str): The domain to check
        
    Returns:
        dict or None: Override classification if available, None otherwise
    """
    domain_lower = domain.lower()
    
    # Check exact match
    if domain_lower in DOMAIN_OVERRIDES:
        logger.info(f"Using override classification for {domain}")
        override = DOMAIN_OVERRIDES[domain_lower]
        
        # Create a standardized result
        result = {
            "domain": domain,
            "predicted_class": override['classification'],
            "confidence_score": override['confidence'],
            "confidence_scores": override.get('confidence_scores', {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0
            }),
            "explanation": override['explanation'],
            "low_confidence": False,
            "detection_method": "manual_override",
            "source": "override",
            "is_parked": False,
            "max_confidence": override['confidence'] / 100.0
        }
        
        # Set service business flag
        result["is_service_business"] = result["predicted_class"] in [
            "Managed Service Provider", 
            "Integrator - Commercial A/V", 
            "Integrator - Residential A/V"
        ]
        
        # Ensure Internal IT Department score is included
        if "Internal IT Department" not in result["confidence_scores"]:
            if result["is_service_business"]:
                # For service businesses, Internal IT Department is always 0
                result["confidence_scores"]["Internal IT Department"] = 0
            else:
                # For non-service businesses, use internal_it_potential or default
                result["confidence_scores"]["Internal IT Department"] = override.get('internal_it_potential', 50)
                # Change the predicted class to "Internal IT Department"
                result["predicted_class"] = "Internal IT Department"
        
        return result
    
    # Check for domain pattern matches (for bulk overrides)
    for pattern, override in DOMAIN_OVERRIDES.items():
        if pattern.startswith('*.') and domain_lower.endswith(pattern[1:]):
            logger.info(f"Using pattern override classification for {domain} (matches {pattern})")
            
            # Create a standardized result (similar to above)
            result = {
                "domain": domain,
                "predicted_class": override['classification'],
                "confidence_score": override['confidence'],
                "confidence_scores": override.get('confidence_scores', {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0
                }),
                "explanation": override['explanation'],
                "low_confidence": False,
                "detection_method": "manual_override",
                "source": "override",
                "is_parked": False,
                "max_confidence": override['confidence'] / 100.0
            }
            
            # Set service business flag
            result["is_service_business"] = result["predicted_class"] in [
                "Managed Service Provider", 
                "Integrator - Commercial A/V", 
                "Integrator - Residential A/V"
            ]
            
            # Ensure Internal IT Department score is included
            if "Internal IT Department" not in result["confidence_scores"]:
                if result["is_service_business"]:
                    # For service businesses, Internal IT Department is always 0
                    result["confidence_scores"]["Internal IT Department"] = 0
                else:
                    # For non-service businesses, use internal_it_potential or default
                    result["confidence_scores"]["Internal IT Department"] = override.get('internal_it_potential', 50)
                    # Change the predicted class to "Internal IT Department"
                    result["predicted_class"] = "Internal IT Department"
            
            return result
    
    return None

def extract_domain_from_email(email):
    """Extract domain from an email address."""
    try:
        # Simple validation of email format
        if not email or '@' not in email:
            return None
            
        # Extract domain portion (after @)
        domain = email.split('@')[-1].strip().lower()
        
        # Basic validation of domain
        if not domain or '.' not in domain:
            return None
            
        return domain
    except Exception as e:
        logger.error(f"Error extracting domain from email: {e}")
        return None

def detect_error_type(error_message):
    """
    Analyze error message to determine the specific type of error.
    
    Args:
        error_message (str): The error message string
        
    Returns:
        tuple: (error_type, detailed_message)
    """
    error_message = str(error_message).lower()
    
    # SSL Certificate errors
    if any(phrase in error_message for phrase in ['certificate has expired', 'certificate verify failed', 'ssl', 'cert']):
        if 'expired' in error_message:
            return "ssl_expired", "The website's SSL certificate has expired."
        elif 'verify failed' in error_message:
            return "ssl_invalid", "The website has an invalid SSL certificate."
        else:
            return "ssl_error", "The website has SSL certificate issues."
    
    # DNS resolution errors
    elif any(phrase in error_message for phrase in ['getaddrinfo failed', 'name or service not known', 'no such host']):
        return "dns_error", "The domain could not be resolved. It may not exist or DNS records may be misconfigured."
    
    # Connection errors
    elif any(phrase in error_message for phrase in ['connection refused', 'connection timed out', 'connection error']):
        return "connection_error", "Could not establish a connection to the website. It may be down or blocking our requests."
    
    # 4XX HTTP errors
    elif any(phrase in error_message for phrase in ['403', 'forbidden', '401', 'unauthorized']):
        return "access_denied", "Access to the website was denied. The site may be blocking automated access."
    elif '404' in error_message or 'not found' in error_message:
        return "not_found", "The requested page was not found on this website."
    
    # 5XX HTTP errors
    elif any(phrase in error_message for phrase in ['500', '502', '503', '504', 'server error']):
        return "server_error", "The website is experiencing server errors."
    
    # Robots.txt or crawling restrictions
    elif any(phrase in error_message for phrase in ['robots.txt', 'disallowed', 'blocked by robots']):
        return "robots_restricted", "The website has restricted automated access in its robots.txt file."
    
    # Default fallback
    return "unknown_error", "An unknown error occurred while trying to access the website."

def crawl_website(url):
    """Crawl a website using Apify with improved multi-stage approach for JavaScript-heavy sites."""
    try:
        logger.info(f"Starting crawl for {url}")
        
        # Quick DNS check before attempting full crawl
        try:
            domain = urlparse(url).netloc
            socket.gethostbyname(domain)
        except socket.gaierror:
            logger.warning(f"Domain {domain} does not resolve - DNS error")
            return None, ("dns_error", "This domain does not exist or cannot be resolved")
        
        # Start the crawl with standard settings
        endpoint = f"https://api.apify.com/v2/actor-tasks/{APIFY_TASK_ID}/runs?token={APIFY_API_TOKEN}"
        payload = {
            "startUrls": [{"url": url}],
            "maxCrawlingDepth": 1,
            "maxCrawlPages": 5,
            "timeoutSecs": 120
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            run_id = response.json()['data']['id']
        except Exception as e:
            logger.error(f"Error starting crawl: {e}")
            return None, detect_error_type(e)
            
        # Wait for crawl to complete
        endpoint = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={APIFY_API_TOKEN}"
        
        max_attempts = 12
        for attempt in range(max_attempts):
            logger.info(f"Checking crawl results, attempt {attempt+1}/{max_attempts}")
            
            try:
                response = requests.get(endpoint, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data:
                        combined_text = ' '.join(item.get('text', '') for item in data if item.get('text'))
                        if combined_text and len(combined_text.strip()) > 100:
                            logger.info(f"Crawl completed, got {len(combined_text)} characters of content")
                            return combined_text, (None, None)
                        elif combined_text:
                            logger.warning(f"Crawl returned minimal content: {len(combined_text)} characters")
                            # Continue trying, might get better results on next attempt
                else:
                    logger.warning(f"Received status code {response.status_code} when checking crawl results")
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout when checking crawl status (attempt {attempt+1})")
            except Exception as e:
                logger.warning(f"Error checking crawl status: {e}")
            
            # Stage 2: Try Puppeteer approach explicitly after a few normal attempts
            if attempt == 4:
                logger.info("Trying Puppeteer-based approach for JavaScript-heavy site...")
                
                # Create a new task run with Puppeteer-specific settings
                puppeteer_payload = {
                    "startUrls": [{"url": url}],
                    "maxCrawlPages": 3,
                    "crawlerType": "playwright:chrome",  # Force Chrome browser
                    "dynamicContentWaitSecs": 15,        # Longer wait for JS content
                    "waitForSelectorSecs": 10,           # Wait for DOM elements
                    "expandIframes": True,               # Get iframe content
                    "clickElementsCssSelector": "button, [aria-expanded='false']",  # Click expandable elements
                    "saveHtml": True,                    # Save raw HTML for backup
                    "proxyConfiguration": {
                        "useApifyProxy": True,
                        "apifyProxyGroups": ["RESIDENTIAL"]  # Use residential proxies
                    }
                }
                
                try:
                    puppeteer_response = requests.post(endpoint.split('dataset')[0] + "?token=" + APIFY_API_TOKEN, 
                                                       json=puppeteer_payload, 
                                                       headers=headers, 
                                                       timeout=30)
                    puppeteer_response.raise_for_status()
                    puppeteer_run_id = puppeteer_response.json()['data']['id']
                    
                    # Wait for Puppeteer crawl to complete (separate from main loop)
                    puppeteer_endpoint = f"https://api.apify.com/v2/actor-runs/{puppeteer_run_id}/dataset/items?token={APIFY_API_TOKEN}"
                    
                    # Give it time to start up
                    time.sleep(5)
                    
                    for p_attempt in range(8):  # Fewer attempts but longer waits
                        logger.info(f"Checking Puppeteer crawl results, attempt {p_attempt+1}/8")
                        
                        try:
                            p_response = requests.get(puppeteer_endpoint, timeout=15)
                            
                            if p_response.status_code == 200:
                                p_data = p_response.json()
                                
                                if p_data:
                                    # Try to get text content from all fields that might have it
                                    text_fields = []
                                    for item in p_data:
                                        if item.get('text'):
                                            text_fields.append(item.get('text', ''))
                                        # Also try to extract from HTML if text is minimal
                                        elif item.get('html') and (not item.get('text') or len(item.get('text', '')) < 100):
                                            # Simple HTML to text extraction
                                            html_text = re.sub(r'<[^>]+>', ' ', item.get('html', ''))
                                            html_text = re.sub(r'\s+', ' ', html_text).strip()
                                            if len(html_text) > 100:
                                                text_fields.append(html_text)
                                    
                                    puppeteer_text = ' '.join(text_fields)
                                    
                                    if puppeteer_text and len(puppeteer_text.strip()) > 100:
                                        logger.info(f"Puppeteer crawl successful, got {len(puppeteer_text)} characters")
                                        return puppeteer_text, (None, None)
                        except Exception as e:
                            logger.warning(f"Error checking Puppeteer crawl: {e}")
                            
                        time.sleep(12)  # Longer wait between Puppeteer checks
                        
                except Exception as e:
                    logger.error(f"Error with Puppeteer approach: {e}")
            
            # Stage 3: Direct request fallback
            if attempt == 7:  # After about 70 seconds
                logger.info("Trying direct request fallback...")
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5'
                    }
                    direct_response = requests.get(url, headers=headers, timeout=15)
                    
                    if direct_response.status_code == 200:
                        # Use a simple content extraction approach
                        text_content = direct_response.text
                        
                        # Extract readable text by removing HTML tags
                        clean_text = re.sub(r'<[^>]+>', ' ', text_content)
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                        
                        if clean_text and len(clean_text) > 100:
                            logger.info(f"Direct request successful, got {len(clean_text)} characters")
                            return clean_text, (None, None)
                        else:
                            logger.warning(f"Direct request returned minimal content: {len(clean_text)} characters")
                except Exception as e:
                    error_type, error_detail = detect_error_type(e)
                    logger.warning(f"Direct request failed: {e} (Type: {error_type})")
            
            # Wait between attempts for the main crawl
            if attempt < max_attempts - 1:
                time.sleep(10)
        
        # If we still don't have good content but have some minimal content,
        # return it rather than failing completely
        if 'combined_text' in locals() and combined_text:
            logger.warning(f"Using minimal content ({len(combined_text)} chars) as fallback")
            return combined_text, (None, None)
            
        # Try one last direct request if we have nothing else
        try:
            logger.info("Final direct request attempt...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            final_response = requests.get(url, headers=headers, timeout=15)
            
            if final_response.status_code == 200:
                # Just get any text we can
                final_text = re.sub(r'<[^>]+>', ' ', final_response.text)
                final_text = re.sub(r'\s+', ' ', final_text).strip()
                
                if len(final_text) > 19:  # Better than the 19 chars we got before
                    logger.info(f"Final direct request got {len(final_text)} characters")
                    return final_text, (None, None)
        except Exception:
            pass
            
        logger.warning(f"Crawl timed out after all attempts")
        return None, ("timeout", "The website took too long to respond or has minimal crawlable content.")
    except Exception as e:
        error_type, error_detail = detect_error_type(e)
        logger.error(f"Error crawling website: {e} (Type: {error_type})")
        return None, (error_type, error_detail)

def save_to_snowflake(domain, url, content, classification):
    """Save classification data to Snowflake"""
    try:
        # Always save the domain content
        logger.info(f"Saving content to Snowflake for {domain}")
        snowflake_conn.save_domain_content(
            domain=domain, url=url, content=content
        )

        # Ensure max_confidence exists
        if 'max_confidence' not in classification:
            confidence_scores = classification.get('confidence_scores', {})
            max_confidence = max(confidence_scores.values()) if confidence_scores else 0.5
            classification['max_confidence'] = max_confidence

        # Set low_confidence flag based on confidence threshold
        if 'low_confidence' not in classification:
            classification['low_confidence'] = classification['max_confidence'] < LOW_CONFIDENCE_THRESHOLD

        # Get explanation directly from classification
        llm_explanation = classification.get('llm_explanation', '')
        
        # If explanation is too long, trim it properly at a sentence boundary
        if len(llm_explanation) > 4000:
            # Find the last period before 3900 chars
            last_period_index = llm_explanation[:3900].rfind('.')
            if last_period_index > 0:
                llm_explanation = llm_explanation[:last_period_index + 1]
            else:
                # If no period found, just truncate with an ellipsis
                llm_explanation = llm_explanation[:3900] + "..."
            
        # Create model metadata
        model_metadata = {
            'model_version': '1.0',
            'llm_model': 'claude-3-haiku-20240307'
        }
        
        # Convert model metadata to JSON string
        model_metadata_json = json.dumps(model_metadata)[:4000]  # Limit size
            
        # Special case for parked domains - save as "Parked Domain" if is_parked flag is set
        company_type = classification.get('predicted_class', 'Unknown')
        if classification.get('is_parked', False):
            company_type = "Parked Domain"
        
        logger.info(f"Saving classification to Snowflake: {domain}, {company_type}")
        snowflake_conn.save_classification(
            domain=domain,
            company_type=str(company_type),
            confidence_score=float(classification['max_confidence']),
            all_scores=json.dumps(classification.get('confidence_scores', {}))[:4000],  # Limit size
            model_metadata=model_metadata_json,
            low_confidence=bool(classification.get('low_confidence', False)),
            detection_method=str(classification.get('detection_method', 'llm_classification')),
            llm_explanation=llm_explanation  # Add explanation directly to save_classification
        )
        
        return True
    except Exception as e:
        logger.error(f"Error saving to Snowflake: {e}\n{traceback.format_exc()}")
        return False

def create_error_result(domain, error_type=None, error_detail=None, email=None):
    """
    Create a standardized error response based on the error type.
    
    Args:
        domain (str): The domain being processed
        error_type (str): The type of error detected
        error_detail (str): Detailed explanation of the error
        email (str, optional): Email address if processing an email
        
    Returns:
        dict: Standardized error response
    """
    # Default error response
    error_result = {
        "domain": domain,
        "error": "Failed to crawl website",
        "predicted_class": "Unknown",
        "confidence_score": 0,
        "confidence_scores": {
            "Managed Service Provider": 0,
            "Integrator - Commercial A/V": 0,
            "Integrator - Residential A/V": 0,
            "Internal IT Department": 0  # Include Internal IT Department with score 0
        },
        "low_confidence": True,
        "is_crawl_error": True
    }
    
    # Add email if provided
    if email:
        error_result["email"] = email
    
    # Default explanation
    explanation = f"We were unable to retrieve content from {domain}. This could be due to a server timeout or the website being unavailable. Without analyzing the website content, we cannot determine the company type."
    
    # Enhanced error handling based on error type
    if error_type:
        error_result["error_type"] = error_type
        
        if error_type.startswith('ssl_'):
            explanation = f"We couldn't analyze {domain} because of SSL certificate issues. "
            if error_type == 'ssl_expired':
                explanation += f"The website's SSL certificate has expired. This is a security issue with the target website, not our classification service."
            elif error_type == 'ssl_invalid':
                explanation += f"The website has an invalid SSL certificate. This is a security issue with the target website, not our classification service."
            else:
                explanation += f"This is a security issue with the target website, not our classification service."
            
            error_result["is_ssl_error"] = True
            
        elif error_type == 'dns_error':
            explanation = f"We couldn't analyze {domain} because the domain could not be resolved. This typically means the domain doesn't exist or its DNS records are misconfigured."
            error_result["is_dns_error"] = True
            
        elif error_type == 'connection_error':
            explanation = f"We couldn't analyze {domain} because a connection couldn't be established. The website may be down, temporarily unavailable, or blocking our requests."
            error_result["is_connection_error"] = True
            
        elif error_type == 'access_denied':
            explanation = f"We couldn't analyze {domain} because access was denied. The website may be blocking automated access or requiring authentication."
            error_result["is_access_denied"] = True
            
        elif error_type == 'not_found':
            explanation = f"We couldn't analyze {domain} because the main page was not found. The website may be under construction or have moved to a different URL."
            error_result["is_not_found"] = True
            
        elif error_type == 'server_error':
            explanation = f"We couldn't analyze {domain} because the website is experiencing server errors. This is an issue with the target website, not our classification service."
            error_result["is_server_error"] = True
            
        elif error_type == 'robots_restricted':
            explanation = f"We couldn't analyze {domain} because the website restricts automated access. This is a policy set by the website owner."
            error_result["is_robots_restricted"] = True
            
        elif error_type == 'timeout':
            explanation = f"We couldn't analyze {domain} because the website took too long to respond. The website may be experiencing performance issues or temporarily unavailable."
            error_result["is_timeout"] = True
            
        # If we have a specific error detail, use it to enhance the explanation
        if error_detail:
            explanation += f" {error_detail}"
    
    error_result["explanation"] = explanation
    return error_result

def validate_result_consistency(result, domain):
    """
    Validate and ensure consistency between predicted_class, confidence scores, and explanation.
    
    Args:
        result (dict): The classification result
        domain (str): The domain
        
    Returns:
        dict: The validated and consistent result
    """
    if not result:
        return result
        
    # First, ensure predicted_class is never null
    if result.get("predicted_class") is None:
        # Extract a class from the explanation if possible
        explanation = result.get("explanation", "")
        
        # Look for existing company type in explanation
        if "managed service provider" in explanation.lower() or "msp" in explanation.lower():
            result["predicted_class"] = "Managed Service Provider"
        elif "commercial a/v" in explanation.lower() or "commercial integrator" in explanation.lower():
            result["predicted_class"] = "Integrator - Commercial A/V"
        elif "residential a/v" in explanation.lower() or "residential integrator" in explanation.lower():
            result["predicted_class"] = "Integrator - Residential A/V"
        elif "non-service business" in explanation.lower() or "not a service" in explanation.lower():
            result["predicted_class"] = "Internal IT Department"  # Changed from "Non-Service Business"
        elif "vacation rental" in explanation.lower() or "travel" in explanation.lower():
            result["predicted_class"] = "Internal IT Department"  # Changed from "Non-Service Business"
        elif "parked domain" in explanation.lower() or "website is parked" in explanation.lower():
            result["predicted_class"] = "Parked Domain"
        else:
            # Default to Unknown
            result["predicted_class"] = "Unknown"
            
        logger.warning(f"Fixed null predicted_class for {domain} to {result['predicted_class']}")
    
    # Check for cases where confidence score is very low for service businesses
    # Only do this for fresh classifications, not cached ones
    if "confidence_score" in result and result["confidence_score"] <= 15 and result.get("source") != "cached":
        explanation = result.get("explanation", "").lower()
        if "non-service business" in explanation or "not a service" in explanation:
            if result["predicted_class"] in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                # If explanation mentions non-service but class is a service type, fix it
                logger.warning(f"Correcting predicted_class from {result['predicted_class']} to Internal IT Department based on explanation")
                result["predicted_class"] = "Internal IT Department"  # Changed from "Non-Service Business"
                
    # Ensure "Process Did Not Complete" has 0% confidence
    if result.get("predicted_class") == "Process Did Not Complete":
        result["confidence_score"] = 0
        if "confidence_scores" in result:
            result["confidence_scores"] = {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0,
                "Internal IT Department": 0  # Ensure Internal IT Department is included with 0 score
            }
    
    # Make sure we have confidence_scores
    if "confidence_scores" not in result:
        result["confidence_scores"] = {}
        
    # For Internal IT Department or Non-Service Business, ensure Internal IT Department score is included
    if result.get("predicted_class") == "Internal IT Department" or result.get("predicted_class") == "Non-Service Business":
        # Convert Non-Service Business to Internal IT Department for consistency
        if result.get("predicted_class") == "Non-Service Business":
            result["predicted_class"] = "Internal IT Department"
            
        # Add Internal IT Department if not present
        if "Internal IT Department" not in result["confidence_scores"]:
            # Try to extract IT potential from explanation
            explanation = result.get("explanation", "")
            it_potential = 60  # Default
            
            # Look for internal IT potential in explanation
            it_match = re.search(r'internal IT.*?(\d+)[/\s]*100', explanation.lower())
            if it_match:
                try:
                    it_potential = int(it_match.group(1))
                except (ValueError, TypeError):
                    pass
            
            # Update confidence scores
            result["confidence_scores"]["Internal IT Department"] = it_potential
            result["confidence_scores"]["Managed Service Provider"] = 5
            result["confidence_scores"]["Integrator - Commercial A/V"] = 3
            result["confidence_scores"]["Integrator - Residential A/V"] = 2
            
            # Set overall confidence to 80% for non-service
            result["confidence_score"] = 80
    
    # For service businesses, ensure Internal IT Department is included with 0 score
    elif result.get("predicted_class") in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
        if "Internal IT Department" not in result["confidence_scores"]:
            result["confidence_scores"]["Internal IT Department"] = 0
    
    # For unknown or parked domains, ensure Internal IT Department is included with 0 score
    elif result.get("predicted_class") in ["Unknown", "Parked Domain"]:
        if "Internal IT Department" not in result["confidence_scores"]:
            result["confidence_scores"]["Internal IT Department"] = 0
    
    # Fix step numbering if it's off (e.g. starting at 6 instead of 1)
    explanation = result.get("explanation", "")
    fixed_steps = []
    step_pattern = re.compile(r'step\s*(\d+)[:\.]?\s*([^$]+?)(?=step\s*\d+[:\.]|$)', re.IGNORECASE | re.DOTALL)
    matches = step_pattern.findall(explanation)
    
    if matches and int(matches[0][0]) > 5:  # If steps start higher than 5
        # Renumber all steps to start from 1
        for i, (_, content) in enumerate(matches, 1):
            fixed_steps.append(f"STEP {i}: {content.strip()}")
            
        # Rebuild explanation with fixed step numbering
        if fixed_steps:
            result["explanation"] = "\n\n".join(fixed_steps)
    
    # Ensure explanation has step-by-step format if it's not a parked domain or process did not complete
    if (result.get("predicted_class") not in ["Parked Domain", "Process Did Not Complete", "Unknown"] 
        and "explanation" in result):
        explanation = result["explanation"]
        
        # Check if the explanation already has the STEP format
        if not any(f"STEP {i}" in explanation for i in range(1, 6)) and not any(f"STEP {i}:" in explanation for i in range(1, 6)):
            # If not already in step format and not numbered like "1:", "2:", etc.
            if not any(f"{i}:" in explanation for i in range(1, 6)):
                domain_name = domain or "This domain"
                predicted_class = result.get("predicted_class", "Unknown")
                is_service = predicted_class in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]
                
                # Create a structured explanation with STEP format
                new_explanation = f"Based on the website content, {domain_name} is classified as a {predicted_class}\n\n"
                new_explanation += f"STEP 1: The website content provides sufficient information to analyze and classify the business, so the processing status is successful\n\n"
                new_explanation += f"STEP 2: The domain is not parked, under construction, or for sale, so it is not a Parked Domain\n\n"
                
                if is_service:
                    confidence = result.get("confidence_score", 80)
                    new_explanation += f"STEP 3: The company is a service business that provides services to other businesses\n\n"
                    new_explanation += f"STEP 4: Based on the service offerings described, this company is classified as a {predicted_class} with {confidence}% confidence\n\n"
                    new_explanation += f"STEP 5: Since this is classified as a service business, the internal IT potential is set to 0/100\n\n"
                else:
                    # Try to extract internal IT potential from confidence scores
                    it_potential = 50  # Default
                    if "confidence_scores" in result and "Internal IT Department" in result["confidence_scores"]:
                        it_potential = result["confidence_scores"]["Internal IT Department"]
                    
                    new_explanation += f"STEP 3: The company is NOT a service/management business that provides ongoing IT or A/V services to clients\n\n"
                    new_explanation += f"STEP 4: Since this is not a service business, we classify it as Internal IT Department\n\n"  # Changed from "Non-Service Business"
                    new_explanation += f"STEP 5: As a non-service business, we assess its internal IT potential at {it_potential}/100\n\n"
                    
                # Include the original explanation as a summary
                new_explanation += f"In summary: {explanation}"
                result["explanation"] = new_explanation
    
    # Ensure explanation is consistent with predicted_class
    if result.get("explanation") and "based on" in result["explanation"].lower():
        explanation = result["explanation"]
        # If explanation mentions company was "previously classified as a None"
        if "previously classified as a None" in explanation:
            # Fix this wording
            explanation = explanation.replace(
                f"previously classified as a None", 
                f"previously classified as a {result.get('predicted_class', 'company')}"
            )
            result["explanation"] = explanation
    
    # Handle any legacy "Corporate IT" keys that might still exist
    if "Corporate IT" in result.get("confidence_scores", {}):
        score = result["confidence_scores"].pop("Corporate IT")
        result["confidence_scores"]["Internal IT Department"] = score
        
    # Handle legacy "Non-Service Business" predicted class
    if result.get("predicted_class") == "Non-Service Business":
        result["predicted_class"] = "Internal IT Department"
    
    return result

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok", 
        "llm_available": llm_classifier is not None,
        "snowflake_connected": getattr(snowflake_conn, 'connected', False)
    }), 200

@app.route('/classify-domain', methods=['POST', 'OPTIONS'])
def classify_domain():
    """Direct API that classifies a domain or email and returns the result"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        input_value = data.get('url', '').strip()
        
        # [UPDATED] Change default for force_reclassify to True
        force_reclassify = data.get('force_reclassify', True)  # Default to TRUE for consistent behavior
        
        use_existing_content = data.get('use_existing_content', False)
        
        if not input_value:
            return jsonify({"error": "URL or email is required"}), 400
        
        # Check if input is an email (contains @)
        is_email = '@' in input_value
        email = None
        if is_email:
            # Extract domain from email
            email = input_value
            domain = extract_domain_from_email(email)
            if not domain:
                return jsonify({"error": "Invalid email format"}), 400
            logger.info(f"Extracted domain '{domain}' from email '{email}'")
        else:
            # Process as domain/URL
            # Format URL properly
            if not input_value.startswith('http'):
                input_value = 'https://' + input_value
                
            # Extract domain
            parsed_url = urlparse(input_value)
            domain = parsed_url.netloc
            if not domain:
                domain = parsed_url.path
                
            # Remove www. if present
            if domain.startswith('www.'):
                domain = domain[4:]
        
        if not domain:
            return jsonify({"error": "Invalid URL or email"}), 400
            
        url = f"https://{domain}"
        logger.info(f"Processing classification request for {domain}")
        
        # Check for domain override before any other processing
        domain_override = check_domain_override(domain)
        if domain_override:
            # Add email to response if input was an email
            if email:
                domain_override["email"] = email
                
            # Add website URL for clickable link
            domain_override["website_url"] = url
            
            # Return the override directly
            logger.info(f"Sending override response to client: {json.dumps(domain_override)}")
            return jsonify(domain_override), 200
        
        # Check for existing classification if not forcing reclassification
        if not force_reclassify:
            existing_record = snowflake_conn.check_existing_classification(domain)
            
            # [UPDATED SECTION] - Fixed cached results handling
            if existing_record:
                logger.info(f"Found existing classification for {domain}")
                logger.info(f"Retrieved record from Snowflake: {existing_record}")
                
                # Extract confidence scores
                confidence_scores = {}
                try:
                    confidence_scores = json.loads(existing_record.get('all_scores', '{}'))
                except Exception as e:
                    logger.warning(f"Could not parse all_scores: {str(e)}")
                
                # Extract LLM explanation directly from the LLM_EXPLANATION column
                llm_explanation = existing_record.get('LLM_EXPLANATION', '')
                
                # If LLM_EXPLANATION is not available, try to get it from model_metadata
                if not llm_explanation:
                    try:
                        metadata = json.loads(existing_record.get('model_metadata', '{}'))
                        llm_explanation = metadata.get('llm_explanation', '')
                    except Exception as e:
                        logger.warning(f"Could not parse model_metadata for {domain}: {e}")
                
                # Ensure we have an explanation
                if not llm_explanation:
                    llm_explanation = f"The domain {domain} was previously classified as a {existing_record.get('company_type')} based on analysis of website content."
                
                # Add low_confidence flag based on confidence score
                confidence_score = existing_record.get('confidence_score', 0.5)
                low_confidence = existing_record.get('low_confidence', confidence_score < LOW_CONFIDENCE_THRESHOLD)
                
                # Check if it's a parked domain (stored as "Parked Domain" in company_type)
                is_parked = existing_record.get('company_type') == "Parked Domain"
                
                # Process confidence scores with type handling
                processed_scores = {}
                for category, score in confidence_scores.items():
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
                
                # Get original predicted class from database 
                orig_predicted_class = existing_record.get('company_type')
                
                # Handle "Non-Service Business" to "Internal IT Department" conversion
                if orig_predicted_class == "Non-Service Business":
                    orig_predicted_class = "Internal IT Department"
                
                # For service businesses, preserve original scores but ensure they're differentiated
                if orig_predicted_class in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                    # Check if we need to fix identical scores
                    identical_values = len(set(processed_scores.values())) <= 1
                    all_values_zero = all(value == 0 for value in processed_scores.values())
                    
                    if identical_values and not all_values_zero:
                        logger.warning("Cached response has identical confidence scores, fixing...")
                        # Set up differentiated scores
                        if orig_predicted_class == "Managed Service Provider":
                            processed_scores = {
                                "Managed Service Provider": 90,
                                "Integrator - Commercial A/V": 10,
                                "Integrator - Residential A/V": 10
                            }
                        elif orig_predicted_class == "Integrator - Commercial A/V":
                            processed_scores = {
                                "Integrator - Commercial A/V": 90,
                                "Managed Service Provider": 10,
                                "Integrator - Residential A/V": 10
                            }
                        elif orig_predicted_class == "Integrator - Residential A/V":
                            processed_scores = {
                                "Integrator - Residential A/V": 90,
                                "Integrator - Commercial A/V": 10, 
                                "Managed Service Provider": 10
                            }
                    
                    # Ensure Internal IT Department is included with score 0 for service businesses
                    processed_scores["Internal IT Department"] = 0
                
                # For Internal IT Department, ensure Internal IT Department score is included
                elif orig_predicted_class == "Internal IT Department":
                    # Try to extract internal IT potential from explanation or set a default
                    it_potential = 60  # Default value
                    it_match = re.search(r'internal IT.*?(\d+)[/\s]*100', llm_explanation)
                    if it_match:
                        try:
                            it_potential = int(it_match.group(1))
                        except (ValueError, TypeError):
                            pass
                        
                    # Add Internal IT Department score
                    processed_scores["Internal IT Department"] = it_potential
                    
                    # Ensure service scores are low
                    processed_scores["Managed Service Provider"] = min(processed_scores.get("Managed Service Provider", 5), 10)
                    processed_scores["Integrator - Commercial A/V"] = min(processed_scores.get("Integrator - Commercial A/V", 3), 10)
                    processed_scores["Integrator - Residential A/V"] = min(processed_scores.get("Integrator - Residential A/V", 2), 10)
                
                # Handle legacy "Corporate IT" key in confidence scores
                if "Corporate IT" in processed_scores:
                    score = processed_scores.pop("Corporate IT")
                    processed_scores["Internal IT Department"] = score
                    
                # Return the cached classification
                result = {
                    "domain": domain,
                    "predicted_class": orig_predicted_class,  # Use original class directly
                    "confidence_score": int(existing_record.get('confidence_score', 0.5) * 100),
                    "confidence_scores": processed_scores,
                    "explanation": llm_explanation,  # Include the explanation here
                    "low_confidence": low_confidence,
                    "detection_method": existing_record.get('detection_method', 'api'),
                    "source": "cached",
                    "is_parked": is_parked,
                    "website_url": url  # Add website URL for clickable link
                }
                
                # Add email to response if input was an email
                if email:
                    result["email"] = email
                
                # Ensure result consistency
                result = validate_result_consistency(result, domain)
                
                # Set proper confidence score for service businesses based on category score
                if result["predicted_class"] in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                    if result["predicted_class"] in result["confidence_scores"]:
                        result["confidence_score"] = result["confidence_scores"][result["predicted_class"]]
                
                # Log the response for debugging
                logger.info(f"Sending response to client: {json.dumps(result)}")
                    
                return jsonify(result), 200
        
        # Try to get content (either from DB or by crawling)
        content = None
        
        # If reclassifying or using existing content, try to get existing content first
        if force_reclassify or use_existing_content:
            try:
                content = snowflake_conn.get_domain_content(domain)
                if content:
                    logger.info(f"Using existing content for {domain}")
            except (AttributeError, Exception) as e:
                logger.warning(f"Could not get existing content, will crawl instead: {e}")
                content = None

        # If we specifically requested to use existing content but none was found
        if use_existing_content and not content:
            error_result = {
                "domain": domain,
                "error": "No existing content found",
                "predicted_class": "Unknown",
                "confidence_score": 0,
                "confidence_scores": {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0,
                    "Internal IT Department": 0
                },
                "explanation": f"We could not find previously stored content for {domain}. Please try recrawling instead.",
                "low_confidence": True,
                "no_existing_content": True,
                "website_url": url  # Add website URL for clickable link
            }
            
            # Add email to response if input was an email
            if email:
                error_result["email"] = email
                
            return jsonify(error_result), 404
        
        # If no content yet and we're not using existing content, crawl the website
        error_type = None
        error_detail = None
        
        if not content and not use_existing_content:
            logger.info(f"Crawling website for {domain}")
            content, (error_type, error_detail) = crawl_website(url)
            
            if not content:
                error_result = create_error_result(domain, error_type, error_detail, email)
                error_result["website_url"] = url  # Add website URL for clickable link
                return jsonify(error_result), 503  # Service Unavailable
        
        # Classify the content
        if not llm_classifier:
            error_result = {
                "domain": domain, 
                "error": "LLM classifier is not available",
                "predicted_class": "Unknown",
                "confidence_score": 0,
                "confidence_scores": {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0,
                    "Internal IT Department": 0
                },
                "explanation": "Our classification system is temporarily unavailable. Please try again later. This issue has been logged and will be addressed by our technical team.",
                "low_confidence": True,
                "website_url": url  # Add website URL for clickable link
            }
            
            # Add email to error response if input was an email
            if email:
                error_result["email"] = email
                
            return jsonify(error_result), 500
            
        logger.info(f"Classifying content for {domain}")
        classification = llm_classifier.classify(content, domain)
        
        if not classification:
            error_result = {
                "domain": domain,
                "error": "Classification failed",
                "predicted_class": "Unknown",
                "confidence_score": 0,
                "confidence_scores": {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0,
                    "Internal IT Department": 0
                },
                "explanation": f"We encountered an issue while analyzing {domain}. Although content was retrieved from the website, our classification system was unable to process it properly. This could be due to unusual formatting or temporary system limitations.",
                "low_confidence": True,
                "website_url": url  # Add website URL for clickable link
            }
            
            # Add email to error response if input was an email
            if email:
                error_result["email"] = email
                
            return jsonify(error_result), 500
        
        # Save to Snowflake (always save, even for reclassifications)
        save_to_snowflake(domain, url, content, classification)
        
        # Create the response with properly differentiated confidence scores
        if classification.get("is_parked", False):
            # Special case for parked domains
            result = {
                "domain": domain,
                "predicted_class": "Parked Domain",  # Clear indicator in the UI
                "confidence_score": 0,  # Zero confidence rather than 5%
                "confidence_scores": {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0,
                    "Internal IT Department": 0
                },
                "explanation": classification.get('llm_explanation', 'This appears to be a parked or inactive domain without business-specific content.'),
                "low_confidence": True,
                "detection_method": classification.get('detection_method', 'parked_domain_detection'),
                "source": "fresh",
                "is_parked": True,
                "website_url": url  # Add website URL for clickable link
            }
        else:
            # [UPDATED SECTION] - Fixed Fresh Classification Handling
            
            # Normal case with confidence scores as integers (1-100)
            # Get max confidence 
            max_confidence = 0
            if "max_confidence" in classification:
                if isinstance(classification["max_confidence"], float) and classification["max_confidence"] <= 1.0:
                    max_confidence = int(classification["max_confidence"] * 100)
                else:
                    max_confidence = int(classification["max_confidence"])
            else:
                # If max_confidence not set, find the highest score
                confidence_scores = classification.get('confidence_scores', {})
                if confidence_scores:
                    max_score = max(confidence_scores.values())
                    if isinstance(max_score, float) and max_score <= 1.0:
                        max_confidence = int(max_score * 100)
                    else:
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
                elif pred_class == "Integrator - Residential A/V":  # Residential A/V
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
                        "Internal IT Department": internal_it_potential  # Add Internal IT Department score
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
                
            result = {
                "domain": domain,
                "predicted_class": classification.get('predicted_class'),
                "confidence_score": max_confidence,
                "confidence_scores": processed_scores,
                "explanation": explanation,  # Include the explanation here
                "low_confidence": classification.get('low_confidence', False),
                "detection_method": classification.get('detection_method', 'api'),
                "source": "fresh",
                "is_parked": False,
                "website_url": url  # Add website URL for clickable link
            }

        # Add email to response if input was an email
        if email:
            result["email"] = email
            
        # Ensure result consistency
        result = validate_result_consistency(result, domain)
        
        # Log the response for debugging
        logger.info(f"Sending response to client: {json.dumps(result)}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {e}\n{traceback.format_exc()}")
        # Try to identify the error type if possible
        error_type, error_detail = detect_error_type(str(e))
        error_result = create_error_result(domain if 'domain' in locals() else "unknown", error_type, error_detail, email if 'email' in locals() else None)
        error_result["error"] = str(e)  # Add the actual error message
        if 'url' in locals():
            error_result["website_url"] = url  # Add website URL for clickable link
        return jsonify(error_result), 500

@app.route('/classify-email', methods=['POST', 'OPTIONS'])
def classify_email():
    """Alias for classify-domain that redirects email classification requests"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        email = data.get('email', '').strip()
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        # Create a new request with the email as URL
        new_data = {
            'url': email,
            'force_reclassify': data.get('force_reclassify', False),
            'use_existing_content': data.get('use_existing_content', False)
        }
        
        # Forward to classify_domain by calling it directly with the new data
        # Use actual request for context but modify just the .json attribute
        # We need to be careful to use a copy and not modify the actual request
        original_json = request.json
        try:
            # Store the original json and use a context-like pattern
            _temp_request_json = new_data
            
            # Since we can't modify request.json directly,
            # we'll monkey patch the request.get_json function temporarily
            original_get_json = request.get_json
            
            def patched_get_json(*args, **kwargs):
                return _temp_request_json
                
            request.get_json = patched_get_json
            
            # Now call classify_domain, which will use our patched get_json
            result = classify_domain()
            
            # Return the result directly
            return result
            
        finally:
            # Restore original get_json
            if 'original_get_json' in locals():
                request.get_json = original_get_json
        
    except Exception as e:
        logger.error(f"Error processing email classification request: {e}\n{traceback.format_exc()}")
        error_type, error_detail = detect_error_type(str(e))
        error_result = create_error_result("unknown", error_type, error_detail)
        error_result["error"] = str(e)
        return jsonify(error_result), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
