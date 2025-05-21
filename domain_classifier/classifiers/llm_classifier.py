"""Updated LLM classifier to handle missing content correctly and fix JSON parsing issues."""
import requests
import logging
import json
import re
import os
import time
import traceback
from typing import Dict, Any, Optional, Tuple, List

from domain_classifier.classifiers.prompt_builder import build_decision_tree_prompt, load_examples_from_knowledge_base
from domain_classifier.classifiers.decision_tree import (
    is_parked_domain,
    check_special_domain_cases,
    create_process_did_not_complete_result,
    create_parked_domain_result,
    check_industry_context
)
from domain_classifier.utils.text_processing import (
    extract_json,
    clean_json_string,
    detect_minimal_content,
    generate_one_line_description,
    extract_company_description
)
from domain_classifier.classifiers.result_validator import validate_classification, check_confidence_alignment, ensure_step_format
from domain_classifier.classifiers.fallback_classifier import fallback_classification, parse_free_text
from domain_classifier.utils.domain_analysis import analyze_domain_words
from domain_classifier.utils.cross_validator import reconcile_classification

# Set up logging
logger = logging.getLogger(__name__)

class LLMClassifier:
    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        """
        Initialize the LLM classifier with API key and model.
        
        Args:
            api_key: The API key for Claude API
            model: The Claude model to use
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. LLM classification will not be available.")
            
        self.model = model
        logger.info(f"Initialized LLM classifier with model: {model}")
        
        # Initialize metrics tracking
        self.vector_attempts = 0
        self.vector_successes = 0
        
        # Cache for examples to avoid reloading
        self.loaded_examples = None
        
        # Performance optimization: Set up a requests session for reuse
        self.session = requests.Session()
        
        # Add a retry adapter with exponential backoff
        retry_strategy = requests.adapters.Retry(
            total=3,  # Maximum number of retries
            backoff_factor=1,  # Will retry in 1, 2, 4 seconds
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["POST"]  # Only retry POST requests
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

    def classify(self, content: str = None, domain: str = None, apollo_data: Optional[Dict] = None, 
                 ai_data: Optional[Dict] = None, use_vector_classification: bool = True, 
                 track_metrics: bool = True) -> Dict[str, Any]:
        """
        Classify using available information - works even with missing content.
        
        Args:
            content: The website content (can be None)
            domain: Domain name (required)
            apollo_data: Optional Apollo data
            ai_data: Optional AI-extracted data
            use_vector_classification: Whether to use vector classification
            track_metrics: Whether to track vector classification metrics
            
        Returns:
            dict: The classification results including predicted class and confidence scores
        """
        logger.info(f"Starting classification for domain: {domain or 'unknown'}")
        
        # STEP 1: Validate domain is provided
        if not domain:
            logger.warning("No domain provided for classification")
            return create_process_did_not_complete_result("unknown")
        
        # STEP 2: Check for parked domain if content is available
        if content and is_parked_domain(content, domain):
            logger.info(f"Domain {domain} is detected as a parked domain")
            return create_parked_domain_result(domain)
            
        # STEP 3: Try vector classification if content is available
        if content and use_vector_classification and len(content.strip()) > 100:
            try:
                if track_metrics:
                    self.vector_attempts += 1
                    
                vector_result = self.classify_with_vectors(content, domain)
                
                # If vector classification returned a result
                if vector_result and vector_result.get("detection_method") == "vector_similarity":
                    # Use vector classification result
                    logger.info(f"Successfully classified {domain} using vector similarity")
                    
                    # Track successful vector classification
                    if track_metrics:
                        self.vector_successes += 1
                        success_rate = (self.vector_successes / self.vector_attempts) * 100
                        logger.info(f"Vector classification success rate: {success_rate:.1f}% ({self.vector_successes}/{self.vector_attempts})")
                    
                    # Store original classification for potential fallback
                    original_predicted_class = vector_result.get("predicted_class")
                    
                    # Perform cross-validation if we have industry data
                    if apollo_data or ai_data:
                        # Check for IT services industry before cross-validation
                        if self._is_it_services_industry(apollo_data, ai_data):
                            logger.info(f"IT services industry detected for {domain}, preserving MSP classification if applicable")
                            # Preserve MSP classification for IT services
                            if original_predicted_class == "Managed Service Provider":
                                # Skip cross-validation for MSPs in IT services industry
                                logger.info(f"Preserving MSP classification for IT services industry: {domain}")
                            else:
                                vector_result = reconcile_classification(vector_result, apollo_data, ai_data)
                        else:
                            # Regular cross-validation for non-IT services
                            vector_result = reconcile_classification(vector_result, apollo_data, ai_data)
                    
                    # Add domain to result
                    vector_result["domain"] = domain
                    
                    # Add one-line description if not present
                    if "company_one_line" not in vector_result:
                        vector_result["company_one_line"] = generate_one_line_description(
                            content=content[:5000] if content else "",
                            predicted_class=vector_result.get('predicted_class', ''),
                            domain=domain,
                            company_description=vector_result.get('company_description', '')
                        )
                    
                    return vector_result
            except Exception as e:
                logger.warning(f"Error during vector-based classification for {domain}: {e}")
                # Continue to LLM classification
        
        # STEP 4: Use LLM for classification - works with or without content
        logger.info(f"Using LLM for classification of {domain}")
        llm_result = self.classify_with_llm(content, domain)
        
        # Store original classification for potential fallback
        original_predicted_class = llm_result.get("predicted_class")
        
        # Check for IT services industry before cross-validation
        if apollo_data or ai_data:
            if self._is_it_services_industry(apollo_data, ai_data):
                logger.info(f"IT services industry detected for {domain}, preserving MSP classification if applicable")
                # Preserve MSP classification for IT services
                if original_predicted_class == "Managed Service Provider":
                    # Skip cross-validation for MSPs in IT services industry
                    logger.info(f"Preserving MSP classification for IT services industry: {domain}")
                else:
                    # Regular cross-validation for other classifications
                    llm_result = reconcile_classification(llm_result, apollo_data, ai_data)
            else:
                # Regular cross-validation for non-IT services
                llm_result = reconcile_classification(llm_result, apollo_data, ai_data)
        
        # Add domain to result if not present
        if "domain" not in llm_result:
            llm_result["domain"] = domain
        
        # Add one-line company description if not present
        if "company_one_line" not in llm_result:
            llm_result["company_one_line"] = generate_one_line_description(
                content=content[:5000] if content else "",
                predicted_class=llm_result.get('predicted_class', ''),
                domain=domain,
                company_description=llm_result.get('company_description', '')
            )
        
        return llm_result
    
    def _is_it_services_industry(self, apollo_data: Optional[Dict], ai_data: Optional[Dict]) -> bool:
        """
        Check if the industry data indicates IT services.
        
        Args:
            apollo_data: Apollo data if available
            ai_data: AI data if available
            
        Returns:
            bool: True if industry data indicates IT services
        """
        it_service_indicators = [
            "information technology",
            "it services",
            "it consulting",
            "managed services",
            "tech services",
            "technology services",
            "computer services",
            "network services",
            "cloud services",
            "msp",
            "managed service"
        ]
        
        # Check Apollo industry data
        if apollo_data and "industry" in apollo_data:
            industry = apollo_data.get("industry", "").lower()
            for indicator in it_service_indicators:
                if indicator in industry:
                    logger.info(f"IT services industry detected in Apollo data: {industry}")
                    return True
                    
        # Check AI-extracted industry data
        if ai_data and "industry" in ai_data:
            ai_industry = ai_data.get("industry", "").lower()
            for indicator in it_service_indicators:
                if indicator in ai_industry:
                    logger.info(f"IT services industry detected in AI data: {ai_industry}")
                    return True
        
        return False
    
    def classify_with_vectors(self, text_content: str, domain: str = None) -> Dict[str, Any]:
        """
        Classify using vector similarity first, fallback to LLM for uncertain cases.
        
        Args:
            text_content: The text content to classify
            domain: Optional domain name for context
            
        Returns:
            dict: The classification results or None if confidence is too low
        """
        try:
            # Check if vector DB and anthropic are available
            try:
                from domain_classifier.storage.vector_db import VectorDBConnector, PINECONE_AVAILABLE, ANTHROPIC_AVAILABLE
                if not PINECONE_AVAILABLE:
                    logger.warning(f"Pinecone not available, skipping vector-based classification for {domain}")
                    return None
            except ImportError:
                logger.warning(f"VectorDBConnector not available, skipping vector-based classification for {domain}")
                return None
                
            # Initialize vector DB connector
            vector_db = VectorDBConnector()
            
            if not vector_db.connected:
                logger.warning(f"Vector DB not connected, skipping vector-based classification for {domain}")
                return None
                
            logger.info(f"Using vector similarity to classify {domain}")
            
            # Get most similar domains by vector - using lower thresholds for hash-based embeddings
            similar_domains = vector_db.query_similar_domains(
                query_text=text_content,
                top_k=10,  # Get more examples for better analysis
                low_confidence_threshold=0.3,  # Lower threshold for hash-based embeddings
                unreliable_threshold=0.2  # Lower threshold for unreliability
            )
            
            if not similar_domains:
                logger.info(f"No similar domains found for {domain}, falling back to LLM")
                return None
            
            # Calculate confidence scores based on similarity and class distribution
            confidence_scores = {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0,
                "Internal IT Department": 0
            }
            
            # Weight by similarity score
            total_similarity = 0
            explanations = []
            company_descriptions = []
            
            # Process the similar domains
            for domain_match in similar_domains:
                class_type = domain_match.get('company_type')
                similarity = domain_match.get('score', 0) 
                confidence_flag = domain_match.get('confidence_flag', None)
                
                # Skip unreliable matches
                if confidence_flag == "vector_match_unreliable":
                    logger.warning(f"Skipping unreliable vector match: {domain_match['domain']}")
                    continue
                
                # Only consider valid classifications
                if class_type in confidence_scores:
                    # Add similarity score to the appropriate category
                    confidence_scores[class_type] += similarity * 100
                    total_similarity += similarity
                    
                    # Collect explanations and descriptions from similar domains
                    if domain_match.get('description'):
                        company_descriptions.append(domain_match.get('description'))
                    if domain_match.get('metadata', {}).get('llm_explanation'):
                        explanations.append(domain_match.get('metadata', {}).get('llm_explanation'))
            
            # Normalize scores if we have matches
            if total_similarity > 0:
                for category in confidence_scores:
                    confidence_scores[category] = int(confidence_scores[category] / total_similarity)
            
            # Get highest confidence class
            max_class = max(confidence_scores, key=confidence_scores.get)
            max_score = confidence_scores[max_class]
            
            # Check if any matches had low confidence flags
            low_confidence_matches = sum(1 for d in similar_domains if d.get('confidence_flag') == "low_confidence_vector")
            has_low_confidence = low_confidence_matches > len(similar_domains) / 2
            
            # Adjusted thresholds for hash-based embeddings
            # The similarity scores will be lower with hash-based embeddings, so we lower our expectations
            if max_score > 60 and total_similarity > 0.25 and not has_low_confidence:  # Lower from 70/0.6 to 60/0.25
                # Create combined explanation
                if explanations:
                    # Use the explanation from the most similar domain
                    explanation = f"Based on similarity analysis, this domain is classified as {max_class}. " + explanations[0]
                else:
                    explanation = f"This domain was classified as {max_class} based on high similarity to other domains in this category."
                    
                # Get or generate company description
                if company_descriptions:
                    company_description = company_descriptions[0]
                else:
                    company_description = f"{domain} appears to be a {max_class}."
                
                # Determine if this is a service business
                is_service_business = max_class != "Internal IT Department"
                
                # Calculate internal IT potential
                internal_it_potential = 0
                if max_class == "Internal IT Department":
                    # Find average internal_it_potential from similar domains
                    potentials = []
                    for similar in similar_domains:
                        if similar.get('metadata', {}).get('internal_it_potential'):
                            try:
                                potentials.append(int(similar['metadata']['internal_it_potential']))
                            except (ValueError, TypeError):
                                pass
                    
                    if potentials:
                        internal_it_potential = int(sum(potentials) / len(potentials))
                    else:
                        internal_it_potential = 60  # Default if no examples
                
                # Create complete result with all expected fields
                result = {
                    "processing_status": 2,  # Success
                    "is_service_business": is_service_business,
                    "predicted_class": max_class,
                    "internal_it_potential": internal_it_potential,
                    "confidence_scores": confidence_scores,
                    "llm_explanation": explanation,
                    "company_description": company_description,
                    "detection_method": "vector_similarity",
                    "low_confidence": False,
                    "max_confidence": max_score / 100.0,
                    "vector_similarity_score": total_similarity  # Add this new metric for tracking
                }
                
                # Add one-line company description
                result["company_one_line"] = generate_one_line_description(
                    content=text_content[:5000] if text_content else "",
                    predicted_class=max_class,
                    domain=domain,
                    company_description=company_description
                )
                
                # Ensure the explanation has the step-by-step format
                result = ensure_step_format(result, domain)
                
                logger.info(f"Successfully classified {domain} using vector similarity as {max_class}")
                return result
            else:
                # Add information about why vector similarity wasn't reliable enough
                if has_low_confidence:
                    logger.info(f"Vector similarity had too many low confidence matches for {domain}")
                elif max_score <= 60:
                    logger.info(f"Vector similarity confidence too low for {domain} ({max_score})")
                elif total_similarity <= 0.25:
                    logger.info(f"Vector similarity total similarity too low for {domain} ({total_similarity:.2f})")
                
                return None
                
        except Exception as e:
            logger.error(f"Error in vector-based classification: {e}")
            return None
    
    def _improved_clean_json_string(self, json_str: str) -> str:
        """
        Enhanced version of clean_json_string that handles numbers with commas.
        
        Args:
            json_str: The JSON string to clean
            
        Returns:
            str: The cleaned JSON string
        """
        # Save original for debugging
        original = json_str
        
        # First, fix json strings that may have been escaped
        json_str = re.sub(r'\\n', '\n', json_str)
        json_str = re.sub(r'\\\"', '\"', json_str)
        json_str = re.sub(r'\\"', '\"', json_str)
        
        # Fix numbers with commas - patterns like "90, 0" should be "90"
        # This regex finds patterns like: "key": X, 0 or "key": X,0
        json_str = re.sub(r'": (\d+),\s*0([,}])', r'": \1\2', json_str)
        
        # Fix JSON with wrapped quotation marks
        if json_str.startswith('"') and json_str.endswith('"'):
            try:
                # Try to parse as JSON-escaped string
                unescaped = json.loads(json_str)
                if isinstance(unescaped, str) and unescaped.startswith('{') and unescaped.endswith('}'):
                    json_str = unescaped
            except:
                # If that fails, just strip the quotes directly
                if json_str.startswith('"') and json_str.endswith('"'):
                    json_str = json_str[1:-1]
        
        # Remove any invalid characters
        json_str = re.sub(r'[^\x20-\x7E]', '', json_str)
        
        # Fix missing quotes around keys
        json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Fix trailing commas in objects/arrays which are invalid JSON
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        # Fix double colons
        json_str = re.sub(r'::', r':', json_str)
        
        # Fix missing quotes around string values for known fields
        for field in ["predicted_class", "detection_method", "llm_explanation", "company_description"]:
            json_str = re.sub(
                r'"{0}":\s*([^"{{\[\d][^,}}]+)([,}}])'.format(field),
                r'"{0}": "\1"\2'.format(field),
                json_str
            )
        
        # If the string is significantly different, log it
        if original != json_str:
            logger.info(f"Original JSON: {original[:100]}...")
            logger.info(f"Cleaned JSON: {json_str[:100]}...")
        
        return json_str
    
    def _improved_extract_predicted_class(self, text_response: str) -> Optional[str]:
        """
        Extract predicted class directly from text response when JSON parsing fails.
        
        Args:
            text_response: The raw text response from the LLM
            
        Returns:
            str: The extracted predicted class or None if not found
        """
        # First, try to find a JSON-like predicted_class field
        class_match = re.search(r'"predicted_class"\s*:\s*"([^"]+)"', text_response)
        if class_match:
            return class_match.group(1)
            
        # Try alternative patterns
        alt_match = re.search(r'predicted_class["\s:]*([A-Za-z\s\-]+)[",}]', text_response)
        if alt_match:
            return alt_match.group(1).strip()
            
        # Look for specific class mentions
        class_types = [
            "Managed Service Provider",
            "Integrator - Commercial A/V", 
            "Integrator - Residential A/V",
            "Internal IT Department"
        ]
        
        # Check for phrases like "I classify this as X" or "This is an X"
        for class_type in class_types:
            patterns = [
                rf"classified as (?:a |an )?{class_type}",
                rf"is (?:a |an )?{class_type}",
                rf"appears to be (?:a |an )?{class_type}",
                rf"identified as (?:a |an )?{class_type}"
            ]
            for pattern in patterns:
                if re.search(pattern, text_response, re.IGNORECASE):
                    return class_type
                    
        # Check for specific MSP indicators when we're unsure
        msp_indicators = [
            r"provides (IT|technology) services to (other|multiple|various) (companies|businesses|organizations)",
            r"offers managed (IT|technology|service) to clients",
            r"managed service provider",
            r"provides managed (IT|tech|services)",
            r"IT service provider"
        ]
        
        for indicator in msp_indicators:
            if re.search(indicator, text_response, re.IGNORECASE):
                return "Managed Service Provider"
                
        return None

    def classify_with_llm(self, text_content: str = None, domain: str = None) -> Dict[str, Any]:
        """
        Classify using LLM with optimized performance and reliability.
        
        Args:
            text_content: The text content to classify (can be None)
            domain: Domain name for context (required)
            
        Returns:
            dict: The classification results
        """
        try:
            if not self.api_key:
                raise ValueError("No API key provided")
                
            # Load examples from knowledge base if not already loaded
            if self.loaded_examples is None:
                self.loaded_examples = load_examples_from_knowledge_base()
                total_examples = sum(len(self.loaded_examples[cat]) for cat in self.loaded_examples)
                logger.info(f"Loaded {total_examples} examples from knowledge base")
                
            # Build system prompt with the decision tree approach
            system_prompt = build_decision_tree_prompt(self.loaded_examples)
                
            # CRITICAL CHANGE: Create appropriate user prompt based on available data
            if text_content and len(text_content.strip()) > 0:
                # Limit the text content to avoid token limits
                max_chars = 12000
                if len(text_content) > max_chars:
                    text_content = text_content[:max_chars]
                    
                # Normal prompt with content
                user_prompt = f"Domain name: {domain or 'unknown'}\n\nWebsite content to classify: {text_content}"
            else:
                # Domain-only prompt for when content is unavailable
                user_prompt = f"""Domain name: {domain or 'unknown'}

Website content to classify: [No content available - the website could not be crawled or returned empty content]

Please classify this domain based on the domain name alone. Consider typical naming patterns for different business types and what this domain name suggests about the company's line of business."""
                
            # Create the request to the Claude API with optimized settings
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            data = {
                "model": self.model,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.1
            }
            
            # Make the request to Claude using the reused session
            logger.info(f"Making request to Claude API for domain {domain or 'unknown'}")
            start_time = time.time()
            response = self.session.post(url, headers=headers, json=data, timeout=60)
            response_data = response.json()
            elapsed = time.time() - start_time
            logger.info(f"Claude API request completed in {elapsed:.2f} seconds")
            
            if "error" in response_data:
                logger.error(f"Error from Claude API: {response_data['error']}")
                raise Exception(f"Claude API error: {response_data['error']}")
                
            # Extract the response text
            if "content" in response_data and len(response_data["content"]) > 0:
                text_response = response_data["content"][0]["text"]
                logger.info(f"Received response from Claude API for {domain}")
            else:
                logger.error("No content in Claude response")
                raise Exception("No content in Claude response")
                
            # Try to extract JSON from the response
            json_str = extract_json(text_response)
            
            if json_str:
                try:
                    # Try to parse the JSON with our enhanced cleaner
                    cleaned_json_str = self._improved_clean_json_string(json_str)
                    logger.info(f"Cleaned JSON string for parsing: {cleaned_json_str[:100]}...")
                    
                    try:
                        parsed_json = json.loads(cleaned_json_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {str(e)} at position {e.pos}")
                        logger.error(f"Character at position {e.pos}: '{cleaned_json_str[max(0, e.pos-10):min(len(cleaned_json_str), e.pos+10)]}'")
                        
                        # Try an even more aggressive cleaning approach
                        ultra_clean = re.sub(r'[^\x20-\x7E]', '', cleaned_json_str)
                        try:
                            parsed_json = json.loads(ultra_clean)
                            logger.info("JSON parsed successfully after ultra cleaning")
                        except:
                            # Extract predicted class from text as a last resort
                            predicted_class = self._improved_extract_predicted_class(text_response)
                            if predicted_class:
                                logger.info(f"Extracted predicted class directly from text: {predicted_class}")
                                # Create a simplified result with the extracted class
                                parsed_json = {
                                    "processing_status": 2,  # Success
                                    "predicted_class": predicted_class,
                                    "is_service_business": predicted_class != "Internal IT Department",
                                    "detection_method": "text_extraction_fallback",
                                    "low_confidence": True,
                                    "max_confidence": 0.5,  # Medium confidence since we had to extract
                                    "confidence_scores": {
                                        "Managed Service Provider": 70 if predicted_class == "Managed Service Provider" else 10,
                                        "Integrator - Commercial A/V": 70 if predicted_class == "Integrator - Commercial A/V" else 10,
                                        "Integrator - Residential A/V": 70 if predicted_class == "Integrator - Residential A/V" else 10,
                                        "Internal IT Department": 70 if predicted_class == "Internal IT Department" else 10
                                    },
                                    "llm_explanation": f"Based on text analysis, this appears to be a {predicted_class}. " +
                                                       f"The domain name '{domain}' and available content suggest this classification."
                                }
                            else:
                                # If that still fails, raise the original error
                                raise e
                    
                    # Validate and normalize the parsed JSON
                    parsed_json = validate_classification(parsed_json, domain)
                    
                    # Add detection method - special note if no content
                    if not text_content or len(text_content.strip()) == 0:
                        parsed_json["detection_method"] = "llm_classification_domain_only"
                    else:
                        parsed_json["detection_method"] = "llm_classification"
                    
                    # Set low_confidence flag appropriately
                    max_confidence = parsed_json.get("max_confidence", 0.5)
                    if isinstance(max_confidence, str):
                        try:
                            max_confidence = float(max_confidence)
                        except (ValueError, TypeError):
                            max_confidence = 0
                    
                    # Lower confidence if no content was available
                    if not text_content or len(text_content.strip()) == 0:
                        # Domain-only classifications are less confident
                        parsed_json["low_confidence"] = True
                        if max_confidence > 0.7:  # Cap confidence for domain-only
                            parsed_json["max_confidence"] = 0.7
                    else:
                        parsed_json["low_confidence"] = max_confidence < 0.4 if parsed_json.get("is_service_business", False) else True
                    
                    logger.info(f"Successful LLM classification for {domain or 'unknown'}: {parsed_json['predicted_class']}")
                    
                    # Ensure the explanation has the step-by-step format
                    parsed_json = ensure_step_format(parsed_json, domain)
                    
                    # Add company description field for new modular design
                    if "company_description" not in parsed_json and "llm_explanation" in parsed_json:
                        parsed_json["company_description"] = extract_company_description(
                            text_content,
                            parsed_json["llm_explanation"],
                            domain
                        )
                    
                    # Add one-line description if not already present
                    if "company_one_line" not in parsed_json and "company_description" in parsed_json:
                        parsed_json["company_one_line"] = generate_one_line_description(
                            content=text_content[:5000] if text_content else "",
                            predicted_class=parsed_json.get('predicted_class', ''),
                            domain=domain,
                            company_description=parsed_json.get('company_description', '')
                        )
                    
                    # Return the validated classification
                    return parsed_json
                    
                except Exception as e:
                    logger.error(f"Error parsing LLM response: {e}")
                    logger.error(f"JSON string: {json_str}")
                    # Continue to text parsing with improved extraction
            
            # If we get here, JSON parsing failed, try improved text parsing
            logger.warning("Could not find JSON in LLM response, falling back to text parsing")
            
            # First try to extract predicted class directly from text
            predicted_class = self._improved_extract_predicted_class(text_response)
            if predicted_class:
                logger.info(f"Direct extraction found predicted class: {predicted_class}")
                # Create result based on extracted class
                result = {
                    "processing_status": 2,  # Success
                    "predicted_class": predicted_class,
                    "is_service_business": predicted_class != "Internal IT Department",
                    "confidence_scores": {
                        "Managed Service Provider": 70 if predicted_class == "Managed Service Provider" else 10,
                        "Integrator - Commercial A/V": 70 if predicted_class == "Integrator - Commercial A/V" else 10,
                        "Integrator - Residential A/V": 70 if predicted_class == "Integrator - Residential A/V" else 10,
                        "Internal IT Department": 70 if predicted_class == "Internal IT Department" else 10
                    },
                    "llm_explanation": f"Based on text analysis, {domain} appears to be a {predicted_class}.",
                    "company_description": f"{domain} is classified as a {predicted_class} based on the available information.",
                    "detection_method": "improved_text_extraction",
                    "low_confidence": True,
                    "max_confidence": 0.6,
                    "domain": domain
                }
                
                # Check for any IT/MSP keywords in the explanation
                if "managed service" in text_response.lower() or "msp" in text_response.lower() or "it service" in text_response.lower():
                    if predicted_class != "Managed Service Provider":
                        logger.info(f"MSP keywords found in explanation, overriding class to MSP for {domain}")
                        result["predicted_class"] = "Managed Service Provider"
                        result["is_service_business"] = True
                        result["confidence_scores"]["Managed Service Provider"] = 70
                        result["confidence_scores"]["Internal IT Department"] = 10
            else:
                # Fall back to the original parser if direct extraction fails
                result = parse_free_text(text_response, domain)
            
            # Set detection method based on available content
            if not text_content or len(text_content.strip()) == 0:
                result["detection_method"] = "text_parsing_domain_only"
            else:
                has_minimal_content = detect_minimal_content(text_content)
                if has_minimal_content:
                    result["detection_method"] = "text_parsing_with_minimal_content"
                else:
                    result["detection_method"] = "text_parsing"
            
            # Ensure the explanation has the step-by-step format
            result = ensure_step_format(result, domain)
            
            # Add domain to result
            result["domain"] = domain
            
            # Add one-line description if not present
            if "company_one_line" not in result and "company_description" in result:
                result["company_one_line"] = generate_one_line_description(
                    content=text_content[:5000] if text_content else "",
                    predicted_class=result.get('predicted_class', ''),
                    domain=domain,
                    company_description=result.get('company_description', '')
                )
            
            logger.info(f"Extracted classification from free text: {result['predicted_class']}")
            return result
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fall back to keyword-based classification
            result = fallback_classification(text_content, domain)
            
            # Check domain for IT/MSP keywords before finalizing fallback
            domain_keywords = analyze_domain_words(domain)
            if any(kw in domain.lower() for kw in ["msp", "itsupport", "itservice", "tech", "managed"]):
                logger.info(f"Domain {domain} contains MSP keywords, setting classification to MSP")
                result["predicted_class"] = "Managed Service Provider"
                result["is_service_business"] = True
            
            # Set detection method based on available content
            if not text_content or len(text_content.strip()) == 0:
                result["detection_method"] = "fallback_classification_domain_only"
            else:
                has_minimal_content = detect_minimal_content(text_content)
                if has_minimal_content:
                    result["detection_method"] = "fallback_with_minimal_content"
                else:
                    result["detection_method"] = "fallback_classification"
            
            # Ensure domain is in result
            result["domain"] = domain
                
            # Ensure the explanation has the step-by-step format
            result = ensure_step_format(result, domain)
            
            # Add one-line description if not present
            if "company_one_line" not in result and "company_description" in result:
                result["company_one_line"] = generate_one_line_description(
                    content=text_content[:5000] if text_content else "",
                    predicted_class=result.get('predicted_class', ''),
                    domain=domain,
                    company_description=result.get('company_description', '')
                )
            
            return result
