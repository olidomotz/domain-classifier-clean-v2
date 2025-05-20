"""Updated LLM classifier to handle missing content correctly."""
import requests
import logging
import json
import re
import os
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
    generate_one_line_description
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
                    
                    # Perform cross-validation if we have industry data
                    if apollo_data or ai_data:
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
        
        # Cross-validate LLM result with industry data
        if apollo_data or ai_data:
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
    
    def classify_with_llm(self, text_content: str = None, domain: str = None) -> Dict[str, Any]:
        """
        Classify using LLM with whatever information is available.
        IMPORTANT: This method now works even if text_content is None.
        
        Args:
            text_content: The text content to classify (can be None)
            domain: Domain name for context (required)
            
        Returns:
            dict: The classification results including predicted class and confidence scores
        """
        try:
            if not self.api_key:
                raise ValueError("No API key provided")
                
            # Load examples from knowledge base
            examples = load_examples_from_knowledge_base()
            
            # Log knowledge base usage
            total_examples = sum(len(examples[cat]) for cat in examples)
            logger.info(f"Loaded {total_examples} examples from knowledge base")
            
            # Build system prompt with the decision tree approach
            system_prompt = build_decision_tree_prompt(examples)
                
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
                
            # Create the request to the Claude API
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
            
            # Make the request to Claude
            logger.info(f"Making request to Claude API for domain {domain or 'unknown'}")
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response_data = response.json()
            
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
                    # Try to parse the JSON
                    parsed_json = json.loads(clean_json_string(json_str))
                    
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
                        from domain_classifier.utils.text_processing import extract_company_description
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
            
            # If we get here, JSON parsing failed, try free text parsing
            logger.warning("Could not find JSON in LLM response, falling back to text parsing")
            parsed_result = parse_free_text(text_response, domain)
            
            # Set detection method based on available content
            if not text_content or len(text_content.strip()) == 0:
                parsed_result["detection_method"] = "text_parsing_domain_only"
            else:
                has_minimal_content = detect_minimal_content(text_content)
                if has_minimal_content:
                    parsed_result["detection_method"] = "text_parsing_with_minimal_content"
                else:
                    parsed_result["detection_method"] = "text_parsing"
            
            # Ensure the explanation has the step-by-step format
            parsed_result = ensure_step_format(parsed_result, domain)
            
            # Add domain to result
            parsed_result["domain"] = domain
            
            # Add one-line description if not present
            if "company_one_line" not in parsed_result and "company_description" in parsed_result:
                parsed_result["company_one_line"] = generate_one_line_description(
                    content=text_content[:5000] if text_content else "",
                    predicted_class=parsed_result.get('predicted_class', ''),
                    domain=domain,
                    company_description=parsed_result.get('company_description', '')
                )
            
            logger.info(f"Extracted classification from free text: {parsed_result['predicted_class']}")
            return parsed_result
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Fall back to keyword-based classification
            result = fallback_classification(text_content, domain)
            
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
