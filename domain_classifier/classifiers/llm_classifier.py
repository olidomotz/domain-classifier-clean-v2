"""LLM classifier for domain classification with improved fallback handling."""
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

    def classify(self, text_content: str, domain: str = None, apollo_data: Optional[Dict] = None, 
                 ai_data: Optional[Dict] = None, use_vector_classification: bool = True, 
                 track_metrics: bool = True, force_llm_classify: bool = False) -> Dict[str, Any]:
        """
        Classify text content, first trying vector-based approach, then falling back to decision tree.
        
        Args:
            text_content: The website content
            domain: Optional domain name for context
            apollo_data: Optional Apollo data
            ai_data: Optional AI-extracted data
            use_vector_classification: Whether to use vector classification
            track_metrics: Whether to track vector classification metrics
            force_llm_classify: Force LLM-based classification even in exceptional cases
            
        Returns:
            dict: The classification results including predicted class and confidence scores
        """
        logger.info(f"Starting classification for domain: {domain or 'unknown'}")
        
        # STEP 1: Check if processing can complete
        if not text_content and not force_llm_classify:
            logger.warning(f"No content provided for domain: {domain or 'unknown'}")
            # CRITICAL CHANGE: If domain is provided, still try to classify with just the domain
            if domain and force_llm_classify:
                logger.info(f"No content but will force LLM classification based on domain name: {domain}")
                # Skip to LLM classification with just domain name
                return self.classify_with_llm_minimal(domain)
            return create_process_did_not_complete_result(domain)
        
        # If we're force classifying with minimal or no content
        if force_llm_classify and (not text_content or len(text_content.strip()) < 100):
            logger.info(f"Force classifying with minimal/no content for {domain}")
            return self.classify_with_llm_minimal(domain, text_content)
        
        # Cache lowercase text for repeated use
        text_lower = text_content.lower() if text_content else ""
        
        # STEP 2: Check if this is a parked/minimal domain
        # Pass domain to is_parked_domain for better detection
        if text_content and is_parked_domain(text_content, domain):
            logger.info(f"Domain {domain or 'unknown'} is detected as a parked domain")
            return create_parked_domain_result(domain)
        
        # Check for minimal content with proxy errors or hosting mentions (common with GoDaddy parked domains)
        if text_content and len(text_content.strip()) < 300 and any(phrase in text_lower for phrase in 
                                                ["proxy error", "error connecting", 
                                                 "godaddy", "domain registration"]):
            logger.info(f"Domain {domain or 'unknown'} appears to be parked based on proxy errors or hosting mentions")
            return create_parked_domain_result(domain)
            
        is_minimal_content = text_content and detect_minimal_content(text_content)
        if is_minimal_content:
            logger.info(f"Domain {domain or 'unknown'} has minimal content")
            
        # Special case handling for specific domains
        if domain:
            # CRITICAL CHANGE: Check for special domain handling
            domain_result = check_special_domain_cases(domain, text_content)
            if domain_result:
                # If force_llm_classify is True, don't use the special case result directly
                if force_llm_classify:
                    logger.info(f"Special domain case detected for {domain}, but will run LLM classification anyway")
                    # Store the special case result to enhance LLM results later
                    special_case = domain_result
                    
                    # Get LLM classification
                    llm_result = self.classify_with_llm(text_content, domain)
                    
                    # If LLM succeeded, combine results, prioritizing special case for key fields
                    if llm_result:
                        # Merge special case hints with LLM classification
                        if "suggested_class" in special_case and special_case["suggested_class"] == llm_result["predicted_class"]:
                            # If suggestions match, boost confidence
                            if "confidence_boost" in special_case:
                                boost = special_case["confidence_boost"]
                                llm_result["max_confidence"] = min(1.0, llm_result.get("max_confidence", 0.5) + boost)
                            
                            # For confidence scores
                            if "confidence_scores" in llm_result and "confidence_boost" in special_case:
                                boost = special_case["confidence_boost"]
                                boost_int = int(boost * 100)
                                class_name = llm_result["predicted_class"]
                                if class_name in llm_result["confidence_scores"]:
                                    # Apply boost to specified class
                                    current_score = llm_result["confidence_scores"][class_name]
                                    llm_result["confidence_scores"][class_name] = min(100, current_score + boost_int)
                        
                        # Special handling for detection method
                        llm_result["detection_method"] = f"{llm_result.get('detection_method', 'llm_classification')}_with_domain_hints"
                        
                        logger.info(f"Enhanced LLM classification with special domain hints for {domain}")
                        return llm_result
                    else:
                        # If LLM fails, fall back to special case
                        logger.warning(f"LLM classification failed, using special case handling for {domain}")
                        return domain_result
                else:
                    # Use special case directly if not force_llm_classify
                    logger.info(f"Using special case handling for domain: {domain}")
                    return domain_result
        
        # STEP 3: Analyze domain words for early signals
        domain_word_scores = analyze_domain_words(domain) if domain else {}
        
        # Flag if domain name suggests manufacturing
        manufacturing_signal = False
        if domain_word_scores.get('manufacturing_score', 0) > domain_word_scores.get('msp_score', 0):
            manufacturing_signal = True
            logger.info(f"Domain name analysis suggests manufacturing for {domain}")
            
        # STEP 4: Check industry context from Apollo and AI data
        is_service_business, industry_confidence = check_industry_context(
            text_content, apollo_data, ai_data
        )
        
        # If high confidence that it's NOT a service business from industry data
        if not is_service_business and industry_confidence >= 0.7:
            logger.info(f"Industry analysis indicates {domain} is NOT a service business with high confidence")
            # Create a Non-Service Business result
            from domain_classifier.classifiers.fallback_classifier import create_non_service_result
            return create_non_service_result(domain, text_content)
            
        # If manufacturing signal from domain name AND industry evidence suggests non-service
        if manufacturing_signal and not is_service_business and industry_confidence >= 0.5:
            logger.info(f"Multiple signals indicate {domain} is a manufacturing company")
            from domain_classifier.classifiers.fallback_classifier import create_non_service_result
            result = create_non_service_result(domain, text_content)
            # Boost manufacturing description
            result["company_description"] = f"{domain} appears to be a manufacturing company specializing in industrial products."
            result["internal_it_potential"] = 70  # Higher IT needs for manufacturing
            # Add one-line description
            result["company_one_line"] = f"{domain} manufactures industrial products and equipment."
            return result
        
        # STEP 5: Try vector-based classification if enabled
        if use_vector_classification and text_content:
            try:
                if track_metrics:
                    self.vector_attempts += 1
                    
                vector_result = self.classify_with_vectors(text_content, domain)
                
                # If vector classification returned a result
                if vector_result and vector_result.get("detection_method") == "vector_similarity":
                    # Check for low confidence flags from vector DB
                    if vector_result.get("vector_match_unreliable"):
                        logger.info(f"Vector match deemed unreliable for {domain}, using LLM classification")
                        # Fall through to LLM classification
                    elif vector_result.get("low_confidence_vector") and (manufacturing_signal or not is_service_business):
                        logger.warning(f"Low confidence vector match for {domain} with contradicting signals, using LLM classification")
                        # Fall through to LLM classification
                    else:
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
                        
                        # Add one-line description if not present
                        if "company_one_line" not in vector_result:
                            vector_result["company_one_line"] = generate_one_line_description(
                                content=text_content[:5000] if text_content else "",
                                predicted_class=vector_result.get('predicted_class', ''),
                                domain=domain,
                                company_description=vector_result.get('company_description', '')
                            )
                        
                        return vector_result
            except Exception as e:
                logger.warning(f"Error during vector-based classification for {domain or 'unknown'}: {e}")
                # Continue to traditional LLM classification
        else:
            logger.info(f"Vector classification disabled for {domain}")
        
        # STEP 6: Use the LLM for classification as fallback
        llm_result = self.classify_with_llm(text_content, domain)
        
        # Cross-validate LLM result with industry data
        if apollo_data or ai_data:
            llm_result = reconcile_classification(llm_result, apollo_data, ai_data)
        
        # Add one-line company description if not present
        if "company_one_line" not in llm_result:
            llm_result["company_one_line"] = generate_one_line_description(
                content=text_content[:5000] if text_content else "",
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
    
    def classify_with_llm_minimal(self, domain: str, minimal_content: str = None) -> Dict[str, Any]:
        """
        Classify using LLM with minimal information (just domain name and optional minimal content).
        Used for sites that can't be crawled but should still be classified.
        
        Args:
            domain: The domain name to classify
            minimal_content: Optional minimal content that might be available
            
        Returns:
            dict: The classification results
        """
        try:
            if not self.api_key:
                raise ValueError("No API key provided")
                
            # Load examples from knowledge base
            examples = load_examples_from_knowledge_base()
            
            # Build system prompt with the decision tree approach
            system_prompt = build_decision_tree_prompt(examples)
            
            # Create specialized user prompt for minimal information
            user_prompt = f"""Classify this domain using only the domain name and any available fragments.
            
Domain name: {domain}
"""
            
            # Add any minimal content if available
            if minimal_content and len(minimal_content.strip()) > 0:
                # Limit content if it exists
                max_chars = 5000  # Smaller limit for minimal content
                content_to_use = minimal_content[:max_chars] if len(minimal_content) > max_chars else minimal_content
                user_prompt += f"\nAvailable content fragments:\n{content_to_use}\n"
            else:
                # Provide context that we're working with minimal info
                user_prompt += "\nNote: No website content is available. Please make your best judgment based on the domain name alone.\n"
                
            user_prompt += "\nPlease do your best to classify this domain even with minimal information. Use domain name analysis, industry patterns, and your knowledge of business naming conventions."
                
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
            logger.info(f"Making minimal-info request to Claude API for domain {domain}")
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response_data = response.json()
            
            if "error" in response_data:
                logger.error(f"Error from Claude API: {response_data['error']}")
                raise Exception(f"Claude API error: {response_data['error']}")
                
            # Extract the response text
            if "content" in response_data and len(response_data["content"]) > 0:
                text_response = response_data["content"][0]["text"]
                logger.info(f"Received response from Claude API for minimal-info classification of {domain}")
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
                    
                    # Add detection method
                    parsed_json["detection_method"] = "llm_minimal_info_classification"
                    
                    # Set low_confidence flag based on highest score
                    max_confidence = parsed_json.get("max_confidence", 0.5)
                    if isinstance(max_confidence, str):
                        try:
                            max_confidence = float(max_confidence)
                        except (ValueError, TypeError):
                            max_confidence = 0
                            
                    # Set confidence lower for minimal info classification
                    parsed_json["low_confidence"] = True
                    parsed_json["max_confidence"] = min(0.7, max_confidence)  # Cap confidence for minimal info
                    
                    logger.info(f"Successful minimal-info LLM classification for {domain}: {parsed_json['predicted_class']}")
                    
                    # Ensure the explanation has the step-by-step format
                    parsed_json = ensure_step_format(parsed_json, domain)
                    
                    # Add company description field for new modular design
                    if "company_description" not in parsed_json and "llm_explanation" in parsed_json:
                        from domain_classifier.utils.text_processing import extract_company_description
                        parsed_json["company_description"] = extract_company_description(
                            "",  # No content
                            parsed_json["llm_explanation"],
                            domain
                        )
                    
                    # Add one-line description if not already present
                    if "company_one_line" not in parsed_json and "company_description" in parsed_json:
                        parsed_json["company_one_line"] = generate_one_line_description(
                            content="",  # No content
                            predicted_class=parsed_json.get('predicted_class', ''),
                            domain=domain,
                            company_description=parsed_json.get('company_description', '')
                        )
                    
                    # Return the validated classification
                    return parsed_json
                    
                except Exception as e:
                    logger.error(f"Error parsing LLM minimal-info response: {e}")
                    logger.error(f"JSON string: {json_str}")
            
            # If JSON parsing fails, try free text parsing
            logger.warning("Could not find JSON in LLM minimal-info response, falling back to text parsing")
            parsed_result = parse_free_text(text_response, domain)
            parsed_result["detection_method"] = "minimal_info_text_parsing"
            parsed_result["low_confidence"] = True  # Always low confidence with minimal info
            
            # Ensure the explanation has the step-by-step format
            parsed_result = ensure_step_format(parsed_result, domain)
            
            # Add one-line description if not present
            if "company_one_line" not in parsed_result and "company_description" in parsed_result:
                parsed_result["company_one_line"] = generate_one_line_description(
                    content="",  # No content
                    predicted_class=parsed_result.get('predicted_class', ''),
                    domain=domain,
                    company_description=parsed_result.get('company_description', '')
                )
            
            logger.info(f"Extracted minimal-info classification from free text: {parsed_result['predicted_class']}")
            return parsed_result
            
        except Exception as e:
            logger.error(f"Minimal-info LLM classification failed: {e}")
            # Fall back to domain name analysis
            result = self._classify_from_domain_name(domain)
            return result
    
    def _classify_from_domain_name(self, domain: str) -> Dict[str, Any]:
        """
        Classify based solely on domain name analysis when all else fails.
        
        Args:
            domain: The domain name
            
        Returns:
            dict: Classification results
        """
        logger.info(f"Using domain name analysis for {domain}")
        
        # Analyze domain words
        domain_scores = analyze_domain_words(domain)
        
        # Determine highest score category
        categories = {
            'msp_score': "Managed Service Provider",
            'commercial_av_score': "Integrator - Commercial A/V",
            'residential_av_score': "Integrator - Residential A/V",
            'manufacturing_score': "Internal IT Department",  # Manufacturing -> Internal IT
            'retail_score': "Internal IT Department"          # Retail -> Internal IT
        }
        
        # Get highest domain score
        highest_category = max(domain_scores.items(), key=lambda x: x[1])
        category_name, score = highest_category
        
        # Convert to predicted class
        predicted_class = categories.get(category_name, "Managed Service Provider")
        
        # For manufacturing or retail, set as Internal IT
        is_service_business = predicted_class != "Internal IT Department"
        
        # Set up confidence scores based on domain word analysis
        confidence_scores = {
            "Managed Service Provider": int(domain_scores.get('msp_score', 0) * 100),
            "Integrator - Commercial A/V": int(domain_scores.get('commercial_av_score', 0) * 100),
            "Integrator - Residential A/V": int(domain_scores.get('residential_av_score', 0) * 100),
            "Internal IT Department": 0  # Default for service businesses
        }
        
        # Ensure all scores are at least 5
        for key in confidence_scores:
            if confidence_scores[key] < 5:
                confidence_scores[key] = 5
        
        # For non-service business, add Internal IT score and adjust others
        if predicted_class == "Internal IT Department":
            # Calculate potential based on industry signals
            internal_it_score = 60  # Default medium-high
            
            # Adjust based on domain signals
            if domain_scores.get('manufacturing_score', 0) > 0.3:
                internal_it_score = 70  # Higher for manufacturing
            elif domain_scores.get('retail_score', 0) > 0.3:
                internal_it_score = 50  # Medium for retail
                
            confidence_scores["Internal IT Department"] = internal_it_score
            confidence_scores["Managed Service Provider"] = 10
            confidence_scores["Integrator - Commercial A/V"] = 5
            confidence_scores["Integrator - Residential A/V"] = 5
        else:
            # For service businesses, make sure Internal IT is 0
            confidence_scores["Internal IT Department"] = 0
            
            # Ensure the predicted class has highest confidence
            highest_score = max(v for k, v in confidence_scores.items() if k != "Internal IT Department")
            confidence_scores[predicted_class] = max(highest_score + 10, confidence_scores[predicted_class])
        
        # Generate explanation
        explanation = f"Based on analysis of the domain name '{domain}', this appears to be a {predicted_class}. "
        
        if domain_scores.get('msp_score', 0) > 0.3:
            explanation += "The domain name contains terms commonly associated with IT service providers. "
        if domain_scores.get('commercial_av_score', 0) > 0.3:
            explanation += "The domain name contains terms associated with commercial audio-visual integration. "
        if domain_scores.get('residential_av_score', 0) > 0.3:
            explanation += "The domain name contains terms associated with residential audio-visual integration. "
        if domain_scores.get('manufacturing_score', 0) > 0.3:
            explanation += "The domain name suggests a manufacturing or industrial company. "
        if domain_scores.get('retail_score', 0) > 0.3:
            explanation += "The domain name suggests a retail or e-commerce business. "
            
        explanation += "Note: This classification is based solely on domain name analysis since no content was available."
        
        # Create a company description
        if predicted_class == "Managed Service Provider":
            company_description = f"{domain} appears to be a managed service provider offering IT services and technology solutions to clients."
            company_one_line = f"{domain} provides managed IT services and technology solutions."
        elif predicted_class == "Integrator - Commercial A/V":
            company_description = f"{domain} appears to be an audio-visual integration company specializing in commercial installations, providing businesses with technology solutions for meeting rooms, signage, and presentations."
            company_one_line = f"{domain} installs audio-visual systems for businesses."
        elif predicted_class == "Integrator - Residential A/V":
            company_description = f"{domain} appears to be a residential audio-visual integration company, providing smart home technology, home theaters, and automation systems for residential clients."
            company_one_line = f"{domain} installs home theater and automation systems."
        else:  # Internal IT
            if domain_scores.get('manufacturing_score', 0) > 0.3:
                company_description = f"{domain} appears to be a manufacturing company with internal IT needs. The domain name suggests an industrial focus rather than a technology service provider."
                company_one_line = f"{domain} is a manufacturing company with internal IT needs."
            elif domain_scores.get('retail_score', 0) > 0.3:
                company_description = f"{domain} appears to be a retail business with internal IT needs. The domain name suggests a focus on selling products rather than providing IT services."
                company_one_line = f"{domain} is a retail business with internal IT needs."
            else:
                company_description = f"{domain} appears to be a business with internal IT needs rather than a technology service provider."
                company_one_line = f"{domain} is a business with internal IT needs."
        
        return {
            "processing_status": 2,  # Success
            "is_service_business": is_service_business,
            "predicted_class": predicted_class,
            "internal_it_potential": 0 if is_service_business else confidence_scores["Internal IT Department"],
            "confidence_scores": confidence_scores,
            "llm_explanation": explanation,
            "company_description": company_description,
            "company_one_line": company_one_line,
            "detection_method": "domain_name_analysis",
            "low_confidence": True,  # Always low confidence with just domain name
            "max_confidence": 0.4  # Lower max confidence when using just domain name
        }
    
    def classify_with_llm(self, text_content: str, domain: str = None) -> Dict[str, Any]:
        """
        Classify text content using LLM decision tree approach.
        This is the traditional/original classification method.
        
        Args:
            text_content: The text content to classify
            domain: Optional domain name for context
            
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
                
            # Limit the text content to avoid token limits
            max_chars = 12000
            if text_content and len(text_content) > max_chars:
                text_content = text_content[:max_chars]
                
            # Create the request to the Claude API
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            # CHANGED: Use safer check for None or empty content
            if not text_content or len(text_content.strip()) == 0:
                user_content = f"Domain name: {domain or 'unknown'}\n\nWebsite content to classify: [No content available, please classify based on domain name alone]"
            else:
                user_content = f"Domain name: {domain or 'unknown'}\n\nWebsite content to classify: {text_content}"
            
            data = {
                "model": self.model,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_content}
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
                    
                    # IMPORTANT: Save the raw response for debugging
                    logger.info(f"Successfully parsed JSON from Claude response for {domain}")
                    logger.debug(f"Parsed JSON for {domain}: {json.dumps(parsed_json)[:500]}...")
                    
                    # Validate and normalize the parsed JSON
                    parsed_json = validate_classification(parsed_json, domain)
                    
                    # Add detection method
                    parsed_json["detection_method"] = "llm_classification"
                    
                    # Set low_confidence flag based on highest score
                    max_confidence = parsed_json.get("max_confidence", 0.5)
                    if isinstance(max_confidence, str):
                        try:
                            max_confidence = float(max_confidence)
                        except (ValueError, TypeError):
                            max_confidence = 0
                            
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
            if detect_minimal_content(text_content) if text_content else True:
                parsed_result["detection_method"] = "text_parsing_with_minimal_content"
            else:
                parsed_result["detection_method"] = "text_parsing"
            
            # Ensure the explanation has the step-by-step format
            parsed_result = ensure_step_format(parsed_result, domain)
            
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
            if detect_minimal_content(text_content) if text_content else True:
                result["detection_method"] = result["detection_method"] + "_with_minimal_content"
                
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
