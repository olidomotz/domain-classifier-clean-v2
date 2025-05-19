"""Optimized bulk processing for large domain batches with crawling support and persistent Snowflake storage."""
import logging
import time
import traceback
import json
import csv
import tempfile
import os
from flask import request, jsonify, send_file
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional
import threading
import queue
import concurrent.futures
from datetime import datetime, timedelta
import hashlib
import random

# Import domain utilities
from domain_classifier.utils.domain_utils import extract_domain_from_email, normalize_domain
from domain_classifier.utils.error_handling import detect_error_type, create_error_result, is_domain_worth_crawling
from domain_classifier.config.overrides import check_domain_override
from domain_classifier.crawlers.apify_crawler import crawl_website

# Set up logging
logger = logging.getLogger(__name__)

# Global settings
MAX_WORKERS = 8  # Maximum concurrent classification threads
MAX_CRAWLS_PER_MINUTE = 30  # Rate limit for crawling
crawl_semaphore = threading.Semaphore(10)  # Limit concurrent crawls
crawl_counter = 0
crawl_counter_lock = threading.Lock()

# In-memory storage for batch processing (fallback if Snowflake fails)
PROCESS_STORAGE = {}

def process_domain_with_crawling(domain: str, use_existing_content: bool, force_reclassify: bool, 
                               llm_classifier=None, snowflake_conn=None, process_id=None):
    """
    Process a single domain with optional crawling for bulk processing.
    
    Args:
        domain: The domain to process
        use_existing_content: Whether to use existing content if available
        force_reclassify: Whether to force reclassification
        llm_classifier: The LLM classifier instance
        snowflake_conn: The Snowflake connector instance
        process_id: The bulk process ID for tracking
        
    Returns:
        dict: The classification result
    """
    global crawl_counter
    
    try:
        # Normalize domain
        domain = normalize_domain(domain)
        
        # Default output structure with domain
        result = {
            "domain": domain,
            "status": "processed",
            "classification_date": datetime.now().isoformat()
        }
        
        # Check for domain override first
        domain_override = check_domain_override(domain)
        if domain_override:
            # Format the override result
            predicted_class = domain_override.get("predicted_class", "Unknown")
            
            # Determine final classification
            final_classification = "2-Internal IT" # Default
            if predicted_class == "Managed Service Provider":
                final_classification = "1-MSP"
            elif predicted_class == "Integrator - Commercial A/V":
                final_classification = "3-Commercial Integrator"
            elif predicted_class == "Integrator - Residential A/V":
                final_classification = "4-Residential Integrator"
            
            # Update result
            result.update({
                "predicted_class": predicted_class,
                "confidence_score": domain_override.get("confidence_score", 90),
                "explanation": domain_override.get("explanation", ""),
                "detection_method": "override",
                "final_classification": final_classification,
                "source": "override"
            })
            
            # Even overrides should be saved with the process_id
            # CHANGE 2: Save overrides with bulk_process_id
            if snowflake_conn and process_id:
                try:
                    from domain_classifier.storage.operations import save_to_snowflake
                    save_to_snowflake(
                        domain=domain, 
                        url=f"https://{domain}", 
                        content=None, 
                        classification=result,  # Pass the override result
                        snowflake_conn=snowflake_conn,
                        bulk_process_id=process_id
                    )
                except Exception as e:
                    logger.error(f"Error saving override to Snowflake: {e}")
            
            return result
        
        # Check for cached result first if not forcing reclassification
        if not force_reclassify and snowflake_conn:
            existing_record = snowflake_conn.check_existing_classification(domain)
            
            if existing_record:
                # Get the classification details
                company_type = existing_record.get('COMPANY_TYPE', 'Unknown')
                confidence_score = existing_record.get('CONFIDENCE_SCORE', 0)
                detection_method = existing_record.get('DETECTION_METHOD', 'cached')
                low_confidence = existing_record.get('LOW_CONFIDENCE', False)
                
                # Try to get all scores
                confidence_scores = {}
                try:
                    if existing_record.get('ALL_SCORES'):
                        confidence_scores = json.loads(existing_record.get('ALL_SCORES', '{}'))
                except Exception as e:
                    logger.warning(f"Error parsing ALL_SCORES for {domain}: {e}")
                
                # Add to result
                result.update({
                    "predicted_class": company_type,
                    "confidence_score": round(float(confidence_score) * 100),
                    "confidence_scores": confidence_scores,
                    "detection_method": detection_method,
                    "source": "cached",
                    "low_confidence": low_confidence
                })
                
                # Add final classification
                if company_type == "Managed Service Provider":
                    result["final_classification"] = "1-MSP"
                elif company_type == "Integrator - Commercial A/V":
                    result["final_classification"] = "3-Commercial Integrator"
                elif company_type == "Integrator - Residential A/V":
                    result["final_classification"] = "4-Residential Integrator"
                elif company_type == "Parked Domain":
                    result["final_classification"] = "6-Parked Domain - no enrichment"
                else:
                    result["final_classification"] = "2-Internal IT"
                
                logger.info(f"Returning cached result for {domain}: {company_type}")
                
                # CHANGE 3: Update the BULK_PROCESS_ID for cached results
                if process_id:
                    try:
                        # Use new method to update the BULK_PROCESS_ID for cached results
                        snowflake_conn.update_classification_bulk_id(domain, process_id)
                    except Exception as e:
                        logger.error(f"Error updating BULK_PROCESS_ID for cached result {domain}: {e}")
                
                # CHANGE 4: Update the summary stats to include cached results
                if process_id in PROCESS_STORAGE:
                    try:
                        if 'summary' not in PROCESS_STORAGE[process_id]:
                            PROCESS_STORAGE[process_id]['summary'] = {
                                'crawled': 0,
                                'cached': 0,
                                'errors': 0,
                                'apollo_enriched': 0,  # Add counter for Apollo enrichment
                                'by_class': {
                                    "Managed Service Provider": 0,
                                    "Integrator - Commercial A/V": 0,
                                    "Integrator - Residential A/V": 0,
                                    "Internal IT Department": 0,
                                    "Parked Domain": 0,
                                    "Unknown": 0
                                }
                            }
                        
                        # Increment cached count
                        PROCESS_STORAGE[process_id]['summary']['cached'] += 1
                        
                        # Update class count
                        if company_type in PROCESS_STORAGE[process_id]['summary']['by_class']:
                            PROCESS_STORAGE[process_id]['summary']['by_class'][company_type] += 1
                        elif company_type == "Unknown" or company_type == "Process Did Not Complete":
                            PROCESS_STORAGE[process_id]['summary']['by_class']['Unknown'] += 1
                    except Exception as stats_error:
                        logger.error(f"Error updating summary stats for cached result: {stats_error}")
                
                # Try to enrich with Apollo data if it's not already enriched
                if not existing_record.get('APOLLO_COMPANY_DATA') and company_type != "Parked Domain":
                    try:
                        # Import Apollo connector
                        from domain_classifier.enrichment.apollo_connector import ApolloConnector
                        
                        # Initialize Apollo with a small delay to avoid rate limiting
                        time.sleep(0.5)  # Small delay
                        apollo = ApolloConnector()
                        
                        # Call the enrichment with retry logic
                        max_retries = 3
                        apollo_data = None
                        
                        for attempt in range(max_retries):
                            try:
                                logger.info(f"Attempting Apollo enrichment for {domain} in batch process (attempt {attempt+1})")
                                apollo_data = apollo.enrich_company(domain)
                                if apollo_data:
                                    result["apollo_data"] = apollo_data
                                    logger.info(f"Successfully enriched {domain} with Apollo data in batch process")
                                    
                                    # Increment Apollo enrichment counter
                                    if process_id in PROCESS_STORAGE and 'summary' in PROCESS_STORAGE[process_id]:
                                        if 'apollo_enriched' not in PROCESS_STORAGE[process_id]['summary']:
                                            PROCESS_STORAGE[process_id]['summary']['apollo_enriched'] = 0
                                        PROCESS_STORAGE[process_id]['summary']['apollo_enriched'] += 1
                                    
                                    # Save to Snowflake with Apollo data
                                    from domain_classifier.storage.operations import save_to_snowflake
                                    save_to_snowflake(
                                        domain=domain, 
                                        url=f"https://{domain}", 
                                        content=None,
                                        classification=result,
                                        snowflake_conn=snowflake_conn,
                                        apollo_company_data=apollo_data,
                                        bulk_process_id=process_id
                                    )
                                    break
                                else:
                                    # Add exponential backoff if Apollo returns no data
                                    backoff_time = 2 ** attempt
                                    logger.info(f"Apollo returned no data for {domain}, backing off for {backoff_time}s")
                                    time.sleep(backoff_time)
                            except Exception as apollo_error:
                                logger.warning(f"Apollo enrichment attempt {attempt+1} failed: {apollo_error}")
                                # Add exponential backoff with jitter
                                backoff_time = (2 ** attempt) + random.uniform(0.1, 1.0)
                                logger.info(f"Backing off for {backoff_time:.2f}s before retry")
                                time.sleep(backoff_time)
                    except Exception as apollo_err:
                        logger.error(f"Error during Apollo enrichment in batch process: {apollo_err}")
                
                return result
                
        # Get content from database if using existing content
        content = None
        if use_existing_content and snowflake_conn:
            try:
                content = snowflake_conn.get_domain_content(domain)
                # If content found, record this in the result
                if content:
                    result["content_source"] = "existing"
                    logger.info(f"Using existing content for {domain}")
            except Exception as e:
                logger.warning(f"Error getting existing content for {domain}: {e}")
        
        # If no content, check if domain is worth crawling
        if not content:
            # Perform domain health check
            worth_crawling, has_dns, dns_error, potentially_flaky = is_domain_worth_crawling(domain)
            
            if not worth_crawling:
                # Domain has issues, record this in the result
                result.update({
                    "status": "error",
                    "error_type": "dns_error" if "DNS" in dns_error else "connection_error",
                    "error_detail": dns_error,
                    "predicted_class": "Unknown",
                    "final_classification": "7-No Website available"
                })
                
                # CHANGE 5: Update summary stats for errors
                if process_id in PROCESS_STORAGE:
                    try:
                        if 'summary' not in PROCESS_STORAGE[process_id]:
                            PROCESS_STORAGE[process_id]['summary'] = {
                                'crawled': 0,
                                'cached': 0,
                                'errors': 0,
                                'apollo_enriched': 0,  # Add counter for Apollo enrichment
                                'by_class': {
                                    "Managed Service Provider": 0,
                                    "Integrator - Commercial A/V": 0,
                                    "Integrator - Residential A/V": 0,
                                    "Internal IT Department": 0,
                                    "Parked Domain": 0,
                                    "Unknown": 0
                                }
                            }
                        
                        # Increment error count
                        PROCESS_STORAGE[process_id]['summary']['errors'] += 1
                        
                        # Update class count
                        PROCESS_STORAGE[process_id]['summary']['by_class']['Unknown'] += 1
                    except Exception as stats_error:
                        logger.error(f"Error updating summary stats for error result: {stats_error}")
                
                # Save error result with process_id
                if snowflake_conn and process_id:
                    try:
                        from domain_classifier.storage.operations import save_to_snowflake
                        save_to_snowflake(
                            domain=domain, 
                            url=f"https://{domain}", 
                            content=None, 
                            classification=result,
                            snowflake_conn=snowflake_conn,
                            bulk_process_id=process_id
                        )
                    except Exception as e:
                        logger.error(f"Error saving error result to Snowflake: {e}")
                        
                return result
            
            # Domain is worth crawling, add to crawl queue
            url = f"https://{domain}"
            
            # Apply rate limiting for crawling
            with crawl_counter_lock:
                global crawl_counter
                crawl_counter += 1
                current_count = crawl_counter
            
            # If we've exceeded rate limit, add delay
            if current_count > MAX_CRAWLS_PER_MINUTE:
                # Add a random delay between 60-90 seconds
                delay = 60 + random.randint(0, 30)
                logger.info(f"Rate limiting active for {domain}, waiting {delay} seconds")
                time.sleep(delay)
                
                # Reset counter occasionally
                if current_count % (MAX_CRAWLS_PER_MINUTE * 2) == 0:
                    with crawl_counter_lock:
                        crawl_counter = 0
            
            # Use semaphore to limit concurrent crawls
            with crawl_semaphore:
                try:
                    logger.info(f"Crawling website for {domain}")
                    result["content_source"] = "fresh_crawl"
                    
                    # Crawl the website
                    content, (error_type, error_detail), crawler_type = crawl_website(url)
                    
                    # Update result with crawler info
                    result["crawler_type"] = crawler_type
                    
                    if not content:
                        # Crawl failed
                        result.update({
                            "status": "error",
                            "error_type": error_type or "crawl_failed",
                            "error_detail": error_detail or "Failed to crawl website",
                            "predicted_class": "Unknown",
                            "final_classification": "7-No Website available" if error_type == "dns_error" else "2-Internal IT"
                        })
                        
                        # CHANGE 6: Update summary stats for errors
                        if process_id in PROCESS_STORAGE:
                            try:
                                if 'summary' not in PROCESS_STORAGE[process_id]:
                                    PROCESS_STORAGE[process_id]['summary'] = {
                                        'crawled': 0,
                                        'cached': 0,
                                        'errors': 0,
                                        'apollo_enriched': 0,  # Add counter for Apollo enrichment
                                        'by_class': {
                                            "Managed Service Provider": 0,
                                            "Integrator - Commercial A/V": 0,
                                            "Integrator - Residential A/V": 0,
                                            "Internal IT Department": 0,
                                            "Parked Domain": 0,
                                            "Unknown": 0
                                        }
                                    }
                                
                                # Increment error count
                                PROCESS_STORAGE[process_id]['summary']['errors'] += 1
                                
                                # Update class count
                                PROCESS_STORAGE[process_id]['summary']['by_class']['Unknown'] += 1
                            except Exception as stats_error:
                                logger.error(f"Error updating summary stats for error result: {stats_error}")
                        
                        # Save error result with process_id
                        if snowflake_conn and process_id:
                            try:
                                from domain_classifier.storage.operations import save_to_snowflake
                                save_to_snowflake(
                                    domain=domain, 
                                    url=f"https://{domain}", 
                                    content=None, 
                                    classification=result,
                                    snowflake_conn=snowflake_conn,
                                    bulk_process_id=process_id
                                )
                            except Exception as e:
                                logger.error(f"Error saving error result to Snowflake: {e}")
                                
                        return result
                    else:
                        # CHANGE 7: Update summary stats for fresh crawls
                        if process_id in PROCESS_STORAGE:
                            try:
                                if 'summary' not in PROCESS_STORAGE[process_id]:
                                    PROCESS_STORAGE[process_id]['summary'] = {
                                        'crawled': 0,
                                        'cached': 0,
                                        'errors': 0,
                                        'apollo_enriched': 0,  # Add counter for Apollo enrichment
                                        'by_class': {
                                            "Managed Service Provider": 0,
                                            "Integrator - Commercial A/V": 0,
                                            "Integrator - Residential A/V": 0,
                                            "Internal IT Department": 0,
                                            "Parked Domain": 0,
                                            "Unknown": 0
                                        }
                                    }
                                
                                # Increment crawled count
                                PROCESS_STORAGE[process_id]['summary']['crawled'] += 1
                            except Exception as stats_error:
                                logger.error(f"Error updating summary stats for crawled result: {stats_error}")
                        
                except Exception as e:
                    # Crawl threw an exception
                    logger.error(f"Error crawling {domain}: {e}")
                    result.update({
                        "status": "error",
                        "error_type": "crawl_exception",
                        "error_detail": str(e),
                        "predicted_class": "Unknown",
                        "final_classification": "2-Internal IT"
                    })
                    
                    # CHANGE 8: Update summary stats for errors
                    if process_id in PROCESS_STORAGE:
                        try:
                            if 'summary' not in PROCESS_STORAGE[process_id]:
                                PROCESS_STORAGE[process_id]['summary'] = {
                                    'crawled': 0,
                                    'cached': 0,
                                    'errors': 0,
                                    'apollo_enriched': 0,  # Add counter for Apollo enrichment
                                    'by_class': {
                                        "Managed Service Provider": 0,
                                        "Integrator - Commercial A/V": 0,
                                        "Integrator - Residential A/V": 0,
                                        "Internal IT Department": 0,
                                        "Parked Domain": 0,
                                        "Unknown": 0
                                    }
                                }
                            
                            # Increment error count
                            PROCESS_STORAGE[process_id]['summary']['errors'] += 1
                            
                            # Update class count
                            PROCESS_STORAGE[process_id]['summary']['by_class']['Unknown'] += 1
                        except Exception as stats_error:
                            logger.error(f"Error updating summary stats for error result: {stats_error}")
                    
                    # Save error result with process_id
                    if snowflake_conn and process_id:
                        try:
                            from domain_classifier.storage.operations import save_to_snowflake
                            save_to_snowflake(
                                domain=domain, 
                                url=f"https://{domain}", 
                                content=None, 
                                classification=result,
                                snowflake_conn=snowflake_conn,
                                bulk_process_id=process_id
                            )
                        except Exception as save_error:
                            logger.error(f"Error saving error result to Snowflake: {save_error}")
                            
                    return result
        
        # If we have content and a classifier, classify the domain
        if content and llm_classifier:
            # Calculate content hash 
            content_hash = hashlib.md5(content.encode()).hexdigest()
            result["content_hash"] = content_hash
            
            # Classify content
            classification = llm_classifier.classify(content, domain)
            
            if classification:
                # Extract key information
                predicted_class = classification.get('predicted_class', 'Unknown')
                detection_method = classification.get('detection_method', 'unknown')
                max_confidence = classification.get('max_confidence', 0.5)
                confidence_scores = classification.get('confidence_scores', {})
                is_parked = classification.get('is_parked', False)
                
                # Set final classification
                final_classification = "2-Internal IT"  # Default
                if predicted_class == "Managed Service Provider":
                    final_classification = "1-MSP"
                elif predicted_class == "Integrator - Commercial A/V":
                    final_classification = "3-Commercial Integrator"
                elif predicted_class == "Integrator - Residential A/V":
                    final_classification = "4-Residential Integrator"
                elif predicted_class == "Parked Domain" or is_parked:
                    final_classification = "6-Parked Domain - no enrichment"
                elif predicted_class == "Internal IT Department":
                    final_classification = "2-Internal IT"
                
                # Update the result
                result.update({
                    "predicted_class": predicted_class,
                    "confidence_score": int(max_confidence * 100),
                    "confidence_scores": confidence_scores,
                    "detection_method": detection_method,
                    "final_classification": final_classification,
                    "source": "classified",
                    "is_parked": is_parked
                })
                
                # Add explanation if available (truncated for bulk processing)
                if "llm_explanation" in classification:
                    explanation = classification["llm_explanation"]
                    if len(explanation) > 500:
                        explanation = explanation[:497] + "..."
                    result["explanation"] = explanation
                
                # CHANGE 9: Update summary stats with classification
                if process_id in PROCESS_STORAGE:
                    try:
                        if 'summary' not in PROCESS_STORAGE[process_id]:
                            PROCESS_STORAGE[process_id]['summary'] = {
                                'crawled': 0,
                                'cached': 0,
                                'errors': 0,
                                'apollo_enriched': 0,  # Add counter for Apollo enrichment
                                'by_class': {
                                    "Managed Service Provider": 0,
                                    "Integrator - Commercial A/V": 0,
                                    "Integrator - Residential A/V": 0,
                                    "Internal IT Department": 0,
                                    "Parked Domain": 0,
                                    "Unknown": 0
                                }
                            }
                        
                        # Update class count
                        if predicted_class in PROCESS_STORAGE[process_id]['summary']['by_class']:
                            PROCESS_STORAGE[process_id]['summary']['by_class'][predicted_class] += 1
                        else:
                            PROCESS_STORAGE[process_id]['summary']['by_class']["Unknown"] += 1
                    except Exception as stats_error:
                        logger.error(f"Error updating summary stats for classification result: {stats_error}")
                
                # Add Apollo enrichment here for fresh classifications
                if predicted_class != "Parked Domain":
                    try:
                        # Import Apollo connector
                        from domain_classifier.enrichment.apollo_connector import ApolloConnector
                        
                        # Initialize Apollo with a small delay to avoid rate limiting
                        time.sleep(0.5)  # Small delay
                        apollo = ApolloConnector()
                        
                        # Call the enrichment with retry logic
                        max_retries = 3
                        apollo_data = None
                        
                        for attempt in range(max_retries):
                            try:
                                logger.info(f"Attempting Apollo enrichment for {domain} in batch process (attempt {attempt+1})")
                                apollo_data = apollo.enrich_company(domain)
                                if apollo_data:
                                    result["apollo_data"] = apollo_data
                                    logger.info(f"Successfully enriched {domain} with Apollo data in batch process")
                                    
                                    # Increment Apollo enrichment counter
                                    if process_id in PROCESS_STORAGE and 'summary' in PROCESS_STORAGE[process_id]:
                                        if 'apollo_enriched' not in PROCESS_STORAGE[process_id]['summary']:
                                            PROCESS_STORAGE[process_id]['summary']['apollo_enriched'] = 0
                                        PROCESS_STORAGE[process_id]['summary']['apollo_enriched'] += 1
                                    
                                    break
                                else:
                                    # Add exponential backoff if Apollo returns no data
                                    backoff_time = 2 ** attempt
                                    logger.info(f"Apollo returned no data for {domain}, backing off for {backoff_time}s")
                                    time.sleep(backoff_time)
                            except Exception as apollo_error:
                                logger.warning(f"Apollo enrichment attempt {attempt+1} failed: {apollo_error}")
                                # Add exponential backoff with jitter
                                backoff_time = (2 ** attempt) + random.uniform(0.1, 1.0)
                                logger.info(f"Backing off for {backoff_time:.2f}s before retry")
                                time.sleep(backoff_time)
                    except Exception as apollo_err:
                        logger.error(f"Error during Apollo enrichment in batch process: {apollo_err}")
                
                # Save to Snowflake if available
                if snowflake_conn:
                    try:
                        from domain_classifier.storage.operations import save_to_snowflake
                        save_to_snowflake(
                            domain=domain, 
                            url=f"https://{domain}", 
                            content=content, 
                            classification=classification, 
                            snowflake_conn=snowflake_conn,
                            crawler_type=crawler_type if "crawler_type" in locals() else result.get("content_source"),
                            classifier_type=detection_method,
                            bulk_process_id=process_id,  # Pass the process_id
                            apollo_company_data=result.get("apollo_data", None)  # Pass Apollo data if available
                        )
                    except Exception as e:
                        logger.error(f"Error saving to Snowflake: {e}")
                        # Continue despite the error
            else:
                # Classification failed
                result.update({
                    "status": "error",
                    "error_type": "classification_failed",
                    "predicted_class": "Unknown",
                    "final_classification": "2-Internal IT"  # Default
                })
                
                # CHANGE 10: Update summary stats for errors
                if process_id in PROCESS_STORAGE:
                    try:
                        if 'summary' not in PROCESS_STORAGE[process_id]:
                            PROCESS_STORAGE[process_id]['summary'] = {
                                'crawled': 0,
                                'cached': 0,
                                'errors': 0,
                                'apollo_enriched': 0,  # Add counter for Apollo enrichment
                                'by_class': {
                                    "Managed Service Provider": 0,
                                    "Integrator - Commercial A/V": 0,
                                    "Integrator - Residential A/V": 0,
                                    "Internal IT Department": 0,
                                    "Parked Domain": 0,
                                    "Unknown": 0
                                }
                            }
                        
                        # Increment error count
                        PROCESS_STORAGE[process_id]['summary']['errors'] += 1
                        
                        # Update class count
                        PROCESS_STORAGE[process_id]['summary']['by_class']["Unknown"] += 1
                    except Exception as stats_error:
                        logger.error(f"Error updating summary stats for error result: {stats_error}")
        elif not content:
            # No content available
            result.update({
                "status": "error",
                "error_type": "no_content",
                "error_detail": "No content available for classification",
                "predicted_class": "Unknown",
                "final_classification": "2-Internal IT"  # Default
            })
            
            # CHANGE 11: Update summary stats for errors
            if process_id in PROCESS_STORAGE:
                try:
                    if 'summary' not in PROCESS_STORAGE[process_id]:
                        PROCESS_STORAGE[process_id]['summary'] = {
                            'crawled': 0,
                            'cached': 0,
                            'errors': 0,
                            'apollo_enriched': 0,  # Add counter for Apollo enrichment
                            'by_class': {
                                "Managed Service Provider": 0,
                                "Integrator - Commercial A/V": 0,
                                "Integrator - Residential A/V": 0,
                                "Internal IT Department": 0,
                                "Parked Domain": 0,
                                "Unknown": 0
                            }
                        }
                    
                    # Increment error count
                    PROCESS_STORAGE[process_id]['summary']['errors'] += 1
                    
                    # Update class count
                    PROCESS_STORAGE[process_id]['summary']['by_class']["Unknown"] += 1
                except Exception as stats_error:
                    logger.error(f"Error updating summary stats for error result: {stats_error}")
        else:
            # No classifier available
            result.update({
                "status": "error",
                "error_type": "no_classifier",
                "error_detail": "Classifier not available",
                "predicted_class": "Unknown",
                "final_classification": "2-Internal IT"  # Default
            })
            
            # CHANGE 12: Update summary stats for errors
            if process_id in PROCESS_STORAGE:
                try:
                    if 'summary' not in PROCESS_STORAGE[process_id]:
                        PROCESS_STORAGE[process_id]['summary'] = {
                            'crawled': 0,
                            'cached': 0,
                            'errors': 0,
                            'apollo_enriched': 0,  # Add counter for Apollo enrichment
                            'by_class': {
                                "Managed Service Provider": 0,
                                "Integrator - Commercial A/V": 0,
                                "Integrator - Residential A/V": 0,
                                "Internal IT Department": 0,
                                "Parked Domain": 0,
                                "Unknown": 0
                            }
                        }
                    
                    # Increment error count
                    PROCESS_STORAGE[process_id]['summary']['errors'] += 1
                    
                    # Update class count
                    PROCESS_STORAGE[process_id]['summary']['by_class']["Unknown"] += 1
                except Exception as stats_error:
                    logger.error(f"Error updating summary stats for error result: {stats_error}")
            
            # Save error result with process_id
            if snowflake_conn and process_id:
                try:
                    from domain_classifier.storage.operations import save_to_snowflake
                    save_to_snowflake(
                        domain=domain, 
                        url=f"https://{domain}", 
                        content=None, 
                        classification=result,
                        snowflake_conn=snowflake_conn,
                        bulk_process_id=process_id
                    )
                except Exception as e:
                    logger.error(f"Error saving error result to Snowflake: {e}")
        
        return result
            
    except Exception as e:
        logger.error(f"Error processing domain {domain}: {e}")
        logger.error(traceback.format_exc())
        
        # Return error result
        error_result = {
            "domain": domain,
            "status": "error",
            "error_type": "processing_error",
            "error_detail": str(e),
            "predicted_class": "Unknown",
            "final_classification": "2-Internal IT"  # Default
        }
        
        # CHANGE 13: Update summary stats for unexpected errors
        if process_id in PROCESS_STORAGE:
            try:
                if 'summary' not in PROCESS_STORAGE[process_id]:
                    PROCESS_STORAGE[process_id]['summary'] = {
                        'crawled': 0,
                        'cached': 0,
                        'errors': 0,
                        'apollo_enriched': 0,  # Add counter for Apollo enrichment
                        'by_class': {
                            "Managed Service Provider": 0,
                            "Integrator - Commercial A/V": 0,
                            "Integrator - Residential A/V": 0,
                            "Internal IT Department": 0,
                            "Parked Domain": 0,
                            "Unknown": 0
                        }
                    }
                
                # Increment error count
                PROCESS_STORAGE[process_id]['summary']['errors'] += 1
                
                # Update class count
                PROCESS_STORAGE[process_id]['summary']['by_class']["Unknown"] += 1
            except Exception as stats_error:
                logger.error(f"Error updating summary stats for error result: {stats_error}")
        
        # Save error result with process_id
        if snowflake_conn and process_id:
            try:
                from domain_classifier.storage.operations import save_to_snowflake
                save_to_snowflake(
                    domain=domain, 
                    url=f"https://{domain}", 
                    content=None, 
                    classification=error_result,
                    snowflake_conn=snowflake_conn,
                    bulk_process_id=process_id
                )
            except Exception as save_error:
                logger.error(f"Error saving error result to Snowflake: {save_error}")
                
        return error_result

def bulk_worker(worker_id, process_id, domains, options, result_queue, llm_classifier, snowflake_conn):
    """Worker function to process domains in bulk."""
    total_domains = len(domains)
    completed = 0
    
    try:
        # CHANGE 14: Ensure summary tracking is initialized
        if process_id in PROCESS_STORAGE and 'summary' not in PROCESS_STORAGE[process_id]:
            PROCESS_STORAGE[process_id]['summary'] = {
                'crawled': 0,
                'cached': 0,
                'errors': 0,
                'apollo_enriched': 0,  # Add counter for Apollo enrichment
                'by_class': {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0,
                    "Internal IT Department": 0,
                    "Parked Domain": 0,
                    "Unknown": 0
                }
            }
        
        for domain in domains:
            # Process the domain with crawling support
            result = process_domain_with_crawling(
                domain=domain,
                use_existing_content=options.get('use_existing_content', True),
                force_reclassify=options.get('force_reclassify', False),
                llm_classifier=llm_classifier,
                snowflake_conn=snowflake_conn,
                process_id=process_id  # Pass the process_id (not the worker_id)
            )
            
            # Put the result in the queue
            result_queue.put(result)
            
            # Update progress
            completed += 1
            
            # Update progress in Snowflake
            if completed % 5 == 0 or completed == total_domains:
                try:
                    conn = snowflake_conn.get_connection()
                    if conn:
                        try:
                            cursor = conn.cursor()
                            # Log the progress update attempt
                            logger.info(f"Worker {worker_id}: Updating progress to {completed}/{total_domains} domains for process {process_id}")
                            cursor.execute("""
                                UPDATE DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                                SET COMPLETED_DOMAINS = %s,
                                    PROGRESS = %s,
                                    UPDATED_AT = CURRENT_TIMESTAMP()
                                WHERE PROCESS_ID = %s
                            """, (completed, int((completed / total_domains) * 100), process_id))
                            conn.commit()
                            cursor.close()
                        except Exception as update_error:
                            logger.error(f"Error updating progress in Snowflake: {update_error}")
                        finally:
                            conn.close()
                except Exception as conn_error:
                    logger.error(f"Error getting Snowflake connection: {conn_error}")
            
            # Log progress occasionally
            if completed % 10 == 0 or completed == total_domains:
                logger.info(f"Worker {worker_id}: Processed {completed}/{total_domains} domains")
                
            # Add a small random delay between domains to avoid overwhelming systems
            time.sleep(random.uniform(0.1, 0.5))
                
    except Exception as e:
        logger.error(f"Error in bulk worker {worker_id}: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Signal completion
        result_queue.put(None)
        logger.info(f"Worker {worker_id} completed processing {completed}/{total_domains} domains")

def run_bulk_process(process_id, domains, options, llm_classifier, snowflake_conn):
    """Run a bulk processing job with Snowflake storage."""
    try:
        # Update process status in Snowflake
        conn = snowflake_conn.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                    SET STATUS = 'running', UPDATED_AT = CURRENT_TIMESTAMP()
                    WHERE PROCESS_ID = %s
                """, (process_id,))
                conn.commit()
                cursor.close()
            except Exception as e:
                logger.error(f"Error updating process status in Snowflake: {e}")
            finally:
                conn.close()
        
        # Create a result queue
        result_queue = queue.Queue()
        
        # Split domains into chunks for parallel processing
        chunk_size = options.get('chunk_size', 500)
        max_workers = options.get('max_workers', 5)
        domain_chunks = [domains[i:i+chunk_size] for i in range(0, len(domains), chunk_size)]
        
        logger.info(f"Bulk process {process_id}: Processing {len(domains)} domains in {len(domain_chunks)} chunks with {max_workers} workers")
        
        # Initialize summary statistics
        # CHANGE 15: Initialize summary statistics properly at the beginning
        PROCESS_STORAGE[process_id]['summary'] = {
            'crawled': 0,
            'cached': 0,
            'errors': 0,
            'apollo_enriched': 0,  # Add counter for Apollo enrichment
            'by_class': {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0,
                "Internal IT Department": 0,
                "Parked Domain": 0,
                "Unknown": 0
            }
        }
        
        # Start worker threads (limited by max_workers)
        workers = []
        active_workers = 0
        
        # Process chunks with controlled parallelism
        for i, chunk in enumerate(domain_chunks):
            # Wait if we already have max_workers active
            while active_workers >= max_workers:
                # Sleep a bit and check if any workers have finished
                time.sleep(1)
                active_workers = sum(1 for w in workers if w.is_alive())
            
            # Generate a unique worker ID
            worker_id = f"{process_id}_{i}"
            
            # Start a new worker - FIXED: pass process_id separately from worker_id
            worker = threading.Thread(
                target=bulk_worker,
                args=(worker_id, process_id, chunk, options, result_queue, llm_classifier, snowflake_conn),
                daemon=True
            )
            worker.start()
            workers.append(worker)
            active_workers += 1
            
            # Log new worker
            logger.info(f"Bulk process {process_id}: Started worker {i+1}/{len(domain_chunks)} for {len(chunk)} domains")
            
            # Add a small delay between starting workers
            time.sleep(1)
            
        # Create temporary files for results
        temp_files = {}
        if options.get('output_format') == 'csv':
            csv_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            fieldnames = [
                'domain', 'predicted_class', 'confidence_score', 'final_classification',
                'detection_method', 'status', 'error_type', 'error_detail', 'content_source'
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            temp_files['csv'] = csv_file.name
        
        json_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        json_file.write(b'[\n')
        json_file.flush()
        temp_files['json'] = json_file.name
        
        # Process results as they come in
        active_workers = len(workers)
        results_count = 0
        results = []
        
        # Periodically update progress even if no results to avoid progress appearing stuck
        last_progress_update = time.time()
        
        while active_workers > 0:
            try:
                # Get result from queue with timeout
                result = result_queue.get(timeout=10)
                
                # Check if worker has completed
                if result is None:
                    active_workers -= 1
                    continue
                    
                # Process the result
                results_count += 1
                results.append(result)
                
                # Update progress in Snowflake frequently
                # FIXED: Update progress more frequently and log the result
                try:
                    conn = snowflake_conn.get_connection()
                    if conn:
                        try:
                            cursor = conn.cursor()
                            
                            # Update progress
                            completed = results_count
                            total = len(domains)
                            progress = int((completed / total) * 100)
                            
                            logger.info(f"Bulk process {process_id}: Updating overall progress to {completed}/{total} domains ({progress}%)")
                            
                            # Only update basic progress info
                            cursor.execute("""
                                UPDATE DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                                SET COMPLETED_DOMAINS = %s,
                                    PROGRESS = %s,
                                    UPDATED_AT = CURRENT_TIMESTAMP()
                                WHERE PROCESS_ID = %s
                            """, (completed, progress, process_id))
                            conn.commit()
                            cursor.close()
                            
                            # Also verify the update was successful by querying the current status
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT COMPLETED_DOMAINS, PROGRESS FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                                WHERE PROCESS_ID = %s
                            """, (process_id,))
                            verify_result = cursor.fetchone()
                            if verify_result:
                                logger.info(f"Verified progress update: COMPLETED_DOMAINS={verify_result[0]}, PROGRESS={verify_result[1]}")
                            cursor.close()
                        except Exception as update_error:
                            logger.error(f"Error updating progress in Snowflake: {update_error}")
                        finally:
                            conn.close()
                    
                    # Update timestamp of last progress update
                    last_progress_update = time.time()
                except Exception as conn_error:
                    logger.error(f"Error getting Snowflake connection: {conn_error}")
                
                # Append result to temp files
                if options.get('output_format') == 'csv' and 'csv' in temp_files:
                    # Write to CSV - only include fields in fieldnames
                    csv_row = {k: str(v) for k, v in result.items() if k in fieldnames}
                    # Open in append mode to avoid issues with concurrent writes
                    with open(temp_files['csv'], 'a', newline='') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writerow(csv_row)
                
                # Write to JSON (always create JSON)
                with open(temp_files['json'], 'a') as f:
                    json.dump(result, f)
                    if active_workers > 0 or results_count < len(domains):
                        f.write(',\n')
                    else:
                        f.write('\n')
                        
                # Log progress occasionally
                if results_count % 100 == 0:
                    logger.info(f"Bulk process {process_id}: Processed {results_count}/{len(domains)} domains")
                    logger.info(f"  - Crawled: {PROCESS_STORAGE[process_id]['summary']['crawled']}, Cached: {PROCESS_STORAGE[process_id]['summary']['cached']}, Errors: {PROCESS_STORAGE[process_id]['summary']['errors']}, Apollo enriched: {PROCESS_STORAGE[process_id]['summary'].get('apollo_enriched', 0)}")
                
            except queue.Empty:
                # No results available in the timeout period
                # FIXED: Update progress periodically even if no results
                current_time = time.time()
                if current_time - last_progress_update > 30:  # Update at least every 30 seconds
                    try:
                        conn = snowflake_conn.get_connection()
                        if conn:
                            # Check if all workers are still alive
                            alive_workers = sum(1 for w in workers if w.is_alive())
                            if alive_workers < active_workers:
                                active_workers = alive_workers
                                logger.info(f"Bulk process {process_id}: {active_workers} workers still active")
                                
                            # Get current progress
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT COMPLETED_DOMAINS, TOTAL_DOMAINS FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                                WHERE PROCESS_ID = %s
                            """, (process_id,))
                            progress_data = cursor.fetchone()
                            
                            if progress_data:
                                completed, total = progress_data
                                progress = int((completed / total) * 100) if total > 0 else 0
                                logger.info(f"Bulk process {process_id}: Progress {completed}/{total} domains ({progress}%)")
                                
                                # Update in-memory storage for better status reporting
                                if process_id in PROCESS_STORAGE:
                                    PROCESS_STORAGE[process_id]['results_count'] = completed
                            
                            cursor.close()
                            conn.close()
                            
                            # Update timestamp
                            last_progress_update = current_time
                    except Exception as e:
                        logger.error(f"Error updating progress during idle period: {e}")
        
        # Finalize JSON file
        with open(temp_files['json'], 'a') as f:
            f.write(']')
            
        # Calculate duration
        started_at = None
        conn = snowflake_conn.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT CREATED_AT FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                    WHERE PROCESS_ID = %s
                """, (process_id,))
                result = cursor.fetchone()
                if result:
                    started_at = result[0]
                cursor.close()
            except Exception as e:
                logger.error(f"Error getting start time from Snowflake: {e}")
            finally:
                conn.close()
                
        completed_at = datetime.now()
        if started_at:
            duration_seconds = (completed_at - started_at).total_seconds()
        else:
            duration_seconds = 0
        
        # Store results location and update status to completed
        results_location = json.dumps({
            'json_file': temp_files.get('json'),
            'csv_file': temp_files.get('csv')
        })
        
        conn = snowflake_conn.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                    SET STATUS = 'completed',
                        COMPLETED_AT = CURRENT_TIMESTAMP(),
                        COMPLETED_DOMAINS = %s,
                        PROGRESS = 100,
                        RESULTS_LOCATION = %s
                    WHERE PROCESS_ID = %s
                """, (results_count, results_location, process_id))
                conn.commit()
                cursor.close()
            except Exception as e:
                logger.error(f"Error updating completion status in Snowflake: {e}")
            finally:
                conn.close()
                
        logger.info(f"Bulk process {process_id} completed: {results_count}/{len(domains)} domains processed in {duration_seconds:.1f} seconds")
        logger.info(f"Summary: Crawled: {PROCESS_STORAGE[process_id]['summary']['crawled']}, Cached: {PROCESS_STORAGE[process_id]['summary']['cached']}, Errors: {PROCESS_STORAGE[process_id]['summary']['errors']}, Apollo enriched: {PROCESS_STORAGE[process_id]['summary'].get('apollo_enriched', 0)}")
        logger.info(f"Classes: {PROCESS_STORAGE[process_id]['summary']['by_class']}")
        
        # Store in memory as well as a backup
        PROCESS_STORAGE[process_id] = {
            'status': 'completed',
            'summary': PROCESS_STORAGE[process_id]['summary'],
            'results_location': results_location,
            'completed_at': completed_at.isoformat(),
            'duration_seconds': duration_seconds,
            'results_count': results_count
        }
        
    except Exception as e:
        logger.error(f"Error in bulk process {process_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Update process status to error
        conn = snowflake_conn.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                    SET STATUS = 'error',
                        ERROR = %s,
                        UPDATED_AT = CURRENT_TIMESTAMP()
                    WHERE PROCESS_ID = %s
                """, (str(e), process_id))
                conn.commit()
                cursor.close()
            except Exception as update_error:
                logger.error(f"Error updating error status in Snowflake: {update_error}")
            finally:
                conn.close()
                
        # Store error in memory as well
        PROCESS_STORAGE[process_id] = {
            'status': 'error',
            'error': str(e)
        }

def register_bulk_routes(app, llm_classifier, snowflake_conn):
    """Register bulk processing routes."""
    
    @app.route('/bulk-process', methods=['POST', 'OPTIONS'])
    def start_bulk_process():
        """Start a bulk processing job for a large list of domains."""
        if request.method == 'OPTIONS':
            return '', 204
            
        try:
            data = request.json
            domains = data.get('domains', [])
            
            if not domains:
                return jsonify({"error": "No domains provided"}), 400
                
            # Generate a unique process ID
            process_id = f"bulk_{int(time.time())}_{hash(str(domains)[:100]) % 1000}"[:30]
            
            # Extract options
            options = {
                'use_existing_content': data.get('use_existing_content', True),
                'force_reclassify': data.get('force_reclassify', False),
                'output_format': data.get('output_format', 'json'),
                'chunk_size': min(data.get('chunk_size', 500), 1000),  # Smaller chunks for crawling
                'max_workers': min(data.get('max_workers', 5), MAX_WORKERS),
                'include_crawling': data.get('include_crawling', True)
            }
            
            # Store basic process info in Snowflake
            conn = snowflake_conn.get_connection()
            if not conn:
                return jsonify({"error": "Could not connect to Snowflake"}), 500
                
            try:
                cursor = conn.cursor()
                # Only store basic information - no JSON/VARIANT fields
                cursor.execute("""
                    INSERT INTO DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                    (PROCESS_ID, TOTAL_DOMAINS, STATUS, PROGRESS, COMPLETED_DOMAINS)
                    VALUES (%s, %s, 'starting', 0, 0)
                """, (
                    process_id,
                    len(domains)
                ))
                conn.commit()
                cursor.close()
            except Exception as e:
                logger.error(f"Error saving bulk process to Snowflake: {e}")
                conn.rollback()
                return jsonify({"error": f"Database error: {str(e)}"}), 500
            finally:
                conn.close()
                
            # Store in memory as backup
            PROCESS_STORAGE[process_id] = {
                'domains': domains,
                'options': options,
                'total_domains': len(domains),
                'status': 'starting',
                'created_at': datetime.now().isoformat(),
                'results_count': 0  # Initialize with 0 completed domains
            }
            
            # Start the processing in a background thread
            threading.Thread(
                target=run_bulk_process,
                args=(process_id, domains, options, llm_classifier, snowflake_conn),
                daemon=True
            ).start()
            
            return jsonify({
                'process_id': process_id,
                'total_domains': len(domains),
                'status': 'starting',
                'status_endpoint': f'/bulk-status/{process_id}',
                'download_endpoint': f'/bulk-download/{process_id}',
                'message': 'Bulk processing job started'
            }), 202
            
        except Exception as e:
            logger.error(f"Error starting bulk process: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500
    
    @app.route('/bulk-status/<process_id>', methods=['GET'])
    def get_bulk_status(process_id):
        """Get the status of a bulk processing job from Snowflake with memory fallback."""
        # Add logging to troubleshoot
        logger.info(f"Checking status for process ID: {process_id}")
        
        # Try Snowflake first
        try:
            conn = snowflake_conn.get_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT PROCESS_ID, TOTAL_DOMAINS, COMPLETED_DOMAINS, PROGRESS, 
                               STATUS, CREATED_AT, COMPLETED_AT, ERROR
                        FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                        WHERE PROCESS_ID = %s
                    """, (process_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        # Get column names from cursor description
                        column_names = [col[0] for col in cursor.description]
                        status_data = dict(zip(column_names, result))
                        
                        # Log the retrieved status data for debugging
                        logger.info(f"Retrieved status data: COMPLETED_DOMAINS={status_data['COMPLETED_DOMAINS']}, PROGRESS={status_data['PROGRESS']}")
                        
                        # Convert timestamps to strings for JSON serialization
                        for key in ['CREATED_AT', 'COMPLETED_AT']:
                            if status_data.get(key):
                                status_data[key] = status_data[key].isoformat()
                                
                        cursor.close()
                                
                        return jsonify({
                            'process_id': status_data['PROCESS_ID'],
                            'total_domains': status_data['TOTAL_DOMAINS'],
                            'completed': status_data['COMPLETED_DOMAINS'],
                            'progress': status_data['PROGRESS'],
                            'status': status_data['STATUS'],
                            'started_at': status_data['CREATED_AT'],
                            'completed_at': status_data.get('COMPLETED_AT'),
                            'error': status_data.get('ERROR'),
                            'is_complete': status_data['STATUS'] == 'completed'
                        }), 200
                except Exception as e:
                    logger.warning(f"Error checking bulk status in Snowflake: {e}")
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"Could not connect to Snowflake for status check: {e}")
            
        # Fall back to in-memory storage if Snowflake query fails
        if process_id in PROCESS_STORAGE:
            process_data = PROCESS_STORAGE[process_id]
            logger.info(f"Using in-memory data: results_count={process_data.get('results_count', 0)}")
            
            return jsonify({
                'process_id': process_id,
                'total_domains': process_data.get('total_domains', 0),
                'completed': process_data.get('results_count', 0),
                'progress': int((process_data.get('results_count', 0) / process_data.get('total_domains', 1)) * 100) if process_data.get('total_domains', 0) > 0 else 0,
                'status': process_data.get('status', 'unknown'),
                'started_at': process_data.get('created_at'),
                'completed_at': process_data.get('completed_at'),
                'error': process_data.get('error'),
                'is_complete': process_data.get('status') == 'completed',
                'from_memory': True
            }), 200
            
        # If not found in either place
        return jsonify({"error": "Process ID not found"}), 404
    
    @app.route('/bulk-results/<process_id>', methods=['GET'])
    def get_bulk_results(process_id):
        """Get the results of a completed bulk job from file system with Snowflake tracking."""
        # Try to get results location from Snowflake
        results_location = None
        try:
            conn = snowflake_conn.get_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT PROCESS_ID, TOTAL_DOMAINS, COMPLETED_DOMAINS, STATUS, 
                               RESULTS_LOCATION, COMPLETED_AT, ERROR
                        FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                        WHERE PROCESS_ID = %s
                    """, (process_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        # Get column names from cursor description
                        column_names = [col[0] for col in cursor.description]
                        process_data = dict(zip(column_names, result))
                        
                        # Only if process is complete
                        if process_data['STATUS'] != 'completed':
                            return jsonify({
                                'process_id': process_id,
                                'status': process_data['STATUS'],
                                'error': process_data.get('ERROR') or 'Process not yet completed',
                                'progress': process_data['COMPLETED_DOMAINS'] / process_data['TOTAL_DOMAINS'] * 100 if process_data['TOTAL_DOMAINS'] > 0 else 0
                            }), 400
                            
                        # Get results location
                        if process_data.get('RESULTS_LOCATION'):
                            results_location = json.loads(process_data['RESULTS_LOCATION'])
                except Exception as e:
                    logger.warning(f"Error getting results location from Snowflake: {e}")
                finally:
                    cursor.close()
                    conn.close()
        except Exception as e:
            logger.warning(f"Could not connect to Snowflake for results: {e}")

        # Fall back to memory if Snowflake fails
        if not results_location and process_id in PROCESS_STORAGE:
            process_data = PROCESS_STORAGE[process_id]
            if process_data.get('status') != 'completed':
                return jsonify({
                    'process_id': process_id,
                    'status': process_data.get('status', 'unknown'),
                    'error': process_data.get('error') or 'Process not yet completed',
                    'progress': int((process_data.get('results_count', 0) / process_data.get('total_domains', 1)) * 100) if process_data.get('total_domains', 0) > 0 else 0,
                    'from_memory': True
                }), 400
                
            # Get results location from memory
            results_location = json.loads(process_data.get('results_location', '{}')) if process_data.get('results_location') else None
            
        # If we have a results location, load the results
        results = {}
        if results_location and 'json_file' in results_location and os.path.exists(results_location['json_file']):
            try:
                with open(results_location['json_file'], 'r') as f:
                    all_results = json.load(f)
                    # Convert to domain -> result mapping
                    for result in all_results:
                        domain = result.get('domain')
                        if domain:
                            results[domain] = {"result": result, "status_code": 200}
            except Exception as e:
                logger.error(f"Error loading results from file: {e}")
                return jsonify({
                    'process_id': process_id,
                    'error': f"Error loading results: {str(e)}",
                    'results': {}
                }), 500
        else:
            # If results file doesn't exist, return a message
            return jsonify({
                'process_id': process_id,
                'status': 'completed',
                'error': 'Results file not found or has been cleaned up',
                'results': {}
            }), 200
            
        # Return successfully loaded results
        total_domains = len(results)
        if process_id in PROCESS_STORAGE:
            total_domains = PROCESS_STORAGE[process_id].get('total_domains', len(results))
            
        return jsonify({
            'process_id': process_id,
            'total_domains': total_domains,
            'results': results
        }), 200
    
    @app.route('/bulk-download/<process_id>', methods=['GET'])
    def download_bulk_results(process_id):
        """Download the results of a bulk processing job."""
        # Try to get results location from Snowflake
        results_location = None
        try:
            conn = snowflake_conn.get_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT PROCESS_ID, STATUS, RESULTS_LOCATION
                        FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                        WHERE PROCESS_ID = %s
                    """, (process_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        # Get column names from cursor description
                        column_names = [col[0] for col in cursor.description]
                        process_data = dict(zip(column_names, result))
                        
                        # Only if process is complete
                        if process_data['STATUS'] != 'completed':
                            return jsonify({
                                'process_id': process_id,
                                'status': process_data['STATUS'],
                                'error': 'Process not yet completed'
                            }), 400
                            
                        # Get results location
                        if process_data.get('RESULTS_LOCATION'):
                            results_location = json.loads(process_data['RESULTS_LOCATION'])
                except Exception as e:
                    logger.warning(f"Error getting results location from Snowflake: {e}")
                finally:
                    cursor.close()
                    conn.close()
        except Exception as e:
            logger.warning(f"Could not connect to Snowflake for download: {e}")
            
        # Fall back to memory if Snowflake fails
        if not results_location and process_id in PROCESS_STORAGE:
            process_data = PROCESS_STORAGE[process_id]
            if process_data.get('status') != 'completed':
                return jsonify({
                    'process_id': process_id,
                    'status': process_data.get('status', 'unknown'),
                    'error': 'Process not yet completed'
                }), 400
                
            # Get results location from memory
            results_location = json.loads(process_data.get('results_location', '{}')) if process_data.get('results_location') else None
        
        # Get output format
        output_format = request.args.get('format', 'json')
        
        # If we have a results location, serve the file
        if results_location:
            if output_format == 'csv' and 'csv_file' in results_location and os.path.exists(results_location['csv_file']):
                return send_file(
                    results_location['csv_file'],
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name=f"bulk_results_{process_id}.csv"
                )
            elif 'json_file' in results_location and os.path.exists(results_location['json_file']):
                return send_file(
                    results_location['json_file'],
                    mimetype='application/json',
                    as_attachment=True,
                    download_name=f"bulk_results_{process_id}.json"
                )
        
        # If results file doesn't exist
        return jsonify({
            'process_id': process_id,
            'error': 'Results file not found or has been cleaned up'
        }), 404
    
    @app.route('/bulk-summary/<process_id>', methods=['GET'])
    def get_bulk_summary(process_id):
        """Get detailed summary statistics for a bulk process."""
        # Try to get data from Snowflake
        completed = 0
        total = 0
        status = 'unknown'
        created_at = None
        completed_at = None
        
        try:
            conn = snowflake_conn.get_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT PROCESS_ID, TOTAL_DOMAINS, COMPLETED_DOMAINS, PROGRESS, 
                               STATUS, CREATED_AT, COMPLETED_AT
                        FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                        WHERE PROCESS_ID = %s
                    """, (process_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        # Get column names from cursor description
                        column_names = [col[0] for col in cursor.description]
                        process_data = dict(zip(column_names, result))
                        
                        # Extract basic metrics
                        total = process_data['TOTAL_DOMAINS']
                        completed = process_data['COMPLETED_DOMAINS']
                        status = process_data['STATUS']
                        created_at = process_data['CREATED_AT']
                        completed_at = process_data.get('COMPLETED_AT')
                except Exception as e:
                    logger.warning(f"Error getting summary data from Snowflake: {e}")
                finally:
                    cursor.close()
                    conn.close()
        except Exception as e:
            logger.warning(f"Could not connect to Snowflake for summary: {e}")
            
        # Get additional details from memory
        summary = {
            'crawled': 0,
            'cached': 0,
            'errors': 0,
            'apollo_enriched': 0,  # Add counter for Apollo enrichment
            'by_class': {}
        }
        
        duration_seconds = 0
        
        if process_id in PROCESS_STORAGE:
            process_data = PROCESS_STORAGE[process_id]
            
            # Use memory data if we couldn't get from Snowflake
            if total == 0:
                total = process_data.get('total_domains', 0)
            if completed == 0:
                completed = process_data.get('results_count', 0)
            if status == 'unknown':
                status = process_data.get('status', 'unknown')
                
            # Get summary stats from memory
            if 'summary' in process_data:
                summary = process_data['summary']
                
            # Get timing data
            if process_data.get('duration_seconds'):
                duration_seconds = process_data['duration_seconds']
            elif process_data.get('completed_at') and process_data.get('created_at'):
                try:
                    completed_time = datetime.fromisoformat(process_data['completed_at'])
                    created_time = datetime.fromisoformat(process_data['created_at'])
                    duration_seconds = (completed_time - created_time).total_seconds()
                except (ValueError, TypeError):
                    duration_seconds = 0
        
        # If not found in either place
        if total == 0 and status == 'unknown':
            return jsonify({"error": "Process ID not found"}), 404
            
        # Calculate additional metrics
        if completed_at and created_at and duration_seconds == 0:
            duration_seconds = (completed_at - created_at).total_seconds()
                
        # Calculate class percentages
        class_percentages = {}
        if completed > 0:
            for cls, count in summary.get('by_class', {}).items():
                class_percentages[cls] = round((count / completed) * 100, 1)
                
        # Calculate crawl/cache percentages
        crawl_percentage = 0
        cache_percentage = 0
        error_percentage = 0
        apollo_percentage = 0  # Add Apollo percentage
        
        if completed > 0:
            crawl_percentage = round((summary.get('crawled', 0) / completed) * 100, 1)
            cache_percentage = round((summary.get('cached', 0) / completed) * 100, 1)
            error_percentage = round((summary.get('errors', 0) / completed) * 100, 1)
            apollo_percentage = round((summary.get('apollo_enriched', 0) / completed) * 100, 1)  # Calculate Apollo enrichment percentage
            
        # Calculate performance metrics
        avg_seconds_per_domain = 0
        domains_per_minute = 0
        time_remaining = "N/A"
        estimated_seconds_remaining = 0
        
        if duration_seconds > 0 and completed > 0:
            avg_seconds_per_domain = round(duration_seconds / completed, 3)
            domains_per_minute = round((completed / (duration_seconds / 60)), 1) if duration_seconds > 0 else 0
            
            # Estimate time remaining
            if completed < total and status == 'running':
                remaining_domains = total - completed
                estimated_seconds_remaining = remaining_domains * avg_seconds_per_domain
                
                # Format time remaining
                if estimated_seconds_remaining < 60:
                    time_remaining = f"{round(estimated_seconds_remaining)} seconds"
                elif estimated_seconds_remaining < 3600:
                    time_remaining = f"{round(estimated_seconds_remaining / 60, 1)} minutes"
                else:
                    time_remaining = f"{round(estimated_seconds_remaining / 3600, 1)} hours"
            
        # Format duration
        format_duration = "Not completed"
        if duration_seconds > 0:
            if duration_seconds < 60:
                format_duration = f"{round(duration_seconds)} seconds"
            elif duration_seconds < 3600:
                format_duration = f"{round(duration_seconds / 60, 1)} minutes"
            else:
                hours = int(duration_seconds / 3600)
                remaining_minutes = int((duration_seconds % 3600) / 60)
                format_duration = f"{hours} hours, {remaining_minutes} minutes"
                
        # Return the summary
        return jsonify({
            'process_id': process_id,
            'status': status,
            'total_domains': total,
            'completed_domains': completed,
            'progress_percentage': int((completed / total) * 100) if total > 0 else 0,
            'started_at': created_at.isoformat() if isinstance(created_at, datetime) else created_at,
            'completed_at': completed_at.isoformat() if isinstance(completed_at, datetime) else completed_at,
            'duration_seconds': duration_seconds,
            'format_duration': format_duration,
            'error_count': summary.get('errors', 0),
            'error_percentage': error_percentage,
            'fresh_crawls': summary.get('crawled', 0),
            'crawl_percentage': crawl_percentage,
            'cached_results': summary.get('cached', 0),
            'cache_percentage': cache_percentage,
            'apollo_enriched': summary.get('apollo_enriched', 0),  # Add Apollo enrichment count
            'apollo_percentage': apollo_percentage,  # Add Apollo enrichment percentage
            'class_counts': summary.get('by_class', {}),
            'class_percentages': class_percentages,
            'performance': {
                'avg_seconds_per_domain': avg_seconds_per_domain,
                'domains_per_minute': domains_per_minute,
                'estimated_seconds_remaining': estimated_seconds_remaining,
                'time_remaining': time_remaining
            }
        }), 200
    
    @app.route('/bulk-cleanup', methods=['DELETE'])
    def cleanup_bulk_processes():
        """Clean up old bulk processes and their temporary files."""
        deleted_count = 0
        error_count = 0
        
        # Try Snowflake cleanup first
        conn = None
        try:
            conn = snowflake_conn.get_connection()
            if conn:
                try:
                    # Get completed processes older than 1 day
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT PROCESS_ID, RESULTS_LOCATION
                        FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                        WHERE STATUS IN ('completed', 'error')
                        AND UPDATED_AT < DATEADD(day, -1, CURRENT_TIMESTAMP())
                    """)
                    
                    old_processes = cursor.fetchall()
                    
                    # Clean up each process
                    for process_id, results_location in old_processes:
                        try:
                            # Delete temporary files
                            if results_location:
                                try:
                                    location_data = json.loads(results_location)
                                    for file_key in ['json_file', 'csv_file']:
                                        if file_key in location_data and os.path.exists(location_data[file_key]):
                                            os.unlink(location_data[file_key])
                                except Exception as e:
                                    logger.warning(f"Error deleting files for process {process_id}: {e}")
                            
                            # Delete from database
                            cursor.execute("""
                                DELETE FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                                WHERE PROCESS_ID = %s
                            """, (process_id,))
                            
                            # Also remove from memory if present
                            if process_id in PROCESS_STORAGE:
                                del PROCESS_STORAGE[process_id]
                                
                            deleted_count += 1
                        except Exception as e:
                            logger.error(f"Error cleaning up process {process_id}: {e}")
                            error_count += 1
                            
                    conn.commit()
                    cursor.close()
                except Exception as e:
                    logger.error(f"Error in Snowflake bulk cleanup: {e}")
                    if conn:
                        conn.rollback()
                finally:
                    if conn:
                        conn.close()
        except Exception as e:
            logger.error(f"Could not connect to Snowflake for cleanup: {e}")
            
        # Also clean up memory storage for old processes
        now = datetime.now()
        memory_processes_to_delete = []
        
        for process_id, process_data in PROCESS_STORAGE.items():
            try:
                # Check if created_at exists and is more than 1 day old
                if 'created_at' in process_data:
                    created_time = datetime.fromisoformat(process_data['created_at']) if isinstance(process_data['created_at'], str) else process_data['created_at']
                    if (now - created_time).total_seconds() > 86400:  # 1 day in seconds
                        # Clean up any files
                        if 'results_location' in process_data:
                            try:
                                location_data = json.loads(process_data['results_location']) if isinstance(process_data['results_location'], str) else process_data['results_location']
                                for file_key in ['json_file', 'csv_file']:
                                    if file_key in location_data and os.path.exists(location_data[file_key]):
                                        os.unlink(location_data[file_key])
                            except Exception as e:
                                logger.warning(f"Error deleting memory-stored files for process {process_id}: {e}")
                                
                        # Mark for deletion
                        memory_processes_to_delete.append(process_id)
                        deleted_count += 1
            except Exception as e:
                logger.error(f"Error cleaning up memory process {process_id}: {e}")
                error_count += 1
                
        # Actually delete the processes
        for process_id in memory_processes_to_delete:
            del PROCESS_STORAGE[process_id]
                    
        return jsonify({
            'deleted_count': deleted_count,
            'error_count': error_count
        }), 200
                
    # Add health check route for bulk module
    @app.route('/bulk-health', methods=['GET'])
    def bulk_health_check():
        """Health check for bulk processing module."""
        conn = snowflake_conn.get_connection()
        if not conn:
            return jsonify({
                "status": "degraded",
                "message": "Could not connect to Snowflake"
            }), 200
            
        try:
            # Check if the table exists
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                    LIMIT 1
                """)
                cursor.fetchone()
                table_exists = True
            except Exception:
                table_exists = False
                
            # Create the table if it doesn't exist
            if not table_exists:
                try:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS (
                            ID NUMBER(38,0) NOT NULL AUTOINCREMENT START 1 INCREMENT 1 NOORDER,
                            PROCESS_ID VARCHAR(255) NOT NULL,
                            TOTAL_DOMAINS NUMBER(38,0) NOT NULL,
                            COMPLETED_DOMAINS NUMBER(38,0) DEFAULT 0,
                            PROGRESS NUMBER(38,0) DEFAULT 0,
                            STATUS VARCHAR(50) DEFAULT 'starting',
                            CREATED_AT TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
                            UPDATED_AT TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
                            COMPLETED_AT TIMESTAMP_NTZ(9),
                            RESULTS_LOCATION VARCHAR(1000),
                            ERROR VARCHAR(1000),
                            UNIQUE (PROCESS_ID),
                            PRIMARY KEY (ID)
                        )
                    """)
                    conn.commit()
                    logger.info("Created simplified BULK_PROCESS_STATUS table in Snowflake")
                    table_exists = True
                except Exception as e:
                    logger.error(f"Error creating BULK_PROCESS_STATUS table: {e}")
                    
            # Get active processes
            cursor.execute("""
                SELECT COUNT(*) AS active
                FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                WHERE STATUS = 'running'
            """)
            active_count = cursor.fetchone()[0]
            
            # Get completed processes
            cursor.execute("""
                SELECT COUNT(*) AS completed
                FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.BULK_PROCESS_STATUS
                WHERE STATUS = 'completed'
            """)
            completed_count = cursor.fetchone()[0]
            
            # Get Apollo enrichment summary if available
            apollo_summary = {
                "total_enriched_count": 0,
                "success_rate": "0%"
            }
            
            # Count domains with Apollo data
            try:
                cursor.execute("""
                    SELECT COUNT(*) AS apollo_count
                    FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.DOMAIN_CLASSIFICATION
                    WHERE APOLLO_COMPANY_DATA IS NOT NULL
                """)
                apollo_count = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) AS total_count
                    FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.DOMAIN_CLASSIFICATION
                """)
                total_domains = cursor.fetchone()[0]
                
                if total_domains > 0:
                    apollo_summary = {
                        "total_enriched_count": apollo_count,
                        "success_rate": f"{(apollo_count / total_domains) * 100:.1f}%",
                        "total_domains": total_domains
                    }
            except Exception as apollo_err:
                logger.warning(f"Error getting Apollo stats: {apollo_err}")
                
            cursor.close()
                
            return jsonify({
                "status": "ok" if table_exists else "degraded",
                "table_exists": table_exists,
                "active_processes": active_count,
                "completed_processes": completed_count,
                "snowflake_connected": True,
                "max_workers": MAX_WORKERS,
                "max_crawls_per_minute": MAX_CRAWLS_PER_MINUTE,
                "apollo_enrichment": apollo_summary
            }), 200
            
        except Exception as e:
            logger.error(f"Error in bulk health check: {e}")
            return jsonify({
                "status": "error",
                "error": str(e)
            }), 500
        finally:
            if conn:
                conn.close()

    # Add a route to monitor crawler usage statistics
    @app.route('/crawler-stats', methods=['GET'])
    def get_crawler_stats():
        """Get crawler usage statistics."""
        try:
            # Import the crawler stats from apify_crawler.py
            from domain_classifier.crawlers.apify_crawler import CRAWLER_STATS
            
            # Calculate percentages
            total = max(1, CRAWLER_STATS["total"])
            stats = {
                "total_crawls": CRAWLER_STATS["total"],
                "direct": {
                    "count": CRAWLER_STATS["direct"],
                    "percentage": round(CRAWLER_STATS["direct"] / total * 100, 1)
                },
                "scrapy": {
                    "count": CRAWLER_STATS["scrapy"],
                    "percentage": round(CRAWLER_STATS["scrapy"] / total * 100, 1)
                },
                "apify": {
                    "count": CRAWLER_STATS["apify"],
                    "percentage": round(CRAWLER_STATS["apify"] / total * 100, 1)
                },
                "last_reset": datetime.fromtimestamp(CRAWLER_STATS["last_reset"]).isoformat()
            }
            
            return jsonify(stats), 200
        except Exception as e:
            logger.error(f"Error getting crawler stats: {e}")
            return jsonify({
                "error": str(e)
            }), 500
    
    # Add a route to show Apollo usage statistics
    @app.route('/apollo-stats', methods=['GET'])
    def get_apollo_stats():
        """Get Apollo usage statistics from batch jobs."""
        try:
            # Get stats from memory
            apollo_stats = {
                "batch_processes": {},
                "overall": {
                    "total_enriched": 0,
                    "total_attempted": 0,
                    "success_rate": 0
                }
            }
            
            # Collect stats from all batch processes
            for process_id, process_data in PROCESS_STORAGE.items():
                if 'summary' in process_data:
                    summary = process_data['summary']
                    if 'apollo_enriched' in summary:
                        total_attempted = summary.get('crawled', 0) + summary.get('cached', 0)
                        apollo_enriched = summary.get('apollo_enriched', 0)
                        
                        apollo_stats["batch_processes"][process_id] = {
                            "total_attempted": total_attempted,
                            "apollo_enriched": apollo_enriched,
                            "success_rate": round((apollo_enriched / max(1, total_attempted)) * 100, 1)
                        }
                        
                        # Add to overall stats
                        apollo_stats["overall"]["total_attempted"] += total_attempted
                        apollo_stats["overall"]["total_enriched"] += apollo_enriched
            
            # Calculate overall success rate
            if apollo_stats["overall"]["total_attempted"] > 0:
                apollo_stats["overall"]["success_rate"] = round(
                    (apollo_stats["overall"]["total_enriched"] / 
                     apollo_stats["overall"]["total_attempted"]) * 100, 1
                )
            
            # Try to get stats from Snowflake for a more complete picture
            try:
                conn = snowflake_conn.get_connection()
                if conn:
                    cursor = conn.cursor()
                    
                    # Count domains with Apollo data
                    cursor.execute("""
                        SELECT COUNT(*) AS apollo_count
                        FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.DOMAIN_CLASSIFICATION
                        WHERE APOLLO_COMPANY_DATA IS NOT NULL
                    """)
                    apollo_count = cursor.fetchone()[0]
                    
                    # Count total domains
                    cursor.execute("""
                        SELECT COUNT(*) AS total_count
                        FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.DOMAIN_CLASSIFICATION
                    """)
                    total_domains = cursor.fetchone()[0]
                    
                    # Add to stats
                    apollo_stats["snowflake"] = {
                        "total_domains": total_domains,
                        "apollo_enriched": apollo_count,
                        "success_rate": round((apollo_count / max(1, total_domains)) * 100, 1)
                    }
                    
                    cursor.close()
                    conn.close()
            except Exception as db_err:
                logger.warning(f"Error getting Apollo stats from Snowflake: {db_err}")
            
            return jsonify(apollo_stats), 200
        except Exception as e:
            logger.error(f"Error getting Apollo stats: {e}")
            return jsonify({
                "error": str(e)
            }), 500

    return app
