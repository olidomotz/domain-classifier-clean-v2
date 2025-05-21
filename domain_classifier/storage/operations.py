"""Storage operations for domain classification."""
import logging
import json
import hashlib
from typing import Dict, Any, Optional, Tuple, List
import traceback

# Import from centralized JSON parser module
from domain_classifier.utils.json_parser import parse_and_validate_json, ensure_dict, safe_get

# Set up logging
logger = logging.getLogger(__name__)

def save_to_snowflake(domain: str, url: str, content: str, classification: Dict[str, Any], 
                     snowflake_conn=None, apollo_company_data=None, crawler_type=None, 
                     classifier_type=None, bulk_process_id=None):
    """Save classification data and Apollo enrichment data to Snowflake"""
    try:
        # If no connector is provided, import and create one
        if snowflake_conn is None:
            from domain_classifier.storage.snowflake_connector import SnowflakeConnector
            snowflake_conn = SnowflakeConnector()
            
        # Handle Snowflake connection failures gracefully
        if not getattr(snowflake_conn, 'connected', False):
            logger.info(f"Fallback: Not saving to Snowflake for {domain} - not connected")
            return True  # Return success anyway to avoid blocking the main flow
            
        # Check if content has changed before saving - only if content is provided
        content_saved = False
        if content:
            try:
                # Generate hash for the new content
                new_hash = hashlib.md5(content.encode()).hexdigest()
                
                # Get a list of domains with recent saves to avoid duplicate lookups
                recent_saves = getattr(snowflake_conn, '_recent_domain_saves', {})
                
                if domain in recent_saves:
                    # If we've saved this domain recently and have its content hash, use that
                    existing_hash = recent_saves[domain]
                    
                    if existing_hash == new_hash:
                        logger.info(f"Content unchanged for {domain}, skipping content save")
                        content_saved = True
                    else:
                        # Content has changed, save the new content
                        logger.info(f"Content changed for {domain}, saving new content")
                        snowflake_conn.save_domain_content(
                            domain=domain, url=url, content=content
                        )
                        # Update the cache
                        recent_saves[domain] = new_hash
                        content_saved = True
                else:
                    # No cached hash available, check Snowflake
                    existing_content = snowflake_conn.get_domain_content(domain)
                    if existing_content:
                        # Generate hash for existing content
                        existing_hash = hashlib.md5(existing_content.encode()).hexdigest()
                        
                        # Add to cache for future reference
                        recent_saves[domain] = existing_hash
                        
                        if existing_hash == new_hash:
                            logger.info(f"Content unchanged for {domain}, skipping content save")
                            content_saved = True
                        else:
                            # Content has changed, save the new content
                            logger.info(f"Content changed for {domain}, saving new content")
                            snowflake_conn.save_domain_content(
                                domain=domain, url=url, content=content
                            )
                            # Update the cache
                            recent_saves[domain] = new_hash
                            content_saved = True
                    else:
                        # No existing content, save the new content
                        logger.info(f"No existing content for {domain}, saving new content")
                        snowflake_conn.save_domain_content(
                            domain=domain, url=url, content=content
                        )
                        # Add to cache
                        recent_saves[domain] = new_hash
                        content_saved = True
                
                # Store the cache back on the connector
                setattr(snowflake_conn, '_recent_domain_saves', recent_saves)
                
            except Exception as e:
                logger.warning(f"Error checking/saving content - continuing: {e}")
                # Continue with classification save anyway
        else:
            logger.warning(f"No content provided for {domain}, skipping content save")
            
        # Ensure max_confidence exists
        if 'max_confidence' not in classification:
            confidence_scores = classification.get('confidence_scores', {})
            max_confidence = max(confidence_scores.values()) if confidence_scores else 0.5
            classification['max_confidence'] = max_confidence
            
        # Set low_confidence flag based on confidence threshold
        if 'low_confidence' not in classification:
            # Import from config if available, otherwise use default
            try:
                from domain_classifier.config.settings import LOW_CONFIDENCE_THRESHOLD
                threshold = LOW_CONFIDENCE_THRESHOLD
            except ImportError:
                threshold = 0.7
                
            classification['low_confidence'] = classification['max_confidence'] < threshold
            
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
        
        # Try to save with timeout handling
        try:
            # CHANGE 1: For cached results, update the BULK_PROCESS_ID
            if classification.get("source") == "cached" and bulk_process_id:
                # Update the existing classification with the new bulk_process_id
                update_success, _ = snowflake_conn.update_classification_bulk_id(
                    domain=domain,
                    bulk_process_id=bulk_process_id
                )
                if not update_success:
                    logger.warning(f"Failed to update BULK_PROCESS_ID for cached result {domain}, but continuing")
            else:
                # Normal insert for new classifications
                save_success, _ = snowflake_conn.save_classification(
                    domain=domain,
                    company_type=str(company_type),
                    confidence_score=float(classification['max_confidence']),
                    all_scores=json.dumps(classification.get('confidence_scores', {}))[:4000],  # Limit size
                    model_metadata=model_metadata_json,
                    low_confidence=bool(classification.get('low_confidence', False)),
                    detection_method=str(classification.get('detection_method', 'llm_classification')),
                    llm_explanation=llm_explanation,  # Add explanation directly to save_classification
                    apollo_company_data=apollo_company_data,
                    crawler_type=crawler_type,
                    classifier_type=classifier_type,
                    bulk_process_id=bulk_process_id  # Add bulk_process_id
                )
                
                if not save_success:
                    logger.warning(f"Failed to save classification for {domain}, but continuing")
        except Exception as e:
            logger.error(f"Error in save_classification: {e}")
            # Continue with vector DB save anyway
        
        # Also save to vector database - still try even if Snowflake save failed
        try:
            # If content is already saved, pass content_saved=True to avoid redundant hash checks
            save_to_vector_db(domain, url, content, classification, vector_db_conn=None, content_saved=content_saved)
        except Exception as e:
            logger.error(f"Error saving to vector DB (non-critical): {e}")
            
        return True
    except Exception as e:
        logger.error(f"Error saving to Snowflake: {e}\n{traceback.format_exc()}")
        return False

def save_to_vector_db(domain: str, url: str, content: str, classification: Dict[str, Any], 
                     vector_db_conn=None, content_saved=False):
    """
    Save domain classification data to vector database.
    
    Args:
        domain: The domain being classified
        url: The URL of the website
        content: The content of the website
        classification: The classification result
        vector_db_conn: Optional vector DB connector instance
        content_saved: If True, skips content hash checks
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # If Pinecone or Anthropic aren't available, just skip silently
        try:
            from domain_classifier.storage.vector_db import PINECONE_AVAILABLE, ANTHROPIC_AVAILABLE
            if not PINECONE_AVAILABLE or not ANTHROPIC_AVAILABLE:
                logger.info(f"Pinecone or Anthropic not available, skipping vector storage for {domain}")
                return False
        except ImportError:
            logger.info(f"Vector DB module not available, skipping vector storage for {domain}")
            return False
            
        # If no connector is provided, import and create one
        if vector_db_conn is None:
            try:
                # Only create the connector once per process if possible
                import builtins
                if not hasattr(builtins, '_vector_db_connector'):
                    logger.info("Creating new VectorDBConnector instance...")
                    from domain_classifier.storage.vector_db import VectorDBConnector
                    builtins._vector_db_connector = VectorDBConnector()
                    logger.info(f"Created new VectorDBConnector instance, connected: {builtins._vector_db_connector.connected}")
                
                vector_db_conn = builtins._vector_db_connector
            except Exception as e:
                logger.error(f"Error creating VectorDBConnector: {e}")
                logger.error(traceback.format_exc())
                try:
                    # Try direct creation as fallback
                    from domain_classifier.storage.vector_db import VectorDBConnector
                    vector_db_conn = VectorDBConnector()
                except Exception:
                    logger.error("Failed to create vector DB connector even with fallback")
                    return False
                
        # Skip if not connected to vector DB
        if not getattr(vector_db_conn, 'connected', False):
            logger.info(f"Vector DB not connected, skipping vector storage for {domain}")
            return False
            
        # Calculate content hash first if we have content
        content_hash = None
        if content:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
        # Check if content has changed (using hash) before re-vectorizing
        if content and not content_saved:
            # Use a cache for content hashes if possible
            hash_cache = getattr(vector_db_conn, '_content_hash_cache', {})
            
            # Check if we have a cached hash for this domain
            if domain in hash_cache and hash_cache[domain] == content_hash:
                logger.info(f"Content unchanged for {domain} based on hash cache, skipping re-vectorization")
                return True
            
            # Cache miss, check if vector with this ID already exists and has the same content hash
            vector_id = vector_db_conn.generate_vector_id(domain)
            try:
                # Fetch existing vector metadata if it exists
                existing_vector = vector_db_conn.index.fetch(ids=[vector_id], namespace="domains")
                
                # Check if vector exists and content hash matches
                if vector_id in existing_vector.vectors:
                    existing_metadata = existing_vector.vectors[vector_id].metadata
                    if existing_metadata.get("content_hash") == content_hash:
                        # Cache the hash for future use
                        hash_cache[domain] = content_hash
                        setattr(vector_db_conn, '_content_hash_cache', hash_cache)
                        
                        logger.info(f"Content unchanged for {domain}, skipping re-vectorization")
                        return True
            except Exception as e:
                # If fetch fails, continue with vectorization
                logger.warning(f"Error checking existing vector: {e}")
            
        # Prepare metadata from classification
        logger.info(f"Preparing metadata for vector storage: {domain}")
        metadata = {
            "domain": domain,
            "url": url,
            "predicted_class": classification.get('predicted_class', 'Unknown'),
            "confidence_score": float(classification.get('max_confidence', 0.5)),
            "is_service_business": classification.get('is_service_business', None),
            "internal_it_potential": classification.get('internal_it_potential', 0),
            "detection_method": classification.get('detection_method', 'unknown'),
            "low_confidence": classification.get('low_confidence', False),
            "is_parked": classification.get('is_parked', False),
            "classification_date": classification.get('classification_date', '')
        }
        
        # Add content hash to metadata if available
        if content_hash:
            metadata["content_hash"] = content_hash
        
        # Add company description if available
        if 'company_description' in classification:
            metadata['company_description'] = classification['company_description']
            
        # Add LLM explanation if available (for better vector search)
        if 'llm_explanation' in classification:
            metadata['llm_explanation'] = classification['llm_explanation']
            
        # Store the vectorized data
        logger.info(f"Saving to vector DB for {domain}")
        success = vector_db_conn.upsert_domain_vector(
            domain=domain,
            content=content,
            metadata=metadata
        )
        
        # Cache the content hash if successful
        if success and content_hash:
            hash_cache = getattr(vector_db_conn, '_content_hash_cache', {})
            hash_cache[domain] = content_hash
            setattr(vector_db_conn, '_content_hash_cache', hash_cache)
            logger.info(f"✅ Successfully stored {domain} in vector database and cached hash")
        
        return success
    except Exception as e:
        logger.error(f"Error saving to vector DB: {e}")
        logger.error(traceback.format_exc())
        return False

def query_similar_domains(query_text: str, top_k: int = 5, filter=None, vector_db_conn=None):
    """
    Query for domains similar to the given text.
    
    Args:
        query_text: The text to find similar domains for
        top_k: The number of results to return
        filter: Optional filter criteria
        vector_db_conn: Optional vector DB connector instance
    
    Returns:
        list: List of similar domains with metadata
    """
    try:
        # If Pinecone or Anthropic aren't available, just return empty
        try:
            from domain_classifier.storage.vector_db import PINECONE_AVAILABLE, ANTHROPIC_AVAILABLE
            if not PINECONE_AVAILABLE or not ANTHROPIC_AVAILABLE:
                logger.info("Pinecone or Anthropic not available, cannot query similar domains")
                return []
        except ImportError:
            logger.info("Vector DB module not available, cannot query similar domains")
            return []
            
        # If no connector is provided, import and create one
        if vector_db_conn is None:
            try:
                # Try to reuse existing connector
                import builtins
                if hasattr(builtins, '_vector_db_connector'):
                    vector_db_conn = builtins._vector_db_connector
                else:
                    from domain_classifier.storage.vector_db import VectorDBConnector
                    vector_db_conn = VectorDBConnector()
                    builtins._vector_db_connector = vector_db_conn
            except Exception as e:
                logger.error(f"Error creating VectorDBConnector: {e}")
                logger.error(traceback.format_exc())
                return []
                
        # Skip if not connected to vector DB
        if not getattr(vector_db_conn, 'connected', False):
            logger.info("Vector DB not connected, cannot query similar domains")
            return []
            
        # Call the vector DB to get similar domains
        logger.info(f"Querying vector DB for domains similar to: {query_text[:50]}...")
        similar_domains = vector_db_conn.query_similar_domains(
            query_text=query_text,
            top_k=top_k,
            filter=filter
        )
        
        logger.info(f"Found {len(similar_domains)} similar domains")
        return similar_domains
    except Exception as e:
        logger.error(f"Error querying similar domains: {e}")
        logger.error(traceback.format_exc())
        return []
