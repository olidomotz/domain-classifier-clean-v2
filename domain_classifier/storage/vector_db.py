"""Pinecone vector database connector with robust retry logic for domain classification."""
import logging
import os
import json
import time
import traceback
import hashlib
import random
from typing import Dict, Any, List, Optional
import socket
import math
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Set up logging
logger = logging.getLogger(__name__)

# Global flags to track availability
PINECONE_AVAILABLE = False

try:
    # Import Pinecone with detailed error logging
    logger.info("Attempting to import pinecone-client...")
    import pinecone
    PINECONE_AVAILABLE = True
    logger.info(f"✅ Pinecone library successfully imported - version: {pinecone.__version__}")
except Exception as e:
    logger.error(f"❌ Error importing Pinecone: {str(e)}")
    logger.error(traceback.format_exc())
    PINECONE_AVAILABLE = False

class VectorDBConnector:
    def __init__(self,
                 api_key: str = None,
                 index_name: str = None,
                 environment: str = None):
        """
        Initialize the Pinecone vector database connector with robust retry handling.
        """
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "domain-embeddings")
        self.environment = environment or os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
        self.host_url = os.environ.get("PINECONE_HOST_URL", "")
        
        # Status indicators
        self.connected = False
        self.index = None
        
        # Track embedding metrics
        self.hash_embeddings_count = 0
        
        # Store version info
        self.pinecone_version = getattr(pinecone, '__version__', 'unknown') if PINECONE_AVAILABLE else 'not available'
        
        # Set up session with retry capabilities for direct API access (as a fallback)
        self.session = self._create_retry_session()
        
        # DETAILED DIAGNOSTICS - Log all initialization values
        logger.info("="*80)
        logger.info("VECTOR DB CONNECTOR INITIALIZATION")
        logger.info(f"Pinecone API Key available: {bool(self.api_key)}")
        logger.info(f"Pinecone Index Name: '{self.index_name}'")
        logger.info(f"Pinecone Environment: '{self.environment}'")
        logger.info(f"Pinecone Host URL: '{self.host_url}'")
        logger.info("="*80)
        
        # Initialize Pinecone connection only if API key is available
        if self.api_key and PINECONE_AVAILABLE:
            try:
                self._init_connection()
                # Run connection test for diagnostics
                self.test_connection()
            except Exception as e:
                logger.error(f"❌ Failed to initialize Pinecone connection: {e}")
                logger.error(traceback.format_exc())
                self.connected = False
        else:
            logger.warning("No Pinecone API key provided or Pinecone not available, vector storage disabled")
    
    def _create_retry_session(self):
        """Create a requests session with retry capability."""
        session = requests.Session()
        
        # Set up retries for connection issues, 
        # with exponential backoff and jitter
        retry_strategy = Retry(
            total=5,  # Maximum number of retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["HEAD", "GET", "POST"],  # Allow retries on these methods
            backoff_factor=1,  # Exponential backoff
            respect_retry_after_header=True,  # Respect Retry-After header
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
            
    def test_connection(self):
        """Test connection to Pinecone and print detailed diagnostics."""
        if not PINECONE_AVAILABLE:
            logger.warning("Cannot test Pinecone connection - library not available")
            return False
            
        logger.info("="*80)
        logger.info("PINECONE CONNECTION TEST")
        logger.info(f"API Key: {self.api_key[:5]}...")
        logger.info(f"Index Name: {self.index_name}")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Host URL: {self.host_url}")
        
        # DNS test for host
        if self.host_url:
            try:
                logger.info(f"Performing DNS test for host: {self.host_url}")
                ip_address = socket.gethostbyname(self.host_url)
                logger.info(f"DNS resolution success - IP: {ip_address}")
            except socket.gaierror as dns_error:
                logger.warning(f"DNS resolution failed: {dns_error}")
            except Exception as e:
                logger.error(f"Error during DNS test: {e}")
        
        # Attempt direct API call to test the connection
        if self.api_key and self.host_url:
            try:
                logger.info("Testing connection with direct API call")
                url = f"https://{self.host_url}/describe_index_stats"
                headers = {
                    "Api-Key": self.api_key,
                    "Content-Type": "application/json"
                }
                
                response = self.session.post(url, headers=headers, json={}, timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"✅ Direct API call successful: {response.text[:200]}")
                    
                    # Mark as connected if direct API succeeds
                    if not self.connected:
                        self.connected = True
                        logger.info("Setting connected=True based on direct API success")
                        
                    return True
                else:
                    logger.error(f"Direct API call failed: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error during direct API test: {e}")
        
        return self.connected

    def _init_connection(self):
        """Initialize connection to Pinecone using v1 API approach with fallback to direct API."""
        try:
            logger.info(f"Initializing Pinecone with api_key={self.api_key[:5]}...")
            
            # For v1.x of the client (older approach) - most compatible
            if self.host_url:
                logger.info(f"Using host-based connection: {self.host_url}")
                pinecone.init(api_key=self.api_key, host=self.host_url)
            else:
                logger.info(f"Using environment-based connection: {self.environment}")
                pinecone.init(api_key=self.api_key, environment=self.environment)
                
            # Connect to the index
            self.index = pinecone.Index(self.index_name)
            
            # Mark as initially connected - we'll verify with operations
            self.connected = True
            logger.info(f"✅ Connected to Pinecone index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"❌ Error in _init_connection: {e}")
            logger.error(traceback.format_exc())
            self.connected = False
            
            # Try to set up direct API access
            logger.info("Setting up for direct API access instead")
            self.connected = True  # Mark as connected, we'll use direct API calls

    def create_embedding(self, text: str) -> List[float]:
        """
        Create a hash-based embedding vector.
        
        Args:
            text: The text to embed
            
        Returns:
            list: The embedding vector
        """
        # Handle None text case
        if text is None:
            logger.warning("Cannot create embedding for None text")
            raise ValueError("Cannot create embedding for None text")
        
        logger.info(f"Creating hash-based embedding for text: '{text[:30]}...' (length: {len(text)})")
        
        # Create a hash-based embedding with 227 dimensions (to match existing index)
        dimensions = 227
        
        # Normalize text
        text = text.lower().strip()
        if not text:
            text = "empty_text"
            
        # Initialize an empty embedding with the right dimensions
        embedding = [0.0] * dimensions
        
        # Generate unique values by combining the text with position indices
        for i in range(dimensions):
            # Create a unique hash for each position
            position_text = f"{text}_{i}"
            hash_value = hashlib.sha256(position_text.encode()).digest()
            
            # Convert the first 4 bytes of hash to a float between -1 and 1
            # Using 4 bytes (32 bits) to get good floating point precision
            byte_value = int.from_bytes(hash_value[:4], byteorder='big')
            normalized = (byte_value / (2**32 - 1)) * 2 - 1  # Scale to [-1, 1]
            
            embedding[i] = normalized
            
        # Normalize the embedding to unit length (important for cosine similarity)
        magnitude = math.sqrt(sum(x**2 for x in embedding))
        if magnitude > 0:
            embedding = [x/magnitude for x in embedding]
            
        self.hash_embeddings_count += 1
        logger.info(f"✅ Created hash-based embedding (dimensions: {len(embedding)})")
        
        return embedding

    def generate_vector_id(self, domain: str, content_type: str = "domain") -> str:
        """
        Generate a unique ID for a vector.
        
        Args:
            domain: The domain name
            content_type: The type of content
            
        Returns:
            str: The unique ID
        """
        # Create a unique ID based on domain and content type
        import hashlib
        unique_str = f"{domain}_{content_type}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def _direct_api_upsert(self, vectors, namespace="domains") -> bool:
        """
        Upsert vectors using direct API call to bypass client issues.
        
        Args:
            vectors: List of tuples (id, vector, metadata)
            namespace: Namespace for the vectors
            
        Returns:
            bool: Success status
        """
        if not self.api_key or not self.host_url:
            return False
            
        # Format vectors for the API
        formatted_vectors = []
        for vec_id, vec, metadata in vectors:
            formatted_vectors.append({
                "id": vec_id,
                "values": vec,
                "metadata": metadata
            })
            
        # Build the API request
        url = f"https://{self.host_url}/vectors/upsert"
        headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "vectors": formatted_vectors,
            "namespace": namespace
        }
        
        # Try making the request with retries
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                logger.info(f"Direct API upsert attempt {attempt+1}/{max_attempts}")
                
                # Add jitter to avoid potential rate limiting issues
                if attempt > 0:
                    jitter = random.uniform(0.1, 0.5)
                    sleep_time = (2 ** attempt) + jitter  # Exponential backoff with jitter
                    logger.info(f"Backing off for {sleep_time:.2f} seconds before retry")
                    time.sleep(sleep_time)
                
                response = self.session.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    logger.info(f"✅ Direct API upsert successful")
                    return True
                else:
                    logger.warning(f"Direct API upsert failed: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error in direct API upsert (attempt {attempt+1}): {e}")
                
        return False

    def _direct_api_query(self, vector, top_k=5, filter=None, namespace="domains") -> Dict:
        """
        Query for similar vectors using direct API call.
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            filter: Optional filter criteria
            namespace: Namespace to query
            
        Returns:
            dict: Query results
        """
        if not self.api_key or not self.host_url:
            return {"matches": []}
            
        # Build the API request
        url = f"https://{self.host_url}/query"
        headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "vector": vector,
            "top_k": top_k,
            "namespace": namespace,
            "include_metadata": True
        }
        
        # Add filter if provided
        if filter:
            payload["filter"] = filter
            
        # Try making the request with retries
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                logger.info(f"Direct API query attempt {attempt+1}/{max_attempts}")
                
                # Add jitter to avoid potential rate limiting issues
                if attempt > 0:
                    jitter = random.uniform(0.1, 0.5)
                    sleep_time = (2 ** attempt) + jitter  # Exponential backoff with jitter
                    logger.info(f"Backing off for {sleep_time:.2f} seconds before retry")
                    time.sleep(sleep_time)
                
                response = self.session.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    logger.info(f"✅ Direct API query successful")
                    return response.json()
                else:
                    logger.warning(f"Direct API query failed: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error in direct API query (attempt {attempt+1}): {e}")
                
        return {"matches": []}

    def upsert_domain_vector(self, domain: str, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Upsert a domain vector to Pinecone with direct API fallback.
        
        Args:
            domain: The domain name
            content: The website content
            metadata: Additional metadata to store
            
        Returns:
            bool: Success status
        """
        if not self.connected:
            logger.warning("Cannot store vector: No active Pinecone connection")
            return False
        
        try:
            # Check if content is None - can't create an embedding from None
            if content is None:
                logger.warning(f"Cannot create embedding for {domain} - content is None")
                return False
            
            # Create embedding for domain
            try:
                vector = self.create_embedding(content)  # Use hash-based embedding
                
                # Validate vector dimensions
                if len(vector) != 227:
                    logger.error(f"Vector dimension mismatch: got {len(vector)}, expected 227")
                    return False
                    
                # Check for NaN or infinite values
                if any(not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) for v in vector):
                    logger.error(f"Invalid vector values found for {domain} - contains NaN or infinite values")
                    return False
                
                # Add detailed logging of vector characteristics
                logger.info(f"Vector stats for {domain}: min={min(vector):.4f}, max={max(vector):.4f}, avg={sum(vector)/len(vector):.4f}")
            except Exception as embed_error:
                logger.error(f"Error creating embedding for {domain}: {embed_error}")
                return False
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Ensure metadata contains domain
            metadata["domain"] = domain
            
            # Sanitize metadata (ensure all values are strings or numbers)
            sanitized_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    sanitized_metadata[key] = value
                elif isinstance(value, dict):
                    # Convert dict to JSON string
                    try:
                        sanitized_metadata[key] = json.dumps(value)
                    except:
                        # If conversion fails, use str()
                        sanitized_metadata[key] = str(value)
                elif value is None:
                    sanitized_metadata[key] = ""
                else:
                    sanitized_metadata[key] = str(value)
            
            # Create a unique ID for the vector
            vector_id = f"domain_{domain.replace('.', '_')}"
            
            # Try standard approach first
            if self.index:
                try:
                    logger.info(f"Attempting to upsert vector for {domain} using standard approach")
                    
                    # Standardize on the 'domains' namespace to match existing data
                    namespace = "domains"
                    
                    # Use the appropriate format based on the Pinecone version
                    self.index.upsert(
                        vectors=[(vector_id, vector, sanitized_metadata)], 
                        namespace=namespace
                    )
                    
                    logger.info(f"✅ Successfully stored vector for {domain} in namespace '{namespace}'")
                    return True
                except Exception as e:
                    logger.warning(f"Standard upsert failed: {e}. Trying direct API approach.")
            
            # If standard approach fails, try direct API call
            direct_result = self._direct_api_upsert(
                vectors=[(vector_id, vector, sanitized_metadata)],
                namespace="domains"
            )
            
            if direct_result:
                logger.info(f"✅ Successfully stored vector for {domain} using direct API call")
                return True
            else:
                logger.error(f"All upsert attempts failed for {domain}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store vector for {domain}: {e}")
            logger.error(traceback.format_exc())
            return False

    def query_similar_domains(self, query_text: str, top_k: int = 5, filter=None,
                             low_confidence_threshold: float = 0.3,
                             unreliable_threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        Query for similar domains using query text.
        
        Args:
            query_text: The text to query with
            top_k: Number of results to return
            filter: Optional filter criteria
            low_confidence_threshold: Threshold for marking results low confidence
            unreliable_threshold: Threshold for marking results unreliable
            
        Returns:
            list: List of similar domains
        """
        if not self.connected:
            logger.warning("Cannot query similar domains: No active Pinecone connection")
            return []
        
        try:
            # Create embedding for query
            try:
                vector = self.create_embedding(query_text)
            except Exception as embed_error:
                logger.error(f"Error creating query embedding: {embed_error}")
                return []
            
            # Query Pinecone - try standard approach first
            result = None
            
            if self.index:
                try:
                    logger.info(f"Querying for similar domains with standard approach")
                    
                    # Set up query parameters
                    namespace = "domains"  # Use the established namespace
                    
                    # Set up query parameters based on filter
                    if filter:
                        result = self.index.query(
                            vector=vector,
                            top_k=top_k,
                            include_metadata=True,
                            filter=filter,
                            namespace=namespace
                        )
                    else:
                        result = self.index.query(
                            vector=vector,
                            top_k=top_k,
                            include_metadata=True,
                            namespace=namespace
                        )
                    
                    logger.info(f"✅ Successfully queried similar domains from namespace '{namespace}'")
                    
                except Exception as e:
                    logger.warning(f"Standard query approach failed: {e}. Trying direct API.")
            
            # If standard approach fails, try direct API call
            if result is None:
                result = self._direct_api_query(
                    vector=vector,
                    top_k=top_k,
                    filter=filter,
                    namespace="domains"
                )
                
                if result and result.get("matches"):
                    logger.info(f"✅ Successfully queried similar domains using direct API")
                else:
                    logger.error("All query attempts failed")
                    return []
            
            # Process results
            similar_domains = []
            
            # Handle different response formats
            if hasattr(result, 'matches'):  # New API
                matches = result.matches
            elif isinstance(result, dict) and 'matches' in result:  # Old API or direct API
                matches = result['matches']
            else:
                logger.warning(f"Unexpected query result format: {type(result)}")
                return []
            
            # Process matches
            for match in matches:
                # Extract score and metadata
                score = 0
                metadata = {}
                
                if hasattr(match, 'score') and hasattr(match, 'metadata'):  # New API
                    score = match.score
                    metadata = match.metadata
                else:  # Old API or direct API
                    score = match.get('score', 0)
                    metadata = match.get('metadata', {})
                
                # Extract information
                domain = metadata.get("domain", "unknown")
                company_type = metadata.get("predicted_class", "Unknown")
                description = metadata.get("company_description", "")
                
                # Add confidence flag
                confidence_flag = None
                if score < unreliable_threshold:
                    confidence_flag = "vector_match_unreliable"
                elif score < low_confidence_threshold:
                    confidence_flag = "low_confidence_vector"
                
                # Create result item
                result_item = {
                    "domain": domain,
                    "score": score,
                    "company_type": company_type,
                    "description": description[:200] + "..." if len(description) > 200 else description,
                    "metadata": metadata,
                    "confidence_flag": confidence_flag
                }
                
                similar_domains.append(result_item)
            
            return similar_domains
            
        except Exception as e:
            logger.error(f"Error querying similar domains: {e}")
            logger.error(traceback.format_exc())
            return []
            
    def diagnose_anthropic_embedding(self):
        """Diagnose embedding capabilities"""
        logger.info("="*80)
        logger.info("DIAGNOSING EMBEDDING CAPABILITY")
        
        # Test the connection directly using the API
        if self.api_key and self.host_url:
            try:
                logger.info("Testing direct API connection")
                url = f"https://{self.host_url}/describe_index_stats"
                headers = {
                    "Api-Key": self.api_key,
                    "Content-Type": "application/json"
                }
                
                response = self.session.post(url, headers=headers, json={}, timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"✅ Direct API connection successful: {response.text[:200]}")
                    return True
                else:
                    logger.error(f"Direct API connection failed: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error testing direct API connection: {e}")
                
        # Test creating a hash-based embedding
        try:
            test_text = "This is a test of the embedding system."
            embedding = self.create_embedding(test_text)
            
            if embedding and len(embedding) == 227:
                logger.info("✅ Successfully created hash-based embedding")
                logger.info(f"Embedding dimensions: {len(embedding)}")
                logger.info(f"Embedding sample: {embedding[:5]}")
                return True
            else:
                logger.warning(f"Created embedding with unexpected dimensions: {len(embedding) if embedding else 'None'}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing embedding: {e}")
            logger.error(traceback.format_exc())
            return False
