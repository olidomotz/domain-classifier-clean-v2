"""Snowflake connector for domain classification data storage."""
import os
import json
import logging
import traceback
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_der_private_key
import snowflake.connector
from snowflake.connector.errors import ProgrammingError, DatabaseError
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

# Set up logging
logger = logging.getLogger(__name__)

# Cache for domain content to reduce database queries
DOMAIN_CONTENT_CACHE = {}
# Cache for classification results
CLASSIFICATION_CACHE = {}
# Cache expiry in seconds (5 minutes)
CACHE_EXPIRY = 300

def safe_params(params):
    """
    Ensure all parameters are safe for Snowflake by converting any dictionaries 
    to JSON strings.
    
    Args:
        params: The parameters tuple/list to process
        
    Returns:
        A new tuple with all dictionaries converted to JSON strings
    """
    if not isinstance(params, (list, tuple)):
        return params
        
    safe_params = []
    for param in params:
        if isinstance(param, dict):
            safe_params.append(json.dumps(param))
        elif isinstance(param, list):
            safe_params.append(json.dumps(param))
        else:
            safe_params.append(param)
            
    return tuple(safe_params)

class SnowflakeConnector:
    def __init__(self):
        """Initialize Snowflake connection with environment variables."""
        self.connected = False
        
        # Initialize instance-level caches
        self._domain_content_cache = {}
        self._classification_cache = {}
        self._recent_domain_saves = {}
        
        # Get connection parameters from environment variables
        self.account = os.environ.get('SNOWFLAKE_ACCOUNT')
        self.user = os.environ.get('SNOWFLAKE_USER')
        self.database = os.environ.get('SNOWFLAKE_DATABASE')
        self.schema = os.environ.get('SNOWFLAKE_SCHEMA')
        self.warehouse = os.environ.get('SNOWFLAKE_WAREHOUSE')
        self.authenticator = os.environ.get('SNOWFLAKE_AUTHENTICATOR')
        self.private_key_path = os.environ.get('SNOWFLAKE_PRIVATE_KEY_PATH')
        
        # Create RSA key file if it doesn't exist
        if not os.path.exists(self.private_key_path) and 'SNOWFLAKE_KEY_BASE64' in os.environ:
            try:
                dir_path = os.path.dirname(self.private_key_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    
                with open(self.private_key_path, 'wb') as key_file:
                    import base64
                    key_data = base64.b64decode(os.environ.get('SNOWFLAKE_KEY_BASE64'))
                    key_file.write(key_data)
                    
                os.chmod(self.private_key_path, 0o600)
                logger.info(f"Created RSA key file at {self.private_key_path}")
            except Exception as e:
                logger.error(f"Failed to create RSA key file: {e}")
        
        # Check for required credentials
        if not self.account or not self.user or not self.private_key_path:
            logger.warning("Missing Snowflake credentials. Using fallback mode.")
            return
        
        try:
            self._init_connection()
        except Exception as e:
            logger.error(f"Failed to initialize Snowflake connection: {e}")
            self.connected = False
    
    def _init_connection(self):
        """Initialize the Snowflake connection."""
        try:
            # Check if the RSA key exists
            if os.path.exists(self.private_key_path):
                logger.info(f"Found RSA key at {self.private_key_path}")
                
                # Test connection
                conn = self.get_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT current_version()")
                    version = cursor.fetchone()[0]
                    logger.info(f"Connected to Snowflake. Version: {version}")
                    
                    # Skip table creation as tables already exist
                    cursor.close()
                    conn.close()
                    self.connected = True
                else:
                    logger.error("Could not establish Snowflake connection")
                    self.connected = False
            else:
                logger.warning(f"RSA key not found at {self.private_key_path}")
                self.connected = False
        except Exception as e:
            logger.error(f"Error connecting to Snowflake: {e}")
            self.connected = False
            raise
    
    def load_private_key(self, path):
        """Load private key from path."""
        try:
            with open(path, "rb") as key_file:
                return key_file.read()
        except Exception as e:
            logger.error(f"Error loading private key: {e}")
            return None
    
    def get_connection(self):
        """Get a new Snowflake connection."""
        if not os.path.exists(self.private_key_path):
            logger.warning(f"Private key not found at {self.private_key_path}")
            return None
            
        try:
            private_key = self.load_private_key(self.private_key_path)
            if not private_key:
                return None
                
            return snowflake.connector.connect(
                user=self.user,
                account=self.account,
                private_key=private_key,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
                authenticator=self.authenticator,
                session_parameters={'QUERY_TAG': 'WebCrawlerBot'},
                # Add these parameters to fix certificate issues
                insecure_mode=True,            # Bypass certificate validation
                ocsp_fail_open=True,           # Continue on OCSP check failures
                login_timeout=10,              # Shorter login timeout
                validate_default_parameters=False  # Skip parameter validation for speed
            )
        except Exception as e:
            logger.error(f"Error getting Snowflake connection: {e}")
            return None
    
    # NEW METHOD: Update BULK_PROCESS_ID for existing classifications
    def update_classification_bulk_id(self, domain: str, bulk_process_id: str) -> Tuple[bool, Optional[str]]:
        """
        Update an existing domain classification's BULK_PROCESS_ID.
        
        Args:
            domain: The domain to update
            bulk_process_id: The bulk process ID to set
            
        Returns:
            tuple: (success, error_message)
        """
        if not self.connected:
            logger.info(f"Fallback: Not updating BULK_PROCESS_ID for {domain} - not connected")
            return True, None
            
        conn = self.get_connection()
        if not conn:
            return False, "Failed to connect to Snowflake"
            
        try:
            cursor = conn.cursor()
            # Set shorter query timeout
            cursor.execute("ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS=20")
            
            # Update the most recent classification for this domain
            query = """
                UPDATE DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.DOMAIN_CLASSIFICATION
                SET BULK_PROCESS_ID = %s
                WHERE DOMAIN = %s
                AND CLASSIFICATION_DATE = (
                    SELECT MAX(CLASSIFICATION_DATE) 
                    FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.DOMAIN_CLASSIFICATION
                    WHERE DOMAIN = %s
                )
            """
            
            cursor.execute(query, (bulk_process_id, domain, domain))
            
            # Also invalidate the cache for this domain
            if domain in self._classification_cache:
                del self._classification_cache[domain]
                
            if domain in CLASSIFICATION_CACHE:
                del CLASSIFICATION_CACHE[domain]
            
            conn.commit()
            logger.info(f"Updated BULK_PROCESS_ID for {domain} to {bulk_process_id}")
            
            return True, None
        except Exception as e:
            error_msg = traceback.format_exc()
            if conn:
                conn.rollback()
            logger.error(f"Error updating BULK_PROCESS_ID: {error_msg}")
            return False, error_msg
        finally:
            if conn:
                conn.close()
    
    def check_existing_classification(self, domain):
        """Check if a domain already has a classification in Snowflake."""
        # First check instance cache
        if domain in self._classification_cache:
            cache_entry = self._classification_cache[domain]
            cache_time, result = cache_entry
            # Check if cache is still valid (less than 5 minutes old)
            if (datetime.now() - cache_time).total_seconds() < CACHE_EXPIRY:
                logger.info(f"Using cached classification for {domain}")
                return result
                
        # Then check global cache
        if domain in CLASSIFICATION_CACHE:
            cache_entry = CLASSIFICATION_CACHE[domain]
            cache_time, result = cache_entry
            # Check if cache is still valid (less than 5 minutes old)
            if (datetime.now() - cache_time).total_seconds() < CACHE_EXPIRY:
                logger.info(f"Using global cached classification for {domain}")
                # Update instance cache
                self._classification_cache[domain] = cache_entry
                return result
        
        # Cache miss or expired, need to query Snowflake    
        if not self.connected:
            logger.info(f"Fallback: No existing classification for {domain}")
            return None
            
        conn = self.get_connection()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor()
            # Look for classifications not older than 30 days and include crawler and classifier type fields
            query = """
                SELECT
                    DOMAIN,
                    COMPANY_TYPE,
                    CONFIDENCE_SCORE,
                    ALL_SCORES,
                    LOW_CONFIDENCE,
                    DETECTION_METHOD,
                    MODEL_METADATA,
                    CLASSIFICATION_DATE,
                    LLM_EXPLANATION,
                    APOLLO_COMPANY_DATA,
                    CRAWLER_TYPE,
                    CLASSIFIER_TYPE,
                    BULK_PROCESS_ID
                FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.DOMAIN_CLASSIFICATION
                WHERE DOMAIN = %s
                AND CLASSIFICATION_DATE > DATEADD(day, -30, CURRENT_TIMESTAMP())
                ORDER BY CLASSIFICATION_DATE DESC
                LIMIT 1
            """
            cursor.execute(query, (domain,))
            
            result = cursor.fetchone()
            if result:
                # Get column names from cursor description
                column_names = [col[0] for col in cursor.description]
                existing_record = dict(zip(column_names, result))
                logger.info(f"Found existing classification for {domain}: {existing_record['COMPANY_TYPE']}")
                
                # Cache the result
                cache_entry = (datetime.now(), existing_record)
                self._classification_cache[domain] = cache_entry
                CLASSIFICATION_CACHE[domain] = cache_entry
                
                return existing_record
            
            logger.info(f"No existing classification found for {domain}")
            return None
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"Error checking existing classification: {error_msg}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_domain_content(self, domain):
        """Get the most recent content for a domain from Snowflake."""
        # Check instance cache first
        if domain in self._domain_content_cache:
            cache_entry = self._domain_content_cache[domain]
            cache_time, content = cache_entry
            # Check if cache is still valid (less than 5 minutes old)
            if (datetime.now() - cache_time).total_seconds() < CACHE_EXPIRY:
                logger.info(f"Using cached content for {domain}")
                return content
                
        # Then check global cache
        if domain in DOMAIN_CONTENT_CACHE:
            cache_entry = DOMAIN_CONTENT_CACHE[domain]
            cache_time, content = cache_entry
            # Check if cache is still valid (less than 5 minutes old)
            if (datetime.now() - cache_time).total_seconds() < CACHE_EXPIRY:
                logger.info(f"Using global cached content for {domain}")
                # Update instance cache
                self._domain_content_cache[domain] = cache_entry
                return content
        
        # Cache miss or expired, need to query Snowflake
        if not self.connected:
            logger.info(f"Fallback: No content for {domain}")
            return None
            
        conn = self.get_connection()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor()
            # Set shorter query timeout
            cursor.execute("ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS=15")
            
            query = """
                SELECT text_content
                FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.DOMAIN_CONTENT
                WHERE domain = %s
                ORDER BY crawl_date DESC
                LIMIT 1
            """
            cursor.execute(query, (domain,))
            
            result = cursor.fetchone()
            if result and result[0]:
                logger.info(f"Found existing content for {domain}")
                
                # Cache the content
                content = result[0]
                cache_entry = (datetime.now(), content)
                self._domain_content_cache[domain] = cache_entry
                DOMAIN_CONTENT_CACHE[domain] = cache_entry
                
                return content
            
            logger.info(f"No existing content found for {domain}")
            return None
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"Error retrieving domain content: {error_msg}")
            return None
        finally:
            if conn:
                conn.close()
    
    def save_domain_content(self, domain, url, content):
        """Save domain content to Snowflake."""
        if not self.connected:
            logger.info(f"Fallback: Not saving domain content for {domain}")
            return True, None
            
        conn = self.get_connection()
        if not conn:
            return False, "Failed to connect to Snowflake"
            
        try:
            cursor = conn.cursor()
            
            # Check if content exists and is not empty
            if not content or len(content.strip()) < 100:
                logger.warning(f"Content for {domain} is too short or empty, not saving")
                return False, "Content too short or empty"
            
            # Insert new record with a timeout
            cursor.execute("ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS=20")
            cursor.execute("""
                INSERT INTO DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.DOMAIN_CONTENT (domain, url, text_content, crawl_date)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP())
            """, (domain, url, content))
            
            conn.commit()
            logger.info(f"Saved domain content for {domain}")
            
            # Update cache
            cache_entry = (datetime.now(), content)
            self._domain_content_cache[domain] = cache_entry
            DOMAIN_CONTENT_CACHE[domain] = cache_entry
            
            # Update hash cache
            import hashlib
            content_hash = hashlib.md5(content.encode()).hexdigest()
            self._recent_domain_saves[domain] = content_hash
            
            return True, None
        except Exception as e:
            error_msg = traceback.format_exc()
            if conn:
                conn.rollback()
            logger.error(f"Error saving domain content: {error_msg}")
            return False, error_msg
        finally:
            if conn:
                conn.close()
    
    def save_classification(self, domain, company_type=None, confidence_score=0, all_scores=None, 
                           model_metadata=None, low_confidence=False, detection_method="auto", 
                           llm_explanation=None, apollo_company_data=None, crawler_type=None, 
                           classifier_type=None, bulk_process_id=None):
        """Save domain classification to Snowflake with explanation and Apollo data."""
        if not self.connected:
            logger.info(f"Fallback: Not saving classification for {domain}")
            return True, None
            
        conn = self.get_connection()
        if not conn:
            return False, "Failed to connect to Snowflake"
            
        try:
            cursor = conn.cursor()
            # Set shorter query timeout
            cursor.execute("ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS=20")
            
            # Ensure explanation isn't too long
            if llm_explanation and len(llm_explanation) > 5000:
                # Truncate at a sentence boundary
                last_period = llm_explanation[:4900].rfind('.')
                if last_period > 0:
                    llm_explanation = llm_explanation[:last_period+1]
                else:
                    llm_explanation = llm_explanation[:4900] + "..."
            
            # Convert all complex parameters to strings - this is critical
            all_scores_str = json.dumps(all_scores) if isinstance(all_scores, (dict, list)) else (all_scores or '{}')
            model_metadata_str = json.dumps(model_metadata) if isinstance(model_metadata, (dict, list)) else (model_metadata or '{}')
            
            # Log the types for debugging
            logger.info(f"Parameter types before SQL: all_scores={type(all_scores_str)}, model_metadata={type(model_metadata_str)}")
            
            # Basic fields that are always included
            basic_query = """
                INSERT INTO DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.DOMAIN_CLASSIFICATION 
                (domain, company_type, confidence_score, all_scores, model_metadata, 
                low_confidence, detection_method, classification_date, llm_explanation,
                crawler_type, classifier_type, bulk_process_id)
                VALUES (%s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), %s, %s, CURRENT_TIMESTAMP(), %s, %s, %s, %s)
            """
            
            # Prepare parameters as a tuple, ensuring no dictionaries
            basic_params = (
                domain, 
                company_type, 
                confidence_score, 
                all_scores_str,
                model_metadata_str,
                low_confidence, 
                detection_method,
                llm_explanation,
                crawler_type,
                classifier_type,
                bulk_process_id
            )
            
            # Execute the basic insert
            cursor.execute(basic_query, basic_params)
            
            # If we have Apollo data, update the record with separate statements
            if apollo_company_data:
                # Convert to JSON string if it's a dict
                if isinstance(apollo_company_data, (dict, list)):
                    apollo_company_json = json.dumps(apollo_company_data)
                else:
                    apollo_company_json = apollo_company_data
                
                # Run an UPDATE statement instead of trying to include in the INSERT
                company_update = """
                    UPDATE DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.DOMAIN_CLASSIFICATION
                    SET apollo_company_data = PARSE_JSON(%s)
                    WHERE domain = %s AND classification_date = (
                        SELECT MAX(classification_date) 
                        FROM DOMOTZ_TESTING_SOURCE.EXTERNAL_PUSH.DOMAIN_CLASSIFICATION
                        WHERE domain = %s
                    )
                """
                cursor.execute(company_update, (apollo_company_json, domain, domain))
            
            conn.commit()
            logger.info(f"Saved classification for {domain}: {company_type}")
            
            # Update the cache with new data
            try:
                # Create a record in the same format as the SELECT query
                new_record = {
                    'DOMAIN': domain,
                    'COMPANY_TYPE': company_type,
                    'CONFIDENCE_SCORE': confidence_score,
                    'ALL_SCORES': all_scores_str,
                    'LOW_CONFIDENCE': low_confidence,
                    'DETECTION_METHOD': detection_method,
                    'MODEL_METADATA': model_metadata_str,
                    'CLASSIFICATION_DATE': datetime.now().isoformat(),
                    'LLM_EXPLANATION': llm_explanation,
                    'APOLLO_COMPANY_DATA': apollo_company_json if 'apollo_company_json' in locals() else None,
                    'CRAWLER_TYPE': crawler_type,
                    'CLASSIFIER_TYPE': classifier_type,
                    'BULK_PROCESS_ID': bulk_process_id
                }
                
                # Update both caches
                cache_entry = (datetime.now(), new_record)
                self._classification_cache[domain] = cache_entry
                CLASSIFICATION_CACHE[domain] = cache_entry
            except Exception as cache_error:
                logger.warning(f"Error updating classification cache: {cache_error}")
                
            return True, None
        except Exception as e:
            error_msg = traceback.format_exc()
            if conn:
                conn.rollback()
            logger.error(f"Error saving classification: {error_msg}")
            return False, error_msg
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
                
    def execute_safe_query(self, query, params):
        """
        Execute a query with parameters safely, ensuring all complex types are properly handled.
        
        Args:
            query: SQL query to execute
            params: Parameters to bind to query
            
        Returns:
            True if successful, False otherwise
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            if not conn:
                logger.error("Failed to get connection for safe query execution")
                return False
                
            cursor = conn.cursor()
            cursor.execute("ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS=20")
            
            # Convert any dictionaries to JSON strings
            safe_parameter_list = []
            for param in params:
                if isinstance(param, (dict, list)):
                    safe_parameter_list.append(json.dumps(param))
                else:
                    safe_parameter_list.append(param)
                    
            # Log the types for debugging
            param_types = [f"{i}: {type(p).__name__}" for i, p in enumerate(safe_parameter_list)]
            logger.info(f"Safe parameter types: {param_types}")
            
            cursor.execute(query, tuple(safe_parameter_list))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error executing safe query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {params}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
