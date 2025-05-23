"""Complete Apollo connector with all methods, including search_person."""

import requests
import logging
import os
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Import from centralized JSON parser module
from domain_classifier.utils.json_parser import parse_and_validate_json, ensure_dict

# Set up logging
logger = logging.getLogger(__name__)

class ApolloConnector:
    def __init__(self, api_key: str = None):
        """
        Initialize the Apollo connector with API key and rate limiting.
        
        Args:
            api_key: The API key for Apollo.io
        """
        self.api_key = api_key or os.environ.get("APOLLO_API_KEY")
        if not self.api_key:
            logger.warning("No Apollo API key provided. Enrichment will not be available.")
            
        self.base_url = "https://api.apollo.io/v1"
        
        # Rate limiting settings
        self.rate_limit = 60  # Maximum calls per hour (conservative for most plans)
        self.calls_this_hour = 0
        self.hour_start = datetime.now()
        self.retry_delay = 5  # Seconds to wait between retries
        
        # Optional: Add a shared in-memory cache
        self._domain_cache = {}  # domain -> (timestamp, data)
        self._cache_expiry = 7 * 24 * 60 * 60  # 7 days in seconds
    
    def enrich_company(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Enrich a company profile using Apollo.io with rate limiting and caching.
        
        Args:
            domain: The domain to enrich
            
        Returns:
            dict: The enriched company data or None if failed
        """
        if not self.api_key:
            logger.warning(f"Cannot enrich {domain}: No Apollo API key")
            return None
        
        # Check cache first
        if domain in self._domain_cache:
            timestamp, data = self._domain_cache[domain]
            age = (datetime.now() - timestamp).total_seconds()
            
            if age < self._cache_expiry:
                logger.info(f"Using cached Apollo data for {domain} ({int(age/3600)} hours old)")
                return data
            else:
                logger.info(f"Cached Apollo data for {domain} expired, refreshing")
        
        # Check rate limits before making API call
        self._check_rate_limit()
        
        try:
            # Apollo's organizations/enrich endpoint
            endpoint = f"{self.base_url}/organizations/enrich"
            
            # Use the X-Api-Key header as required by Apollo
            headers = {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "X-Api-Key": self.api_key
            }
            
            payload = {
                "domain": domain
            }
            
            logger.info(f"Attempting to enrich data for domain: {domain}")
            
            # Record this API call for rate limiting
            self.calls_this_hour += 1
            
            # Try up to 3 times with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(endpoint, json=payload, headers=headers, timeout=15)
                    
                    # Check if we hit rate limits (429 Too Many Requests)
                    if response.status_code == 429:
                        logger.warning(f"Apollo API rate limit hit for {domain}. Attempt {attempt+1}/{max_retries}")
                        retry_after = int(response.headers.get('Retry-After', self.retry_delay * (2 ** attempt)))
                        time.sleep(retry_after)
                        continue
                    
                    # Process successful response
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get('organization'):
                            logger.info(f"Successfully enriched {domain} with Apollo data")
                            
                            company_data = self._format_company_data(data['organization'])
                            
                            # Cache the result
                            self._domain_cache[domain] = (datetime.now(), company_data)
                            
                            return company_data
                        else:
                            logger.warning(f"Apollo API returned 200 but no organization data for {domain}")
                            logger.debug(f"Apollo API response: {json.dumps(data)[:500]}...")
                            
                            # Cache the empty result with a shorter expiry (1 day)
                            empty_result = {"name": domain, "no_data": True}
                            self._domain_cache[domain] = (datetime.now() - timedelta(days=6), empty_result)
                            
                            # Exit the retry loop on 200 response
                            break
                    else:
                        # Enhanced error logging
                        try:
                            error_details = response.json()
                            logger.error(f"Apollo API error ({response.status_code}) for {domain}: {error_details}")
                        except:
                            logger.error(f"Apollo API error ({response.status_code}) for {domain}: {response.text[:500]}")
                        
                        # Don't retry certain error codes
                        if response.status_code in [400, 401, 403]:
                            break
                        
                        # For other errors, retry with backoff
                        retry_delay = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Retrying Apollo API call in {retry_delay} seconds. Attempt {attempt+1}/{max_retries}")
                        time.sleep(retry_delay)
                
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout connecting to Apollo API for {domain} (attempt {attempt+1})")
                    retry_delay = self.retry_delay * (2 ** attempt)
                    time.sleep(retry_delay)
                
                except requests.exceptions.ConnectionError:
                    logger.warning(f"Connection error with Apollo API for {domain} (attempt {attempt+1})")
                    retry_delay = self.retry_delay * (2 ** attempt)
                    time.sleep(retry_delay)
                    
                except Exception as e:
                    logger.error(f"Error making Apollo API call for {domain}: {e}")
                    retry_delay = self.retry_delay * (2 ** attempt)
                    time.sleep(retry_delay)
            
            # If we get here, all attempts failed or returned no data
            return None
            
        except Exception as e:
            logger.error(f"Error enriching company with Apollo: {e}")
            return None
    
    def _check_rate_limit(self):
        """Check and enforce rate limits."""
        now = datetime.now()
        
        # Reset counter if a new hour has started
        if now - self.hour_start > timedelta(hours=1):
            logger.info(f"Resetting Apollo API call counter: {self.calls_this_hour} calls in the last hour")
            self.calls_this_hour = 0
            self.hour_start = now
            return
        
        # If we're approaching the limit, sleep until the next hour
        if self.calls_this_hour >= self.rate_limit:
            seconds_until_reset = 3600 - (now - self.hour_start).seconds
            logger.warning(f"Apollo API rate limit reached ({self.calls_this_hour} calls). Sleeping for {seconds_until_reset} seconds")
            
            # Sleep in smaller increments with logging
            sleep_chunk = 30  # Sleep in 30-second chunks
            remaining = seconds_until_reset
            while remaining > 0:
                sleep_time = min(sleep_chunk, remaining)
                time.sleep(sleep_time)
                remaining -= sleep_time
                if remaining > 0:
                    logger.info(f"Rate limit sleep: {remaining} seconds remaining until reset")
            
            # Reset counters after sleep
            self.calls_this_hour = 0
            self.hour_start = datetime.now()
            logger.info("Apollo API rate limit reset, continuing with requests")
    
    def _format_company_data(self, apollo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format Apollo data with extended fields."""
        try:
            # Extract all available fields for maximum data extraction
            formatted_data = {}
            
            # Core fields (always attempt to extract these)
            core_fields = [
                "name", "website_url", "industry", "estimated_num_employees", "employee_count",
                "estimated_annual_revenue", "founded_year", "linkedin_url", "phone",
                "technologies", "description", "primary_domain"
            ]
            
            # Extract core fields
            for field in core_fields:
                if field in apollo_data and apollo_data[field] is not None:
                    # Special case for employee_count
                    if field == "estimated_num_employees":
                        formatted_data["employee_count"] = apollo_data[field]
                    # Special case for website
                    elif field == "website_url":
                        formatted_data["website"] = apollo_data[field]
                    # All other fields
                    else:
                        formatted_data[field] = apollo_data[field]
            
            # Process address separately
            formatted_data["address"] = self._format_address(apollo_data)
            
            # Extended fields - capture everything else that might be available
            for key, value in apollo_data.items():
                # Skip fields already processed
                if key in core_fields or key in ["street_address", "city", "state", "country", "postal_code"]:
                    continue
                    
                # Add all other fields that have values
                if value is not None and value != "":
                    formatted_data[key] = value
                    
                    # Log new fields that we haven't seen before
                    if key not in core_fields:
                        logger.info(f"Found additional Apollo field: {key}")
            
            # Log which fields we actually found
            available_fields = [k for k, v in formatted_data.items() if v]
            logger.info(f"Available Apollo data fields for {apollo_data.get('name', 'Unknown')}: {available_fields}")
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Error formatting Apollo data: {e}")
            return {
                "name": apollo_data.get('name', 'Unknown'),
                "website": apollo_data.get('website_url', 'Unknown'),
                "error": "Data formatting error"
            }
    
    def _format_address(self, apollo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format address data from Apollo.
        
        Args:
            apollo_data: The raw Apollo organization data
            
        Returns:
            dict: Formatted address data
        """
        return {
            "street": apollo_data.get('street_address'),
            "city": apollo_data.get('city'),
            "state": apollo_data.get('state'),
            "country": apollo_data.get('country'),
            "zip": apollo_data.get('postal_code')
        }
        
    def search_person(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Search for a person by email.
        
        Args:
            email: The email to search for
            
        Returns:
            dict: The person data or None if not found
        """
        if not self.api_key:
            logger.warning(f"Cannot search for {email}: No Apollo API key")
            return None
        
        # Check rate limits before making API call
        self._check_rate_limit()
        
        try:
            # Apollo's people/search endpoint
            endpoint = f"{self.base_url}/people/search"
            
            # Use the X-Api-Key header as required by Apollo
            headers = {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "X-Api-Key": self.api_key
            }
            
            payload = {
                "q_person_email": email,
                "page": 1,
                "per_page": 1
            }
            
            logger.info(f"Searching for person with email: {email}")
            
            # Record this API call for rate limiting
            self.calls_this_hour += 1
            
            # Make the request with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(endpoint, json=payload, headers=headers, timeout=15)
                    
                    # Check if we hit rate limits
                    if response.status_code == 429:
                        logger.warning(f"Apollo API rate limit hit for email search {email}. Attempt {attempt+1}/{max_retries}")
                        retry_after = int(response.headers.get('Retry-After', self.retry_delay * (2 ** attempt)))
                        time.sleep(retry_after)
                        continue
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get('people') and len(data['people']) > 0:
                            logger.info(f"Found person data for email {email}")
                            return self._format_person_data(data['people'][0])
                        else:
                            logger.warning(f"No person found for email {email}")
                            break
                    else:
                        # Enhanced error logging
                        try:
                            error_details = response.json()
                            logger.error(f"Apollo API error ({response.status_code}) for email search {email}: {error_details}")
                        except:
                            logger.error(f"Apollo API error ({response.status_code}) for email search {email}: {response.text[:500]}")
                        
                        # Don't retry certain error codes
                        if response.status_code in [400, 401, 403]:
                            break
                        
                        # For other errors, retry with backoff
                        retry_delay = self.retry_delay * (2 ** attempt)
                        time.sleep(retry_delay)
                        
                except Exception as e:
                    logger.error(f"Error during Apollo person search: {e}")
                    retry_delay = self.retry_delay * (2 ** attempt)
                    time.sleep(retry_delay)
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching for person with Apollo: {e}")
            return None
    
    def _format_person_data(self, person_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format person data from Apollo.
        
        Args:
            person_data: The raw Apollo person data
            
        Returns:
            dict: Formatted person data
        """
        return {
            "name": f"{person_data.get('first_name', '')} {person_data.get('last_name', '')}".strip(),
            "first_name": person_data.get('first_name'),
            "last_name": person_data.get('last_name'),
            "title": person_data.get('title'),
            "seniority": person_data.get('seniority'),
            "email": person_data.get('email'),
            "linkedin_url": person_data.get('linkedin_url'),
            "phone": person_data.get('phone_number'),
            "department": person_data.get('department'),
            "company": person_data.get('organization_name')
        }
