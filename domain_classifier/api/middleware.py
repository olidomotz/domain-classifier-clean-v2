"""Middleware for the domain classifier API."""
from flask_cors import CORS
import logging
import time
from flask import request, g
from functools import wraps

# Set up logging
logger = logging.getLogger(__name__)

def setup_cors(app):
    """Set up CORS middleware for the application."""
    if app is None:
        logger.error("App object is None in setup_cors")
        from flask import Flask
        app = Flask(__name__)
        logger.info("Created new Flask app as fallback in setup_cors")
        
    logger.info("Setting up CORS for app")
    CORS(app)
    logger.info("CORS setup complete")
    return app

def setup_performance_monitoring(app):
    """Set up middleware for monitoring API performance."""
    if app is None:
        logger.error("App object is None in setup_performance_monitoring")
        from flask import Flask
        app = Flask(__name__)
        logger.info("Created new Flask app as fallback in setup_performance_monitoring")
    
    @app.before_request
    def start_timer():
        g.start_time = time.time()
    
    @app.after_request
    def log_request_time(response):
        if hasattr(g, 'start_time'):
            total_time = time.time() - g.start_time
            endpoint = request.endpoint or request.path
            status_code = response.status_code
            
            # Log performance metrics
            if total_time > 10:  # Only log slow requests
                logger.warning(f"Slow request to {endpoint}: {total_time:.2f}s (Status: {status_code})")
            elif total_time > 5:
                logger.info(f"Medium-speed request to {endpoint}: {total_time:.2f}s (Status: {status_code})")
            
            # Add processing time to response headers
            response.headers['X-Processing-Time'] = f"{total_time:.3f}s"
            
        return response
    
    logger.info("Performance monitoring middleware added")
    return app

def timed_function(f):
    """Decorator to time function execution for performance monitoring."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Log if function takes too long
        if elapsed_time > 5:
            logger.warning(f"Slow function {f.__name__}: {elapsed_time:.2f}s")
        
        return result
    return wrapper
