from flask_cors import CORS
import logging

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
