"""
Flask application factory for the domain classifier API.
This module creates and configures the Flask application with all routes and middleware.
"""
import os
import logging
import time
from flask import Flask, jsonify
from domain_classifier.api.middleware import setup_cors
# Import register_all_routes inside create_app function to avoid circular imports
from domain_classifier.config.settings import get_port
from domain_classifier.classifiers.llm_classifier import LLMClassifier
from domain_classifier.storage.snowflake_connector import SnowflakeConnector
from domain_classifier.storage.vector_db import VectorDBConnector

# Set up logging
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    logger.info("Creating Flask application...")
    app = Flask(__name__)
    
    # Log app creation
    logger.info(f"Flask app created: {app}")

    # Set up middleware
    logger.info("Setting up CORS middleware")
    app = setup_cors(app)
    logger.info("CORS middleware set up successfully")

    # Initialize services
    logger.info("Initializing LLM classifier")
    llm_classifier = LLMClassifier(
        api_key=os.environ.get("ANTHROPIC_API_KEY"), 
        model="claude-3-haiku-20240307"
    )
    
    logger.info("Initializing Snowflake connector")
    snowflake_conn = SnowflakeConnector()
    
    logger.info("Initializing Vector DB connector")
    vector_db_conn = VectorDBConnector()

    # Register routes with initialized services
    # Import register_all_routes here (inside function) to avoid circular imports
    logger.info("Importing route registration function")
    from domain_classifier.api.routes import register_all_routes
    
    logger.info("Registering routes")
    app = register_all_routes(app, llm_classifier, snowflake_conn, vector_db_conn)
    logger.info("Routes registered successfully")

    # Set up JSON encoder
    logger.info("Setting up JSON encoder")
    from domain_classifier.utils.json_encoder import CustomJSONEncoder
    app.json_encoder = CustomJSONEncoder
    logger.info("JSON encoder set up successfully")

    # Final check that app isn't None
    if app is None:
        logger.critical("App is None after all initialization - creating minimal app")
        app = Flask(__name__)
        
        @app.route('/health')
        def health():
            return jsonify({"status": "minimal fallback app is running, initialization failed"}), 200

    logger.info("Application creation completed successfully")
    return app
