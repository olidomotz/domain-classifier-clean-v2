"""
Flask application factory for the domain classifier API.
This module creates and configures the Flask application with all routes and middleware.
"""
import os
import logging
import time
import traceback
from flask import Flask, jsonify, request
from domain_classifier.api.middleware import setup_cors, setup_performance_monitoring  # Import both from middleware
from domain_classifier.config.settings import get_port
from domain_classifier.classifiers.llm_classifier import LLMClassifier
from domain_classifier.storage.snowflake_connector import SnowflakeConnector
from domain_classifier.storage.vector_db import VectorDBConnector

# Set up logging
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application with robust error handling."""
    try:
        logger.info("Creating Flask application...")
        app = Flask(__name__)
        
        # Log app creation
        logger.info(f"Flask app created: {app}")

        # Set up middleware
        logger.info("Setting up CORS middleware")
        app = setup_cors(app)
        logger.info("CORS middleware set up successfully")
        
        # Set up performance monitoring
        logger.info("Setting up performance monitoring middleware")
        app = setup_performance_monitoring(app)
        logger.info("Performance monitoring set up successfully")

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

        # Add universal error handler
        @app.errorhandler(Exception)
        def handle_exception(e):
            logger.error(f"Unhandled exception: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": "Internal server error",
                "message": str(e),
                "status": "error"
            }), 500
            
        # Add route timeout middleware
        @app.before_request
        def timeout_middleware():
            # Set a max request timeout of 120 seconds
            request.environ['REQUEST_TIMEOUT'] = 120
        
        logger.info("Application creation completed successfully")
        return app
    except Exception as e:
        logger.critical(f"Error creating Flask application: {e}")
        logger.critical(traceback.format_exc())
        
        # Create a minimal fallback app
        fallback_app = Flask(__name__)
        
        @fallback_app.route('/health')
        def health():
            return jsonify({
                "status": "warning",
                "message": "Running in fallback mode",
                "error": str(e)
            }), 200
            
        @fallback_app.route('/', defaults={'path': ''})
        @fallback_app.route('/<path:path>')
        def catch_all(path):
            return jsonify({
                "error": "Application initialization failed",
                "message": f"The application is running in fallback mode due to initialization error: {str(e)}",
                "status": "error"
            }), 500
        
        logger.info("Returning fallback app due to initialization error")
        return fallback_app
