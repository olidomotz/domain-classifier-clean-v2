"""Routes initialization module for the domain classifier API."""
from domain_classifier.api.routes.health import register_health_routes
from domain_classifier.api.routes.classify import register_classify_routes
from domain_classifier.api.routes.enrich import register_enrich_routes
from domain_classifier.api.routes.similarity import register_similarity_routes
from domain_classifier.api.routes.batch import register_batch_routes
from domain_classifier.api.routes.bulk import register_bulk_routes
import logging

# Set up logging
logger = logging.getLogger(__name__)

def register_all_routes(app, llm_classifier, snowflake_conn, vector_db_conn):
    """Register all routes with the app."""
    if app is None:
        logger.error("App object is None in register_all_routes")
        from flask import Flask
        app = Flask(__name__)
        logger.info("Created new Flask app as fallback")
    
    # Register health routes
    logger.info("Registering health routes")
    app = register_health_routes(app, llm_classifier, snowflake_conn, vector_db_conn)
    
    # Register classification routes
    logger.info("Registering classification routes")
    app = register_classify_routes(app, llm_classifier, snowflake_conn)
    
    # Register enrichment routes
    logger.info("Registering enrichment routes")
    app = register_enrich_routes(app, snowflake_conn)
    
    # Register similarity routes
    logger.info("Registering similarity routes")
    app = register_similarity_routes(app, snowflake_conn)
    
    # Register batch processing routes
    logger.info("Registering batch processing routes")
    app = register_batch_routes(app, llm_classifier, snowflake_conn)
    
    # Register bulk processing routes (optimized for large domain sets)
    logger.info("Registering bulk processing routes")
    app = register_bulk_routes(app, llm_classifier, snowflake_conn)
    
    logger.info("All routes registered successfully")
    
    return app
