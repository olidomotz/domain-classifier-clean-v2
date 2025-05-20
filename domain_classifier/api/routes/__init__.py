"""Routes initialization module for the domain classifier API."""
import logging

# Set up logging
logger = logging.getLogger(__name__)

# DO NOT import route modules at the top level - only import them inside the function

def register_all_routes(app, llm_classifier, snowflake_conn, vector_db_conn):
    """Register all routes with the app."""
    logger.info("Starting route registration process")
    
    if app is None:
        logger.error("App object is None in register_all_routes")
        from flask import Flask
        app = Flask(__name__)
        logger.info("Created new Flask app as fallback")
    
    # Import and register route modules INSIDE this function to avoid circular imports
    from domain_classifier.api.routes.health import register_health_routes
    logger.info("Registering health routes")
    app = register_health_routes(app, llm_classifier, snowflake_conn, vector_db_conn)
    
    from domain_classifier.api.routes.classify import register_classify_routes
    logger.info("Registering classification routes")
    app = register_classify_routes(app, llm_classifier, snowflake_conn)
    
    from domain_classifier.api.routes.enrich import register_enrich_routes
    logger.info("Registering enrichment routes")
    app = register_enrich_routes(app, snowflake_conn)
    
    from domain_classifier.api.routes.similarity import register_similarity_routes
    logger.info("Registering similarity routes")
    app = register_similarity_routes(app, snowflake_conn)
    
    from domain_classifier.api.routes.batch import register_batch_routes
    logger.info("Registering batch processing routes")
    app = register_batch_routes(app, llm_classifier, snowflake_conn)
    
    from domain_classifier.api.routes.bulk import register_bulk_routes
    logger.info("Registering bulk processing routes")
    app = register_bulk_routes(app, llm_classifier, snowflake_conn)
    
    logger.info("All routes registered successfully")
    
    return app
