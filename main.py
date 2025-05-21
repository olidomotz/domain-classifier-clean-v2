"""
Domain Classifier Main Entry Point

This is the main entry point for the Domain Classifier application.

It creates and configures the Flask application using the modular structure.
"""

import os
import sys
import logging
import traceback
from flask import Flask, jsonify
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

print("="*80)
print("STARTING DOMAIN CLASSIFIER")
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir("."))
print("="*80)

# Create a minimal fallback app in case all imports fail
fallback_app = Flask(__name__)

@fallback_app.route('/health')
def fallback_health():
    return jsonify({"status": "minimal fallback healthy", "version": "fallback", "timestamp": time.time()})

@fallback_app.route('/')
def fallback_root():
    return jsonify({
        "status": "ok",
        "message": "Domain Classifier API is running in fallback mode",
        "version": "fallback",
        "timestamp": time.time()
    })

# Default app is fallback
app = fallback_app

# Try to import from modular structure
try:
    # Create RSA key from base64 environment variable if needed
    if "SNOWFLAKE_KEY_BASE64" in os.environ and "SNOWFLAKE_PRIVATE_KEY_PATH" in os.environ:
        key_path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH", "/workspace/rsa_key.der")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        
        # Write key file if it doesn't exist or we're forcing refresh
        if not os.path.exists(key_path) or os.environ.get("FORCE_KEY_REFRESH", "false").lower() == "true":
            try:
                import base64
                with open(key_path, "wb") as key_file:
                    key_data = base64.b64decode(os.environ.get("SNOWFLAKE_KEY_BASE64"))
                    key_file.write(key_data)
                os.chmod(key_path, 0o600)
                logger.info(f"Created Snowflake key file at {key_path}")
            except Exception as e:
                logger.error(f"Failed to create Snowflake key file: {e}")
        else:
            logger.info(f"Using existing Snowflake key file at {key_path}")
    elif not "SNOWFLAKE_KEY_BASE64" in os.environ:
        logger.warning("SNOWFLAKE_KEY_BASE64 not set. Snowflake integration will be disabled.")
    
    # Import the app factory - this needs to be done after the routes/__init__.py and enrich.py fix
    logger.info("Attempting to import app factory")
    from domain_classifier.api.app import create_app
    logger.info("Successfully imported app factory")
    
    # Apply fixes before creating the app
    try:
        import fixes
        fixes.apply_patches()
        logger.info("Applied domain classifier fixes")
    except Exception as fix_error:
        logger.error(f"Failed to apply fixes: {fix_error}")
        logger.error(traceback.format_exc())
    
    # Create the Flask application
    logger.info("Creating app using create_app()")
    new_app = create_app()
    
    # Verify app was created properly
    if new_app is not None and hasattr(new_app, 'route'):
        app = new_app
        logger.info("Successfully created app using modular structure")
    else:
        logger.error("create_app() returned None or invalid app object")
        raise ValueError("App creation failed, app is None or invalid")

except ImportError as e:
    logger.error(f"ImportError in modular structure: {e}")
    logger.error(f"Error details: {traceback.format_exc()}")
    logger.warning("This is likely a circular import issue between modules")
    # Continue to fallback mechanism

except Exception as e:
    logger.error(f"Error using modular structure: {e}")
    logger.error(traceback.format_exc())
    
    # Try to fall back to old structure if the file exists
    try:
        # Check for both possible filenames
        if os.path.exists("api_service_old.py"):
            logger.info("Falling back to old structure (api_service_old.py)")
            
            # Update the import to handle the llm_classifier not found error
            try:
                # First ensure the llm_classifier module exists by copying from the new structure
                if not os.path.exists("llm_classifier.py"):
                    # If the old module doesn't exist, but we have the new one, copy it
                    logger.info("Creating llm_classifier.py for legacy support")
                    with open("llm_classifier.py", "w") as f:
                        f.write("# Legacy adapter\nfrom domain_classifier.classifiers.llm_classifier import LLMClassifier\n")
                
                import api_service_old
                
                if hasattr(api_service_old, 'app') and api_service_old.app is not None:
                    app = api_service_old.app
                    logger.info("Successfully loaded app from api_service_old.py")
                else:
                    logger.error("api_service_old.py doesn't contain a valid app")
            
            except Exception as import_error:
                logger.error(f"Error importing api_service_old: {import_error}")
        
        elif os.path.exists("api_service.py"):
            logger.info("Falling back to old structure (api_service.py)")
            
            try:
                # Similar fix for api_service.py
                if not os.path.exists("llm_classifier.py"):
                    logger.info("Creating llm_classifier.py for legacy support")
                    with open("llm_classifier.py", "w") as f:
                        f.write("# Legacy adapter\nfrom domain_classifier.classifiers.llm_classifier import LLMClassifier\n")
                
                import api_service
                
                if hasattr(api_service, 'app') and api_service.app is not None:
                    app = api_service.app
                    logger.info("Successfully loaded app from api_service.py")
                else:
                    logger.error("api_service.py doesn't contain a valid app")
            
            except Exception as import_error:
                logger.error(f"Error importing api_service: {import_error}")
        
        else:
            logger.warning("Neither api_service_old.py nor api_service.py found")
            # Keep using the fallback app
    
    except Exception as e:
        logger.error(f"Error falling back to old structure: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Using minimalist fallback app")

# Add health check to verify we have a real app
if hasattr(app, 'route') and not any(rule.rule == '/health' for rule in app.url_map.iter_rules()):
    @app.route('/health')
    def health():
        return jsonify({"status": "ok", "message": "Domain Classifier API is running", "version": "1.0.0"}), 200

# Add a universal error handler
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({
        "error": "Internal server error",
        "message": str(e),
        "status": "error"
    }), 500

# For direct execution
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting app on port {port}")
    app.run(host="0.0.0.0", port=port)
