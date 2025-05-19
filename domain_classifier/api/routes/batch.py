"""Batch processing API routes for domain classifier."""
import logging
import traceback
import json
import time
from flask import request, jsonify
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional
import threading
import queue

# Import domain utilities
from domain_classifier.utils.domain_utils import extract_domain_from_email
from domain_classifier.utils.error_handling import detect_error_type, create_error_result
from domain_classifier.config.overrides import check_domain_override

# Set up logging
logger = logging.getLogger(__name__)

# Create a processing queue and results store
processing_queue = queue.Queue()
batch_results = {}
worker_thread = None
is_worker_running = False

def process_domain_worker():
    """Worker thread to process domains from the queue."""
    global is_worker_running
    is_worker_running = True
    
    while True:
        try:
            # Get batch_id, domain, and other parameters from the queue
            batch_id, domain, force_reclassify, use_existing_content = processing_queue.get(timeout=1)
            
            if domain == "STOP":
                # Special signal to stop the worker
                processing_queue.task_done()
                break
                
            try:
                # Import needed components
                from domain_classifier.api.routes.classify import classify_domain
                from flask import Flask, request as flask_request
                
                # Create a temporary Flask app for context
                app = Flask(__name__)
                
                # Create a proxy request object
                class RequestProxy:
                    @property
                    def json(self):
                        return {
                            'url': domain,
                            'force_reclassify': force_reclassify,
                            'use_existing_content': use_existing_content
                        }
                    
                    @property
                    def method(self):
                        return 'POST'
                
                # Process the domain within the app context
                with app.test_request_context():
                    # Replace the request object
                    flask_request.json = RequestProxy().json
                    
                    # Call classify_domain directly
                    result, status_code = classify_domain()
                    
                    # Store the result
                    if batch_id in batch_results:
                        batch_results[batch_id]['results'][domain] = {
                            'result': result.get_json(),
                            'status_code': status_code
                        }
                        batch_results[batch_id]['completed'] += 1
                        
                        # Update progress
                        total = len(batch_results[batch_id]['domains'])
                        completed = batch_results[batch_id]['completed']
                        batch_results[batch_id]['progress'] = int((completed / total) * 100)
                        
                        logger.info(f"Batch {batch_id}: Processed {completed}/{total} domains ({batch_results[batch_id]['progress']}%)")
            except Exception as e:
                logger.error(f"Error processing domain {domain} in batch {batch_id}: {e}")
                logger.error(traceback.format_exc())
                
                # Record the error
                if batch_id in batch_results:
                    batch_results[batch_id]['results'][domain] = {
                        'result': {
                            'domain': domain,
                            'error': str(e),
                            'predicted_class': 'Error',
                            'confidence_score': 0
                        },
                        'status_code': 500
                    }
                    batch_results[batch_id]['completed'] += 1
            
            # Mark the task as done
            processing_queue.task_done()
            
        except queue.Empty:
            # Check if there are any pending batches
            pending_batches = False
            for batch_id, batch_data in batch_results.items():
                if batch_data['completed'] < len(batch_data['domains']):
                    pending_batches = True
                    break
                    
            # If no pending batches for 30 seconds, exit the worker
            if not pending_batches:
                # Sleep for 30 seconds before checking again
                time.sleep(30)
                
                # Check again if any new batches have been added
                pending_batches = False
                for batch_id, batch_data in batch_results.items():
                    if batch_data['completed'] < len(batch_data['domains']):
                        pending_batches = True
                        break
                        
                if not pending_batches:
                    logger.info("No pending batches, stopping worker thread")
                    break
        except Exception as e:
            logger.error(f"Error in worker thread: {e}")
            logger.error(traceback.format_exc())
            
    # Mark as not running
    is_worker_running = False
    logger.info("Worker thread stopped")

def ensure_worker_running():
    """Ensure the worker thread is running."""
    global worker_thread, is_worker_running
    
    if worker_thread is None or not worker_thread.is_alive():
        logger.info("Starting worker thread")
        worker_thread = threading.Thread(target=process_domain_worker)
        worker_thread.daemon = True
        worker_thread.start()
        is_worker_running = True
    else:
        logger.debug("Worker thread already running")

def register_batch_routes(app, llm_classifier, snowflake_conn):
    """Register batch processing routes."""
    
    @app.route('/batch-classify', methods=['POST', 'OPTIONS'])
    def batch_classify():
        """Process multiple domains in batch."""
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return '', 204
        
        try:
            data = request.json
            domains = data.get('domains', [])
            force_reclassify = data.get('force_reclassify', False)
            use_existing_content = data.get('use_existing_content', True)  # Changed default to True for batch processing
            
            if not domains:
                return jsonify({"error": "No domains provided"}), 400
                
            # Generate a unique batch ID
            import uuid
            batch_id = str(uuid.uuid4())
            
            # Initialize batch status
            batch_results[batch_id] = {
                'domains': domains,
                'results': {},
                'completed': 0,
                'progress': 0,
                'creation_time': time.time()  # Store creation time for cleanup
            }
            
            # Ensure worker thread is running
            ensure_worker_running()
            
            # Add domains to the queue
            for domain in domains:
                processing_queue.put((batch_id, domain, force_reclassify, use_existing_content))
                
            # Return the batch ID immediately
            return jsonify({
                "batch_id": batch_id,
                "total_domains": len(domains),
                "message": "Batch processing started",
                "status_endpoint": f"/batch-status/{batch_id}",
                "results_endpoint": f"/batch-results/{batch_id}"
            }), 202
            
        except Exception as e:
            logger.error(f"Error starting batch process: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500
    
    @app.route('/batch-status/<batch_id>', methods=['GET'])
    def batch_status(batch_id):
        """Get the status of a batch process."""
        if batch_id not in batch_results:
            return jsonify({"error": "Batch ID not found"}), 404
            
        # Return the current status
        batch = batch_results[batch_id]
        return jsonify({
            "batch_id": batch_id,
            "total_domains": len(batch['domains']),
            "completed": batch['completed'],
            "progress": batch['progress'],
            "is_complete": batch['completed'] == len(batch['domains'])
        }), 200
    
    @app.route('/batch-results/<batch_id>', methods=['GET'])
    def get_batch_results(batch_id):
        """Get the results of a completed batch process."""
        if batch_id not in batch_results:
            return jsonify({"error": "Batch ID not found"}), 404
            
        # Check if batch is complete
        batch = batch_results[batch_id]
        if batch['completed'] < len(batch['domains']):
            return jsonify({
                "batch_id": batch_id,
                "error": "Batch processing not complete",
                "progress": batch['progress'],
                "completed": batch['completed'],
                "total": len(batch['domains'])
            }), 400
            
        # Return the results
        return jsonify({
            "batch_id": batch_id,
            "total_domains": len(batch['domains']),
            "results": batch['results']
        }), 200
    
    # Add endpoint to get summary statistics for a batch
    @app.route('/batch-summary/<batch_id>', methods=['GET'])
    def batch_summary(batch_id):
        """Get summary statistics for a batch process."""
        if batch_id not in batch_results:
            return jsonify({"error": "Batch ID not found"}), 404
            
        # Get the batch data
        batch = batch_results[batch_id]
        
        # Calculate summary statistics
        total = len(batch['domains'])
        completed = batch['completed']
        
        # Count by class
        class_counts = {
            "Managed Service Provider": 0,
            "Integrator - Commercial A/V": 0,
            "Integrator - Residential A/V": 0,
            "Internal IT Department": 0,
            "Parked Domain": 0,
            "Error": 0,
            "Unknown": 0
        }
        
        # Count by final classification
        final_class_counts = {
            "1-MSP": 0,
            "2-Internal IT": 0,
            "3-Commercial Integrator": 0, 
            "4-Residential Integrator": 0,
            "5-Parked Domain with partial enrichment": 0,
            "6-Parked Domain - no enrichment": 0,
            "7-No Website available": 0
        }
        
        # Average confidence scores
        confidence_sum = 0
        confidence_count = 0
        
        # Detection methods
        detection_methods = {}
        
        # Process results
        for domain, result_data in batch['results'].items():
            result = result_data.get('result', {})
            
            # Count by class
            predicted_class = result.get('predicted_class', 'Unknown')
            if predicted_class in class_counts:
                class_counts[predicted_class] += 1
            else:
                class_counts['Unknown'] += 1
                
            # Count by final classification  
            final_classification = result.get('final_classification', 'Unknown')
            if final_classification in final_class_counts:
                final_class_counts[final_classification] += 1
                
            # Sum confidence scores
            confidence_score = result.get('confidence_score', 0)
            if confidence_score > 0:
                confidence_sum += confidence_score
                confidence_count += 1
                
            # Count detection methods
            detection_method = result.get('detection_method', 'Unknown')
            if detection_method in detection_methods:
                detection_methods[detection_method] += 1
            else:
                detection_methods[detection_method] = 1
                
        # Calculate average confidence
        avg_confidence = round(confidence_sum / max(1, confidence_count), 2)
        
        # Return the summary
        return jsonify({
            "batch_id": batch_id,
            "total_domains": total,
            "completed": completed,
            "progress": batch['progress'],
            "is_complete": completed == total,
            "class_counts": class_counts,
            "final_classification_counts": final_class_counts,
            "average_confidence": avg_confidence,
            "detection_methods": detection_methods
        }), 200
    
    # Clean up old batch results periodically
    @app.before_request
    def cleanup_old_batches():
        """Remove old batch results to prevent memory leaks."""
        try:
            # Keep only batches from the last 24 hours
            current_time = time.time()
            batches_to_remove = []
            
            for batch_id, batch in batch_results.items():
                # Check if batch has a creation_time
                if 'creation_time' not in batch:
                    batch['creation_time'] = current_time
                    continue
                    
                # Check if batch is older than 24 hours
                if current_time - batch['creation_time'] > 86400:  # 24 hours in seconds
                    batches_to_remove.append(batch_id)
            
            # Remove old batches
            for batch_id in batches_to_remove:
                logger.info(f"Removing old batch {batch_id}")
                del batch_results[batch_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up old batches: {e}")
    
    # Add endpoint to delete a batch
    @app.route('/batch-delete/<batch_id>', methods=['DELETE'])
    def delete_batch(batch_id):
        """Delete a batch and its results."""
        if batch_id not in batch_results:
            return jsonify({"error": "Batch ID not found"}), 404
            
        # Delete the batch
        del batch_results[batch_id]
        
        return jsonify({
            "batch_id": batch_id,
            "message": "Batch deleted successfully"
        }), 200
    
    return app
