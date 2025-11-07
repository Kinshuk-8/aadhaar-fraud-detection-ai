import os
import sys
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add backend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Frontend paths
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), 'frontend')

print("🚀 Starting Aadhaar Verification API on Google Cloud Run")
print(f"📁 Current directory: {os.getcwd()}")
print(f"📁 Files in root: {os.listdir('.')}")

# Import backend modules with proper error handling
try:
    from backend.utils.processor import process_single_image_bytes, process_zip_bytes
    BACKEND_IMPORTS_WORKING = True
    print("✅ Successfully imported backend modules")
    
    # Test if model file exists
    model_path = "backend/models/yolov8n.pt"
    if os.path.exists(model_path):
        print(f"✅ YOLO model found: {model_path}")
    else:
        print(f"❌ YOLO model not found at: {model_path}")
        # List backend directory to debug
        if os.path.exists('backend'):
            print("📁 Backend directory contents:")
            for item in os.listdir('backend'):
                print(f"   - {item}")
                if item == 'models' and os.path.exists('backend/models'):
                    print("📁 Models directory contents:")
                    for model_item in os.listdir('backend/models'):
                        print(f"     - {model_item}")
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"❌ Traceback: {traceback.format_exc()}")
    BACKEND_IMPORTS_WORKING = False
    
    # Create fallback functions
    def process_single_image_bytes(*args, **kwargs):
        return {
            "error": "Backend modules not loaded", 
            "message": "Processor module import failed",
            "assessment": "ERROR"
        }
    
    def process_zip_bytes(*args, **kwargs):
        return [{
            "error": "Backend modules not loaded", 
            "message": "Processor module import failed",
            "assessment": "ERROR"
        }]

# Frontend Routes
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_PATH, "index.html")

@app.route("/<path:page>")
def serve_pages(page):
    if page in ["services", "about", "contact"]:
        return send_from_directory(FRONTEND_PATH, f"{page}.html")
    try:
        return send_from_directory(FRONTEND_PATH, page)
    except:
        return send_from_directory(FRONTEND_PATH, "index.html")

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(FRONTEND_PATH, "css"), filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(os.path.join(FRONTEND_PATH, "js"), filename)

# API Routes
@app.route("/api/health")
def health_check():
    """Comprehensive health check endpoint"""
    import subprocess
    
    health_info = {
        "status": "running" if BACKEND_IMPORTS_WORKING else "degraded",
        "backend_imports": BACKEND_IMPORTS_WORKING,
        "model_exists": os.path.exists("backend/models/yolov8n.pt"),
        "service": "AadhaarVerify API",
        "platform": "Google Cloud Run",
        "dependencies": {}
    }
    
    # Test Tesseract availability
    try:
        result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
        health_info["dependencies"]["tesseract"] = {
            "available": result.returncode == 0,
            "path": result.stdout.strip() if result.returncode == 0 else "Not found"
        }
    except Exception as e:
        health_info["dependencies"]["tesseract"] = {"available": False, "error": str(e)}
    
    # Test Python imports
    try:
        import pytesseract
        health_info["dependencies"]["pytesseract"] = {"available": True}
    except ImportError as e:
        health_info["dependencies"]["pytesseract"] = {"available": False, "error": str(e)}
    
    try:
        import cv2
        health_info["dependencies"]["opencv"] = {"available": True, "version": cv2.__version__}
    except ImportError as e:
        health_info["dependencies"]["opencv"] = {"available": False, "error": str(e)}
    
    try:
        from ultralytics import YOLO
        health_info["dependencies"]["ultralytics"] = {"available": True}
    except ImportError as e:
        health_info["dependencies"]["ultralytics"] = {"available": False, "error": str(e)}
    
    return jsonify(health_info)

@app.route("/api/verify_single", methods=["POST"])
def api_verify_single():
    """Single Aadhaar card verification endpoint"""
    try:
        if not BACKEND_IMPORTS_WORKING:
            return jsonify({
                "success": False,
                "error": "Backend modules not loaded",
                "message": "Processor functions are not available"
            }), 503

        if 'front' not in request.files:
            return jsonify({"error": "Front image is required"}), 400

        front = request.files['front']
        if front.filename == '':
            return jsonify({"error": "No front image selected"}), 400

        front_bytes = front.read()
        
        print(f"✅ Processing single image: {front.filename}")
        print(f"📏 File size: {len(front_bytes)} bytes")
        
        result = process_single_image_bytes(
            front_bytes,
            back_bytes=None,
            do_qr_check=False,
            model_path="backend/models/yolov8n.pt",  # Updated to use yolov8n.pt
            device="cpu"
        )

        return jsonify({"success": True, "result": result})

    except Exception as e:
        print(f"❌ Error in verify_single: {str(e)}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/api/verify_batch", methods=["POST"])
def api_verify_batch():
    """Batch Aadhaar card verification endpoint with progress tracking"""
    try:
        if not BACKEND_IMPORTS_WORKING:
            return jsonify({
                "success": False,
                "error": "Backend modules not loaded", 
                "message": "Processor functions are not available"
            }), 503

        zip_file = request.files.get("zip")
        if not zip_file or zip_file.filename == '':
            return jsonify({"error": "ZIP file is required"}), 400

        zip_bytes = zip_file.read()
        print(f"✅ Processing batch images: {zip_file.filename}")
        print(f"📦 ZIP file size: {len(zip_bytes)} bytes")
        
        # Optional: Limit files for very large batches to prevent timeout
        max_files = request.form.get("max_files")
        if max_files:
            max_files = int(max_files)
            print(f"🔧 Processing limit set to {max_files} files")
        
        results = process_zip_bytes(
            zip_bytes,
            model_path="backend/models/yolov8n.pt",  # Updated to use yolov8n.pt
            do_qr_check=False,
            device="cpu",
            max_files=max_files  # Pass the optional limit
        )

        # Generate batch summary
        total_files = len(results)
        valid_aadhaar = len([r for r in results if not r.get('error') or r.get('error') == 'NOT_AADHAAR'])
        non_aadhaar = len([r for r in results if r.get('error') == 'NOT_AADHAAR'])
        errors = len([r for r in results if r.get('error') and r.get('error') != 'NOT_AADHAAR'])
        
        summary = {
            "total_files_processed": total_files,
            "valid_aadhaar_cards": valid_aadhaar - non_aadhaar,  # Exclude non-Aadhaar files
            "non_aadhaar_files": non_aadhaar,
            "processing_errors": errors,
            "success_rate": f"{((valid_aadhaar - non_aadhaar) / total_files * 100):.1f}%" if total_files > 0 else "0%"
        }

        print(f"📊 Batch processing complete: {total_files} files processed")

        return jsonify({
            "success": True, 
            "results": results,
            "summary": summary,
            "total_files": total_files
        })

    except Exception as e:
        print(f"❌ Error in verify_batch: {str(e)}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Test endpoint to verify basic functionality
@app.route("/api/test", methods=["GET"])
def test_endpoint():
    """Test endpoint to verify basic API functionality"""
    return jsonify({
        "message": "AadhaarVerify API is running on Google Cloud Run",
        "status": "operational",
        "backend_loaded": BACKEND_IMPORTS_WORKING,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large"}), 413

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Changed default to 8080 for Cloud Run
    print(f"🚀 Starting AadhaarVerify on port {port}")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Backend imports working: {BACKEND_IMPORTS_WORKING}")
    
    # List files for debugging
    print("📁 Root directory contents:")
    for item in os.listdir('.'):
        print(f"   - {item}")
    
    if os.path.exists('backend'):
        print("📁 Backend directory contents:")
        for item in os.listdir('backend'):
            print(f"   - {item}")
            if item == 'models' and os.path.exists('backend/models'):
                print("📁 Models directory contents:")
                for model_item in os.listdir('backend/models'):
                    print(f"     - {model_item}")
    
    if os.path.exists('frontend'):
        print("📁 Frontend directory contents:")
        for item in os.listdir('frontend'):
            print(f"   - {item}")
    
    app.run(host="0.0.0.0", port=port, debug=False)