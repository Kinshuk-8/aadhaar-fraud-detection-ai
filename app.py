import os
import sys
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add backend to Python path - Render specific path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, 'backend')
sys.path.append(backend_path)

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Frontend paths - Render specific
FRONTEND_PATH = os.path.join(current_dir, 'frontend')

print("🚀 Starting Aadhaar Verification API on Render")
print(f"📁 Current directory: {current_dir}")
print(f"📁 Python path: {sys.path}")

# Import backend modules with proper error handling
try:
    from utils.processor import process_single_image_bytes, process_zip_bytes
    BACKEND_IMPORTS_WORKING = True
    print("✅ Successfully imported backend modules")
    
    # Check if model file exists
    model_path = os.path.join(current_dir, "backend", "models", "yolov8n.pt")
    if os.path.exists(model_path):
        print(f"✅ YOLO model found: {model_path}")
    else:
        print(f"❌ YOLO model not found at: {model_path}")
        # List available files for debugging
        models_dir = os.path.join(current_dir, "backend", "models")
        if os.path.exists(models_dir):
            print(f"📁 Models directory contents: {os.listdir(models_dir)}")
        
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
    return jsonify({
        "status": "running" if BACKEND_IMPORTS_WORKING else "degraded",
        "backend_imports": BACKEND_IMPORTS_WORKING,
        "model_exists": os.path.exists("backend/models/yolov8n.pt"),
        "platform": "Render",
        "current_directory": current_dir
    })

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
        
        # Use absolute path for model
        model_path = os.path.join(current_dir, "backend", "models", "yolov8n.pt")
        result = process_single_image_bytes(
            front_bytes,
            back_bytes=None,
            do_qr_check=False,
            model_path=model_path,
            device="cpu"
        )

        return jsonify({"success": True, "result": result})

    except Exception as e:
        print(f"❌ Error in verify_single: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/api/verify_batch", methods=["POST"])
def api_verify_batch():
    """Batch Aadhaar card verification endpoint"""
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
        print("✅ Processing batch images...")
        
        # Use absolute path for model
        model_path = os.path.join(current_dir, "backend", "models", "yolov8n.pt")
        results = process_zip_bytes(
            zip_bytes,
            model_path=model_path, 
            do_qr_check=False,
            device="cpu"
        )

        # Generate batch summary
        total_files = len(results)
        valid_aadhaar = len([r for r in results if not r.get('error') or r.get('error') == 'NOT_AADHAAR'])
        non_aadhaar = len([r for r in results if r.get('error') == 'NOT_AADHAAR'])
        errors = len([r for r in results if r.get('error') and r.get('error') != 'NOT_AADHAAR'])
        
        summary = {
            "total_files_processed": total_files,
            "valid_aadhaar_cards": valid_aadhaar - non_aadhaar,
            "non_aadhaar_files": non_aadhaar,
            "processing_errors": errors,
        }

        return jsonify({
            "success": True, 
            "results": results,
            "summary": summary,
            "total_files": total_files
        })

    except Exception as e:
        print(f"❌ Error in verify_batch: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/api/test")
def test_endpoint():
    return jsonify({
        "message": "AadhaarVerify API is running on Render",
        "status": "operational",
        "backend_loaded": BACKEND_IMPORTS_WORKING
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting AadhaarVerify on port {port}")
    print(f"📁 Backend imports working: {BACKEND_IMPORTS_WORKING}")
    
    # Debug file structure
    print("📁 Root directory contents:")
    for item in os.listdir(current_dir):
        print(f"   - {item}")
    
    app.run(host="0.0.0.0", port=port, debug=False)