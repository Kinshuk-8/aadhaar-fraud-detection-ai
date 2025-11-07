import os
import sys
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add backend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Frontend paths
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), 'frontend')

# Import backend modules with proper error handling
try:
    from backend.utils.processor import process_single_image_bytes, process_zip_bytes
    BACKEND_IMPORTS_WORKING = True
    print("✅ Successfully imported backend modules")
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
        "model_exists": os.path.exists("backend/models/best.pt"),
        "service": "AadhaarVerify API"
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
        
        print("✅ Processing single image...")
        result = process_single_image_bytes(
            front_bytes,
            back_bytes=None,
            do_qr_check=False,
            model_path="backend/models/best.pt",
            device="cpu"
        )

        return jsonify({"success": True, "result": result})

    except Exception as e:
        print(f"❌ Error in verify_single: {str(e)}")
        print(f"❌ Traceback: {traceback.format_exc()}")
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
        
        results = process_zip_bytes(
            zip_bytes,
            model_path="backend/models/best.pt",
            do_qr_check=False,
            device="cpu"
        )

        return jsonify({"success": True, "results": results})

    except Exception as e:
        print(f"❌ Error in verify_batch: {str(e)}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
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
    
    app.run(host="0.0.0.0", port=port, debug=False)