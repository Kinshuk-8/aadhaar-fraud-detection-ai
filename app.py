import os
import sys
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add backend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# ✅ Ensure YOLO models are downloaded before backend imports
from backend.load_model import ensure_models
model_paths = ensure_models()
os.environ["MODEL_PATH"] = model_paths["best.pt"]
os.environ["FACE_MODEL_PATH"] = model_paths["yolov8n.pt"]

app = Flask(__name__)
CORS(app)

# Increase file upload limit (50 MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Frontend directory
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), 'frontend')

# ✅ Import backend modules with error handling
try:
    from backend.utils.processor import process_single_image_bytes, process_zip_bytes
    BACKEND_IMPORTS_WORKING = True
    print("✅ Successfully imported backend modules")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"❌ Traceback: {traceback.format_exc()}")
    BACKEND_IMPORTS_WORKING = False
    
    # Fallback functions in case imports fail
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

# ─────────────────────────────────────────────
# 🌐 FRONTEND ROUTES
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# ⚙️ API ROUTES
# ─────────────────────────────────────────────

@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "running" if BACKEND_IMPORTS_WORKING else "degraded",
        "backend_imports": BACKEND_IMPORTS_WORKING,
        "model_best_exists": os.path.exists(os.environ.get("MODEL_PATH", "")),
        "model_yolo_exists": os.path.exists(os.environ.get("FACE_MODEL_PATH", "")),
        "service": "AadhaarVerify API"
    })

@app.route("/api/verify_single", methods=["POST"])
def api_verify_single():
    """Single Aadhaar card verification endpoint."""
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
            model_path=os.environ.get("MODEL_PATH", "backend/models/best.pt"),
            device="cpu"
        )

        return jsonify({"success": True, "result": result})

    except Exception as e:
        print(f"❌ Error in verify_single: {str(e)}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/verify_batch", methods=["POST"])
def api_verify_batch():
    """Batch Aadhaar card verification endpoint with progress tracking."""
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

        # Optional: limit max number of files per batch
        max_files = request.form.get("max_files")
        if max_files:
            max_files = int(max_files)
            print(f"🔧 Processing limit set to {max_files} files")
        
        results = process_zip_bytes(
            zip_bytes,
            model_path=os.environ.get("MODEL_PATH", "backend/models/best.pt"), 
            do_qr_check=False,
            device="cpu",
            max_files=max_files
        )

        total_files = len(results)
        valid_aadhaar = len([r for r in results if not r.get('error') or r.get('error') == 'NOT_AADHAAR'])
        non_aadhaar = len([r for r in results if r.get('error') == 'NOT_AADHAAR'])
        errors = len([r for r in results if r.get('error') and r.get('error') != 'NOT_AADHAAR'])
        
        summary = {
            "total_files_processed": total_files,
            "valid_aadhaar_cards": valid_aadhaar - non_aadhaar,
            "non_aadhaar_files": non_aadhaar,
            "processing_errors": errors,
            "success_rate": f"{((valid_aadhaar - non_aadhaar) / total_files * 100):.1f}%" if total_files > 0 else "0%"
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

# ─────────────────────────────────────────────
# 🧠 APP STARTUP
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
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
