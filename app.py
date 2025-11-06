import os
import io
import base64
import zipfile
import json
import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    from backend.utils.processor import process_single_image_bytes, process_zip_bytes
except ImportError as e:
    print(f"Import error: {e}")

# -------------------- CONFIG --------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(ROOT, "uploads")
MODEL_PATH = os.path.join(ROOT, "backend", "models", "best.pt")
FRONTEND_PATH = os.path.join(ROOT, "frontend")

# Create required directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# -------------------- FRONTEND ROUTES --------------------
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_PATH, "index.html")

@app.route("/<path:page>")
def serve_pages(page):
    # Serve HTML pages
    if page in ["services", "about", "contact"]:
        return send_from_directory(FRONTEND_PATH, f"{page}.html")
    
    # Try to serve static files
    full_path = os.path.join(FRONTEND_PATH, page)
    if os.path.exists(full_path):
        return send_from_directory(FRONTEND_PATH, page)
    
    # Fallback to index.html for client-side routing
    return send_from_directory(FRONTEND_PATH, "index.html")

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(FRONTEND_PATH, "css"), filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(os.path.join(FRONTEND_PATH, "js"), filename)

# -------------------- API ROUTES --------------------
@app.route("/api/verify_single", methods=["POST"])
def api_verify_single():
    """Single Aadhaar card verification endpoint"""
    try:
        if 'front' not in request.files:
            return jsonify({"error": "Front image is required"}), 400

        front = request.files['front']
        back = request.files.get("back")
        qr = request.form.get("qr", "true").lower() == "true"

        if front.filename == '':
            return jsonify({"error": "No front image selected"}), 400

        front_bytes = front.read()
        back_bytes = back.read() if back and back.filename != '' else None

        # Process the image
        result = process_single_image_bytes(
            front_bytes,
            back_bytes,
            do_qr_check=qr,
            model_path=MODEL_PATH,
            device="cpu"
        )

        # Handle non-Aadhaar case
        if result.get("error") == "NOT_AADHAAR":
            return jsonify({
                "success": False,
                "error": "NOT_AADHAAR",
                "message": result.get("message"),
                "confidence_score": result.get("confidence_score"),
                "aadhaar_verification_details": result.get("aadhaar_verification_details")
            })

        # Encode annotated image for frontend display
        if result.get("annotated_img_bytes"):
            result["annotated_b64"] = base64.b64encode(
                result["annotated_img_bytes"]
            ).decode("utf-8")
            del result["annotated_img_bytes"]

        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/api/verify_batch", methods=["POST"])
def api_verify_batch():
    """Batch Aadhaar card verification endpoint"""
    try:
        zip_file = request.files.get("zip")
        
        if not zip_file or zip_file.filename == '':
            return jsonify({"error": "ZIP file is required for batch processing"}), 400

        zip_bytes = zip_file.read()
        results = process_zip_bytes(
            zip_bytes,
            model_path=MODEL_PATH,
            do_qr_check=False,
            device="cpu"
        )

        # Encode annotated images
        for result in results:
            if result.get("annotated_img_bytes"):
                result["annotated_b64"] = base64.b64encode(
                    result["annotated_img_bytes"]
                ).decode("utf-8")
                del result["annotated_img_bytes"]

        return jsonify({"success": True, "results": results})

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_exists": os.path.exists(MODEL_PATH),
        "frontend_exists": os.path.exists(FRONTEND_PATH)
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)