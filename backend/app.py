# backend/app.py
import os
import io
import base64
import zipfile
import json
import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from utils.processor import process_single_image_bytes, process_zip_bytes


# -------------------- CONFIG --------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(ROOT, "uploads")
MODEL_PATH = os.path.join(ROOT, "models", "best.pt")
FRONTEND_PATH = os.path.join(ROOT, "..", "frontend")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

app = Flask(
    __name__,
    static_folder=FRONTEND_PATH,
    template_folder=FRONTEND_PATH
)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# -------------------- FRONTEND ROUTES --------------------
@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def frontend(path):
    """Serve frontend HTML files directly."""
    full_path = os.path.join(app.static_folder, path)
    if os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# -------------------- API: SINGLE VERIFICATION --------------------
@app.route("/api/verify_single", methods=["POST"])
def api_verify_single():
    """
    Single Aadhaar card verification.
    Accepts 'front', optional 'back', and 'qr' flag in form-data.
    """
    try:
        if 'front' not in request.files:
            return jsonify({"error": "Front image is required"}), 400

        front = request.files['front']
        back = request.files.get("back")
        qr = request.form.get("qr", "true").lower() == "true"
        device = request.form.get("device", "cpu")

        if front.filename == '':
            return jsonify({"error": "No front image selected"}), 400

        front_bytes = front.read()
        back_bytes = back.read() if back and back.filename != '' else None

        # Process Aadhaar verification
        result = process_single_image_bytes(
            front_bytes,
            back_bytes,
            do_qr_check=qr,
            model_path=MODEL_PATH,
            device=device
        )

        # Handle non-Aadhaar image case
        if result.get("error") == "NOT_AADHAAR":
            return jsonify({
                "success": False,
                "error": "NOT_AADHAAR",
                "message": result.get("message"),
                "confidence_score": result.get("confidence_score"),
                "aadhaar_verification_details": result.get("aadhaar_verification_details")
            })

        # Encode annotated image for display
        if result.get("annotated_img_bytes"):
            result["annotated_b64"] = base64.b64encode(
                result["annotated_img_bytes"]
            ).decode("utf-8")
            del result["annotated_img_bytes"]

      
       

        return jsonify({"success": True, "result": result})

    except Exception as e:
        import traceback
        return jsonify({"error": f"Server error: {str(e)}\n{traceback.format_exc()}"}), 500

# -------------------- API: BATCH VERIFICATION --------------------
@app.route("/api/verify_batch", methods=["POST"])
def api_verify_batch():
    """
    Batch Aadhaar card verification.
    Accepts ZIP file or multiple image uploads.
    """
    try:
        #qr = request.form.get("qr", "true").lower() == "true"
        qr = False
        device = request.form.get("device", "cpu")
        results = []

        zip_file = request.files.get("zip")
        if zip_file and zip_file.filename != '':
            zip_bytes = zip_file.read()
            results = process_zip_bytes(
                zip_bytes,
                model_path=MODEL_PATH,
                do_qr_check=qr,
                device=device
            )
        else:
            images = request.files.getlist("images")
            if not images or all(img.filename == '' for img in images):
                return jsonify({"error": "No ZIP or images uploaded"}), 400

            # Create a temp ZIP in memory for uniform handling
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, mode="w") as z:
                for f in images:
                    if f.filename != '':
                        z.writestr(secure_filename(f.filename), f.read())
            mem.seek(0)
            results = process_zip_bytes(
                mem.read(),
                model_path=MODEL_PATH,
                do_qr_check=qr,
                device=device
            )

        # Base64 encode annotated images
        for r in results:
            if r.get("annotated_img_bytes"):
                r["annotated_b64"] = base64.b64encode(
                    r["annotated_img_bytes"]
                ).decode("utf-8")
                del r["annotated_img_bytes"]

       
       

        return jsonify({"success": True, "results": results})

    except Exception as e:
        import traceback
        return jsonify({"error": f"Server error: {str(e)}\n{traceback.format_exc()}"}), 500


# -------------------- HEALTH CHECK --------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_exists": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH
    })

# -------------------- MAIN --------------------
if __name__ == "__main__":
    print(f"Starting Aadhaar Fraud Detection API...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"Frontend path: {FRONTEND_PATH}")
    print(f"Frontend exists: {os.path.exists(FRONTEND_PATH)}")
    
    app.run(host="0.0.0.0", port=5000, debug=True)