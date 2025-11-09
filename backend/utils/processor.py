import os
import io
import re
import zipfile
import datetime
import json
import cv2
import numpy as np
import tempfile
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# --- Environment-aware paths for Render & local ---
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("backend", "models", "best.pt"))
FACE_MODEL_PATH = os.environ.get("FACE_MODEL_PATH", os.path.join("backend", "models", "yolov8n.pt"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join("backend", "uploads"))

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Import with error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available - running in test mode")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ Tesseract not available")

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from pyaadhaar.utils import isSecureQr
    from pyaadhaar.decode import AadhaarSecureQr
    PYAADHAAR_AVAILABLE = True
except ImportError:
    PYAADHAAR_AVAILABLE = False
    print("⚠️ PyAadhaar not available - QR decoding disabled")

# Import verification rules
try:
    from .verification_rules import (
        validate_aadhaar_number, validate_name, 
        validate_dob, validate_gender, correct_common_ocr_errors
    )
    VERIFICATION_RULES_AVAILABLE = True
except ImportError:
    VERIFICATION_RULES_AVAILABLE = False
    print("⚠️ Verification rules not available")

# -------------------- AADHAAR IMAGE VERIFICATION --------------------
def is_aadhaar_image(image_bytes):
    """Verify if the uploaded image is actually an Aadhaar card."""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Smart resize: keep detail but limit memory
        max_dim = 1280  # limit for Render safety but good OCR detail
        width, height = image.size
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        img_np = np.array(image)

        # Basic checks without OCR if Tesseract not available
        if not TESSERACT_AVAILABLE:
            width, height = image.size
            aspect_ratio = width / height
            valid_aspect = 1.5 <= aspect_ratio <= 2.0
            min_dimension = min(width, height)
            valid_size = min_dimension >= 300
            
            confidence = 50 if valid_aspect and valid_size else 20
            
            return confidence >= 50, confidence, {
                "keywords_found": 0,
                "aadhaar_numbers_found": 0,
                "aspect_ratio_valid": valid_aspect,
                "size_valid": valid_size,
                "detected_text_snippets": "OCR not available"
            }

        # ✅ Environment-safe paths for Render
        os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"
        os.environ["TMPDIR"] = "/tmp"

        # Heuristic 1: Aadhaar-specific text patterns
        processed_img = preprocess_for_ocr_full(image)
        text = pytesseract.image_to_string(
            processed_img, config="--psm 6 --oem 1", timeout=10
        ).lower()

        aadhaar_keywords = [
            'aadhaar', 'aadhar', 'uidai', 'government of india',
            'unique identification authority', 'dob', 'date of birth',
            'year of birth', 'male', 'female', 'gender'
        ]
        
        keyword_matches = sum(1 for keyword in aadhaar_keywords if keyword in text)
        
        # Heuristic 2: Aadhaar 12-digit number pattern
        aadhaar_pattern = re.findall(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text)
        
        # Heuristic 3: Aspect ratio
        width, height = image.size
        aspect_ratio = width / height
        valid_aspect = 1.5 <= aspect_ratio <= 2.0
        
        # Heuristic 4: Image dimensions
        min_dimension = min(width, height)
        valid_size = min_dimension >= 300
        
        # Confidence score
        confidence = 0
        if keyword_matches >= 2:
            confidence += 40
        elif keyword_matches >= 1:
            confidence += 20
        if aadhaar_pattern:
            confidence += 30
        if valid_aspect:
            confidence += 15
        if valid_size:
            confidence += 15
        
        return confidence >= 50, confidence, {
            "keywords_found": keyword_matches,
            "aadhaar_numbers_found": len(aadhaar_pattern),
            "aspect_ratio_valid": valid_aspect,
            "size_valid": valid_size,
            "detected_text_snippets": text[:200] + "..." if len(text) > 200 else text
        }

    except Exception as e:
        return False, 0, {"error": str(e)}

def preprocess_for_ocr_full(image):
    """Preprocessing for full image OCR verification"""
    gray = image.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray

# -------------------- OCR PREPROCESSING --------------------
def preprocess_for_ocr(crop):
    """Preprocessing for Tesseract on cropped images."""
    gray = crop.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    gray = gray.filter(ImageFilter.SHARPEN)
    width, height = gray.width, gray.height
    gray = gray.resize((int(width * 2.0), int(height * 2.0)), Image.Resampling.LANCZOS)
    return gray

def ocr_text(image, label):
    """OCR text extraction, configured for cropped fields."""
    if not TESSERACT_AVAILABLE:
        return f"OCR_{label}"  # Mock fallback
    
    label_lower = label.lower()
    
    if 'aadhaar' in label_lower or 'number' in label_lower:
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    elif 'dob' in label_lower or 'date' in label_lower:
        config = "--psm 7"
    else:
        config = "--psm 6"
    
    # Safe OCR with timeout
    try:
        text = pytesseract.image_to_string(image, config=config, timeout=10)
    except Exception as e:
        print(f"⚠️ OCR timeout for {label}: {e}")
        return ""
    return text.strip().replace('\n', ' ')

# -------------------- QR CODE DECODING --------------------
def decode_secure_qr(image_np):
    """Decodes the Secure QR code from a NumPy image array."""
    if not PYAADHAAR_AVAILABLE:
        return {"error": "QR decoding disabled - dependencies not available"}
    
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        code = pyzbar_decode(gray)
        if not code:
            return {"error": "QR Code not found or could not be read"}
        
        qrData = code[0].data
        if isSecureQr(qrData):
            secure_qr = AadhaarSecureQr(int(qrData))
            decoded_data = secure_qr.decodeddata()
            return dict(decoded_data) if hasattr(decoded_data, '__dict__') else decoded_data
        else:
            return {"error": "QR code is not a valid Secure Aadhaar QR."}
    except Exception as e:
        return {"error": f"QR decoding failed: {str(e)}"}

# -------------------- FIELD EXTRACTION HELPERS --------------------
def find_key_by_substr(data_dict, substr):
    """Helper to find a value in a dict where the key contains a substring."""
    substr = substr.lower()
    for key, value in data_dict.items():
        if substr in key.lower():
            return value
    return ""

def extract_dob_from_text(raw_dob_text):
    """Enhanced DOB extraction with error correction"""
    if not raw_dob_text:
        return ""
    
    # Apply error correction if available
    if VERIFICATION_RULES_AVAILABLE:
        cleaned_raw_dob = correct_common_ocr_errors(raw_dob_text)
    else:
        cleaned_raw_dob = raw_dob_text
    
    # Priority 1: User's strict DOB:DD/MM/YYYY pattern
    dob_match = re.search(r'(DOB|DoB|0OB)\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})', cleaned_raw_dob, re.IGNORECASE)
    if dob_match: 
        return dob_match.group(2).strip()
    
    # Priority 2: Full date pattern (DD/MM/YYYY)
    dob_match_full = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', cleaned_raw_dob)
    if dob_match_full:
        return dob_match_full.group(1).strip()
    
    # Priority 3: Year of Birth pattern
    year_match = re.search(r'(Year of Birth)\s*[:\-]?\s*(\d{4})', cleaned_raw_dob, re.IGNORECASE)
    if year_match: 
        return year_match.group(2).strip()
    
    # Fallback: Just a 4-digit number (potential year)
    year_only_match = re.search(r'\b(\d{4})\b', cleaned_raw_dob)
    if year_only_match:
        return year_only_match.group(1).strip()
    
    return ""

def correct_aadhaar_number(ocr_aadhaar_num):
    """Apply Aadhaar number correction heuristics"""
    if not ocr_aadhaar_num:
        return ""
    
    cleaned_num = re.sub(r'\s+', '', ocr_aadhaar_num)
    
    # Heuristic 1: If 12 digits, fails checksum, and starts with '9', try '8' 
    if len(cleaned_num) == 12 and cleaned_num.startswith('9'):
        if VERIFICATION_RULES_AVAILABLE:
            from .verification_rules import verhoeff_validate
            if not verhoeff_validate(cleaned_num):
                potential_fix = '8' + cleaned_num[1:]
                if verhoeff_validate(potential_fix):
                    return potential_fix
    
    # Heuristic 2: Replace common OCR confusions
    cleaned_num = cleaned_num.replace('O', '0').replace('I', '1').replace('o', '0').replace('l', '1')
    
    return cleaned_num

# -------------------- MAIN PROCESSING --------------------
def process_single_image_bytes(front_bytes, back_bytes=None, do_qr_check=False, model_path=None, device="cpu"):
    """
    Complete Aadhaar verification pipeline - JSON serializable version
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # If YOLO is not available, return basic analysis
    if not YOLO_AVAILABLE:
        return {
            "error": "MODEL_UNAVAILABLE",
            "message": "YOLO model not available - running in basic mode",
            "assessment": "UNKNOWN",
            "fraud_score": 0,
            "filename": f"single_{int(datetime.datetime.now().timestamp())}",
            "timestamp": ts,
            "indicators": ["⚠️ Running in basic mode - YOLO not available"]
        }
    
    # --- Verify if image is actually an Aadhaar card ---
    is_aadhaar, aadhaar_confidence, aadhaar_verification_details = is_aadhaar_image(front_bytes)
    
    if not is_aadhaar:
        return {
            "error": "NOT_AADHAAR",
            "message": "The uploaded image does not appear to be an Aadhaar card",
            "aadhaar_verification_details": aadhaar_verification_details,
            "confidence_score": aadhaar_confidence,
            "timestamp": ts,
            "filename": f"single_{int(datetime.datetime.now().timestamp())}",
            "assessment": "INVALID_INPUT"
        }
    
    # Load models
    try:
    # ✅ Load your fine-tuned Aadhaar YOLO model (best.pt)
        if model_path and os.path.exists(model_path):
            custom_model = YOLO(model_path)
            custom_model.to("cpu")
        else:
            # Fallback to MODEL_PATH environment variable (set in app.py)
            custom_model_path = os.environ.get("MODEL_PATH", os.path.join("backend", "models", "best.pt"))
            custom_model = YOLO(custom_model_path)
            custom_model.to("cpu")
    # ✅ Load general YOLO model (yolov8n.pt) for face detection
        face_model_path = os.environ.get("FACE_MODEL_PATH", os.path.join("backend", "models", "yolov8n.pt"))
        general_model = YOLO(face_model_path)
        general_model.to("cpu")

    # ✅ Move both to selected device (CPU or GPU)
        custom_model.to(device)
        general_model.to(device)

    except Exception as e:
        device = "cpu"  # Fallback to CPU for safety
        print(f"⚠️ Model loading error: {e}")

    # Convert bytes to PIL
    front_image_pil = Image.open(io.BytesIO(front_bytes)).convert("RGB")
    back_image_pil = Image.open(io.BytesIO(back_bytes)).convert("RGB") if back_bytes else None
    
    # Initialize results - ONLY JSON-SERIALIZABLE DATA
    results = {
        "fraud_score": 0,
        "indicators": [],
        "ocr_data": {},
        "qr_data": {},
        "assessment": "LOW",
        "filename": f"single_{int(datetime.datetime.now().timestamp())}",
        "timestamp": ts,
        "extracted": {},
        "back_image_qr_data": None,
        "aadhaar_verification": {
            "is_aadhaar_card": True,
            "confidence_score": aadhaar_confidence,
            "verification_details": aadhaar_verification_details
        }
    }

    # --- A: Front Image OCR & Bounding Boxes ---
    try:
        img_np = np.array(front_image_pil)
        yolo_results = custom_model(img_np, device=device, conf=0.25, verbose=False)
        
        # Create annotated image but don't store bytes in JSON
        annotated_img = front_image_pil.copy()
        draw = ImageDraw.Draw(annotated_img)
        
        # Extract text from detected fields
        if yolo_results[0].boxes:
            for box in yolo_results[0].boxes:
                class_id = int(box.cls[0])
                label = custom_model.names[class_id]
                
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords

                crop = front_image_pil.crop((x1, y1, x2, y2))
                processed_crop = preprocess_for_ocr(crop)
                text = ocr_text(processed_crop, label)

                if text:
                    results["ocr_data"][label] = text
        
        # Store annotation as base64 string if needed, but for now skip to avoid size
        # We'll remove annotated_img_bytes entirely from JSON response
        
    except Exception as e:
        results["fraud_score"] += 5
        results["indicators"].append("🔴 HIGH: Error in field detection.")

    # --- B: Face Detection ---
    try:
        face_results = general_model(img_np, classes=[0], device=device, conf=0.4, verbose=False)
        if len(face_results[0].boxes) > 0:
            results["indicators"].append("✅ LOW: Face detected on card.")
        else:
            results["fraud_score"] += 3
            results["indicators"].append("🔴 HIGH: No face detected on the card.")
    except Exception as e:
        results["indicators"].append("⚠️ Face detection failed.")

    # --- C: Data Extraction and Validation ---
    ocr_aadhaar_num = find_key_by_substr(results["ocr_data"], "number")
    ocr_name = find_key_by_substr(results["ocr_data"], "name")
    ocr_gender = find_key_by_substr(results["ocr_data"], "gender")
    
    # Extract DOB with cleaning
    raw_dob_text = ""
    for key, value in results["ocr_data"].items():
        if "dob" in key.lower() or "date" in key.lower():
            raw_dob_text = value
            break
    ocr_dob = extract_dob_from_text(raw_dob_text)

    # Apply Aadhaar number correction
    ocr_aadhaar_num = correct_aadhaar_number(ocr_aadhaar_num)

    # Store extracted data
    results["extracted"] = {
        "name": ocr_name,
        "dob": ocr_dob,
        "gender": ocr_gender,
        "aadhaar": ocr_aadhaar_num
    }

    # Validation checks with fallbacks
    if VERIFICATION_RULES_AVAILABLE:
        an_val = validate_aadhaar_number(ocr_aadhaar_num)
        name_val = validate_name(ocr_name)
        dob_val = validate_dob(ocr_dob)
        gender_val = validate_gender(ocr_gender)
    else:
        # Basic validation without verification_rules
        an_val = "Valid" if ocr_aadhaar_num and len(ocr_aadhaar_num) >= 12 else "Missing"
        name_val = "Valid" if ocr_name and len(ocr_name) >= 2 else "Missing"
        dob_val = "Valid" if ocr_dob else "Missing"
        gender_val = "Valid" if ocr_gender else "Missing"

    # Update fraud score based on validation
    if an_val == "Missing":
        results["fraud_score"] += 2
        results["indicators"].append("🔴 HIGH: Aadhaar number is missing.")
    elif "Invalid" in an_val:
        results["fraud_score"] += 3
        results["indicators"].append(f"🔴 HIGH: Aadhaar number '{ocr_aadhaar_num}' is {an_val}.")
    else:
        results["indicators"].append(f"✅ LOW: Aadhaar number '{ocr_aadhaar_num}' is valid.")

    if name_val == "Missing":
        results["fraud_score"] += 1
        results["indicators"].append("🟡 MEDIUM: Name is missing.")
    elif "Invalid" in name_val:
        results["fraud_score"] += 1
        results["indicators"].append(f"🟡 MEDIUM: Name '{ocr_name}' is {name_val}.")
    else:
        results["indicators"].append(f"✅ LOW: Name '{ocr_name}' format is valid.")

    if dob_val == "Missing":
        results["fraud_score"] += 1
        results["indicators"].append(f"🟡 MEDIUM: Date of Birth is missing.")
    elif "Invalid" in dob_val:
        results["fraud_score"] += 2
        results["indicators"].append(f"🔴 HIGH: DOB '{ocr_dob}' is {dob_val}.")
    else:
        results["indicators"].append(f"✅ LOW: DOB '{ocr_dob}' format is valid.")

    if gender_val == "Missing":
        results["fraud_score"] += 1
        results["indicators"].append("🟡 MEDIUM: Gender is missing.")
    elif "Invalid" in gender_val:
        results["fraud_score"] += 1
        results["indicators"].append(f"🟡 MEDIUM: Gender '{ocr_gender}' is {gender_val}.")
    else:
        results["indicators"].append(f"✅ LOW: Gender '{ocr_gender}' format is valid.")

    # --- D: QR Code Verification ---
    if do_qr_check and PYAADHAAR_AVAILABLE:
        try:
            image_np_bgr_front = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            qr_data_front = decode_secure_qr(image_np_bgr_front)
            
            if "error" not in qr_data_front:
                results["qr_data"] = qr_data_front
                results["indicators"].append("✅ LOW: Secure QR Code decoded successfully.")
            else:
                results["indicators"].append(f"⚠️ QR Code: {qr_data_front.get('error')}")
        except Exception as e:
            results["indicators"].append("⚠️ QR decoding error.")
    else:
        results["indicators"].append("⚪ INFO: QR Code check was disabled.")

    # Final assessment
    if results["fraud_score"] >= 3:
        results["assessment"] = "HIGH"
    elif results["fraud_score"] >= 1:
        results["assessment"] = "MODERATE"
    else:
        results["assessment"] = "LOW"
        if not any(ind.startswith("🔴") or ind.startswith("🟡") for ind in results["indicators"]):
             results["indicators"].append("✅ LOW: All checks passed.")

    # Ensure all data is JSON serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bytes, bytearray)):
            return "<binary_data>"
        elif hasattr(obj, '__dict__'):
            return {k: make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        else:
            return obj

    return make_serializable(results)

# -------------------- BATCH PROCESSING --------------------
# -------------------- BATCH PROCESSING --------------------
def process_zip_bytes(zip_bytes, model_path=None, do_qr_check=False, device="cpu", max_files=None):
    """Process multiple images from ZIP file with memory management and Render-safe OCR"""
    results = []
    
    if not YOLO_AVAILABLE:
        return [{
            "error": "MODEL_UNAVAILABLE",
            "message": "YOLO model not available",
            "assessment": "UNKNOWN"
        }]
    
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            # Get all image files
            members = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
            
            # Optional: Limit files for very large batches
            if max_files and len(members) > max_files:
                print(f"⚠️ Limiting processing to first {max_files} files out of {len(members)}")
                members = members[:max_files]
            
            print(f"📦 Processing {len(members)} images from ZIP file")
            
            processed_count = 0
            success_count = 0
            error_count = 0
            
            for name in members:
                try:
                    processed_count += 1
                    with z.open(name) as f:
                        img_bytes = f.read()

                    print(f"🔍 [{processed_count}/{len(members)}] Processing: {name}")

                    # ✅ Render memory safety: skip files over ~6 MB
                    if len(img_bytes) > 6 * 1024 * 1024:
                        print(f"⚠️ Skipping {name} - too large ({len(img_bytes)/1024/1024:.2f} MB)")
                        results.append({
                            "filename": name,
                            "error": "TOO_LARGE",
                            "message": "File exceeds safe size limit for Render free tier",
                            "assessment": "SKIPPED"
                        })
                        error_count += 1
                        continue

                    # ✅ Verify if it's an Aadhaar image first (with downscaling + timeout safety)
                    is_aadhaar, confidence, details = is_aadhaar_image(img_bytes)

                    if not is_aadhaar:
                        error_count += 1
                        results.append({
                            "filename": name,
                            "error": "NOT_AADHAAR",
                            "message": "The image does not appear to be an Aadhaar card", 
                            "confidence_score": confidence,
                            "aadhaar_verification_details": details,
                            "assessment": "INVALID_INPUT"
                        })
                        continue

                    # ✅ Process as Aadhaar card (Render-safe)
                    rec = process_single_image_bytes(
                        img_bytes, 
                        back_bytes=None, 
                        do_qr_check=do_qr_check, 
                        model_path=model_path, 
                        device=device
                    )

                    rec["filename"] = name

                    if rec.get("error"):
                        error_count += 1
                    else:
                        success_count += 1

                    results.append(rec)

                    print(f"✅ [{processed_count}/{len(members)}] Completed: {name} - Status: {rec.get('assessment', 'UNKNOWN')}")

                    # ✅ Memory cleanup between files
                    import gc
                    import time
                    time.sleep(0.5)
                    gc.collect()

                except Exception as e:
                    error_count += 1
                    print(f"❌ [{processed_count}/{len(members)}] Error processing {name}: {str(e)}")
                    results.append({
                        "filename": name,
                        "error": f"Processing error: {str(e)}",
                        "assessment": "ERROR"
                    })

    except Exception as e:
        print(f"❌ ZIP processing failed: {str(e)}")
        results.append({
            "filename": "batch_processing", 
            "error": f"ZIP processing failed: {str(e)}",
            "assessment": "ERROR"
        })
    
    print(f"📊 Batch processing complete: {success_count} successful, {error_count} errors out of {len(results)} files")
    return results
