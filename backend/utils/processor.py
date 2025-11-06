# backend/utils/processor.py
import os
import io
import re
import zipfile
import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from ultralytics import YOLO
import pytesseract
from pyzbar.pyzbar import decode as pyzbar_decode
from pyaadhaar.utils import isSecureQr
from pyaadhaar.decode import AadhaarSecureQr

from .verification_rules import (
    validate_aadhaar_number, validate_name, 
    validate_dob, validate_gender, correct_common_ocr_errors
)

# -------------------- AADHAAR IMAGE VERIFICATION --------------------
def is_aadhaar_image(image_bytes):
    """
    Verify if the uploaded image is actually an Aadhaar card.
    Uses multiple heuristics to detect Aadhaar card characteristics.
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
        
        # Heuristic 1: Check for Aadhaar-specific text patterns
        processed_img = preprocess_for_ocr_full(image)
        text = pytesseract.image_to_string(processed_img, config="--psm 6").lower()
        
        aadhaar_keywords = [
            'aadhaar', 'aadhar', 'uidai', 'government of india',
            'unique identification authority', 'dob', 'date of birth',
            'year of birth', 'male', 'female', 'gender'
        ]
        
        keyword_matches = sum(1 for keyword in aadhaar_keywords if keyword in text)
        
        # Heuristic 2: Check for 12-digit Aadhaar number pattern
        aadhaar_pattern = re.findall(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text)
        
        # Heuristic 3: Check aspect ratio (Aadhaar cards are typically rectangular)
        width, height = image.size
        aspect_ratio = width / height
        valid_aspect = 1.5 <= aspect_ratio <= 2.0  # Typical Aadhaar card aspect ratios
        
        # Heuristic 4: Check image dimensions (should be reasonable for a document)
        min_dimension = min(width, height)
        valid_size = min_dimension >= 300  # At least 300px on smaller side
        
        # Calculate confidence score
        confidence = 0
        
        # Text content (40% weight)
        if keyword_matches >= 2:
            confidence += 40
        elif keyword_matches >= 1:
            confidence += 20
            
        # Aadhaar number pattern (30% weight)
        if aadhaar_pattern:
            confidence += 30
            
        # Image characteristics (30% weight)
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
    label_lower = label.lower()
    
    if 'aadhaar' in label_lower or 'number' in label_lower:
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    elif 'dob' in label_lower or 'date' in label_lower:
        config = "--psm 7"
    else:
        config = "--psm 6"
    
    text = pytesseract.image_to_string(image, config=config)
    return text.strip().replace('\n', ' ')

# -------------------- QR CODE DECODING --------------------
def decode_secure_qr(image_np):
    """Decodes the Secure QR code from a NumPy image array."""
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        code = pyzbar_decode(gray)
        if not code:
            return {"error": "QR Code not found or could not be read"}
        
        qrData = code[0].data
        if isSecureQr(qrData):
            secure_qr = AadhaarSecureQr(int(qrData))
            decoded_data = secure_qr.decodeddata()
            return decoded_data
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
    
    # Apply error correction
    cleaned_raw_dob = correct_common_ocr_errors(raw_dob_text)
    
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
    Complete Aadhaar verification pipeline synchronized with original logic
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- NEW: Verify if image is actually an Aadhaar card ---
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
    if model_path and os.path.exists(model_path):
        custom_model = YOLO(model_path)
    else:
        custom_model = YOLO("yolov8n.pt")  # Fallback
    
    general_model = YOLO("yolov8n.pt")  # For face detection
    
    try:
        custom_model.to(device)
        general_model.to(device)
    except Exception:
        device = "cpu"  # Fallback to CPU

    # Convert bytes to PIL
    front_image_pil = Image.open(io.BytesIO(front_bytes)).convert("RGB")
    back_image_pil = Image.open(io.BytesIO(back_bytes)).convert("RGB") if back_bytes else None
    
    # Initialize results
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
    img_np = np.array(front_image_pil)
    yolo_results = custom_model(img_np, device=device, conf=0.25, verbose=False)
    
    # Create annotated image
    annotated_img = front_image_pil.copy()
    draw = ImageDraw.Draw(annotated_img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

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
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                draw.text((x1, max(0, y1 - 25)), f"{label}: {text[:30]}", fill="green", font=font)
    else:
        results["fraud_score"] += 5
        results["indicators"].append("🔴 HIGH: Could not extract any text fields from the Front Image.")

    # Convert annotated image to bytes
    img_byte_arr = io.BytesIO()
    annotated_img.save(img_byte_arr, format='JPEG')
    results["annotated_img_bytes"] = img_byte_arr.getvalue()

    # --- B: Face Detection ---
    face_results = general_model(img_np, classes=[0], device=device, conf=0.4, verbose=False)
    if len(face_results[0].boxes) > 0:
        results["indicators"].append("✅ LOW: Face detected on card.")
    else:
        results["fraud_score"] += 3
        results["indicators"].append("🔴 HIGH: No face detected on the card.")

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

    # Validation checks
    an_val = validate_aadhaar_number(ocr_aadhaar_num)
    name_val = validate_name(ocr_name)
    dob_val = validate_dob(ocr_dob)
    gender_val = validate_gender(ocr_gender)

    # Update fraud score based on validation
    if an_val == "Missing":
        results["fraud_score"] += 2
        results["indicators"].append("🔴 HIGH: Aadhaar number is missing.")
    elif "Invalid" in an_val:
        results["fraud_score"] += 3
        results["indicators"].append(f"🔴 HIGH: Aadhaar number '{ocr_aadhaar_num}' is {an_val}.")
    else:
        results["indicators"].append(f"✅ LOW: Aadhaar number '{ocr_aadhaar_num}' is valid (Checksum OK).")

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
        results["indicators"].append(f"🟡 MEDIUM: Date of Birth is missing. (Raw OCR: '{raw_dob_text}')")
    elif "Invalid" in dob_val:
        results["fraud_score"] += 2
        results["indicators"].append(f"🔴 HIGH: DOB '{ocr_dob}' is {dob_val}. (Raw OCR: '{raw_dob_text}')")
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
    if do_qr_check:
        qr_found = False
        
        # Check Front Image
        image_np_bgr_front = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        qr_data_front = decode_secure_qr(image_np_bgr_front)
        
        if "error" not in qr_data_front:
            results["qr_data"] = qr_data_front
            results["indicators"].append("✅ LOW: Secure QR Code decoded successfully from Front Image.")
            qr_found = True
        elif back_image_pil is not None:
            # Check Back Image if front failed and back exists
            image_np_back = np.array(back_image_pil)
            image_np_bgr_back = cv2.cvtColor(image_np_back, cv2.COLOR_RGB2BGR)
            qr_data_back = decode_secure_qr(image_np_bgr_back)
            
            if "error" not in qr_data_back:
                results["qr_data"] = qr_data_back
                results["back_image_qr_data"] = qr_data_back
                results["indicators"].append("✅ LOW: Secure QR Code decoded successfully from Back Image.")
                qr_found = True
            else:
                results["fraud_score"] += 3
                results["indicators"].append(f"🔴 HIGH: QR Code Error - Failed on both images.")
        else:
            results["fraud_score"] += 3
            results["indicators"].append(f"🔴 HIGH: QR Code Error - {qr_data_front.get('error')}")
        
        # Cross-reference if QR found
        if qr_found:
            qr_data = results["qr_data"]
            qr_name = qr_data.get("name", "")
            qr_dob = qr_data.get("dob", "")
            qr_gender = qr_data.get("gender", "")
            qr_aadhaar_last_4 = qr_data.get("aadhar_last_4_digit", "")

            # Name check
            clean_ocr_name = re.sub(r'\s+', '', ocr_name).lower()
            clean_qr_name = re.sub(r'\s+', '', qr_name).lower()
            
            if ocr_name and qr_name and clean_ocr_name not in clean_qr_name and clean_qr_name not in clean_ocr_name:
                 results["fraud_score"] += 3
                 results["indicators"].append(f"🔴 HIGH: Name Mismatch. OCR: '{ocr_name}', QR: '{qr_name}'")
            
            # Gender check
            if ocr_gender and qr_gender:
                ocr_g = 'M' if ocr_gender.lower() == 'male' else ('F' if ocr_gender.lower() == 'female' else 'O')
                if ocr_g != qr_gender:
                    results["fraud_score"] += 3
                    results["indicators"].append(f"🔴 HIGH: Gender Mismatch. OCR: '{ocr_gender}', QR: '{qr_gender}'")

            # Aadhaar last 4
            cleaned_ocr_aadhaar = re.sub(r'\s+', '', ocr_aadhaar_num)
            if cleaned_ocr_aadhaar and qr_aadhaar_last_4 and cleaned_ocr_aadhaar[-4:] != qr_aadhaar_last_4:
                 results["fraud_score"] += 3
                 results["indicators"].append(f"🔴 HIGH: Aadhaar Number Mismatch. OCR ends in: '{cleaned_ocr_aadhaar[-4:]}', QR ends in: '{qr_aadhaar_last_4}'")
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

    return results

# -------------------- BATCH PROCESSING --------------------
def process_zip_bytes(zip_bytes, model_path=None, do_qr_check=False, device="cpu"):
    """Process multiple images from ZIP file"""
    results = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        members = [n for n in z.namelist() if n.lower().endswith((".jpg",".jpeg",".png",".bmp",".tiff"))]
        for name in members:
            try:
                with z.open(name) as f:
                    img_bytes = f.read()
                    
                    # Verify if it's an Aadhaar image first
                    is_aadhaar, confidence, details = is_aadhaar_image(img_bytes)
                    
                    if not is_aadhaar:
                        results.append({
                            "filename": name,
                            "error": "NOT_AADHAAR",
                            "message": "The image does not appear to be an Aadhaar card",
                            "aadhaar_verification_details": details,
                            "confidence_score": confidence,
                            "assessment": "INVALID_INPUT"
                        })
                        continue
                    
                    # Process as Aadhaar card
                    rec = process_single_image_bytes(
                        img_bytes, 
                        back_bytes=None, 
                        do_qr_check=do_qr_check, 
                        model_path=model_path, 
                        device=device
                    )
                    rec["filename"] = name
                    results.append(rec)
                    
            except Exception as e:
                results.append({
                    "filename": name, 
                    "error": str(e),
                    "assessment": "ERROR",
                    "fraud_score": 100,
                    "indicators": [f"🔴 HIGH: Processing error - {str(e)}"]
                })
    return results