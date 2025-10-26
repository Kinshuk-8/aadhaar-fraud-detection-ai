import streamlit as st
import os
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import numpy as np
import pytesseract
from ultralytics import YOLO
import tempfile
import cv2
from pyzbar.pyzbar import decode as pyzbar_decode
from pyaadhaar.utils import isSecureQr
from pyaadhaar.decode import AadhaarSecureQr
import re
import torch  # <-- CHANGED: Added torch import

# --- Page Configuration ---
st.set_page_config(page_title="Aadhaar Fraud Detection", layout="wide")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- CHANGED: Global Device Setup ---
def get_device():
    """Checks for available hardware accelerator."""
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return 'cuda'
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return 'mps'
    print("Using CPU")
    return 'cpu'

DEVICE = get_device()
# ------------------------------------

# --- 1. MODEL LOADING (Cached) ---
# We cache the models so they only load once
@st.cache_resource
def load_yolo_model(model_path):
    print(f"Loading custom model from {model_path}...")
    try:
        model = YOLO(model_path)
        print("Custom model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading custom model: {e}")
        st.error(f"Error loading custom model: {e}")
        return None

@st.cache_resource
def load_general_model():
    print("Loading general model (yolov8n.pt)...")
    try:
        model = YOLO("yolov8n.pt")
        print("General model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading general model: {e}")
        st.error(f"Error loading general model: {e}")
        return None

# --- 2. PREPROCESSING & OCR (Your Code) ---
def preprocess_for_ocr(crop):
    """Your preprocessing function."""
    gray = crop.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    gray = gray.filter(ImageFilter.SHARPEN)
    # Resize for better OCR - let's be gentle to avoid pixelation
    width, height = gray.width, gray.height
    gray = gray.resize((int(width * 1.5), int(height * 1.5)), Image.LANCZOS)
    return gray

def ocr_text(image, label):
    """Your OCR text extraction function."""
    # Allowlist depending on field type
    label_lower = label.lower()
    
    if 'aadhaar' in label_lower or 'number' in label_lower:
        # 1. FIX: Removed the space from the whitelist to avoid shlex quoting errors.
        # This will capture the digits, and we clean up any spaces in the extracted text later.
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        
    elif 'dob' in label_lower or 'date' in label_lower:
        # Date characters (digits, slash, hyphen)
        config = "--psm 7 -c tessedit_char_whitelist=0123456789/-"
        
    else:
        # PSM 6: Assume a single uniform block of text for names/gender/etc.
        config = "--psm 6"
    
    # This line (86) should now run without the ValueError
    text = pytesseract.image_to_string(image, config=config)
    
    # We strip and replace newlines regardless of the config used
    text = text.strip().replace('\n', ' ')
    
    return text

# --- 3. QR CODE & CHECKSUM LOGIC (From original API) ---

# Verhoeff Checksum (for Aadhaar Number validation)
_d = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
)
_p = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8)
)
_inv = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)

def validate_checksum(num_str_with_checksum):
    """Validates a number string using Verhoeff checksum."""
    c = 0
    num_digits = [int(d) for d in num_str_with_checksum]
    for i, digit in enumerate(reversed(num_digits)):
        c = _d[c][_p[(i % 8)][digit]]
    return c == 0

# --- CHANGED: Now accepts a NumPy array ---
def decode_aadhaar_qr(image_np):
    """Decodes the QR code from a NumPy image array."""
    try:
        # Image is already a NumPy array, just convert to gray
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

# --- 4. HELPER FUNCTIONS ---
def find_key_by_substr(data_dict, substr):
    """Helper to find a value in a dict where the key contains a substring."""
    substr = substr.lower()
    for key, value in data_dict.items():
        if substr in key.lower():
            return value
    return ""

# --- CHANGED: Added new helper for robust date comparison ---
def normalize_date(date_str):
    """Normalizes date string by replacing separators with '/'."""
    if not date_str:
        return ""
    return date_str.replace('-', '/').replace('.', '/')
# -----------------------------------------------------------

# --- 5. MAIN ANALYSIS PIPELINE ---
def perform_analysis(front_image_pil, back_image_pil, do_qr_check, custom_model, general_model):
    """
    Runs the full analysis pipeline on the uploaded images.
    """
    
    # --- Setup ---
    analysis_results = {
        "fraud_score": 0,
        "indicators": [],
        "ocr_data": {},
        "qr_data": {},
        "face_detected": False,
        "annotated_image": None,
        "assessment": "LOW"
    }
    
    # --- CHANGED: Removed local device detection, now uses global DEVICE ---
    
    # --- A: Front Image OCR & Bounding Boxes ---
    img_np = np.array(front_image_pil.convert("RGB"))
    # --- CHANGED: Uses global DEVICE ---
    yolo_results = custom_model(img_np, device=DEVICE, conf=0.25, verbose=False)
    
    annotated_img = front_image_pil.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    if yolo_results[0].boxes:
        for box in yolo_results[0].boxes:
            class_id = int(box.cls[0])
            label = custom_model.names[class_id]
            
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords

            # Crop region with padding
            padding = 5
            crop = front_image_pil.crop((max(0, x1 - padding), max(0, y1 - padding), 
                                         min(front_image_pil.width, x2 + padding), 
                                         min(front_image_pil.height, y2 + padding)))

            # Preprocess and OCR
            processed_crop = preprocess_for_ocr(crop)
            text = ocr_text(processed_crop, label)

            if text:
                analysis_results["ocr_data"][label] = text
                # Draw bounding boxes
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                draw.text((x1, max(0, y1 - 25)), f"{label}: {text[:30]}", fill="green", font=font)
    
    analysis_results["annotated_image"] = annotated_img

    # --- B: Face Detection ---
    # --- CHANGED: Uses global DEVICE ---
    face_results = general_model(img_np, classes=[0], device=DEVICE, conf=0.4, verbose=False) # class 0 is 'person'
    if len(face_results[0].boxes) > 0:
        analysis_results["face_detected"] = True
    else:
        analysis_results["fraud_score"] += 3
        analysis_results["indicators"].append("🔴 HIGH: No face ('person') detected on the card.")
        
    # --- C: Aadhaar Number Validation (Checksum) ---
    aadhaar_num_str = find_key_by_substr(analysis_results["ocr_data"], "number")
    aadhaar_num_clean = re.sub(r'\s+', '', aadhaar_num_str)
    
    if len(aadhaar_num_clean) == 12 and aadhaar_num_clean.isdigit():
        if not validate_checksum(aadhaar_num_clean):
            analysis_results["fraud_score"] += 3
            analysis_results["indicators"].append(f"🔴 HIGH: Aadhaar number '{aadhaar_num_str}' failed the Verhoeff checksum. It is mathematically invalid.")
    elif aadhaar_num_clean:
        analysis_results["fraud_score"] += 1
        analysis_results["indicators"].append(f"🟡 MEDIUM: Aadhaar number '{aadhaar_num_str}' is not 12 digits or contains invalid characters.")
    else:
        analysis_results["fraud_score"] += 1
        analysis_results["indicators"].append("🟡 MEDIUM: Could not extract Aadhaar number.")

    # --- D: QR Code Verification (if toggled) ---
    if do_qr_check:
        if not back_image_pil:
            analysis_results["fraud_score"] += 1
            analysis_results["indicators"].append("🟡 MEDIUM: QR Code check was enabled, but no back image was uploaded.")
        else:
            # --- CHANGED: Removed tempfile. Now converts PIL to NumPy in memory ---
            # 1. Convert PIL (RGB) to NumPy array
            back_img_np = np.array(back_image_pil.convert("RGB"))
            # 2. Convert RGB to BGR for OpenCV functions
            back_img_bgr = cv2.cvtColor(back_img_np, cv2.COLOR_RGB2BGR)
            
            # 3. Pass the NumPy array directly to the decode function
            qr_data = decode_aadhaar_qr(back_img_bgr) 
            analysis_results["qr_data"] = qr_data
            # ---------------------------------------------------------------------
            
            if "error" in qr_data:
                analysis_results["fraud_score"] += 2
                analysis_results["indicators"].append(f"🔴 HIGH: QR Code Error - {qr_data['error']}.")
            else:
                analysis_results["indicators"].append("✅ LOW: Secure QR Code decoded successfully.")
                
                # Compare OCR data with QR data
                ocr_name = find_key_by_substr(analysis_results["ocr_data"], "name")
                ocr_dob = find_key_by_substr(analysis_results["ocr_data"], "dob")
                
                qr_name = qr_data.get("name", "")
                qr_dob = qr_data.get("dob", "")
                qr_aadhaar_last_4 = qr_data.get("aadhar_last_4_digit", "")
                
                if ocr_name and qr_name and ocr_name.lower() not in qr_name.lower() and qr_name.lower() not in ocr_name.lower():
                    analysis_results["fraud_score"] += 3
                    analysis_results["indicators"].append(f"🔴 HIGH: Name Mismatch. OCR: '{ocr_name}', QR: '{qr_name}'")

                # --- CHANGED: Uses normalize_date for robust comparison ---
                if ocr_dob and qr_dob and normalize_date(ocr_dob) != normalize_date(qr_dob):
                    analysis_results["fraud_score"] += 3
                    analysis_results["indicators"].append(f"🔴 HIGH: DOB Mismatch. OCR: '{ocr_dob}', QR: '{qr_dob}'")
                # ---------------------------------------------------------

                if aadhaar_num_clean and qr_aadhaar_last_4 and aadhaar_num_clean[-4:] != qr_aadhaar_last_4:
                    analysis_results["fraud_score"] += 3
                    analysis_results["indicators"].append(f"🔴 HIGH: Aadhaar Number Mismatch. OCR ends in: '{aadhaar_num_clean[-4:]}', QR ends in: '{qr_aadhaar_last_4}'")

    else:
        analysis_results["indicators"].append("⚪ INFO: QR Code check was disabled by the user.")

    # --- E: Final Assessment ---
    if analysis_results["fraud_score"] >= 3:
        analysis_results["assessment"] = "HIGH"
    elif analysis_results["fraud_score"] >= 1:
        analysis_results["assessment"] = "MODERATE"
    else:
        analysis_results["assessment"] = "LOW"
        # Only show "all checks passed" if the score is 0 and no other indicators were added
        if not any(ind.startswith("🔴") or ind.startswith("🟡") for ind in analysis_results["indicators"]):
             analysis_results["indicators"].append("✅ LOW: All checks passed.")

    return analysis_results


# --- 6. STREAMLIT UI ---

st.title("Aadhaar Card Fraud Detection 🔎")
st.markdown("Upload the front and (optionally) back of an Aadhaar card to analyze it for signs of fraud.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙ Configuration")
    
    # Path to your custom model
    # IMPORTANT: Make sure 'best.pt' is in the same directory as this script
    MODEL_PATH = 'best.pt'
    
    front_image = st.file_uploader("Upload Front Image", type=["jpg", "jpeg", "png"])
    back_image = st.file_uploader("Upload Back Image (Optional)", type=["jpg", "jpeg", "png"])
    
    qr_toggle = st.toggle("Perform QR Code Verification", value=True, help="Requires the back image to be uploaded.")
    
    analyze_button = st.button("Analyze Card", type="primary", use_container_width=True)

# --- Main Area for Results ---
if analyze_button:
    if not front_image:
        st.error("Please upload the Front Image to begin analysis.")
    else:
        # Load models
        custom_model = load_yolo_model(MODEL_PATH)
        general_model = load_general_model()
        
        if custom_model is None or general_model is None:
            st.error("Models could not be loaded. Please check the console for errors.")
        else:
            with st.spinner("🕵 Analyzing image... This may take a moment."):
                # Open images with PIL
                pil_front_image = Image.open(front_image)
                pil_back_image = Image.open(back_image) if back_image else None
                
                # Perform the full analysis
                results = perform_analysis(pil_front_image, pil_back_image, qr_toggle, custom_model, general_model)
            
            st.success("Analysis complete!")

            # --- Display Final Result ---
            assessment = results["assessment"]
            if assessment == "HIGH":
                st.error(f"**Assessment: HIGH FRAUD RISK** (Score: {results['fraud_score']})")
            elif assessment == "MODERATE":
                st.warning(f"**Assessment: MODERATE FRAUD RISK** (Score: {results['fraud_score']})")
            else:
                st.success(f"**Assessment: LOW FRAUD RISK** (Score: {results['fraud_score']})")

            # --- Display Detailed Breakdown ---
            with st.expander("**See Detailed Analysis Breakdown**", expanded=True):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Annotated Front Image")
                    if results["annotated_image"]:
                        st.image(results["annotated_image"], caption="Detections on Front Image", use_column_width=True)
                    else:
                        st.image(pil_front_image, caption="Original Front Image", use_column_width=True)


                with col2:
                    st.subheader("Fraud Indicators")
                    for indicator in results["indicators"]:
                        if "HIGH" in indicator:
                            st.error(indicator)
                        elif "MEDIUM" in indicator:
                            st.warning(indicator)
                        elif "LOW" in indicator:
                            st.success(indicator)
                        else:
                            st.info(indicator)
                    
                    st.subheader("Face Detection")
                    if results["face_detected"]:
                        st.success("✅ Face detected on card.")
                    else:
                        st.error("❌ No face detected.")

                st.divider()

                col3, col4 = st.columns(2)
                
                with col3:
                    st.subheader("Extracted Text (OCR)")
                    if results["ocr_data"]:
                        st.json(results["ocr_data"])
                    else:
                        st.info("No text could be extracted.")

                with col4:
                    st.subheader("Decoded QR Data")
                    if not qr_toggle:
                        st.info("QR check was disabled.")
                    elif not back_image:
                        st.warning("No back image uploaded for QR check.")
                    elif results["qr_data"]:
                        st.json(results["qr_data"])
                    else:
                        st.error("Could not retrieve any data from QR code.")

# Instructions
st.info(
    "**How this works:**\n"
    "1.  **Text Extraction (OCR):** `best.pt` model finds fields on the front image, which are then preprocessed and read by Tesseract.\n"
    "2.  **Face Detection:** `yolov8n.pt` checks for a 'person' in the photo area.\n"
    "3.  **Checksum:** The extracted Aadhaar number is mathematically validated using the Verhoeff algorithm.\n"
    "4.  **QR Validation (Optional):** The QR code on the back is decoded and its data is cross-referenced with the text from the front.",
    icon="ℹ️"
)