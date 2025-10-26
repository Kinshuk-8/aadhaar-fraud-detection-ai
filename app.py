import streamlit as st
import os
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import numpy as np
import pytesseract
from ultralytics import YOLO
import torch
import cv2
from pyzbar.pyzbar import decode as pyzbar_decode
from pyaadhaar.utils import isSecureQr
from pyaadhaar.decode import AadhaarSecureQr
import re
import platform
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Aadhaar Fraud Detection", layout="wide")

# --- Configure Tesseract path based on environment ---
if platform.system() == "Windows":
    # Change this to your local Windows Tesseract path if different
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pytesseract.get_tesseract_version() # Test if Tesseract is found
    except Exception as e:
        st.error(f"Tesseract not found at r'C:\Program Files\Tesseract-OCR\tesseract.exe'. Please update the path. Error: {e}")
        st.stop()
elif platform.system() in ["Linux", "Darwin"]:
    # Streamlit Cloud / Linux default path, relies on packages.txt for installation
    if os.path.exists("/usr/bin/tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    else:
        # This error is usually suppressed on Streamlit Cloud if packages.txt is correct.
        st.warning(
            "Tesseract not found at expected path. Ensure 'tesseract-ocr' is listed in packages.txt"
        )

# --- Global Device Setup (For YOLO) ---
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

# --- 1. MODEL LOADING (Cached) ---
@st.cache_resource
def load_yolo_model(model_path):
    print(f"Loading custom model from {model_path}...")
    if not os.path.exists(model_path):
        st.error(f"Custom model not found at path: {model_path}. Make sure 'best.pt' is uploaded.")
        return None
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

# --- 2. PREPROCESSING & OCR ---
def preprocess_for_ocr(crop):
    """Preprocessing for Tesseract on cropped images."""
    gray = crop.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    gray = gray.filter(ImageFilter.SHARPEN)
    width, height = gray.width, gray.height
    
    # Increased scaling factor from 1.5 to 2.0 for better detail recognition
    gray = gray.resize((int(width * 2.0), int(height * 2.0)), Image.Resampling.LANCZOS)
    return gray

def ocr_text(image, label):
    """OCR text extraction, configured for cropped fields."""
    label_lower = label.lower()
    
    if 'aadhaar' in label_lower or 'number' in label_lower:
        # Use simple whitelist for digits only
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    
    elif 'dob' in label_lower or 'date' in label_lower:
        # Allow Tesseract to read all characters in a single line
        config = "--psm 7"
    
    else:
        # PSM 6: Assume a single uniform block of text for names/gender
        config = "--psm 6"
    
    text = pytesseract.image_to_string(image, config=config)
    text = text.strip().replace('\n', ' ')
    return text

# --- 3. OCR CORRECTION ---
def correct_common_ocr_errors(text):
    """
    Applies common corrections for errors often seen in Aadhaar OCR.
    """
    if not text:
        return ""
    
    # Keep original case for regex, but clean spaces
    corrected_text = text.replace(' ', '')
    
    # Handle '45' misread for '15' (often '1' is misread as '4' in certain fonts)
    corrected_text = re.sub(r'4([/.-])', r'1\1', corrected_text) 
    
    # Replace common separators with /
    corrected_text = corrected_text.replace('-', '/').replace('.', '/')
    
    return corrected_text

# --- 4. VERHOEFF ALGORITHM ---
d_table = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

p_table = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

def verhoeff_validate(num):
    """Return True if Aadhaar passes Verhoeff checksum validation"""
    try:
        c = 0
        num = num[::-1]
        for i, item in enumerate(num):
            c = d_table[c][p_table[i % 8][int(item)]]
        return c == 0
    except (ValueError, TypeError):
        return False

# --- 5. VALIDATION RULES ---
def validate_aadhaar_number(aadhaar_number):
    if not aadhaar_number:
        return "Missing"
    aadhaar_number = re.sub(r'\s+', '', aadhaar_number) # Clean spaces
    if not re.fullmatch(r'\d{12}', aadhaar_number):
        return "Invalid (must be 12 digits)"
    if not verhoeff_validate(aadhaar_number):
        return "Invalid (checksum failed – possible tampering)"
    return "Valid"

def validate_name(name):
    if not name:
        return "Missing"
    if not re.fullmatch(r"[A-Za-z. ]{3,50}", name):
        return "Invalid (special characters or too short/long)"
    return "Valid"

def validate_dob(dob):
    if not dob:
        return "Missing"
    try:
        if len(dob) == 4: # Case for "Year of Birth"
            year = int(dob)
            current_year = datetime.now().year
            if 1900 <= year <= current_year:
                return "Valid"
            else:
                return "Invalid (year out of range)"
        else:
            # Requires clean DD/MM/YYYY format
            parsed = datetime.strptime(dob, "%d/%m/%Y")
            if parsed > datetime.now():
                return "Invalid (future date)"
            return "Valid"
    except ValueError:
        return "Invalid (wrong format or impossible date)"

def validate_gender(gender):
    if not gender:
        return "Missing"
    if gender.lower() not in ["male", "female"]:
        return "Invalid (must be Male/Female)"
    return "Valid"

# --- 6. QR CODE & HELPER FUNCTIONS ---
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

def find_key_by_substr(data_dict, substr):
    """Helper to find a value in a dict where the key contains a substring."""
    substr = substr.lower()
    for key, value in data_dict.items():
        if substr in key.lower():
            return value
    return ""

def normalize_date(date_str):
    """Normalizes date string by replacing separators with '/'."""
    if not date_str:
        return ""
    return date_str.replace('-', '/').replace('.', '/')

# --- 7. MAIN ANALYSIS PIPELINE ---
def perform_analysis(image_pil, do_qr_check, custom_model, general_model):
    """
    Runs the full analysis pipeline using YOLO for ROIs.
    """
    
    analysis_results = {
        "fraud_score": 0,
        "indicators": [],
        "ocr_data": {},
        "qr_data": {},
        "annotated_image": None,
        "assessment": "LOW"
    }
    
    # --- A: Front Image OCR & Bounding Boxes ---
    img_np = np.array(image_pil.convert("RGB"))
    yolo_results = custom_model(img_np, device=DEVICE, conf=0.25, verbose=False)
    
    annotated_img = image_pil.copy()
    draw = ImageDraw.Draw(annotated_img)
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

            crop = image_pil.crop((x1, y1, x2, y2))
            processed_crop = preprocess_for_ocr(crop)
            text = ocr_text(processed_crop, label)

            if text:
                analysis_results["ocr_data"][label] = text
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                draw.text((x1, max(0, y1 - 25)), f"{label}: {text[:30]}", fill="green", font=font)
    
    analysis_results["annotated_image"] = annotated_img

    if not analysis_results["ocr_data"]:
        analysis_results["fraud_score"] += 5
        analysis_results["indicators"].append("🔴 HIGH: Could not extract any text fields. Check image quality or model.")

    # --- B: Face Detection ---
    face_results = general_model(img_np, classes=[0], device=DEVICE, conf=0.4, verbose=False)
    if len(face_results[0].boxes) > 0:
        analysis_results["indicators"].append("✅ LOW: Face detected on card.")
    else:
        analysis_results["fraud_score"] += 3
        analysis_results["indicators"].append("🔴 HIGH: No face ('person') detected on the card.")
        
    # --- C: Data Cleaning and Validation Prep  ---
    
    # 1. Get raw OCR data
    ocr_aadhaar_num = find_key_by_substr(analysis_results["ocr_data"], "number")
    ocr_name = find_key_by_substr(analysis_results["ocr_data"], "name")
    ocr_gender = find_key_by_substr(analysis_results["ocr_data"], "gender")
    
    # Get raw DOB text using a generic search that is likely to catch the key
    raw_dob_text = ""
    for key, value in analysis_results["ocr_data"].items():
        if "dob" in key.lower() or "date" in key.lower():
            raw_dob_text = value
            break

    # 2. Clean and Extract DOB (FIXED LOGIC to ensure ocr_dob is captured)
    ocr_dob = ""
    if raw_dob_text:
        # Step 1: Apply error correction (converts 4->1 and separators to '/')
        cleaned_raw_dob = correct_common_ocr_errors(raw_dob_text)
        
        # Step 2: Apply user's specific regex *or* look for the date/year pattern directly.
        
        # Priority 1: User's strict DOB:DD/MM/YYYY pattern
        dob_match = re.search(r'(DOB|DoB|0OB)\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})', cleaned_raw_dob, re.IGNORECASE)
        
        if dob_match: 
            ocr_dob = dob_match.group(2).strip()
        else:
            # Priority 2: Full date pattern (DD/MM/YYYY) anywhere in the text
            dob_match_full = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', cleaned_raw_dob)
            if dob_match_full:
                 # Normalization already happened in correct_common_ocr_errors, just clean.
                ocr_dob = dob_match_full.group(1).strip()
            else:
                 # Priority 3: Year of Birth pattern
                year_match = re.search(r'(Year of Birth)\s*[:\-]?\s*(\d{4})', cleaned_raw_dob, re.IGNORECASE)
                if year_match: 
                    ocr_dob = year_match.group(2).strip()
                else:
                    # Fallback 4: Just a 4-digit number (potential year)
                    year_only_match = re.search(r'\b(\d{4})\b', cleaned_raw_dob)
                    if year_only_match:
                         ocr_dob = year_only_match.group(1).strip()


    # 3. Update the dictionary with the *cleaned* DOB for JSON output
    # This loop ensures the JSON output reflects the date used for validation
    for key in list(analysis_results["ocr_data"].keys()):
        if "dob" in key.lower() or "date" in key.lower():
            analysis_results["ocr_data"][key] = ocr_dob
    
    # 4. Validation Checks
    
    # Aadhaar Number
    an_val = validate_aadhaar_number(ocr_aadhaar_num)
    if an_val == "Valid":
        analysis_results["indicators"].append(f"✅ LOW: Aadhaar number '{ocr_aadhaar_num}' is valid (Checksum OK).")
    elif an_val == "Missing":
        analysis_results["fraud_score"] += 2
        analysis_results["indicators"].append("🔴 HIGH: Aadhaar number is missing.")
    else:
        analysis_results["fraud_score"] += 3
        analysis_results["indicators"].append(f"🔴 HIGH: Aadhaar number '{ocr_aadhaar_num}' is {an_val}.")

    # Name
    name_val = validate_name(ocr_name)
    if name_val == "Valid":
        analysis_results["indicators"].append(f"✅ LOW: Name '{ocr_name}' format is valid.")
    elif name_val == "Missing":
        analysis_results["fraud_score"] += 1
        analysis_results["indicators"].append("🟡 MEDIUM: Name is missing.")
    else:
        analysis_results["fraud_score"] += 1
        analysis_results["indicators"].append(f"🟡 MEDIUM: Name '{ocr_name}' is {name_val}.")

    # Date of Birth (Validate the *cleaned* ocr_dob)
    dob_val = validate_dob(ocr_dob)
    if dob_val == "Valid":
        # Only append SUCCESS if it's clean and valid
        analysis_results["indicators"].append(f"✅ LOW: DOB '{ocr_dob}' format is valid.")
    elif dob_val == "Missing":
        analysis_results["fraud_score"] += 1
        # Include a better message for debugging missing DOB
        reason = f"(Raw OCR: '{raw_dob_text}')" if raw_dob_text else "(No text detected for DOB)"
        # This will now only trigger if ocr_dob is genuinely empty.
        analysis_results["indicators"].append(f"🟡 MEDIUM: Date of Birth is missing. {reason}")
    else:
        analysis_results["fraud_score"] += 2
        analysis_results["indicators"].append(f"🔴 HIGH: DOB '{ocr_dob}' is {dob_val}. (Raw OCR: '{raw_dob_text}')")

    # Gender
    gender_val = validate_gender(ocr_gender)
    if gender_val == "Valid":
        analysis_results["indicators"].append(f"✅ LOW: Gender '{ocr_gender}' format is valid.")
    elif gender_val == "Missing":
        analysis_results["fraud_score"] += 1
        analysis_results["indicators"].append("🟡 MEDIUM: Gender is missing.")
    else:
        analysis_results["fraud_score"] += 1
        analysis_results["indicators"].append(f"🟡 MEDIUM: Gender '{ocr_gender}' is {gender_val}.")

    # --- D: QR Code Verification (if toggled) ---
    if do_qr_check:
        image_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        qr_data = decode_secure_qr(image_np_bgr)
        analysis_results["qr_data"] = qr_data
        
        if "error" in qr_data:
            analysis_results["fraud_score"] += 3
            analysis_results["indicators"].append(f"🔴 HIGH: QR Code Error - {qr_data['error']}.")
        else:
            analysis_results["indicators"].append("✅ LOW: Secure QR Code decoded successfully.")
            
            # --- E: Cross-Reference OCR with QR Data ---
            qr_name = qr_data.get("name", "")
            qr_dob = qr_data.get("dob", "")
            qr_gender = qr_data.get("gender", "")
            qr_aadhaar_last_4 = qr_data.get("aadhar_last_4_digit", "")

            # 1. Name check
            if ocr_name and qr_name and ocr_name.lower() not in qr_name.lower() and qr_name.lower() not in ocr_name.lower():
                 analysis_results["fraud_score"] += 3
                 analysis_results["indicators"].append(f"🔴 HIGH: Name Mismatch. OCR: '{ocr_name}', QR: '{qr_name}'")
            
            # 2. DOB check (THIS SECTION IS COMMENTED OUT AS REQUESTED)
            # if ocr_dob and qr_dob and normalize_date(ocr_dob) != normalize_date(qr_dob):
            #      analysis_results["fraud_score"] += 3
            #      analysis_results["indicators"].append(f"🔴 HIGH: DOB Mismatch. OCR: '{ocr_dob}', QR: '{qr_dob}'")

            # 3. Gender check
            if ocr_gender and qr_gender:
                ocr_g = 'M' if ocr_gender.lower() == 'male' else ('F' if ocr_gender.lower() == 'female' else 'O')
                if ocr_g != qr_gender:
                    analysis_results["fraud_score"] += 3
                    analysis_results["indicators"].append(f"🔴 HIGH: Gender Mismatch. OCR: '{ocr_gender}', QR: '{qr_gender}'")

            # 4. Aadhaar last 4
            cleaned_ocr_aadhaar = re.sub(r'\s+', '', ocr_aadhaar_num)
            if cleaned_ocr_aadhaar and qr_aadhaar_last_4 and cleaned_ocr_aadhaar[-4:] != qr_aadhaar_last_4:
                 analysis_results["fraud_score"] += 3
                 analysis_results["indicators"].append(f"🔴 HIGH: Aadhaar Number Mismatch. OCR ends in: '{cleaned_ocr_aadhaar[-4:]}', QR ends in: '{qr_aadhaar_last_4}'")

    else:
        analysis_results["indicators"].append("⚪ INFO: QR Code check was disabled by the user.")

    # --- F: Final Assessment ---
    if analysis_results["fraud_score"] >= 3:
        analysis_results["assessment"] = "HIGH"
    elif analysis_results["fraud_score"] >= 1:
        analysis_results["assessment"] = "MODERATE"
    else:
        analysis_results["assessment"] = "LOW"
        if not any(ind.startswith("🔴") or ind.startswith("🟡") for ind in analysis_results["indicators"]):
             analysis_results["indicators"].append("✅ LOW: All checks passed.")

    return analysis_results


# --- 8. STREAMLIT UI ---

st.title("Aadhaar Card Fraud Detection 🔎")
st.markdown("Upload an image of an Aadhaar card to analyze it for signs of fraud.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙ Configuration")
    
    # IMPORTANT: Make sure 'best.pt' is in the same directory as this script
    MODEL_PATH = 'best.pt'
    
    aadhaar_image = st.file_uploader("Upload Aadhaar Image", type=["jpg", "jpeg", "png"])
    
    qr_toggle = st.toggle("Perform QR Code Verification", value=True, help="Scans the uploaded image for a Secure QR code and cross-references it.")
    
    analyze_button = st.button("Analyze Card", type="primary", use_container_width=True)

# --- Main Area for Results ---
if analyze_button:
    if not aadhaar_image:
        st.error("Please upload an Aadhaar Image to begin analysis.")
    else:
        # Load models
        custom_model = load_yolo_model(MODEL_PATH)
        general_model = load_general_model()
        
        if custom_model is None or general_model is None:
            st.error("Models could not be loaded. Please check the console for errors and ensure 'best.pt' is in the correct folder.")
        else:
            with st.spinner("🕵 Analyzing image... This may take a moment."):
                pil_image = Image.open(aadhaar_image)
                
                # Perform the full analysis
                results = perform_analysis(pil_image, qr_toggle, custom_model, general_model)
            
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
                    st.subheader("Annotated Image")
                    if results["annotated_image"]:
                        st.image(results["annotated_image"], caption="Detections on Image", use_container_width=True)
                    else:
                        st.image(pil_image, caption="Original Image", use_container_width=True)

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

                st.divider()

                col3, col4 = st.columns(2)
                
                with col3:
                    st.subheader("Extracted Text (OCR)")
                    st.info("This JSON shows the *cleaned* data after extraction.")
                    if results["ocr_data"]:
                        st.json(results["ocr_data"])
                    else:
                        st.info("No text could be extracted.")

                with col4:
                    st.subheader("Decoded QR Data")
                    if not qr_toggle:
                        st.info("QR check was disabled.")
                    elif "error" in results["qr_data"]:
                        st.error(results["qr_data"]["error"])
                    elif results["qr_data"]:
                        st.json(results["qr_data"])
                    else:
                        st.info("QR check was not performed or no data was found.")

# Instructions (Simplified)
st.info(
    "**How this works:**\n"
    "This tool performs a deep analysis of the Aadhaar card image through a 7-step process:\n"
    "* **1. Image Preparation:** The image is enhanced to improve text recognition.\n"
    "* **2. Data Location:** An AI model finds and draws boxes around key fields (Name, DOB, Aadhaar Number).\n"
    "* **3. Text Reading (OCR):** Software extracts the text from these fields.\n"
    "* **4. Error Correction:** Common scanning and reading errors in the text, especially in dates, are automatically fixed.\n"
    "* **5. Data Validation:** The extracted Name, DOB, and Gender are checked for correct format and logical consistency.\n"
    "* **6. Checksum Verification:** The 12-digit Aadhaar number is verified using the official Verhoeff algorithm to detect tampering.\n"
    "* **7. QR Code Cross-Check (Optional):** If enabled, the card's Secure QR code is decoded, and its data is compared against the text read from the front of the card.",
    icon="ℹ️"
)