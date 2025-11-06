# backend/utils/ocr_utils.py
import io
import os
import platform
import pytesseract
import re
from PIL import Image, ImageEnhance, ImageFilter

# --- Configure Tesseract path based on environment ---
if platform.system() == "Windows":
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
        pytesseract.get_tesseract_version()
    except Exception as e:
        print(f"[ERROR] Tesseract not found at default path. Error: {e}")
elif platform.system() in ["Linux", "Darwin"]:
    if os.path.exists("/usr/bin/tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def pil_from_bytes(image_bytes):
    """Converts raw image bytes into a PIL image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def preprocess_for_ocr(image_pil):
    """Apply preprocessing to improve OCR quality."""
    gray = image_pil.convert("L")
    enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
    filtered = enhanced.filter(ImageFilter.SHARPEN)
    
    # Resize for better recognition
    width, height = filtered.width, filtered.height
    filtered = filtered.resize((int(width * 2.0), int(height * 2.0)), Image.Resampling.LANCZOS)
    
    return filtered

def ocr_text_with_config(image_pil, config="--psm 6"):
    """Extracts text from an image using Tesseract OCR with config."""
    text = pytesseract.image_to_string(image_pil, config=config)
    return text.strip()

def extract_aadhaar_fields(text):
    """Rule-based extraction of Aadhaar fields from OCR text."""
    uid = None
    name = None
    
    # Extract UID (12-digit number)
    digits = "".join(filter(str.isdigit, text))
    if len(digits) >= 12:
        uid = digits[:12]

    # Simple heuristic for name
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines:
        if any(keyword in line.lower() for keyword in ["name", "nam", "nane"]):
            # Remove the keyword and clean up
            clean_line = re.sub(r'^(name[:\-\s]*)', '', line, flags=re.I).strip()
            if len(clean_line) >= 3:
                name = clean_line
                break

    return {
        "uid": uid,
        "name": name
    }