# backend/utils/verification_rules.py
import re
from datetime import datetime

# Verhoeff tables
d_table = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],
    [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],
    [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0]
]

p_table = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],
    [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],
    [7,0,4,6,9,1,3,2,5,8]
]

def verhoeff_validate(num):
    """Return True if Aadhaar passes Verhoeff checksum validation"""
    try:
        c = 0
        num = str(num)[::-1]
        for i, item in enumerate(num):
            c = d_table[c][p_table[i % 8][int(item)]]
        return c == 0
    except (ValueError, TypeError):
        return False

def validate_aadhaar_number(aadhaar_number):
    if not aadhaar_number:
        return "Missing"
    aadhaar_number = re.sub(r'\s+', '', str(aadhaar_number))
    if not re.fullmatch(r'\d{12}', aadhaar_number):
        return "Invalid (must be 12 digits)"
    if not verhoeff_validate(aadhaar_number):
        return "Invalid (checksum failed – possible tampering)"
    return "Valid"

def validate_name(name):
    if not name:
        return "Missing"
    if not re.fullmatch(r"[A-Za-z. ]{3,50}", str(name).strip()):
        return "Invalid (special characters or too short/long)"
    return "Valid"

def validate_dob(dob):
    if not dob:
        return "Missing"
    try:
        if len(str(dob)) == 4: # Case for "Year of Birth"
            year = int(dob)
            current_year = datetime.now().year
            if 1900 <= year <= current_year:
                return "Valid"
            else:
                return "Invalid (year out of range)"
        else:
            # Requires clean DD/MM/YYYY format
            parsed = datetime.strptime(str(dob), "%d/%m/%Y")
            if parsed > datetime.now():
                return "Invalid (future date)"
            return "Valid"
    except ValueError:
        return "Invalid (wrong format or impossible date)"

def validate_gender(gender):
    if not gender:
        return "Missing"
    gender_str = str(gender).lower()
    if gender_str not in ["male", "female", "m", "f"]:
        return "Invalid (must be Male/Female)"
    return "Valid"

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