[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aadhaar-fraud-detection-ai.streamlit.app)


🔎 Aadhaar Card Fraud Detection System

This project implements a comprehensive system for analyzing uploaded Aadhaar card images (front and back) to detect potential signs of fraud. The system utilizes Computer Vision (YOLOv8) for field localization, Optical Character Recognition (OCR) for text extraction, and advanced cryptographic checks (Verhoeff checksum, Secure QR code decoding) for data validation and cross-referencing.

🌟 Key Features

🧠 Object Detection: Uses a fine-tuned YOLOv8 model (best.pt) to accurately localize key fields (Name, DoB, Aadhaar Number, Photo) on the card front.

✅ Data Validation: Implements the Verhoeff algorithm to mathematically validate the 12-digit Aadhaar number checksum.

🔐 Secure QR Code Verification: Decodes the Secure QR code on the back of the card to cross-reference personal data (Name, DoB, last 4 digits of Aadhaar) against the OCR extracted text.

👤 Face Presence Check: Uses a general YOLO model to confirm the presence of a 'person' object within the card's photo area.

💻 Streamlit Interface: Provides an intuitive web interface for easy file uploads and result visualization.

🛠️ Installation and Setup
Prerequisites

You must have Python 3.8+ installed. Additionally, this project requires the Tesseract OCR engine to be installed system-wide.

Install Tesseract OCR:
Download and install the Tesseract executable for your operating system (Windows/macOS/Linux).
Note for Windows Users: Update the path to your Tesseract executable in the script:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

Project Setup

Clone the repository:
git clone https://github.com/your-username/aadhaar-fraud-detection.git
cd aadhaar-fraud-detection

Create and activate a virtual environment:
python -m venv venv
Windows: .\venv\Scripts\activate
Linux/macOS: source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Model Files:
Ensure you have the following trained model files in the root directory:

best.pt → Custom YOLO model for field detection

yolov8n.pt → General YOLO model (auto-downloaded by Ultralytics)

🚀 Usage

Run the Streamlit web application from the project root directory:
streamlit run app.py
Then open the local URL provided by Streamlit (e.g., http://localhost:8501) to use the web interface.

🧩 Analysis Pipeline

The perform_analysis() function executes the following steps:

Object Detection (Front): custom_model (best.pt) identifies and crops Name, DoB, and Aadhaar Number fields.

OCR Extraction: Cropped fields are preprocessed and sent to Tesseract to extract text.

Checksum Validation: The extracted 12-digit Aadhaar number is validated using the Verhoeff checksum algorithm.

Face Detection: general_model (yolov8n.pt) confirms that a face is present on the card photo.

QR Code Decoding (Back): The Secure QR code is decoded using the pyaadhaar library.

Cross-Reference: Extracted OCR data (Name, DoB, Aadhaar last 4 digits) is compared with decoded QR data.

Fraud Scoring: A fraud score is computed based on checksum failures, data mismatches, and detection errors, resulting in a final risk assessment: Low Risk 🟢 | Moderate Risk 🟡 | High Risk 🔴

📦 Dependencies

Refer to the requirements.txt file for a complete list of dependencies.

📘 Author

Developed by Maddineni Kinshuk
🎓 B.Tech CSE | VIT-AP
💼 Project: Aadhaar Fraud Detection System

🧾 License

This project is licensed under the MIT License. See the LICENSE file for more information.
