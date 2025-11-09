import os
import requests

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive direct download URLs
MODELS = {
    "best.pt": "https://drive.google.com/uc?export=download&id=14-7ql-EW4-6trjB7lJRCmKRFtIXj_rk8",
    "yolov8n.pt": "https://drive.google.com/uc?export=download&id=111vAQgZKO2JkDt52lJIUwEijr6wXXODf"
}

def ensure_models():
    """Downloads both YOLO models from Google Drive if missing, and returns their paths."""
    model_paths = {}
    
    for name, url in MODELS.items():
        model_path = os.path.join(MODEL_DIR, name)
        model_paths[name] = model_path

        if not os.path.exists(model_path):
            print(f"📥 Downloading {name} from Google Drive...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ {name} downloaded successfully!")
            except Exception as e:
                print(f"❌ Failed to download {name}: {e}")
        else:
            print(f"✅ {name} already exists locally.")

    return model_paths
