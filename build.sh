#!/bin/bash
echo "🚀 Starting build process on Render..."

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get update
apt-get install -y tesseract-ocr libzbar0 libgl1-mesa-glx

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Ensure backend/models directory exists
echo "📁 Setting up directory structure..."
mkdir -p backend/models
mkdir -p backend/uploads

# Copy YOLO model to correct location
if [ -f "yolov8n.pt" ]; then
    echo "✅ Copying YOLO model to backend/models/"
    cp yolov8n.pt backend/models/
else
    echo "❌ yolov8n.pt not found in root directory"
    echo "📁 Current files:"
    ls -la
fi

echo "✅ Build completed successfully!"