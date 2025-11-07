# Use official Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libzbar0 \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary folders
RUN mkdir -p backend/models backend/uploads /tmp/uploads

# Expose port
EXPOSE 5000

# Start app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
