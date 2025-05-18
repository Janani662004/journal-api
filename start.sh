#!/bin/bash

echo "Downloading model files from Google Drive..."

# Replace with your actual file IDs!
gdown --id 1JfP8om_C1yP-zFWF5-QsGlCw4X99wsko -O reduced_xlnet.zip
gdown --id 1vidNSlVwQOFMSYhXa0JkVNvTc-404G3h -O wav2vec2.zip

# Unzip the files
unzip reduced_xlnet.zip
unzip wav2vec2.zip

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI
uvicorn main:app --host 0.0.0.0 --port 10000
