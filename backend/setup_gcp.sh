#!/bin/bash

# This script helps set up GCP credentials for the Synthetic Image Detector backend

# Check if the credentials file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-gcp-credentials-json>"
    echo "Example: $0 /path/to/service-account-key.json"
    exit 1
fi

# Check if the file exists
if [ ! -f "$1" ]; then
    echo "Error: Credentials file not found at $1"
    exit 1
fi

# Create the credentials directory if it doesn't exist
mkdir -p gcp-credentials

# Copy the credentials file to the gcp-credentials directory
cp "$1" gcp-credentials/service-account-key.json

echo "GCP credentials set up successfully!"
echo "The credentials file has been copied to gcp-credentials/service-account-key.json"
echo "Make sure to update the BUCKET_NAME and MODEL_PATH variables in app.py to point to your model in GCP storage." 