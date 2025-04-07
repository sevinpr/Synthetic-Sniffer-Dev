@echo off
REM This script helps set up GCP credentials for the Synthetic Image Detector backend

REM Check if the credentials file is provided
if "%~1"=="" (
    echo Usage: %0 ^<path-to-gcp-credentials-json^>
    echo Example: %0 C:\path\to\service-account-key.json
    exit /b 1
)

REM Check if the file exists
if not exist "%~1" (
    echo Error: Credentials file not found at %~1
    exit /b 1
)

REM Create the credentials directory if it doesn't exist
if not exist "gcp-credentials" mkdir gcp-credentials

REM Copy the credentials file to the gcp-credentials directory
copy "%~1" "gcp-credentials\service-account-key.json"

echo GCP credentials set up successfully!
echo The credentials file has been copied to gcp-credentials\service-account-key.json
echo Make sure to update the BUCKET_NAME and MODEL_PATH variables in app.py to point to your model in GCP storage. 