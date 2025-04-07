@echo off
echo Setting up Cloud Build for Synthetic Sniffer...

REM Check if gcloud is installed
where gcloud >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Google Cloud SDK is not installed or not in PATH.
    echo Please install it from https://cloud.google.com/sdk/docs/install
    exit /b 1
)

REM Check if user is authenticated
gcloud auth list --filter=status:ACTIVE --format="value(account)" > temp.txt
set /p ACTIVE_ACCOUNT=<temp.txt
del temp.txt

if "%ACTIVE_ACCOUNT%"=="" (
    echo You are not authenticated with Google Cloud.
    echo Please run: gcloud auth login
    exit /b 1
)

REM Get project ID
echo Enter your Google Cloud Project ID:
set /p PROJECT_ID=

REM Enable required APIs
echo Enabling required APIs...
gcloud services enable cloudbuild.googleapis.com --project=%PROJECT_ID%
gcloud services enable containerregistry.googleapis.com --project=%PROJECT_ID%
gcloud services enable run.googleapis.com --project=%PROJECT_ID%
gcloud services enable secretmanager.googleapis.com --project=%PROJECT_ID%

REM Create secret for GCP credentials
echo.
echo Do you want to create a secret for your GCP credentials? (Y/N)
set /p CREATE_SECRET=

if /i "%CREATE_SECRET%"=="Y" (
    echo.
    echo Enter the path to your service account key JSON file:
    set /p KEY_FILE=
    
    if not exist "%KEY_FILE%" (
        echo Error: File not found at %KEY_FILE%
        exit /b 1
    )
    
    echo Creating secret...
    gcloud secrets create synthetic-sniffer-credentials --data-file="%KEY_FILE%" --project=%PROJECT_ID%
    
    echo Granting access to Cloud Build service account...
    for /f "tokens=*" %%a in ('gcloud projects describe %PROJECT_ID% --format="value(projectNumber)"') do set PROJECT_NUMBER=%%a
    gcloud secrets add-iam-policy-binding synthetic-sniffer-credentials --member=serviceAccount:%PROJECT_NUMBER%@cloudbuild.gserviceaccount.com --role=roles/secretmanager.secretAccessor --project=%PROJECT_ID%
)

REM Create Cloud Build trigger
echo.
echo Do you want to create a Cloud Build trigger? (Y/N)
set /p CREATE_TRIGGER=

if /i "%CREATE_TRIGGER%"=="Y" (
    echo.
    echo Enter your repository name (e.g., username/repo):
    set /p REPO_NAME=
    
    echo Enter your branch name (e.g., main):
    set /p BRANCH_NAME=
    
    echo Creating Cloud Build trigger...
    gcloud builds triggers create github --repo-name=%REPO_NAME% --branch-pattern=^%BRANCH_NAME%$ --build-config=cloudbuild.yaml --project=%PROJECT_ID%
)

echo.
echo Setup complete! You can now push your code to trigger a build.
echo.
echo Next steps:
echo 1. Push your code to your repository
echo 2. Check the Cloud Build console to monitor your build
echo 3. If deploying to Cloud Run, uncomment the deployment step in cloudbuild.yaml 