# Setting Up Cloud Build for Synthetic Sniffer

This guide explains how to connect your Synthetic Sniffer project to Google Cloud Build for automated building and deployment.

## Prerequisites

1. A Google Cloud Platform account
2. The Google Cloud SDK installed on your machine
3. Docker installed on your machine (for local testing)
4. Your project code pushed to a Git repository (GitHub, GitLab, or Cloud Source Repositories)

## Step 1: Enable Required APIs

Run the following commands to enable the necessary Google Cloud APIs:

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com  # If deploying to Cloud Run
gcloud services enable secretmanager.googleapis.com  # For storing credentials
```

## Step 2: Set Up Cloud Build Trigger

### Option 1: Using the Google Cloud Console

1. Go to the [Cloud Build Triggers page](https://console.cloud.google.com/cloud-build/triggers)
2. Click "Create Trigger"
3. Connect your repository (GitHub, GitLab, or Cloud Source Repositories)
4. Configure the trigger:
   - Name: `synthetic-sniffer-build`
   - Event: `Push to a branch`
   - Source: Your repository
   - Branch: `^main$` (or your main branch name)
   - Configuration: `Cloud Build configuration file (yaml or json)`
   - Cloud Build configuration file location: `/cloudbuild.yaml`
5. Click "Create"

### Option 2: Using the gcloud CLI

```bash
gcloud builds triggers create github \
  --repo-name=YOUR_REPO_NAME \
  --branch-pattern=^main$ \
  --build-config=cloudbuild.yaml
```

## Step 3: Set Up Secret Management for GCP Credentials

For security, you should store your GCP service account credentials as a secret in Cloud Build:

1. Create a secret in Secret Manager:

```bash
gcloud secrets create synthetic-sniffer-credentials --data-file=path/to/your/service-account-key.json
```

2. Grant Cloud Build access to the secret:

```bash
gcloud secrets add-iam-policy-binding synthetic-sniffer-credentials \
  --member=serviceAccount:YOUR_PROJECT_NUMBER@cloudbuild.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

3. Update the `cloudbuild.yaml` file to use the secret:

```yaml
steps:
  # Create a directory for credentials
  - name: "gcr.io/cloud-builders/gcloud"
    entrypoint: "bash"
    args:
      - "-c"
      - |
        mkdir -p /workspace/backend/gcp-credentials
        echo "$$SECRET" > /workspace/backend/gcp-credentials/service-account-key.json
    secretEnv: ["SECRET"]

  # Build the Docker image
  - name: "gcr.io/cloud-builders/docker"
    args: ["build", "-t", "gcr.io/$PROJECT_ID/synthetic-sniffer", "."]

  # Push the image to Container Registry
  - name: "gcr.io/cloud-builders/docker"
    args: ["push", "gcr.io/$PROJECT_ID/synthetic-sniffer"]

availableSecrets:
  secretManager:
    - versionName: projects/$PROJECT_ID/secrets/synthetic-sniffer-credentials/versions/latest
      env: "SECRET"

images:
  - "gcr.io/$PROJECT_ID/synthetic-sniffer"

# Explicitly specify that we want to use Docker for building
options:
  machineType: "E2_HIGHCPU_8"
  logging: CLOUD_LOGGING_ONLY
```

## Step 4: Test Your Cloud Build Configuration

1. Push your changes to your repository
2. Go to the [Cloud Build History page](https://console.cloud.google.com/cloud-build/builds)
3. Check the status of your build

## Step 5: Deploy to Cloud Run (Optional)

If you want to deploy your application to Cloud Run, uncomment the deployment step in the `cloudbuild.yaml` file and ensure you have the necessary permissions.

## Troubleshooting

### Common Errors

#### "invalid app path 'main': evaluate symlink: lstat main: no such file or directory"

This error occurs when Cloud Build tries to use Cloud Native Buildpacks instead of Docker. To fix this:

1. Make sure your `cloudbuild.yaml` file explicitly uses Docker for building:

   ```yaml
   steps:
     - name: "gcr.io/cloud-builders/docker"
       args: ["build", "-t", "gcr.io/$PROJECT_ID/synthetic-sniffer", "."]
   ```

2. Add the `options` section to your `cloudbuild.yaml` file:

   ```yaml
   options:
     machineType: "E2_HIGHCPU_8"
     logging: CLOUD_LOGGING_ONLY
   ```

3. Ensure you have a valid Dockerfile in your repository root.

#### "Build failed: Docker build failed"

If your Docker build fails:

1. Test your Dockerfile locally first:

   ```bash
   docker build -t synthetic-sniffer .
   ```

2. Check the build logs for specific errors.

3. Make sure all required files are included in your repository and not ignored by `.dockerignore`.

#### "Authentication Issues"

If you encounter authentication issues:

1. Ensure your service account has the necessary permissions.
2. Check that your secret is correctly set up in Secret Manager.
3. Verify that Cloud Build has access to the secret.

## Additional Resources

- [Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Container Registry Documentation](https://cloud.google.com/container-registry/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Docker Documentation](https://docs.docker.com/)
