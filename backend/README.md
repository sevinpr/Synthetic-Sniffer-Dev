# Synthetic Image Detector Backend

This is a Flask-based backend API for the Synthetic Image Detector application, which classifies images as either "GAN-generated" or "Real" using a PyTorch model.

## Setup Instructions

1. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```

2. **Set up Google Cloud Storage authentication**:

   - Create a service account in the Google Cloud Console with access to your storage bucket
   - Download the JSON key file for the service account
   - Set the environment variable to point to your credentials file:
     ```
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
     ```
   - Alternatively, place the JSON key file in the `/app/gcp-credentials` directory when running in Docker

3. **Configure the model path**:

   - Update the `BUCKET_NAME` and `MODEL_PATH` variables in `app.py` to point to your model in GCP storage
   - Example: `BUCKET_NAME = 'my-model-bucket'` and `MODEL_PATH = 'models/hde_model_complete.pth'`

4. **Customize model architecture**:

   - The current implementation provides a simplified CNN architecture in the `GanDetector` class
   - **Important**: You must update this class to match the exact architecture of your trained model
   - If your model was saved with `torch.save(model)` instead of `torch.save(model.state_dict())`, you'll need to modify the loading code

5. **Adjust preprocessing**:

   - Modify the `preprocess` transform in `app.py` to match the preprocessing used during your model training
   - The default preprocessing is for models trained on ImageNet

6. **Run the server**:
   ```
   python app.py
   ```
   - The server will run on http://localhost:5000

## API Endpoint

### POST /predict

Accepts an image file upload and returns a prediction.

**Request**:

- Method: POST
- Content-Type: multipart/form-data
- Form Parameter: 'image' (the image file)

**Response**:

```json
{
  "prediction": "GAN-generated", // or "Real"
  "confidence": 0.95 // confidence score between 0 and 1
}
```

## Troubleshooting

- **Model Architecture Issues**: If you get a shape mismatch error, ensure your `GanDetector` class matches exactly the architecture used when training the model
- **CUDA/CPU Issues**: The code loads the model to CPU by default. If using CUDA, modify the device in the `load_model` function
- **GCP Authentication Issues**: Ensure your service account has the necessary permissions to access the storage bucket
