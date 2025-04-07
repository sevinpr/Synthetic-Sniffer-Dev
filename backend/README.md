# Synthetic Image Detector Backend

This is a Flask-based backend API for the Synthetic Image Detector application, which classifies images as either "GAN-generated" or "Real" using a PyTorch model.

## Setup Instructions

1. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```

2. **Prepare your PyTorch model**:

   - Place your trained `.pth` model file in the backend directory
   - By default, the app expects the file to be named `model.pth`
   - You can modify the `MODEL_PATH` variable in `app.py` if your model has a different name

3. **Customize model architecture**:

   - The current implementation provides a simplified CNN architecture in the `GanDetector` class
   - **Important**: You must update this class to match the exact architecture of your trained model
   - If your model was saved with `torch.save(model)` instead of `torch.save(model.state_dict())`, you'll need to modify the loading code

4. **Adjust preprocessing**:

   - Modify the `preprocess` transform in `app.py` to match the preprocessing used during your model training
   - The default preprocessing is for models trained on ImageNet

5. **Run the server**:
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
