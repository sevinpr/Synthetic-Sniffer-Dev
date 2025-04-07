import os
import torch
import torch.nn as nn
import torch.nn.functional as F # Needed for DecisionFusion and info_nce
import torch.fft             # Needed for FrequencyPathway
import torchvision.models as models # Needed for DRGANEncoder's ResNet
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from google.cloud import storage
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- START: COPY MODEL DEFINITIONS FROM JUPYTER NOTEBOOK ---
# Make sure all necessary classes are defined here

class DRGANEncoder(nn.Module):
    def __init__(self, latent_dim=128, freeze_backbone=True): # Use the ResNet version
        super(DRGANEncoder, self).__init__()
        # Example using ResNet50
        # Use pretrained=False if you don't have internet access in the backend
        # or if the weights were already included during training save
        # Use weights=None for newer torchvision versions if you don't want pretrained
        # For simplicity, assuming pretrained=True is fine here. Adjust if needed.
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) # Or weights=None
        # Remove the original classifier
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            # Freeze only if they were frozen during training
            # It's generally safer to load the state dict which includes requires_grad info
            # But explicitly setting it ensures consistency if loading on a different setup.
            # Assuming they WERE frozen during the training run based on the notebook code.
            for param in self.features.parameters():
                param.requires_grad = False
        # else: # Ensure gradients are enabled if they weren't frozen
        #     for param in self.features.parameters():
        #          param.requires_grad = True


        # Adjust the input dimension of the FC layer based on backbone output
        num_ftrs = backbone.fc.in_features # 2048 for ResNet50
        self.fc = nn.Linear(num_ftrs, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten features
        latent = self.fc(x)
        return latent

class DRGANDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(DRGANDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512*16*16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32->64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 64->128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),     # 128->256
            nn.Tanh()  # Assuming image pixels in [-1, 1] (due to Normalize(0.5, 0.5))
        )

    def forward(self, latent):
        x = self.fc(latent)
        x = x.view(x.size(0), 512, 16, 16)
        rec = self.deconv(x)
        return rec

class DRGAN(nn.Module):
    def __init__(self, latent_dim=128):
        super(DRGAN, self).__init__()
        self.encoder = DRGANEncoder(latent_dim=latent_dim)
        self.decoder = DRGANDecoder(latent_dim=latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # 2 classes: fake (0), real (1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        rec = self.decoder(latent)
        class_logits = self.classifier(latent)
        return latent, rec, class_logits

class SemanticPathway(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super(SemanticPathway, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, latent):
        return self.net(latent)

class FrequencyPathway(nn.Module):
    def __init__(self, image_size=256, out_dim=64):
        super(FrequencyPathway, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 8)) # Pool to a fixed size (8x8)
        fft_flat_dim = 8 * 8 # Size after pooling

        self.net = nn.Sequential(
            nn.Linear(fft_flat_dim, 128), # Intermediate layer
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim)
        )

    def forward(self, image):
        fft_coeffs = torch.fft.rfft2(image, norm="ortho")
        fft_mag = torch.abs(fft_coeffs)
        fft_mag_pooled_channel = fft_mag.mean(dim=1, keepdim=True)
        pooled_fft_mag = self.pool(fft_mag_pooled_channel)
        flat_fft = torch.flatten(pooled_fft_mag, 1)
        return self.net(flat_fft)

class BiologicalPathway(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super(BiologicalPathway, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, latent):
        return self.net(latent)

class ContrastiveRefinement(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super(ContrastiveRefinement, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, out_dim)
        )

    def forward(self, latent):
        projection = self.projector(latent)
        return projection

class NashAE(nn.Module):
    def __init__(self, in_dim=128, latent_dim=64):
        super(NashAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_dim),
            nn.ReLU(inplace=True)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, latent):
        z = self.encoder(latent)
        rec = self.decoder(z)
        d_score = self.discriminator(z)
        return z, rec, d_score

class DecisionFusion(nn.Module):
    def __init__(self, in_dims=[64, 64, 64, 64, 64], hidden_dim=64, out_dim=2):
        super(DecisionFusion, self).__init__()
        num_pathways = len(in_dims)
        total_in_dim = sum(in_dims)
        self.pathway_weights = nn.Parameter(torch.ones(num_pathways))
        self.fc = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, semantic, frequency, biological, contrastive, nash):
        features = [semantic, frequency, biological, contrastive, nash]
        batch_size = features[0].size(0)
        normalized_weights = F.softmax(self.pathway_weights, dim=0)
        weighted_features = []
        for i, feat in enumerate(features):
            weight = normalized_weights[i].view(1, 1).expand(batch_size, 1)
            weighted_features.append(feat * weight.expand_as(feat))
        fused_features = torch.cat(weighted_features, dim=1)
        return self.fc(fused_features)

    def get_weights(self):
        return F.softmax(self.pathway_weights, dim=0)


class HDE(nn.Module):
    def __init__(self, latent_dim=128, image_size=256):
        super(HDE, self).__init__()
        self.drgan = DRGAN(latent_dim=latent_dim)
        self.semantic = SemanticPathway(in_dim=latent_dim, out_dim=64)
        self.frequency = FrequencyPathway(image_size=image_size, out_dim=64)
        self.biological = BiologicalPathway(in_dim=latent_dim, out_dim=64)
        self.contrastive = ContrastiveRefinement(in_dim=latent_dim, out_dim=64)
        self.nash = NashAE(in_dim=latent_dim, latent_dim=64)
        self.fusion = DecisionFusion(in_dims=[64, 64, 64, 64, 64], out_dim=2)

    def forward(self, x):
        latent, rec, class_logits = self.drgan(x)
        semantic_feat = self.semantic(latent)
        frequency_feat = self.frequency(x)
        biological_feat = self.biological(latent)
        contrastive_feat = self.contrastive(latent)
        nash_z, nash_rec, nash_dscore = self.nash(latent)
        fused_logits = self.fusion(semantic_feat, frequency_feat, biological_feat, contrastive_feat, nash_z)

        # Return the dictionary, although we only need 'fused_logits' for prediction
        return {
            'latent': latent, 'drgan_rec': rec, 'drgan_class': class_logits,
            'semantic': semantic_feat, 'frequency': frequency_feat, 'biological': biological_feat,
            'contrastive': contrastive_feat, 'nash_z': nash_z, 'nash_rec': nash_rec,
            'nash_dscore': nash_dscore, 'fused_logits': fused_logits
        }

    def get_contrastive_projection(self, x_aug):
         with torch.no_grad():
             latent_aug = self.drgan.encoder(x_aug)
         contrastive_feat_aug = self.contrastive(latent_aug)
         return contrastive_feat_aug

# --- END: COPY MODEL DEFINITIONS ---


# --- Configuration ---
BUCKET_NAME = 'hde-model'  # Replace with your actual GCP bucket name
MODEL_PATH = 'hde_model_complete.pth'  # The path to the model file in your bucket
LATENT_DIM = 128             # Must match the trained model
IMAGE_SIZE = 256             # Must match the trained model
GCP_CREDENTIALS_PATH = 'synthetic-sniffer-cb226412de29.json'


# --- Load the CORRECT Model from GCP Bucket ---
def load_model():
    try:
        # Instantiate the HDE model
        model = HDE(latent_dim=LATENT_DIM, image_size=IMAGE_SIZE)
        
        # Set up GCP credentials
        if os.path.exists(GCP_CREDENTIALS_PATH):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCP_CREDENTIALS_PATH
            print(f"Using GCP credentials from: {GCP_CREDENTIALS_PATH}")
        else:
            print("Warning: GCP credentials file not found. Using default authentication method.")
            print("Make sure you have set up authentication via gcloud CLI or environment variables.")
        
        # Initialize GCP Storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_PATH)
        
        # Download the model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            blob.download_to_filename(temp_file.name)
            # Load the state dict from the temporary file
            model.load_state_dict(torch.load(temp_file.name, map_location=torch.device('cpu')))
            model.eval()  # Set to evaluation mode
            print("HDE Model loaded successfully from GCP bucket")
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        return model
    except Exception as e:
        print(f"Error loading HDE model from GCP bucket: {e}")
        # Print traceback for detailed debugging if needed
        import traceback
        traceback.print_exc()
        return None

model = load_model()

# --- Define Image Preprocessing (MATCH THE NOTEBOOK'S val_test_transforms) ---
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Resize to the model's expected input size
    transforms.ToTensor(),                       # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) # Use the SAME normalization as training/validation
])

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure model loaded correctly
    if model is None:
         return jsonify({'error': 'Model not loaded, check server logs.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Get the image from the request
        image_file = request.files['image']
        image_bytes = image_file.read()
        # Ensure image is in RGB format
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess the image
        image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            # Get the final classification logits from the fusion layer
            logits = outputs['fused_logits']
            # Get probabilities and predicted class index
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        # Interpret results (Based on notebook: 0 = fake, 1 = real)
        predicted_label = predicted_idx.item()
        result = "Real" if predicted_label == 1 else "Fake" # "Fake" corresponds to GAN-generated

        return jsonify({
            'prediction': result,
            'predicted_class_index': predicted_label, # Optionally return the index
            'confidence': float(confidence.item())
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc() # Print full traceback to server logs
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Check if model loaded before starting server
    if model is None:
        print("Exiting: Model could not be loaded.")
    else:
        print("Starting Flask server...")
        # Use host='0.0.0.0' to make it accessible externally
        # debug=True is helpful for development, consider turning off for production
        app.run(debug=True, host='0.0.0.0', port=5000)