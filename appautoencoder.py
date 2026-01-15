import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import cv2

# ---------------------------
# Define Autoencoder class
# ---------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    model = Autoencoder()
    checkpoint = torch.load("model/simple_autoencoder_l2_loss.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

model = load_model()

# ---------------------------
# Image Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------
# Anomaly Score Calculation
# ---------------------------
def compute_anomaly_score(input_tensor, recon_tensor):
    return ((input_tensor - recon_tensor) ** 2).mean().item()

# ---------------------------
# Heatmap Generation
# ---------------------------
def generate_heatmap(input_tensor, recon_tensor):
    error_map = ((input_tensor - recon_tensor) ** 2).mean(dim=1).squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.imshow(error_map, cmap='jet')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    return buf

# ---------------------------
# Draw error mask over original image
# ---------------------------
def overlay_error_mask(original_img, input_tensor, recon_tensor):
    input_np = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    recon_np = recon_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    diff = np.abs(input_np - recon_np).mean(axis=2)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    original_cv = np.array(original_img.resize((224, 224)))[:, :, ::-1]  # RGB to BGR
    overlay = cv2.addWeighted(original_cv, 0.6, heatmap, 0.4, 0)
    return overlay

# ---------------------------
# Threshold Calibration
# ---------------------------
@st.cache_resource
def calibrate_threshold(val_folder="val", percentile=65):
    val_scores = []
    for fname in os.listdir(val_folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(val_folder, fname)).convert('RGB')
            input_tensor = transform(img).unsqueeze(0)
            recon_tensor = model(input_tensor)
            score = compute_anomaly_score(input_tensor, recon_tensor)
            val_scores.append(score)
    threshold = np.percentile(val_scores, percentile)
    return threshold

# ---------------------------
# Streamlit App
# ---------------------------
st.title("🔍 Anomaly Detection using Autoencoder")

# Auto-calibrated threshold
threshold = calibrate_threshold()
st.info(f"📊 Auto-calibrated Threshold: {threshold:.6f}")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    recon_tensor = model(input_tensor)

    # Anomaly score
    score = compute_anomaly_score(input_tensor, recon_tensor)
    st.metric("Anomaly Score", f"{score:.6f}")

    # Classification
    if score >= threshold:
        st.error("🔴 Defective Part Detected")
    else:
        st.success("🟢 Part is Good")

    # Show heatmap
    st.subheader("🔥 Anomaly Heatmap")
    heatmap_buf = generate_heatmap(input_tensor, recon_tensor)
    st.image(heatmap_buf, caption='Error Heatmap', use_column_width=True)

    # Show overlay
    st.subheader("📸 Overlay of Defective Areas")
    overlay_img = overlay_error_mask(image, input_tensor, recon_tensor)
    st.image(overlay_img, channels="BGR", caption="Error Overlay", use_column_width=True)


# 

