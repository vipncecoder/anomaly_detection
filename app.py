import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import Autoencoder  # or paste model class above this line

# Load model
model = Autoencoder()
model.load_state_dict(torch.load("autoencoder_with_resnet_deep_features.pth", map_location=torch.device('cpu')))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Threshold for anomaly detection
THRESHOLD = 0.01

# Title
st.title("Anomaly Detection - Defective or Not")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 256, 256)

    # Run through autoencoder
    with torch.no_grad():
        output = model(input_tensor)
        loss = torch.mean((input_tensor - output) ** 2).item()

    # Show result
    st.write(f"Reconstruction Error: {loss:.5f}")
    if loss > THRESHOLD:
        st.error("❌ Defective")
    else:
        st.success("✅ Non-Defective")