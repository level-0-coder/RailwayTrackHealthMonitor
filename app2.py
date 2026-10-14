import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image

# -----------------------------
# Load Model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Metal Crack Segmentation")
st.write("Upload a metal surface image and click 'Predict' to see the crack mask.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["bmp", "jpg", "png"])

# Only show the button if a file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.subheader("Original Image")
    st.image(image_np, width=500)

    # Predict button
    if st.button("Predict"):
        orig_h, orig_w = image_np.shape[:2]

        # Preprocess
        img_resized = cv2.resize(image_np, (256, 256))
        img_tensor = torch.tensor(img_resized.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
            mask = (pred > 0.5).float().squeeze().cpu().numpy()
            mask_img = (mask * 255).astype(np.uint8)

        # Resize mask to original size
        mask_img = cv2.resize(mask_img, (orig_w, orig_h))

        # Overlay
        overlay = cv2.addWeighted(image_np, 0.7, cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR), 0.3, 0)

        # Display results
        st.subheader("Predicted Crack Mask")
        st.image(mask_img, width=500, clamp=True)

        st.subheader("Overlay")
        st.image(overlay, width=500)