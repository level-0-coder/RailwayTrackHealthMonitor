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
# print(device)
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
st.title("Railway Track Crack Segmentation")
st.write("Upload a Railway Track surface image.")

uploaded_file = st.file_uploader("Choose an image", type=["bmp", "jpg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    orig_h, orig_w = image.shape[:2]

    if st.button("Predict"):
        # Preprocess
        img_resized = cv2.resize(image, (256, 256))
        img_tensor = torch.tensor(img_resized.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
            mask = (pred > 0.5).float().squeeze().cpu().numpy()
            mask_img = (mask * 255).astype(np.uint8)

        # Resize mask back to original size
        mask_img = cv2.resize(mask_img, (orig_w, orig_h))

        # Overlay
        overlay = cv2.addWeighted(image, 0.7, cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR), 0.3, 0)

        # Show images
        st.subheader("Original Image")
        st.image(image, width=500)

        st.subheader("Predicted Crack Mask")
        st.image(mask_img, width=500, clamp=True)

        st.subheader("Overlay")
        st.image(overlay, width=500)