import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,  # inference uses trained weights
    in_channels=3,
    classes=1,
).to(device)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Load test image
img_path = "dataset/input/1.bmp"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))
image = image.astype(np.float32) / 255.0
image_tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).to(device)

# Predict mask
with torch.no_grad():
    pred = model(image_tensor)
    pred = torch.sigmoid(pred)
    pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()

# Convert to 0-255 image
pred_mask_img = (pred_mask * 255).astype(np.uint8)
cv2.imwrite("pred_mask.png", pred_mask_img)
print("Prediction saved as pred_mask.png")