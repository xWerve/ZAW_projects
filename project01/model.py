import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import os

TARGET_SIZE = (256, 256)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cable_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)
model.eval()

def predict(image: np.ndarray) -> np.ndarray:
    original_h, original_w = image.shape[:2]


    img_pil = Image.fromarray(image).convert('RGB').resize(TARGET_SIZE, resample=Image.BILINEAR)
    img_np = np.array(img_pil).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device).float()

    with torch.no_grad():
        output = model(img_tensor)
        mask_prob = torch.sigmoid(output)
        mask = mask_prob.squeeze().cpu().numpy()

    binary_mask = (mask > 0.5).astype(np.uint8) * 255

    if np.sum(binary_mask > 0) < 50:
        binary_mask = np.zeros_like(binary_mask)

    mask_pil = Image.fromarray(binary_mask).resize((original_w, original_h), resample=Image.NEAREST)

    return np.array(mask_pil)