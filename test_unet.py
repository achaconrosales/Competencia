# test_unet.py

import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF

from unet import UNet

# -------------- CONFIGURACIÓN --------------------
MODEL_PATH = "unet_sporas.pth"
IMAGE_DIR = "./validacion"
SAVE_DIR = "./predictions"
IMAGE_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear directorio de salida si no existe
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------- CARGAR MODELO --------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Modelo cargado")

# -------------- PREDICCIÓN -----------------------
def predict_mask(image_path):
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize(IMAGE_SIZE)
    input_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    return mask

# -------------- TESTEO --------------------------
image_paths = [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR) if fname.endswith('.png')]
for path in image_paths:
    filename = os.path.basename(path)
    mask = predict_mask(path)

    # Cargar imagen original y redimensionar la máscara a RGB
    original_img = Image.open(path).convert('RGB').resize(IMAGE_SIZE)
    mask_img = Image.fromarray(mask * 255).convert('L').resize(IMAGE_SIZE)
    mask_img_rgb = Image.merge("RGB", (mask_img, mask_img, mask_img))

    # Concatenar imagen original y máscara predicha horizontalmente
    combined = Image.new('RGB', (IMAGE_SIZE[0]*2, IMAGE_SIZE[1]))
    combined.paste(original_img, (0, 0))
    combined.paste(mask_img_rgb, (IMAGE_SIZE[0], 0))

    # Guardar imagen combinada
    combined.save(os.path.join(SAVE_DIR, f"combined_{filename}"), dpi = (100, 100))

