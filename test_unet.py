# test_unet.py

import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF

from dataloader import SporeDataModule
from unet import UNet

# -------------- CONFIGURACIÓN --------------------
MODEL_PATH = "unet_model.pth"
IMAGE_DIR = './dataset'
SAVE_DIR = "./predictions"
IMAGE_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear directorio de salida si no existe
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------- CARGAR MODELO --------------------
model = UNet().to(DEVICE)
model.load(MODEL_PATH, map_location=DEVICE)  # Usar el método load definido en tu clase
model.eval()
    
# -------------- CARGAR DATOS DE VALIDACIÓN -------
data_module = SporeDataModule(IMAGE_DIR, image_size=IMAGE_SIZE, batch_size=1, verbose=False)
_, val_loader = data_module.get_dataloaders()

# -------------- PREDICCIÓN Y GUARDADO -----------
for idx, (image, mask) in enumerate(val_loader):


    image = image.to(DEVICE)
    with torch.no_grad():
        output = model(image)
        pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Recuperar la imagen original del dataset
    # (asumiendo que val_loader usa SporeDataset y tiene .val_images)
    original_path = data_module.val_images[idx]
    original_img = Image.open(original_path).convert('RGB').resize(IMAGE_SIZE)
    mask_img = Image.fromarray(pred_mask * 255).convert('L').resize(IMAGE_SIZE)
    mask_img_rgb = Image.merge("RGB", (mask_img, mask_img, mask_img))

    combined = Image.new('RGB', (IMAGE_SIZE[0]*2, IMAGE_SIZE[1]))
    combined.paste(original_img, (0, 0))
    combined.paste(mask_img_rgb, (IMAGE_SIZE[0], 0))

    filename = os.path.basename(original_path)
    combined.save(os.path.join(SAVE_DIR, f"combined_{filename}"), dpi=(100, 100))