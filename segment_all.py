# segment_all.py

import os
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
from unet import UNet

# Configuración
MODEL_PATH = "unet_model.pth"
INPUT_DIR = "./dataset_bc"
OUTPUT_DIR = "./dataset_bc_segmented"
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar modelo
model = UNet().to(DEVICE)
model.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# Buscar imágenes en todas las subcarpetas
for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(root, file)
            # Construir ruta de salida manteniendo estructura
            rel_dir = os.path.relpath(root, INPUT_DIR)
            save_subdir = os.path.join(OUTPUT_DIR, rel_dir)
            os.makedirs(save_subdir, exist_ok=True)
            save_path = os.path.join(save_subdir, file)

            # Cargar y preprocesar imagen
            img = Image.open(input_path).convert('RGB').resize(IMAGE_SIZE)
            tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(tensor)
                pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

            mask_img = Image.fromarray(pred_mask).convert('L').resize(IMAGE_SIZE)
            mask_img_rgb = Image.merge("RGB", (mask_img, mask_img, mask_img))

            # Combinar original y máscara lado a lado
            combined = Image.new('RGB', (IMAGE_SIZE[0]*2, IMAGE_SIZE[1]))
            combined.paste(img, (0, 0))
            combined.paste(mask_img_rgb, (IMAGE_SIZE[0], 0))

            combined.save(save_path)

print("Segmentación completada. Combinaciones guardadas en:", OUTPUT_DIR)