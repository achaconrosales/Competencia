# test_unet.py

import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF

from dataloader import SporeDataModule  # Asegúrate de que este archivo exista y sea accesible
from unet import UNet  # Asegúrate de que este archivo exista y sea accesible

# -------------- CONFIGURACIÓN --------------------
MODEL_PATH = "unet_model.pth"
IMAGE_DIR = './dataset'
SAVE_DIR = "./predictions"  # Cambiado para guardar los plots
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Crear directorio de salida para los plots si no existe
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------- CARGAR MODELO --------------------
model = UNet().to(DEVICE)
model.load(MODEL_PATH, map_location=DEVICE)  # Usar el método load definido en tu clase UNet
model.eval()
 
# -------------- CARGAR DATOS DE VALIDACIÓN -------
# Asegúrate de que batch_size sea 1 para poder acceder a las muestras individuales
data_module = SporeDataModule(IMAGE_DIR, image_size=IMAGE_SIZE, batch_size=1, verbose=False)
_, val_loader = data_module.get_dataloaders()

# -------------- PREDICCIÓN Y GENERACIÓN DE PLOTS -----------
images_to_plot = []
masks_gt_to_plot = []
masks_pred_to_plot = []
plot_counter = 0

for idx, (image, mask) in enumerate(val_loader):
    # Procesar la imagen para la predicción
    image = image.to(DEVICE)
    with torch.no_grad():
        output = model(image)
        # Asumiendo salida binaria (1 canal, sigmoid)
        pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)



    original_path = data_module.val_images[idx]
    original_img = Image.open(original_path).convert('RGB').resize(IMAGE_SIZE)
    mask_img = Image.fromarray(pred_mask * 255).convert('L').resize(IMAGE_SIZE)
    mask_img_rgb = Image.merge("RGB", (mask_img, mask_img, mask_img))

    mask_gt_np = mask.squeeze().cpu().numpy() # Eliminar dim de batch y canal

    images_to_plot.append(original_img)
    masks_gt_to_plot.append(mask_gt_np)
    masks_pred_to_plot.append(mask_img_rgb)

    # Generar el plot cada 3 muestras
    if (idx + 1) % 3 == 0:
        fig, axes = plt.subplots(3, 3, figsize=(9, 9)) # 3 filas, 3 columnas (Img, GT, Pred)
        
        for i in range(3):
            # Imagen Original
            axes[i, 0].imshow(images_to_plot[i])
            axes[i, 0].set_title(f'Original {plot_counter * 3 + i}', fontsize=25)
            axes[i, 0].axis('off')

            # Máscara Ground Truth
            axes[i, 1].imshow(masks_gt_to_plot[i], cmap='gray') # Asumiendo máscara binaria
            axes[i, 1].set_title(f'Ground Truth {plot_counter * 3 + i}', fontsize=25)
            axes[i, 1].axis('off')

            # Máscara Predicha
            axes[i, 2].imshow(masks_pred_to_plot[i], cmap='gray') # Asumiendo máscara binaria
            axes[i, 2].set_title(f'Predicted {plot_counter * 3 + i}', fontsize=25 )
            axes[i, 2].axis('off')

        plt.tight_layout()
        plot_filename = os.path.join(SAVE_DIR, f"predictions_batch_{plot_counter}.png")
        plt.savefig(plot_filename)
        plt.close(fig) # Cierra la figura para liberar memoria

        print(f"Plot guardado: {plot_filename}")

        # Limpiar las listas para el siguiente lote de plots
        images_to_plot = []
        masks_gt_to_plot = []
        masks_pred_to_plot = []
        plot_counter += 1

# Si hay muestras restantes que no completaron un lote de 3
if len(images_to_plot) > 0:
    fig, axes = plt.subplots(len(images_to_plot), 3, figsize=(9, len(images_to_plot) * 3))
    
    # Asegurarse de que axes sea siempre 2D, incluso si solo hay 1 fila
    if len(images_to_plot) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(len(images_to_plot)):
        axes[i, 0].imshow(images_to_plot[i])
        axes[i, 0].set_title(f'Original {plot_counter * 3 + i}', fontsize=25)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masks_gt_to_plot[i], cmap='gray')
        axes[i, 1].set_title(f'Ground Truth {plot_counter * 3 + i}', fontsize=25)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(masks_pred_to_plot[i], cmap='gray')
        axes[i, 2].set_title(f'Predicted {plot_counter * 3 + i}', fontsize=25)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plot_filename = os.path.join(SAVE_DIR, f"predictions_batch_final_{plot_counter}.png")
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Plot final guardado: {plot_filename}")

print("Visualización de predicciones completada.")