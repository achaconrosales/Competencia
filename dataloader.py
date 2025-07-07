import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import numpy as np
from torchvision import transforms

class SporeSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        # Solo archivos que tengan máscara correspondiente
        self.image_filenames = [
            f for f in sorted(os.listdir(self.image_dir))
            if os.path.isfile(os.path.join(self.mask_dir, f))
        ]
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Cargar la imagen (convertir a RGB para asegurar 3 canales)
        image = Image.open(img_path).convert("RGB")
        # Cargar la máscara (convertir a escala de grises para que sea 1 canal)
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convertir la máscara a un tensor de tipo 'long' si tus píxeles son IDs de clase (0, 1, 2, etc.)
        # Esto es común para CrossEntropyLoss.
        #mask = torch.from_numpy(np.array(mask)).long()

        # Si tus máscaras son binarias (0 o 1) y usas BCEWithLogitsLoss, podrías necesitar:
        mask = torch.from_numpy(np.array(mask)).float()
        # Asegúrate de que la máscara tenga la dimensión de canal si tu modelo la espera (ej. [1, H, W])
        # Si ToTensor ya se encargó de esto, no necesitas `unsqueeze`. Si no, tal vez sí:
        # mask = mask.unsqueeze(0) # Añade una dimensión de canal si no la tiene (e.g., de [H, W] a [1, H, W])

        return image, mask

def get_dataloaders(root_dir, batch_size, image_transform=None, mask_transform=None, train_split=0.8):
    """
    Genera DataLoaders de entrenamiento y validación.

    Args:
        root_dir (str): Ruta a tu carpeta de dataset (ej. 'path/to/tu/dataset').
                        Debe contener las subcarpetas 'images' y 'masks'.
        batch_size (int): El tamaño de lote deseado.
        image_transform (callable, opcional): Transformación a aplicar a las imágenes.
        mask_transform (callable, opcional): Transformación a aplicar a las máscaras.
        train_split (float): Proporción del dataset a usar para entrenamiento (0.0 a 1.0).

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    dataset = SporeSegmentationDataset(root_dir=root_dir)

    # Si se proporcionan transformaciones separadas, úsalas en __getitem__
    # Para simplicidad en este ejemplo, las aplicamos directamente en __init__
    # o asumiendo que un solo 'transform' manejará ambas.
    # Una mejor práctica para transforms que modifican la geometría (e.g., aleatorios)
    # es aplicarlos de forma coordinada a la imagen y a la máscara dentro de __getitem__.
    # Aquí, asumimos que 'transform' en SporeSegmentationDataset.init es común para ambos.

    # Calcular los tamaños de los conjuntos
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    # Dividir el dataset aleatoriamente
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Opcional: Asignar transformaciones a los sub-datasets si son diferentes
    # En este diseño, la transformación se pasa al constructor original del Dataset.
    # Si quieres transforms diferentes para train/val (ej. más aumentos para train),
    # puedes clonar y modificar los datasets o pasar transforms al constructor
    # de SporeSegmentationDataset directamente al crearlo para cada split (más complejo).

    # Crear DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # num_workers para carga más rápida
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader

if __name__ == '__main__':
    # --- Ejemplo de Uso ---
    # 1. Crear carpetas de dataset de ejemplo si no existen
    # En un escenario real, reemplazarías 'dataset' con la ruta real a tu dataset.
    dummy_dataset_path = 'dataset'
    os.makedirs(os.path.join(dummy_dataset_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dummy_dataset_path, 'masks'), exist_ok=True)

    # Crear algunas imágenes y máscaras de ejemplo (20 en total)
    for i in range(20):
        dummy_image = Image.new('RGB', (128, 128), color = (i*10, i*5, i*2))
        dummy_mask = Image.new('L', (128, 128), color = (i % 2) * 255) # Máscara binaria simple (0 o 255)
        dummy_image.save(os.path.join(dummy_dataset_path, 'images', f'image_{i:02d}.png'))
        dummy_mask.save(os.path.join(dummy_dataset_path, 'masks', f'image_{i:02d}.png'))

    # 2. Definir tus transformaciones
    # Para U-Net, querrás redimensionar tus imágenes a un tamaño fijo (ej. 256x256).
    # ToTensor convierte la imagen PIL a un Tensor de PyTorch y normaliza los píxeles a [0, 1].
    # Normalize se usa comúnmente para imágenes RGB con medias y desviaciones estándar predefinidas.
    # Para las máscaras, usualmente solo necesitas Resize y ToTensor (sin Normalize).
    common_transform = transforms.Compose([
        transforms.Resize((256, 256)), # Redimensionar imagen y máscara
        transforms.ToTensor(),         # Convierte PIL Image a Tensor de PyTorch
    ])

    # Si usas Normalize para las imágenes, aplícala solo a la imagen DESPUÉS de ToTensor.
    # La máscara no se normaliza de esta manera. Podrías pasar dos transforms diferentes
    # a la clase SporeSegmentationDataset, o aplicarlas condicionalmente dentro de __getitem__.
    # Por simplicidad aquí, common_transform se aplica a ambos.

    # 3. Obtener tus dataloaders
    batch_size = 4
    train_dl, val_dl = get_dataloaders(
        root_dir=dummy_dataset_path,
        batch_size=batch_size,
        image_transform=common_transform, # Aquí pasarías la transformación para imágenes
        mask_transform=common_transform   # Y aquí la transformación para máscaras
    )

    print(f"Número de muestras de entrenamiento: {len(train_dl.dataset)}")
    print(f"Número de muestras de validación: {len(val_dl.dataset)}")
    print(f"Número de lotes de entrenamiento: {len(train_dl)}")
    print(f"Número de lotes de validación: {len(val_dl)}")

    # 4. Iterar a través de un lote para ver la salida
    print("\n--- Lote de ejemplo del Dataloader de Entrenamiento ---")
    for images, masks in train_dl:
        print(f"Forma del lote de imágenes: {images.shape}") # Debería ser [batch_size, C, H, W]
        print(f"Forma del lote de máscaras: {masks.shape}")   # Debería ser [batch_size, H, W] (para máscaras de ID)
        break # Tomar solo un lote

    print("\n--- Lote de ejemplo del Dataloader de Validación ---")
    for images, masks in val_dl:
        print(f"Forma del lote de imágenes: {images.shape}")
        print(f"Forma del lote de máscaras: {masks.shape}")
        break