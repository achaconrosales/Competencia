import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import numpy as np
from torchvision import transforms

class SporeSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')

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

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.train:
            # --- Aumentos de datos coherentes entre imagen y máscara ---
            if torch.rand(1).item() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

            if torch.rand(1).item() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

            if torch.rand(1).item() > 0.5:
                angle = torch.randint(-30, 30, (1,)).item()
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)

            # ColorJitter solo en imagen (no en máscara)
            color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )
            image = color_jitter(image)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.from_numpy(np.array(mask)).float()
        return image, mask

def get_dataloaders(root_dir, batch_size, image_transform=None, mask_transform=None, train_split=0.8):
    full_dataset = SporeSegmentationDataset(root_dir=root_dir, transform=image_transform, train=True)

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Cambiar flag de augmentación para val
    val_dataset.dataset.train = False

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
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
    if __name__ == '__main__':
    # --- Ejemplo de Uso ---
        dummy_dataset_path = 'dataset'
        os.makedirs(os.path.join(dummy_dataset_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dummy_dataset_path, 'masks'), exist_ok=True)

    # Crear imágenes .png y .jpg
        for i in range(10):
            dummy_image_png = Image.new('RGB', (128, 128), color=(i*10, i*5, i*2))
            dummy_mask_png = Image.new('L', (128, 128), color=(i % 2) * 255)
            dummy_image_png.save(os.path.join(dummy_dataset_path, 'images', f'image_png_{i:02d}.png'))
            dummy_mask_png.save(os.path.join(dummy_dataset_path, 'masks', f'image_png_{i:02d}.png'))

        for i in range(10):
            dummy_image_jpg = Image.new('RGB', (128, 128), color=(i*7, i*9, i*4))
            dummy_mask_jpg = Image.new('L', (128, 128), color=(i % 2) * 255)
            dummy_image_jpg.save(os.path.join(dummy_dataset_path, 'images', f'image_jpg_{i:02d}.jpg'))
            dummy_mask_jpg.save(os.path.join(dummy_dataset_path, 'masks', f'image_jpg_{i:02d}.jpg'))

        # Transformaciones
        common_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
    ])

        # Obtener dataloaders
        batch_size = 4
        train_dl, val_dl = get_dataloaders(
            root_dir=dummy_dataset_path,
            batch_size=batch_size,
            image_transform=common_transform,
            mask_transform=common_transform
        )

        print(f"Número de muestras de entrenamiento: {len(train_dl.dataset)}")
        print(f"Número de muestras de validación: {len(val_dl.dataset)}")
        print(f"Número de lotes de entrenamiento: {len(train_dl)}")
        print(f"Número de lotes de validación: {len(val_dl)}")

        print("\n--- Lote de ejemplo del Dataloader de Entrenamiento ---")
        for images, masks in train_dl:
            print(f"Forma del lote de imágenes: {images.shape}")
            print(f"Forma del lote de máscaras: {masks.shape}")
            break

        print("\n--- Lote de ejemplo del Dataloader de Validación ---")
        for images, masks in val_dl:
            print(f"Forma del lote de imágenes: {images.shape}")
            print(f"Forma del lote de máscaras: {masks.shape}")
            break