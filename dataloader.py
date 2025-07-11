# spore_dataloader.py

import os
from glob import glob
from PIL import Image
from PIL import ImageFile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
import random
from tqdm import tqdm

class SporeDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=True, image_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.image_size = image_size
        self.mean,self.std = self.compute_mean_std(image_size=image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # Resize
        image = image.resize(self.image_size)
        mask = mask.resize(self.image_size)

        # Aumentos básicos (solo en entrenamiento)
        if self.augment:
            if torch.rand(1).item() < 0.2:
                image = image.convert('L').convert('RGB')
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if torch.rand(1).item() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            # Rotación aleatoria
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
            # Crop aleatorio
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            # Ajuste de brillo y contraste solo a la imagen
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        # Convertir a tensores
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask
    
    @staticmethod
    def compute_mean_std(image_size=(224, 224)):
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # ← AÑADIR esto
        
        image_paths = sorted(glob(os.path.join('./dataset/images', '*.[jp][pn]g')))
        sums = torch.zeros(3)
        sumsq = torch.zeros(3)
        n_pixels = 0

        for path in tqdm(image_paths, desc="Procesando imágenes"):
            try:
                img = Image.open(path).convert('RGB').resize(image_size)
                tensor = TF.to_tensor(img)
                sums += tensor.sum(dim=[1, 2])
                sumsq += (tensor ** 2).sum(dim=[1, 2])
                n_pixels += tensor.shape[1] * tensor.shape[2]
            except Exception as e:
                print(f"[WARNING] Imagen corrupta omitida: {path} — {e}")

        mean = sums / n_pixels
        std = (sumsq / n_pixels - mean ** 2).sqrt()
        print(f"Media: {mean}, Desviación estándar: {std}")
        return mean.tolist(), std.tolist()



class SporeDataModule:
    def __init__(self, dataset_dir, image_size=(256, 256), batch_size=8, verbose=True):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.batch_size = batch_size

        self.image_paths = sorted(glob(os.path.join(dataset_dir, 'images', '*.[jp][pn]g')))
        self.mask_paths = sorted(glob(os.path.join(dataset_dir, 'masks', '*.[jp][pn]g')))

        assert len(self.image_paths) == len(self.mask_paths), "Las imágenes y máscaras no coinciden en número"

        # Orden alfabético (split no aleatorio)
        total = len(self.image_paths)
        split_idx = int(0.8 * total)

        self.train_images = self.image_paths[:split_idx]
        self.train_masks = self.mask_paths[:split_idx]
        self.val_images = self.image_paths[split_idx:]
        self.val_masks = self.mask_paths[split_idx:]
        if verbose:
            print(f"Total images: {total}")
            print(f"Training images: {len(self.train_images)}")
            print(f"Validation images: {len(self.val_images)}")


    def get_dataloaders(self):
        train_dataset = SporeDataset(self.train_images, self.train_masks, augment=True, image_size=self.image_size)
        val_dataset = SporeDataset(self.val_images, self.val_masks, augment=False, image_size=self.image_size)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader
    
    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
