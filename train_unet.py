# train_unet.py
print("Importing libraries...")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from dataloader import SporeSegmentationDataset
from unet import UNet

# Configuración
IMAGE_DIR = './dataset'
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
IMAGE_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dice_coef(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def iou_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-8)


def train():
    dataset = SporeSegmentationDataset(IMAGE_DIR, transform=transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_dice += dice_coef(outputs, masks).item() * images.size(0)
            running_iou += iou_score(outputs, masks).item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        epoch_dice = running_dice / len(dataset)
        epoch_iou = running_iou / len(dataset)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f} | IoU: {epoch_iou:.4f}")



    torch.save(model.state_dict(), "unet_sporas.pth")
    print("✅ Modelo guardado como unet_sporas.pth")

if __name__ == "__main__":
    train()
