# train_unet.py
print("Importing libraries...")
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from dataloader import SporeDataModule
from unet import UNet

# Configuración
model_name = "unet_model.pth"
IMAGE_DIR = './dataset'
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
IMAGE_SIZE = (224, 224)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    wandb.login(key="604cb8bc212df5c53f97526f8520c686e12d8588") #CUENTA DE AARON
    wandb.init(project="SporeSegmentation", 
               name=f"UNet_img{IMAGE_SIZE[0]}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{wandb.util.generate_id()[:4]}",
               config={"model": "UNet",
                        "epochs": NUM_EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "image_size": IMAGE_SIZE,})
    
    dataset = SporeDataModule(IMAGE_DIR, IMAGE_SIZE, BATCH_SIZE)
    train_loader, val_loader = dataset.get_dataloaders()

    model = UNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = running_dice = running_iou = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_dice += model.dice_coef(outputs, masks).item() * images.size(0)
            running_iou += model.iou_score(outputs, masks).item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_dice = running_dice / len(train_loader.dataset)
        epoch_iou = running_iou / len(train_loader.dataset)

        # VALIDATION
        model.eval()
        val_loss = val_dice = val_iou = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_dice += model.dice_coef(outputs, masks).item() * images.size(0)
                val_iou += model.iou_score(outputs, masks).item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        wandb.log({
            "Train Loss": epoch_loss,
            "Train Dice": epoch_dice,
            "Train IoU": epoch_iou,
            "Val Loss": val_loss,
            "Val Dice": val_dice,
            "Val IoU": val_iou

        })
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Train Dice: {epoch_dice:.4f} | Train IoU: {epoch_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

    model.save(model_name)
if __name__ == "__main__":
    train()
