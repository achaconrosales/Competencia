%cd /content/drive/MyDrive/Competencia
# train_unet.py
print("Importing libraries...")
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from dataloader import SporeDataModule
from unet import UNet

sweep_config = {
    'method': 'random',  # Método de búsqueda: 'random' (aleatorio), 'grid' (malla), 'bayes' (bayesiano)
    'metric': {          # Métrica a optimizar durante el sweep
        'name': 'Val IoU', # La recompensa media por episodio es una métrica estándar de SB3
        'goal': 'maximize'             # Queremos maximizar esta métrica
    },
    'parameters': {      # Definición de los hiperparámetros y sus valores/rangos
        'img_size': {
            'values': [(256, 256), (320, 320), (384, 384)] # Valores específicos a probar para el tamaño de imagen
        },
        'batch_size': {
            'values': [2, 4, 8] # Valores específicos a probar para
        },
        'learning_rate': {
            'values': [1e-3, 1e-4, 1e-5] # Valores específicos a probar para la tasa de aprendizaje
        },
        'num_epochs': {
            'values': [1] # Valores específicos para el número de épocas de entrenamiento
        }
    }
}


# Configuración
model_name = "unet_model.pth"
IMAGE_DIR = './dataset'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    wandb.login(key="604cb8bc212df5c53f97526f8520c686e12d8588") #CUENTA DE AARON
    wandb.init(project="SporeSegmentation")
    config = wandb.config
    
    # Extrae los hiperparámetros de la configuración actual del sweep
    IMAGE_SIZE = config.img_size
    BATCH_SIZE = config.batch_size
    LEARNING_RATE = config.learning_rate
    NUM_EPOCHS = config.num_epochs


    wandb.run.name =f"UNet_img{IMAGE_SIZE[0]}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{wandb.util.generate_id()[:4]}"

    
    dataset = SporeDataModule(IMAGE_DIR, IMAGE_SIZE, BATCH_SIZE)
    dataset.set_seed(42)
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

    #model.save(model_name)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="SporeSegmentation")
    wandb.agent(sweep_id, function=train, count=27)


