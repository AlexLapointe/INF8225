import os
import glob
import torch
import lightning as L
from tqdm import tqdm
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import CacheDataset, DataLoader, NibabelReader
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityRanged,
    ResizeWithPadOrCropd, RandCropByPosNegLabeld, ToTensord, EnsureChannelFirstd, RandCropByLabelClassesd, AsDiscreted
)
from torch.optim import Adam
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore", message="no available indices of class.*to crop.*")

# ======================
# 1. CONFIGURATION
# ======================
DATASET_DIR = 'mbh/nii'
SAVE_DIR = "test/first/"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# 2. DATA PIPELINE (same as your original)
# ======================
transforms = Compose([
    LoadImaged(keys=["image", "seg"], reader=NibabelReader()),
    EnsureChannelFirstd(keys=["image", "seg"]),
    ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
    ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=(512, 512, 32)),
    ToTensord(keys=["image", "seg"], dtype=torch.float32),  # Convert to float32
])

def get_data_files(img_dir, seg_dir):
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
    labels = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.endswith('.nii.gz')])
    return [{"image": img, "seg": lbl} for img, lbl in zip(images, labels)]

# ======================
# 3. LIGHTNING MODULE
# ======================
class HemorrhageModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=6,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.loss_fn = DiceCELoss(include_background=False,to_onehot_y=True, softmax=True, lambda_dice=0.7, lambda_ce=0.3)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch",get_not_nans=True, ignore_empty=True)
        self.best_dice = 0.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["seg"]  # y is one-hot from AsDiscreted transform
        y_logits = self(x)

        # Loss computation (automatically applies softmax)
        loss = self.loss_fn(y_logits, y)

        # Metrics (must apply softmax first)
        y_prob = torch.softmax(y_logits, dim=1)
        self.dice_metric(y_prob, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["seg"]
        y_logits = self(x)

        # Loss
        loss = self.loss_fn(y_logits, y)

        # Metrics
        y_prob = torch.softmax(y_logits, dim=1)
        self.dice_metric(y_prob, y)

        # Logs
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            print(f"\nTrain Loss (epoch {self.current_epoch}): {train_loss.item():.4f}")

    def on_validation_epoch_end(self):
        dice_scores,_  = self.dice_metric.aggregate()  # Shape: [num_classes]

        self.dice_metric.reset()

        # Log individual class scores
        class_names = ["EDH", "IPH", "IVH","SAH","SDH"]  # Adjust to your classes

        for i, score in enumerate(dice_scores):
            # Extract scalar value for each class
            self.log(f"val_dice_{class_names[i]}", score.item())  # .item() converts to Python float

        # Log mean Dice (optional)
        mean_dice = dice_scores.mean()
        self.log("val_dice_mean", mean_dice, prog_bar=True)

        print(f"\nDice Scores:")
        for name, score in zip(class_names, dice_scores):
            print(f"{name}: {score.item():.4f}")
        print(f"Mean: {mean_dice.item():.4f}")

        # Save best model
        if mean_dice > self.best_dice:
            self.best_dice = mean_dice
            self.trainer.save_checkpoint(os.path.join(SAVE_DIR, "best.ckpt"))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

# ======================
# 4. TRAINING SETUP
# ======================
def main():
    # Load data (same as original)
    train_files = get_data_files(f"{DATASET_DIR}/train/img", f"{DATASET_DIR}/train/seg")
    val_files = get_data_files(f"{DATASET_DIR}/val/img", f"{DATASET_DIR}/val/seg")

    train_dataset = CacheDataset(
        train_files,
        transform=transforms,
        cache_rate=1.0,  # Cache tout le dataset
        num_workers=8
    )

    val_dataset = CacheDataset(
        val_files,
        transform=transforms,
        cache_rate=1.0,  # Cache tout le dataset
        num_workers=8
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize model with checkpoint if available
    model = HemorrhageModel()

    # Configure trainer with progress bar and checkpointing
    trainer = L.Trainer(
        log_every_n_steps=5,
        max_epochs=1000,
        accelerator="auto",
        devices=[0],
        default_root_dir=SAVE_DIR,
        callbacks=[
            L.pytorch.callbacks.TQDMProgressBar(refresh_rate=10),
            L.pytorch.callbacks.ModelCheckpoint(
            dirpath=SAVE_DIR,
            filename="best_{epoch}_{val_dice_mean:.4f}",  # Nom basé sur la métrique
            monitor="val_dice_mean",  # Métrique à surveiller
            mode="max",              # Maximiser le Dice
            save_top_k=3,            # Garder les 3 meilleurs modèles
            every_n_epochs=10,       # Sauvegarde périodique (optionnel)
            save_last=True
            )
        ]
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
