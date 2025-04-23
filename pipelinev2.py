import os
import glob
import torch
import lightning as L
from tqdm import tqdm
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric , DiceHelper
from monai.data import CacheDataset, DataLoader, NibabelReader,PersistentDataset, MetaTensor
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
DATASET_DIR = '/home/tibia/Projet_Hemorragie/mbh_seg/nii'
SAVE_DIR = "/home/tibia/Projet_Hemorragie/test/first/"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# 2. DATA PIPELINE (same as your original)
# ======================
transforms = Compose([
    LoadImaged(keys=["image", "seg"], reader=NibabelReader()),
    EnsureChannelFirstd(keys=["image", "seg"]),
    ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
    ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=(512, 512, 32)),
    
    #RandCropByLabelClassesd(
    #keys=["image", "seg"],
    #label_key="seg",
    #spatial_size=(96, 96, 32),
    #num_classes=6,    # number of foreground classes (1, 2, 3,4,5)
    #ratios=[0, 1, 1, 1,1, 1],        # sample all classes equally
   #num_samples=4,),
   # ToTensord(keys=["image", "seg"])
])

def get_data_files(img_dir, seg_dir):
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
    labels = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.endswith('.nii.gz')])
    return [{"image": img, "seg": lbl} for img, lbl in zip(images, labels)]



class DiceHelperAggregator:
    def __init__(self, num_classes=6):
        self.all_preds = []
        self.all_labels = []
        self.num_classes = num_classes
        self.class_names = ["EDH", "IPH", "IVH", "SAH", "SDH"]  # Ajustez selon vos classes

    def update(self, y_pred_logits, y_labels):
        """Stocke les prédictions et labels bruts"""
        self.all_preds.append(y_pred_logits.detach().cpu())
        self.all_labels.append(y_labels.detach().cpu())

    def compute(self):
        """Calcule le Dice final sur toutes les données accumulées"""
        # Concaténation de tous les batchs
        y_preds = torch.cat(self.all_preds)  # Shape [N, 6, H, W, D]
        y_labels = torch.cat(self.all_labels) # Shape [N, 1, H, W, D]
        
        # Calcul du Dice avec DiceHelper
        dice_helper = DiceHelper(
            include_background=False,
            softmax=True,  # Applique softmax + argmax automatiquement
            reduction="mean_batch",
            num_classes=self.num_classes
        )
        dice_scores, _ = dice_helper(y_preds, y_labels)
        
        return dice_scores
    
    def reset(self):
        self.all_preds = []
        self.all_labels = []
# ======================
# 3. LIGHTNING MODULE 
# ======================
class HemorrhageModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.dice_aggregator = DiceHelperAggregator(num_classes=6)

        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=6,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.loss_fn = DiceCELoss(include_background=False,to_onehot_y=True, softmax=True, lambda_dice=0.7, lambda_ce=0.3)
        self.dice_metric = DiceHelper(include_background=None,  softmax=True, get_not_nans=True, reduction="mean_batch", ignore_empty=True)
        self.best_dice = 0.0
        
        
        

    def forward(self, x):
        return self.model(x)

    
    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["seg"]      # y: [B,1,H,W,D], valeurs 0..5
        y_logits = self(x)                       # [B,6,H,W,D]
        loss = self.loss_fn(y_logits, y)         # softmax + to_onehot_y à l’intérieur

        y_prob = torch.softmax(y_logits, dim=1)  # [B,6,H,W,D]
        self.dice_metric(y_prob, y)              # y reste indexé

        self.log("train_loss", loss, prog_bar=True)
        return loss
    
   

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["seg"]  # y: [B, 1, H, W, D]
        y_logits = self(x)  #y_logits [B, 6, H, W, D]
        self.dice_aggregator.update(y_logits, y)
        # Loss
        loss = self.loss_fn(y_logits, y)

        # Metrics

        

        # Logs
        self.log("val_loss", loss, prog_bar=True)
        return loss

       
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            print(f"\nTrain Loss (epoch {self.current_epoch}): {train_loss.item():.4f}")

    def on_validation_epoch_end(self):
    # 1. Calcul des scores finaux
        dice_scores = self.dice_aggregator.compute()  # Shape: [num_classes]
    
    # 2. Définition des noms de classes (ajustez selon votre cas)
        class_names = ["EDH", "IPH", "IVH", "SAH", "SDH"]  # Exclut le background
    
    # 3. Calcul du Dice moyen (sans le background)
        mean_dice = dice_scores.mean()
    
    # 4. Affichage détaillé
        print("\n=== Dice Scores ===")
        for name, score in zip(class_names, dice_scores):
            print(f"{name}: {score.item():.4f}")
        print(f"Mean Dice: {mean_dice.item():.4f}")
    
    # 5. Logging pour TensorBoard/MLflow
        for name, score in zip(class_names, dice_scores):
            self.log(f"val_dice_{name}", score.item(), prog_bar=False)
        self.log("val_dice_mean", mean_dice.item(), prog_bar=True)
    
    # 6. Sauvegarde du meilleur modèle
        if mean_dice > self.best_dice:
            self.best_dice = mean_dice
            best_path = os.path.join(SAVE_DIR, f"best_model_epoch_{self.current_epoch}_dice_{mean_dice:.4f}.ckpt")
            self.trainer.save_checkpoint(best_path)
            print(f"\n Nouveau meilleur modèle sauvegardé (Dice: {mean_dice:.4f}) ")
    
    # 7. Réinitialisation pour la prochaine epoch
        self.dice_aggregator.reset()
       

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

  

# ======================
# 4. TRAINING SETUP
# ======================
def main():
    # Load data (same as original)
    train_files = get_data_files(f"{DATASET_DIR}/train/img", f"{DATASET_DIR}/train/seg")
    val_files = get_data_files(f"{DATASET_DIR}/val/img", f"{DATASET_DIR}/val/seg")
    
    train_dataset = PersistentDataset(
        train_files, 
        transform=transforms,
        cache_dir=os.path.join(SAVE_DIR, "cache_train")
    )
    
    val_dataset = PersistentDataset(
        val_files,
        transform=transforms,
        cache_dir=os.path.join(SAVE_DIR, "cache_val")
    )

    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

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
            save_weights_only=True, 
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