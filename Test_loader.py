import os
import numpy as np
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



DATASET_DIR = 'mbh_seg/nii'
print(os.listdir(DATASET_DIR))
SAVE_DIR = "/test/first/"
os.makedirs(SAVE_DIR, exist_ok=True)

transforms = Compose([
    LoadImaged(keys=["image", "seg"], reader=NibabelReader()),
    EnsureChannelFirstd(keys=["image", "seg"]),
    ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
    ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=(512, 512, 32)),
    ToTensord(keys=["image", "seg"])
])

def get_data_files(img_dir, seg_dir):
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
    labels = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.endswith('.nii.gz')])
    return [{"image": img, "seg": lbl} for img, lbl in zip(images, labels)]

train_files = get_data_files(f"{DATASET_DIR}/train/img", f"{DATASET_DIR}/train/seg")
val_files = get_data_files(f"{DATASET_DIR}/val/img", f"{DATASET_DIR}/val/seg")
    

# 2. Vérification
print(f"Nombre d'images d'entraînement: {len(train_files)}")

#train_dataset = CacheDataset(data=train_files, transform=transforms)
val_dataset = CacheDataset(data=val_files, transform=transforms)


print("=== Structure des fichiers bruts ===")
print(f"Nombre total d'échantillons d'entraînement: {len(val_files)}")
print("\nExemple du premier élément (avant transformations):")
print(val_files[0])  # Affiche {'image': 'chemin/img1.nii.gz', 'seg': 'chemin/seg1.nii.gz'}

# ===== 2. Inspection du CacheDataset =====
print("\n\n=== Structure du CacheDataset ===")
print(f"Type de train_dataset: {type(val_dataset)}")  # <class 'monai.data.dataset.CacheDataset'>

print(f"Nombre d'éléments dans le cache : {len(val_dataset._cache)}")

# Affichez les 3 premiers éléments du cache (format liste)
for i, cached_item in enumerate(val_dataset._cache[3:6]):
    print(f"\n=== Élément {i} ===")
    
    # Affiche les clés disponibles dans l'élément cache
    print("Clés disponibles :", cached_item.keys())  # Normalement ['image', 'seg']
    
    # Shape des données
    print("Image shape :", cached_item['image'].shape)
    print("Seg shape :", cached_item['seg'].shape)
    
    # Plage de valeurs
    print("Valeurs uniques dans seg :", np.unique(cached_item['seg']))
    print("Intensité image (min/max) :", np.min(cached_item['image']), np.max(cached_item['image']))

   # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

first_batch = next(iter(val_loader))

print("\n=== Structure du batch ===")
print("Clés disponibles :", first_batch.keys())  # ['image', 'seg']
print("Shape des images :", first_batch['image'].shape)  # [2x4, C, H, W, D]
print("Shape des segs :", first_batch['seg'].shape)      # [2x4, 6, H, W, D]

# Détail du premier échantillon du batch
print("\n=== Premier échantillon ===")
print("Intensité image (min/max) :", 
      first_batch['image'][0].min().item(), first_batch['image'][0].max().item())
print("Valeurs uniques seg :", torch.unique(first_batch['seg'][0]))

# Vérification des workers
print("\n=== Workers ===")
print("Nombre de workers actifs :", val_loader.num_workers)
print("Échantillons dans le dataset :", len(val_loader.dataset))
