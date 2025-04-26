###########################
# U-Net 2D for Segmentation
###########################


###########################
# Importations Librairies:
###########################
# PyTorch:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

#MONAI:
from monai.transforms import Compose,RandFlipd,RandRotate90d,RandAffined
from monai.transforms import RandGridDistortiond,RandAdjustContrastd
from monai.transforms import NormalizeIntensityd,EnsureChannelFirstd,ToTensord

# Traitement image:
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from PIL import Image
import cv2
import nibabel as nib
from collections import defaultdict

# Visualisation:
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Files Management:
import os
import zipfile
import wandb
from google.colab import drive

###########################
# Configuration:
###########################

# Device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# print(torch.__version__)

drive.mount('/content/drive')
zip_path = "/content/drive/MyDrive/dataset.zip"  # chemin zip dans Drive
extract_path = "/content/dataset"                # où décompresser dans Colab

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Décompression terminée")

###########################
# Création du dataset:
###########################

class NiftiSegmentationDataset(Dataset):
    """ NiftiSegmentationDataset for loading 3D NIfTI images 
        and their corresponding segmentation masks.

    Args:
        image_dir (str): Directory containing the NIfTI images.
        mask_dir (str): Directory containing the segmentation masks.
        transform (callable, optional): Optional transform to be applied on a sample.
        class_map (dict, optional): Mapping of class indices for remapping the segmentation masks.
    
    Returns:
        image_slices (list): List of 2D image slices.
        mask_slices (list): List of 2D mask slices.
    """
    def __init__(self, image_dir, mask_dir, transform=False, class_map=None):
        self.image_slices = []
        self.mask_slices = []
        self.transform = transform
        self.class_map = class_map

        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        mask_files = set(os.listdir(mask_dir))

        total_slices = 0
        kept_slices = 0

        for img_file in image_files:
            index = os.path.splitext(os.path.splitext(img_file)[0])[0]
            mask_file = f"{index}.nii.gz"
            if mask_file in mask_files:
                img_path = os.path.join(image_dir, img_file)
                msk_path = os.path.join(mask_dir, mask_file)

                img = nib.load(img_path).get_fdata()
                msk = nib.load(msk_path).get_fdata()

                assert img.shape == msk.shape, f"Shape mismatch in {img_file}"

                for z in range(img.shape[-1]):
                    slice_img = img[:, :, z]
                    slice_msk = msk[:, :, z]

                    total_slices += 1
                    if np.any(slice_msk != 0):  # Ignore slices with only background
                        self.image_slices.append(slice_img)
                        self.mask_slices.append(slice_msk)
                        kept_slices += 1

        print(f"Total slices scanned : {total_slices}")
        print(f"Kept (non-empty) slices : {kept_slices}")

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        mask = self.mask_slices[idx]

        if self.class_map:
            mask = remap_mask(mask, self.class_map)

        if self.transform:
            augmented = self.transform(image=image.astype(np.float32), mask=mask.astype(np.uint8))
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

def generate_image_batch(data_batch: list) -> tuple:
    """
    Assemble a batch of image/mask pairs into batched tensors.

    Args:
        data_batch (list): List of (image, mask) tuples.
            - image: Tensor of shape (C, H, W)
            - mask : Tensor of shape (H, W)

    Returns:
        image_batch (Tensor): Batched images of shape (B, C, H, W)
        mask_batch  (Tensor): Batched masks of shape (B, H, W)
    """
    images, masks = zip(*data_batch)
    image_batch = torch.stack(images)  # Shape: (B, C, H, W)
    mask_batch = torch.stack(masks)    # Shape: (B, H, W)
    return image_batch, mask_batch

def remap_mask(mask: np.ndarray, mapping: dict) -> np.ndarray:
    """
    Remappe les valeurs des masques selon un dictionnaire de correspondance.

    Args:
        mask (np.ndarray): masque original
        mapping (dict): ex. {0:0, 2:1, 3:2, 4:3, 5:4}

    Returns:
        np.ndarray: masque remappé avec classes consécutives
    """
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for old_val, new_val in mapping.items():
        remapped[mask == old_val] = new_val
    return remapped

###########################
# Test de Visualisation des données:
###########################

img_path = "/content/dataset/dataset/mbh/nii/test/img/ID_0219ef88_ID_e5c1a31210.nii.gz"
img = nib.load(img_path)
data = img.get_fdata()

print("Shape :", data.shape)              # (H, W, D) ? (D, H, W) ?
print("Data type :", data.dtype)
print("Min/Max :", data.min(), data.max())

print(f"Nombre de dimensions : {data.ndim}")
for i, dim in enumerate(data.shape):
    print(f"Dim {i} size = {dim}")


plt.imshow(data[:, :, data.shape[2] // 2], cmap="gray")
plt.title("Coupe axiale (tranche centrale)")
plt.show()

img_path = "/content/dataset/dataset/mbh/nii/test/seg/ID_0219ef88_ID_e5c1a31210.nii.gz"
img = nib.load(img_path)
data = img.get_fdata()

print("Shape :", data.shape)              # (H, W, D) ? (D, H, W) ?
print("Data type :", data.dtype)
print("Min/Max :", data.min(), data.max())

print(f"Nombre de dimensions : {data.ndim}")
for i, dim in enumerate(data.shape):
    print(f"Dim {i} size = {dim}")


plt.imshow(data[:, :, data.shape[2] // 2], cmap="gray")
plt.title("Coupe axiale (tranche centrale)")
plt.show()

###########################
# Initilisation des datasets:
###########################

base_path_nii = "/content/dataset/dataset/mbh/nii"

# TRAIN
train_image_dir = os.path.join(base_path_nii, "train", "img")
train_mask_dir  = os.path.join(base_path_nii, "train", "seg")

# VAL
val_image_dir = os.path.join(base_path_nii, "val", "img")
val_mask_dir  = os.path.join(base_path_nii, "val", "seg")

# TEST
test_image_dir = os.path.join(base_path_nii, "test", "img")
test_mask_dir  = os.path.join(base_path_nii, "test", "seg")

# Sans augmentation:
train_dataset = NiftiSegmentationDataset(train_image_dir, train_mask_dir)
val_dataset   = NiftiSegmentationDataset(val_image_dir, val_mask_dir)
test_dataset  = NiftiSegmentationDataset(test_image_dir, test_mask_dir)

idx = 2  # change l'index si nécessaire
image, mask = test_dataset[idx]  # train_dataset est une instance de NiftiSegmentationDataset

# # Affichage d'une image et de son masque
# plt.figure(figsize=(10, 4))
# plt.suptitle(f"Tranche {idx} - Retenue")
# plt.subplot(1, 2, 1)
# plt.title("Image")
# plt.imshow(image.squeeze().numpy(), cmap="gray")
# plt.axis("off")
# plt.subplot(1, 2, 2)
# plt.title("Masque")
# plt.imshow(mask.numpy(), cmap="jet")
# plt.axis("off")
# plt.show()

###########################
# Initialisation des datasets augmentés:
###########################

train_transform = Compose([
    EnsureChannelFirstd(keys=["image", "seg"]),
    RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),  # Horizontal (X)
    RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),  # Vertical (Y)
    RandRotate90d(keys=["image", "seg"], prob=0.5, max_k=3),     # 90° rotation
    RandAffined(
        keys=["image", "seg"],
        prob=0.5,
        rotate_range=(0.2, 0.2, 0.2),  # approx ±25°
        shear_range=(0.1, 0.1, 0.1),
        padding_mode='border'
    ),
    RandGridDistortiond(
        keys=["image", "seg"],
        prob=0.3,
        distort_limit=0.3
    ),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
    NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
    ToTensord(keys=["image", "seg"])
])

val_transform = Compose([
    EnsureChannelFirstd(keys=["image", "seg"]),
    NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
    ToTensord(keys=["image", "seg"])
])

######################################################
# Initialisation du U-Net:
######################################################

class DoubleConv(nn.Module):
    """
    Double convolution block for U-Net.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        x (Tensor): Output tensor after double convolution.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    Classic U-Net architecture for image segmentation.

    Args:
        n_channels (int): Number of input channels (e.g., 1 for grayscale).
        n_classes (int): Number of output classes for segmentation.
    
    Returns:
        logits (Tensor): Output tensor of shape (B, n_classes, H, W).
    """
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = self.down_block(64, 128)
        self.down2 = self.down_block(128, 256)
        self.down3 = self.down_block(256, 512)
        self.down4 = self.down_block(512, 1024)

        self.up1 = self.up_block(1024, 512)
        self.up2 = self.up_block(512, 256)
        self.up3 = self.up_block(256, 128)
        self.up4 = self.up_block(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.crop_and_concat(self.up1[0](x5), x4)
        x = self.up1[1](x)

        x = self.crop_and_concat(self.up2[0](x), x3)
        x = self.up2[1](x)

        x = self.crop_and_concat(self.up3[0](x), x2)
        x = self.up3[1](x)

        x = self.crop_and_concat(self.up4[0](x), x1)
        x = self.up4[1](x)

        logits = self.outc(x)
        return logits

    def crop_and_concat(self, upsampled, bypass):
        _, _, H, W = upsampled.shape
        bypass_cropped = self.center_crop(bypass, H, W)
        return torch.cat([bypass_cropped, upsampled], dim=1)

    def center_crop(self, tensor, target_h, target_w):
        _, _, h, w = tensor.shape
        start_x = (w - target_w) // 2
        start_y = (h - target_h) // 2
        return tensor[:, :, start_y:start_y + target_h, start_x:start_x + target_w]

def count_unique_labels(dataset):
    """
    Count unique labels in the dataset for testing.
    Args:
        dataset (Dataset): PyTorch dataset containing images and masks.
    Returns:
        list: Sorted list of unique labels.
    """
    unique_labels = set()
    for i in range(len(dataset)):
        _, mask = dataset[i]
        unique_labels.update(torch.unique(mask).tolist())
    return sorted(unique_labels)

######################################################
# Initialisation des fonctions nécessaires pour le training:
######################################################

def compute_weight_map(mask, w0=20, sigma=3):
    """
    Compute pixel-wise weight map for segmentation training.

    Args:
        mask (np.array): Labeled 2D mask.
        w0 (float): Weight for separation term.
        sigma (float): Std-dev for border emphasis.

    Returns:
        np.array: weight map w(x)
    """
    labeled_mask = label(mask)
    num_objects = np.max(labeled_mask)

    if num_objects < 2:
        return np.ones_like(mask, dtype=np.float32)

    # Distances aux autres objets
    dists = np.zeros((num_objects, *mask.shape), dtype=np.float32)
    for i in range(1, num_objects + 1):
        dists[i - 1] = distance_transform_edt(labeled_mask != i)

    d1 = np.min(dists, axis=0)
    dists[dists == d1] = np.inf
    d2 = np.min(dists, axis=0)
    separation = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))

    # Pondération par fréquence de classe
    class_weights = np.ones_like(mask, dtype=np.float32)
    labels, counts = np.unique(mask, return_counts=True)
    total = np.sum(counts)
    freqs = {val: count / total for val, count in zip(labels, counts)}

    for val in labels:
        # if val == 0:
        #     continue
        class_weights[mask == val] = 1.0 / (freqs[val] + 1e-6)

    # Normaliser pour stabiliser:
    class_weights /= np.mean(class_weights)

    return class_weights + separation

def compute_weight_batch(batch_masks):
    """
    Compute the weight map for each mask in a batch.
    Returns a tensor of shape (B, H, W)
    
    Args:
        batch_masks (Tensor): Shape (B, H, W) — batch of masks
    
    Returns:
        Tensor: Shape (B, H, W) — batch of weight maps
    """
    weights = []
    for mask in batch_masks:
        mask_np = mask.cpu().numpy()
        mask_np = (mask_np > 0).astype(np.uint8)  # binarize the mask
        w = compute_weight_map(mask_np)
        weights.append(torch.tensor(w, dtype=torch.float32))
    return torch.stack(weights).to(device)

def weighted_cross_entropy_loss(logits, targets, weight_map):
    """
    Computes the pixel-wise weighted cross-entropy loss as defined in Eq. (1) of the U-Net paper.

    Args:
        logits (Tensor): Network output, shape (B, C, H, W), unnormalized scores for each class.
        targets (Tensor): Ground truth segmentation, shape (B, H, W), with class indices.
        weight_map (Tensor): Weight map w(x), shape (B, H, W), computed from the ground truth mask.

    Returns:
        Tensor: Scalar loss value.
    """
    assert logits.device == targets.device == weight_map.device
    assert logits.dtype in [torch.float32, torch.float16, torch.bfloat16]
    assert weight_map.dtype in [torch.float32, torch.float16, torch.bfloat16]

    B, C, H, W = logits.shape

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=1)

    # Select log-probabilities of the correct class at each pixel
    log_probs_target = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Apply pixel-wise weights
    loss = -weight_map * log_probs_target
    return loss.mean()

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Compute class-weighted multi-class Dice Loss between logits and ground truth labels.

    Args:
        logits (Tensor): Shape (B, C, H, W) — raw output from the model (before softmax)
        targets (Tensor): Shape (B, H, W) — ground truth class indices
        smooth (float): smoothing constant to avoid division by zero

    Returns:
        Tensor: Dice loss value (scalar)
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)                           # (B, C, H, W)
    targets_one_hot = F.one_hot(targets, num_classes)          # (B, H, W, C)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

    intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)

    # Calcul dynamique de la fréquence de pixels par classe
    class_pixel_counts = targets_one_hot.sum(dim=(0, 2, 3))  # (C,)
    class_weights = 1.0 / (class_pixel_counts + 1e-6)
    class_weights = torch.clamp(class_weights, max=10.0)
    class_weights /= class_weights.sum()

    weighted_dice_loss = (1.0 - dice) * class_weights
    return weighted_dice_loss.sum()

def combined_loss(logits: torch.Tensor, targets: torch.Tensor, weight_map: torch.Tensor, smooth: float = 1.0, w_ce: float = 0.3, w_dice: float = 0.7) -> torch.Tensor:
    """
    Combines the Weighted Cross-Entropy Loss and Dice Loss.

    Args:
        logits (Tensor): Shape (B, C, H, W) — raw output from the model (before softmax)
        targets (Tensor): Shape (B, H, W) — ground truth class indices
        weight_map (Tensor): Shape (B, H, W) — pixel-wise weight map
        smooth (float): Smoothing constant for Dice Loss to avoid division by zero
        w_ce (float): Weight for the Cross-Entropy loss component
        w_dice (float): Weight for the Dice loss component

    Returns:
        Tensor: Combined loss value
    """
    # Compute Dice Loss
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)                    # (B, C, H, W)
    targets_one_hot = F.one_hot(targets, num_classes)   # (B, H, W, C)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

    intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()

    # Compute Weighted Cross-Entropy Loss
    assert logits.device == targets.device == weight_map.device
    assert logits.dtype in [torch.float32, torch.float16, torch.bfloat16]
    assert weight_map.dtype in [torch.float32, torch.float16, torch.bfloat16]

    B, C, H, W = logits.shape

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=1)

    # Select log-probabilities of the correct class at each pixel
    log_probs_target = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Apply pixel-wise weights
    ce_loss = -weight_map * log_probs_target

    # Combine both losses
    total_loss = (w_ce * ce_loss.mean()) + (w_dice * dice_loss)

    return total_loss

def loss_batch_unet(model: nn.Module, images: torch.Tensor, masks: torch.Tensor, config: dict) -> dict:
    """
    Compute loss and metrics for a U-Net batch (segmentation).

    Args:
        model (nn.Module): U-Net model
        images (Tensor): shape (B, C, H, W)
        masks (Tensor): shape (B, H, W) with integer class labels
        config (dict): contains:
            - 'device': torch.device
            - 'loss': loss function
            - 'weight_map_fn': function that generates weight maps from masks

    Returns:
        dict: {'loss': ..., 'iou': ..., 'accuracy': ...}
    """
    device = config['device']
    loss_fn = config['loss']
    weight_map_fn = config.get('weight_map_fn', None)

    images = images.to(device)
    masks = masks.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(images)  # shape: (B, C, H, W)

        if weight_map_fn:
            weight_map = weight_map_fn(masks)
            loss = loss_fn(outputs, masks, weight_map)
        else:
            loss = loss_fn(outputs, masks)

        # Get predicted classes
        preds = torch.argmax(outputs, dim=1)  # (B, H, W)

        # IoU
        intersection = torch.logical_and(preds == 1, masks == 1).sum(dim=(1, 2)).float()
        union = torch.logical_or(preds == 1, masks == 1).sum(dim=(1, 2)).float()
        iou = (intersection / (union + 1e-6)).mean().item()

        # Pixel accuracy
        correct = (preds == masks).float().mean().item()

    return {'loss': loss.item(), 'iou': iou, 'accuracy': correct}


######################################################
# Initialisation des fonctions d'évaluation
######################################################

def eval_model(model: nn.Module, dataloader: DataLoader, config: dict, topk_list=[1, 3, 5]) -> dict:
    """
    Evaluate U-Net on a dataset and compute multiple metrics.

    Args:
        model (nn.Module): Trained segmentation model.
        dataloader (DataLoader): Validation or test loader.
        config (dict): Must contain 'device', 'loss', and optionally 'weight_map_fn'.
        topk_list (list): List of top-k accuracies to compute.

    Returns:
        dict: Averaged metrics over the dataset: loss, accuracy, mean IoU, top-k.
    """
    device = config["device"]
    loss_fn = config["loss_fn"]
    weight_map_fn = config.get("weight_map_fn", None)
    logs = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Loss fonction:
            if weight_map_fn:
                weight_map = weight_map_fn(masks).to(device)
                loss = loss_fn(outputs, masks, weight_map)
            else:
                loss = loss_fn(outputs, masks)

            logs["loss"].append(loss.item())

            # Pixel accuracy
            correct = (preds == masks).float().mean().item()
            logs["accuracy"].append(correct)

            # Per-class IoU (excluding background = 0)
            ious = []
            num_classes = outputs.shape[1]
            for cls in range(1, num_classes):  # ignore background
                pred_cls = (preds == cls)
                mask_cls = (masks == cls)
                intersection = torch.logical_and(pred_cls, mask_cls).sum().float()
                union = torch.logical_or(pred_cls, mask_cls).sum().float()
                if union > 0:
                    iou = intersection / (union + 1e-6)
                    ious.append(iou.item())

            mean_iou = np.mean(ious) if ious else 0.0
            logs["IoU"].append(mean_iou)

            # Top-k pixel accuracy
            for k in topk_list:
                acc_k = topk_pixel_accuracy(outputs, masks, k)
                logs[f"top-{k}"].append(acc_k)

    return {k: np.mean(v) for k, v in logs.items()}

def topk_pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Compute pixel-wise top-k accuracy for segmentation.

    Args:
        logits (Tensor): shape (B, C, H, W)
        targets (Tensor): shape (B, H, W)
        k (int): top-k value

    Returns:
        float: top-k pixel accuracy
    """
    B, C, H, W = logits.shape
    targets_flat = targets.view(B, -1)
    logits_flat = logits.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)

    topk = logits_flat.topk(k, dim=-1).indices           # (B, H*W, k)
    targets_exp = targets_flat.unsqueeze(-1).expand_as(topk)  # (B, H*W, k)

    correct = (topk == targets_exp).any(dim=-1).float()
    return correct.mean().item()

def show_predictions(model, dataset, device, n=3):
    """
    Visualize `n` predictions from the dataset using the trained model.

    Args:
        model (nn.Module): Trained U-Net model
        dataset (Dataset): PyTorch dataset (ex: val_dataset)
        device (torch.device): "cuda" or "cpu"
        n (int): Number of samples to display
    """
    model.eval()
    n = min(n, len(dataset))  # ne pas dépasser la taille du dataset

    plt.figure(figsize=(12, 4 * n))
    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]
            image = image.to(device).unsqueeze(0)  # (1, 1, H, W)
            pred = model(image)  # (1, C, H, W)
            pred_mask = torch.argmax(pred.squeeze(), dim=0).cpu()

            plt.subplot(n, 3, i * 3 + 1)
            plt.title("Input Image")
            plt.imshow(image.squeeze().cpu().numpy(), cmap="gray")
            plt.axis("off")

def print_logs(dataset_type: str, logs: dict):
    """
    Nicely format and print log metrics for a given dataset type.

    Args:
        dataset_type (str): "Train", "Val", or "Test"
        logs (dict): Dictionary of metric names and values (floats)
    """
    formatted = [
        f"{name}: {value:.8f}"
        for name, value in logs.items()
    ]
    desc = f"{dataset_type} —\t" + "\t".join(formatted)
    print(desc)

######################################################
# Training Function:
######################################################

def train_and_validate(model, train_loader, val_loader, optimizer, config, n_epochs, log_wandb=True, 
                       show_preds=None, dataset_for_display=None, n_logged_images=3, save_path="Lattest_best_model.pth"):
    """
    Train and validate the model for a specified number of epochs.  
    
    Args:   
        model (nn.Module): U-Net model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (torch.optim.Optimizer): Optimizer for training
        config (dict): Configuration dictionary containing:
            - 'device': torch.device
            - 'loss_fn': loss function
            - 'weight_map_fn': function to compute weight maps (optional)
            - 'scheduler': learning rate scheduler (optional)
        n_epochs (int): Number of epochs to train
        log_wandb (bool): Whether to log metrics to Weights & Biases
        show_preds (bool): Whether to show predictions during training
        dataset_for_display (Dataset): Dataset to use for displaying predictions
        n_logged_images (int): Number of images to log
        save_path (str): Path to save the best model
    """
    device = config["device"]
    loss_fn = config["loss_fn"]
    weight_map_fn = config.get("weight_map_fn", None)
    scheduler = config.get("scheduler", None)  # ajout pour compatibilité scheduler
    best_val_iou = 0.0
    num_used_batches = 0

    if log_wandb:
        log_table = wandb.Table(columns=["epoch", "image", "ground_truth", "prediction"])

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        model.train()
        total_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            if torch.all(masks == 0):
                # print(" Batch sans aucune classe foreground — ignoré")
                continue
            num_used_batches += 1

            optimizer.zero_grad()
            images = images.to(device)
            masks = masks.to(device)

            weight_map = weight_map_fn(masks).to(device) if weight_map_fn else None
            outputs = model(images)

            assert masks.max() < outputs.shape[1], \
                f"Target max label {masks.max().item()} >= n_classes={outputs.shape[1]}"
            loss = loss_fn(outputs, masks, weight_map) if weight_map is not None else loss_fn(outputs, masks)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / num_used_batches
        print(f"\nTraining Loss: {avg_train_loss:.8f}")

        # Validation
        val_logs = eval_model(model, val_loader, config, topk_list=[1, 3, 5])
        print_logs("Validation", val_logs)

        val_iou = val_logs.get("IoU", 0.0)
        if val_iou > best_val_iou:
            print(f"New best IoU: {val_iou:.8f} (previous: {best_val_iou:.8f}) — saving model.")
            torch.save(model.state_dict(), save_path)
            best_val_iou = val_iou

        if log_wandb and (epoch + 1) % 2 == 0:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                **{f"val_{k}": v for k, v in val_logs.items()}
            })

        if scheduler is not None:
            scheduler.step()  # mise à jour du learning rate

        if show_preds and dataset_for_display is not None and (epoch) % 10 == 0:
            model.eval()
            with torch.no_grad():
                for i in range(min(n_logged_images, len(dataset_for_display))):
                    if i % 75 != 0:
                        continue
                    img, mask = dataset_for_display[i]
                    input_tensor = img.unsqueeze(0).to(device)
                    pred = model(input_tensor)
                    pred_mask = torch.argmax(pred.squeeze(), dim=0).cpu()

                    if log_wandb:
                        wandb.log({
                            f"Example/{i}/image": wandb.Image(img.squeeze().cpu().numpy(), caption="Input"),
                            f"Example/{i}/mask": wandb.Image(mask.cpu().numpy(), caption="Ground Truth"),
                            f"Example/{i}/prediction": wandb.Image(pred_mask.numpy(), caption="Prediction")
                        })
                        log_table.add_data(
                            epoch + 1,
                            wandb.Image(img.squeeze().cpu().numpy(), caption="Input"),
                            wandb.Image(mask.cpu().numpy(), caption="Ground Truth"),
                            wandb.Image(pred_mask.numpy(), caption="Prediction")
                        )

    if log_wandb:
        wandb.log({"Predictions Table": log_table})
        print("Final wandb table logged.")

######################################################
# Training and evaluation loop:
######################################################

torch.cuda.empty_cache()

batch_sizes = [16]
epoch_values = [500]
learning_rates = [1e-4]
n_classes = 6

for lr in learning_rates:
    print(lr)
    for epochs in epoch_values:
        torch.cuda.empty_cache()
        print(epochs)
        print()
        for batch_size in batch_sizes:
            model = UNet(n_channels=1, n_classes=n_classes)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Cosine Decay
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            ### Version dataset 3D:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False
            )

##### Configurations pour le training avec différentes fonctions de perte:
### 1) Weighted Cross Entropy Loss
            config = {
                      "architecture": "U-Net",
                      "loss": "Weighted Cross Entropy",
                      "epochs": epochs,
                      "batch_size": batch_size,
                      "optimizer": "Adam",
                      "lr": lr,
                      "n_classes": n_classes,
                      "dataset": base_path_nii,
                      "device": device,
                      "loss_fn": weighted_cross_entropy_loss,
                      "train_loader": train_loader,
                      "val_loader": val_loader,
                      "optimizer_instance": optimizer,
                      "scheduler": scheduler,
                      "weight_map_fn": compute_weight_batch
                  }


### 2) Dice Loss
            # config = {
            #           "architecture": "U-Net",
            #           "loss": "Dice Loss",
            #           "epochs": epochs,
            #           "batch_size": batch_size,
            #           "optimizer": "Adam",
            #           "lr": lr,
            #           "n_classes": n_classes,
            #           "dataset": base_path_nii,
            #           "device": device,
            #           "loss_fn": dice_loss,
            #           "train_loader": train_loader,
            #           "val_loader": val_loader,
            #           "optimizer_instance": optimizer,
            #           "scheduler": scheduler,
            #           "weight_map_fn": None  # pas avec DiceLoss
            #       }

### 3) Combined Loss (Weighted Cross Entropy + Dice Loss)
            # config = {
            #           "architecture": "U-Net",
            #           "loss": "combined_loss",
            #           "epochs": epochs,
            #           "batch_size": batch_size,
            #           "optimizer": "Adam",
            #           "lr": lr,
            #           "n_classes": n_classes,
            #           "dataset": base_path_nii,
            #           "device": device,
            #           "loss_fn": combined_loss,
            #           "train_loader": train_loader,
            #           "val_loader": val_loader,
            #           "optimizer_instance": optimizer,
            #           "scheduler": scheduler,
            #           "weight_map_fn": compute_weight_batch
            #       }

            model.to(config["device"])
            run_name = f"U-Net_3DCE_1"

#### Compilation des logs pour wandb:
            with wandb.init(
                config=config,
                project=f"Projet U-Net - LR = 1e-03",
                group="U-Net Cell Segmentation",
                name=run_name,
                save_code=True,
            ):
                train_and_validate(
                    model=model,
                    train_loader=config["train_loader"],
                    val_loader=config["val_loader"],
                    optimizer=config["optimizer_instance"],
                    config={
                        "device": config["device"],
                        "loss_fn": config["loss_fn"],
                        "weight_map_fn": config["weight_map_fn"],
                        "scheduler": config["scheduler"]
                    },
                    n_epochs=config["epochs"],
                    log_wandb=True,
                    show_preds=True,
                    dataset_for_display=val_dataset
                )

