"""
Dataset and Data Loading for Satellite Damage Detection

Contains:
- Scene-wise dataset splitting (prevents data leakage)
- Custom Dataset class with Albumentations transforms
- Data loaders for training and validation
"""

import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import TrainingConfig, get_config


def discover_scenes(data_root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Scan processed_data folder and group files by scene_id.

    This ensures tiles from the same scene are kept together
    during train/val split to prevent data leakage.

    Args:
        data_root: Path to processed_data/combined folder

    Returns:
        Dictionary mapping scene_id to file paths:
        {
            "georgia-flooding_00001_tile_0_0": {
                "pre": Path(...),
                "post": Path(...),
                "mask": Path(...),
                "diff": Path(...) or None
            },
            ...
        }
    """
    scenes = defaultdict(dict)

    # Get all directories
    pre_dir = data_root / "pre"
    post_dir = data_root / "post"
    mask_dir = data_root / "masks"
    diff_dir = data_root / "diff"

    if not pre_dir.exists():
        raise ValueError(f"Pre-image directory not found: {pre_dir}")

    # Scan pre directory as reference
    for pre_file in sorted(pre_dir.glob("*.png")):
        # Extract scene_id from filename
        # Format: tier1_georgia-flooding_00001_tile_0_0.png
        scene_id = pre_file.stem  # Remove .png

        # Find corresponding files
        post_file = post_dir / pre_file.name
        mask_file = mask_dir / pre_file.name
        diff_file = diff_dir / pre_file.name if diff_dir.exists() else None

        # Verify files exist
        if post_file.exists() and mask_file.exists():
            scenes[scene_id] = {
                "pre": pre_file,
                "post": post_file,
                "mask": mask_file,
                "diff": diff_file if diff_file and diff_file.exists() else None,
            }

    return dict(scenes)


def extract_base_scene_id(tile_id: str) -> str:
    """
    Extract base scene ID from tile ID to group tiles from same scene.

    Example:
        "tier1_georgia-flooding_00001_tile_0_0" -> "tier1_georgia-flooding_00001"

    This prevents tiles from the same original image being split
    between train and validation sets.
    """
    # Pattern: tierX_disaster-name_XXXXX_tile_row_col
    match = re.match(r"(.+)_tile_\d+_\d+$", tile_id)
    if match:
        return match.group(1)
    return tile_id  # Fallback to full ID


def scene_wise_split(
    scenes: Dict[str, Dict[str, Path]],
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split scenes into train and validation sets BY SCENE, not by tile.

    This is CRITICAL to prevent data leakage between tiles from the same image.

    Args:
        scenes: Dictionary of all scenes
        train_ratio: Proportion for training (default: 0.8)
        random_seed: For reproducibility

    Returns:
        (train_tile_ids, val_tile_ids)
    """
    # Group tile IDs by their base scene
    scene_to_tiles = defaultdict(list)
    for tile_id in scenes.keys():
        base_scene = extract_base_scene_id(tile_id)
        scene_to_tiles[base_scene].append(tile_id)

    # Get unique base scenes
    unique_scenes = list(scene_to_tiles.keys())

    # Shuffle scenes (not tiles!)
    random.seed(random_seed)
    random.shuffle(unique_scenes)

    # Split by scenes
    split_idx = int(len(unique_scenes) * train_ratio)
    train_scenes = unique_scenes[:split_idx]
    val_scenes = unique_scenes[split_idx:]

    # Collect all tiles for each split
    train_tiles = []
    for scene in train_scenes:
        train_tiles.extend(scene_to_tiles[scene])

    val_tiles = []
    for scene in val_scenes:
        val_tiles.extend(scene_to_tiles[scene])

    print(f"\n{'='*60}")
    print("SCENE-WISE DATA SPLIT")
    print(f"{'='*60}")
    print(f"Total unique scenes: {len(unique_scenes)}")
    print(f"Total tiles: {len(scenes)}")
    print(f"\nTrain: {len(train_scenes)} scenes -> {len(train_tiles)} tiles")
    print(f"Val:   {len(val_scenes)} scenes -> {len(val_tiles)} tiles")
    print(f"{'='*60}\n")

    return train_tiles, val_tiles


def get_train_transforms(config: TrainingConfig) -> A.Compose:
    """
    Get Albumentations transforms for training.

    Includes:
    - Geometric: RandomRotate90, Flips, ShiftScaleRotate
    - Photometric: ColorJitter (RandomBrightnessContrast, HueSaturationValue)
    - Normalization
    """
    return A.Compose(
        [
            # Geometric transforms
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=45,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),

            # Photometric transforms (applied to images only, not mask)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=1.0,
                ),
            ], p=0.5),

            # Blur and noise (occasionally)
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
            ], p=0.2),

            # Normalization
            A.Normalize(
                mean=config.normalize_mean,
                std=config.normalize_std,
                max_pixel_value=255.0,
            ),

            # Convert to tensor
            ToTensorV2(),
        ],
        # Apply same geometric transforms to all images and mask
        additional_targets={
            "post_image": "image",
            "diff_image": "image",
        },
    )


def get_val_transforms(config: TrainingConfig) -> A.Compose:
    """
    Get Albumentations transforms for validation.

    Only includes normalization - no augmentation.
    """
    return A.Compose(
        [
            # Normalization only
            A.Normalize(
                mean=config.normalize_mean,
                std=config.normalize_std,
                max_pixel_value=255.0,
            ),

            # Convert to tensor
            ToTensorV2(),
        ],
        additional_targets={
            "post_image": "image",
            "diff_image": "image",
        },
    )


class SatelliteDataset(Dataset):
    """
    PyTorch Dataset for xBD Satellite Damage Detection.

    Loads:
    - pre_img: Pre-disaster satellite image
    - post_img: Post-disaster satellite image
    - diff_img: Difference image (|post - pre|)
    - mask: Damage segmentation mask (5 classes)

    Handles:
    - Albumentations transforms (applied consistently to all images)
    - Missing diff images (computes on-the-fly)
    - IGNORE_INDEX for unlabeled pixels
    """

    def __init__(
        self,
        tile_ids: List[str],
        scenes: Dict[str, Dict[str, Path]],
        transforms: Optional[A.Compose] = None,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Args:
            tile_ids: List of tile IDs to include
            scenes: Dictionary mapping tile_id to file paths
            transforms: Albumentations transform pipeline
            config: Training configuration
        """
        self.tile_ids = tile_ids
        self.scenes = scenes
        self.transforms = transforms
        self.config = config or get_config()

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and transform a single sample.

        Returns:
            Dictionary with:
            - pre_img: (3, H, W) tensor
            - post_img: (3, H, W) tensor
            - diff_img: (3, H, W) tensor
            - mask: (H, W) tensor
            - tile_id: string identifier
        """
        tile_id = self.tile_ids[idx]
        scene_data = self.scenes[tile_id]

        # Load images (BGR -> RGB)
        pre_img = cv2.imread(str(scene_data["pre"]))
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)

        post_img = cv2.imread(str(scene_data["post"]))
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

        # Load or compute diff image
        if scene_data["diff"] is not None:
            diff_img = cv2.imread(str(scene_data["diff"]))
            diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)
        else:
            # Compute diff on-the-fly
            diff_img = cv2.absdiff(post_img, pre_img)

        # Load mask (grayscale)
        mask = cv2.imread(str(scene_data["mask"]), cv2.IMREAD_GRAYSCALE)

        # Apply transforms
        if self.transforms is not None:
            transformed = self.transforms(
                image=pre_img,
                post_image=post_img,
                diff_image=diff_img,
                mask=mask,
            )
            pre_img = transformed["image"]
            post_img = transformed["post_image"]
            diff_img = transformed["diff_image"]
            mask = transformed["mask"]
        else:
            # Convert to tensor manually if no transforms
            pre_img = torch.from_numpy(pre_img).permute(2, 0, 1).float() / 255.0
            post_img = torch.from_numpy(post_img).permute(2, 0, 1).float() / 255.0
            diff_img = torch.from_numpy(diff_img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        # Ensure mask is long tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        mask = mask.long()

        return {
            "pre_img": pre_img,
            "post_img": post_img,
            "diff_img": diff_img,
            "mask": mask,
            "tile_id": tile_id,
        }


def get_dataloaders(
    config: TrainingConfig,
    data_root: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation data loaders.

    Args:
        config: Training configuration
        data_root: Override data root path

    Returns:
        (train_loader, val_loader, metadata)
    """
    data_root = data_root or config.data_root

    # Discover all scenes
    print(f"[INFO] Discovering scenes from: {data_root}")
    scenes = discover_scenes(data_root)

    if len(scenes) == 0:
        raise ValueError(f"No valid scenes found in {data_root}")

    print(f"[INFO] Found {len(scenes)} valid tile-scene pairs")

    # Scene-wise split
    train_tiles, val_tiles = scene_wise_split(
        scenes,
        train_ratio=config.train_split,
        random_seed=config.random_seed,
    )

    # Create datasets
    train_dataset = SatelliteDataset(
        tile_ids=train_tiles,
        scenes=scenes,
        transforms=get_train_transforms(config),
        config=config,
    )

    val_dataset = SatelliteDataset(
        tile_ids=val_tiles,
        scenes=scenes,
        transforms=get_val_transforms(config),
        config=config,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Metadata for logging
    metadata = {
        "total_scenes": len(scenes),
        "train_tiles": len(train_tiles),
        "val_tiles": len(val_tiles),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
    }

    return train_loader, val_loader, metadata


def verify_dataset(config: TrainingConfig, num_samples: int = 3):
    """
    Verify dataset loading by displaying sample images.

    Useful for debugging data pipeline issues.
    """
    import matplotlib.pyplot as plt

    train_loader, val_loader, metadata = get_dataloaders(config)

    print(f"\n{'='*60}")
    print("DATASET VERIFICATION")
    print(f"{'='*60}")
    print(f"Train batches: {metadata['train_batches']}")
    print(f"Val batches: {metadata['val_batches']}")

    # Get a batch
    batch = next(iter(train_loader))

    print(f"\nBatch contents:")
    print(f"  pre_img: {batch['pre_img'].shape}")
    print(f"  post_img: {batch['post_img'].shape}")
    print(f"  diff_img: {batch['diff_img'].shape}")
    print(f"  mask: {batch['mask'].shape}")
    print(f"  mask unique values: {torch.unique(batch['mask']).tolist()}")

    # Visualize
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for i in range(min(num_samples, batch['pre_img'].shape[0])):
        # Denormalize for visualization
        mean = torch.tensor(config.normalize_mean).view(3, 1, 1)
        std = torch.tensor(config.normalize_std).view(3, 1, 1)

        pre = batch['pre_img'][i] * std + mean
        post = batch['post_img'][i] * std + mean
        diff = batch['diff_img'][i] * std + mean
        mask = batch['mask'][i]

        # Plot
        axes[i, 0].imshow(pre.permute(1, 2, 0).clamp(0, 1).numpy())
        axes[i, 0].set_title(f"Pre-disaster")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(post.permute(1, 2, 0).clamp(0, 1).numpy())
        axes[i, 1].set_title(f"Post-disaster")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(diff.permute(1, 2, 0).clamp(0, 1).numpy())
        axes[i, 2].set_title(f"Difference")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=4)
        axes[i, 3].set_title(f"Mask (unique: {torch.unique(mask).tolist()})")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(config.visualization_dir / "dataset_verification.png", dpi=150)
    plt.close()

    print(f"\nVisualization saved to: {config.visualization_dir / 'dataset_verification.png'}")
