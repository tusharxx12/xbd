"""
Main Training Script for Satellite Damage Detection

This script ties everything together:
- Model initialization
- Data loading
- Loss function setup
- Training loop execution

Usage:
    python -m training.train

    # Or from Kaggle notebook:
    from training.train import main
    main()
"""

import os
import sys
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import TrainingConfig, get_config, DAMAGE_CLASSES
from training.dataset import get_dataloaders, verify_dataset
from training.metrics import DamageMetrics
from training.trainer import Trainer

from models.damage_detection_model import (
    SatelliteDamageDetectionModel,
    DamageDetectionLoss,
    create_model,
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_banner():
    """Print training banner"""
    print("\n" + "=" * 70)
    print("  🛰️  SATELLITE DAMAGE DETECTION TRAINING  🛰️")
    print("  xBD (xView2) Building Damage Assessment")
    print("=" * 70)
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def print_config(config: TrainingConfig):
    """Print training configuration"""
    print("\n📋 TRAINING CONFIGURATION")
    print("-" * 50)
    print(f"  Data root:       {config.data_root}")
    print(f"  Output dir:      {config.output_dir}")
    print(f"  Device:          {config.device}")
    print("-" * 50)
    print(f"  Backbone:        {config.backbone_name}")
    print(f"  Pretrained:      {config.pretrained}")
    print(f"  Num classes:     {config.num_classes}")
    print("-" * 50)
    print(f"  Learning rate:   {config.lr}")
    print(f"  Batch size:      {config.batch_size}")
    print(f"  Epochs:          {config.epochs}")
    print(f"  Optimizer:       {config.optimizer}")
    print(f"  Scheduler:       {config.scheduler}")
    print("-" * 50)
    print(f"  Mixed precision: {config.use_amp}")
    print(f"  Grad clipping:   {config.grad_clip_norm}")
    print(f"  Deep supervision:{config.use_deep_supervision}")
    print("-" * 50)
    print(f"  Early stopping:  patience={config.early_stopping_patience}")
    print(f"  Wandb:           {config.use_wandb}")
    print("-" * 50 + "\n")


def main(
    config: TrainingConfig = None,
    debug: bool = False,
):
    """
    Main training function.

    Args:
        config: Training configuration (optional, uses defaults if not provided)
        debug: If True, use smaller dataset for testing
    """
    print_banner()

    # Get configuration
    if config is None:
        config = get_config()

    # Override for debug mode
    if debug:
        config.epochs = 2
        config.batch_size = 2
        config.use_wandb = False
        config.num_workers = 0
        print("⚠️  DEBUG MODE: Using reduced settings")

    print_config(config)

    # Set random seed
    print(f"🎲 Setting random seed: {config.random_seed}")
    set_seed(config.random_seed)

    # Check for GPU
    if torch.cuda.is_available():
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU available, training on CPU")

    # ============================================
    # 1. Load Data
    # ============================================
    print("\n" + "=" * 50)
    print("📁 LOADING DATA")
    print("=" * 50)

    try:
        train_loader, val_loader, metadata = get_dataloaders(config)
        print(f"\n✅ Data loaded successfully!")
        print(f"   Train: {metadata['train_tiles']} tiles, {metadata['train_batches']} batches")
        print(f"   Val:   {metadata['val_tiles']} tiles, {metadata['val_batches']} batches")
    except Exception as e:
        print(f"\n❌ Failed to load data: {e}")
        print("   Make sure preprocessed data exists at:", config.data_root)
        raise

    # ============================================
    # 2. Create Model
    # ============================================
    print("\n" + "=" * 50)
    print("🏗️  CREATING MODEL")
    print("=" * 50)

    model = create_model(
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        use_deep_supervision=config.use_deep_supervision,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✅ Model created!")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size:           {total_params * 4 / 1e6:.1f} MB (FP32)")

    # ============================================
    # 3. Create Loss Function
    # ============================================
    print("\n" + "=" * 50)
    print("📉 CREATING LOSS FUNCTION")
    print("=" * 50)

    # Prepare class weights
    class_weights = None
    if config.class_weights is not None:
        class_weights = torch.tensor(config.class_weights).float()
        print(f"\n   Class weights: {config.class_weights}")

    loss_fn = DamageDetectionLoss(
        num_classes=config.num_classes,
        class_weights=class_weights,
        dice_weight=config.dice_weight,
        ce_weight=config.ce_weight,
        focal_gamma=config.focal_gamma,
        use_focal=config.use_focal_loss,
        ignore_index=config.IGNORE_INDEX,
    )

    print(f"\n✅ Loss function created!")
    print(f"   Focal loss:   {config.use_focal_loss} (gamma={config.focal_gamma})")
    print(f"   Dice weight:  {config.dice_weight}")
    print(f"   CE weight:    {config.ce_weight}")
    print(f"   Ignore index: {config.IGNORE_INDEX}")

    # ============================================
    # 4. Create Trainer
    # ============================================
    print("\n" + "=" * 50)
    print("🏋️  CREATING TRAINER")
    print("=" * 50)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    print(f"\n✅ Trainer created!")
    print(f"   Optimizer:    AdamW (lr={config.lr}, wd={config.weight_decay})")
    print(f"   Scheduler:    {config.scheduler}")
    print(f"   AMP:          {config.use_amp}")

    # ============================================
    # 5. Start Training
    # ============================================
    results = trainer.train()

    # ============================================
    # 6. Print Final Results
    # ============================================
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Results:")
    print(f"   Best Validation F1:  {results['best_val_f1']*100:.2f}%")
    print(f"   Total Training Time: {results['total_time']/60:.1f} minutes")
    print(f"\n📁 Outputs saved to:")
    print(f"   Checkpoints:     {config.checkpoint_dir}")
    print(f"   Visualizations:  {config.visualization_dir}")
    print("=" * 70 + "\n")

    return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Satellite Damage Detection Model")

    parser.add_argument(
        "--data-root",
        type=str,
        default="/kaggle/working/processed_data/combined",
        help="Path to preprocessed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced settings",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create config from arguments
    config = get_config(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_wandb=not args.no_wandb,
    )

    # Run training
    main(config=config, debug=args.debug)
