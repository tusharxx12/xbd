"""
Configuration for Satellite Damage Detection Training

Contains all hyperparameters, paths, and training settings.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import torch


@dataclass
class TrainingConfig:
    """
    Central configuration class for training pipeline.

    All hyperparameters and settings are defined here for:
    - Easy experimentation
    - Reproducibility
    - Clear documentation
    """

    # ============================================
    # Paths
    # ============================================
    data_root: Path = Path("/kaggle/working/processed_data/combined")
    output_dir: Path = Path("/kaggle/working/outputs")
    checkpoint_dir: Path = Path("/kaggle/working/checkpoints")
    visualization_dir: Path = Path("/kaggle/working/visualizations")

    # ============================================
    # Model Configuration
    # ============================================
    backbone_name: str = "swin_tiny_patch4_window7_224"
    num_classes: int = 5
    pretrained: bool = True
    use_deep_supervision: bool = True

    # ============================================
    # Training Hyperparameters
    # ============================================
    lr: float = 1e-4
    backbone_lr_mult: float = 0.1  # Backbone learns slower
    weight_decay: float = 0.01
    batch_size: int = 4
    epochs: int = 50
    num_workers: int = 4

    # ============================================
    # Optimizer Settings
    # ============================================
    optimizer: str = "adamw"
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # ============================================
    # Scheduler Settings
    # ============================================
    scheduler: str = "onecycle"  # "onecycle" or "cosine"
    warmup_epochs: int = 3
    min_lr: float = 1e-7

    # ============================================
    # Mixed Precision & Gradient Settings
    # ============================================
    use_amp: bool = True  # Mixed precision training
    grad_clip_norm: float = 1.0  # Gradient clipping for transformer stability
    accumulation_steps: int = 1  # Gradient accumulation

    # ============================================
    # Early Stopping
    # ============================================
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_f1_macro"
    early_stopping_mode: str = "max"  # "max" for F1, "min" for loss

    # ============================================
    # Data Split
    # ============================================
    train_split: float = 0.8
    val_split: float = 0.2
    random_seed: int = 42

    # ============================================
    # Normalization Statistics (xBD Dataset)
    # ============================================
    # Computed from xBD satellite imagery
    normalize_mean: Tuple[float, float, float] = (0.3260, 0.3753, 0.2511)
    normalize_std: Tuple[float, float, float] = (0.1074, 0.0881, 0.0880)

    # ============================================
    # Special Indices
    # ============================================
    IGNORE_INDEX: int = 255  # Pixels to ignore in loss and metrics

    # ============================================
    # Loss Function Settings
    # ============================================
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    dice_weight: float = 0.5
    ce_weight: float = 0.5

    # Class weights for imbalanced data (computed from preprocessing stats)
    # Higher weights for rare classes (destroyed, major damage)
    class_weights: Optional[List[float]] = field(
        default_factory=lambda: [0.1, 1.0, 2.0, 3.0, 4.0]
    )

    # ============================================
    # Logging & Visualization
    # ============================================
    use_wandb: bool = True
    wandb_project: str = "xbd-damage-detection"
    wandb_entity: Optional[str] = None  # Set to your wandb username/team
    wandb_run_name: Optional[str] = None

    log_interval: int = 10  # Log every N batches
    save_interval: int = 5  # Save checkpoint every N epochs
    visualize_interval: int = 1  # Visualize predictions every N epochs
    num_visualizations: int = 4  # Number of samples to visualize

    # ============================================
    # Reproducibility
    # ============================================
    deterministic: bool = True

    # ============================================
    # Device Configuration
    # ============================================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """Create directories after initialization"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging"""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }


# Damage class names for visualization and logging
DAMAGE_CLASSES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed",
}

# Color map for visualization (RGB)
DAMAGE_COLORS = {
    0: (0, 0, 0),        # Background - Black
    1: (0, 255, 0),      # No Damage - Green
    2: (255, 255, 0),    # Minor Damage - Yellow
    3: (255, 165, 0),    # Major Damage - Orange
    4: (255, 0, 0),      # Destroyed - Red
}


def get_config(**kwargs) -> TrainingConfig:
    """
    Factory function to create config with optional overrides.

    Args:
        **kwargs: Override any config parameter

    Returns:
        TrainingConfig instance
    """
    return TrainingConfig(**kwargs)
