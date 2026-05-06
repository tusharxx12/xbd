"""
Training package for Satellite Damage Detection

Contains:
- config: Training configuration
- dataset: Data loading and augmentation
- metrics: Research-quality metrics engine
- trainer: Training loop with all features
- train: Main training script
"""

from .config import (
    TrainingConfig,
    get_config,
    DAMAGE_CLASSES,
    DAMAGE_COLORS,
)

from .dataset import (
    SatelliteDataset,
    get_dataloaders,
    get_train_transforms,
    get_val_transforms,
    scene_wise_split,
    discover_scenes,
)

from .metrics import (
    DamageMetrics,
    MetricTracker,
)

from .trainer import (
    Trainer,
    EarlyStopping,
)

from .train import main

__all__ = [
    # Config
    "TrainingConfig",
    "get_config",
    "DAMAGE_CLASSES",
    "DAMAGE_COLORS",
    # Dataset
    "SatelliteDataset",
    "get_dataloaders",
    "get_train_transforms",
    "get_val_transforms",
    "scene_wise_split",
    "discover_scenes",
    # Metrics
    "DamageMetrics",
    "MetricTracker",
    # Trainer
    "Trainer",
    "EarlyStopping",
    # Main
    "main",
]
