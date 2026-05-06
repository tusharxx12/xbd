"""
Models package for Satellite Damage Detection

Contains:
- SatelliteDamageDetectionModel: Main model architecture
- DamageDetectionLoss: Combined loss function
- create_model: Factory function
"""

from .damage_detection_model import (
    SatelliteDamageDetectionModel,
    DamageDetectionLoss,
    CrossAttentionFusion,
    DiffCNNBranch,
    DecoderBlock,
    SwinEncoderWrapper,
    create_model,
)

__all__ = [
    'SatelliteDamageDetectionModel',
    'DamageDetectionLoss',
    'CrossAttentionFusion',
    'DiffCNNBranch',
    'DecoderBlock',
    'SwinEncoderWrapper',
    'create_model',
]
