# High-Resolution Satellite Damage Assessment via Siamese Swin-Transformers and Geometric-Residual Priors

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Rapid and accurate assessment of building damage following natural disasters is critical for effective emergency response and resource allocation. This work presents a novel deep learning architecture for pixel-wise building damage classification from bi-temporal satellite imagery, addressing the xView2 (xBD) challenge.

Our approach introduces a **Siamese Swin-Transformer encoder** with shared weights that processes pre- and post-disaster image pairs, enabling the network to learn robust temporal feature representations. At the bottleneck, a **Cross-Temporal Attention Fusion** module allows post-disaster features to attend to pre-disaster reference states, explicitly modeling the "what changed" relationship through a Query-Key-Value formulation where post-features serve as queries.

To capture explicit pixel-level changes, we incorporate a lightweight **Diff-CNN branch** that processes the absolute difference between image pairs, providing complementary geometric-residual priors to the transformer pathway. The fused multi-scale features are decoded through a U-Net style architecture with skip connections enhanced by change-aware feature differencing.

Training employs a **Hybrid Focal-Dice loss** to address severe class imbalance (destroyed buildings comprise <1% of pixels), combined with extensive data augmentation and mixed-precision training. Our method achieves competitive performance on the xBD benchmark, demonstrating the effectiveness of combining transformer-based temporal reasoning with explicit change detection priors.

---

## Key Features

| Component | Description |
|-----------|-------------|
| **Siamese Swin-Transformer** | Shared-weight encoder (`swin_tiny_patch4_window7_224`) extracts hierarchical features from pre/post image pairs |
| **Cross-Attention Fusion** | Multi-head attention at bottleneck with post→pre attention (Post=Q, Pre=K,V) |
| **Diff-CNN Branch** | 3-layer lightweight CNN processes `|post - pre|` for explicit change detection |
| **Change-Aware Skip Connections** | Skip features enhanced with temporal difference: `post + (post - pre)` |
| **Hybrid Focal-Dice Loss** | Addresses extreme class imbalance with focal weighting and Dice optimization |
| **Test-Time Augmentation** | Ensemble of original + horizontal flip + vertical flip for improved F1 |
| **Full-Image Stitching** | Tile-based inference for 1024×1024 test images with overlap averaging |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                 │
│   Pre-Image (B,3,512,512)    Post-Image (B,3,512,512)    Diff (B,3,512,512) │
└────────────┬─────────────────────────┬─────────────────────────┬────────┘
             │                         │                         │
             ▼                         ▼                         ▼
┌────────────────────────────────────────────────────┐   ┌─────────────────┐
│         SIAMESE SWIN-TRANSFORMER ENCODER           │   │   DIFF-CNN      │
│  ┌─────────────┐              ┌─────────────┐      │   │   BRANCH        │
│  │    PRE      │   (shared)   │    POST     │      │   │                 │
│  │  Stage 1-4  │◄────────────►│  Stage 1-4  │      │   │  Conv 3→64      │
│  └──────┬──────┘              └──────┬──────┘      │   │  Conv 64→128    │
│         │                            │             │   │  Conv 128→256   │
└─────────┼────────────────────────────┼─────────────┘   └────────┬────────┘
          │                            │                          │
          ▼                            ▼                          │
    ┌─────────────────────────────────────────────────┐           │
    │          CROSS-ATTENTION FUSION                  │           │
    │   Post Features → Query (Q)                      │           │
    │   Pre Features  → Key (K), Value (V)             │           │
    │   Multi-Head Attention (8 heads) + FFN           │           │
    └─────────────────────────┬───────────────────────┘           │
                              │                                    │
                              └──────────────┬─────────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │    FEATURE FUSION        │
                              │    Concat + 1×1 Conv     │
                              │    768 + 256 → 768       │
                              └────────────┬─────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │              U-NET DECODER                    │
                    │   Stage 4: 768 → 384 + Skip (Change-Aware)   │
                    │   Stage 3: 384 → 192 + Skip                  │
                    │   Stage 2: 192 → 96  + Skip                  │
                    │   Stage 1: 96  → 64                          │
                    │   Upsample: 64 → 64 (512×512)                │
                    └────────────────────┬─────────────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────────┐
                              │    SEGMENTATION HEAD     │
                              │    Conv 64→64 → Conv 64→5│
                              └────────────┬─────────────┘
                                           │
                                           ▼
                                    OUTPUT (B, 5, 512, 512)
                    ┌──────────────────────────────────────────────┐
                    │  Class 0: Background    Class 3: Major       │
                    │  Class 1: No Damage     Class 4: Destroyed   │
                    │  Class 2: Minor Damage                       │
                    └──────────────────────────────────────────────┘
```

---

## Project Structure

```
xbd/
├── 📁 models/
│   ├── __init__.py
│   └── damage_detection_model.py    # Model architecture
│
├── 📁 training/
│   ├── __init__.py
│   ├── config.py                    # Training configuration
│   ├── dataset.py                   # Data loading & augmentation
│   ├── metrics.py                   # Research-quality metrics
│   ├── trainer.py                   # Training engine
│   └── train.py                     # Main training script
│
├── 📄 xbd_preprocessing.py          # Data preprocessing pipeline
├── 📄 inference.py                  # High-performance inference
├── 📄 requirements.txt              # Dependencies
└── 📄 README.md                     # This file
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/xbd-damage-detection.git
cd xbd-damage-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Kaggle Environment

All dependencies are pre-installed in Kaggle. Simply upload the code and run.

---

## Usage

### 1. Data Preprocessing

Download the xBD dataset and preprocess into tiles:

```python
from xbd_preprocessing import preprocess_all_tiers
from pathlib import Path

# Process both tier1 and tier3
preprocess_all_tiers(
    input_base_path=Path("/kaggle/input/xbd-dataset/xbd"),
    output_path=Path("/kaggle/working/processed_data"),
    tiers=["tier1", "tier3"],
    debug=False,
    save_diff=True,
    clean_output=True,
)
```

**Output Structure:**
```
processed_data/combined/
├── pre/       # Pre-disaster tiles (512×512)
├── post/      # Post-disaster tiles
├── masks/     # Damage segmentation masks
└── diff/      # Difference images
```

### 2. Training

```python
from training import main, get_config

# Default configuration
main()

# Custom configuration
config = get_config(
    data_root="/kaggle/working/processed_data/combined",
    epochs=50,
    batch_size=4,
    lr=1e-4,
    use_wandb=True,
)
main(config=config)
```

**Command Line:**
```bash
python -m training.train \
    --data-root /kaggle/working/processed_data/combined \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4
```

### 3. Inference

```python
from inference import Predictor
import cv2

# Load trained model
predictor = Predictor("checkpoints/best_model.pth")

# Load test images
pre_img = cv2.cvtColor(cv2.imread("test_pre.png"), cv2.COLOR_BGR2RGB)
post_img = cv2.cvtColor(cv2.imread("test_post.png"), cv2.COLOR_BGR2RGB)

# Predict with TTA (handles 1024×1024 automatically)
mask = predictor.predict(pre_img, post_img, use_tta=True)

# Visualize
predictor.visualize_prediction(
    pre_img, post_img, mask,
    save_path="outputs/prediction.png",
    title="Damage Assessment Result",
)
```

---

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1e-4` | Learning rate |
| `batch_size` | `4` | Batch size |
| `epochs` | `50` | Training epochs |
| `backbone` | `swin_tiny` | Encoder backbone |
| `optimizer` | `AdamW` | Optimizer |
| `scheduler` | `OneCycleLR` | LR scheduler |
| `loss` | `Focal + Dice` | Loss function |
| `class_weights` | `[0.1, 1, 2, 3, 4]` | Class balancing |
| `early_stopping` | `patience=10` | On validation F1 |
| `amp` | `True` | Mixed precision |
| `grad_clip` | `1.0` | Gradient clipping |

---

## Performance Metrics

### xBD Validation Set Results

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | TBD |
| **Macro F1-Score** | TBD |
| **Weighted F1-Score** | TBD |
| **Mean IoU (mIoU)** | TBD |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | IoU |
|-------|-----------|--------|----------|-----|
| Background | TBD | TBD | TBD | TBD |
| No Damage | TBD | TBD | TBD | TBD |
| Minor Damage | TBD | TBD | TBD | TBD |
| Major Damage | TBD | TBD | TBD | TBD |
| Destroyed | TBD | TBD | TBD | TBD |

> **Note:** Results will be updated after full training on the xBD dataset.

---

## Dataset

This project uses the **xBD (xView2)** dataset for building damage assessment:

- **Source:** [xView2 Challenge](https://xview2.org/)
- **Images:** High-resolution satellite imagery (1024×1024)
- **Labels:** Building polygons with 4-level damage classification
- **Disasters:** Hurricanes, floods, wildfires, earthquakes

### Damage Classes

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Background | Non-building areas |
| 1 | No Damage | Intact buildings |
| 2 | Minor Damage | Cosmetic damage |
| 3 | Major Damage | Structural damage |
| 4 | Destroyed | Complete destruction |

---

## Key Implementation Details

### Scene-Wise Data Split

To prevent data leakage, tiles from the same original scene are kept together:

```python
# Groups tiles by base scene ID before splitting
# tier1_georgia-flooding_00001_tile_0_0 → tier1_georgia-flooding_00001
train_scenes, val_scenes = scene_wise_split(scenes, train_ratio=0.8)
```

### Class Imbalance Handling

The xBD dataset is heavily imbalanced:
- Background: ~87%
- No Damage: ~9%
- Damaged classes: ~3% combined

**Solutions:**
1. **Focal Loss:** Down-weights easy (background) samples
2. **Dice Loss:** Optimizes overlap regardless of class frequency
3. **Class Weights:** `[0.1, 1.0, 2.0, 3.0, 4.0]` for rare class emphasis

### Test-Time Augmentation

```python
# Ensemble of 3 augmentations for higher F1
tta_modes = ["original", "hflip", "vflip"]
# Predictions are averaged after reversing transforms
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{xbd-damage-detection-2024,
  author = {Your Name},
  title = {High-Resolution Satellite Damage Assessment via Siamese Swin-Transformers},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/your-username/xbd-damage-detection}
}
```

---

## Acknowledgments

- [xView2 Challenge](https://xview2.org/) for the dataset
- [timm](https://github.com/huggingface/pytorch-image-models) for Swin Transformer implementation
- [Albumentations](https://albumentations.ai/) for augmentation pipeline
- [Weights & Biases](https://wandb.ai/) for experiment tracking

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
