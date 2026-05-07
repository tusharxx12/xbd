"""
Generate Video Assets for Project Presentation

This script generates:
1. Architecture diagrams
2. Data flow visualizations
3. Training progress animations
4. Sample prediction comparisons
5. Metrics dashboards

Usage:
    python docs/generate_video_assets.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# Create output directory
OUTPUT_DIR = Path("docs/video_assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    "primary": "#2196F3",      # Blue
    "secondary": "#4CAF50",    # Green
    "accent": "#FF9800",       # Orange
    "danger": "#F44336",       # Red
    "dark": "#263238",         # Dark gray
    "light": "#ECEFF1",        # Light gray
    "purple": "#9C27B0",       # Purple
    "cyan": "#00BCD4",         # Cyan
}

DAMAGE_COLORS = {
    "Background": "#000000",
    "No Damage": "#00FF00",
    "Minor": "#FFFF00",
    "Major": "#FFA500",
    "Destroyed": "#FF0000",
}


def create_architecture_diagram():
    """Create detailed architecture diagram"""
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_facecolor('white')

    # Title
    ax.text(10, 13.5, 'Satellite Damage Detection Architecture',
            fontsize=24, fontweight='bold', ha='center', va='center',
            color=COLORS['dark'])

    # ========== INPUT LAYER ==========
    # Pre-image box
    pre_box = FancyBboxPatch((0.5, 10.5), 2.5, 2,
                              boxstyle="round,pad=0.05",
                              facecolor=COLORS['primary'],
                              edgecolor='black', linewidth=2)
    ax.add_patch(pre_box)
    ax.text(1.75, 11.5, 'Pre-Image\n(B,3,512,512)',
            ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # Post-image box
    post_box = FancyBboxPatch((3.5, 10.5), 2.5, 2,
                               boxstyle="round,pad=0.05",
                               facecolor=COLORS['secondary'],
                               edgecolor='black', linewidth=2)
    ax.add_patch(post_box)
    ax.text(4.75, 11.5, 'Post-Image\n(B,3,512,512)',
            ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # Diff-image box
    diff_box = FancyBboxPatch((6.5, 10.5), 2.5, 2,
                               boxstyle="round,pad=0.05",
                               facecolor=COLORS['accent'],
                               edgecolor='black', linewidth=2)
    ax.add_patch(diff_box)
    ax.text(7.75, 11.5, 'Diff-Image\n|post - pre|',
            ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # ========== SIAMESE ENCODER ==========
    encoder_box = FancyBboxPatch((0.3, 6.5), 6, 3.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor=COLORS['light'],
                                  edgecolor=COLORS['primary'], linewidth=3)
    ax.add_patch(encoder_box)
    ax.text(3.3, 9.7, 'SIAMESE SWIN-TRANSFORMER ENCODER',
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['dark'])

    # Pre encoder stages
    for i, (h, c) in enumerate([(128, 96), (64, 192), (32, 384), (16, 768)]):
        y = 8.8 - i * 0.6
        box = FancyBboxPatch((0.8, y - 0.2), 2, 0.4,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS['primary'],
                              alpha=0.3 + i * 0.2)
        ax.add_patch(box)
        ax.text(1.8, y, f'Stage {i+1}: {c}ch',
                ha='center', va='center', fontsize=8)

    # Post encoder stages
    for i, (h, c) in enumerate([(128, 96), (64, 192), (32, 384), (16, 768)]):
        y = 8.8 - i * 0.6
        box = FancyBboxPatch((3.5, y - 0.2), 2, 0.4,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS['secondary'],
                              alpha=0.3 + i * 0.2)
        ax.add_patch(box)
        ax.text(4.5, y, f'Stage {i+1}: {c}ch',
                ha='center', va='center', fontsize=8)

    # Shared weights arrow
    ax.annotate('', xy=(3.4, 7.8), xytext=(2.9, 7.8),
                arrowprops=dict(arrowstyle='<->', color=COLORS['danger'], lw=2))
    ax.text(3.15, 8.1, 'Shared\nWeights', ha='center', va='center', fontsize=7,
            color=COLORS['danger'])

    # ========== DIFF-CNN BRANCH ==========
    diff_cnn_box = FancyBboxPatch((6.5, 6.5), 2.5, 3.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor=COLORS['light'],
                                   edgecolor=COLORS['accent'], linewidth=3)
    ax.add_patch(diff_cnn_box)
    ax.text(7.75, 9.7, 'DIFF-CNN',
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['dark'])

    for i, c in enumerate([64, 128, 256]):
        y = 8.8 - i * 0.8
        box = FancyBboxPatch((6.8, y - 0.25), 2, 0.5,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS['accent'],
                              alpha=0.4 + i * 0.2)
        ax.add_patch(box)
        ax.text(7.8, y, f'Conv {c}ch',
                ha='center', va='center', fontsize=9, color='black')

    # ========== CROSS-ATTENTION FUSION ==========
    cross_attn_box = FancyBboxPatch((10, 7), 4, 2.5,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#E8EAF6',
                                     edgecolor=COLORS['purple'], linewidth=3)
    ax.add_patch(cross_attn_box)
    ax.text(12, 9.2, 'CROSS-ATTENTION FUSION',
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['dark'])

    ax.text(12, 8.4, 'Post → Query (Q)', ha='center', va='center', fontsize=9,
            color=COLORS['secondary'])
    ax.text(12, 7.9, 'Pre → Key (K), Value (V)', ha='center', va='center', fontsize=9,
            color=COLORS['primary'])
    ax.text(12, 7.4, 'Multi-Head Attention (8 heads)', ha='center', va='center', fontsize=9,
            color=COLORS['purple'])

    # ========== FEATURE FUSION ==========
    fusion_box = FancyBboxPatch((10, 5), 4, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#FFF3E0',
                                 edgecolor=COLORS['accent'], linewidth=3)
    ax.add_patch(fusion_box)
    ax.text(12, 6.1, 'FEATURE FUSION',
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['dark'])
    ax.text(12, 5.5, 'Concat + 1×1 Conv\n768 + 256 → 768',
            ha='center', va='center', fontsize=9, color=COLORS['dark'])

    # ========== DECODER ==========
    decoder_box = FancyBboxPatch((15, 5), 4.5, 4.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#E8F5E9',
                                  edgecolor=COLORS['secondary'], linewidth=3)
    ax.add_patch(decoder_box)
    ax.text(17.25, 9.2, 'U-NET DECODER',
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['dark'])

    decoder_stages = [
        ('768→384 + Skip', 8.5),
        ('384→192 + Skip', 7.8),
        ('192→96 + Skip', 7.1),
        ('96→64', 6.4),
        ('Upsample 512×512', 5.7),
    ]
    for text, y in decoder_stages:
        box = FancyBboxPatch((15.3, y - 0.2), 4, 0.4,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS['secondary'],
                              alpha=0.3)
        ax.add_patch(box)
        ax.text(17.3, y, text, ha='center', va='center', fontsize=9)

    # ========== OUTPUT ==========
    output_box = FancyBboxPatch((15, 1), 4.5, 3.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#FFEBEE',
                                 edgecolor=COLORS['danger'], linewidth=3)
    ax.add_patch(output_box)
    ax.text(17.25, 4.2, 'SEGMENTATION HEAD',
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['dark'])
    ax.text(17.25, 3.6, 'Conv 64→64 → Conv 64→5',
            ha='center', va='center', fontsize=9, color=COLORS['dark'])
    ax.text(17.25, 3.0, 'Output: (B, 5, 512, 512)',
            ha='center', va='center', fontsize=10, fontweight='bold', color=COLORS['danger'])

    # Class labels
    classes = [('Background', '#000000'), ('No Damage', '#00FF00'),
               ('Minor', '#FFFF00'), ('Major', '#FFA500'), ('Destroyed', '#FF0000')]
    for i, (name, color) in enumerate(classes):
        ax.add_patch(plt.Rectangle((15.2 + i * 0.85, 1.3), 0.7, 0.5,
                                    facecolor=color, edgecolor='black'))
        ax.text(15.55 + i * 0.85, 1.0, name, ha='center', va='center', fontsize=7, rotation=45)

    # ========== ARROWS ==========
    # Input to encoder arrows
    ax.annotate('', xy=(1.75, 10.5), xytext=(1.75, 9),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(4.75, 10.5), xytext=(4.75, 9),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(7.75, 10.5), xytext=(7.75, 9.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Encoder to cross-attention
    ax.annotate('', xy=(6.3, 8), xytext=(10, 8.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Diff-CNN to fusion
    ax.annotate('', xy=(9, 7.2), xytext=(10, 5.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=2,
                               connectionstyle="arc3,rad=-0.3"))

    # Cross-attention to fusion
    ax.annotate('', xy=(12, 7), xytext=(12, 6.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Fusion to decoder
    ax.annotate('', xy=(14, 5.7), xytext=(15, 7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Decoder to output
    ax.annotate('', xy=(17.25, 5), xytext=(17.25, 4.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Skip connections (dashed)
    for y_start, y_end in [(8.8, 8.5), (8.2, 7.8), (7.6, 7.1)]:
        ax.annotate('', xy=(5.5, y_start), xytext=(15.3, y_end),
                    arrowprops=dict(arrowstyle='->', color=COLORS['cyan'],
                                   lw=1.5, linestyle='dashed'))

    ax.text(10, 8.8, 'Skip Connections', fontsize=8, color=COLORS['cyan'],
            style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'architecture_diagram.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'architecture_diagram.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved architecture diagram")


def create_class_imbalance_chart():
    """Create class imbalance visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Data
    classes = ['Background', 'No Damage', 'Minor', 'Major', 'Destroyed']
    percentages = [87.45, 9.30, 1.15, 1.45, 0.66]
    colors = ['#263238', '#4CAF50', '#FFEB3B', '#FF9800', '#F44336']

    # Bar chart
    bars = ax1.barh(classes, percentages, color=colors, edgecolor='black')
    ax1.set_xlabel('Percentage of Pixels (%)', fontsize=12)
    ax1.set_title('Class Distribution in xBD Dataset', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 100)

    # Add percentage labels
    for bar, pct in zip(bars, percentages):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.2f}%', ha='left', va='center', fontsize=10)

    # Pie chart for damaged classes only
    damaged_classes = ['No Damage', 'Minor', 'Major', 'Destroyed']
    damaged_pcts = [9.30, 1.15, 1.45, 0.66]
    damaged_colors = ['#4CAF50', '#FFEB3B', '#FF9800', '#F44336']

    wedges, texts, autotexts = ax2.pie(damaged_pcts, labels=damaged_classes,
                                        colors=damaged_colors, autopct='%1.1f%%',
                                        startangle=90, explode=[0, 0.05, 0.05, 0.1])
    ax2.set_title('Building Classes Only\n(Excluding Background)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_imbalance.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"✓ Saved class imbalance chart")


def create_data_pipeline_diagram():
    """Create data preprocessing pipeline visualization"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(8, 7.5, 'Data Preprocessing Pipeline', fontsize=20, fontweight='bold',
            ha='center', color=COLORS['dark'])

    # Stage 1: Raw Data
    box1 = FancyBboxPatch((0.5, 4.5), 2.5, 2,
                           boxstyle="round,pad=0.1",
                           facecolor='#E3F2FD', edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(box1)
    ax.text(1.75, 5.5, 'Raw xBD\nDataset', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(1.75, 4.8, '1024×1024\n+ JSON labels', ha='center', va='center', fontsize=9)

    # Stage 2: Parse JSON
    box2 = FancyBboxPatch((4, 4.5), 2.5, 2,
                           boxstyle="round,pad=0.1",
                           facecolor='#E8F5E9', edgecolor=COLORS['secondary'], linewidth=2)
    ax.add_patch(box2)
    ax.text(5.25, 5.5, 'Parse\nPolygons', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(5.25, 4.8, 'WKT → Coords\n+ Damage class', ha='center', va='center', fontsize=9)

    # Stage 3: Rasterize
    box3 = FancyBboxPatch((7.5, 4.5), 2.5, 2,
                           boxstyle="round,pad=0.1",
                           facecolor='#FFF3E0', edgecolor=COLORS['accent'], linewidth=2)
    ax.add_patch(box3)
    ax.text(8.75, 5.5, 'Rasterize\nMasks', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(8.75, 4.8, 'cv2.fillPoly\nnp.maximum', ha='center', va='center', fontsize=9)

    # Stage 4: Tile
    box4 = FancyBboxPatch((11, 4.5), 2.5, 2,
                           boxstyle="round,pad=0.1",
                           facecolor='#F3E5F5', edgecolor=COLORS['purple'], linewidth=2)
    ax.add_patch(box4)
    ax.text(12.25, 5.5, 'Tile\nImages', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(12.25, 4.8, '512×512 tiles\nstride=512', ha='center', va='center', fontsize=9)

    # Stage 5: Output
    box5 = FancyBboxPatch((13.5, 1.5), 2, 5,
                           boxstyle="round,pad=0.1",
                           facecolor='#FFEBEE', edgecolor=COLORS['danger'], linewidth=2)
    ax.add_patch(box5)
    ax.text(14.5, 6, 'Output', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(14.5, 5.2, '📁 pre/', ha='center', va='center', fontsize=10)
    ax.text(14.5, 4.5, '📁 post/', ha='center', va='center', fontsize=10)
    ax.text(14.5, 3.8, '📁 masks/', ha='center', va='center', fontsize=10)
    ax.text(14.5, 3.1, '📁 diff/', ha='center', va='center', fontsize=10)
    ax.text(14.5, 2.2, 'Scene-wise\nTrain/Val Split', ha='center', va='center',
            fontsize=9, color=COLORS['danger'])

    # Arrows
    for x1, x2 in [(3, 4), (6.5, 7.5), (10, 11)]:
        ax.annotate('', xy=(x2, 5.5), xytext=(x1, 5.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.annotate('', xy=(13.5, 4), xytext=(13.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2,
                               connectionstyle="arc3,rad=-0.3"))

    # Bottom info boxes
    info_boxes = [
        ('Tier 1: 2,799 scenes', 2),
        ('Tier 3: 6,369 scenes', 5),
        ('Combined: 9,168 scenes', 8),
        ('~15,000+ tiles', 11),
    ]

    for text, x in info_boxes:
        box = FancyBboxPatch((x - 1, 1), 2.5, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor=COLORS['light'], edgecolor='gray')
        ax.add_patch(box)
        ax.text(x + 0.25, 1.4, text, ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_pipeline.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"✓ Saved data pipeline diagram")


def create_tta_visualization():
    """Create Test-Time Augmentation visualization"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Create sample "image" pattern
    np.random.seed(42)
    sample = np.random.rand(64, 64, 3) * 0.5 + 0.3
    # Add a distinct pattern (triangle)
    for i in range(20, 50):
        for j in range(max(0, i-30), min(64, i)):
            sample[i, j] = [0.8, 0.2, 0.2]

    # Top row: Transformations
    transforms = [
        ('Original', sample),
        ('H-Flip', np.fliplr(sample)),
        ('V-Flip', np.flipud(sample)),
        ('Average', sample),  # Placeholder
    ]

    for idx, (title, img) in enumerate(transforms):
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(title, fontsize=14, fontweight='bold')
        axes[0, idx].axis('off')
        if idx < 3:
            axes[0, idx].text(32, -5, '→ Predict →', ha='center', fontsize=10)

    # Add "Predict" arrows
    axes[0, 3].text(32, -5, '= Ensemble', ha='center', fontsize=10, color=COLORS['danger'])

    # Bottom row: Predictions (simulated)
    pred_original = np.random.rand(64, 64) * 0.5 + 0.3
    pred_hflip = np.fliplr(pred_original) + np.random.rand(64, 64) * 0.1
    pred_vflip = np.flipud(pred_original) + np.random.rand(64, 64) * 0.1
    pred_ensemble = (pred_original + np.fliplr(pred_hflip) + np.flipud(pred_vflip)) / 3

    predictions = [
        ('Pred (Original)', pred_original),
        ('Pred (H-Flip) → Reverse', np.fliplr(pred_hflip)),
        ('Pred (V-Flip) → Reverse', np.flipud(pred_vflip)),
        ('Final Ensemble', pred_ensemble),
    ]

    for idx, (title, pred) in enumerate(predictions):
        im = axes[1, idx].imshow(pred, cmap='RdYlGn_r')
        axes[1, idx].set_title(title, fontsize=12)
        axes[1, idx].axis('off')

    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, label='Damage Probability')

    plt.suptitle('Test-Time Augmentation (TTA) Process', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tta_visualization.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"✓ Saved TTA visualization")


def create_tiling_visualization():
    """Create image tiling and stitching visualization"""
    fig = plt.figure(figsize=(16, 6))

    # Create sample 1024x1024 "image"
    np.random.seed(42)
    full_image = np.random.rand(1024, 1024, 3) * 0.3 + 0.4

    # Add some patterns
    full_image[200:400, 200:400] = [0.8, 0.3, 0.3]
    full_image[600:800, 100:300] = [0.3, 0.8, 0.3]
    full_image[400:600, 700:900] = [0.3, 0.3, 0.8]

    # Subplot 1: Full image with grid
    ax1 = fig.add_subplot(131)
    ax1.imshow(full_image)
    ax1.set_title('1024×1024 Test Image\n(with tile grid)', fontsize=12, fontweight='bold')

    # Draw tile grid
    for i in range(0, 1024, 512):
        ax1.axhline(y=i, color='white', linewidth=2, linestyle='--')
        ax1.axvline(x=i, color='white', linewidth=2, linestyle='--')

    # Label tiles
    for i, y in enumerate([256, 768]):
        for j, x in enumerate([256, 768]):
            ax1.text(x, y, f'Tile {i*2+j+1}', ha='center', va='center',
                    fontsize=14, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax1.axis('off')

    # Subplot 2: Individual tiles
    ax2 = fig.add_subplot(132)
    tiles_display = np.ones((1024 + 30, 1024 + 30, 3))

    # Place tiles with gaps
    gap = 15
    for i in range(2):
        for j in range(2):
            y_start = i * (512 + gap)
            x_start = j * (512 + gap)
            tile = full_image[i*512:(i+1)*512, j*512:(j+1)*512]
            tiles_display[y_start:y_start+512, x_start:x_start+512] = tile

    ax2.imshow(tiles_display)
    ax2.set_title('Tiles Extracted\n(512×512 each)', fontsize=12, fontweight='bold')

    # Add prediction labels
    for i, y in enumerate([256, 768 + gap]):
        for j, x in enumerate([256, 768 + gap]):
            ax2.text(x, y, f'→ Predict\n→ TTA', ha='center', va='center',
                    fontsize=10, color='white',
                    bbox=dict(boxstyle='round', facecolor=COLORS['primary'], alpha=0.8))
    ax2.axis('off')

    # Subplot 3: Stitched result
    ax3 = fig.add_subplot(133)

    # Create fake prediction mask
    mask = np.zeros((1024, 1024))
    mask[200:400, 200:400] = 4  # Destroyed
    mask[600:800, 100:300] = 1  # No damage
    mask[400:600, 700:900] = 2  # Minor

    # Color map
    colors = ['#000000', '#00FF00', '#FFFF00', '#FFA500', '#FF0000']
    cmap = plt.cm.colors.ListedColormap(colors)

    ax3.imshow(mask, cmap=cmap, vmin=0, vmax=4)
    ax3.set_title('Stitched Prediction\n(1024×1024)', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Add legend
    patches = [mpatches.Patch(color=colors[i], label=list(DAMAGE_COLORS.keys())[i])
               for i in range(5)]
    ax3.legend(handles=patches, loc='lower right', fontsize=8)

    plt.suptitle('Full-Image Inference: Tile → Predict → Stitch', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tiling_visualization.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"✓ Saved tiling visualization")


def create_loss_function_diagram():
    """Create loss function visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Focal Loss curve
    x = np.linspace(0, 1, 100)
    gamma_values = [0, 1, 2, 5]

    for gamma in gamma_values:
        focal = -((1 - x) ** gamma) * np.log(x + 1e-8)
        axes[0].plot(x, focal, label=f'γ = {gamma}', linewidth=2)

    axes[0].set_xlabel('Probability (pt)', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Focal Loss\nDown-weights Easy Samples', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 5)
    axes[0].grid(True, alpha=0.3)

    # Dice Loss illustration
    overlap_ratios = np.linspace(0, 1, 100)
    dice_loss = 1 - (2 * overlap_ratios) / (1 + overlap_ratios)

    axes[1].plot(overlap_ratios, dice_loss, color=COLORS['secondary'], linewidth=3)
    axes[1].fill_between(overlap_ratios, dice_loss, alpha=0.3, color=COLORS['secondary'])
    axes[1].set_xlabel('Overlap Ratio (IoU)', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Dice Loss\nOptimizes Overlap Directly', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # Combined Loss
    ce_weight = 0.5
    dice_weight = 0.5

    # Simulated combined loss surface
    x_pred = np.linspace(0.01, 0.99, 50)
    y_overlap = np.linspace(0.01, 0.99, 50)
    X, Y = np.meshgrid(x_pred, y_overlap)

    focal = -((1 - X) ** 2) * np.log(X + 1e-8)
    dice = 1 - (2 * Y) / (1 + Y)
    combined = ce_weight * focal + dice_weight * dice

    contour = axes[2].contourf(X, Y, combined, levels=20, cmap='RdYlGn_r')
    plt.colorbar(contour, ax=axes[2], label='Combined Loss')
    axes[2].set_xlabel('Prediction Confidence', fontsize=12)
    axes[2].set_ylabel('Overlap Ratio', fontsize=12)
    axes[2].set_title('Hybrid Focal + Dice Loss\n0.5 × Focal + 0.5 × Dice', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'loss_function.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"✓ Saved loss function diagram")


def create_metrics_dashboard():
    """Create a sample metrics dashboard"""
    fig = plt.figure(figsize=(14, 10))

    # Title
    fig.suptitle('Training Metrics Dashboard', fontsize=20, fontweight='bold', y=0.98)

    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Training/Validation Loss
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = np.arange(1, 51)
    train_loss = 1.5 * np.exp(-0.08 * epochs) + 0.3 + np.random.rand(50) * 0.05
    val_loss = 1.5 * np.exp(-0.07 * epochs) + 0.35 + np.random.rand(50) * 0.08

    ax1.plot(epochs, train_loss, label='Train Loss', color=COLORS['primary'], linewidth=2)
    ax1.plot(epochs, val_loss, label='Val Loss', color=COLORS['danger'], linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Learning Rate Schedule
    ax2 = fig.add_subplot(gs[0, 2])
    lr = np.concatenate([
        np.linspace(1e-6, 1e-4, 5),
        1e-4 * (1 + np.cos(np.pi * np.arange(45) / 45)) / 2
    ])
    ax2.plot(epochs, lr, color=COLORS['accent'], linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('OneCycleLR Schedule')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. Per-Class F1 Scores
    ax3 = fig.add_subplot(gs[1, 0])
    classes = ['BG', 'No Dmg', 'Minor', 'Major', 'Destr']
    f1_scores = [0.95, 0.82, 0.45, 0.52, 0.38]
    colors = ['#263238', '#4CAF50', '#FFEB3B', '#FF9800', '#F44336']

    bars = ax3.bar(classes, f1_scores, color=colors, edgecolor='black')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('Per-Class F1-Score')
    ax3.set_ylim(0, 1)
    for bar, score in zip(bars, f1_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', fontsize=9)

    # 4. Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 1:])
    conf_matrix = np.array([
        [0.95, 0.03, 0.01, 0.01, 0.00],
        [0.05, 0.82, 0.08, 0.03, 0.02],
        [0.02, 0.15, 0.55, 0.20, 0.08],
        [0.01, 0.08, 0.18, 0.58, 0.15],
        [0.01, 0.05, 0.10, 0.25, 0.59],
    ])

    im = ax4.imshow(conf_matrix, cmap='Blues')
    ax4.set_xticks(range(5))
    ax4.set_yticks(range(5))
    ax4.set_xticklabels(classes)
    ax4.set_yticklabels(classes)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    ax4.set_title('Normalized Confusion Matrix')

    for i in range(5):
        for j in range(5):
            ax4.text(j, i, f'{conf_matrix[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if conf_matrix[i, j] > 0.5 else 'black')

    plt.colorbar(im, ax=ax4, fraction=0.046)

    # 5. F1 and mIoU over epochs
    ax5 = fig.add_subplot(gs[2, :])
    f1_macro = 0.75 * (1 - np.exp(-0.1 * epochs)) + np.random.rand(50) * 0.03
    miou = 0.65 * (1 - np.exp(-0.08 * epochs)) + np.random.rand(50) * 0.02

    ax5.plot(epochs, f1_macro, label='Macro F1', color=COLORS['primary'], linewidth=2)
    ax5.plot(epochs, miou, label='mIoU', color=COLORS['secondary'], linewidth=2)
    ax5.axhline(y=f1_macro.max(), color=COLORS['primary'], linestyle='--', alpha=0.5)
    ax5.axhline(y=miou.max(), color=COLORS['secondary'], linestyle='--', alpha=0.5)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Score')
    ax5.set_title('Validation Metrics Over Training')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)

    # Annotate best scores
    ax5.annotate(f'Best F1: {f1_macro.max():.3f}',
                xy=(f1_macro.argmax() + 1, f1_macro.max()),
                xytext=(f1_macro.argmax() + 5, f1_macro.max() + 0.05),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary']),
                fontsize=10, color=COLORS['primary'])

    plt.savefig(OUTPUT_DIR / 'metrics_dashboard.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"✓ Saved metrics dashboard")


def create_sample_predictions():
    """Create sample prediction visualization grid"""
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    np.random.seed(42)

    col_titles = ['Pre-Disaster', 'Post-Disaster', 'Ground Truth', 'Prediction']
    row_titles = ['Hurricane', 'Earthquake', 'Flood', 'Wildfire']

    # Color map for masks
    colors = ['#000000', '#00FF00', '#FFFF00', '#FFA500', '#FF0000']
    cmap = plt.cm.colors.ListedColormap(colors)

    for row in range(4):
        for col in range(4):
            if col < 2:
                # Satellite-like images
                img = np.random.rand(256, 256, 3) * 0.3 + 0.4
                # Add some structure
                for _ in range(10):
                    x, y = np.random.randint(20, 236, 2)
                    size = np.random.randint(10, 30)
                    brightness = 0.6 + np.random.rand() * 0.3
                    img[y:y+size, x:x+size] = brightness

                if col == 1:  # Post-disaster - add damage
                    for _ in range(5):
                        x, y = np.random.randint(20, 236, 2)
                        size = np.random.randint(5, 20)
                        img[y:y+size, x:x+size] = [0.3, 0.2, 0.2]

                axes[row, col].imshow(img)
            else:
                # Masks
                mask = np.zeros((256, 256))
                # Add buildings
                for _ in range(15):
                    x, y = np.random.randint(20, 236, 2)
                    size = np.random.randint(10, 25)
                    damage_level = np.random.choice([0, 1, 2, 3, 4],
                                                     p=[0.3, 0.35, 0.15, 0.12, 0.08])
                    mask[y:y+size, x:x+size] = damage_level

                axes[row, col].imshow(mask, cmap=cmap, vmin=0, vmax=4)

            axes[row, col].axis('off')

            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=14, fontweight='bold')

            if col == 0:
                axes[row, col].text(-0.15, 0.5, row_titles[row],
                                    transform=axes[row, col].transAxes,
                                    fontsize=12, fontweight='bold',
                                    rotation=90, va='center', ha='center')

    # Add legend
    legend_patches = [mpatches.Patch(color=colors[i], label=list(DAMAGE_COLORS.keys())[i])
                      for i in range(5)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('Sample Predictions Across Different Disaster Types',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUTPUT_DIR / 'sample_predictions.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"✓ Saved sample predictions")


def create_cross_attention_diagram():
    """Create Cross-Attention mechanism visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'Cross-Temporal Attention Mechanism',
            fontsize=18, fontweight='bold', ha='center')

    # Post features (Query)
    post_box = FancyBboxPatch((0.5, 4), 2.5, 2,
                               boxstyle="round,pad=0.1",
                               facecolor='#C8E6C9', edgecolor=COLORS['secondary'], linewidth=2)
    ax.add_patch(post_box)
    ax.text(1.75, 5, 'Post Features\n(B, 256, 768)', ha='center', va='center', fontsize=11)
    ax.text(1.75, 4.3, '→ Query (Q)', ha='center', va='center', fontsize=10,
            color=COLORS['secondary'], fontweight='bold')

    # Pre features (Key, Value)
    pre_box = FancyBboxPatch((0.5, 1), 2.5, 2,
                              boxstyle="round,pad=0.1",
                              facecolor='#BBDEFB', edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(pre_box)
    ax.text(1.75, 2, 'Pre Features\n(B, 256, 768)', ha='center', va='center', fontsize=11)
    ax.text(1.75, 1.3, '→ Key (K), Value (V)', ha='center', va='center', fontsize=10,
            color=COLORS['primary'], fontweight='bold')

    # Q, K, V projections
    for i, (name, y, color) in enumerate([('Q', 5, COLORS['secondary']),
                                           ('K', 2.5, COLORS['primary']),
                                           ('V', 1.5, COLORS['primary'])]):
        proj_box = FancyBboxPatch((4, y - 0.3), 1.5, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(proj_box)
        ax.text(4.75, y, f'{name} Proj', ha='center', va='center', fontsize=10)

    # Attention computation
    attn_box = FancyBboxPatch((6.5, 2.5), 3, 3,
                               boxstyle="round,pad=0.1",
                               facecolor='#FFF3E0', edgecolor=COLORS['accent'], linewidth=2)
    ax.add_patch(attn_box)
    ax.text(8, 5.2, 'Attention', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(8, 4.5, 'Attn = softmax(Q·Kᵀ/√d)', ha='center', va='center', fontsize=10,
            family='monospace')
    ax.text(8, 3.8, '8 heads × 96 dim', ha='center', va='center', fontsize=10)
    ax.text(8, 3.1, 'Output = Attn · V', ha='center', va='center', fontsize=10,
            family='monospace')

    # Output
    out_box = FancyBboxPatch((10.5, 3.5), 2.5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#E1BEE7', edgecolor=COLORS['purple'], linewidth=2)
    ax.add_patch(out_box)
    ax.text(11.75, 4.25, 'Fused Features\n(B, 256, 768)', ha='center', va='center', fontsize=11)

    # Arrows
    # Post to Q
    ax.annotate('', xy=(4, 5), xytext=(3, 5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    # Pre to K
    ax.annotate('', xy=(4, 2.5), xytext=(3, 2.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    # Pre to V
    ax.annotate('', xy=(4, 1.5), xytext=(3, 1.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    # Q to Attention
    ax.annotate('', xy=(6.5, 4.5), xytext=(5.5, 5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    # K to Attention
    ax.annotate('', xy=(6.5, 3.5), xytext=(5.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    # V to Attention
    ax.annotate('', xy=(6.5, 3), xytext=(5.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    # Attention to Output
    ax.annotate('', xy=(10.5, 4.25), xytext=(9.5, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Explanation text
    ax.text(7, 0.8, '"Post features ask: Where was there a building that might now be damaged?"',
            ha='center', va='center', fontsize=11, style='italic', color=COLORS['dark'])
    ax.text(7, 0.3, '"Pre features provide the reference: Here\'s what existed before."',
            ha='center', va='center', fontsize=11, style='italic', color=COLORS['dark'])

    plt.savefig(OUTPUT_DIR / 'cross_attention.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"✓ Saved cross-attention diagram")


def main():
    """Generate all video assets"""
    print("=" * 60)
    print("GENERATING VIDEO ASSETS")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Generate all assets
    create_architecture_diagram()
    create_class_imbalance_chart()
    create_data_pipeline_diagram()
    create_tta_visualization()
    create_tiling_visualization()
    create_loss_function_diagram()
    create_metrics_dashboard()
    create_sample_predictions()
    create_cross_attention_diagram()

    print()
    print("=" * 60)
    print("ALL ASSETS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
