"""
Simple Video Generator using Matplotlib + MoviePy

This is an easier alternative to Manim that works out of the box.

Run with:
    python docs/video/simple_video.py

Requirements:
    pip install matplotlib numpy moviepy pillow

Output:
    docs/video/output/satellite_damage_video.mp4
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# Create output directory
OUTPUT_DIR = Path("docs/video/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR = OUTPUT_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Video settings
FPS = 30
RESOLUTION = (1920, 1080)
DURATION_PER_SLIDE = 5  # seconds

# Colors
COLORS = {
    "bg": "#1a1a2e",
    "primary": "#4fc3f7",
    "secondary": "#81c784",
    "accent": "#ffb74d",
    "danger": "#e57373",
    "purple": "#ba68c8",
    "text": "#ffffff",
    "text_dim": "#b0bec5",
}


def create_text_frame(title, subtitle=None, points=None, bg_color=COLORS["bg"]):
    """Create a single frame with text"""
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    ax.set_xlim(0, 19.2)
    ax.set_ylim(0, 10.8)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    ax.axis('off')

    # Title
    ax.text(9.6, 8.5, title, fontsize=48, color=COLORS["text"],
            ha='center', va='center', fontweight='bold')

    if subtitle:
        ax.text(9.6, 7.2, subtitle, fontsize=28, color=COLORS["primary"],
                ha='center', va='center')

    if points:
        y_start = 5.5
        for i, point in enumerate(points):
            ax.text(2, y_start - i * 1.0, f"• {point}",
                   fontsize=24, color=COLORS["text_dim"], ha='left', va='center')

    return fig, ax


def create_slide_1_title():
    """Slide 1: Title"""
    fig, ax = create_text_frame("")

    # Main title
    ax.text(9.6, 6.5, "Satellite Damage Detection", fontsize=56,
            color=COLORS["text"], ha='center', va='center', fontweight='bold')
    ax.text(9.6, 5.0, "using Siamese Swin-Transformers", fontsize=36,
            color=COLORS["primary"], ha='center', va='center')

    # Subtitle
    ax.text(9.6, 3.5, "xView2 Building Damage Assessment Challenge", fontsize=24,
            color=COLORS["text_dim"], ha='center', va='center')

    # Decorative line
    ax.plot([4, 15.2], [4.2, 4.2], color=COLORS["primary"], linewidth=2)

    return fig


def create_slide_2_problem():
    """Slide 2: The Problem"""
    fig, ax = create_text_frame("The Challenge")

    problems = [
        "🌪️  Natural disasters cause massive destruction",
        "⏱️  Manual inspection takes days or weeks",
        "🛰️  Satellite imagery available within hours",
        "🤖  AI can automate damage assessment",
    ]

    y_start = 6.0
    for i, problem in enumerate(problems):
        ax.text(3, y_start - i * 1.2, problem, fontsize=28,
                color=COLORS["text"], ha='left', va='center')

    # Goal box
    goal_box = FancyBboxPatch((3, 1.5), 13.2, 1.5, boxstyle="round,pad=0.05",
                               facecolor=COLORS["secondary"], alpha=0.2,
                               edgecolor=COLORS["secondary"], linewidth=2)
    ax.add_patch(goal_box)
    ax.text(9.6, 2.25, "Goal: Classify building damage from pre/post satellite image pairs",
            fontsize=24, color=COLORS["secondary"], ha='center', va='center', fontweight='bold')

    return fig


def create_slide_3_dataset():
    """Slide 3: Dataset"""
    fig, ax = create_text_frame("xBD Dataset")

    # Stats
    stats = [
        "📊  9,168 image pairs (1024×1024 pixels)",
        "🌍  19 disaster events worldwide",
        "🏠  ~800,000 building annotations",
    ]

    y_start = 7.0
    for i, stat in enumerate(stats):
        ax.text(2, y_start - i * 0.8, stat, fontsize=24,
                color=COLORS["text"], ha='left', va='center')

    # Damage classes
    ax.text(2, 4.5, "Damage Classes:", fontsize=28,
            color=COLORS["accent"], ha='left', va='center', fontweight='bold')

    classes = [
        ("Background", "#263238", "87.45%"),
        ("No Damage", "#4CAF50", "9.30%"),
        ("Minor Damage", "#FFEB3B", "1.15%"),
        ("Major Damage", "#FF9800", "1.45%"),
        ("Destroyed", "#F44336", "0.66%"),
    ]

    for i, (name, color, pct) in enumerate(classes):
        rect = Rectangle((2, 3.8 - i * 0.6), 0.4, 0.4, facecolor=color)
        ax.add_patch(rect)
        ax.text(2.6, 4.0 - i * 0.6, f"{name}: {pct}", fontsize=20,
                color=COLORS["text"], ha='left', va='center')

    # Warning
    ax.text(12, 3.0, "⚠️ SEVERE CLASS IMBALANCE!", fontsize=32,
            color=COLORS["danger"], ha='center', va='center', fontweight='bold')
    ax.text(12, 2.2, "Destroyed buildings: only 0.66%", fontsize=20,
            color=COLORS["danger"], ha='center', va='center')

    return fig


def create_slide_4_architecture():
    """Slide 4: Architecture Overview"""
    fig, ax = create_text_frame("Model Architecture")

    # Input boxes
    components = [
        ("Pre-Image", 3, 7, COLORS["primary"]),
        ("Post-Image", 7, 7, COLORS["secondary"]),
        ("Diff-Image", 11, 7, COLORS["accent"]),
    ]

    for name, x, y, color in components:
        box = FancyBboxPatch((x, y), 2.5, 1, boxstyle="round,pad=0.02",
                              facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 1.25, y + 0.5, name, fontsize=16, color=color,
                ha='center', va='center', fontweight='bold')

    # Siamese Encoder
    encoder_box = FancyBboxPatch((2.5, 5), 6, 1.2, boxstyle="round,pad=0.02",
                                  facecolor=COLORS["purple"], alpha=0.2,
                                  edgecolor=COLORS["purple"], linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(5.5, 5.6, "Siamese Swin-Transformer", fontsize=18,
            color=COLORS["purple"], ha='center', va='center', fontweight='bold')

    # Diff-CNN
    diff_box = FancyBboxPatch((10, 5), 3, 1.2, boxstyle="round,pad=0.02",
                               facecolor=COLORS["accent"], alpha=0.2,
                               edgecolor=COLORS["accent"], linewidth=2)
    ax.add_patch(diff_box)
    ax.text(11.5, 5.6, "Diff-CNN", fontsize=18,
            color=COLORS["accent"], ha='center', va='center', fontweight='bold')

    # Cross-Attention
    cross_box = FancyBboxPatch((4, 3.3), 5, 1, boxstyle="round,pad=0.02",
                                facecolor=COLORS["primary"], alpha=0.2,
                                edgecolor=COLORS["primary"], linewidth=2)
    ax.add_patch(cross_box)
    ax.text(6.5, 3.8, "Cross-Attention Fusion", fontsize=16,
            color=COLORS["primary"], ha='center', va='center', fontweight='bold')

    # Decoder
    decoder_box = FancyBboxPatch((6, 1.8), 4, 1, boxstyle="round,pad=0.02",
                                  facecolor=COLORS["secondary"], alpha=0.2,
                                  edgecolor=COLORS["secondary"], linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(8, 2.3, "U-Net Decoder → 5 Classes", fontsize=16,
            color=COLORS["secondary"], ha='center', va='center', fontweight='bold')

    # Arrows
    ax.annotate('', xy=(5.5, 6.2), xytext=(5.5, 7),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.annotate('', xy=(11.5, 6.2), xytext=(11.5, 7),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.annotate('', xy=(6.5, 4.3), xytext=(5.5, 5),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.annotate('', xy=(8, 2.8), xytext=(8, 3.3),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))

    return fig


def create_slide_5_cross_attention():
    """Slide 5: Cross-Attention"""
    fig, ax = create_text_frame("Cross-Temporal Attention")

    # Key insight
    ax.text(9.6, 7.5, '"Post features ASK what changed,', fontsize=28,
            color=COLORS["accent"], ha='center', va='center', style='italic')
    ax.text(9.6, 6.8, 'Pre features ANSWER what existed"', fontsize=28,
            color=COLORS["accent"], ha='center', va='center', style='italic')

    # Q, K, V diagram
    # Post -> Q
    post_box = FancyBboxPatch((2, 4), 3, 1, boxstyle="round,pad=0.02",
                               facecolor=COLORS["secondary"], alpha=0.3,
                               edgecolor=COLORS["secondary"], linewidth=2)
    ax.add_patch(post_box)
    ax.text(3.5, 4.5, "Post Features", fontsize=18,
            color=COLORS["secondary"], ha='center', va='center')

    q_box = FancyBboxPatch((6.5, 4), 1.5, 1, boxstyle="round,pad=0.02",
                            facecolor=COLORS["secondary"], alpha=0.6,
                            edgecolor=COLORS["secondary"], linewidth=2)
    ax.add_patch(q_box)
    ax.text(7.25, 4.5, "Q", fontsize=24, color='white', ha='center', va='center', fontweight='bold')

    ax.annotate('', xy=(6.5, 4.5), xytext=(5, 4.5),
                arrowprops=dict(arrowstyle='->', color=COLORS["secondary"], lw=2))

    # Pre -> K, V
    pre_box = FancyBboxPatch((2, 2), 3, 1, boxstyle="round,pad=0.02",
                              facecolor=COLORS["primary"], alpha=0.3,
                              edgecolor=COLORS["primary"], linewidth=2)
    ax.add_patch(pre_box)
    ax.text(3.5, 2.5, "Pre Features", fontsize=18,
            color=COLORS["primary"], ha='center', va='center')

    k_box = FancyBboxPatch((6.5, 2.8), 1.5, 0.8, boxstyle="round,pad=0.02",
                            facecolor=COLORS["primary"], alpha=0.6,
                            edgecolor=COLORS["primary"], linewidth=2)
    ax.add_patch(k_box)
    ax.text(7.25, 3.2, "K", fontsize=24, color='white', ha='center', va='center', fontweight='bold')

    v_box = FancyBboxPatch((6.5, 1.8), 1.5, 0.8, boxstyle="round,pad=0.02",
                            facecolor=COLORS["primary"], alpha=0.6,
                            edgecolor=COLORS["primary"], linewidth=2)
    ax.add_patch(v_box)
    ax.text(7.25, 2.2, "V", fontsize=24, color='white', ha='center', va='center', fontweight='bold')

    ax.annotate('', xy=(6.5, 3.2), xytext=(5, 2.7),
                arrowprops=dict(arrowstyle='->', color=COLORS["primary"], lw=2))
    ax.annotate('', xy=(6.5, 2.2), xytext=(5, 2.3),
                arrowprops=dict(arrowstyle='->', color=COLORS["primary"], lw=2))

    # Attention
    attn_box = FancyBboxPatch((9, 2.5), 4, 2.5, boxstyle="round,pad=0.02",
                               facecolor=COLORS["purple"], alpha=0.2,
                               edgecolor=COLORS["purple"], linewidth=2)
    ax.add_patch(attn_box)
    ax.text(11, 4.3, "Attention", fontsize=20,
            color=COLORS["purple"], ha='center', va='center', fontweight='bold')
    ax.text(11, 3.5, "softmax(QKᵀ/√d)·V", fontsize=16,
            color='white', ha='center', va='center', family='monospace')
    ax.text(11, 2.8, "8 heads × 96 dim", fontsize=14,
            color=COLORS["text_dim"], ha='center', va='center')

    # Arrows to attention
    ax.annotate('', xy=(9, 4), xytext=(8, 4.5),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.annotate('', xy=(9, 3.2), xytext=(8, 3.2),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.annotate('', xy=(9, 2.7), xytext=(8, 2.2),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))

    # Output
    out_box = FancyBboxPatch((14, 3), 3, 1.5, boxstyle="round,pad=0.02",
                              facecolor=COLORS["accent"], alpha=0.3,
                              edgecolor=COLORS["accent"], linewidth=2)
    ax.add_patch(out_box)
    ax.text(15.5, 3.75, "Fused Features", fontsize=18,
            color=COLORS["accent"], ha='center', va='center', fontweight='bold')

    ax.annotate('', xy=(14, 3.75), xytext=(13, 3.75),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))

    return fig


def create_slide_6_loss():
    """Slide 6: Loss Function"""
    fig, ax = create_text_frame("Hybrid Focal-Dice Loss")

    # Problem
    ax.text(9.6, 7.5, "Problem: Background 87%, Destroyed only 0.66%", fontsize=24,
            color=COLORS["danger"], ha='center', va='center')

    # Formula
    formula_box = FancyBboxPatch((4, 5.5), 11.2, 1.2, boxstyle="round,pad=0.02",
                                  facecolor=COLORS["primary"], alpha=0.1,
                                  edgecolor=COLORS["primary"], linewidth=2)
    ax.add_patch(formula_box)
    ax.text(9.6, 6.1, "L = 0.5 × Focal + 0.5 × Dice", fontsize=32,
            color='white', ha='center', va='center', family='monospace')

    # Focal Loss
    ax.text(5, 4.5, "Focal Loss", fontsize=24,
            color=COLORS["secondary"], ha='center', va='center', fontweight='bold')
    ax.text(5, 3.8, "• Down-weights easy samples", fontsize=18,
            color=COLORS["text_dim"], ha='center', va='center')
    ax.text(5, 3.2, "• Focuses on hard examples", fontsize=18,
            color=COLORS["text_dim"], ha='center', va='center')
    ax.text(5, 2.5, "FL = -(1-p)² × log(p)", fontsize=16,
            color='white', ha='center', va='center', family='monospace')

    # Dice Loss
    ax.text(14, 4.5, "Dice Loss", fontsize=24,
            color=COLORS["accent"], ha='center', va='center', fontweight='bold')
    ax.text(14, 3.8, "• Optimizes overlap directly", fontsize=18,
            color=COLORS["text_dim"], ha='center', va='center')
    ax.text(14, 3.2, "• Handles small regions", fontsize=18,
            color=COLORS["text_dim"], ha='center', va='center')
    ax.text(14, 2.5, "DL = 1 - 2|A∩B|/(|A|+|B|)", fontsize=16,
            color='white', ha='center', va='center', family='monospace')

    # Class weights
    ax.text(9.6, 1.5, "Class Weights: [0.1, 1.0, 2.0, 3.0, 4.0]", fontsize=24,
            color=COLORS["danger"], ha='center', va='center', fontweight='bold')

    return fig


def create_slide_7_tta():
    """Slide 7: Test-Time Augmentation"""
    fig, ax = create_text_frame("Test-Time Augmentation (TTA)")

    # Original image representation
    transforms = [
        ("Original", 3.5, COLORS["primary"]),
        ("H-Flip", 8, COLORS["secondary"]),
        ("V-Flip", 12.5, COLORS["accent"]),
    ]

    for name, x, color in transforms:
        # Image box
        box = Rectangle((x, 5), 2.5, 2.5, facecolor=color, alpha=0.3,
                        edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 1.25, 4.3, name, fontsize=18, color=color,
                ha='center', va='center', fontweight='bold')

        # Triangle to show orientation
        if name == "Original":
            triangle = plt.Polygon([(x+0.5, 5.5), (x+2, 5.5), (x+1.25, 7)],
                                   facecolor='white', alpha=0.7)
        elif name == "H-Flip":
            triangle = plt.Polygon([(x+2, 5.5), (x+0.5, 5.5), (x+1.25, 7)],
                                   facecolor='white', alpha=0.7)
        else:  # V-Flip
            triangle = plt.Polygon([(x+0.5, 7), (x+2, 7), (x+1.25, 5.5)],
                                   facecolor='white', alpha=0.7)
        ax.add_patch(triangle)

        # Predict arrow
        ax.annotate('', xy=(x+1.25, 3.8), xytext=(x+1.25, 5),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2))
        ax.text(x+2, 4.4, "Predict", fontsize=12, color=COLORS["text_dim"])

    # Average box
    avg_box = FancyBboxPatch((5.5, 1.5), 6, 1.5, boxstyle="round,pad=0.02",
                              facecolor=COLORS["danger"], alpha=0.2,
                              edgecolor=COLORS["danger"], linewidth=2)
    ax.add_patch(avg_box)
    ax.text(8.5, 2.5, "Reverse transforms", fontsize=18,
            color='white', ha='center', va='center')
    ax.text(8.5, 1.9, "→ Average all predictions", fontsize=18,
            color=COLORS["danger"], ha='center', va='center', fontweight='bold')

    # Arrows to average
    for x in [3.5, 8, 12.5]:
        ax.annotate('', xy=(8.5, 3), xytext=(x+1.25, 3.8),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    # Result
    ax.text(15.5, 2.25, "~3-5% F1\nimprovement!", fontsize=24,
            color=COLORS["secondary"], ha='center', va='center', fontweight='bold')

    return fig


def create_slide_8_results():
    """Slide 8: Results"""
    fig, ax = create_text_frame("Results")

    # Metrics table
    metrics = [
        ("Macro F1-Score", "XX.XX%"),
        ("Weighted F1-Score", "XX.XX%"),
        ("Mean IoU (mIoU)", "XX.XX%"),
        ("Overall Accuracy", "XX.XX%"),
    ]

    # Table background
    table_box = FancyBboxPatch((2, 3.5), 5, 4, boxstyle="round,pad=0.02",
                                facecolor=COLORS["primary"], alpha=0.1,
                                edgecolor=COLORS["primary"], linewidth=2)
    ax.add_patch(table_box)

    y_start = 6.8
    for metric, value in metrics:
        ax.text(2.3, y_start, metric, fontsize=18, color=COLORS["text_dim"], ha='left')
        ax.text(6.7, y_start, value, fontsize=18, color=COLORS["secondary"],
                ha='right', fontweight='bold')
        y_start -= 0.9

    # Per-class bars
    ax.text(12, 7.5, "Per-Class F1", fontsize=24,
            color=COLORS["accent"], ha='center', va='center', fontweight='bold')

    classes = [
        ("BG", 0.95, "#263238"),
        ("NoDmg", 0.82, "#4CAF50"),
        ("Minor", 0.45, "#FFEB3B"),
        ("Major", 0.52, "#FF9800"),
        ("Destr", 0.38, "#F44336"),
    ]

    for i, (name, score, color) in enumerate(classes):
        bar_height = score * 4
        bar = Rectangle((9 + i * 1.3, 3), 1, bar_height, facecolor=color)
        ax.add_patch(bar)
        ax.text(9.5 + i * 1.3, 2.5, name, fontsize=12, color='white',
                ha='center', va='center', rotation=45)
        ax.text(9.5 + i * 1.3, 3.2 + bar_height, f"{score:.2f}", fontsize=12,
                color='white', ha='center', va='center')

    return fig


def create_slide_9_contributions():
    """Slide 9: Key Contributions"""
    fig, ax = create_text_frame("Key Contributions")

    contributions = [
        ("1. Siamese Swin-Transformer", "First application to bi-temporal damage assessment", COLORS["primary"]),
        ("2. Cross-Temporal Attention", "Novel Q/K/V formulation for change detection", COLORS["purple"]),
        ("3. Geometric-Residual Priors", "Diff-CNN complements semantic features", COLORS["accent"]),
        ("4. Production-Ready Pipeline", "Scene-wise splitting, TTA, full-image stitching", COLORS["secondary"]),
    ]

    y_start = 7.0
    for title, desc, color in contributions:
        ax.text(2, y_start, title, fontsize=26, color=color,
                ha='left', va='center', fontweight='bold')
        ax.text(2.5, y_start - 0.5, desc, fontsize=18,
                color=COLORS["text_dim"], ha='left', va='center')
        y_start -= 1.5

    return fig


def create_slide_10_thanks():
    """Slide 10: Thank You"""
    fig, ax = create_text_frame("")

    ax.text(9.6, 6.5, "Thank You!", fontsize=72,
            color='white', ha='center', va='center', fontweight='bold')

    ax.plot([4, 15.2], [5.5, 5.5], color=COLORS["primary"], linewidth=3)

    ax.text(9.6, 4.2, "📧  your.email@example.com", fontsize=24,
            color=COLORS["text_dim"], ha='center', va='center')
    ax.text(9.6, 3.4, "🐙  github.com/your-username/xbd", fontsize=24,
            color=COLORS["text_dim"], ha='center', va='center')

    ax.text(9.6, 2.0, "Questions?", fontsize=36,
            color=COLORS["accent"], ha='center', va='center')

    return fig


def save_all_frames():
    """Save all slides as frames"""
    slides = [
        ("01_title", create_slide_1_title),
        ("02_problem", create_slide_2_problem),
        ("03_dataset", create_slide_3_dataset),
        ("04_architecture", create_slide_4_architecture),
        ("05_cross_attention", create_slide_5_cross_attention),
        ("06_loss", create_slide_6_loss),
        ("07_tta", create_slide_7_tta),
        ("08_results", create_slide_8_results),
        ("09_contributions", create_slide_9_contributions),
        ("10_thanks", create_slide_10_thanks),
    ]

    print("Generating frames...")
    for name, create_func in slides:
        fig = create_func()
        filepath = FRAMES_DIR / f"{name}.png"
        fig.savefig(filepath, dpi=100, facecolor=COLORS["bg"],
                   edgecolor='none', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"  ✓ Saved {filepath}")

    return [FRAMES_DIR / f"{name}.png" for name, _ in slides]


def create_video_from_frames(frame_paths, output_path, fps=1, duration_per_frame=5):
    """Create video from frame images using moviepy"""
    try:
        from moviepy.editor import ImageSequenceClip, concatenate_videoclips, ImageClip

        print("\nCreating video with MoviePy...")

        # Create clips from images
        clips = []
        for frame_path in frame_paths:
            clip = ImageClip(str(frame_path)).set_duration(duration_per_frame)
            clips.append(clip)

        # Concatenate all clips
        final_clip = concatenate_videoclips(clips, method="compose")

        # Write video
        final_clip.write_videofile(
            str(output_path),
            fps=24,
            codec='libx264',
            audio=False,
            verbose=False,
            logger=None
        )

        print(f"  ✓ Video saved to: {output_path}")
        return True

    except ImportError:
        print("\n⚠️ MoviePy not installed. Install with: pip install moviepy")
        print("  Falling back to GIF creation...")
        return False


def create_gif_from_frames(frame_paths, output_path, duration_per_frame=3000):
    """Create GIF from frame images using PIL"""
    print("\nCreating GIF with PIL...")

    frames = []
    for frame_path in frame_paths:
        img = Image.open(frame_path)
        frames.append(img)

    # Save as GIF
    gif_path = output_path.with_suffix('.gif')
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_per_frame,
        loop=0
    )

    print(f"  ✓ GIF saved to: {gif_path}")
    return gif_path


def main():
    """Main function to generate video"""
    print("=" * 60)
    print("GENERATING PROJECT VIDEO")
    print("=" * 60)

    # Generate all frames
    frame_paths = save_all_frames()

    # Try to create video
    video_path = OUTPUT_DIR / "satellite_damage_video.mp4"
    video_created = create_video_from_frames(frame_paths, video_path)

    # Create GIF as backup/alternative
    gif_path = create_gif_from_frames(frame_paths, OUTPUT_DIR / "satellite_damage_video")

    print("\n" + "=" * 60)
    print("VIDEO GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  • Frames: {FRAMES_DIR}/")
    if video_created:
        print(f"  • Video:  {video_path}")
    print(f"  • GIF:    {gif_path}")
    print("\nTo customize:")
    print("  1. Edit the create_slide_X functions")
    print("  2. Adjust DURATION_PER_SLIDE for timing")
    print("  3. Run this script again")


if __name__ == "__main__":
    main()
