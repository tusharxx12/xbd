"""
xBD (xView2) Dataset Preprocessing Script
==========================================
Preprocesses satellite imagery for building damage segmentation.

Features:
- File pairing (pre/post images + JSON labels)
- Mask generation from WKT polygons
- 512x512 tiling with building density filtering
- Memory-efficient processing

Author: AI Assistant
Environment: Kaggle
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import cv2
from shapely import wkt
from shapely.geometry import Polygon
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for preprocessing parameters."""

    # Paths (Kaggle environment)
    INPUT_PATH = Path("/kaggle/input/xview2-challenge-dataset")
    OUTPUT_PATH = Path("/kaggle/working/processed_data")

    # Image parameters
    IMAGE_SIZE = 1024
    TILE_SIZE = 512
    TILE_STRIDE = 512  # No overlap

    # Filtering
    MIN_BUILDING_RATIO = 0.01  # Skip tiles with < 1% building pixels

    # Damage class mapping
    DAMAGE_CLASSES = {
        "no-damage": 1,
        "minor-damage": 2,
        "major-damage": 3,
        "destroyed": 4,
        "un-classified": 0,  # Treat as background
    }

    # Debug mode
    DEBUG_LIMIT = 10  # Process only N scenes in debug mode


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_json(json_path: Path) -> Optional[Dict]:
    """
    Load and parse JSON label file.

    Args:
        json_path: Path to JSON file

    Returns:
        Parsed JSON dict or None if error
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[WARNING] Failed to load {json_path}: {e}")
        return None


def extract_scene_id(filename: str) -> str:
    """
    Extract scene ID from filename.

    Example:
        "georgia-flooding_00001_pre_disaster.png" -> "georgia-flooding_00001"

    Args:
        filename: Image or JSON filename

    Returns:
        Scene ID string
    """
    # Remove extension and split by '_'
    name = Path(filename).stem
    parts = name.split('_')

    # Scene ID is everything before 'pre' or 'post'
    scene_parts = []
    for part in parts:
        if part in ('pre', 'post'):
            break
        scene_parts.append(part)

    return '_'.join(scene_parts)


def parse_polygon_from_wkt(wkt_string: str) -> Optional[np.ndarray]:
    """
    Parse WKT polygon string to numpy coordinate array.

    Args:
        wkt_string: WKT format polygon string

    Returns:
        Numpy array of shape (N, 2) with coordinates, or None if invalid
    """
    try:
        geom = wkt.loads(wkt_string)

        if geom.is_empty:
            return None

        # Handle both Polygon and MultiPolygon
        if geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords)
        elif geom.geom_type == 'MultiPolygon':
            # Take the largest polygon
            largest = max(geom.geoms, key=lambda p: p.area)
            coords = np.array(largest.exterior.coords)
        else:
            return None

        return coords

    except Exception as e:
        # Silently handle malformed WKT
        return None


def clip_coordinates(coords: np.ndarray, image_size: int = 1024) -> np.ndarray:
    """
    Clip polygon coordinates to image bounds.

    Args:
        coords: Numpy array of coordinates (N, 2)
        image_size: Image dimension (assumes square)

    Returns:
        Clipped coordinates
    """
    coords = np.clip(coords, 0, image_size - 1)
    return coords.astype(np.int32)


# ============================================================================
# MASK GENERATION
# ============================================================================

def create_mask(json_data: Dict, image_size: int = 1024) -> np.ndarray:
    """
    Create segmentation mask from JSON label data.

    Args:
        json_data: Parsed JSON dictionary with building annotations
        image_size: Output mask size (assumes square)

    Returns:
        Numpy array mask of shape (image_size, image_size)
        Values: 0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed
    """
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    # Get features from JSON
    features = json_data.get('features', {}).get('xy', [])

    if not features:
        return mask

    for feature in features:
        properties = feature.get('properties', {})
        wkt_string = properties.get('feature_wkt', '')

        # Get damage class
        damage_type = properties.get('subtype', 'no-damage')
        damage_class = Config.DAMAGE_CLASSES.get(damage_type, 0)

        # Skip background/unclassified
        if damage_class == 0:
            continue

        # Parse polygon
        coords = parse_polygon_from_wkt(wkt_string)

        if coords is None:
            continue

        # Clip to image bounds
        coords = clip_coordinates(coords, image_size)

        # Reshape for cv2.fillPoly: expects (1, N, 2)
        polygon = coords.reshape((-1, 1, 2))

        # Fill polygon on mask
        cv2.fillPoly(mask, [polygon], color=int(damage_class))

    return mask


def create_building_mask(json_data: Dict, image_size: int = 1024) -> np.ndarray:
    """
    Create binary building mask (for localization task).

    Args:
        json_data: Parsed JSON dictionary
        image_size: Output mask size

    Returns:
        Binary mask where 1=building, 0=background
    """
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    features = json_data.get('features', {}).get('xy', [])

    for feature in features:
        properties = feature.get('properties', {})
        wkt_string = properties.get('feature_wkt', '')

        coords = parse_polygon_from_wkt(wkt_string)

        if coords is None:
            continue

        coords = clip_coordinates(coords, image_size)
        polygon = coords.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon], color=1)

    return mask


# ============================================================================
# TILING
# ============================================================================

def tile_image(image: np.ndarray,
               tile_size: int = 512,
               stride: int = 512) -> List[Tuple[np.ndarray, int, int]]:
    """
    Split image into tiles.

    Args:
        image: Input image (H, W) or (H, W, C)
        tile_size: Size of each tile
        stride: Step size between tiles

    Returns:
        List of (tile, row_idx, col_idx) tuples
    """
    tiles = []
    h, w = image.shape[:2]

    for row_idx, y in enumerate(range(0, h - tile_size + 1, stride)):
        for col_idx, x in enumerate(range(0, w - tile_size + 1, stride)):
            if len(image.shape) == 3:
                tile = image[y:y+tile_size, x:x+tile_size, :]
            else:
                tile = image[y:y+tile_size, x:x+tile_size]

            tiles.append((tile, row_idx, col_idx))

    return tiles


def compute_building_ratio(mask_tile: np.ndarray) -> float:
    """
    Compute ratio of building pixels in a tile.

    Args:
        mask_tile: Tile from damage mask

    Returns:
        Ratio of non-zero (building) pixels
    """
    total_pixels = mask_tile.size
    building_pixels = np.count_nonzero(mask_tile)
    return building_pixels / total_pixels


# ============================================================================
# FILE DISCOVERY AND PAIRING
# ============================================================================

def discover_scenes(data_path: Path) -> Dict[str, Dict[str, Path]]:
    """
    Discover and pair all scene files.

    Args:
        data_path: Path to dataset directory (contains images/ and labels/)

    Returns:
        Dictionary mapping scene_id to {'pre': path, 'post': path, 'json': path}
    """
    scenes = defaultdict(dict)

    images_dir = data_path / "images"
    labels_dir = data_path / "labels"

    # Check if directories exist
    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        return {}

    # Discover pre-disaster images
    for img_path in images_dir.glob("*_pre_disaster.png"):
        scene_id = extract_scene_id(img_path.name)
        scenes[scene_id]['pre'] = img_path

    # Discover post-disaster images
    for img_path in images_dir.glob("*_post_disaster.png"):
        scene_id = extract_scene_id(img_path.name)
        scenes[scene_id]['post'] = img_path

    # Discover JSON labels
    if labels_dir.exists():
        for json_path in labels_dir.glob("*_post_disaster.json"):
            scene_id = extract_scene_id(json_path.name)
            scenes[scene_id]['json'] = json_path

    # Filter to only complete scenes
    complete_scenes = {
        scene_id: files
        for scene_id, files in scenes.items()
        if all(k in files for k in ['pre', 'post', 'json'])
    }

    incomplete = len(scenes) - len(complete_scenes)
    if incomplete > 0:
        print(f"[INFO] Skipped {incomplete} incomplete scenes")

    return complete_scenes


# ============================================================================
# SCENE PROCESSING
# ============================================================================

def process_scene(scene_id: str,
                  files: Dict[str, Path],
                  output_dirs: Dict[str, Path],
                  config: Config = Config) -> int:
    """
    Process a single scene: load, create mask, tile, and save.

    Args:
        scene_id: Scene identifier
        files: Dictionary with 'pre', 'post', 'json' paths
        output_dirs: Dictionary with 'pre', 'post', 'masks' output paths
        config: Configuration object

    Returns:
        Number of tiles saved
    """
    tiles_saved = 0

    # Load images
    pre_img = cv2.imread(str(files['pre']))
    post_img = cv2.imread(str(files['post']))

    if pre_img is None or post_img is None:
        print(f"[WARNING] Failed to load images for {scene_id}")
        return 0

    # Convert BGR to RGB
    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

    # Load JSON and create mask
    json_data = load_json(files['json'])

    if json_data is None:
        return 0

    mask = create_mask(json_data, config.IMAGE_SIZE)

    # Tile images and mask
    pre_tiles = tile_image(pre_img, config.TILE_SIZE, config.TILE_STRIDE)
    post_tiles = tile_image(post_img, config.TILE_SIZE, config.TILE_STRIDE)
    mask_tiles = tile_image(mask, config.TILE_SIZE, config.TILE_STRIDE)

    # Process each tile
    for (pre_tile, row, col), (post_tile, _, _), (mask_tile, _, _) in zip(
        pre_tiles, post_tiles, mask_tiles
    ):
        # Check building density
        building_ratio = compute_building_ratio(mask_tile)

        if building_ratio < config.MIN_BUILDING_RATIO:
            continue

        # Generate filename
        tile_name = f"{scene_id}_tile_{row}_{col}.png"

        # Save tiles (convert RGB back to BGR for cv2)
        cv2.imwrite(
            str(output_dirs['pre'] / tile_name),
            cv2.cvtColor(pre_tile, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            str(output_dirs['post'] / tile_name),
            cv2.cvtColor(post_tile, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            str(output_dirs['masks'] / tile_name),
            mask_tile
        )

        tiles_saved += 1

    return tiles_saved


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_samples(output_dirs: Dict[str, Path], num_samples: int = 3):
    """
    Display random samples for visual verification.

    Args:
        output_dirs: Dictionary with output directory paths
        num_samples: Number of samples to display
    """
    # Get list of processed tiles
    tile_files = list(output_dirs['pre'].glob("*.png"))

    if not tile_files:
        print("[WARNING] No processed tiles found for visualization")
        return

    # Random sample
    samples = random.sample(tile_files, min(num_samples, len(tile_files)))

    # Color map for damage classes
    damage_colors = {
        0: [0, 0, 0],        # Background - black
        1: [0, 255, 0],      # No damage - green
        2: [255, 255, 0],    # Minor - yellow
        3: [255, 165, 0],    # Major - orange
        4: [255, 0, 0],      # Destroyed - red
    }

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, pre_path in enumerate(samples):
        tile_name = pre_path.name

        # Load all three images
        pre_img = cv2.imread(str(output_dirs['pre'] / tile_name))
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)

        post_img = cv2.imread(str(output_dirs['post'] / tile_name))
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(output_dirs['masks'] / tile_name), cv2.IMREAD_GRAYSCALE)

        # Create colored mask for visualization
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in damage_colors.items():
            colored_mask[mask == class_id] = color

        # Plot
        axes[idx, 0].imshow(pre_img)
        axes[idx, 0].set_title(f"Pre-Disaster\n{tile_name}")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(post_img)
        axes[idx, 1].set_title("Post-Disaster")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(colored_mask)
        axes[idx, 2].set_title("Damage Mask")
        axes[idx, 2].axis('off')

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=np.array(c)/255, label=l)
        for l, c in [
            ("Background", [0, 0, 0]),
            ("No Damage", [0, 255, 0]),
            ("Minor Damage", [255, 255, 0]),
            ("Major Damage", [255, 165, 0]),
            ("Destroyed", [255, 0, 0]),
        ]
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dirs['pre'].parent / "sample_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Visualization saved to {output_dirs['pre'].parent / 'sample_visualization.png'}")


def print_dataset_stats(output_dirs: Dict[str, Path]):
    """
    Print statistics about the processed dataset.

    Args:
        output_dirs: Dictionary with output directory paths
    """
    tile_files = list(output_dirs['masks'].glob("*.png"))

    if not tile_files:
        print("[WARNING] No processed tiles found")
        return

    # Count damage class distribution
    class_counts = defaultdict(int)
    total_pixels = 0

    for mask_path in tqdm(tile_files, desc="Computing statistics"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unique, counts = np.unique(mask, return_counts=True)

        for cls, cnt in zip(unique, counts):
            class_counts[cls] += cnt
            total_pixels += cnt

    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total tiles: {len(tile_files)}")
    print(f"\nDamage class distribution:")

    class_names = {0: "Background", 1: "No Damage", 2: "Minor", 3: "Major", 4: "Destroyed"}

    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        percentage = (count / total_pixels) * 100
        print(f"  {class_names.get(cls, f'Class {cls}')}: {count:,} pixels ({percentage:.2f}%)")


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def preprocess_dataset(input_path: Path = None,
                       output_path: Path = None,
                       split: str = "train",
                       debug: bool = False,
                       debug_limit: int = None,
                       visualize: bool = True):
    """
    Main preprocessing pipeline.

    Args:
        input_path: Path to xBD dataset (default: Config.INPUT_PATH)
        output_path: Path for processed output (default: Config.OUTPUT_PATH)
        split: Dataset split to process ('train', 'test', 'hold')
        debug: Enable debug mode (process limited scenes)
        debug_limit: Number of scenes to process in debug mode
        visualize: Show sample visualizations after processing
    """
    # Set paths
    input_path = input_path or Config.INPUT_PATH
    output_path = output_path or Config.OUTPUT_PATH
    debug_limit = debug_limit or Config.DEBUG_LIMIT

    data_path = input_path / split

    print("="*60)
    print("xBD DATASET PREPROCESSING")
    print("="*60)
    print(f"Input path: {data_path}")
    print(f"Output path: {output_path / split}")
    print(f"Debug mode: {debug} (limit: {debug_limit if debug else 'N/A'})")
    print("="*60)

    # Create output directories
    output_dirs = {
        'pre': output_path / split / 'pre',
        'post': output_path / split / 'post',
        'masks': output_path / split / 'masks',
    }

    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created directory: {dir_path}")

    # Discover scenes
    print("\n[INFO] Discovering scenes...")
    scenes = discover_scenes(data_path)

    if not scenes:
        print("[ERROR] No complete scenes found!")
        return

    print(f"[INFO] Found {len(scenes)} complete scenes")

    # Limit scenes in debug mode
    if debug:
        scene_ids = list(scenes.keys())[:debug_limit]
        scenes = {sid: scenes[sid] for sid in scene_ids}
        print(f"[DEBUG] Processing {len(scenes)} scenes only")

    # Process scenes
    total_tiles = 0

    for scene_id, files in tqdm(scenes.items(), desc="Processing scenes"):
        try:
            tiles_saved = process_scene(scene_id, files, output_dirs)
            total_tiles += tiles_saved
        except Exception as e:
            print(f"\n[ERROR] Failed to process {scene_id}: {e}")
            continue

    print(f"\n[INFO] Processing complete!")
    print(f"[INFO] Total tiles saved: {total_tiles}")

    # Print statistics
    print_dataset_stats(output_dirs)

    # Visualize samples
    if visualize and total_tiles > 0:
        print("\n[INFO] Generating visualization...")
        visualize_samples(output_dirs, num_samples=3)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run preprocessing
    # Set debug=True for testing with limited samples
    # Set debug=False for full dataset processing

    preprocess_dataset(
        input_path=Config.INPUT_PATH,
        output_path=Config.OUTPUT_PATH,
        split="train",
        debug=True,          # Set to False for full processing
        debug_limit=10,      # Number of scenes in debug mode
        visualize=True       # Show sample visualizations
    )

    # To process full dataset, uncomment below:
    # preprocess_dataset(
    #     split="train",
    #     debug=False,
    #     visualize=True
    # )
