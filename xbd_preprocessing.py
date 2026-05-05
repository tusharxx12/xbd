"""
xBD (xView2) Dataset Preprocessing Script
==========================================
Preprocesses satellite imagery for building damage segmentation.

Features:
- File pairing (pre/post images + JSON labels)
- Mask generation from WKT polygons
- 512x512 tiling with building density filtering
- Memory-efficient processing
- Optional change/diff channel generation

Author: AI Assistant
Environment: Kaggle

CHANGELOG:
- Fixed MultiPolygon handling (now processes ALL polygons)
- Fixed mask overwrite with damage priority (higher class wins)
- Added robust JSON parsing with validation
- Removed fixed image size assumption (dynamic sizing)
- Added safe tile matching via index-based approach
- Fixed building pixel check using np.sum(mask > 0)
- Added optional change/diff channel
- Added coordinate safety checks
- Added cv2.setNumThreads(0) for performance
- Improved visualization with overlay
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import cv2
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm
import matplotlib.pyplot as plt

# [FIX #9] Performance improvement - disable OpenCV threading overhead
cv2.setNumThreads(0)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for preprocessing parameters."""

    # Paths (Kaggle environment)
    INPUT_PATH = Path("/kaggle/input/xview2-challenge-dataset")
    OUTPUT_PATH = Path("/kaggle/working/processed_data")

    # Image parameters
    # [FIX #4] IMAGE_SIZE is now only used as fallback, actual size is read from image
    DEFAULT_IMAGE_SIZE = 1024  # Fallback only
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

    # [FIX #2] Damage priority for mask overwrite (higher value = higher priority)
    DAMAGE_PRIORITY = {
        0: 0,  # Background - lowest
        1: 1,  # No damage
        2: 2,  # Minor damage
        3: 3,  # Major damage
        4: 4,  # Destroyed - highest priority
    }

    # Debug mode
    DEBUG_LIMIT = 10  # Process only N scenes in debug mode

    # [FIX #7] Optional change channel
    SAVE_DIFF_CHANNEL = True  # Set to True to save diff tiles


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


def parse_polygons_from_wkt(wkt_string: str) -> List[np.ndarray]:
    """
    [FIX #1] Parse WKT polygon string to list of numpy coordinate arrays.
    Now handles ALL polygons in MultiPolygon, not just the largest.

    Args:
        wkt_string: WKT format polygon string

    Returns:
        List of numpy arrays, each of shape (N, 2) with coordinates
    """
    polygons = []

    try:
        geom = wkt.loads(wkt_string)

        if geom.is_empty:
            return polygons

        # [FIX #1] Handle both Polygon and MultiPolygon - process ALL polygons
        if geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords)
            polygons.append(coords)
        elif geom.geom_type == 'MultiPolygon':
            # [FIX #1] CRITICAL: Process ALL polygons, not just largest
            for poly in geom.geoms:
                if not poly.is_empty:
                    coords = np.array(poly.exterior.coords)
                    polygons.append(coords)
        # Handle GeometryCollection (rare but possible)
        elif geom.geom_type == 'GeometryCollection':
            for g in geom.geoms:
                if g.geom_type == 'Polygon' and not g.is_empty:
                    coords = np.array(g.exterior.coords)
                    polygons.append(coords)

    except Exception as e:
        # Silently handle malformed WKT
        pass

    return polygons


def clip_coordinates(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    [FIX #4 & #8] Clip polygon coordinates to image bounds.
    Now takes separate height and width parameters.

    Args:
        coords: Numpy array of coordinates (N, 2) in (x, y) format
        height: Image height
        width: Image width

    Returns:
        Clipped coordinates as int32
    """
    # [FIX #8] Ensure coordinates are within bounds and converted to int32
    clipped = coords.copy()
    clipped[:, 0] = np.clip(coords[:, 0], 0, width - 1)   # x coordinates
    clipped[:, 1] = np.clip(coords[:, 1], 0, height - 1)  # y coordinates
    return clipped.astype(np.int32)


# ============================================================================
# MASK GENERATION
# ============================================================================

def create_mask(json_data: Dict, height: int, width: int) -> np.ndarray:
    """
    [FIX #1, #2, #3, #4] Create segmentation mask from JSON label data.
    - Processes ALL polygons in MultiPolygon
    - Higher damage classes overwrite lower ones
    - Robust JSON parsing with validation
    - Dynamic image size

    Args:
        json_data: Parsed JSON dictionary with building annotations
        height: Output mask height
        width: Output mask width

    Returns:
        Numpy array mask of shape (height, width)
        Values: 0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Get features from JSON
    features = json_data.get('features', {}).get('xy', [])

    if not features:
        return mask

    # [FIX #2] Collect all polygons with their damage classes
    polygon_data = []

    for feature in features:
        properties = feature.get('properties', {})

        # [FIX #3] Robust JSON parsing - skip if subtype is missing
        if 'subtype' not in properties:
            continue

        damage_type = properties.get('subtype', '')

        # [FIX #3] Skip if damage type is empty or not recognized
        if not damage_type or damage_type not in Config.DAMAGE_CLASSES:
            continue

        damage_class = Config.DAMAGE_CLASSES.get(damage_type, 0)

        # Skip background/unclassified
        if damage_class == 0:
            continue

        # [FIX #3] Skip if feature_wkt is missing or empty
        wkt_string = properties.get('feature_wkt', '')
        if not wkt_string or not wkt_string.strip():
            continue

        # [FIX #1] Parse ALL polygons from WKT
        polygons = parse_polygons_from_wkt(wkt_string)

        for coords in polygons:
            if coords is not None and len(coords) >= 3:
                polygon_data.append((coords, damage_class))

    # [FIX #2] Sort by damage class (ascending) so higher classes are drawn last
    # This ensures destroyed (4) overwrites major (3) overwrites minor (2), etc.
    polygon_data.sort(key=lambda x: Config.DAMAGE_PRIORITY[x[1]])

    # Draw polygons in sorted order
    for coords, damage_class in polygon_data:
        # [FIX #4 & #8] Clip to actual image bounds
        coords = clip_coordinates(coords, height, width)

        # Reshape for cv2.fillPoly: expects (N, 1, 2)
        polygon = coords.reshape((-1, 1, 2))

        # Fill polygon on mask
        cv2.fillPoly(mask, [polygon], color=int(damage_class))

    return mask


def create_mask_with_priority(json_data: Dict, height: int, width: int) -> np.ndarray:
    """
    [FIX #2 Alternative] Create segmentation mask using np.maximum for priority.
    This method guarantees higher damage classes always win, even with overlaps.

    Args:
        json_data: Parsed JSON dictionary with building annotations
        height: Output mask height
        width: Output mask width

    Returns:
        Numpy array mask of shape (height, width)
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    features = json_data.get('features', {}).get('xy', [])

    if not features:
        return mask

    for feature in features:
        properties = feature.get('properties', {})

        # [FIX #3] Robust validation
        if 'subtype' not in properties:
            continue

        damage_type = properties.get('subtype', '')
        if not damage_type or damage_type not in Config.DAMAGE_CLASSES:
            continue

        damage_class = Config.DAMAGE_CLASSES.get(damage_type, 0)
        if damage_class == 0:
            continue

        wkt_string = properties.get('feature_wkt', '')
        if not wkt_string or not wkt_string.strip():
            continue

        polygons = parse_polygons_from_wkt(wkt_string)

        for coords in polygons:
            if coords is not None and len(coords) >= 3:
                coords = clip_coordinates(coords, height, width)
                polygon = coords.reshape((-1, 1, 2))

                # [FIX #2] Create temporary mask for this polygon
                temp_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [polygon], color=int(damage_class))

                # [FIX #2] Use np.maximum to ensure higher damage class wins
                mask = np.maximum(mask, temp_mask)

    return mask


def create_building_mask(json_data: Dict, height: int, width: int) -> np.ndarray:
    """
    Create binary building mask (for localization task).
    [FIX #4] Now uses dynamic image size.

    Args:
        json_data: Parsed JSON dictionary
        height: Output mask height
        width: Output mask width

    Returns:
        Binary mask where 1=building, 0=background
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    features = json_data.get('features', {}).get('xy', [])

    for feature in features:
        properties = feature.get('properties', {})

        # [FIX #3] Skip if feature_wkt is missing
        wkt_string = properties.get('feature_wkt', '')
        if not wkt_string or not wkt_string.strip():
            continue

        # [FIX #1] Get all polygons
        polygons = parse_polygons_from_wkt(wkt_string)

        for coords in polygons:
            if coords is not None and len(coords) >= 3:
                # [FIX #4 & #8] Clip to actual bounds
                coords = clip_coordinates(coords, height, width)
                polygon = coords.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [polygon], color=1)

    return mask


# ============================================================================
# TILING
# ============================================================================

def tile_image(image: np.ndarray,
               tile_size: int = 512,
               stride: int = 512) -> Dict[Tuple[int, int], np.ndarray]:
    """
    [FIX #5] Split image into tiles with index-based dictionary for safe matching.

    Args:
        image: Input image (H, W) or (H, W, C)
        tile_size: Size of each tile
        stride: Step size between tiles

    Returns:
        Dictionary mapping (row_idx, col_idx) to tile array
    """
    tiles = {}
    h, w = image.shape[:2]

    for row_idx, y in enumerate(range(0, h - tile_size + 1, stride)):
        for col_idx, x in enumerate(range(0, w - tile_size + 1, stride)):
            if len(image.shape) == 3:
                tile = image[y:y+tile_size, x:x+tile_size, :]
            else:
                tile = image[y:y+tile_size, x:x+tile_size]

            # [FIX #5] Use tuple key for safe matching
            tiles[(row_idx, col_idx)] = tile

    return tiles


def compute_building_ratio(mask_tile: np.ndarray) -> float:
    """
    [FIX #6] Compute ratio of building pixels in a tile.
    Now uses np.sum(mask > 0) instead of np.count_nonzero.

    Args:
        mask_tile: Tile from damage mask

    Returns:
        Ratio of non-zero (building) pixels
    """
    total_pixels = mask_tile.size
    # [FIX #6] Use np.sum(mask > 0) for explicit building pixel count
    building_pixels = np.sum(mask_tile > 0)
    return building_pixels / total_pixels


# ============================================================================
# CHANGE DETECTION
# ============================================================================

def compute_change_image(pre_img: np.ndarray, post_img: np.ndarray) -> np.ndarray:
    """
    [FIX #7] Compute absolute difference between pre and post images.

    Args:
        pre_img: Pre-disaster image (H, W, C)
        post_img: Post-disaster image (H, W, C)

    Returns:
        Difference image (H, W, C)
    """
    return cv2.absdiff(post_img, pre_img)


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
                  save_diff: bool = False) -> int:
    """
    [FIX #4, #5, #7] Process a single scene: load, create mask, tile, and save.
    - Dynamic image size (no fixed 1024 assumption)
    - Safe tile matching via index-based approach
    - Optional diff channel saving

    Args:
        scene_id: Scene identifier
        files: Dictionary with 'pre', 'post', 'json' paths
        output_dirs: Dictionary with 'pre', 'post', 'masks', optionally 'diff' output paths
        save_diff: Whether to save diff channel

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

    # [FIX #4] Get actual image dimensions
    height, width = pre_img.shape[:2]

    # Convert BGR to RGB
    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

    # [FIX #7] Compute change/diff image if needed
    diff_img = None
    if save_diff:
        diff_img = compute_change_image(pre_img, post_img)

    # Load JSON and create mask
    json_data = load_json(files['json'])

    if json_data is None:
        return 0

    # [FIX #4] Pass actual image dimensions to mask creation
    mask = create_mask(json_data, height, width)

    # [FIX #5] Tile images and mask with index-based dictionary
    pre_tiles = tile_image(pre_img, Config.TILE_SIZE, Config.TILE_STRIDE)
    post_tiles = tile_image(post_img, Config.TILE_SIZE, Config.TILE_STRIDE)
    mask_tiles = tile_image(mask, Config.TILE_SIZE, Config.TILE_STRIDE)

    diff_tiles = None
    if save_diff and diff_img is not None:
        diff_tiles = tile_image(diff_img, Config.TILE_SIZE, Config.TILE_STRIDE)

    # [FIX #5] Process each tile using index-based matching
    for (row, col), pre_tile in pre_tiles.items():
        # [FIX #5] Safe matching - verify indices exist in all tile dicts
        if (row, col) not in post_tiles or (row, col) not in mask_tiles:
            continue

        post_tile = post_tiles[(row, col)]
        mask_tile = mask_tiles[(row, col)]

        # [FIX #6] Check building density with corrected method
        building_ratio = compute_building_ratio(mask_tile)

        if building_ratio < Config.MIN_BUILDING_RATIO:
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

        # [FIX #7] Save diff tile if enabled
        if save_diff and diff_tiles is not None and (row, col) in diff_tiles:
            diff_tile = diff_tiles[(row, col)]
            cv2.imwrite(
                str(output_dirs['diff'] / tile_name),
                cv2.cvtColor(diff_tile, cv2.COLOR_RGB2BGR)
            )

        tiles_saved += 1

    return tiles_saved


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_samples(output_dirs: Dict[str, Path], num_samples: int = 3, show_diff: bool = False):
    """
    [FIX #10] Display random samples for visual verification.
    Now includes overlay visualization.

    Args:
        output_dirs: Dictionary with output directory paths
        num_samples: Number of samples to display
        show_diff: Whether to show diff channel
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

    # [FIX #10] Determine number of columns based on whether diff is shown
    has_diff = show_diff and 'diff' in output_dirs and output_dirs['diff'].exists()
    num_cols = 5 if has_diff else 4  # pre, post, mask, overlay (+ diff if enabled)

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4 * num_cols, 5 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, pre_path in enumerate(samples):
        tile_name = pre_path.name

        # Load all images
        pre_img = cv2.imread(str(output_dirs['pre'] / tile_name))
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)

        post_img = cv2.imread(str(output_dirs['post'] / tile_name))
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(output_dirs['masks'] / tile_name), cv2.IMREAD_GRAYSCALE)

        # Create colored mask for visualization
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in damage_colors.items():
            colored_mask[mask == class_id] = color

        # [FIX #10] Create overlay visualization
        overlay = post_img.copy()
        alpha = 0.4  # Transparency for overlay

        for class_id, color in damage_colors.items():
            if class_id == 0:  # Skip background
                continue
            mask_region = (mask == class_id)
            overlay[mask_region] = (
                (1 - alpha) * overlay[mask_region] + alpha * np.array(color)
            ).astype(np.uint8)

        # Plot pre-disaster
        axes[idx, 0].imshow(pre_img)
        axes[idx, 0].set_title(f"Pre-Disaster\n{tile_name[:30]}...")
        axes[idx, 0].axis('off')

        # Plot post-disaster
        axes[idx, 1].imshow(post_img)
        axes[idx, 1].set_title("Post-Disaster")
        axes[idx, 1].axis('off')

        # Plot colored mask
        axes[idx, 2].imshow(colored_mask)
        axes[idx, 2].set_title("Damage Mask")
        axes[idx, 2].axis('off')

        # [FIX #10] Plot overlay
        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title("Overlay (Post + Mask)")
        axes[idx, 3].axis('off')

        # [FIX #7] Plot diff if available
        if has_diff:
            diff_path = output_dirs['diff'] / tile_name
            if diff_path.exists():
                diff_img = cv2.imread(str(diff_path))
                diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)
                axes[idx, 4].imshow(diff_img)
                axes[idx, 4].set_title("Change (Diff)")
            else:
                axes[idx, 4].text(0.5, 0.5, "N/A", ha='center', va='center')
            axes[idx, 4].axis('off')

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
    plt.subplots_adjust(bottom=0.08)
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
                       visualize: bool = True,
                       save_diff: bool = None):
    """
    Main preprocessing pipeline.
    [FIX #7] Added save_diff parameter for optional change channel.

    Args:
        input_path: Path to xBD dataset (default: Config.INPUT_PATH)
        output_path: Path for processed output (default: Config.OUTPUT_PATH)
        split: Dataset split to process ('train', 'test', 'hold')
        debug: Enable debug mode (process limited scenes)
        debug_limit: Number of scenes to process in debug mode
        visualize: Show sample visualizations after processing
        save_diff: Save difference/change channel (default: Config.SAVE_DIFF_CHANNEL)
    """
    # Set paths
    input_path = input_path or Config.INPUT_PATH
    output_path = output_path or Config.OUTPUT_PATH
    debug_limit = debug_limit or Config.DEBUG_LIMIT
    save_diff = save_diff if save_diff is not None else Config.SAVE_DIFF_CHANNEL

    data_path = input_path / split

    print("="*60)
    print("xBD DATASET PREPROCESSING")
    print("="*60)
    print(f"Input path: {data_path}")
    print(f"Output path: {output_path / split}")
    print(f"Debug mode: {debug} (limit: {debug_limit if debug else 'N/A'})")
    print(f"Save diff channel: {save_diff}")
    print("="*60)

    # Create output directories
    output_dirs = {
        'pre': output_path / split / 'pre',
        'post': output_path / split / 'post',
        'masks': output_path / split / 'masks',
    }

    # [FIX #7] Add diff directory if enabled
    if save_diff:
        output_dirs['diff'] = output_path / split / 'diff'

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
            tiles_saved = process_scene(scene_id, files, output_dirs, save_diff=save_diff)
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
        visualize_samples(output_dirs, num_samples=3, show_diff=save_diff)


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
        debug=True,              # Set to False for full processing
        debug_limit=10,          # Number of scenes in debug mode
        visualize=True,          # Show sample visualizations
        save_diff=True           # [FIX #7] Save change/diff channel
    )

    # To process full dataset, uncomment below:
    # preprocess_dataset(
    #     split="train",
    #     debug=False,
    #     visualize=True,
    #     save_diff=True
    # )
