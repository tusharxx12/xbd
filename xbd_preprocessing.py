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
- Multi-tier processing (tier1 + tier3) with combined statistics

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
- Added multi-tier processing support
- Added combined statistics across tiers
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
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

    # Paths (Kaggle environment) - Updated for multi-tier support
    INPUT_BASE_PATH = Path("/kaggle/input/xview2-challenge-dataset")
    OUTPUT_PATH = Path("/kaggle/working/processed_data")

    # Available tiers
    TIERS = ["tier1", "tier3"]

    # Image parameters
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

    # Damage class names for display
    CLASS_NAMES = {
        0: "Background",
        1: "No Damage",
        2: "Minor",
        3: "Major",
        4: "Destroyed"
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

    # Optional change channel
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
    name = Path(filename).stem
    parts = name.split('_')

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

        if geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords)
            polygons.append(coords)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                if not poly.is_empty:
                    coords = np.array(poly.exterior.coords)
                    polygons.append(coords)
        elif geom.geom_type == 'GeometryCollection':
            for g in geom.geoms:
                if g.geom_type == 'Polygon' and not g.is_empty:
                    coords = np.array(g.exterior.coords)
                    polygons.append(coords)

    except Exception as e:
        pass

    return polygons


def clip_coordinates(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    [FIX #4 & #8] Clip polygon coordinates to image bounds.

    Args:
        coords: Numpy array of coordinates (N, 2) in (x, y) format
        height: Image height
        width: Image width

    Returns:
        Clipped coordinates as int32
    """
    clipped = coords.copy()
    clipped[:, 0] = np.clip(coords[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(coords[:, 1], 0, height - 1)
    return clipped.astype(np.int32)


# ============================================================================
# MASK GENERATION
# ============================================================================

def create_mask(json_data: Dict, height: int, width: int) -> np.ndarray:
    """
    Create segmentation mask from JSON label data.

    Args:
        json_data: Parsed JSON dictionary with building annotations
        height: Output mask height
        width: Output mask width

    Returns:
        Numpy array mask of shape (height, width)
        Values: 0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    features = json_data.get('features', {}).get('xy', [])

    if not features:
        return mask

    polygon_data = []

    for feature in features:
        properties = feature.get('properties', {})

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
                polygon_data.append((coords, damage_class))

    polygon_data.sort(key=lambda x: Config.DAMAGE_PRIORITY[x[1]])

    for coords, damage_class in polygon_data:
        coords = clip_coordinates(coords, height, width)
        polygon = coords.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon], color=int(damage_class))

    return mask


def create_building_mask(json_data: Dict, height: int, width: int) -> np.ndarray:
    """
    Create binary building mask (for localization task).

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

        wkt_string = properties.get('feature_wkt', '')
        if not wkt_string or not wkt_string.strip():
            continue

        polygons = parse_polygons_from_wkt(wkt_string)

        for coords in polygons:
            if coords is not None and len(coords) >= 3:
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
    Split image into tiles with index-based dictionary for safe matching.

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

            tiles[(row_idx, col_idx)] = tile

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
    building_pixels = np.sum(mask_tile > 0)
    return building_pixels / total_pixels


# ============================================================================
# CHANGE DETECTION
# ============================================================================

def compute_change_image(pre_img: np.ndarray, post_img: np.ndarray) -> np.ndarray:
    """
    Compute absolute difference between pre and post images.

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

    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        return {}

    for img_path in images_dir.glob("*_pre_disaster.png"):
        scene_id = extract_scene_id(img_path.name)
        scenes[scene_id]['pre'] = img_path

    for img_path in images_dir.glob("*_post_disaster.png"):
        scene_id = extract_scene_id(img_path.name)
        scenes[scene_id]['post'] = img_path

    if labels_dir.exists():
        for json_path in labels_dir.glob("*_post_disaster.json"):
            scene_id = extract_scene_id(json_path.name)
            scenes[scene_id]['json'] = json_path

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
                  save_diff: bool = False,
                  tier_prefix: str = "") -> int:
    """
    Process a single scene: load, create mask, tile, and save.

    Args:
        scene_id: Scene identifier
        files: Dictionary with 'pre', 'post', 'json' paths
        output_dirs: Dictionary with output paths
        save_diff: Whether to save diff channel
        tier_prefix: Prefix to add to filenames (e.g., "tier1_")

    Returns:
        Number of tiles saved
    """
    tiles_saved = 0

    pre_img = cv2.imread(str(files['pre']))
    post_img = cv2.imread(str(files['post']))

    if pre_img is None or post_img is None:
        print(f"[WARNING] Failed to load images for {scene_id}")
        return 0

    height, width = pre_img.shape[:2]

    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

    diff_img = None
    if save_diff:
        diff_img = compute_change_image(pre_img, post_img)

    json_data = load_json(files['json'])

    if json_data is None:
        return 0

    mask = create_mask(json_data, height, width)

    pre_tiles = tile_image(pre_img, Config.TILE_SIZE, Config.TILE_STRIDE)
    post_tiles = tile_image(post_img, Config.TILE_SIZE, Config.TILE_STRIDE)
    mask_tiles = tile_image(mask, Config.TILE_SIZE, Config.TILE_STRIDE)

    diff_tiles = None
    if save_diff and diff_img is not None:
        diff_tiles = tile_image(diff_img, Config.TILE_SIZE, Config.TILE_STRIDE)

    for (row, col), pre_tile in pre_tiles.items():
        if (row, col) not in post_tiles or (row, col) not in mask_tiles:
            continue

        post_tile = post_tiles[(row, col)]
        mask_tile = mask_tiles[(row, col)]

        building_ratio = compute_building_ratio(mask_tile)

        if building_ratio < Config.MIN_BUILDING_RATIO:
            continue

        # Add tier prefix to filename for unique identification
        tile_name = f"{tier_prefix}{scene_id}_tile_{row}_{col}.png"

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

        if save_diff and diff_tiles is not None and (row, col) in diff_tiles:
            diff_tile = diff_tiles[(row, col)]
            cv2.imwrite(
                str(output_dirs['diff'] / tile_name),
                cv2.cvtColor(diff_tile, cv2.COLOR_RGB2BGR)
            )

        tiles_saved += 1

    return tiles_saved


# ============================================================================
# STATISTICS
# ============================================================================

def compute_dataset_stats(output_dirs: Dict[str, Path]) -> Dict:
    """
    Compute statistics about the processed dataset.

    Args:
        output_dirs: Dictionary with output directory paths

    Returns:
        Dictionary with statistics
    """
    tile_files = list(output_dirs['masks'].glob("*.png"))

    if not tile_files:
        return {}

    class_counts = defaultdict(int)
    total_pixels = 0

    for mask_path in tqdm(tile_files, desc="Computing statistics"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unique, counts = np.unique(mask, return_counts=True)

        for cls, cnt in zip(unique, counts):
            class_counts[cls] += cnt
            total_pixels += cnt

    return {
        'total_tiles': len(tile_files),
        'total_pixels': total_pixels,
        'class_counts': dict(class_counts)
    }


def print_dataset_stats(stats: Dict, title: str = "DATASET STATISTICS"):
    """
    Print statistics about the processed dataset.

    Args:
        stats: Dictionary with statistics
        title: Title for the statistics section
    """
    if not stats:
        print("[WARNING] No statistics available")
        return

    print("\n" + "="*60)
    print(title)
    print("="*60)
    print(f"Total tiles: {stats['total_tiles']:,}")
    print(f"\nDamage class distribution:")

    total_pixels = stats['total_pixels']
    class_counts = stats['class_counts']

    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        percentage = (count / total_pixels) * 100
        class_name = Config.CLASS_NAMES.get(cls, f'Class {cls}')
        print(f"  {class_name}: {count:,} pixels ({percentage:.2f}%)")


def merge_stats(stats_list: List[Dict]) -> Dict:
    """
    Merge statistics from multiple tiers.

    Args:
        stats_list: List of statistics dictionaries

    Returns:
        Merged statistics dictionary
    """
    merged = {
        'total_tiles': 0,
        'total_pixels': 0,
        'class_counts': defaultdict(int)
    }

    for stats in stats_list:
        if stats:
            merged['total_tiles'] += stats['total_tiles']
            merged['total_pixels'] += stats['total_pixels']
            for cls, count in stats['class_counts'].items():
                merged['class_counts'][cls] += count

    merged['class_counts'] = dict(merged['class_counts'])
    return merged


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_samples(output_dirs: Dict[str, Path], num_samples: int = 3, show_diff: bool = False):
    """
    Display random samples for visual verification.

    Args:
        output_dirs: Dictionary with output directory paths
        num_samples: Number of samples to display
        show_diff: Whether to show diff channel
    """
    tile_files = list(output_dirs['pre'].glob("*.png"))

    if not tile_files:
        print("[WARNING] No processed tiles found for visualization")
        return

    samples = random.sample(tile_files, min(num_samples, len(tile_files)))

    damage_colors = {
        0: [0, 0, 0],
        1: [0, 255, 0],
        2: [255, 255, 0],
        3: [255, 165, 0],
        4: [255, 0, 0],
    }

    has_diff = show_diff and 'diff' in output_dirs and output_dirs['diff'].exists()
    num_cols = 5 if has_diff else 4

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4 * num_cols, 5 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, pre_path in enumerate(samples):
        tile_name = pre_path.name

        pre_img = cv2.imread(str(output_dirs['pre'] / tile_name))
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)

        post_img = cv2.imread(str(output_dirs['post'] / tile_name))
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(output_dirs['masks'] / tile_name), cv2.IMREAD_GRAYSCALE)

        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in damage_colors.items():
            colored_mask[mask == class_id] = color

        overlay = post_img.copy()
        alpha = 0.4

        for class_id, color in damage_colors.items():
            if class_id == 0:
                continue
            mask_region = (mask == class_id)
            overlay[mask_region] = (
                (1 - alpha) * overlay[mask_region] + alpha * np.array(color)
            ).astype(np.uint8)

        axes[idx, 0].imshow(pre_img)
        axes[idx, 0].set_title(f"Pre-Disaster\n{tile_name[:30]}...")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(post_img)
        axes[idx, 1].set_title("Post-Disaster")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(colored_mask)
        axes[idx, 2].set_title("Damage Mask")
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title("Overlay (Post + Mask)")
        axes[idx, 3].axis('off')

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


def print_class_weights(stats: Dict):
    """
    Calculate and print recommended class weights for training.

    Args:
        stats: Dataset statistics dictionary
    """
    if not stats or not stats.get('class_counts'):
        return

    print("\n" + "="*60)
    print("RECOMMENDED CLASS WEIGHTS (for imbalanced training)")
    print("="*60)

    class_counts = stats['class_counts']

    # Remove background for weight calculation (or keep if needed)
    building_classes = {k: v for k, v in class_counts.items() if k > 0}

    if not building_classes:
        print("[WARNING] No building classes found")
        return

    total_building_pixels = sum(building_classes.values())
    num_classes = len(building_classes)

    print("\nMethod 1: Inverse Frequency")
    print("-" * 40)
    weights_inv = {}
    for cls, count in sorted(building_classes.items()):
        weight = total_building_pixels / (num_classes * count)
        weights_inv[cls] = weight
        print(f"  Class {cls} ({Config.CLASS_NAMES[cls]}): {weight:.4f}")

    print("\nMethod 2: Effective Number of Samples (beta=0.99)")
    print("-" * 40)
    beta = 0.99
    weights_eff = {}
    for cls, count in sorted(building_classes.items()):
        effective_num = (1 - beta**count) / (1 - beta)
        weight = (1 - beta) / (1 - beta**count)
        weights_eff[cls] = weight
        print(f"  Class {cls} ({Config.CLASS_NAMES[cls]}): {weight:.6f}")

    # Normalize weights
    print("\nNormalized Weights (sum=num_classes):")
    print("-" * 40)
    weight_sum = sum(weights_inv.values())
    for cls, weight in sorted(weights_inv.items()):
        normalized = (weight / weight_sum) * num_classes
        print(f"  Class {cls} ({Config.CLASS_NAMES[cls]}): {normalized:.4f}")


# ============================================================================
# SINGLE TIER PROCESSING
# ============================================================================

def preprocess_single_tier(input_path: Path,
                           output_path: Path,
                           tier_name: str,
                           debug: bool = False,
                           debug_limit: int = None,
                           save_diff: bool = True) -> Dict:
    """
    Process a single tier of the dataset.

    Args:
        input_path: Path to tier data
        output_path: Output path for processed data
        tier_name: Name of the tier (e.g., "tier1")
        debug: Enable debug mode
        debug_limit: Number of scenes in debug mode
        save_diff: Save diff channel

    Returns:
        Statistics dictionary for this tier
    """
    debug_limit = debug_limit or Config.DEBUG_LIMIT

    print("\n" + "="*60)
    print(f"PROCESSING {tier_name.upper()}")
    print("="*60)
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Debug mode: {debug} (limit: {debug_limit if debug else 'N/A'})")
    print("="*60)

    # Create output directories
    output_dirs = {
        'pre': output_path / 'pre',
        'post': output_path / 'post',
        'masks': output_path / 'masks',
    }

    if save_diff:
        output_dirs['diff'] = output_path / 'diff'

    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Discover scenes
    print("\n[INFO] Discovering scenes...")
    scenes = discover_scenes(input_path)

    if not scenes:
        print(f"[ERROR] No complete scenes found in {tier_name}!")
        return {}

    print(f"[INFO] Found {len(scenes)} complete scenes")

    if debug:
        scene_ids = list(scenes.keys())[:debug_limit]
        scenes = {sid: scenes[sid] for sid in scene_ids}
        print(f"[DEBUG] Processing {len(scenes)} scenes only")

    # Process scenes with tier prefix
    total_tiles = 0
    tier_prefix = f"{tier_name}_"

    for scene_id, files in tqdm(scenes.items(), desc=f"Processing {tier_name}"):
        try:
            tiles_saved = process_scene(
                scene_id, files, output_dirs,
                save_diff=save_diff,
                tier_prefix=tier_prefix
            )
            total_tiles += tiles_saved
        except Exception as e:
            print(f"\n[ERROR] Failed to process {scene_id}: {e}")
            continue

    print(f"\n[INFO] {tier_name} processing complete!")
    print(f"[INFO] Tiles saved: {total_tiles}")

    # Compute statistics
    stats = compute_dataset_stats(output_dirs)
    stats['tier'] = tier_name
    stats['scenes_processed'] = len(scenes)

    return stats


# ============================================================================
# MULTI-TIER PROCESSING
# ============================================================================

def preprocess_all_tiers(input_base_path: Path = None,
                         output_path: Path = None,
                         tiers: List[str] = None,
                         debug: bool = False,
                         debug_limit: int = None,
                         save_diff: bool = True,
                         visualize: bool = True):
    """
    Process all tiers and output combined statistics.

    Args:
        input_base_path: Base path containing tier directories
        output_path: Output path for processed data
        tiers: List of tier names to process
        debug: Enable debug mode
        debug_limit: Number of scenes per tier in debug mode
        save_diff: Save diff channel
        visualize: Show sample visualizations
    """
    input_base_path = input_base_path or Config.INPUT_BASE_PATH
    output_path = output_path or Config.OUTPUT_PATH
    tiers = tiers or Config.TIERS

    print("="*60)
    print("xBD MULTI-TIER DATASET PREPROCESSING")
    print("="*60)
    print(f"Input base path: {input_base_path}")
    print(f"Output path: {output_path}")
    print(f"Tiers to process: {tiers}")
    print(f"Save diff channel: {save_diff}")
    print("="*60)

    all_stats = []
    combined_output_dirs = None

    for tier in tiers:
        tier_input_path = input_base_path / tier
        tier_output_path = output_path / "combined"  # All tiers go to same output

        if not tier_input_path.exists():
            print(f"\n[WARNING] Tier path not found: {tier_input_path}")
            continue

        stats = preprocess_single_tier(
            input_path=tier_input_path,
            output_path=tier_output_path,
            tier_name=tier,
            debug=debug,
            debug_limit=debug_limit,
            save_diff=save_diff
        )

        if stats:
            all_stats.append(stats)
            # Store output dirs for visualization
            combined_output_dirs = {
                'pre': tier_output_path / 'pre',
                'post': tier_output_path / 'post',
                'masks': tier_output_path / 'masks',
            }
            if save_diff:
                combined_output_dirs['diff'] = tier_output_path / 'diff'

    # Print individual tier statistics
    for stats in all_stats:
        print_dataset_stats(stats, f"{stats['tier'].upper()} STATISTICS")

    # Print combined statistics
    if len(all_stats) > 1:
        combined_stats = merge_stats(all_stats)
        print_dataset_stats(combined_stats, "COMBINED STATISTICS (ALL TIERS)")
        print_class_weights(combined_stats)
    elif len(all_stats) == 1:
        print_class_weights(all_stats[0])

    # Visualize samples from combined output
    if visualize and combined_output_dirs:
        print("\n[INFO] Generating visualization...")
        visualize_samples(combined_output_dirs, num_samples=3, show_diff=save_diff)

    # Save statistics to JSON
    if all_stats:
        stats_output = {
            'tiers': [s['tier'] for s in all_stats],
            'individual': all_stats,
            'combined': merge_stats(all_stats) if len(all_stats) > 1 else all_stats[0]
        }

        stats_file = output_path / "combined" / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_output, f, indent=2)
        print(f"\n[INFO] Statistics saved to {stats_file}")


# ============================================================================
# LEGACY SINGLE SPLIT PROCESSING (for backward compatibility)
# ============================================================================

def preprocess_dataset(input_path: Path = None,
                       output_path: Path = None,
                       split: str = "train",
                       debug: bool = False,
                       debug_limit: int = None,
                       visualize: bool = True,
                       save_diff: bool = None):
    """
    Legacy function for single split processing.
    For multi-tier processing, use preprocess_all_tiers() instead.
    """
    input_path = input_path or Config.INPUT_BASE_PATH
    output_path = output_path or Config.OUTPUT_PATH
    debug_limit = debug_limit or Config.DEBUG_LIMIT
    save_diff = save_diff if save_diff is not None else Config.SAVE_DIFF_CHANNEL

    data_path = input_path / split

    stats = preprocess_single_tier(
        input_path=data_path,
        output_path=output_path / split,
        tier_name=split,
        debug=debug,
        debug_limit=debug_limit,
        save_diff=save_diff
    )

    if stats:
        print_dataset_stats(stats)
        print_class_weights(stats)

    if visualize and stats.get('total_tiles', 0) > 0:
        output_dirs = {
            'pre': output_path / split / 'pre',
            'post': output_path / split / 'post',
            'masks': output_path / split / 'masks',
        }
        if save_diff:
            output_dirs['diff'] = output_path / split / 'diff'

        print("\n[INFO] Generating visualization...")
        visualize_samples(output_dirs, num_samples=3, show_diff=save_diff)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # =========================================================
    # OPTION 1: Process ALL tiers (tier1 + tier3) combined
    # =========================================================
    preprocess_all_tiers(
        input_base_path=Path("/kaggle/input/xview2-challenge-dataset"),
        output_path=Path("/kaggle/working/processed_data"),
        tiers=["tier1", "tier3"],  # Process both tiers
        debug=False,               # Set to True for testing
        debug_limit=10,            # Scenes per tier in debug mode
        save_diff=True,
        visualize=True
    )

    # =========================================================
    # OPTION 2: Process single tier only
    # =========================================================
    # preprocess_dataset(
    #     input_path=Path("/kaggle/input/xview2-challenge-dataset"),
    #     output_path=Path("/kaggle/working/processed_data"),
    #     split="tier1",
    #     debug=False,
    #     visualize=True,
    #     save_diff=True
    # )
