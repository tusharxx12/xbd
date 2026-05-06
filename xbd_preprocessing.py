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
- FIXED: JSON structure parsing for different xBD formats
- ADDED: Debug mode with detailed logging
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

# Performance improvement - disable OpenCV threading overhead
cv2.setNumThreads(0)


# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    """Central configuration for preprocessing parameters."""

    # Paths (Kaggle environment) - Updated for multi-tier support
    # FIX 1: Correct default input path to xBD dataset location
    INPUT_BASE_PATH = Path("/kaggle/input/datasets/qianlanzz/xbd-dataset/xbd")
    OUTPUT_PATH = Path("/kaggle/working/processed_data")

    # Available tiers
    TIERS = ["tier1", "tier3"]

    # Image parameters
    DEFAULT_IMAGE_SIZE = 1024  # Fallback only
    TILE_SIZE = 512
    TILE_STRIDE = 512  # No overlap

    # Filtering
    # FIX 6: Lower min building ratio for better balance between recall and noise
    MIN_BUILDING_RATIO = 0.005  # Skip tiles with < 0.5% building pixels

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
        4: "Destroyed",
    }

    # Damage priority for mask overwrite (higher value = higher priority)
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

    # Verbose logging for debugging
    VERBOSE = False


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
        with open(json_path, "r") as f:
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
    parts = name.split("_")

    scene_parts = []
    for part in parts:
        if part in ("pre", "post"):
            break
        scene_parts.append(part)

    return "_".join(scene_parts)


def parse_polygons_from_wkt(wkt_string: str) -> List[np.ndarray]:
    """
    Parse WKT polygon string to list of numpy coordinate arrays.
    Handles ALL polygons in MultiPolygon.

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

        if geom.geom_type == "Polygon":
            coords = np.array(geom.exterior.coords)
            polygons.append(coords)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                if not poly.is_empty:
                    coords = np.array(poly.exterior.coords)
                    polygons.append(coords)
        elif geom.geom_type == "GeometryCollection":
            for g in geom.geoms:
                if g.geom_type == "Polygon" and not g.is_empty:
                    coords = np.array(g.exterior.coords)
                    polygons.append(coords)

    except Exception as e:
        if Config.VERBOSE:
            print(f"[WARNING] Failed parsing WKT: {e}")

    return polygons


def clip_coordinates(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Clip polygon coordinates to image bounds.

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
# JSON PARSING - FLEXIBLE FOR DIFFERENT xBD FORMATS
# ============================================================================


def extract_features_from_json(json_data: Dict) -> List[Dict]:
    """
    Extract features from JSON, handling different xBD formats.

    The xBD dataset has different JSON structures:
    - Format 1: json_data['features']['xy'] -> list of features
    - Format 2: json_data['features']['lng_lat'] -> list of features
    - Format 3: json_data directly contains features list

    Args:
        json_data: Parsed JSON dictionary

    Returns:
        List of feature dictionaries
    """
    features = []

    # Try different JSON structures
    if "features" in json_data:
        feat_container = json_data["features"]

        # Check if features is a dict with 'xy' or 'lng_lat' keys
        if isinstance(feat_container, dict):
            if "xy" in feat_container:
                features = feat_container["xy"]
            elif "lng_lat" in feat_container:
                features = feat_container["lng_lat"]
            else:
                # Maybe features dict has other structure
                for key, val in feat_container.items():
                    if isinstance(val, list):
                        features = val
                        break
        elif isinstance(feat_container, list):
            features = feat_container

    # GeoJSON format - features at top level
    if not features and isinstance(json_data, dict):
        if json_data.get("type") == "FeatureCollection":
            features = json_data.get("features", [])

    return features if isinstance(features, list) else []


def extract_polygon_and_damage(feature: Dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract polygon WKT and damage type from a feature.
    Handles different property structures in xBD.

    Args:
        feature: Feature dictionary

    Returns:
        Tuple of (wkt_string, damage_type) or (None, None) if not found
    """
    wkt_string = None
    damage_type = None

    # FIX 3: Only use feature properties, no fallback to entire feature
    properties = feature.get("properties", {})

    # FIX 2: Try to get WKT polygon - prioritize FEATURE LEVEL first, then properties level
    # Skip empty or null values
    for wkt_key in ["wkt", "feature_wkt", "pixelWkt"]:
        # Search WKT at FEATURE LEVEL first
        if wkt_key in feature:
            val = feature[wkt_key]

        # fallback to properties level
        elif wkt_key in properties:
            val = properties[wkt_key]

        else:
            continue

        # Skip empty, null, or non-string values
        if val and isinstance(val, str) and val.strip():
            wkt_string = val
            break

    # If no WKT, try geometry field
    if not wkt_string and "geometry" in feature:
        geom = feature["geometry"]
        if isinstance(geom, dict) and "coordinates" in geom:
            # Convert GeoJSON geometry to WKT
            try:
                from shapely.geometry import shape

                shapely_geom = shape(geom)
                wkt_string = shapely_geom.wkt
            except:
                pass

    # Try to get damage type
    # Common keys: 'subtype', 'damage', 'damage_type', '_damage'
    for damage_key in ["subtype", "damage", "damage_type", "_damage"]:
        if damage_key in properties:
            damage_type = properties[damage_key]
            break

    return wkt_string, damage_type


# ============================================================================
# MASK GENERATION
# ============================================================================


def create_mask(
    json_data: Dict, height: int, width: int, verbose: bool = False
) -> np.ndarray:
    """
    Create segmentation mask from JSON label data.
    Handles multiple JSON formats used in xBD dataset.

    Args:
        json_data: Parsed JSON dictionary with building annotations
        height: Output mask height
        width: Output mask width
        verbose: Print debug information

    Returns:
        Numpy array mask of shape (height, width)
        Values: 0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Extract features using flexible parser
    features = extract_features_from_json(json_data)

    if not features:
        if verbose:
            print(f"    [DEBUG] No features found in JSON")
        return mask

    if verbose:
        print(f"    [DEBUG] Found {len(features)} features")

    polygon_data = []
    skipped_no_wkt = 0
    skipped_no_damage = 0
    skipped_invalid = 0

    for feature in features:
        # Extract polygon and damage using flexible parser
        wkt_string, damage_type = extract_polygon_and_damage(feature)

        # Skip if no WKT
        if not wkt_string or not wkt_string.strip():
            skipped_no_wkt += 1
            continue

        # Skip if no damage type or unrecognized
        if not damage_type or damage_type not in Config.DAMAGE_CLASSES:
            skipped_no_damage += 1
            continue

        damage_class = Config.DAMAGE_CLASSES.get(damage_type, 0)

        # Skip background/unclassified
        if damage_class == 0:
            skipped_invalid += 1
            continue

        # Parse polygons from WKT
        polygons = parse_polygons_from_wkt(wkt_string)

        for coords in polygons:
            if coords is not None and len(coords) >= 3:
                polygon_data.append((coords, damage_class))

    if verbose:
        print(
            f"    [DEBUG] Valid polygons: {len(polygon_data)}, "
            f"Skipped (no wkt): {skipped_no_wkt}, "
            f"Skipped (no damage): {skipped_no_damage}, "
            f"Skipped (invalid): {skipped_invalid}"
        )

    # FIX 5: OPTIMIZE MASK DRAWING PERFORMANCE
    # Sort polygons by damage class (ascending) so higher classes overwrite lower classes naturally
    # Remove temp mask creation for better performance
    polygon_data.sort(key=lambda x: x[1])  # Sort by damage class (ascending)
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

    features = extract_features_from_json(json_data)

    for feature in features:
        wkt_string, _ = extract_polygon_and_damage(feature)

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


def tile_image(
    image: np.ndarray, tile_size: int = 512, stride: int = 512
) -> Dict[Tuple[int, int], np.ndarray]:
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
                tile = image[y : y + tile_size, x : x + tile_size, :]
            else:
                tile = image[y : y + tile_size, x : x + tile_size]

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
        scenes[scene_id]["pre"] = img_path

    for img_path in images_dir.glob("*_post_disaster.png"):
        scene_id = extract_scene_id(img_path.name)
        scenes[scene_id]["post"] = img_path

    if labels_dir.exists():
        for json_path in labels_dir.glob("*_post_disaster.json"):
            scene_id = extract_scene_id(json_path.name)
            scenes[scene_id]["json"] = json_path

    complete_scenes = {
        scene_id: files
        for scene_id, files in scenes.items()
        if all(k in files for k in ["pre", "post", "json"])
    }

    incomplete = len(scenes) - len(complete_scenes)
    if incomplete > 0:
        print(f"[INFO] Skipped {incomplete} incomplete scenes")

    return complete_scenes


# ============================================================================
# DEBUG: ANALYZE SINGLE JSON
# ============================================================================


def debug_analyze_json(json_path: Path):
    """
    Analyze a single JSON file to understand its structure.
    Useful for debugging parsing issues.

    Args:
        json_path: Path to JSON file
    """
    print(f"\n[DEBUG] Analyzing JSON: {json_path}")

    json_data = load_json(json_path)
    if json_data is None:
        print("  Failed to load JSON")
        return

    print(f"  Top-level keys: {list(json_data.keys())}")

    if "features" in json_data:
        feat = json_data["features"]
        print(f"  'features' type: {type(feat)}")

        if isinstance(feat, dict):
            print(f"  'features' keys: {list(feat.keys())}")
            for key in feat.keys():
                val = feat[key]
                if isinstance(val, list):
                    print(f"    '{key}': list with {len(val)} items")
                    if val:
                        first_item = val[0]
                        print(
                            f"      First item keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'not a dict'}"
                        )
                        if isinstance(first_item, dict):
                            # Show ALL keys at top level of feature
                            print(f"      === FULL FIRST FEATURE STRUCTURE ===")
                            for k, v in first_item.items():
                                if isinstance(v, dict):
                                    print(
                                        f"        '{k}': dict with keys {list(v.keys())}"
                                    )
                                elif isinstance(v, str) and len(v) > 100:
                                    print(f"        '{k}': '{v[:100]}...' (truncated)")
                                else:
                                    print(f"        '{k}': {v}")
                            if "properties" in first_item:
                                print(f"      === PROPERTIES ===")
                                for pk, pv in first_item["properties"].items():
                                    if isinstance(pv, str) and len(pv) > 100:
                                        print(
                                            f"        '{pk}': '{pv[:100]}...' (truncated)"
                                        )
                                    else:
                                        print(f"        '{pk}': {pv}")
        elif isinstance(feat, list):
            print(f"  'features' is a list with {len(feat)} items")
            if feat:
                print(
                    f"    First item keys: {list(feat[0].keys()) if isinstance(feat[0], dict) else 'not a dict'}"
                )

    # Try extracting features
    features = extract_features_from_json(json_data)
    print(f"  Extracted {len(features)} features")

    # Show what we find in first feature with buildings
    if features:
        for i, f in enumerate(features[:3]):
            print(f"  Feature {i} raw: {f}")

    if features:
        # Check first feature
        first = features[0]
        wkt_str, damage = extract_polygon_and_damage(first)
        print(f"  First feature - WKT: {'Yes' if wkt_str else 'No'}, Damage: {damage}")


# ============================================================================
# SCENE PROCESSING
# ============================================================================


def process_scene(
    scene_id: str,
    files: Dict[str, Path],
    output_dirs: Dict[str, Path],
    save_diff: bool = False,
    tier_prefix: str = "",
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Process a single scene: load, create mask, tile, and save.

    Args:
        scene_id: Scene identifier
        files: Dictionary with 'pre', 'post', 'json' paths
        output_dirs: Dictionary with output paths
        save_diff: Whether to save diff channel
        tier_prefix: Prefix to add to filenames (e.g., "tier1_")
        verbose: Print debug information

    Returns:
        Tuple of (tiles_saved, tiles_skipped)
    """
    tiles_saved = 0
    tiles_skipped = 0

    pre_img = cv2.imread(str(files["pre"]))
    post_img = cv2.imread(str(files["post"]))

    if pre_img is None or post_img is None:
        if verbose:
            print(f"[WARNING] Failed to load images for {scene_id}")
        return 0, 0

    height, width = pre_img.shape[:2]

    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

    diff_img = None
    if save_diff:
        diff_img = compute_change_image(pre_img, post_img)

    json_data = load_json(files["json"])

    if json_data is None:
        return 0, 0

    mask = create_mask(json_data, height, width, verbose=verbose)

    # Check if mask has any buildings
    total_building_pixels = np.sum(mask > 0)
    if verbose and total_building_pixels == 0:
        print(f"  [DEBUG] {scene_id}: Empty mask (no buildings)")

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
            tiles_skipped += 1
            continue

        # Add tier prefix to filename for unique identification
        tile_name = f"{tier_prefix}{scene_id}_tile_{row}_{col}.png"

        cv2.imwrite(
            str(output_dirs["pre"] / tile_name),
            cv2.cvtColor(pre_tile, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(output_dirs["post"] / tile_name),
            cv2.cvtColor(post_tile, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(output_dirs["masks"] / tile_name), mask_tile)

        if save_diff and diff_tiles is not None and (row, col) in diff_tiles:
            diff_tile = diff_tiles[(row, col)]
            cv2.imwrite(
                str(output_dirs["diff"] / tile_name),
                cv2.cvtColor(diff_tile, cv2.COLOR_RGB2BGR),
            )

        tiles_saved += 1

    return tiles_saved, tiles_skipped


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
    tile_files = list(output_dirs["masks"].glob("*.png"))

    # FIX 2: Return proper structure even when no tiles exist to avoid KeyError
    if not tile_files:
        return {
            "total_tiles": 0,
            "total_pixels": 0,
            "class_counts": {},
        }

    class_counts = defaultdict(int)
    total_pixels = 0

    for mask_path in tqdm(tile_files, desc="Computing statistics"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unique, counts = np.unique(mask, return_counts=True)

        # FIX 3: Convert numpy types to Python native types for JSON serialization
        for cls, cnt in zip(unique, counts):
            class_counts[int(cls)] += int(cnt)
            total_pixels += int(cnt)

    return {
        "total_tiles": len(tile_files),
        "total_pixels": total_pixels,
        "class_counts": dict(class_counts),
    }


def print_dataset_stats(stats: Dict, title: str = "DATASET STATISTICS"):
    """
    Print statistics about the processed dataset.

    Args:
        stats: Dictionary with statistics
        title: Title for the statistics section
    """
    if not stats or "total_tiles" not in stats:
        print(f"\n[WARNING] No statistics available for {title}")
        return

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Total tiles: {stats['total_tiles']:,}")
    print(f"\nDamage class distribution:")

    total_pixels = stats["total_pixels"]
    class_counts = stats["class_counts"]

    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        percentage = (count / total_pixels) * 100
        class_name = Config.CLASS_NAMES.get(cls, f"Class {cls}")
        print(f"  {class_name}: {count:,} pixels ({percentage:.2f}%)")


def merge_stats(stats_list: List[Dict]) -> Dict:
    """
    Merge statistics from multiple tiers.

    Args:
        stats_list: List of statistics dictionaries

    Returns:
        Merged statistics dictionary
    """
    merged = {"total_tiles": 0, "total_pixels": 0, "class_counts": defaultdict(int)}

    for stats in stats_list:
        if stats:
            merged["total_tiles"] += stats["total_tiles"]
            merged["total_pixels"] += stats["total_pixels"]
            for cls, count in stats["class_counts"].items():
                merged["class_counts"][cls] += count

    merged["class_counts"] = dict(merged["class_counts"])
    return merged


# ============================================================================
# VISUALIZATION
# ============================================================================


def visualize_samples(
    output_dirs: Dict[str, Path], num_samples: int = 3, show_diff: bool = False
):
    """
    Display random samples for visual verification.

    Args:
        output_dirs: Dictionary with output directory paths
        num_samples: Number of samples to display
        show_diff: Whether to show diff channel
    """
    tile_files = list(output_dirs["pre"].glob("*.png"))

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

    has_diff = show_diff and "diff" in output_dirs and output_dirs["diff"].exists()
    num_cols = 5 if has_diff else 4

    fig, axes = plt.subplots(
        num_samples, num_cols, figsize=(4 * num_cols, 5 * num_samples)
    )

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, pre_path in enumerate(samples):
        tile_name = pre_path.name

        pre_img = cv2.imread(str(output_dirs["pre"] / tile_name))
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)

        post_img = cv2.imread(str(output_dirs["post"] / tile_name))
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(output_dirs["masks"] / tile_name), cv2.IMREAD_GRAYSCALE)

        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in damage_colors.items():
            colored_mask[mask == class_id] = color

        overlay = post_img.copy()
        alpha = 0.4

        for class_id, color in damage_colors.items():
            if class_id == 0:
                continue
            mask_region = mask == class_id
            overlay[mask_region] = (
                (1 - alpha) * overlay[mask_region] + alpha * np.array(color)
            ).astype(np.uint8)

        axes[idx, 0].imshow(pre_img)
        axes[idx, 0].set_title(f"Pre-Disaster\n{tile_name[:30]}...")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(post_img)
        axes[idx, 1].set_title("Post-Disaster")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(colored_mask)
        axes[idx, 2].set_title("Damage Mask")
        axes[idx, 2].axis("off")

        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title("Overlay (Post + Mask)")
        axes[idx, 3].axis("off")

        if has_diff:
            diff_path = output_dirs["diff"] / tile_name
            if diff_path.exists():
                diff_img = cv2.imread(str(diff_path))
                diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)
                axes[idx, 4].imshow(diff_img)
                axes[idx, 4].set_title("Change (Diff)")
            else:
                axes[idx, 4].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[idx, 4].axis("off")

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=np.array(c) / 255, label=l)
        for l, c in [
            ("Background", [0, 0, 0]),
            ("No Damage", [0, 255, 0]),
            ("Minor Damage", [255, 255, 0]),
            ("Major Damage", [255, 165, 0]),
            ("Destroyed", [255, 0, 0]),
        ]
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(
        output_dirs["pre"].parent / "sample_visualization.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
    print(
        f"[INFO] Visualization saved to {output_dirs['pre'].parent / 'sample_visualization.png'}"
    )


def print_class_weights(stats: Dict):
    """
    Calculate and print recommended class weights for training.

    Args:
        stats: Dataset statistics dictionary
    """
    if not stats or not stats.get("class_counts"):
        return

    print("\n" + "=" * 60)
    print("RECOMMENDED CLASS WEIGHTS (for imbalanced training)")
    print("=" * 60)

    class_counts = stats["class_counts"]

    # Remove background for weight calculation
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


def preprocess_single_tier(
    input_path: Path,
    output_path: Path,
    tier_name: str,
    debug: bool = False,
    debug_limit: int = None,
    save_diff: bool = True,
    verbose: bool = False,
    clean_output: bool = False,
) -> Dict:
    """
    Process a single tier of the dataset.

    Args:
        input_path: Path to tier data
        output_path: Output path for processed data
        tier_name: Name of the tier (e.g., "tier1")
        debug: Enable debug mode
        debug_limit: Number of scenes in debug mode
        save_diff: Save diff channel
        verbose: Print detailed debug info
        clean_output: Clean existing output directories before processing

    Returns:
        Statistics dictionary for this tier
    """
    debug_limit = debug_limit or Config.DEBUG_LIMIT

    print("\n" + "=" * 60)
    print(f"PROCESSING {tier_name.upper()}")
    print("=" * 60)
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Debug mode: {debug} (limit: {debug_limit if debug else 'N/A'})")
    print("=" * 60)

    # Create output directories
    output_dirs = {
        "pre": output_path / "pre",
        "post": output_path / "post",
        "masks": output_path / "masks",
    }

    if save_diff:
        output_dirs["diff"] = output_path / "diff"

    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Discover scenes
    print("\n[INFO] Discovering scenes...")
    scenes = discover_scenes(input_path)

    if not scenes:
        print(f"[ERROR] No complete scenes found in {tier_name}!")
        return {}

    print(f"[INFO] Found {len(scenes)} complete scenes")

    # Debug: Analyze JSON files to check structure - find one with features
    if verbose or debug:
        print("\n[DEBUG] Looking for a JSON with features to analyze...")
        for scene_id, scene_files in list(scenes.items())[:20]:
            json_data = load_json(scene_files["json"])
            if json_data:
                features = extract_features_from_json(json_data)
                if features:
                    print(
                        f"[DEBUG] Found JSON with {len(features)} features: {scene_files['json']}"
                    )
                    debug_analyze_json(scene_files["json"])
                    break
        else:
            # If no JSON with features found, just show first one
            first_scene = list(scenes.values())[0]
            debug_analyze_json(first_scene["json"])

    if debug:
        scene_ids = list(scenes.keys())[:debug_limit]
        scenes = {sid: scenes[sid] for sid in scene_ids}
        print(f"[DEBUG] Processing {len(scenes)} scenes only")

    # Process scenes with tier prefix
    total_tiles = 0
    total_skipped = 0
    empty_mask_count = 0
    tier_prefix = f"{tier_name}_"

    for scene_id, files in tqdm(scenes.items(), desc=f"Processing {tier_name}"):
        try:
            tiles_saved, tiles_skipped = process_scene(
                scene_id,
                files,
                output_dirs,
                save_diff=save_diff,
                tier_prefix=tier_prefix,
                verbose=verbose,
            )
            total_tiles += tiles_saved
            total_skipped += tiles_skipped

            if tiles_saved == 0 and tiles_skipped == 0:
                empty_mask_count += 1

        except Exception as e:
            print(f"\n[ERROR] Failed to process {scene_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n[INFO] {tier_name} processing complete!")
    print(f"[INFO] Tiles saved: {total_tiles}")
    print(f"[INFO] Tiles skipped (low building ratio): {total_skipped}")
    print(f"[INFO] Scenes with empty masks: {empty_mask_count}")

    # Compute statistics
    stats = compute_dataset_stats(output_dirs)
    stats["tier"] = tier_name
    stats["scenes_processed"] = len(scenes)
    stats["tiles_skipped"] = total_skipped
    stats["empty_mask_scenes"] = empty_mask_count

    return stats


# ============================================================================
# MULTI-TIER PROCESSING
# ============================================================================


def preprocess_all_tiers(
    input_base_path: Path = None,
    output_path: Path = None,
    tiers: List[str] = None,
    debug: bool = False,
    debug_limit: int = None,
    save_diff: bool = True,
    visualize: bool = True,
    verbose: bool = False,
    clean_output: bool = False,
):
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
        verbose: Print detailed debug info
        clean_output: Clean existing output directories before processing
    """
    input_base_path = input_base_path or Config.INPUT_BASE_PATH
    output_path = output_path or Config.OUTPUT_PATH
    tiers = tiers or Config.TIERS

    print("=" * 60)
    print("xBD MULTI-TIER DATASET PREPROCESSING")
    print("=" * 60)
    print(f"Input base path: {input_base_path}")
    print(f"Output path: {output_path}")
    print(f"Tiers to process: {tiers}")
    print(f"Save diff channel: {save_diff}")
    print("=" * 60)

    # FIX 1: Move output cleaning logic here - clean ONLY ONCE before tier loop
    if clean_output:
        import shutil

        combined_output_path = output_path / "combined"
        dirs_to_clean = [
            combined_output_path / "pre",
            combined_output_path / "post",
            combined_output_path / "masks",
            combined_output_path / "diff",
        ]
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"[INFO] Cleaned existing directory: {dir_path}")

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
            save_diff=save_diff,
            verbose=verbose,
        )

        if stats:
            all_stats.append(stats)
            # Store output dirs for visualization
            combined_output_dirs = {
                "pre": tier_output_path / "pre",
                "post": tier_output_path / "post",
                "masks": tier_output_path / "masks",
            }
            if save_diff:
                combined_output_dirs["diff"] = tier_output_path / "diff"

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
            "tiers": [s["tier"] for s in all_stats],
            "individual": all_stats,
            "combined": merge_stats(all_stats) if len(all_stats) > 1 else all_stats[0],
        }

        stats_file = output_path / "combined" / "dataset_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats_output, f, indent=2)
        print(f"\n[INFO] Statistics saved to {stats_file}")


# ============================================================================
# LEGACY SINGLE SPLIT PROCESSING (for backward compatibility)
# ============================================================================


def preprocess_dataset(
    input_path: Path = None,
    output_path: Path = None,
    split: str = "train",
    debug: bool = False,
    debug_limit: int = None,
    visualize: bool = True,
    save_diff: bool = None,
):
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
        save_diff=save_diff,
    )

    if stats:
        print_dataset_stats(stats)
        print_class_weights(stats)

    if visualize and stats.get("total_tiles", 0) > 0:
        output_dirs = {
            "pre": output_path / split / "pre",
            "post": output_path / split / "post",
            "masks": output_path / split / "masks",
        }
        if save_diff:
            output_dirs["diff"] = output_path / split / "diff"

        print("\n[INFO] Generating visualization...")
        visualize_samples(output_dirs, num_samples=3, show_diff=save_diff)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # =========================================================
    # OPTION 1: Process ALL tiers (tier1 + tier3) combined
    # =========================================================
    # FIX 1: Correct input path to match dataset location
    preprocess_all_tiers(
        input_base_path=Path("/kaggle/input/datasets/qianlanzz/xbd-dataset/xbd"),
        output_path=Path("/kaggle/working/processed_data"),
        tiers=["tier1", "tier3"],  # Process both tiers
        debug=True,  # Set to True for testing first!
        debug_limit=5,  # Test with 5 scenes first
        save_diff=True,
        visualize=True,
        verbose=True,  # Enable detailed logging
        clean_output=True,  # FIX 1: Clean output directories once before processing
    )

    # =========================================================
    # OPTION 2: Full processing (after debug passes)
    # =========================================================
    # preprocess_all_tiers(
    #     input_base_path=Path("/kaggle/input/datasets/qianlanzz/xbd-dataset/xbd"),
    #     output_path=Path("/kaggle/working/processed_data"),
    #     tiers=["tier1", "tier3"],
    #     debug=False,
    #     save_diff=True,
    #     visualize=True,
    #     verbose=False,
    #     clean_output=True
    # )
import shutil
from pathlib import Path

def combine_processed_tiers(output_path: Path, tiers: list):
    """
    Combine multiple processed tiers into one unified dataset.

    Creates:
    processed_data/combined/
        ├── pre/
        ├── post/
        ├── masks/
        └── diff/
    """

    combined_dir = output_path / "combined"

    # Create directories
    for subdir in ["pre", "post", "masks", "diff"]:
        (combined_dir / subdir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("COMBINING TIERS")
    print("=" * 60)

    for tier in tiers:
        tier_dir = output_path / tier

        for subdir in ["pre", "post", "masks", "diff"]:
            src_dir = tier_dir / subdir
            dst_dir = combined_dir / subdir

            if not src_dir.exists():
                continue

            files = list(src_dir.glob("*"))

            print(f"[INFO] Copying {len(files)} files from {tier}/{subdir}")

            for file in files:
                dst_file = dst_dir / file.name

                # Avoid overwrite collisions
                if dst_file.exists():
                    dst_file = dst_dir / f"{tier}_{file.name}"

                shutil.copy2(file, dst_file)

    print("\n[INFO] Combined dataset created at:")
    print(combined_dir)
