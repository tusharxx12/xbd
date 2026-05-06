"""
High-Performance Inference for Satellite Damage Detection

Features:
- Model loading from checkpoint
- Test Time Augmentation (TTA) for higher F1 scores
- Full-image stitching for 1024x1024 test images
- Research paper quality visualizations
- Batch and single-image inference

Usage:
    from inference import Predictor

    predictor = Predictor("checkpoints/best_model.pth")
    mask = predictor.predict(pre_img, post_img, diff_img, use_tta=True)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.damage_detection_model import create_model, SatelliteDamageDetectionModel
from training.config import TrainingConfig, get_config, DAMAGE_CLASSES, DAMAGE_COLORS


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    # Model settings
    checkpoint_path: Path = Path("checkpoints/best_model.pth")
    num_classes: int = 5

    # Input settings
    tile_size: int = 512
    stride: int = 512  # Non-overlapping by default

    # TTA settings
    use_tta: bool = True
    tta_modes: Tuple[str, ...] = ("original", "hflip", "vflip")

    # Normalization (must match training)
    normalize_mean: Tuple[float, float, float] = (0.3260, 0.3753, 0.2511)
    normalize_std: Tuple[float, float, float] = (0.1074, 0.0881, 0.0880)

    # Performance
    use_amp: bool = True
    batch_size: int = 4

    # Output
    output_dir: Path = Path("inference_outputs")

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Predictor:
    """
    High-performance predictor for satellite damage detection.

    Features:
    - Test Time Augmentation (TTA) with flip ensembles
    - Full-image stitching for large test images
    - Mixed precision inference
    - Research-quality visualizations
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config: Optional[InferenceConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize predictor with trained model.

        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            config: Inference configuration
            device: Override device (cuda/cpu)
        """
        self.config = config or InferenceConfig()
        self.device = torch.device(device or self.config.device)

        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Predictor initialized on {self.device}")
        print(f"[INFO] TTA enabled: {self.config.use_tta}")
        print(f"[INFO] TTA modes: {self.config.tta_modes}")

    def _load_model(self, checkpoint_path: Union[str, Path]) -> nn.Module:
        """Load model from checkpoint"""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Create model architecture
        model = create_model(
            pretrained=False,
            num_classes=self.config.num_classes,
            use_deep_supervision=False,  # Not needed for inference
        )

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Handle potential key mismatches
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)

        print(f"[INFO] Model loaded from: {checkpoint_path}")

        return model

    def _normalize(self, image: np.ndarray) -> torch.Tensor:
        """Normalize image and convert to tensor"""
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply normalization
        mean = np.array(self.config.normalize_mean, dtype=np.float32)
        std = np.array(self.config.normalize_std, dtype=np.float32)
        image = (image - mean) / std

        # Convert to tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1)

        return tensor

    def _denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Denormalize tensor for visualization"""
        mean = torch.tensor(self.config.normalize_mean).view(3, 1, 1)
        std = torch.tensor(self.config.normalize_std).view(3, 1, 1)

        tensor = tensor * std + mean
        tensor = tensor.clamp(0, 1)

        # Convert to numpy (C, H, W) -> (H, W, C)
        image = tensor.permute(1, 2, 0).numpy()

        return image

    def _apply_tta_transform(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        diff: torch.Tensor,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply TTA transform to inputs"""
        if mode == "original":
            return pre, post, diff
        elif mode == "hflip":
            return torch.flip(pre, [-1]), torch.flip(post, [-1]), torch.flip(diff, [-1])
        elif mode == "vflip":
            return torch.flip(pre, [-2]), torch.flip(post, [-2]), torch.flip(diff, [-2])
        elif mode == "rot90":
            return torch.rot90(pre, 1, [-2, -1]), torch.rot90(post, 1, [-2, -1]), torch.rot90(diff, 1, [-2, -1])
        elif mode == "rot180":
            return torch.rot90(pre, 2, [-2, -1]), torch.rot90(post, 2, [-2, -1]), torch.rot90(diff, 2, [-2, -1])
        elif mode == "rot270":
            return torch.rot90(pre, 3, [-2, -1]), torch.rot90(post, 3, [-2, -1]), torch.rot90(diff, 3, [-2, -1])
        else:
            raise ValueError(f"Unknown TTA mode: {mode}")

    def _reverse_tta_transform(self, output: torch.Tensor, mode: str) -> torch.Tensor:
        """Reverse TTA transform on output"""
        if mode == "original":
            return output
        elif mode == "hflip":
            return torch.flip(output, [-1])
        elif mode == "vflip":
            return torch.flip(output, [-2])
        elif mode == "rot90":
            return torch.rot90(output, -1, [-2, -1])
        elif mode == "rot180":
            return torch.rot90(output, -2, [-2, -1])
        elif mode == "rot270":
            return torch.rot90(output, -3, [-2, -1])
        else:
            raise ValueError(f"Unknown TTA mode: {mode}")

    def _tile_image(
        self,
        image: np.ndarray,
        tile_size: int,
        stride: int,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Split image into overlapping tiles.

        Args:
            image: Input image (H, W, C)
            tile_size: Size of each tile
            stride: Stride between tiles

        Returns:
            tiles: List of tile arrays
            positions: List of (row, col) positions
        """
        h, w = image.shape[:2]
        tiles = []
        positions = []

        # Calculate number of tiles needed
        for row in range(0, h, stride):
            for col in range(0, w, stride):
                # Handle edge cases
                row_end = min(row + tile_size, h)
                col_end = min(col + tile_size, w)

                # Adjust start to ensure full tile size (if possible)
                row_start = max(0, row_end - tile_size)
                col_start = max(0, col_end - tile_size)

                tile = image[row_start:row_end, col_start:col_end]

                # Pad if necessary (edge tiles)
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    padded = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded

                tiles.append(tile)
                positions.append((row_start, col_start))

        return tiles, positions

    def _stitch_predictions(
        self,
        predictions: List[np.ndarray],
        positions: List[Tuple[int, int]],
        output_shape: Tuple[int, int],
        tile_size: int,
    ) -> np.ndarray:
        """
        Stitch tile predictions back into full image.

        Uses averaging for overlapping regions.

        Args:
            predictions: List of prediction arrays (tile_size, tile_size, num_classes)
            positions: List of (row, col) positions
            output_shape: (H, W) of full output
            tile_size: Size of each tile

        Returns:
            Full stitched prediction (H, W, num_classes)
        """
        h, w = output_shape
        num_classes = predictions[0].shape[-1] if predictions[0].ndim == 3 else self.config.num_classes

        # Accumulator and counter for averaging
        output = np.zeros((h, w, num_classes), dtype=np.float32)
        counts = np.zeros((h, w, 1), dtype=np.float32)

        for pred, (row, col) in zip(predictions, positions):
            # Calculate actual region (handle edge cases)
            row_end = min(row + tile_size, h)
            col_end = min(col + tile_size, w)
            pred_h = row_end - row
            pred_w = col_end - col

            # Add prediction to accumulator
            if pred.ndim == 2:
                # Convert class labels to one-hot
                pred_onehot = np.eye(num_classes)[pred[:pred_h, :pred_w]]
            else:
                pred_onehot = pred[:pred_h, :pred_w]

            output[row:row_end, col:col_end] += pred_onehot
            counts[row:row_end, col:col_end] += 1

        # Average overlapping regions
        output = output / np.maximum(counts, 1)

        return output

    @torch.no_grad()
    def predict_tile(
        self,
        pre_tile: torch.Tensor,
        post_tile: torch.Tensor,
        diff_tile: torch.Tensor,
        use_tta: bool = True,
    ) -> torch.Tensor:
        """
        Predict on a single tile with optional TTA.

        Args:
            pre_tile: Pre-disaster tile (1, 3, H, W)
            post_tile: Post-disaster tile (1, 3, H, W)
            diff_tile: Difference tile (1, 3, H, W)
            use_tta: Whether to use test-time augmentation

        Returns:
            Prediction probabilities (1, num_classes, H, W)
        """
        if use_tta:
            tta_outputs = []

            for mode in self.config.tta_modes:
                # Apply transform
                pre_t, post_t, diff_t = self._apply_tta_transform(
                    pre_tile, post_tile, diff_tile, mode
                )

                # Forward pass
                with autocast(enabled=self.config.use_amp):
                    output = self.model(pre_t, post_t, diff_t)
                    if isinstance(output, tuple):
                        output = output[0]

                # Softmax to get probabilities
                probs = F.softmax(output, dim=1)

                # Reverse transform
                probs = self._reverse_tta_transform(probs, mode)
                tta_outputs.append(probs)

            # Average TTA predictions
            output = torch.stack(tta_outputs).mean(dim=0)

        else:
            with autocast(enabled=self.config.use_amp):
                output = self.model(pre_tile, post_tile, diff_tile)
                if isinstance(output, tuple):
                    output = output[0]
            output = F.softmax(output, dim=1)

        return output

    def predict(
        self,
        pre_img: np.ndarray,
        post_img: np.ndarray,
        diff_img: Optional[np.ndarray] = None,
        use_tta: Optional[bool] = None,
        return_probs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict damage mask for full-size image.

        Handles images larger than tile_size by tiling and stitching.

        Args:
            pre_img: Pre-disaster image (H, W, 3) RGB uint8
            post_img: Post-disaster image (H, W, 3) RGB uint8
            diff_img: Difference image (optional, computed if not provided)
            use_tta: Override TTA setting
            return_probs: If True, also return probability map

        Returns:
            mask: Predicted damage mask (H, W) with class labels
            probs: (optional) Probability map (H, W, num_classes)
        """
        use_tta = use_tta if use_tta is not None else self.config.use_tta

        # Compute diff if not provided
        if diff_img is None:
            diff_img = cv2.absdiff(post_img, pre_img)

        h, w = pre_img.shape[:2]
        tile_size = self.config.tile_size
        stride = self.config.stride

        # Check if tiling is needed
        if h <= tile_size and w <= tile_size:
            # Single tile prediction
            pre_tensor = self._normalize(pre_img).unsqueeze(0).to(self.device)
            post_tensor = self._normalize(post_img).unsqueeze(0).to(self.device)
            diff_tensor = self._normalize(diff_img).unsqueeze(0).to(self.device)

            probs = self.predict_tile(pre_tensor, post_tensor, diff_tensor, use_tta)
            probs = probs.squeeze(0).cpu().numpy()
            probs = probs.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

            # Crop to original size if needed
            probs = probs[:h, :w]

        else:
            # Tile and stitch
            pre_tiles, positions = self._tile_image(pre_img, tile_size, stride)
            post_tiles, _ = self._tile_image(post_img, tile_size, stride)
            diff_tiles, _ = self._tile_image(diff_img, tile_size, stride)

            tile_predictions = []

            # Process tiles in batches
            num_tiles = len(pre_tiles)
            batch_size = self.config.batch_size

            for batch_start in range(0, num_tiles, batch_size):
                batch_end = min(batch_start + batch_size, num_tiles)

                # Prepare batch
                pre_batch = torch.stack([
                    self._normalize(pre_tiles[i])
                    for i in range(batch_start, batch_end)
                ]).to(self.device)

                post_batch = torch.stack([
                    self._normalize(post_tiles[i])
                    for i in range(batch_start, batch_end)
                ]).to(self.device)

                diff_batch = torch.stack([
                    self._normalize(diff_tiles[i])
                    for i in range(batch_start, batch_end)
                ]).to(self.device)

                # Predict batch (TTA applied per-tile)
                batch_probs = self.predict_tile(
                    pre_batch, post_batch, diff_batch, use_tta
                )

                # Convert to numpy
                batch_probs = batch_probs.cpu().numpy()

                for i in range(batch_probs.shape[0]):
                    tile_probs = batch_probs[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                    tile_predictions.append(tile_probs)

            # Stitch predictions
            probs = self._stitch_predictions(
                tile_predictions, positions, (h, w), tile_size
            )

        # Get class labels
        mask = np.argmax(probs, axis=-1).astype(np.uint8)

        if return_probs:
            return mask, probs
        return mask

    def predict_batch(
        self,
        pre_images: List[np.ndarray],
        post_images: List[np.ndarray],
        diff_images: Optional[List[np.ndarray]] = None,
        use_tta: Optional[bool] = None,
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """
        Predict damage masks for multiple images.

        Args:
            pre_images: List of pre-disaster images
            post_images: List of post-disaster images
            diff_images: List of difference images (optional)
            use_tta: Override TTA setting
            show_progress: Show progress bar

        Returns:
            List of predicted damage masks
        """
        if diff_images is None:
            diff_images = [None] * len(pre_images)

        masks = []
        iterator = zip(pre_images, post_images, diff_images)

        if show_progress:
            iterator = tqdm(list(iterator), desc="Predicting")

        for pre, post, diff in iterator:
            mask = self.predict(pre, post, diff, use_tta=use_tta)
            masks.append(mask)

        return masks

    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert class mask to RGB colored image"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in DAMAGE_COLORS.items():
            colored[mask == class_id] = color

        return colored

    def visualize_prediction(
        self,
        pre_img: np.ndarray,
        post_img: np.ndarray,
        prediction: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
        dpi: int = 300,
        figsize: Tuple[int, int] = (20, 5),
    ) -> plt.Figure:
        """
        Create research paper quality visualization.

        Shows: Pre | Post | Prediction | (Ground Truth if provided)

        Args:
            pre_img: Pre-disaster image (H, W, 3)
            post_img: Post-disaster image (H, W, 3)
            prediction: Predicted mask (H, W)
            ground_truth: Optional ground truth mask (H, W)
            save_path: Path to save figure
            title: Optional figure title
            dpi: Resolution for saving
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        num_cols = 4 if ground_truth is not None else 3

        fig, axes = plt.subplots(1, num_cols, figsize=figsize)

        # Pre-disaster
        axes[0].imshow(pre_img)
        axes[0].set_title("Pre-Disaster", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        # Post-disaster
        axes[1].imshow(post_img)
        axes[1].set_title("Post-Disaster", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        # Prediction
        pred_colored = self._colorize_mask(prediction)
        axes[2].imshow(pred_colored)
        axes[2].set_title("Prediction", fontsize=14, fontweight="bold")
        axes[2].axis("off")

        # Ground truth (if provided)
        if ground_truth is not None:
            gt_colored = self._colorize_mask(ground_truth)
            axes[3].imshow(gt_colored)
            axes[3].set_title("Ground Truth", fontsize=14, fontweight="bold")
            axes[3].axis("off")

        # Create legend
        legend_patches = [
            mpatches.Patch(
                color=np.array(DAMAGE_COLORS[i]) / 255.0,
                label=DAMAGE_CLASSES[i]
            )
            for i in range(5)
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=5,
            fontsize=11,
            frameon=True,
            bbox_to_anchor=(0.5, -0.05),
        )

        if title:
            fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"[INFO] Saved visualization: {save_path}")

        return fig

    def visualize_comparison_grid(
        self,
        samples: List[Dict[str, np.ndarray]],
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Damage Detection Results",
        dpi: int = 300,
    ) -> plt.Figure:
        """
        Create a grid visualization of multiple samples for papers.

        Args:
            samples: List of dicts with keys: pre, post, prediction, (gt optional)
            save_path: Path to save figure
            title: Figure title
            dpi: Resolution

        Returns:
            matplotlib Figure
        """
        num_samples = len(samples)
        has_gt = "gt" in samples[0]
        num_cols = 4 if has_gt else 3

        fig, axes = plt.subplots(
            num_samples, num_cols,
            figsize=(5 * num_cols, 4 * num_samples)
        )

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        col_titles = ["Pre-Disaster", "Post-Disaster", "Prediction"]
        if has_gt:
            col_titles.append("Ground Truth")

        for row, sample in enumerate(samples):
            # Pre
            axes[row, 0].imshow(sample["pre"])
            if row == 0:
                axes[row, 0].set_title(col_titles[0], fontsize=12, fontweight="bold")
            axes[row, 0].axis("off")

            # Post
            axes[row, 1].imshow(sample["post"])
            if row == 0:
                axes[row, 1].set_title(col_titles[1], fontsize=12, fontweight="bold")
            axes[row, 1].axis("off")

            # Prediction
            pred_colored = self._colorize_mask(sample["prediction"])
            axes[row, 2].imshow(pred_colored)
            if row == 0:
                axes[row, 2].set_title(col_titles[2], fontsize=12, fontweight="bold")
            axes[row, 2].axis("off")

            # Ground truth
            if has_gt:
                gt_colored = self._colorize_mask(sample["gt"])
                axes[row, 3].imshow(gt_colored)
                if row == 0:
                    axes[row, 3].set_title(col_titles[3], fontsize=12, fontweight="bold")
                axes[row, 3].axis("off")

        # Legend
        legend_patches = [
            mpatches.Patch(
                color=np.array(DAMAGE_COLORS[i]) / 255.0,
                label=DAMAGE_CLASSES[i]
            )
            for i in range(5)
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=5,
            fontsize=10,
            frameon=True,
            bbox_to_anchor=(0.5, -0.02),
        )

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"[INFO] Saved grid visualization: {save_path}")

        return fig

    def evaluate_on_dataset(
        self,
        data_loader,
        use_tta: Optional[bool] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset and compute metrics.

        Args:
            data_loader: PyTorch DataLoader with validation/test data
            use_tta: Override TTA setting
            verbose: Print detailed metrics

        Returns:
            Dictionary of computed metrics
        """
        from training.metrics import DamageMetrics
        from training.config import get_config

        config = get_config()
        metrics = DamageMetrics(
            num_classes=self.config.num_classes,
            ignore_index=config.IGNORE_INDEX,
        )

        use_tta = use_tta if use_tta is not None else self.config.use_tta

        # Denormalization parameters
        mean = torch.tensor(self.config.normalize_mean).view(1, 3, 1, 1)
        std = torch.tensor(self.config.normalize_std).view(1, 3, 1, 1)

        for batch in tqdm(data_loader, desc="Evaluating"):
            pre_imgs = batch["pre_img"]
            post_imgs = batch["post_img"]
            diff_imgs = batch["diff_img"]
            masks = batch["mask"]

            # Move to device
            pre_imgs = pre_imgs.to(self.device)
            post_imgs = post_imgs.to(self.device)
            diff_imgs = diff_imgs.to(self.device)

            # Predict
            probs = self.predict_tile(pre_imgs, post_imgs, diff_imgs, use_tta)

            # Update metrics
            metrics.update(probs, masks)

        # Get results
        results = metrics.get_all_metrics()

        if verbose:
            metrics.print_summary("EVALUATION RESULTS")

        return results


def run_inference_demo():
    """Demo inference on sample images"""
    print("=" * 60)
    print("INFERENCE DEMO")
    print("=" * 60)

    # Check for checkpoint
    checkpoint_path = Path("checkpoints/best_model.pth")

    if not checkpoint_path.exists():
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        print("Please train the model first or provide a valid checkpoint path.")
        return

    # Initialize predictor
    predictor = Predictor(checkpoint_path)

    # Create dummy test images
    pre_img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    post_img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    print("\n[INFO] Running inference on 1024x1024 test image...")
    start_time = time.time()

    # Predict with TTA
    mask = predictor.predict(pre_img, post_img, use_tta=True)

    inference_time = time.time() - start_time
    print(f"[INFO] Inference time: {inference_time:.2f}s")
    print(f"[INFO] Output shape: {mask.shape}")
    print(f"[INFO] Unique classes: {np.unique(mask)}")

    # Visualize
    predictor.visualize_prediction(
        pre_img, post_img, mask,
        save_path="inference_outputs/demo_prediction.png",
        title="Demo Prediction (Random Input)",
    )

    print("\n[INFO] Demo complete!")


if __name__ == "__main__":
    run_inference_demo()
