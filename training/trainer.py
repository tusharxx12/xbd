"""
Training Engine for Satellite Damage Detection

Contains:
- Trainer class with full training loop
- Mixed precision (FP16) training
- Gradient clipping for transformer stability
- Learning rate scheduling (OneCycleLR / CosineAnnealingLR)
- Early stopping based on validation F1
- Visual validation with side-by-side comparisons
- Weights & Biases integration
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .config import TrainingConfig, DAMAGE_CLASSES, DAMAGE_COLORS
from .metrics import DamageMetrics, MetricTracker


class EarlyStopping:
    """
    Early stopping based on validation metric.

    Stops training if metric doesn't improve for `patience` epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 1e-4,
    ):
        """
        Args:
            patience: Epochs to wait before stopping
            mode: "max" if higher is better, "min" if lower is better
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

    def step(self, value: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value
            epoch: Current epoch

        Returns:
            True if training should stop
        """
        if self.mode == "max":
            improved = value > (self.best_value + self.min_delta)
        else:
            improved = value < (self.best_value - self.min_delta)

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Training engine for Satellite Damage Detection model.

    Features:
    - Mixed precision (FP16) training with automatic scaling
    - Gradient clipping for transformer stability
    - Learning rate scheduling (OneCycle or Cosine)
    - Early stopping on validation F1
    - Visual validation with qualitative analysis plots
    - Weights & Biases logging
    - Checkpoint saving/loading
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: The damage detection model
            loss_fn: Loss function module
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device (default: from config)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(config.device)

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup optimizer with differential learning rates
        self._setup_optimizer()

        # Setup scheduler
        self._setup_scheduler()

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=config.use_amp)

        # Metrics
        self.train_metrics = DamageMetrics(
            num_classes=config.num_classes,
            ignore_index=config.IGNORE_INDEX,
        )
        self.val_metrics = DamageMetrics(
            num_classes=config.num_classes,
            ignore_index=config.IGNORE_INDEX,
        )
        self.metric_tracker = MetricTracker()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode=config.early_stopping_mode,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_f1 = 0.0

        # Wandb
        self.wandb_run = None
        if config.use_wandb:
            self._setup_wandb()

    def _setup_optimizer(self):
        """Setup AdamW optimizer with differential learning rates"""
        # Get parameter groups
        if hasattr(self.model, "get_parameter_groups"):
            param_groups = self.model.get_parameter_groups(
                lr=self.config.lr,
                backbone_lr_mult=self.config.backbone_lr_mult,
            )
        else:
            param_groups = [{"params": self.model.parameters(), "lr": self.config.lr}]

        self.optimizer = AdamW(
            param_groups,
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        total_steps = len(self.train_loader) * self.config.epochs

        if self.config.scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                total_steps=total_steps,
                pct_start=self.config.warmup_epochs / self.config.epochs,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=1000.0,
            )
            self.scheduler_step_per_batch = True
        else:  # cosine
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr,
            )
            self.scheduler_step_per_batch = False

    def _setup_wandb(self):
        """Initialize Weights & Biases logging"""
        try:
            import wandb

            # Generate run name if not specified
            run_name = self.config.wandb_run_name or \
                f"xbd_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=run_name,
                config=self.config.to_dict(),
                reinit=True,
            )

            # Watch model gradients
            wandb.watch(self.model, log="gradients", log_freq=100)

            print(f"[INFO] Wandb initialized: {run_name}")

        except ImportError:
            print("[WARNING] wandb not installed. Logging disabled.")
            self.config.use_wandb = False
        except Exception as e:
            print(f"[WARNING] wandb initialization failed: {e}")
            self.config.use_wandb = False

    def _log_wandb(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb"""
        if self.config.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()

        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_dice_loss = 0.0
        num_batches = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            pre_img = batch["pre_img"].to(self.device)
            post_img = batch["post_img"].to(self.device)
            diff_img = batch["diff_img"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                # Get model outputs
                outputs = self.model(pre_img, post_img, diff_img)

                # Handle deep supervision
                if isinstance(outputs, tuple):
                    logits, aux_outputs = outputs
                    losses = self.loss_fn(logits, mask, aux_outputs)
                else:
                    logits = outputs
                    losses = self.loss_fn(logits, mask)

                loss = losses["total_loss"]

                # Scale loss for gradient accumulation
                if self.config.accumulation_steps > 1:
                    loss = loss / self.config.accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping for transformer stability
                if self.config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip_norm,
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Scheduler step (for OneCycleLR)
                if self.scheduler_step_per_batch:
                    self.scheduler.step()

            # Update metrics
            self.train_metrics.update(logits.detach(), mask)

            # Accumulate losses
            epoch_loss += losses["total_loss"].item()
            epoch_ce_loss += losses.get("ce_loss", torch.tensor(0.0)).item()
            epoch_dice_loss += losses.get("dice_loss", torch.tensor(0.0)).item()
            num_batches += 1

            self.global_step += 1

            # Log batch metrics
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  Batch [{batch_idx:4d}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {current_lr:.2e}"
                )

                # Log to wandb
                self._log_wandb(
                    {
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": current_lr,
                    },
                    step=self.global_step,
                )

        # Compute epoch metrics
        epoch_time = time.time() - start_time
        train_metrics_dict = self.train_metrics.get_all_metrics()

        results = {
            "train_loss": epoch_loss / num_batches,
            "train_ce_loss": epoch_ce_loss / num_batches,
            "train_dice_loss": epoch_dice_loss / num_batches,
            "train_time": epoch_time,
            **{f"train_{k}": v for k, v in train_metrics_dict.items()},
        }

        # Step scheduler (for CosineAnnealingLR)
        if not self.scheduler_step_per_batch:
            self.scheduler.step()

        return results

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on the validation set.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()

        epoch_loss = 0.0
        num_batches = 0

        start_time = time.time()

        for batch in self.val_loader:
            # Move data to device
            pre_img = batch["pre_img"].to(self.device)
            post_img = batch["post_img"].to(self.device)
            diff_img = batch["diff_img"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(pre_img, post_img, diff_img)

                # Handle deep supervision (only use main output for val)
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs

                losses = self.loss_fn(logits, mask)
                loss = losses["total_loss"]

            # Update metrics
            self.val_metrics.update(logits, mask)
            epoch_loss += loss.item()
            num_batches += 1

        # Compute epoch metrics
        val_time = time.time() - start_time
        val_metrics_dict = self.val_metrics.get_all_metrics()

        results = {
            "val_loss": epoch_loss / num_batches,
            "val_time": val_time,
            **{f"val_{k}": v for k, v in val_metrics_dict.items()},
        }

        return results

    def visualize_predictions(
        self,
        num_samples: int = 4,
        save_path: Optional[Path] = None,
    ):
        """
        Generate visual validation with side-by-side comparisons.

        Creates images showing: Pre | Post | Ground Truth | Prediction
        for qualitative analysis in research papers.
        """
        self.model.eval()

        # Get samples from validation set
        samples = []
        for batch in self.val_loader:
            for i in range(batch["pre_img"].shape[0]):
                samples.append({
                    "pre_img": batch["pre_img"][i],
                    "post_img": batch["post_img"][i],
                    "diff_img": batch["diff_img"][i],
                    "mask": batch["mask"][i],
                    "tile_id": batch["tile_id"][i],
                })
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break

        # Create figure
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        # Denormalization
        mean = torch.tensor(self.config.normalize_mean).view(3, 1, 1)
        std = torch.tensor(self.config.normalize_std).view(3, 1, 1)

        with torch.no_grad():
            for idx, sample in enumerate(samples):
                # Get prediction
                pre = sample["pre_img"].unsqueeze(0).to(self.device)
                post = sample["post_img"].unsqueeze(0).to(self.device)
                diff = sample["diff_img"].unsqueeze(0).to(self.device)

                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(pre, post, diff)
                    if isinstance(outputs, tuple):
                        logits, _ = outputs
                    else:
                        logits = outputs
                    pred = logits.argmax(dim=1).squeeze(0).cpu()

                # Denormalize images
                pre_vis = (sample["pre_img"] * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()
                post_vis = (sample["post_img"] * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()

                # Create colored masks
                gt_colored = self._colorize_mask(sample["mask"].numpy())
                pred_colored = self._colorize_mask(pred.numpy())

                # Plot
                axes[idx, 0].imshow(pre_vis)
                axes[idx, 0].set_title(f"Pre-disaster\n{sample['tile_id'][:30]}", fontsize=10)
                axes[idx, 0].axis("off")

                axes[idx, 1].imshow(post_vis)
                axes[idx, 1].set_title("Post-disaster", fontsize=10)
                axes[idx, 1].axis("off")

                axes[idx, 2].imshow(gt_colored)
                axes[idx, 2].set_title("Ground Truth", fontsize=10)
                axes[idx, 2].axis("off")

                axes[idx, 3].imshow(pred_colored)
                axes[idx, 3].set_title("Prediction", fontsize=10)
                axes[idx, 3].axis("off")

        # Add legend
        legend_patches = [
            mpatches.Patch(color=np.array(DAMAGE_COLORS[i]) / 255.0, label=DAMAGE_CLASSES[i])
            for i in range(5)
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=5,
            fontsize=12,
            bbox_to_anchor=(0.5, -0.02),
        )

        plt.suptitle(
            f"Epoch {self.current_epoch} - Validation Predictions",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.config.visualization_dir / f"epoch_{self.current_epoch:03d}_predictions.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Saved visualization to: {save_path}")

        # Log to wandb
        if self.config.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.log({"val/predictions": wandb.Image(str(save_path))})

    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert class mask to RGB colored image"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in DAMAGE_COLORS.items():
            colored[mask == class_id] = color

        return colored

    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        """Save training checkpoint"""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch:03d}.pth"

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_f1": self.best_val_f1,
            "config": self.config.to_dict(),
        }

        save_path = self.config.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"[INFO] Saved checkpoint: {save_path}")

        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"[INFO] Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_f1 = checkpoint["best_val_f1"]

        print(f"[INFO] Loaded checkpoint from epoch {self.current_epoch}")

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Training history and best metrics
        """
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.lr}")
        print(f"Mixed precision: {self.config.use_amp}")
        print(f"Early stopping patience: {self.config.early_stopping_patience}")
        print("=" * 70 + "\n")

        training_start = time.time()
        history = {"train": [], "val": []}

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1}/{self.config.epochs}")
            print(f"{'='*70}")

            # Train
            train_metrics = self.train_epoch()
            history["train"].append(train_metrics)

            print(f"\n📊 Training Results:")
            print(f"  Loss: {train_metrics['train_loss']:.4f}")
            print(f"  F1 (Macro): {train_metrics['train_f1_macro']*100:.2f}%")
            print(f"  Time: {train_metrics['train_time']:.1f}s")

            # Validate
            val_metrics = self.validate()
            history["val"].append(val_metrics)

            print(f"\n📊 Validation Results:")
            print(f"  Loss: {val_metrics['val_loss']:.4f}")
            print(f"  F1 (Macro): {val_metrics['val_f1_macro']*100:.2f}%")
            print(f"  mIoU: {val_metrics['val_miou']*100:.2f}%")
            print(f"  Time: {val_metrics['val_time']:.1f}s")

            # Track best model
            current_f1 = val_metrics["val_f1_macro"]
            is_best = current_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = current_f1
                print(f"\n🎉 New best validation F1: {self.best_val_f1*100:.2f}%")

            # Log to wandb
            all_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self._log_wandb(all_metrics, step=epoch)
            self.metric_tracker.update(all_metrics, epoch)

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            # Visual validation
            if (epoch + 1) % self.config.visualize_interval == 0:
                self.visualize_predictions(num_samples=self.config.num_visualizations)

            # Early stopping
            if self.early_stopping.step(current_f1, epoch):
                print(f"\n⚠️ Early stopping triggered at epoch {epoch + 1}")
                print(f"   Best F1: {self.early_stopping.best_value*100:.2f}% at epoch {self.early_stopping.best_epoch + 1}")
                break

            # Print detailed metrics periodically
            if (epoch + 1) % 5 == 0:
                self.val_metrics.print_summary(f"EPOCH {epoch + 1} DETAILED METRICS")

        # Training complete
        total_time = time.time() - training_start
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best validation F1: {self.best_val_f1*100:.2f}%")
        print("=" * 70)

        # Final visualization
        self.visualize_predictions(
            num_samples=self.config.num_visualizations,
            save_path=self.config.visualization_dir / "final_predictions.png",
        )

        # Close wandb
        if self.config.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.finish()

        return {
            "history": history,
            "best_val_f1": self.best_val_f1,
            "total_time": total_time,
        }
