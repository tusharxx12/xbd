"""
Research-Quality Metrics Engine for Damage Detection

Contains:
- DamageMetrics: Confusion matrix-based metrics calculator
- Per-class and aggregated metrics (F1, IoU)
- IGNORE_INDEX handling throughout
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .config import TrainingConfig, DAMAGE_CLASSES


class DamageMetrics:
    """
    Research-quality metrics calculator for multi-class segmentation.

    Maintains a confusion matrix and computes:
    - Per-class Precision, Recall, F1-Score
    - Macro/Weighted F1-Score
    - Per-class IoU (Intersection over Union)
    - Mean IoU (mIoU)

    CRITICAL: Strictly ignores IGNORE_INDEX (255) in all calculations.
    """

    def __init__(
        self,
        num_classes: int = 5,
        ignore_index: int = 255,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        Args:
            num_classes: Number of classes (including background)
            ignore_index: Label to ignore in calculations
            class_names: Optional mapping from class ID to name
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or DAMAGE_CLASSES

        # Initialize confusion matrix
        self.reset()

    def reset(self):
        """Reset confusion matrix for new epoch/evaluation"""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64,
        )
        self._sample_count = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Update confusion matrix with batch predictions.

        Args:
            predictions: (B, C, H, W) logits or (B, H, W) class predictions
            targets: (B, H, W) ground truth labels
        """
        # Handle logits vs class predictions
        if predictions.dim() == 4:
            preds = predictions.argmax(dim=1)  # (B, H, W)
        else:
            preds = predictions

        # Move to CPU and convert to numpy
        preds = preds.detach().cpu().numpy().astype(np.int64)
        targets = targets.detach().cpu().numpy().astype(np.int64)

        # Flatten
        preds = preds.flatten()
        targets = targets.flatten()

        # Create mask to ignore specified index
        valid_mask = targets != self.ignore_index

        # Filter out ignored pixels
        preds = preds[valid_mask]
        targets = targets[valid_mask]

        # Clip predictions to valid range
        preds = np.clip(preds, 0, self.num_classes - 1)

        # Update confusion matrix
        for t, p in zip(targets, preds):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1

        self._sample_count += 1

    def compute_per_class_metrics(self) -> Dict[str, np.ndarray]:
        """
        Compute per-class precision, recall, and F1-score.

        Returns:
            Dictionary with:
            - precision: (num_classes,) array
            - recall: (num_classes,) array
            - f1: (num_classes,) array
        """
        # True Positives: diagonal elements
        tp = np.diag(self.confusion_matrix)

        # False Positives: column sum - diagonal
        fp = self.confusion_matrix.sum(axis=0) - tp

        # False Negatives: row sum - diagonal
        fn = self.confusion_matrix.sum(axis=1) - tp

        # Precision = TP / (TP + FP)
        precision = np.divide(
            tp,
            tp + fp,
            out=np.zeros_like(tp, dtype=np.float64),
            where=(tp + fp) != 0,
        )

        # Recall = TP / (TP + FN)
        recall = np.divide(
            tp,
            tp + fn,
            out=np.zeros_like(tp, dtype=np.float64),
            where=(tp + fn) != 0,
        )

        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def compute_per_class_iou(self) -> np.ndarray:
        """
        Compute per-class IoU (Intersection over Union).

        IoU = TP / (TP + FP + FN)

        Returns:
            (num_classes,) array of IoU values
        """
        # True Positives: diagonal elements
        tp = np.diag(self.confusion_matrix)

        # False Positives: column sum - diagonal
        fp = self.confusion_matrix.sum(axis=0) - tp

        # False Negatives: row sum - diagonal
        fn = self.confusion_matrix.sum(axis=1) - tp

        # IoU = TP / (TP + FP + FN)
        iou = np.divide(
            tp,
            tp + fp + fn,
            out=np.zeros_like(tp, dtype=np.float64),
            where=(tp + fp + fn) != 0,
        )

        return iou

    def compute_macro_f1(self, exclude_background: bool = True) -> float:
        """
        Compute macro-averaged F1-score.

        Args:
            exclude_background: If True, exclude class 0 from calculation

        Returns:
            Macro F1-score
        """
        per_class = self.compute_per_class_metrics()
        f1_scores = per_class["f1"]

        if exclude_background:
            # Exclude background class (index 0)
            f1_scores = f1_scores[1:]

        # Filter out classes with no support
        valid_mask = f1_scores > 0
        if valid_mask.sum() == 0:
            return 0.0

        return float(f1_scores[valid_mask].mean())

    def compute_weighted_f1(self, exclude_background: bool = True) -> float:
        """
        Compute weighted F1-score (weighted by class support).

        Args:
            exclude_background: If True, exclude class 0 from calculation

        Returns:
            Weighted F1-score
        """
        per_class = self.compute_per_class_metrics()
        f1_scores = per_class["f1"]

        # Class support (number of true samples per class)
        support = self.confusion_matrix.sum(axis=1)

        if exclude_background:
            f1_scores = f1_scores[1:]
            support = support[1:]

        # Weighted average
        total_support = support.sum()
        if total_support == 0:
            return 0.0

        weighted_f1 = (f1_scores * support).sum() / total_support
        return float(weighted_f1)

    def compute_miou(self, exclude_background: bool = True) -> float:
        """
        Compute mean IoU across classes.

        Args:
            exclude_background: If True, exclude class 0 from calculation

        Returns:
            Mean IoU
        """
        iou = self.compute_per_class_iou()

        if exclude_background:
            iou = iou[1:]

        # Only average over classes with support
        valid_mask = iou > 0
        if valid_mask.sum() == 0:
            return 0.0

        return float(iou[valid_mask].mean())

    def compute_overall_accuracy(self) -> float:
        """
        Compute overall pixel accuracy.

        Returns:
            Accuracy = correct predictions / total predictions
        """
        total = self.confusion_matrix.sum()
        if total == 0:
            return 0.0

        correct = np.diag(self.confusion_matrix).sum()
        return float(correct / total)

    def get_all_metrics(self, exclude_background: bool = True) -> Dict[str, float]:
        """
        Compute all metrics at once.

        Args:
            exclude_background: If True, exclude class 0 from aggregated metrics

        Returns:
            Dictionary with all computed metrics
        """
        per_class = self.compute_per_class_metrics()
        iou = self.compute_per_class_iou()

        metrics = {
            # Aggregated metrics
            "accuracy": self.compute_overall_accuracy(),
            "f1_macro": self.compute_macro_f1(exclude_background),
            "f1_weighted": self.compute_weighted_f1(exclude_background),
            "miou": self.compute_miou(exclude_background),
        }

        # Per-class metrics
        for i in range(self.num_classes):
            class_name = self.class_names.get(i, f"class_{i}")
            class_name = class_name.lower().replace(" ", "_")

            metrics[f"f1_{class_name}"] = float(per_class["f1"][i])
            metrics[f"iou_{class_name}"] = float(iou[i])
            metrics[f"precision_{class_name}"] = float(per_class["precision"][i])
            metrics[f"recall_{class_name}"] = float(per_class["recall"][i])

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the raw confusion matrix"""
        return self.confusion_matrix.copy()

    def print_summary(self, title: str = "METRICS SUMMARY"):
        """Print a formatted summary of all metrics"""
        metrics = self.get_all_metrics()
        per_class = self.compute_per_class_metrics()
        iou = self.compute_per_class_iou()

        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")

        print(f"\n📊 AGGREGATED METRICS:")
        print(f"  Accuracy:       {metrics['accuracy']*100:.2f}%")
        print(f"  Macro F1:       {metrics['f1_macro']*100:.2f}%")
        print(f"  Weighted F1:    {metrics['f1_weighted']*100:.2f}%")
        print(f"  Mean IoU:       {metrics['miou']*100:.2f}%")

        print(f"\n📋 PER-CLASS METRICS:")
        print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}")
        print(f"  {'-'*55}")

        for i in range(self.num_classes):
            class_name = self.class_names.get(i, f"Class {i}")
            print(
                f"  {class_name:<15} "
                f"{per_class['precision'][i]*100:>9.2f}% "
                f"{per_class['recall'][i]*100:>9.2f}% "
                f"{per_class['f1'][i]*100:>9.2f}% "
                f"{iou[i]*100:>9.2f}%"
            )

        print(f"\n📈 CONFUSION MATRIX:")
        print(f"  (Rows: True, Columns: Predicted)")
        print(f"  {'':<15}", end="")
        for i in range(self.num_classes):
            print(f"{self.class_names.get(i, f'C{i}')[:6]:>8}", end="")
        print()

        for i in range(self.num_classes):
            print(f"  {self.class_names.get(i, f'C{i}'):<15}", end="")
            for j in range(self.num_classes):
                print(f"{self.confusion_matrix[i, j]:>8}", end="")
            print()

        print(f"{'='*70}\n")


class MetricTracker:
    """
    Tracks metrics across training epochs for analysis and early stopping.
    """

    def __init__(self):
        self.history = defaultdict(list)
        self.best_values = {}
        self.best_epoch = {}

    def update(self, metrics: Dict[str, float], epoch: int):
        """Record metrics for an epoch"""
        for key, value in metrics.items():
            self.history[key].append((epoch, value))

            # Track best values
            if key not in self.best_values:
                self.best_values[key] = value
                self.best_epoch[key] = epoch
            else:
                # Assume higher is better for F1/IoU, lower for loss
                if "loss" in key.lower():
                    if value < self.best_values[key]:
                        self.best_values[key] = value
                        self.best_epoch[key] = epoch
                else:
                    if value > self.best_values[key]:
                        self.best_values[key] = value
                        self.best_epoch[key] = epoch

    def get_best(self, metric_name: str) -> Tuple[float, int]:
        """Get best value and epoch for a metric"""
        return self.best_values.get(metric_name, 0.0), self.best_epoch.get(metric_name, 0)

    def get_history(self, metric_name: str) -> List[Tuple[int, float]]:
        """Get full history for a metric"""
        return self.history.get(metric_name, [])
