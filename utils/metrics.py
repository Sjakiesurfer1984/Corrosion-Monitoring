# Import PyTorch for handling tensors and GPU operations
import torch
# Type hints for function arguments and returns
from typing import List, Tuple
# Import NumPy for numerical operations (not strictly required here, but often helpful)
import numpy as np
import sklearn

# Import useful classification evaluation tools from scikit-learn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

def calculate_mask_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
    """
    Calculates Intersection-over-Union (IoU) between a predicted mask and the ground truth mask.

    IoU is defined as:
        Area of Overlap / Area of Union

    This tells us how closely the predicted shape (the mask) overlaps with the real (ground truth) shape.

    Args:
        pred_mask (torch.Tensor): The predicted binary mask of shape [H, W].
        true_mask (torch.Tensor): The corresponding ground truth binary mask of shape [H, W].

    Returns:
        float: IoU value between 0.0 and 1.0, where 1.0 means perfect overlap.
    """

    # Convert both masks to boolean (True where mask is non-zero)
    pred = pred_mask.bool()
    truth = true_mask.bool()

    # Compute the intersection: where both prediction and ground truth are True
    intersection = torch.logical_and(pred, truth).sum().item()

    # Compute the union: any pixel that's True in either prediction or truth
    union = torch.logical_or(pred, truth).sum().item()

    # Avoid division by zero: if union is 0, return 1 if intersection is also 0 (i.e., both empty), else 0
    if union == 0:
        return float(intersection == 0)

    return intersection / union


def evaluate_detection_labels(
    true_labels: List[int],
    pred_labels: List[int],
    label_names: List[str] = None
) -> None:
    """
    Evaluates how well the model predicted the correct classes for each detected instance.

    This version includes precision, recall, F1-score, and UAR (Unweighted Average Recall).

    Args:
        true_labels (List[int]): The actual class labels (from ground truth).
        pred_labels (List[int]): The predicted class labels (from the model).
        label_names (List[str], optional): Human-readable names for each class.

    Returns:
        None (prints metrics to console)
    """

    print("Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=label_names, digits=3))

    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))

    # === Aggregate Metrics ===
    # Precision: Of all predictions made, how many were correct?
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    print(f"Weighted Precision: {precision:.3f}")

    # Recall: Of all actual class instances, how many did we correctly predict?
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    print(f"Weighted Recall:    {recall:.3f}")

    # F1: Harmonic mean of precision and recall
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    print(f"Weighted F1 Score:  {f1:.3f}")

    # UAR (Unweighted Average Recall): Treats all classes equally
    per_class_recalls = recall_score(true_labels, pred_labels, average=None, zero_division=0)
    uar = per_class_recalls.mean()
    print(f"Unweighted Avg Recall (UAR): {uar:.3f}")



def compute_segmentation_metrics(
    preds: torch.Tensor,  # shape [B, H, W]
    targets: torch.Tensor  # shape [B, H, W]
) -> tuple[list[int], list[int], list[float]]:
    """
    Compute per-pixel accuracy and IoU for semantic segmentation tasks.
    Returns flattened labels and IoU scores per sample.
    """
    all_true = []
    all_pred = []
    all_iou = []

    num_classes = torch.max(torch.cat([preds, targets])) + 1

    for pred, target in zip(preds, targets):
        pred_flat = pred.cpu().numpy().flatten()
        target_flat = target.cpu().numpy().flatten()

        all_true.extend(target_flat)
        all_pred.extend(pred_flat)

        # Simple per-class IoU calculation
        ious = []
        for cls in range(1, num_classes):  # Skip background = 0
            pred_mask = pred == cls
            target_mask = target == cls
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)

        all_iou.append(np.mean(ious) if ious else 0.0)

    return all_true, all_pred, all_iou




def segmentation_metrics(
    iou_scores: List[float],
    iou_thresholds: List[float] = [0.5, 0.75]
) -> None:
    """
    Evaluates segmentation performance using IoU-based metrics.

    This is complementary to label classification metrics. It focuses on *how well*
    the shapes of predicted masks match the ground truth.

    Args:
        iou_scores (List[float]): List of IoU values for all matched mask pairs.
        iou_thresholds (List[float]): Thresholds to report IoU-based accuracy at.

    Returns:
        None — prints results to console.
    """

    # === Step 1: Mean IoU (across all matched instances) ===
    if not iou_scores:
        print("No matched masks — cannot compute IoU metrics.")
        return

    miou = sum(iou_scores) / len(iou_scores)
    print(f"Mean IoU (mIoU) across matched masks: {miou:.3f}")

    # === Step 2: Report % of masks with IoU ≥ threshold (like COCO does) ===
    for thresh in iou_thresholds:
        above = [iou for iou in iou_scores if iou >= thresh]
        acc = len(above) / len(iou_scores)
        print(f"IoU ≥ {thresh:.2f}: {acc * 100:.1f}% of masks")



def compute_classification_metrics(
    true_labels: List[int],
    pred_labels: List[int]
) -> dict:
    """
    Computes core classification metrics and returns them in a loggable format.

    Args:
        true_labels (List[int]): Ground truth class IDs.
        pred_labels (List[int]): Predicted class IDs.

    Returns:
        dict: A dictionary containing F1, recall, accuracy, precision, and UAR.
    """
    metrics = {}

    # Avoid failure in case of no predictions
    if len(pred_labels) == 0 or len(true_labels) == 0:
        return {
            "f1": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "uar": 0.0,
        }

    metrics["f1"] = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    metrics["recall"] = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    metrics["accuracy"] = accuracy_score(true_labels, pred_labels)
    metrics["precision"] = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)

    per_class_recalls = recall_score(true_labels, pred_labels, average=None, zero_division=0)
    metrics["uar"] = per_class_recalls.mean()

    return metrics



