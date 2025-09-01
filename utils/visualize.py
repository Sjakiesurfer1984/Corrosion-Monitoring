# visualize.py

# This visualize.py does 3 things:

    # Draws predicted masks, boxes, and labels

    # Compares prediction vs truth

    # Lets you see where your model performs well or fails — crucial for debugging

# Import the libraries needed for visualization and image processing
import matplotlib.pyplot as plt              # For plotting images and graphs
import numpy as np                           # For array manipulation (if needed later)
import torch                                 # Core PyTorch library
import cv2                                   # OpenCV (not used below, but often useful)
import random                                # For random colors (if extended later)

# torchvision provides utilities to draw bounding boxes and segmentation masks
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image  # Converts tensors to PIL images


def draw_predictions(image: torch.Tensor,
                     masks: torch.Tensor,
                     boxes: torch.Tensor,
                     labels: torch.Tensor,
                     scores: torch.Tensor = None,
                     score_threshold: float = 0.5) -> torch.Tensor:
    """
    Draws segmentation masks, bounding boxes, and class labels on an image.

    This is typically used to visualize what the model predicted.

    Args:
        image (torch.Tensor): The input image as a tensor of shape [3, H, W]. Values should be 0-255.
        masks (torch.Tensor): Predicted masks of shape [N, H, W], where N = number of detected objects.
        boxes (torch.Tensor): Bounding boxes of shape [N, 4] with coordinates (x1, y1, x2, y2).
        labels (torch.Tensor): Class labels (integers) for each object.
        scores (torch.Tensor): Optional confidence scores per prediction (between 0 and 1).
        score_threshold (float): Only predictions with score >= threshold are shown.

    Returns:
        torch.Tensor: The annotated image as a tensor that can be displayed or saved.
    """

    # If confidence scores are provided, filter out weak predictions
    if scores is not None:
        keep = scores >= score_threshold
        masks = masks[keep]        # Keep only high-confidence masks
        boxes = boxes[keep]        # Keep corresponding boxes
        labels = labels[keep]      # Keep corresponding labels

    # Ensure the input image is on the CPU and in unsigned 8-bit format (0–255 range)
    image = image.clone().detach().cpu().to(torch.uint8)

    # Draw segmentation masks onto the image
    # masks.bool() ensures that only binary (True/False) masks are used
    colored = draw_segmentation_masks(image,
                                      masks.bool(),
                                      alpha=0.5)  # alpha controls transparency of masks

    # Draw bounding boxes and overlay class labels as text
    colored = draw_bounding_boxes(colored,
                                  boxes,
                                  labels=[str(l.item()) for l in labels],
                                  colors="red")  # Use red boxes for all detections

    # Convert tensor image to PIL format for plotting or saving
    return to_pil_image(colored)


def show_image(image_pil, title: str = ""):
    """
    Displays a PIL image using matplotlib.

    Args:
        image_pil: A PIL-format image (e.g., from to_pil_image)
        title (str): Optional title to show above the image
    """
    plt.figure(figsize=(8, 8))     # Set figure size
    plt.imshow(image_pil)          # Show the image
    plt.title(title)               # Add title
    plt.axis("off")                # Hide axis ticks and labels
    plt.tight_layout()             # Clean up layout
    plt.show()                     # Display the image window


def visualize_sample(model, dataset, idx: int, device="cpu", score_threshold=0.5):
    """
    Displays the predicted and ground truth segmentation for one image from the dataset.

    Args:
        model: The trained Mask R-CNN model.
        dataset: The dataset object that returns (image, target) pairs.
        idx (int): Index of the image to visualize.
        device (str): Either 'cpu' or 'cuda'. Use same device as model.
        score_threshold (float): Minimum confidence required to display predictions.
    """
    # Ensure the model is in evaluation mode (not training)
    model.eval()

    # Get the image and ground truth target from the dataset
    image, target = dataset[idx]
    image = image.to(device)  # Move image to same device as model

    # Run the model on the image (forward pass)
    with torch.no_grad():  # No need to compute gradients during inference
        prediction = model([image])[0]  # Model returns a list of predictions, one per image

    # Draw predictions (from model output)
    pred_viz = draw_predictions(
        image * 255,                             # Rescale from 0-1 to 0-255
        prediction['masks'].squeeze(1),         # Shape [N, 1, H, W] -> [N, H, W]
        prediction['boxes'],
        prediction['labels'],
        prediction.get('scores'),
        score_threshold=score_threshold
    )

    # Draw ground truth (from dataset annotations)
    gt_viz = draw_predictions(
        image * 255,                             # Same rescaling
        target['masks'],
        target['boxes'],
        target['labels']
    )

    # Show both ground truth and prediction side by side
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(gt_viz)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_viz)
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
