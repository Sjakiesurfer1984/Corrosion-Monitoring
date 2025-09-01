# run_inference.py

import torch
from PIL import Image
from tkinter import Tk, filedialog, messagebox
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pytorch_lightning as pl

# Import the PyTorch Lightning model class
from model.lightning_model import CorrosionSegmenter

# === Color map: class_id → hex color ===
CLASS_COLORS = {
    0: "#000000",   # background
    1: "#00FF00",   # mild corrosion (green)
    2: "#FFFF00",   # middle corrosion (yellow)
    3: "#FF0000",   # severe corrosion (red)
}

def load_model(model_path: str) -> pl.LightningModule:
    """
    Loads a PyTorch Lightning model from a checkpoint.
    """
    # Use PyTorch Lightning's built-in method to load the model and its state.
    # It handles all the key mapping and hyperparameter loading automatically.
    # The 'map_location' is set to CPU to ensure it works even without a GPU.
    model = CorrosionSegmenter.load_from_checkpoint(checkpoint_path=model_path, map_location="cpu")
    
    # Set the model to evaluation mode
    model.eval()
    
    # Return the underlying PyTorch model from the Lightning wrapper
    return model.model  # <-- Note this change

def decode_segmentation(mask: np.ndarray) -> Image.Image:
    """
    Turn a (HxW) array of class IDs into an RGB PIL image using CLASS_COLORS.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, hexcol in CLASS_COLORS.items():
        rgb = tuple(int(hexcol[i : i+2], 16) for i in (1, 3, 5))
        color_mask[mask == cls_id] = rgb
    return Image.fromarray(color_mask)

def run_inference(model: torch.nn.Module, image_path: str) -> None:
    # 1) Load & preprocess
    image = Image.open(image_path).convert("RGB")
    
    inference_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = inference_transforms(image).unsqueeze(0)

    # 2) Forward pass
    with torch.no_grad():
        # The model returned from load_model is now the raw PyTorch model,
        # so we can call it directly.
        out = model(input_tensor)["out"]    # [1, num_classes, H, W]
        pred_mask = torch.argmax(out.squeeze(), dim=0)  # [H, W]

    # 3) (Optional) Debug: check if anything but background is predicted
    print("Unique classes in prediction:", torch.unique(pred_mask))
    plt.figure(figsize=(4,4))
    plt.title("Raw mask (jet cmap)")
    plt.imshow(pred_mask.cpu(), cmap="jet")
    plt.colorbar()
    plt.show()

    # 4) Color-decode & display side by side
    mask_image = decode_segmentation(pred_mask.cpu().numpy())
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(mask_image)
    plt.title("Predicted Segmentation")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def pick_image_and_run(model_path: str):
    """
    GUI flow: load model, let user pick an image, then run inference.
    """
    model = load_model(model_path)

    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Select image to segment",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not file_path:
        messagebox.showinfo("Cancelled", "No image selected.")
        return

    run_inference(model, file_path)

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    messagebox.showinfo("Select Model", "Please choose your .pth or .ckpt model file.")
    model_file = filedialog.askopenfilename(
        title="Select model file",
        filetypes=[("PyTorch Model", "*.pth;*.ckpt")]
    )
    if not model_file:
        messagebox.showinfo("Cancelled", "No model selected.")
    else:
        pick_image_and_run(model_file)