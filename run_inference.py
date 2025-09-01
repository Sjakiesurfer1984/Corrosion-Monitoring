import torch
from PIL import Image
from tkinter import Tk, filedialog, messagebox
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dataset.corrosion_dataset import get_transform
from model.model import get_semantic_segmentation_model

# === Color map: class_id → hex color ===
CLASS_COLORS = {
    0: "#000000",  # background
    1: "#00FF00",  # mild corrosion (green)
    2: "#FFFF00",  # middle corrosion (yellow)
    3: "#FF0000",  # severe corrosion (red)
}

def load_model(model_path: str, num_classes: int) -> torch.nn.Module:
    """
    Load the DeepLabV3 model with `num_classes` output channels.
    """
    model = get_semantic_segmentation_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def decode_segmentation(mask: np.ndarray) -> Image.Image:
    """
    Turn a (H×W) array of class IDs into an RGB PIL image using CLASS_COLORS.
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
    transform = get_transform(train=False)    # ← match your training pipeline
    input_tensor = transform(image).unsqueeze(0)

    # 2) Forward pass
    with torch.no_grad():
        out = model(input_tensor)["out"]     # [1, num_classes, H, W]
        pred_mask = torch.argmax(out.squeeze(), dim=0)  # [H, W]

    # 3) (Optional) Debug: check if anything but background is predicted
    print("Unique classes in prediction:", torch.unique(pred_mask))
    plt.figure(figsize=(4,4))
    plt.title("Raw mask (jet cmap)")
    plt.imshow(pred_mask.cpu(), cmap="jet")
    plt.colorbar()
    plt.show()

    # 4) Color‐decode & display side by side
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

def pick_image_and_run(model_path: str, num_classes: int = 4):
    """
    GUI flow: load model, let user pick an image, then run inference.
    """
    model = load_model(model_path, num_classes)

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
    # 1) Ask user to select the trained model
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    messagebox.showinfo("Select Model", "Please choose your .pth model file.")
    model_file = filedialog.askopenfilename(
        title="Select model file",
        filetypes=[("PyTorch Model", "*.pth")]
    )
    if not model_file:
        messagebox.showinfo("Cancelled", "No model selected.")
    else:
        pick_image_and_run(model_file, num_classes=4)




# # run_inference.py

# import torch
# from torchvision import transforms as T
# from PIL import Image
# from tkinter import Tk, filedialog
# from pathlib import Path
# import matplotlib.pyplot as plt

# from utils.visualize import show_image
# from model.model import get_semantic_segmentation_model  # if you want a cleaner wrapper
# from tkinter import Tk, filedialog, messagebox

# # Color map: index to RGB (hex-style)
# # === Color map: class_id -> hex color ===
# CLASS_COLORS = {
#     0: "#000000",  # background (black)
#     1: "#00FF00",  # mild corrosion = green
#     2: "#FFFF00",  # middle corrosion = yellow
#     3: "#FF0000",  # severe corrosion = red
# }
# # === Load model ===

# def load_model(model_path: str, num_classes: int) -> torch.nn.Module:
#     """
#     Loads the trained DeepLabV3 model with the correct head and aux classifier.
#     """
#     model = get_semantic_segmentation_model(num_classes=num_classes)
#     model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     model.eval()
#     return model



# # === Convert prediction to RGB mask ===
# def decode_segmentation(mask: torch.Tensor) -> Image.Image:
#     h, w = mask.shape
#     color_mask = torch.zeros(3, h, w, dtype=torch.uint8)
#     for idx, hex_color in CLASS_COLORS.items():
#         r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
#         color_mask[0][mask == idx] = r
#         color_mask[1][mask == idx] = g
#         color_mask[2][mask == idx] = b
#     return T.ToPILImage()(color_mask)


# # === Inference on a single image ===
# def run_inference(model: torch.nn.Module, image_path: str) -> None:
#     image = Image.open(image_path).convert("RGB")
#     transform = T.ToTensor()
#     input_tensor = transform(image).unsqueeze(0)

#     with torch.no_grad():
#         output = model(input_tensor)["out"]
#         pred_mask = torch.argmax(output.squeeze(), dim=0)

#     mask_image = decode_segmentation(pred_mask)

#     # Show original + prediction side by side
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     plt.title("Original")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(mask_image)
#     plt.title("Predicted Segmentation")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()


# # === GUI file picker ===
# def pick_image_and_run(model_path: str, num_classes: int = 4):
#     print("Starting image segmentation visualisation...\n")
#     print("Select an image\n")
#     root = Tk()
#     root.withdraw()
#     file_path = filedialog.askopenfilename(
#         title="Select image",
#         filetypes=[("Image files", "*.jpg *.jpeg *.png")]
#     )
#     if not file_path:
#         print("No image selected.")
#         return

#     model = load_model(model_path, num_classes)
#     run_inference(model, file_path)


# if __name__ == "__main__":
#     # GUI for selecting model file
#     root = Tk()
#     root.withdraw()
#     root.attributes('-topmost', True)

#     messagebox.showinfo("Set Model", "Please select your trained model (.pth) file.")

#     model_path = filedialog.askopenfilename(
#         title="Select Model File",
#         filetypes=[("PyTorch Model", "*.pth")]
#     )

#     if model_path:
#         num_classes = 4  # Adjust this number as required
#         pick_image_and_run(model_path, num_classes)
#     else:
#         messagebox.showinfo("Operation Cancelled", "No model file was selected.")

