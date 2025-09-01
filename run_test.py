# run_test.py

import argparse
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tkinter import Tk, filedialog, messagebox
from typing import Optional

# Import your custom dataset and the PyTorch Lightning model
from dataset.corrosion_dataset import CorrosionDataset
from model.lightning_model import CorrosionSegmenter

def get_best_model_path() -> Optional[Path]:
    """
    Tries to find the best model checkpoint automatically in the _saved_models directory.
    """
    model_dir = Path("_saved_models")
    if not model_dir.exists():
        return None
        
    # Find the newest file with a .ckpt extension
    try:
        # Use glob to find all checkpoint files
        checkpoints = sorted(list(model_dir.glob("*.ckpt")), key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0] if checkpoints else None
    except IndexError:
        return None

def main():
    # Hardcoded paths to the test data
    test_image_dir = r"data\processed\test\images_512" # still needs to be created, this folder. should contain the 512x512 test images
    test_mask_dir = r"data\processed\test\mask_512"    # still needs to be created, this folder. should contain the 512x512 test masks
    batch_size = 4
    num_workers = 8

    # --- 1. FIND OR SELECT THE MODEL ---
    model_path = get_best_model_path()
    
    if not model_path:
        # If no model is found, fall back to the GUI
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showinfo("Select Model", "No models found automatically. Please choose a .ckpt model file.")
        file_path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("PyTorch Checkpoint", "*.ckpt")]
        )
        if not file_path:
            messagebox.showinfo("Cancelled", "No model selected. Exiting.")
            return
        model_path = Path(file_path)

    # --- 2. LOAD THE BEST MODEL ---
    print(f"Loading model from checkpoint: {model_path}")
    # PyTorch Lightning handles all the hyperparameter loading and state_dict mapping.
    model = CorrosionSegmenter.load_from_checkpoint(checkpoint_path=model_path, map_location=torch.device('cpu'))
    model.eval() # Ensure the model is in evaluation mode.

    # --- 3. SET UP THE TEST DATA ---
    print("Setting up test data loaders...")
    test_dataset = CorrosionDataset(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        augmentations=None # We never use random augmentations on test data.
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # We don't shuffle test data for consistent evaluation.
        num_workers=num_workers,
        persistent_workers=True
    )
    
    # --- 4. RUN THE EVALUATION ---
    print("Starting final evaluation on the test set...")
    trainer = pl.Trainer(accelerator="auto", devices="auto")
    
    # The `trainer.test()` method will automatically call the `test_step()`
    # and `on_test_epoch_end()` methods in your LightningModule.
    trainer.test(model, dataloaders=test_loader)
    
if __name__ == "__main__":
    main()