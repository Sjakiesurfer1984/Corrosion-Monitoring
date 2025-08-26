# run_train.py

import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import List, Callable, Optional

# Callbacks are special objects that can perform actions at various stages of training.
# They are like plugins for our training loop.
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Loggers handle the process of sending our metrics (loss, mIoU, etc.) to a
# visualization service like Weights & Biases or TensorBoard.
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Import our data-handling classes and the new LightningModule
from dataset.corrosion_dataset import (
    CorrosionDataset,
    JointTransforms,
    JointRandomHorizontalFlip,
    JointRandomRotation
    )
from model.lightning_model import CorrosionSegmenter


def main() -> None:
    """
    The main function that sets up the data, model, and trainer, then starts the training.
    This function does not return anything, hence '-> None'.
    """
     # --- 1. ARGUMENT PARSING ---
    # We use Python's `argparse` library to create a command-line interface for our script.
    # This allows us to easily experiment with different settings (like learning rate or model architecture)
    # without having to change the code every time.
    parser = argparse.ArgumentParser(description="Train a corrosion segmentation model with PyTorch Lightning.")
    parser.add_argument("--model-name", type=str, default="deeplabv3_resnet50", help="Model architecture to use.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training and validation.")
    parser.add_argument("--max-epochs", type=int, default=2, help="Maximum number of epochs to train.")
    parser.add_argument("--image-dir", type=str, required=False, help="Path to the training images.")
    parser.add_argument("--mask-dir", type=str, required=False, help="Path to the training masks.")
    parser.add_argument("--val-image-dir", type=str, required=False, help="Path to the validation images.")
    parser.add_argument("--val-mask-dir", type=str, required=False, help="Path to the validation masks.")
    parser.add_argument("--hard-coded", action="store_true", help="Use hard-coded arguments instead of command-line args.")
    
    args: argparse.Namespace = parser.parse_args()

    # --- 2. DYNAMIC ARGUMENT SELECTION ---
    if args.hard_coded:
       # --- Hard-coded values for convenience ---
       model_name = "deeplabv3_resnet50"
       learning_rate = 1e-4
       batch_size = 4
       max_epochs = 2
       # Update these paths to match where your data is located on the VM!
       image_dir = "data/train/images"
       mask_dir = "data/train/masks"
       val_image_dir = "data/val/images"
       val_mask_dir = "data/val/masks"
    else:
       # --- Use parsed arguments from the command line ---
       model_name = args.model_name
       learning_rate = args.learning_rate
       batch_size = args.batch_size
       max_epochs = args.max_epochs
       image_dir = args.image_dir
       mask_dir = args.mask_dir
       val_image_dir = args.val_image_dir
       val_mask_dir = args.val_mask_dir

    # --- 3. DATA SETUP ---
    train_augs: Callable = JointTransforms([
       JointRandomHorizontalFlip(p=0.5),
       JointRandomRotation(degrees=(-15, 15))
    ])

    train_dataset: CorrosionDataset = CorrosionDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        augmentations=train_augs
    )
    
    val_dataset: CorrosionDataset = CorrosionDataset(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        augmentations=None
    )

    train_loader: DataLoader = DataLoader(
       train_dataset,
       batch_size=batch_size,
       shuffle=True,
       num_workers=8,
       persistent_workers=True
    )
    
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        persistent_workers=True # This line can speed up data loading on systems with enough RAM. It does htis by keeping worker processes alive between epochs.
    )

    # --- 4. MODEL INITIALIZATION ---
    model: CorrosionSegmenter = CorrosionSegmenter(
       model_name=model_name,
       num_classes=4, 
       learning_rate=learning_rate
    )

    # --- 5. CALLBACKS & LOGGERS ---
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        monitor="val/mIoU", 
        mode="max",    
        save_top_k=1,               dirpath="_saved_models/",
        filename=f"{model_name}-{{epoch:02d}}-{{val/mIoU:.4f}}"
    )
        
    early_stopping_callback: EarlyStopping = EarlyStopping(
        monitor="val/loss",
        patience=5
    )
        # Instantiate both loggers as a list
    loggers = [
       WandbLogger(project="corrosion-segmentation-lightning"),
       TensorBoardLogger(save_dir="tb_logs/", name="corrosion_model")
    ]
        # --- 6. TRAINER INITIALIZATION ---
    trainer: pl.Trainer = pl.Trainer(
       max_epochs=max_epochs,
       accelerator="auto",
       devices="auto",
       logger=loggers,
       callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # --- 7. START TRAINING ---
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()