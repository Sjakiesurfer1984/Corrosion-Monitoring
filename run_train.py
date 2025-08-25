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
from pytorch_lightning.loggers import WandbLogger

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
    parser.add_argument("--image-dir", type=str, required=True, help="Path to the training images.")
    parser.add_argument("--mask-dir", type=str, required=True, help="Path to the training masks.")
    parser.add_argument("--val-image-dir", type=str, required=True, help="Path to the validation images.")
    parser.add_argument("--val-mask-dir", type=str, required=True, help="Path to the validation masks.")
    
    # `parse_args()` returns an object where each argument is an attribute.
    args: argparse.Namespace = parser.parse_args()

    # --- 2. DATA SETUP ---
    # Define the augmentation pipeline for the training data. Augmentations like flipping and rotating
    # create new variations of our training images. This helps the model generalize better and prevents it
    # from simply memorizing the training set.
    train_augs: Callable = JointTransforms([
        JointRandomHorizontalFlip(p=0.5),
        JointRandomRotation(degrees=(-15, 15))
    ])

    # Create the training dataset instance, passing in our augmentation pipeline.
    train_dataset: CorrosionDataset = CorrosionDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        augmentations=train_augs
    )
    
    # IMPORTANT: The validation set should NOT have random augmentations. We need a consistent,
    # stable benchmark to measure the model's true performance at the end of each epoch.
    val_dataset: CorrosionDataset = CorrosionDataset(
        image_dir=args.val_image_dir,
        mask_dir=args.val_mask_dir,
        augmentations=None
    )

    # The DataLoader is a PyTorch utility that takes our Dataset and automatically handles
    # batching, shuffling, and multi-threaded data loading. `num_workers > 0` uses
    # separate processes to load data, which prevents the GPU from waiting for data to be ready.
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8, # This uses 8 worker processes to load data in parallel. A worker is a separate process that loads data.
        persistent_workers=True # This ensures that the worker processes stay alive after the first epoch,
        # which can speed up subsequent epochs by avoiding the overhead of starting new processes.
    )
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        persistent_workers=True 
    )

    # --- 3. MODEL INITIALIZATION ---
    # We create an instance of our `CorrosionSegmenter` LightningModule, passing in the hyperparameters
    # from our command-line arguments.
    model: CorrosionSegmenter = CorrosionSegmenter(
        model_name=args.model_name,
        num_classes=4,  # Background + 3 corrosion types
        learning_rate=args.learning_rate
    )

    # --- 4. CALLBACKS & LOGGERS ---
    
    # The ModelCheckpoint callback is responsible for saving our model during training.
    # It will monitor a specific metric and save the best-performing version of the model.

    # TO DO: We have to save every epoch, because the best model might be at the end. Hence, a crash before reaching the best performance would lose all the effort and time spent.
    # However, this takes a lot of disk space. A better solution would be to save only the best model so far, and delete the previous best model.
    # This can be done by setting `save_top_k=1` in the ModelCheckpoint callback.
    # Hence, this checkpoint callback is setup correctly. If the programme were to crash mid-training, we would only lose the progress since the last best model was saved.
    # and we could resume training from that point.
    # How? By adding `resume_from_checkpoint="path/to/checkpoint.ckpt"` to the `pl.Trainer()` initialization below.
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        monitor="val/mIoU",   # The metric to watch.
        mode="max",           # 'max' means we want the highest mIoU, as higher is better.
        save_top_k=1,         # Save only the single best model checkpoint.
        dirpath="_saved_models/", # Directory where the checkpoints will be saved.
        filename=f"{args.model_name}-{{epoch:02d}}-{{val/mIoU:.4f}}" # A descriptive filename.
    )
    
    # The EarlyStopping callback can stop the training process automatically if the model's
    # performance on the validation set stops improving, which helps prevent overfitting.
    early_stopping_callback: EarlyStopping = EarlyStopping(
        monitor="val/loss", # The metric to watch for improvement.
        patience=5          # Number of epochs to wait for improvement before stopping.
    )
    
    # The logger handles sending all the metrics we log with `self.log()` in our LightningModule
    # to an external service. Here, we're using Weights & Biases.
    wandb_logger: WandbLogger = WandbLogger(project="corrosion-segmentation-lightning")

    # --- 5. TRAINER INITIALIZATION ---
    # The `Trainer` is the heart of PyTorch Lightning. It automates every aspect of the training loop,
    # including device management (moving data to the GPU), calling the optimizer, backpropagation,
    # and executing the callbacks and loggers at the correct times.
    trainer: pl.Trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto", # Automatically selects the best hardware (GPU, TPU, etc.).
        devices="auto",     # Automatically uses all available devices of the selected type.
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # --- 6. START TRAINING ---
    # This single line kicks off the entire training and validation process. The Trainer
    # will now run for `max_epochs`, calling the appropriate methods in our `CorrosionSegmenter`
    # and using the dataloaders, callbacks, and logger we provided.
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    # This standard Python construct ensures that the `main()` function is called only
    # when this script is executed directly from the command line.
    main()