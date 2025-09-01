# run_train.py

# This file is the main entry point for our training pipeline.
# It sets up all the components (data, model, callbacks) and starts the training.

# --- Core Library Imports ---
import argparse
from typing import Callable, List, Optional
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

# --- PyTorch Lightning Callbacks and Loggers ---
# Callbacks are like plugins for our training loop. They perform actions
# at specific moments, such as saving the best model or stopping early.
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Loggers are responsible for sending our training metrics (like loss and accuracy)
# to a visualization service so we can track our model's performance.
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# --- Project-Specific Imports ---
# We import our custom dataset class, which handles loading our images and masks.
from dataset.corrosion_dataset import (
    CorrosionDataset,
    JointTransforms,
    JointRandomHorizontalFlip,
    JointRandomRotation
)
# We import our PyTorch Lightning model defined in ligthning_model.py, which contains the full training logic.
from model.lightning_model import CorrosionSegmenter


def main() -> None:
    # --- 1. THE SINGLE CONTROL CENTER ---
    # This is the one and only place you need to change settings.
    # You can choose between using hard-coded values or command-line arguments,
    # and you can turn on debug mode for quick testing.

    # If this is 'True', the script will use the values defined below in this block.
    # If this is 'False', the script will expect you to provide arguments via the command line.
    USE_HARD_CODED_CONFIG: bool = True
    
    # If this is 'True', the training will use a tiny subset of the data
    # to run a quick test. This is useful for debugging your code.
    RUN_IN_DEBUG_MODE: bool = True

    # --- HARD-CODED VALUES ---
    # These values will be used if 'USE_HARD_CODED_CONFIG' is set to 'True'.
    model_name: str = "deeplabv3_resnet50"
    learning_rate: float = 1e-4
    batch_size: int = 4
    max_epochs: int = 5
    image_dir: str = r"data\processed\train\images_512"
    mask_dir: str = r"data\processed\train\mask_512"
    val_image_dir: str = r"data\processed\val\images_512"
    val_mask_dir: str = r"data\processed\val\mask_512"

    # --- 2. ARGUMENT SELECTION ---
    # This section gets the correct values based on the flag you set above.
    if USE_HARD_CODED_CONFIG:
        model_name_selected = model_name
        learning_rate_selected = learning_rate
        batch_size_selected = batch_size
        max_epochs_selected = max_epochs
        image_dir_selected = image_dir
        mask_dir_selected = mask_dir
        val_image_dir_selected = val_image_dir
        val_mask_dir_selected = val_mask_dir
        debug_mode_selected = RUN_IN_DEBUG_MODE
        
    else:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Train a corrosion segmentation model with PyTorch Lightning.")
        parser.add_argument("--model-name", type=str, default="deeplabv3_resnet50", help="Model architecture to use.")
        parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate.")
        parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training and validation.")
        parser.add_argument("--max-epochs", type=int, default=2, help="Maximum number of epochs to train.")
        parser.add_argument("--image-dir", type=str, required=True, help="Path to the training images.")
        parser.add_argument("--mask-dir", type=str, required=True, help="Path to the training masks.")
        parser.add_argument("--val-image-dir", type=str, required=True, help="Path to the validation images.")
        parser.add_argument("--val-mask-dir", type=str, required=True, help="Path to the validation masks.")
        parser.add_argument("--debug", action="store_true", help="Run in debug mode with a very small dataset.")
        
        args: argparse.Namespace = parser.parse_args()
        
        model_name_selected = args.model_name
        learning_rate_selected = args.learning_rate
        batch_size_selected = args.batch_size
        max_epochs_selected = args.max_epochs
        image_dir_selected = args.image_dir
        mask_dir_selected = args.mask_dir
        val_image_dir_selected = args.val_image_dir
        val_mask_dir_selected = args.val_mask_dir
        debug_mode_selected = args.debug

    # --- 3. DATA SETUP ---
    # This is where we define how to transform our images.
    train_augs: Callable = JointTransforms([
        JointRandomHorizontalFlip(p=0.5), # Randomly flips the image to prevent overfitting.
        JointRandomRotation(degrees=(-15, 15)) # Randomly rotates the image slightly.
    ])

    # We create a dataset object for our full training set.
    train_dataset_full: CorrosionDataset = CorrosionDataset(
        image_dir=image_dir_selected,
        mask_dir=mask_dir_selected,
        augmentations=train_augs
    )
    
    # We create a dataset object for our full validation set.
    val_dataset_full: CorrosionDataset = CorrosionDataset(
        image_dir=val_image_dir_selected,
        mask_dir=val_mask_dir_selected,
        augmentations=None # We do NOT apply random augmentations to the validation set.
    )
    
    # --- DEBUG MODE LOGIC ---
    # If the debug flag is 'True', we use a tiny subset of the dataset.
    if debug_mode_selected:
        print("Running in debug mode. Using a very small dataset.")
        # We use PyTorch's `Subset` class to create a small dataset.
        train_dataset: Subset = Subset(train_dataset_full, range(4))
        val_dataset: Subset = Subset(val_dataset_full, range(4))
    else:
        # If not in debug mode, we use the full datasets.
        train_dataset: CorrosionDataset = train_dataset_full
        val_dataset: CorrosionDataset = val_dataset_full
    
    # The DataLoader automatically handles batching and data loading.
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size_selected,
        shuffle=True, # We shuffle the training data to ensure the model sees a random order.
        num_workers=8, # This uses 8 separate processes to load data, which speeds things up.
        persistent_workers=True # Keeps the data-loading processes alive between epochs.
    )
    
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size_selected,
        shuffle=False, # We do NOT shuffle the validation data for consistent evaluation.
        num_workers=8,
        persistent_workers=True
    )

    # --- 4. MODEL INITIALIZATION ---
    # We create an instance of our PyTorch Lightning model.
    model: CorrosionSegmenter = CorrosionSegmenter(
        model_name=model_name_selected,
        num_classes=4, # Corresponds to our 4 classes (Background, Fair, Poor, Severe).
        learning_rate=learning_rate_selected
    )

    # --- 5. CALLBACKS & LOGGERS ---
    # These are the "plugins" for our training process.
    
    # The ModelCheckpoint saves the best model during training.
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        monitor="val/mIoU", # We monitor the mean IoU on the validation set.
        mode="max", # We want to save the model with the highest value for this metric.
        save_top_k=1, # We only save the single best model to save disk space.
        dirpath="_saved_models/", # This is where the model checkpoints will be stored.
        filename=f"{model_name_selected}-{{epoch:02d}}-{{val/mIoU:.4f}}" # Creates a descriptive filename.
    )
    
    # The EarlyStopping callback prevents overfitting by stopping training if
    # the model's performance on the validation set stops improving.
    early_stopping_callback: EarlyStopping = EarlyStopping(
        monitor="val/loss", # We watch the validation loss.
        patience=5 # We wait for 5 epochs without improvement before stopping.
    )
    
    # We set up the loggers to track our metrics.
    loggers: List[pl.loggers.WandbLogger | pl.loggers.TensorBoardLogger] = [
        # This logger sends our metrics to the Weights & Biases platform.
        WandbLogger(project="corrosion-segmentation-lightning"),
        # This logger saves our metrics to a local folder for TensorBoard.
        TensorBoardLogger(save_dir="tb_logs/", name="corrosion_model")
    ]
    
    # --- 6. TRAINER INITIALIZATION ---
    # The `Trainer` is the heart of PyTorch Lightning. It automates every part of the training loop.
    trainer: pl.Trainer = pl.Trainer(
        max_epochs=max_epochs_selected,
        accelerator="auto", # Automatically uses the best hardware (e.g., GPU).
        devices="auto", # Automatically uses all available devices of that type.
        logger=loggers, # We pass our list of loggers here.
        callbacks=[checkpoint_callback, early_stopping_callback] # We pass our list of callbacks here.
    )

    # --- 7. START TRAINING ---
    # This single line kicks off the entire process.
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


    # --- 8. POST-TRAINING EVALUATION (OPTIONAL) ---
    # After training, we can evaluate the best model on the validation set.
    print("Training complete. Best model saved at:", checkpoint_callback.best_model_path)
    print("Best validation mIoU:", checkpoint_callback.best_model_score.item())

    # --- 9. TEST SET EVALUATION (OPTIONAL) ---
    # If you have a separate test set, you can evaluate the best model on it here
    # by loading the best checkpoint and calling `trainer.test()`.
    # Uncomment and modify the following lines if you have a test set.
    # test_image_dir = "data/test/images"
    # test_mask_dir = "data/test/masks"
    # test_dataset = CorrosionDataset(image_dir=test_image_dir, mask_dir=test_mask_dir, augmentations=None)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size_selected, shuffle=False, num_workers=8, persistent_workers=True)
    # best_model = CorrosionSegmenter.load_from_checkpoint(checkpoint_callback.best_model_path)
    # trainer.test(best_model, dataloaders=test_loader)
    

if __name__ == "__main__":
    # This standard Python construct ensures that the `main()` function is called
    # when the script is run directly from the command line.
    main()