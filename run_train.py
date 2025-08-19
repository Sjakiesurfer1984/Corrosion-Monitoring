#run_train.py
# This file is the "conductor". It's responsible for preparing all the pieces (data, model, optimizer) 
# and then handing them over to the Trainer to run the show.
# run_train.py
import torch
from torch.utils.data import DataLoader
import argparse  # We import the argparse library to handle command-line arguments.
from typing import List

# Import our refactored, flexible components
# This correctly imports from the 'model.py' file inside the 'model' folder
from model.model import get_model
from dataset.corrosion_dataset import (
    CorrosionDataset,
    JointTransforms,
    JointRandomHorizontalFlip,
    JointRandomRotation
)
from train.trainer import Trainer

# --- CONFIGURATION VIA ARGUMENTS ---
def get_args() -> argparse.Namespace:
    """
    This function sets up and parses command-line arguments.
    It's a clean way to manage all the settings for a training run.
    """
    # Create a parser object
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")

    # --- Experiment Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of samples per batch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes including background.")

    # --- Data Paths (Set your default paths ONCE here) ---
    parser.add_argument("--train_img_dir", type=str, default="data/Train/images_512", help="Path to training images.")
    parser.add_argument("--train_mask_dir", type=str, default="data/Train/mask_512", help="Path to training masks.")
    parser.add_argument("--val_img_dir", type=str, default="data/val/images_512", help="Path to validation images.")
    parser.add_argument("--val_mask_dir", type=str, default="data/val/mask_512", help="Path to validation masks.")
    
    # --- System Configuration ---
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (cuda or cpu).")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker processes for data loading.")

    # The `parse_args()` method reads the command-line inputs and returns an object
    # containing all the values.
    return parser.parse_args()

def main() -> None:
    """The main function that orchestrates the entire training process."""
    args = get_args()

    # Set the device. It's crucial to send your model and data to the GPU if you have one.
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Define the number of classes. It's 4 because we have 3 corrosion types + 1 background class.
    NUM_CLASSES: int = 4

    # 1. --- DATASETS and DATALOADERS ---

    # Define the augmentation pipeline for the training set. We can chain multiple
    # transformations together using our JointTransforms helper class.
    train_augs = JointTransforms([
        JointRandomHorizontalFlip(p=0.5),
        JointRandomRotation(degrees=(-15, 15))
        # You could add a JointRandomCrop class here for more augmentation!
    ])

    # Create the training dataset, passing in our augmentation pipeline.
    train_dataset = CorrosionDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        augmentations=train_augs
    )

    # Create the validation dataset.
    # IMPORTANT: The validation set should NOT have random augmentations.
    # We need a stable, consistent benchmark to measure our model's progress.
    val_dataset = CorrosionDataset(
        image_dir=args.val_image_dir,
        mask_dir=args.val_mask_dir,
        augmentations=None  # No augmentations!
    )

    # The DataLoader is a PyTorch utility that takes our Dataset and automatically
    # handles batching, shuffling, and multi-threaded data loading for us.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. --- MODEL ---
    # We call our model factory to get the desired model architecture.
    model: torch.nn.Module = get_model(model_name=args.model_name, num_classes=NUM_CLASSES).to(DEVICE)

    # 3. --- LOSS, OPTIMIZER, SCHEDULER ---
    # The loss function measures how wrong the model's predictions are.
    # CrossEntropyLoss is the standard for multi-class segmentation.
    criterion = torch.nn.CrossEntropyLoss()

    # The optimizer's job is to update the model's weights to reduce the loss.
    # Adam is a very popular and effective general-purpose optimizer.
    # We could also try SGD (stochastic gradient descent)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # A learning rate scheduler can adjust the learning rate during training
    # (e.g., reduce it if the validation loss stops improving). This can help find a better solution.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    # 4. --- INITIALIZE and RUN THE TRAINER ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        scheduler=scheduler
    )

    # We can create a unique output directory for each run based on the model name.
    output_dir = f"_saved_models/{args.model_name}"
    trainer.run(num_epochs=args.epochs, output_dir=output_dir)

if __name__ == "__main__":
    # This standard Python construct ensures that main() is called only when
    # this script is executed directly (not when it's imported by another script).
    main()