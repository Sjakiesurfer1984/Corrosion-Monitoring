# train/trainer.py

import os
import torch
import torch.nn as nn
import wandb
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from utils.logger import logger # Assuming logger is in the same directory or utils
# Make sure to import your metric calculation functions
from utils.metrics import compute_segmentation_metrics, compute_classification_metrics
from typing import Tuple, Dict
import numpy as np
import json
# Keep your train_one_epoch and evaluate functions as they are,
# but they will become methods of the Trainer class.

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, device, scheduler=None):
        """
        Initializes the Trainer object.
        Args:
            model (nn.Module): The neural network model to train.
            optimizer (torch.optim.Optimizer): The optimizer.
            criterion (nn.Module): The loss function.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            device (torch.device): The device to run on (e.g., 'cuda' or 'cpu').
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scheduler = scheduler
        self.best_loss = float("inf")
        self.model_save_dir = None # Will be set during training run

    def _train_one_epoch(self, epoch):
        """Private method for training one epoch."""
        self.model.train()
        total_loss = 0.0
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            imgs = torch.stack([img.to(self.device) for img in images])
            targs = torch.stack([m.to(self.device) for m in masks])

            self.optimizer.zero_grad()
            outputs = self.model(imgs)["out"]
            loss = self.criterion(outputs, targs)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            # print the loss every 5 epochs
            if batch_idx % 5 == 0:
                print(f"[Epoch {epoch+1}] Batch {batch_idx}/{len(self.train_loader)} — loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")
        return avg_loss

    # TO DO: define this method. 
    # Add metrics: Intersection over Union (IoU), Recall, F1, Precision, Accuracy. 
    @torch.no_grad()
    def _validate_one_epoch(self) -> Tuple[float, float, Dict[str, float]]:
        """
        Performs one full evaluation pass on the validation dataset.
        This now calculates and returns loss, mean IoU, and a dictionary of classification metrics.

        Returns:
            Tuple[float, float, Dict[str, float]]: A tuple containing:
                - The average validation loss.
                - The mean Intersection over Union (mIoU).
                - A dictionary of classification metrics (F1, recall, etc.).
        """
        # Set the model to evaluation mode. This is crucial as it disables layers like
        # Dropout, which should only be active during training.
        self.model.eval()

        total_loss = 0.0
        # These lists will collect the results from all batches so we can
        # calculate metrics over the entire validation set at the end.
        all_true_labels, all_pred_labels, all_iou_scores = [], [], []

        # Loop through each batch in the validation data loader.
        for images, masks in self.val_loader:
            # Send the data to the correct device (CPU or GPU).
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Get the model's raw output (logits).
            outputs = self.model(images)["out"]

            # Calculate the loss for this batch and add it to our running total.
            loss = self.criterion(outputs, masks)
            total_loss += loss.item()

            # Convert the raw logits into final class predictions (0, 1, 2, or 3 for each pixel).
            preds = torch.argmax(outputs, dim=1)

            # Use our helper function from metrics.py to get flattened labels and IoU scores for the batch.
            true, pred, iou = compute_segmentation_metrics(preds, masks)
            # We use the extend method because
            all_true_labels.extend(true)
            all_pred_labels.extend(pred)
            all_iou_scores.extend(iou)

        # --- METRIC CALCULATION ACROSS THE ENTIRE DATASET ---
        # 1. Calculate the final average loss.
        avg_loss = total_loss / len(self.val_loader)

        # 2. Calculate Mean IoU. We filter out any NaN values just in case.
        miou = np.nanmean(all_iou_scores) if all_iou_scores else 0.0

        # 3. Calculate all the classification metrics using the aggregated predictions.
        cls_metrics = compute_classification_metrics(all_true_labels, all_pred_labels)

        print(f"Validation Loss: {avg_loss:.4f} | Mean IoU: {miou:.4f} | F1-Score: {cls_metrics.get('f1', 0):.4f}")

        return avg_loss, miou, cls_metrics


    def _setup_run_save_directory(self, output_dir):
        """Sets up the output directory for saving models."""
        project_root = Path(__file__).resolve().parent.parent
        self.model_save_dir = project_root / output_dir / f"seg_{datetime.now():%Y%m%d_%H%M%S}"
        os.makedirs(self.model_save_dir, exist_ok=True)


    def run(self, num_epochs: int, output_dir: str = "_saved_models") -> None:
        """
        The main training loop. This method orchestrates the training and validation
        process over a specified number of epochs, logs comprehensive metrics,
        and saves model checkpoints.

        Args:
            num_epochs (int): The total number of epochs to train for.
            output_dir (str): The parent directory where model checkpoints will be saved.
                            A unique sub-folder will be created inside this directory.
        """
        # This helper method creates a unique, timestamped directory for this specific training run.
        self._setup_run_save_directory(output_dir)
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Models and logs will be saved to: {self.model_save_dir}")

        # The main loop that iterates through each epoch. An epoch is one full pass
        # through the entire training dataset.
        val_loss_lst = []
        train_loss_lst = []
        miou_lst = []
        cls_metrics_lst = []

        for epoch in range(num_epochs):
            print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

            # --- TRAINING STEP ---
            # Call the private method to perform one full pass over the training data.
            # This is where the model learns and its weights are updated.
            train_loss = self._train_one_epoch(epoch)
            train_loss_lst.append(train_loss)
            # --- VALIDATION STEP ---
            # Call the private method to evaluate the model's performance on the
            # validation set, which it has not seen during training. This gives us an
            # unbiased measure of how well the model is generalizing.
            val_loss, miou, cls_metrics = self._validate_one_epoch()
            val_loss_lst.append(val_loss)
            miou_lst.append(miou)
            cls_metrics_lst.append(cls_metrics)

            # we should also keep track of all the metrics in a list, so we can later create local plots


            # --- LOGGING ---
            # Log all the important metrics to a tracker like Weights & Biases (wandb).
            # This allows you to visualize the model's performance in real-time.
            if wandb.run:
                # Create a dictionary to hold all the data for this epoch.
                log_data = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/mIoU": miou,
                }
                # Add all the classification metrics (F1, recall, etc.) to the log
                # with a "val/" prefix to keep things organized.
                # e.g., {'f1': 0.8} becomes {'val/f1': 0.8}
                for key, value in cls_metrics.items():
                    log_data[f"val/{key}"] = value
                
                wandb.log(log_data)
                logger.info(f"Logged metrics to W&B for epoch {epoch + 1}.")

            # --- SAVE EPOCH CHECKPOINT ---
            # We save the model's state after every single epoch. This is crucial for
            # being able to go back and analyze the model at any point in its training history. It is also prudent to save each epoch, to 
            # allow us to continue our training after an unexpected interruption (e.g. a power failure, or a crash, or a bug).
            # The `state_dict` is a Python dictionary that maps each layer to its learnable parameters (weights and biases).
            epoch_ckpt_path = self.model_save_dir / f"epoch_{epoch:02d}.pth"
            torch.save(self.model.state_dict(), epoch_ckpt_path)

            # --- TRACK AND SAVE THE BEST MODEL ---
            # We check if the model's performance this epoch (MEASURED BY mIoU) is the
            # best we've seen so far. Since higher mIoU is better, we check for ">".
            # Note that the early stopper Class and its instance check for VALIDATION LOSS. If val. loss does not improve
            # for a number of 
            if miou > self.best_miou:
                self.best_miou = miou
                
                # 1. If a previous best model file exists, delete it.
                if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
                    print(f"Removing old best model: {self.best_ckpt_path}")
                    os.remove(self.best_ckpt_path)

                # 2. Define the new path and save the new model.
                new_best_path = self.model_save_dir / f"model_epoch_{epoch}_miou_{miou:.4f}.pth"
                torch.save(self.model.state_dict(), new_best_path)
                print(f"Saved NEW BEST model to: {new_best_path}")

                # 3. Update the tracker to the new path.
                self.best_ckpt_path = new_best_path
            
            # --- SAVE HISTORY EVERY EPOCH ---
            history = {
                "train_loss": train_loss_lst,
                "val_loss": val_loss_lst,
                "miou": miou_lst,
                "class_metrics_per_epoch": cls_metrics_lst
            }
            history_path = self.model_save_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)     

            # --- LEARNING RATE SCHEDULING ---
            # If a scheduler is provided, we update it based on the validation loss.
            # For ReduceLROnPlateau, this will decrease the learning rate if the
            # validation loss plateaus (stops improving).
            if self.scheduler:
                self.scheduler.step(val_loss)

        print("\n\nTraining complete!")
        logger.info(f"Best validation mIoU achieved during training: {self.best_miou:.4f}")