# lightning_model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score
from typing import Tuple

# Import your model factory from your existing model.py file.
# This assumes the file structure is model/model.py
from model.model import get_model

class CorrosionSegmenter(pl.LightningModule):
    """
    A PyTorch Lightning Module for the corrosion segmentation task.

    This class encapsulates the entire system: the model, the loss function,
    the optimization logic, and the training/validation/testing steps.
    """
    def __init__(
        # the attributes of the class are defined here.
        # however, we use `self.hparams` to store hyperparameters, like model_name, num_classes, and learning_rate.
        # This is a PyTorch Lightning convention.
        # It allows us to save and load the model with all its hyperparameters,
        # so we can recreate the model later without needing to remember
        # every single parameter.
        # Note: We do NOT use `self.model_name`, `self.num_classes`, etc. directly.
        # The assignment to `self.hparams` is done automatically by PyTorch Lightning
        self,
        model_name: str = "deeplabv3_resnet50",
        num_classes: int = 4,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = False
    ):
        """
        The constructor for our LightningModule.

        Args:
            model_name (str): The name of the model architecture to use from the factory.
            num_classes (int): The number of classes (e.g., 4 for background + 3 corrosion types).
            learning_rate (float): The learning rate for the optimizer.
        """
        super().__init__()

        # This is a PyTorch Lightning best practice. It saves the arguments passed to __init__
        # (like model_name, num_classes, lr) as hyperparameters, which are then
        # automatically saved with the model checkpoints.
        self.save_hyperparameters()

        # --- Core Components ---
        # 1. The Model (The "Engine")
        # We use the get_model factory from our model.py file to create the raw
        # neural network architecture.
        # ------------------------------ FETCHING THE MODEL, DEFINED IN MODEL.PY -----------------------------
        self.model = get_model(model_name=self.hparams.model_name, num_classes=self.hparams.num_classes)

        # ------------------ ADDED LOGIC FOR FREEZING THE BACKBONE ------------------
        if freeze_backbone:
            # Check the model name to find the correct backbone to freeze
            if self.hparams.model_name == "deeplabv3_resnet50":
                # The backbone for DeepLabV3 is the `model.backbone` attribute.
                # We iterate over its parameters and set `requires_grad` to False.
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
                print("Backbone for deeplabv3_resnet50 is frozen.")
            elif self.hparams.model_name == "fcn_resnet50":
                # The backbone for FCN is also `model.backbone`.
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
                print("Backbone for fcn_resnet50 is frozen.")
            else:
                print("Backbone freezing not supported for this model.")



        # 2. The Loss Function
        # We define the criterion that will be used to measure the error of the predictions.
        self.criterion = nn.CrossEntropyLoss()

        # 3. The Metrics
        # We use the highly optimized `torchmetrics` library. It's important to define
        # separate metric objects for validation and testing to keep their internal states separate.
        self.val_iou = MulticlassJaccardIndex(num_classes=self.hparams.num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes)
        
        self.test_iou = MulticlassJaccardIndex(num_classes=self.hparams.num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The raw output (logits) from the model.
        """
        # We simply pass the input through our underlying model.
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Defines the logic for a single training step (for one batch).
        This method is called automatically by the Lightning Trainer.
        """
        images, masks = batch

        # Get the raw model output (logits)
        outputs = self.forward(images)['out']

        # Calculate the loss
        loss = self.criterion(outputs, masks)

        # Log the training loss. `prog_bar=True` shows it in the progress bar.
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Defines the logic for a single validation step (for one batch).
        This method is called automatically by the Lightning Trainer.
        """
        images, masks = batch

        # Get the raw model output (logits)
        outputs = self.forward(images)['out']

        # Calculate the loss
        loss = self.criterion(outputs, masks)

        # Convert logits to final predictions (class index per pixel)
        preds = torch.argmax(outputs, dim=1)

        # Update our validation metrics with the predictions and ground truth
        self.val_iou.update(preds, masks)
        self.val_f1.update(preds, masks)

        # Log the loss and metrics. By default, these are aggregated over the epoch.
        self.log("val/loss", loss, logger=True)
        self.log("val/mIoU", self.val_iou, logger=True)
        self.log("val/f1_score", self.val_f1, logger=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Defines the logic for a single test step (for one batch).
        This method is called automatically when you run `trainer.test()`.
        """
        images, masks = batch
        outputs = self.forward(images)['out']
        loss = self.criterion(outputs, masks)
        preds = torch.argmax(outputs, dim=1)

        # Update our test metrics
        self.test_iou.update(preds, masks)
        self.test_f1.update(preds, masks)

        # Log the test metrics
        self.log("test/loss", loss, logger=True)
        self.log("test/mIoU", self.test_iou, logger=True)
        self.log("test/f1_score", self.test_f1, logger=True)

    def configure_optimizers(self) -> dict:
        """
        Sets up the optimizer and (optional) learning rate scheduler.
        This method is called automatically by PyTorch Lightning.

        Returns:
            dict: A dictionary containing the optimizer and scheduler configuration.
        """
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/mIoU", # The metric to monitor for the scheduler
            },
        }