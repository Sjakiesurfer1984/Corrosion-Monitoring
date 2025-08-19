# Corrosion_dataset.py

# === Dataset class for semantic segmentation of corrosion ===
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Callable
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor
from torchvision.transforms import Normalize

# A Recipe for Building a Neural Network with PyTorch
# Building a neural network involves a systematic process that transforms raw data into a predictive model. This guide outlines the essential steps, providing a clear "recipe" to follow. We'll use PyTorch as our framework.

# Step 1: Data Inception and Verification 🕵️‍♀️
# This is the starting point. You need to understand the data you're working with.

# Acquire Data: Load your data from its source (e.g., CSV files, databases, an API). For this recipe, let's assume you're loading a CSV file using the pandas library.

# Initial Inspection:

# View Shape: Check the number of rows and columns (df.shape).

# Examine Head/Tail: Look at the first and last few rows to get a feel for the features and target variable (df.head()).

# Check Data Types: Ensure columns have the correct data types (numeric, categorical, etc.) using df.info() or df.dtypes.

# Summary Statistics: For numerical features, generate descriptive statistics (mean, std, min, max) with df.describe().

# Check for Missing Values: Identify any missing data points (df.isnull().sum()). Decide on a strategy to handle them (e.g., imputation, removal).

# Step 2: Data Processing and Transformations ⚙️
# Raw data is rarely ready for a neural network. It needs to be cleaned, normalized, and converted into a numerical format.

# Handle Missing Data: Based on your verification, fill in or drop missing values.

# Encode Categorical Features: Neural networks only understand numbers. Convert categorical string data into a numerical format.

# One-Hot Encoding: Creates a new binary column for each category. Ideal for nominal data (no inherent order).

# Label Encoding: Assigns a unique integer to each category. Suitable for ordinal data (where order matters).

# Feature Scaling (Normalization/Standardization): This is crucial! It helps the network learn more efficiently by ensuring all features are on a similar scale.

# Normalization (Min-Max Scaling): Scales data to a fixed range, usually 0 to 1.

# Standardization (Z-score Normalization): Rescales data to have a mean of 0 and a standard deviation of 1.

# Split Data: Divide your dataset into three distinct sets:

# Training Set: Used to train the model (typically 70-80% of the data).

# Validation Set: Used to tune hyperparameters and monitor for overfitting during training (typically 10-15%).

# Test Set: Used for the final, unbiased evaluation of the model after training is complete (typically 10-15%).

# Convert to Tensors: PyTorch works with torch.Tensor objects. Convert your processed data (likely NumPy arrays at this stage) into tensors.

# Step 3: Create a Dataset and DataLoader 📦
# PyTorch provides convenient abstractions for handling data in batches.

# Create a Custom Dataset: This is a class that inherits from torch.utils.data.Dataset. It needs three methods:

# __init__(): To initialize the dataset, loading the features and labels.

# __len__(): To return the total number of samples in the dataset.

# __getitem__(idx): To retrieve a single sample (features and label) at a given index idx.

# Create DataLoader Instances: The DataLoader takes your Dataset object and automates the process of batching, shuffling, and loading data in parallel. You'll create one for each of your data splits (train, validation, test).

# batch_size: Defines how many samples are processed before the model's weights are updated.

# shuffle=True: For the training loader, this randomizes the order of data at each epoch to prevent the model from learning the data's sequence.

# Step 4: Create the Neural Network Model 🧠
# Define the architecture of your neural network.

# Define the Class: Create a class that inherits from torch.nn.Module.

# Initialize Layers in __init__(): Define all the layers of your network (e.g., linear layers, activation functions) in the constructor. The layers are the building blocks.

# torch.nn.Linear(in_features, out_features): A fully connected layer.

# Activation Functions (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Softmax): Introduce non-linearity, allowing the model to learn complex patterns.

# Define the Forward Pass in forward(): This method specifies how data flows through the layers you defined. The input x is passed sequentially through each layer and activation function, and the final output (logits) is returned.

# Step 5: Train the Model 🏋️‍♂️
# This is the core learning phase where the model adjusts its weights.

# Instantiate the Model, Loss Function, and Optimizer:

# Model: model = YourNetworkClass()

# Loss Function: Measures how far the model's prediction is from the actual target. The choice depends on the task (e.g., nn.BCELoss for binary classification, nn.CrossEntropyLoss for multi-class, nn.MSELoss for regression).

# Optimizer: Implements an algorithm to update the model's weights based on the loss. Common choices include torch.optim.Adam or torch.optim.SGD.

# The Training Loop: This loop iterates over the data for a specified number of epochs (one full pass through the entire training dataset).

# Set the model to training mode: model.train().

# Loop through the train_loader.

# Zero Gradients: optimizer.zero_grad() clears gradients from the previous iteration.

# Forward Pass: outputs = model(features) gets the model's predictions.

# Calculate Loss: loss = criterion(outputs, labels).

# Backward Pass (Backpropagation): loss.backward() computes the gradients of the loss with respect to model parameters.

# Update Weights: optimizer.step() updates the model's weights using the computed gradients.

# Step 6: Validate the Model ✅
# During training, you must check how the model is performing on data it hasn't seen (the validation set).

# The Validation Loop: This is typically nested inside the epoch loop, after a full pass of training.

# Set the model to evaluation mode: model.eval(). This disables layers like dropout that behave differently during training and inference.

# Use with torch.no_grad(): to disable gradient calculation, as you are not training here. This saves memory and computation.

# Loop through the validation_loader.

# Make predictions and calculate the loss and other metrics (e.g., accuracy).

# Track the validation loss. If it stops decreasing or starts increasing, it's a sign of overfitting, and you should consider stopping the training (early stopping).

# Step 7: Test the Model 🧪
# This is the final step. After training is complete, you evaluate the final model on the test set.

# The Test Loop: This is done only once after the model is fully trained.

# Load the best model weights you saved during training (based on the best validation performance).

# Set the model to evaluation mode: model.eval().

# Use with torch.no_grad():.

# Loop through the test_loader.

# Make predictions and calculate the final performance metrics (accuracy, precision, recall, F1-score, etc.).

# This result is your best estimate of how the model will perform on new, unseen data in the real world.





# Potential Improvements 💡
# Hardcoded RGB Values: The RGB values for the different corrosion levels are hardcoded directly inside __getitem__. This works, but it can be brittle. If you ever get a new mask with slightly different RGB values (e.g., [254, 0, 0] instead of [255, 0, 0]), this logic will fail.

# Suggestion: Define a color map at the top of your file or as a class attribute. This makes it easier to manage and modify the colors.

# Python

# # At the top of your file or inside the class
# COLOR_MAP = {
#     (255, 0, 0): 1,   # Red -> Fair
#     (0, 255, 0): 2,   # Green -> Poor
#     (255, 255, 0): 3, # Yellow -> Severe
# }


# dataset/corrosion_dataset.py

# Import necessary libraries
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

# --- The Dataset Class ---
# This class is a blueprint for how PyTorch should load our data.
# It inherits from `torch.utils.data.Dataset`.
class CorrosionDataset(Dataset):
    """
    A custom PyTorch Dataset for loading corrosion images and their segmentation masks.
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        # 'Optional[Callable]' means this argument can either be a callable function
        # (like our augmentation pipeline) or None.
        augmentations: Optional[Callable] = None
    ) -> None:
        """
        The constructor for our dataset. This runs once when we create the dataset object.

        Args:
            image_dir (str): The path to the folder containing the input images.
            mask_dir (str): The path to the folder containing the segmentation masks.
            augmentations (Optional[Callable]): A function/callable to apply for data augmentation.
        """
        # We use pathlib.Path for a modern, object-oriented way to handle file paths.
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.augmentations = augmentations

        # We find all images and masks and sort them to ensure they align.
        # For example, img_01.jpeg will match mask_01.png.
        self.image_paths = sorted(list(self.image_dir.glob("*.jpeg")))
        self.mask_paths = sorted(list(self.mask_dir.glob("*.png")))

        # A crucial sanity check: make sure we have the same number of images and masks.
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch in image and mask count in {image_dir} and {mask_dir}"

    def __len__(self) -> int:
        """This method returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method is the most important. It loads and returns ONE sample from the dataset
        at the given index 'idx'. This is called by the DataLoader for every single image.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the processed image
                                               tensor and the corresponding label mask tensor.
        """
        # Get the file paths for the specific image and mask at this index.
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Open the image and mask files using the PIL library.
        # We .convert("RGB") to ensure they are in the standard 3-channel format.
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # --- SYNCHRONIZED AUGMENTATION STEP ---
        # If we provided an augmentation function during initialization, we call it here.
        # It takes both the image and mask and applies the same random transform to both.
        if self.augmentations:
            image, mask = self.augmentations(image, mask)

        # --- MASK CONVERSION: From Color to Class Index ---
        # A neural network doesn't understand colors; it understands numbers.
        # We need to convert the RGB mask (e.g., red pixels [255,0,0]) into a
        # single-channel map of class indices (e.g., all red pixels become the number 1).
        mask_np = np.array(mask)
        label_mask = np.zeros(mask_np.shape[:2], dtype=np.uint8) # Create an empty (all zeros) map

        # Find all pixels that are pure red and set their value in our label_mask to 1.
        label_mask[(mask_np == [255, 0, 0]).all(axis=-1)] = 1  # Fair
        # Find all pixels that are pure green and set their value to 2.
        label_mask[(mask_np == [0, 255, 0]).all(axis=-1)] = 2  # Poor
        # Find all pixels that are pure yellow and set their value to 3.
        label_mask[(mask_np == [255, 255, 0]).all(axis=-1)] = 3  # Severe
        # The background pixels remain 0 by default.

        # --- FINAL CONVERSIONS TO TENSOR ---
        # 1. Convert the augmented PIL image to a PyTorch Tensor. This also scales
        #    pixel values from the [0, 255] range to the [0.0, 1.0] range.
        image_tensor = F.to_tensor(image)

        # 2. Normalize the image tensor. This is a standard practice where we subtract the
        #    mean and divide by the standard deviation of the ImageNet dataset. It helps
        #    the model train more effectively.
        image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 3. Convert our numpy label map to a PyTorch tensor.
        #    It must be of type 'long' (64-bit integer) for the loss function.
        label_tensor = torch.from_numpy(label_mask).long()

        return image_tensor, label_tensor

# --- NEW Augmentation Classes for Synchronized Transforms ---
# These classes are designed to be "callable" (meaning they have a __call__ method),
# so we can treat an object of the class as if it were a function.

class JointRandomHorizontalFlip:
    """
    A callable class that randomly flips an image and its mask horizontally
    with a given probability.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # torch.rand(1) generates a single random number between 0 and 1.
        if torch.rand(1) < self.p:
            # If the number is less than our probability 'p', we apply the flip
            # to BOTH the image and the mask.
            return F.hflip(img), F.hflip(mask)
        # Otherwise, we return them unchanged.
        return img, mask

class JointRandomRotation:
    """
    A callable class that randomly rotates an image and its mask by the same angle.
    """
    def __init__(self, degrees: Tuple[float, float]):
        self.degrees = degrees

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Get a single random angle within the specified degree range.
        angle = T.RandomRotation.get_params(self.degrees)
        # Apply the rotation with the *exact same angle* to both image and mask.
        return F.rotate(img, angle), F.rotate(mask, angle)

class JointTransforms:
    """A helper class to chain multiple joint transformations together."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask