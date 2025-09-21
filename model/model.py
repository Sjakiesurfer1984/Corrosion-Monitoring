# model.py
# This file is responsible for defining models. Some are simple, others are more complex. 
# Hence a separate file, to contain and abstract away the definition of models. 

# Import necessary modules from torchvision.
# We now switch to a semantic segmentation model (DeepLabV3)
# because our masks are pixel-wise class maps, not instance-based.

# We import the necessary PyTorch libraries.
# 'torch' is the main library, and 'seg_models' is a specific part of torchvision
# that contains pre-built segmentation models like DeepLabV3 and FCN.
import torch
import torchvision.models.segmentation as seg_models

# We also import the pre-trained 'weights' for these models.
# Using pre-trained weights (from a huge dataset like COCO) is a technique
# called 'transfer learning'. It's like hiring an expert who already knows a lot
# about general images, so they can learn our specific corrosion task much faster.
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights
)

# --- The Model Factory Function ---
def get_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """
    This function acts as a factory for creating neural network models.
    Based on the 'model_name' string you provide, it returns the
    corresponding model architecture, ready for training.

    Args:
        model_name (str): The name of the model you want to use.
        (e.g., "deeplabv3_resnet50", "fcn_resnet50").
        we make this case-insensitive.

        num_classes (int): The number of categories you want the model to predict.
        For our task, this would be 4 (Background, Fair, Poor, Severe).

    Returns:
        torch.nn.Module: The constructed PyTorch model. `torch.nn.Module` is the
        base class for all neural networks in PyTorch.
    """
    # It's good practice to convert the input name to lowercase
    # to avoid errors from simple capitalization mistakes (e.g., "DeepLab" vs "deeplab").
    model_name = model_name.lower()

    # --- Model Selection Logic ---
    # This is like a switchboard. It checks the model_name and runs the
    # appropriate code to build that model.

    if model_name == "deeplabv3_resnet50":
        # Load the DeepLabV3 model with a ResNet-50 backbone.
        # `weights=...DEFAULT` tells PyTorch to download and use the
        # best available pre-trained weights.
        model = seg_models.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

        # --- IMPORTANT: Modifying the final layer ---
        # The pre-trained model was trained on the COCO dataset, which has 21 classes.
        # We only have 4 classes. So, we MUST replace the final layer (the "classifier")
        # with a new one that outputs the correct number of channels (one for each of our classes).
        # For DeepLabV3, the final layer is at `model.classifier[4]`.
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    # --- ADD NEW MODELS HERE ---
    # To add another model, we just add another 'elif' block. 
    elif model_name == "fcn_resnet50":
        # Here we do the same for the FCN (Fully Convolutional Network) model.
        model = seg_models.fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)

        # We replace its final layer. Note that the number of input channels (512)
        # might be different from DeepLab's (256). You have to know your model architecture.
        model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=1)

    else:
        # If the user provides a name we don't recognize, we raise an error
        # to stop the program with a clear message.
        raise ValueError(f"Model '{model_name}' not recognized in the factory.")

    # Some models, like DeepLabV3, have an 'auxiliary classifier' which helps in training.
    # We should also modify this layer if it exists. `hasattr` checks if the model
    # object has an attribute named 'aux_classifier' before we try to access it.
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        # We get the number of input channels from the existing layer before replacing it.
        in_channels = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1)

    return model