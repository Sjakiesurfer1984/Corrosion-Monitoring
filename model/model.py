# model.py
# This file serves as a centralised factory for creating our neural network models.
# It uses a registry pattern to allow for easy addition of new model architectures
# without needing to modify the core `get_model` function.

# --- Core Library Imports ---
import torch
import torchvision.models.segmentation as seg_models
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights
)

# --- 1. THE MODEL REGISTRY ---
# A dictionary that acts as a registry to map model names (strings) to their
# corresponding creation functions. This is a core component of the plugin architecture.
MODEL_REGISTRY = {}

def register_model(name: str):
    """
    A decorator to register a model creation function in the global registry.
    This pattern ensures the registry is populated automatically when a file is imported.
    A decorator in Python is a function that takes another function as its argument, performs some action, 
    and then returns a new, modified function or the original function. Essentially, it "decorates" or wraps another function, 
    giving it new or altered behaviour without permanently changing its source code. 
    It's a syntactic shortcut for wrapping a function call.
    """
    def decorator(func):
        # We store the function in our global registry, using a lower case name for consistency.
        MODEL_REGISTRY[name.lower()] = func
        return func
    return decorator

# --- 2. REGISTERING THE MODELS ---
# Each model's creation and configuration logic is encapsulated in its own function.
# The `@register_model` decorator automatically adds these functions to the registry.

@register_model("deeplabv3_resnet50")
def create_deeplabv3_resnet50(num_classes: int) -> torch.nn.Module:
    """
    Creates and configures the DeepLabV3 model with a ResNet-50 backbone.
    """
    # Load the pre-trained model and weights. This is an example of transfer learning.
    model = seg_models.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    # We must replace the final classifier layer to match our number of classes.
    # The original model was trained on 21 classes from the COCO dataset.
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    # If the model has an auxiliary classifier (common in segmentation models),
    # we also need to modify it to output the correct number of classes.
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        in_channels = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    return model

@register_model("fcn_resnet50")
def create_fcn_resnet50(num_classes: int) -> torch.nn.Module:
    """
    Creates and configures the Fully Convolutional Network (FCN) model with a ResNet-50 backbone.
    """
    model = seg_models.fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)

    # The same logic applies here; we replace the final classifier.
    # Note that the input channels (512) are different for this architecture.
    model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=1)

    # And we modify the auxiliary classifier as well.
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        in_channels = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1)

    return model

# --- 3. THE SINGLE POINT OF ACCESS ---
# The main function to be called from external scripts (e.g., run_train.py).

def get_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """
    Acts as a factory for creating neural network models using a registry lookup.
    
    Args:
        model_name (str): The name of the model to retrieve from the registry.
        num_classes (int): The number of output classes for the final layer.
        
    Returns:
        torch.nn.Module: The configured PyTorch model.
        
    Raises:
        ValueError: If the provided `model_name` is not found in the registry.
    """
    model_name_lower = model_name.lower()
    
    # Check if the requested model exists in our registry.
    if model_name_lower not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not recognised. Available models: {list(MODEL_REGISTRY.keys())}")
        
    # Retrieve the model creation function from the registry.
    model_creator = MODEL_REGISTRY[model_name_lower]
    
    # Execute the function to get the configured model instance.
    model = model_creator(num_classes)
    
    return model