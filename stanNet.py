# networks.py
# Complex-valued neural network for yoga pose classification using HSV color space features

import torch
from torch import nn

# Import complex-valued layers and operations
from complexnn import (
    ComplexConv2d,           # Complex convolution layer
    ComplexBatchNorm2d,      # Complex batch normalization
    ComplexMaxPool2d,        # Complex max pooling
    ComplexAdaptiveAvgPool2d, # Complex adaptive average pooling
    ComplexDropout,          # Complex dropout regularization
)

# Import complex-valued activation functions
from complex_activations import CReLU  # Complex ReLU (CPReLU is also available)


class stanNet(nn.Module):
    """
    stanNet: A complex-valued convolutional neural network for image classification.
    
    This architecture processes complex-valued input features (H, S, V-derived) and
    progressively increases channel depth across three processing stages, followed by
    global average pooling and a classification head.
    
    Args:
        num_classes (int): Number of output classes for classification.
        width (int): Base channel width for the first layer (default: 32).
                    Subsequent stages use 2x, 4x, and 8x this width.
        p_drop (float): Dropout probability applied after max pooling in each stage (default: 0.1).
    """
    
    def __init__(self, num_classes: int, width: int = 32, p_drop: float = 0.1):
        super().__init__()
        w = width  # Shorthand for base channel width
        
        # Stem: Initial feature extraction layer
        # Input: complex64 tensor with 3 channels (from ToComplex on H, S, V features)
        # Output: w channels after convolution, batch norm, and activation
        self.stem = nn.Sequential(
            ComplexConv2d(3, w, kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(w),
            CReLU(),
        )
        
        # Stage 1: First processing stage with downsampling
        # Channels: w -> 2*w
        # Spatial dimensions reduced by 2x via max pooling
        self.stage1 = nn.Sequential(
            ComplexConv2d(w, 2*w, kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(2*w),
            CReLU(),
            ComplexMaxPool2d(kernel_size=2),  # Downsample by 2x
            ComplexDropout(p_drop),            # Regularization
        )
        
        # Stage 2: Second processing stage with further downsampling
        # Channels: 2*w -> 4*w
        # Spatial dimensions reduced by 4x total (2x from each of stages 1 and 2)
        self.stage2 = nn.Sequential(
            ComplexConv2d(2*w, 4*w, kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(4*w),
            CReLU(),
            ComplexMaxPool2d(kernel_size=2),  # Downsample by 2x
            ComplexDropout(p_drop),            # Regularization
        )
        
        # Stage 3: Third processing stage with final downsampling
        # Channels: 4*w -> 8*w
        # Spatial dimensions reduced by 8x total (2x from each of stages 1, 2, and 3)
        self.stage3 = nn.Sequential(
            ComplexConv2d(4*w, 8*w, kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(8*w),
            CReLU(),
            ComplexMaxPool2d(kernel_size=2),  # Downsample by 2x
            ComplexDropout(p_drop),            # Regularization
        )
        
        # Global average pooling: Reduces spatial dimensions to 1x1
        # Preserves channel information for classification
        self.gap = ComplexAdaptiveAvgPool2d((1, 1))
        
        # Classification head: 1x1 complex convolution mapping 8*w channels to num_classes
        # Output: Complex logits (B, num_classes, 1, 1)
        self.head = ComplexConv2d(8*w, num_classes, kernel_size=1, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Complex64 input tensor of shape (B, 3, H, W)
                            where B is batch size, H and W are spatial dimensions.
                            Expected input: 3 complex channels (e.g., from ToHSV() + ToComplex()).
        
        Returns:
            torch.Tensor: Complex logits of shape (B, num_classes, 1, 1).
                         Typically passed to a complex-valued loss function.
        """
        # Pass through stem for initial feature extraction
        x = self.stem(x)
        
        # Pass through three progressive processing stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Global average pooling to aggregate spatial information
        x = self.gap(x)
        
        # Classification head produces final logits
        x = self.head(x)
        
        return x  # Shape: (B, num_classes, 1, 1) as complex logits


def stanNet_complex(num_classes: int):
    """
    Factory function to create a stanNet instance with default hyperparameters.
    
    Args:
        num_classes (int): Number of output classes for the classification task.
    
    Returns:
        stanNet: A stanNet model instance with base width 32 and dropout probability 0.1.
    """
    return stanNet(num_classes=num_classes, width=32, p_drop=0.1)
