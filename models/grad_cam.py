"""
Grad-CAM implementation for model interpretability
Class Activation Maps for visualizing model predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    Visualizes which parts of the image influence the model's decision
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Get the target layer
        if isinstance(self.target_layer, str):
            # Navigate through model to find layer
            layer = self.model
            for attr in self.target_layer.split('.'):
                layer = getattr(layer, attr)
        else:
            layer = self.target_layer
        
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class for visualization
        
        Returns:
            CAM heatmap as numpy array (0-255)
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target_score = output[0, target_class]
        target_score.backward()
        
        # Calculate weights
        if self.gradients is None or self.activations is None:
            return np.ones((224, 224, 3), dtype=np.uint8) * 255
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input size
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = np.uint8(255 * cam)
        
        # Apply colormap
        cam_colored = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        return cam_colored

class ActivationMap:
    """
    Simple activation map visualization
    Shows raw feature activations from intermediate layers
    """
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.features = None
        
        # Register hook
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook"""
        def hook(module, input, output):
            self.features = output.detach()
        
        layer = self.model
        for attr in self.layer_name.split('.'):
            layer = getattr(layer, attr)
        
        layer.register_forward_hook(hook)
    
    def visualize(self, input_tensor, num_filters=16):
        """
        Visualize top activation filters
        
        Args:
            input_tensor: Input image tensor
            num_filters: Number of filters to visualize
        
        Returns:
            Combined visualization image
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if self.features is None:
            return np.ones((256, 256, 3), dtype=np.uint8) * 255
        
        features = self.features[0]  # Take first image in batch
        num_channels = min(features.shape[0], num_filters)
        
        # Create grid
        img_per_row = 4
        rows = (num_channels + img_per_row - 1) // img_per_row
        
        grid_height = rows * 64
        grid_width = img_per_row * 64
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        for idx in range(num_channels):
            row = idx // img_per_row
            col = idx % img_per_row
            
            # Get feature map
            feature = features[idx].cpu().numpy()
            feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
            feature = (feature * 255).astype(np.uint8)
            feature = cv2.resize(feature, (64, 64))
            
            # Place in grid
            grid[row*64:(row+1)*64, col*64:(col+1)*64] = feature
        
        # Convert to RGB
        grid_rgb = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)
        return grid_rgb

class SaliencyMap:
    """
    Compute input gradients for saliency visualization
    Shows which pixels most influence the prediction
    """
    def __init__(self, model):
        self.model = model
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate saliency map
        
        Args:
            input_tensor: Input image tensor (requires grad)
            target_class: Target class for visualization
        
        Returns:
            Saliency map as numpy array
        """
        input_tensor.requires_grad = True
        
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        target_score = output[0, target_class]
        target_score.backward()
        
        # Get gradients
        gradients = input_tensor.grad.data.abs()
        gradients = gradients.squeeze().cpu().numpy()
        
        # Max across color channel if needed
        if len(gradients.shape) == 3:
            gradients = gradients.max(axis=0)
        
        # Normalize
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
        saliency = np.uint8(255 * gradients)
        
        return saliency

if __name__ == "__main__":
    print("Grad-CAM utilities loaded successfully")
