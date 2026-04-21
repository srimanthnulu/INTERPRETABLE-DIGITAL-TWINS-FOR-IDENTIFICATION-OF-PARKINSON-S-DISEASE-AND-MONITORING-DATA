"""
Grad-CAM Implementation for Visual Explanations
Highlights regions that contribute most to predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The neural network model
            target_layer: Layer to extract gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image: torch.Tensor, 
                    target_class: int = None) -> np.ndarray:
        """
        Generate class activation map
        
        Args:
            input_image: Input image tensor [1, 3, 224, 224]
            target_class: Target class index (None for predicted class)
            
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output, _ = self.model(input_image, output_attentions=False)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [hidden_size]
        activations = self.activations[0]  # [num_patches, hidden_size]
        
        # Weight the channels by gradient
        weights = gradients.mean(dim=0, keepdim=True)  # Global average pooling
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1)  # [num_patches]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Reshape to spatial dimensions
        grid_size = int(np.sqrt(cam.shape[0] - 1))  # Exclude CLS token
        cam = cam[1:].reshape(grid_size, grid_size)  # Skip CLS token
        
        return cam.cpu().numpy()


class ViTGradCAM:
    """Grad-CAM adapted for Vision Transformer"""
    
    def __init__(self, model):
        """
        Args:
            model: ViT model
        """
        self.model = model
        self.gradients = None
        
    def generate_cam(self, input_image: torch.Tensor, 
                    target_class: int = None) -> np.ndarray:
        """
        Generate CAM for ViT using attention rollout and gradients
        
        Args:
            input_image: Input image [1, 3, 224, 224]
            target_class: Target class (None for predicted)
            
        Returns:
            CAM heatmap
        """
        self.model.eval()
        input_image.requires_grad = True
        
        # Forward pass
        output, attentions = self.model(input_image, output_attentions=True)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients w.r.t input
        gradients = input_image.grad.data[0]  # [3, 224, 224]
        
        # Use attention from last layer
        attention = attentions[-1][0].mean(dim=0)  # Average over heads
        
        # Get attention to CLS token
        cls_attention = attention[0, 1:]  # Exclude CLS token
        
        # Reshape
        grid_size = int(np.sqrt(cls_attention.shape[0]))
        attention_map = cls_attention.reshape(grid_size, grid_size)
        
        # Combine with gradient magnitude
        grad_magnitude = gradients.abs().mean(dim=0)  # [224, 224]
        grad_magnitude = F.interpolate(
            grad_magnitude.unsqueeze(0).unsqueeze(0),
            size=(grid_size, grid_size),
            mode='bilinear',
            align_corners=False
        )[0, 0]
        
        # Combine attention and gradients
        cam = attention_map * grad_magnitude
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().detach().numpy()


def visualize_cam(image_path: str, cam: np.ndarray, 
                 save_path: str = None) -> np.ndarray:
    """
    Overlay CAM heatmap on original image
    
    Args:
        image_path: Path to original image
        cam: CAM heatmap
        save_path: Path to save visualization
        
    Returns:
        Overlaid image
    """
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (224, 224))
    
    # Apply colormap
    heatmap = cm.jet(cam_resized)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Overlay
    overlaid = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    if save_path:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cam_resized, cmap='jet')
        plt.title('Attention Heatmap')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlaid)
        plt.title('Overlaid Explanation')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return overlaid


def generate_attention_rollout(attentions: Tuple, head_fusion: str = 'mean') -> np.ndarray:
    """
    Generate attention rollout visualization
    
    Args:
        attentions: Tuple of attention tensors from all layers
        head_fusion: How to combine attention heads ('mean', 'max', 'min')
        
    Returns:
        Attention rollout map
    """
    # Stack all layer attentions
    all_attentions = torch.stack([att.squeeze() for att in attentions])
    
    # Fuse attention heads
    if head_fusion == 'mean':
        fused = all_attentions.mean(dim=1)
    elif head_fusion == 'max':
        fused = all_attentions.max(dim=1)[0]
    else:
        fused = all_attentions.min(dim=1)[0]
    
    # Add identity matrix
    identity = torch.eye(fused.size(-1))
    fused = fused + identity
    
    # Normalize
    fused = fused / fused.sum(dim=-1, keepdim=True)
    
    # Recursively multiply attention matrices
    rollout = fused[0]
    for i in range(1, fused.size(0)):
        rollout = torch.matmul(fused[i], rollout)
    
    # Get attention to CLS token
    cls_attention = rollout[0, 1:]
    
    # Reshape to spatial grid
    grid_size = int(np.sqrt(cls_attention.shape[0]))
    attention_map = cls_attention.reshape(grid_size, grid_size)
    
    return attention_map.cpu().numpy()
