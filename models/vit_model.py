"""
Vision Transformer Model for Parkinson's Disease Detection
Includes transfer learning, fine-tuning, and interpretability features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from typing import Tuple, Optional
import numpy as np


class ViTForParkinsons(nn.Module):
    """Vision Transformer model fine-tuned for Parkinson's detection"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, 
                 freeze_base: bool = False):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_base: Whether to freeze base ViT layers
        """
        super(ViTForParkinsons, self).__init__()
        
        # Load pretrained ViT-Base model
        if pretrained:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        else:
            config = ViTConfig()
            self.vit = ViTModel(config)
        
        # Freeze base layers if specified
        if freeze_base:
            for param in self.vit.parameters():
                param.requires_grad = False
            
            # Unfreeze last 2 transformer layers for fine-tuning
            for param in self.vit.encoder.layer[-2:].parameters():
                param.requires_grad = True
        
        # Get hidden size
        hidden_size = self.vit.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Store attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, pixel_values: torch.Tensor, 
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model
        
        Args:
            pixel_values: Input images [batch_size, 3, 224, 224]
            output_attentions: Whether to return attention weights
            
        Returns:
            Logits and optionally attention weights
        """
        # Get ViT outputs
        outputs = self.vit(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            return_dict=True
        )
        
        # Get [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Classification
        logits = self.classifier(cls_output)
        
        # Store attention weights if requested
        if output_attentions:
            self.attention_weights = outputs.attentions
            return logits, outputs.attentions
        
        return logits, None
    
    def get_attention_map(self, image: torch.Tensor, layer_idx: int = -1) -> np.ndarray:
        """
        Extract attention map for visualization
        
        Args:
            image: Input image [1, 3, 224, 224]
            layer_idx: Which transformer layer to visualize
            
        Returns:
            Attention map as numpy array
        """
        self.eval()
        with torch.no_grad():
            _, attentions = self.forward(image, output_attentions=True)
        
        # Get attention from specified layer
        attention = attentions[layer_idx]  # [batch, num_heads, num_patches, num_patches]
        
        # Average across heads
        attention = attention.mean(dim=1)  # [batch, num_patches, num_patches]
        
        # Get attention to CLS token
        cls_attention = attention[0, 0, 1:]  # [num_patches-1]
        
        # Reshape to spatial dimensions (14x14 for 224x224 image with patch size 16)
        grid_size = int(np.sqrt(cls_attention.shape[0]))
        attention_map = cls_attention.reshape(grid_size, grid_size)
        
        return attention_map.cpu().numpy()


class BaselineCNN(nn.Module):
    """Baseline CNN model for comparison"""
    
    def __init__(self, num_classes: int = 2):
        """
        Args:
            num_classes: Number of output classes
        """
        super(BaselineCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            
        Returns:
            Logits
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
