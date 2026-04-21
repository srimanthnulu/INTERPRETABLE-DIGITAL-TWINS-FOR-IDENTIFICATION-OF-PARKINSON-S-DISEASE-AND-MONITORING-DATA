"""
Data Preprocessing Module for Parkinson's Disease Detection
Handles image loading, augmentation, and preprocessing
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from typing import Tuple, List, Optional
import cv2


class ParkinsonsDataset(Dataset):
    """Custom dataset for Parkinson's disease image classification"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, augment: bool = False):
        """
        Args:
            image_paths: List of paths to images
            labels: List of labels (0: Healthy, 1: Parkinson's)
            transform: Preprocessing transforms
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path


def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    Get preprocessing transforms for images
    
    Args:
        augment: Whether to include data augmentation
        
    Returns:
        Composed transforms
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def prepare_data_loaders(train_paths: List[str], train_labels: List[int],
                        val_paths: List[str], val_labels: List[int],
                        batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and validation data loaders
    
    Args:
        train_paths: Training image paths
        train_labels: Training labels
        val_paths: Validation image paths
        val_labels: Validation labels
        batch_size: Batch size for data loaders
        
    Returns:
        Train and validation data loaders
    """
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    train_dataset = ParkinsonsDataset(train_paths, train_labels, 
                                     train_transform, augment=True)
    val_dataset = ParkinsonsDataset(val_paths, val_labels, 
                                   val_transform, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def preprocess_single_image(image_path: str) -> torch.Tensor:
    """
    Preprocess a single image for inference
    
    Args:
        image_path: Path to the image
        
    Returns:
        Preprocessed image tensor
    """
    transform = get_transforms(augment=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor


def enhance_image_quality(image_path: str) -> np.ndarray:
    """
    Enhance image quality for better analysis
    
    Args:
        image_path: Path to the image
        
    Returns:
        Enhanced image as numpy array
    """
    img = cv2.imread(image_path)
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced_rgb
