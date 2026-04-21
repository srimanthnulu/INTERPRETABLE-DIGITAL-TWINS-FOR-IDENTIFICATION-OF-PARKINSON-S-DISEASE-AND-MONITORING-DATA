"""
Training Module for Parkinson's Detection Model
Includes training loop, validation, and comprehensive evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, Tuple, List
import json
import os


class Trainer:
    """Training manager for Parkinson's detection models"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda',
                 learning_rate: float = 1e-4, weight_decay: float = 0.01):
        """
        Args:
            model: Neural network model
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss and accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs, _ = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 20, save_path: str = 'best_model.pth') -> Dict:
        """
        Train the model for multiple epochs
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        best_val_acc = 0.0
        
        print(f"\n{'='*50}")
        print(f"Starting Training for {epochs} epochs")
        print(f"{'='*50}\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, save_path)
                print(f"✓ Best model saved with validation accuracy: {val_acc:.2f}%")
        
        print(f"\n{'='*50}")
        print(f"Training Complete! Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"{'='*50}\n")
        
        return self.history


class Evaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: Trained model
            device: Device for inference
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions on dataset
        
        Args:
            data_loader: Data loader
            
        Returns:
            True labels, predicted labels, and probabilities
        """
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(data_loader, desc='Predicting'):
                images = images.to(self.device)
                
                outputs, _ = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of PD class
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def evaluate(self, data_loader: DataLoader, save_dir: str = 'results') -> Dict:
        """
        Comprehensive evaluation with all metrics
        
        Args:
            data_loader: Data loader
            save_dir: Directory to save results
            
        Returns:
            Dictionary of metrics
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Get predictions
        y_true, y_pred, y_prob = self.predict(data_loader)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_true, y_prob)
        }
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        print("="*50 + "\n")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(cm, save_dir)
        
        # ROC Curve
        self._plot_roc_curve(y_true, y_prob, metrics['roc_auc'], save_dir)
        
        # Classification Report
        report = classification_report(y_true, y_pred, 
                                      target_names=['Healthy', 'Parkinsons'])
        print("\nClassification Report:")
        print(report)
        
        # Save metrics
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        return metrics
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_dir: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Healthy', 'Parkinsons'],
                   yticklabels=['Healthy', 'Parkinsons'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                       auc: float, save_dir: str):
        """Plot and save ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=150)
        plt.close()


def plot_training_history(history: Dict, save_path: str = 'training_history.png'):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
