"""
Main Training Script for Parkinson's Detection Model
Trains both ViT and baseline CNN models with comprehensive evaluation
"""

import torch
import os
import sys
import argparse
from sklearn.model_selection import train_test_split
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vit_model import ViTForParkinsons, BaselineCNN, count_parameters
from utils.data_preprocessing import prepare_data_loaders
from utils.training import Trainer, Evaluator, plot_training_history


def load_dataset_paths(data_dir: str):
    """
    Load image paths and labels from directory structure
    
    Expected structure:
    data_dir/
        healthy/
            spiral/
            wave/
            handwriting/
        parkinsons/
            spiral/
            wave/
            handwriting/
    
    Args:
        data_dir: Root data directory
        
    Returns:
        Lists of image paths and labels
    """
    image_paths = []
    labels = []
    
    categories = ['healthy', 'parkinsons']
    image_types = ['spiral', 'wave', 'handwriting']
    
    for label, category in enumerate(categories):
        for image_type in image_types:
            type_dir = os.path.join(data_dir, category, image_type)
            
            if not os.path.exists(type_dir):
                print(f"Warning: Directory not found: {type_dir}")
                continue
            
            for img_file in os.listdir(type_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(type_dir, img_file)
                    image_paths.append(img_path)
                    labels.append(label)
    
    print(f"Loaded {len(image_paths)} images")
    print(f"  Healthy: {labels.count(0)}")
    print(f"  Parkinson's: {labels.count(1)}")
    
    return image_paths, labels


def create_sample_data(data_dir: str, num_samples: int = 100):
    """
    Create sample dataset structure for demonstration
    
    Args:
        data_dir: Directory to create sample data
        num_samples: Number of sample images per category
    """
    import cv2
    
    print("Creating sample dataset...")
    
    categories = ['healthy', 'parkinsons']
    image_types = ['spiral', 'wave', 'handwriting']
    
    for category in categories:
        for image_type in image_types:
            type_dir = os.path.join(data_dir, category, image_type)
            os.makedirs(type_dir, exist_ok=True)
            
            # Create synthetic images (in practice, use real data)
            for i in range(num_samples // len(image_types)):
                # Create random image (placeholder)
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                
                img_path = os.path.join(type_dir, f'{image_type}_{i:04d}.png')
                cv2.imwrite(img_path, img)
    
    print(f"Sample data created in {data_dir}")


def train_model(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    if not os.path.exists(args.data_dir):
        print(f"Data directory not found. Creating sample data...")
        create_sample_data(args.data_dir, num_samples=200)
    
    image_paths, labels = load_dataset_paths(args.data_dir)
    
    if len(image_paths) == 0:
        raise ValueError("No images found. Please provide a valid dataset.")
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_paths)} images")
    print(f"  Validation: {len(val_paths)} images")
    
    # Create data loaders
    print("\nPreparing data loaders...")
    train_loader, val_loader = prepare_data_loaders(
        train_paths, train_labels,
        val_paths, val_labels,
        batch_size=args.batch_size
    )
    
    # Train ViT model
    print("\n" + "="*80)
    print("TRAINING VISION TRANSFORMER MODEL")
    print("="*80)
    
    vit_model = ViTForParkinsons(num_classes=2, pretrained=True, 
                                freeze_base=args.freeze_base)
    print(f"\nViT Model Parameters: {count_parameters(vit_model):,}")
    
    vit_trainer = Trainer(vit_model, device=device, 
                         learning_rate=args.learning_rate,
                         weight_decay=args.weight_decay)
    
    vit_history = vit_trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs,
        save_path=os.path.join(args.results_dir, 'vit_best_model.pth')
    )
    
    # Plot ViT training history
    plot_training_history(vit_history, 
                         os.path.join(args.results_dir, 'vit_training_history.png'))
    
    # Evaluate ViT model
    print("\n" + "="*80)
    print("EVALUATING VISION TRANSFORMER MODEL")
    print("="*80)
    
    vit_evaluator = Evaluator(vit_model, device=device)
    vit_metrics = vit_evaluator.evaluate(
        val_loader,
        save_dir=os.path.join(args.results_dir, 'vit_evaluation')
    )
    
    # Train baseline CNN if requested
    if args.train_baseline:
        print("\n" + "="*80)
        print("TRAINING BASELINE CNN MODEL")
        print("="*80)
        
        cnn_model = BaselineCNN(num_classes=2)
        print(f"\nCNN Model Parameters: {count_parameters(cnn_model):,}")
        
        cnn_trainer = Trainer(cnn_model, device=device,
                            learning_rate=args.learning_rate * 2,  # Higher LR for CNN
                            weight_decay=args.weight_decay)
        
        cnn_history = cnn_trainer.fit(
            train_loader, val_loader,
            epochs=args.epochs,
            save_path=os.path.join(args.results_dir, 'cnn_best_model.pth')
        )
        
        # Plot CNN training history
        plot_training_history(cnn_history,
                            os.path.join(args.results_dir, 'cnn_training_history.png'))
        
        # Evaluate CNN model
        print("\n" + "="*80)
        print("EVALUATING BASELINE CNN MODEL")
        print("="*80)
        
        cnn_evaluator = Evaluator(cnn_model, device=device)
        cnn_metrics = cnn_evaluator.evaluate(
            val_loader,
            save_dir=os.path.join(args.results_dir, 'cnn_evaluation')
        )
        
        # Compare models
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(f"\n{'Metric':<20} {'ViT':<15} {'CNN':<15} {'Difference':<15}")
        print("-" * 65)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            vit_val = vit_metrics[metric]
            cnn_val = cnn_metrics[metric]
            diff = vit_val - cnn_val
            print(f"{metric.upper():<20} {vit_val:<15.4f} {cnn_val:<15.4f} "
                  f"{diff:+.4f}")
        
        print("\n" + "="*80)
    
    print("\nTraining complete! Results saved to:", args.results_dir)

def analyze_dataset(image_paths, labels):
    """
    Analyze dataset distribution and basic statistics
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    print("\n" + "="*80)
    print("DATASET ANALYSIS")
    print("="*80)

    total_images = len(image_paths)
    label_counts = Counter(labels)

    print(f"Total Images: {total_images}")
    print(f"Healthy Images: {label_counts[0]}")
    print(f"Parkinson's Images: {label_counts[1]}")

    # Class balance
    healthy_pct = (label_counts[0] / total_images) * 100
    parkinsons_pct = (label_counts[1] / total_images) * 100

    print(f"\nClass Distribution:")
    print(f"Healthy: {healthy_pct:.2f}%")
    print(f"Parkinson's: {parkinsons_pct:.2f}%")

    # Image type distribution
    type_counts = {"spiral": 0, "wave": 0, "handwriting": 0}

    for path in image_paths:
        for t in type_counts:
            if t in path.lower():
                type_counts[t] += 1

    print("\nImage Type Distribution:")
    for k, v in type_counts.items():
        print(f"{k.capitalize()}: {v}")

    # Plot class distribution
    plt.figure()
    plt.bar(['Healthy', 'Parkinsons'], [label_counts[0], label_counts[1]])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.savefig("results/class_distribution.png")
    plt.close()

    print("\nSaved: results/class_distribution.png")

def main():
    parser = argparse.ArgumentParser(
        description='Train Parkinson\'s Disease Detection Models'
    )
    
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for regularization')
    parser.add_argument('--freeze_base', action='store_true',
                       help='Freeze base ViT layers (fine-tune only last layers)')
    parser.add_argument('--train_baseline', action='store_true',
                       help='Also train baseline CNN for comparison')
    
    args = parser.parse_args()
    
    train_model(args)


if __name__ == '__main__':
    main()
