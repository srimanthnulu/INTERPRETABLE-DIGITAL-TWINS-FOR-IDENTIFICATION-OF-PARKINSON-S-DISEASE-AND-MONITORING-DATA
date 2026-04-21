"""
Inference Script for Single Image Prediction
Provides predictions with interpretability visualizations
"""

import torch
import os
import sys
import argparse
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vit_model import ViTForParkinsons
from utils.data_preprocessing import preprocess_single_image
from utils.interpretability import ViTGradCAM, visualize_cam
from utils.digital_twin import DigitalTwin


def predict_single_image(model_path: str, image_path: str, 
                         patient_id: str = None, save_results: bool = True):
    """
    Predict Parkinson's disease from a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        patient_id: Optional patient ID for digital twin tracking
        save_results: Whether to save visualization results
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ViTForParkinsons(num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\nModel loaded from: {model_path}")
    print(f"Analyzing image: {image_path}")
    
    # Preprocess image
    image_tensor = preprocess_single_image(image_path).to(device)
    
    # Get prediction
    with torch.no_grad():
        logits, attentions = model(image_tensor, output_attentions=True)
        probabilities = torch.softmax(logits, dim=1)
        prediction = logits.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item()
        pd_probability = probabilities[0, 1].item()
    
    # Get class name
    class_names = ['Healthy', 'Parkinson\'s Disease']
    predicted_class = class_names[prediction]
    
    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Class:      {predicted_class}")
    print(f"Confidence:           {confidence:.2%}")
    print(f"PD Probability:       {pd_probability:.2%}")
    print(f"Healthy Probability:  {probabilities[0, 0].item():.2%}")
    print("="*60 + "\n")
    
    # Generate interpretability visualizations
    if save_results:
        results_dir = 'results/inference'
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate Grad-CAM
        print("Generating interpretability visualizations...")
        grad_cam = ViTGradCAM(model)
        cam = grad_cam.generate_cam(image_tensor, target_class=1)
        
        # Get image name
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save visualization
        viz_path = os.path.join(results_dir, f'{image_name}_explanation.png')
        visualize_cam(image_path, cam, save_path=viz_path)
        
        print(f"Visualization saved to: {viz_path}")
        
        # Get attention map
        attention_map = model.get_attention_map(image_tensor, layer_idx=-1)
        
        # Digital Twin integration
        if patient_id:
            print(f"\nUpdating Digital Twin for patient: {patient_id}")
            twin = DigitalTwin()
            
            # Determine image type from path
            image_type = 'unknown'
            if 'spiral' in image_path.lower():
                image_type = 'spiral'
            elif 'wave' in image_path.lower():
                image_type = 'wave'
            elif 'handwriting' in image_path.lower():
                image_type = 'handwriting'
            
            # Add prediction
            twin.add_prediction(
                patient_id=patient_id,
                image_type=image_type,
                prediction=prediction,
                probability=pd_probability,
                attention_map_path=viz_path
            )
            
            print(f"Prediction recorded in Digital Twin")
    
    return {
        'prediction': prediction,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'pd_probability': pd_probability,
        'probabilities': probabilities[0].cpu().numpy()
    }


def batch_predict(model_path: str, image_dir: str, output_file: str = None):
    """
    Predict on multiple images
    
    Args:
        model_path: Path to trained model
        image_dir: Directory containing images
        output_file: Optional CSV file to save results
    """
    import pandas as pd
    
    # Get all images
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images")
    
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        
        try:
            result = predict_single_image(
                model_path, img_path, 
                save_results=False
            )
            
            results.append({
                'image': img_file,
                'prediction': result['predicted_class'],
                'confidence': result['confidence'],
                'pd_probability': result['pd_probability']
            })
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("BATCH PREDICTION SUMMARY")
    print("="*60)
    print(df)
    print("="*60)
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Predict Parkinson\'s Disease from images'
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image', type=str,
                       help='Path to single image')
    parser.add_argument('--image_dir', type=str,
                       help='Directory of images for batch prediction')
    parser.add_argument('--patient_id', type=str,
                       help='Patient ID for Digital Twin tracking')
    parser.add_argument('--output', type=str,
                       help='Output CSV file for batch predictions')
    
    args = parser.parse_args()
    
    if args.image:
        predict_single_image(
            args.model, 
            args.image,
            patient_id=args.patient_id
        )
    elif args.image_dir:
        batch_predict(args.model, args.image_dir, args.output)
    else:
        print("Error: Please provide either --image or --image_dir")


if __name__ == '__main__':
    main()
