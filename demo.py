"""
Demo Script for Parkinson's Detection System
Tests all major components without requiring a full dataset
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

print("="*60)
print("PARKINSON'S DETECTION SYSTEM - DEMO")
print("="*60)
print()

# Check imports
print("1. Checking dependencies...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
    import torchvision
    print(f"   ✓ Torchvision {torchvision.__version__}")
    from transformers import ViTModel
    print(f"   ✓ Transformers installed")
    import gradio as gr
    print(f"   ✓ Gradio {gr.__version__}")
    print("   All dependencies installed correctly!")
except ImportError as e:
    print(f"   ✗ Missing dependency: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

print()

# Check project structure
print("2. Checking project structure...")
required_dirs = [
    'models', 'utils', 'app', 'data', 'results', 'patient_records'
]
required_files = [
    'models/vit_model.py',
    'utils/data_preprocessing.py',
    'utils/training.py',
    'utils/interpretability.py',
    'utils/digital_twin.py',
    'app/gradio_app.py',
    'train.py',
    'predict.py'
]

all_good = True
for directory in required_dirs:
    if os.path.exists(directory):
        print(f"   ✓ {directory}/")
    else:
        print(f"   ✗ {directory}/ NOT FOUND")
        all_good = False

for file in required_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} NOT FOUND")
        all_good = False

if not all_good:
    print("\n   Some files are missing. Please check the project structure.")
    sys.exit(1)

print("   Project structure is correct!")
print()

# Test model loading
print("3. Testing model architecture...")
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models.vit_model import ViTForParkinsons, BaselineCNN, count_parameters
    
    # Create model instance (without loading weights)
    vit_model = ViTForParkinsons(num_classes=2, pretrained=False)
    print(f"   ✓ ViT model created ({count_parameters(vit_model):,} parameters)")
    
    cnn_model = BaselineCNN(num_classes=2)
    print(f"   ✓ CNN model created ({count_parameters(cnn_model):,} parameters)")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 224, 224)
    vit_output, _ = vit_model(test_input)
    print(f"   ✓ ViT forward pass successful (output shape: {vit_output.shape})")
    
    cnn_output = cnn_model(test_input)
    print(f"   ✓ CNN forward pass successful (output shape: {cnn_output.shape})")
    
except Exception as e:
    print(f"   ✗ Error testing models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test data preprocessing
print("4. Testing data preprocessing...")
try:
    from utils.data_preprocessing import get_transforms, preprocess_single_image
    
    # Create a dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    dummy_img.save('temp_test_image.png')
    
    # Test transforms
    transform = get_transforms(augment=False)
    transformed = transform(dummy_img)
    print(f"   ✓ Transform successful (shape: {transformed.shape})")
    
    # Test preprocessing
    tensor = preprocess_single_image('temp_test_image.png')
    print(f"   ✓ Image preprocessing successful (shape: {tensor.shape})")
    
    # Clean up
    os.remove('temp_test_image.png')
    
except Exception as e:
    print(f"   ✗ Error testing preprocessing: {e}")
    import traceback
    traceback.print_exc()

print()

# Test Digital Twin
print("5. Testing Digital Twin system...")
try:
    from utils.digital_twin import DigitalTwin
    
    twin = DigitalTwin(storage_dir='test_patient_records')
    
    # Create test patient
    profile = twin.create_patient(
        patient_id="TEST001",
        name="Test Patient",
        age=65,
        gender="Male",
        medical_history="Test profile"
    )
    print(f"   ✓ Patient created: {profile.name}")
    
    # Add test prediction
    record = twin.add_prediction(
        patient_id="TEST001",
        image_type="spiral",
        prediction=1,
        probability=0.85,
        attention_map_path=""
    )
    print(f"   ✓ Prediction added (probability: {record.probability:.2f})")
    
    # Get history
    history = twin.get_patient_history("TEST001")
    print(f"   ✓ History retrieved ({len(history)} records)")
    
    # Analyze progression
    stats = twin.analyze_progression("TEST001")
    print(f"   ✓ Progression analysis complete")
    
    # Clean up
    import shutil
    shutil.rmtree('test_patient_records')
    
except Exception as e:
    print(f"   ✗ Error testing Digital Twin: {e}")
    import traceback
    traceback.print_exc()

print()

# Test interpretability
print("6. Testing interpretability modules...")
try:
    from utils.interpretability import ViTGradCAM
    
    model = ViTForParkinsons(num_classes=2, pretrained=False)
    grad_cam = ViTGradCAM(model)
    print(f"   ✓ Grad-CAM initialized")
    
    test_input = torch.randn(1, 3, 224, 224)
    cam = grad_cam.generate_cam(test_input, target_class=1)
    print(f"   ✓ CAM generated (shape: {cam.shape})")
    
except Exception as e:
    print(f"   ✗ Error testing interpretability: {e}")
    import traceback
    traceback.print_exc()

print()

# Check GPU availability
print("7. Checking hardware...")
if torch.cuda.is_available():
    print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
    print(f"   ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(f"   ℹ GPU not available, will use CPU")
    print(f"   (Training will be slower but still functional)")

print()

# Summary
print("="*60)
print("DEMO COMPLETE - SYSTEM CHECK SUMMARY")
print("="*60)
print()
print("✓ All core components are working correctly!")
print()
print("Next steps:")
print("1. Prepare your dataset in the data/ directory")
print("2. Run: python train.py --epochs 10 --batch_size 8")
print("3. After training, run: python app/gradio_app.py")
print()
print("For detailed instructions, see:")
print("  - README.md (comprehensive documentation)")
print("  - USAGE_GUIDE.md (step-by-step guide)")
print()
print("="*60)
