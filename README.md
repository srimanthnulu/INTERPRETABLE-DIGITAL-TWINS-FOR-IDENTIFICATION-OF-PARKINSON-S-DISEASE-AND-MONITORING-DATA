# 🧠 Interpretable Digital Twin for Parkinson's Disease Detection

## Advanced AI System using Vision Transformer with Explainable AI

A comprehensive deep learning system for detecting Parkinson's Disease from spiral, wave, and handwriting images using state-of-the-art Vision Transformer (ViT) architecture with interpretability features and Digital Twin patient monitoring.

---

## 🌟 Features

### Core Capabilities
- **Vision Transformer Model**: Pre-trained ViT-Base with transfer learning
- **Interpretable AI**: Grad-CAM and attention heatmaps for explainable predictions
- **Digital Twin System**: Patient tracking and disease progression monitoring
- **Multi-Modal Analysis**: Supports spiral drawings, wave patterns, and handwriting
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC metrics
- **Baseline Comparison**: CNN baseline model for performance validation
- **Interactive Web App**: Gradio-based user interface for easy access

### Advanced Features
- Transfer learning with fine-tuning of last transformer layers
- Data augmentation for improved generalization
- AdamW optimizer with learning rate scheduling
- Attention rollout visualization
- Patient history tracking and progression analysis
- Automated report generation

---

## 📁 Project Structure

```
parkinsons_digital_twin/
│
├── app/
│   └── gradio_app.py           # Web application interface
│
├── models/
│   └── vit_model.py             # ViT and CNN model architectures
│
├── utils/
│   ├── data_preprocessing.py    # Data loading and augmentation
│   ├── training.py              # Training and evaluation utilities
│   ├── interpretability.py      # Grad-CAM and attention visualization
│   └── digital_twin.py          # Patient monitoring system
│
├── data/                        # Dataset directory (create this)
│   ├── healthy/
│   │   ├── spiral/
│   │   ├── wave/
│   │   └── handwriting/
│   └── parkinsons/
│       ├── spiral/
│       ├── wave/
│       └── handwriting/
│
├── results/                     # Training results and models
├── patient_records/             # Digital Twin patient data
│
├── train.py                     # Main training script
├── predict.py                   # Inference script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 🚀 Installation & Setup

### Step 1: Clone/Download the Project

Download the project folder to your local machine.

### Step 2: Install Dependencies

```bash
# Navigate to project directory
cd parkinsons_digital_twin

# Install required packages
pip install -r requirements.txt
```

**System Requirements:**
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB RAM minimum, 16GB recommended
- 10GB free disk space

### Step 3: Prepare Your Dataset

Organize your dataset in the following structure:

```
data/
├── healthy/
│   ├── spiral/      # Healthy spiral drawings
│   ├── wave/        # Healthy wave drawings
│   └── handwriting/ # Healthy handwriting samples
└── parkinsons/
    ├── spiral/      # Parkinson's spiral drawings
    ├── wave/        # Parkinson's wave drawings
    └── handwriting/ # Parkinson's handwriting samples
```

**Note**: If you don't have data yet, the training script will create sample synthetic data for demonstration purposes.

---

## 📖 Usage Guide

### 1. Training the Model

#### Basic Training (ViT only)

```bash
python train.py --data_dir data --epochs 20 --batch_size 16
```

#### Training with Baseline Comparison

```bash
python train.py \
    --data_dir data \
    --epochs 20 \
    --batch_size 16 \
    --train_baseline \
    --freeze_base
```

#### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_dir` | Path to dataset directory | `data` |
| `--results_dir` | Directory to save results | `results` |
| `--batch_size` | Batch size for training | `16` |
| `--epochs` | Number of training epochs | `20` |
| `--learning_rate` | Learning rate | `1e-4` |
| `--weight_decay` | Weight decay for regularization | `0.01` |
| `--freeze_base` | Freeze base ViT layers | `False` |
| `--train_baseline` | Also train CNN baseline | `False` |

#### Example Commands

```bash
# Quick training (CPU compatible)
python train.py --epochs 10 --batch_size 8

# Full training with comparison
python train.py --epochs 30 --train_baseline --freeze_base

# GPU training with larger batch
python train.py --epochs 25 --batch_size 32
```

**Training Output:**
- Best model saved to `results/vit_best_model.pth`
- Training history plots
- Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix and ROC curve

---

### 2. Single Image Prediction

Predict Parkinson's Disease from a single image:

```bash
python predict.py \
    --model results/vit_best_model.pth \
    --image path/to/your/image.png \
    --patient_id P001
```

**Output:**
- Prediction class (Healthy or Parkinson's)
- Confidence scores
- Probability distribution
- Grad-CAM explanation heatmap
- Digital Twin record (if patient_id provided)

---

### 3. Batch Prediction

Predict on multiple images at once:

```bash
python predict.py \
    --model results/vit_best_model.pth \
    --image_dir path/to/images/ \
    --output predictions.csv
```

**Output:**
- CSV file with all predictions
- Summary statistics

---

### 4. Launch Web Application

Start the interactive Gradio web interface:

```bash
python app/gradio_app.py
```


The application will launch at `http://localhost:7860`

**Web App Features:**
1. **Image Analysis Tab**: Upload and analyze medical images
2. **Patient Management Tab**: Create and manage patient profiles
3. **Digital Twin Monitoring Tab**: Track patient history and progression
4. **Generate Reports Tab**: Create comprehensive patient reports
5. **About Tab**: System information and documentation

---

## 🖥️ Web Application Guide

### Loading the Model

1. Enter model path (default: `results/vit_best_model.pth`)
2. Click "Load Model" button
3. Wait for confirmation message

### Analyzing an Image

1. Upload image (spiral, wave, or handwriting)
2. Select image type from radio buttons
3. (Optional) Enter patient ID to track in Digital Twin
4. Click "🔍 Analyze Image"
5. View results:
   - Prediction and confidence scores
   - Probability distribution chart
   - AI explanation heatmap showing important regions

### Creating a Patient Profile

1. Go to "👤 Patient Management" tab
2. Fill in patient details:
   - Patient ID (required)
   - Full Name (required)
   - Age
   - Gender
   - Medical History
3. Click "Create Patient Profile"

### Viewing Patient History

1. Go to "📊 Digital Twin Monitoring" tab
2. Enter Patient ID
3. Click "View Patient History"
4. Review:
   - Patient information
   - Testing summary
   - Disease progression chart
   - Complete test history table

### Generating Reports

1. Go to "📄 Generate Reports" tab
2. Enter Patient ID
3. Click "Generate Report"
4. View comprehensive patient report with all historical data

---

## 📊 Model Architecture

### Vision Transformer (ViT)

- **Base Model**: google/vit-base-patch16-224
- **Input Size**: 224×224 RGB images
- **Patch Size**: 16×16
- **Embedding Dimension**: 768
- **Transformer Layers**: 12
- **Attention Heads**: 12
- **Parameters**: ~86M (trainable varies based on fine-tuning strategy)

### Fine-Tuning Strategy

1. Load pre-trained ViT-Base weights
2. Optionally freeze base layers
3. Fine-tune last 2 transformer layers
4. Add custom classification head:
   - LayerNorm
   - Dropout (0.3)
   - Linear (768 → 512)
   - GELU activation
   - Dropout (0.2)
   - Linear (512 → 2)

### Baseline CNN

- 3 convolutional blocks with batch normalization
- Max pooling for spatial reduction
- Adaptive average pooling
- Fully connected classification head
- Parameters: ~15M

---

## 🔍 Interpretability Features

### Grad-CAM (Gradient-weighted Class Activation Mapping)

Highlights regions that most influence the model's prediction:

```python
from utils.interpretability import ViTGradCAM, visualize_cam

grad_cam = ViTGradCAM(model)
cam = grad_cam.generate_cam(image_tensor, target_class=1)
visualize_cam(image_path, cam, save_path='explanation.png')
```

### Attention Visualization

Shows where the transformer pays attention:

```python
attention_map = model.get_attention_map(image_tensor, layer_idx=-1)
```

### Attention Rollout

Traces attention through all transformer layers:

```python
from utils.interpretability import generate_attention_rollout

rollout = generate_attention_rollout(attentions, head_fusion='mean')
```

---

## 🏥 Digital Twin System

### Patient Profile Management

```python
from utils.digital_twin import DigitalTwin

twin = DigitalTwin()

# Create patient
profile = twin.create_patient(
    patient_id="P001",
    name="John Doe",
    age=65,
    gender="Male",
    medical_history="No prior neurological conditions"
)

# Add prediction
twin.add_prediction(
    patient_id="P001",
    image_type="spiral",
    prediction=1,
    probability=0.87,
    attention_map_path="explanation.png"
)

# Get history
history = twin.get_patient_history("P001")

# Analyze progression
stats = twin.analyze_progression("P001", save_path="progression.png")

# Generate report
report = twin.generate_report("P001", save_path="report.txt")
```

---

## 📈 Evaluation Metrics

The system provides comprehensive evaluation:

### Classification Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

### Visualizations

- Confusion Matrix
- ROC Curve
- Training History (Loss and Accuracy)
- Disease Progression Charts

---

## 🎯 Best Practices

### Data Collection

1. **Image Quality**: Ensure clear, high-resolution images
2. **Consistency**: Use consistent lighting and background
3. **Balance**: Maintain balanced classes (equal healthy and PD samples)
4. **Diversity**: Include various severity levels and patient demographics

### Model Training

1. **Start with frozen base**: Use `--freeze_base` for faster initial training
2. **Fine-tune incrementally**: Gradually unfreeze more layers
3. **Monitor overfitting**: Watch validation loss
4. **Use data augmentation**: Helps with limited data
5. **Save checkpoints**: Regularly save best models

### Clinical Usage

1. **Professional oversight**: Always involve medical professionals
2. **Multiple tests**: Use multiple test types (spiral, wave, handwriting)
3. **Temporal tracking**: Monitor progression over time
4. **Context matters**: Consider patient history and symptoms
5. **Interpretability**: Review attention maps for clinical relevance

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size
python train.py --batch_size 8
```

**Issue**: Model not found
```bash
# Solution: Check model path
python predict.py --model results/vit_best_model.pth --image test.png
```

**Issue**: Gradio app won't start
```bash
# Solution: Check port availability or change port
python app/gradio_app.py
# Or modify server_port in gradio_app.py
```

**Issue**: No images found during training
```bash
# Solution: Check data directory structure
ls -R data/
# Ensure proper folder organization
```

---

## 📚 References

### Key Papers

1. **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021
2. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
3. **Parkinson's Detection**: Various studies on motor function analysis

### Datasets

- Parkinson's Disease Spiral Drawings (UCI Machine Learning Repository)
- NewHandPD Dataset
- Custom clinical datasets (with proper authorization)

---

## ⚠️ Important Disclaimers

1. **Research Use Only**: This system is intended for research and educational purposes
2. **Not a Medical Device**: Does not replace professional medical diagnosis
3. **Clinical Validation Required**: Requires extensive clinical validation before medical use
4. **Privacy Compliance**: Ensure compliance with HIPAA/GDPR for patient data
5. **Informed Consent**: Obtain proper consent for data collection and analysis

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional interpretability methods
- Multi-task learning (severity prediction)
- Ensemble methods
- Real-time video analysis
- Mobile deployment
- Integration with EHR systems

---

## 📧 Support

For questions or issues:
1. Check this README first
2. Review code comments and docstrings
3. Examine example outputs in results/
4. Contact your system administrator

---

## 📄 License

This project is for educational and research purposes. Ensure proper licensing for clinical use.

---

## 🎓 Citation

If you use this system in your research, please cite:

```bibtex
@software{parkinsons_digital_twin,
  title={Interpretable Digital Twin for Parkinson's Disease Detection},
  author={Your Name},
  year={2024},
  description={Vision Transformer-based system with Grad-CAM interpretability}
}
```

---

**Built with ❤️ for advancing medical AI research**

Last Updated: February 2026
