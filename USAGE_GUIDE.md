# 📂 FILE ORGANIZATION AND USAGE GUIDE

## Complete Directory Structure

```
parkinsons_digital_twin/          # Main project directory (save everything here)
│
├── 📁 app/                        # Web application files
│   ├── __init__.py
│   └── gradio_app.py             # Main web interface (RUN THIS for web app)
│
├── 📁 models/                     # Model architecture files
│   ├── __init__.py
│   └── vit_model.py              # ViT and CNN models
│
├── 📁 utils/                      # Utility modules
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data loading and augmentation
│   ├── training.py               # Training utilities
│   ├── interpretability.py       # Grad-CAM and visualization
│   └── digital_twin.py           # Patient monitoring system
│
├── 📁 data/                       # YOUR DATASET GOES HERE
│   ├── healthy/
│   │   ├── spiral/               # Put healthy spiral images here
│   │   ├── wave/                 # Put healthy wave images here
│   │   └── handwriting/          # Put healthy handwriting images here
│   └── parkinsons/
│       ├── spiral/               # Put Parkinson's spiral images here
│       ├── wave/                 # Put Parkinson's wave images here
│       └── handwriting/          # Put Parkinson's handwriting images here
│
├── 📁 results/                    # Training outputs (created automatically)
│   ├── vit_best_model.pth        # Trained ViT model (generated after training)
│   ├── cnn_best_model.pth        # Trained CNN model (if baseline used)
│   ├── vit_training_history.png  # Training plots
│   ├── vit_evaluation/           # ViT evaluation results
│   │   ├── metrics.json
│   │   ├── confusion_matrix.png
│   │   └── roc_curve.png
│   └── inference/                # Inference visualizations
│       └── *_explanation.png     # Grad-CAM heatmaps
│
├── 📁 patient_records/            # Digital Twin data (created automatically)
│   └── [patient_id]/             # Each patient gets a folder
│       ├── profile.json          # Patient information
│       └── predictions.json      # Prediction history
│
├── 📄 train.py                    # RUN THIS to train models
├── 📄 predict.py                  # RUN THIS for single predictions
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # Main documentation
├── 📄 setup.sh                    # Setup script (Linux/Mac)
└── 📄 USAGE_GUIDE.md             # This file
```

---

## 🎯 WHERE TO SAVE EACH FILE

### Core Python Files (Already Created)
These files form the main codebase. Save them exactly as shown in the structure above.

**Location**: `parkinsons_digital_twin/`

1. **train.py** → Root directory
2. **predict.py** → Root directory
3. **requirements.txt** → Root directory
4. **README.md** → Root directory
5. **setup.sh** → Root directory

### Model Files
**Location**: `parkinsons_digital_twin/models/`

1. **vit_model.py** → Contains ViT and CNN architectures
2. **__init__.py** → Python package initialization

### Utility Files
**Location**: `parkinsons_digital_twin/utils/`

1. **data_preprocessing.py** → Data loading and augmentation
2. **training.py** → Training loop and evaluation
3. **interpretability.py** → Grad-CAM and attention visualization
4. **digital_twin.py** → Patient monitoring system
5. **__init__.py** → Python package initialization

### Application Files
**Location**: `parkinsons_digital_twin/app/`

1. **gradio_app.py** → Web interface
2. **__init__.py** → Python package initialization

### Your Dataset (You Need to Provide)
**Location**: `parkinsons_digital_twin/data/`

Organize your images like this:
```
data/
├── healthy/
│   ├── spiral/          # Healthy spiral drawings (.png, .jpg)
│   ├── wave/            # Healthy wave drawings
│   └── handwriting/     # Healthy handwriting samples
└── parkinsons/
    ├── spiral/          # PD spiral drawings
    ├── wave/            # PD wave drawings
    └── handwriting/     # PD handwriting samples
```

**Example file names:**
- `healthy_spiral_001.png`
- `healthy_wave_045.jpg`
- `pd_spiral_023.png`
- `pd_handwriting_012.jpg`

### Logo Image (Optional)
If you have a logo image:
**Location**: `parkinsons_digital_twin/app/logo.png`

Then modify `gradio_app.py` to include:
```python
gr.Image("logo.png", label="System Logo")
```

---

## 🚀 HOW TO RUN EACH FILE

### 1️⃣ Initial Setup (Do This First!)

**Option A: Using Setup Script (Linux/Mac)**
```bash
cd parkinsons_digital_twin
chmod +x setup.sh
./setup.sh
```

**Option B: Manual Setup**
```bash
cd parkinsons_digital_twin

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/healthy/spiral data/healthy/wave data/healthy/handwriting
mkdir -p data/parkinsons/spiral data/parkinsons/wave data/parkinsons/handwriting
mkdir -p results patient_records
```

---

### 2️⃣ Training the Model

**Basic Training:**
```bash
python train.py --data_dir data --epochs 20 --batch_size 16
```

**What it does:**
- Loads images from `data/` directory
- Trains Vision Transformer model
- Saves best model to `results/vit_best_model.pth`
- Creates evaluation plots in `results/`

**Training with CNN Baseline:**
```bash
python train.py --data_dir data --epochs 20 --train_baseline
```

**Advanced Options:**
```bash
# Train with frozen base (faster, less GPU memory)
python train.py --epochs 15 --freeze_base

# Train with smaller batch (if GPU memory limited)
python train.py --epochs 20 --batch_size 8

# Full training with all features
python train.py --epochs 30 --batch_size 32 --train_baseline
```

**Expected Output Files:**
- `results/vit_best_model.pth` ← Main trained model
- `results/vit_training_history.png` ← Training curves
- `results/vit_evaluation/` ← Evaluation metrics
- `results/cnn_best_model.pth` ← Baseline model (if used)

---

### 3️⃣ Making Predictions

**Single Image Prediction:**
```bash
python predict.py \
    --model results/vit_best_model.pth \
    --image path/to/test_image.png \
    --patient_id P001
```

**What it does:**
- Loads the trained model
- Analyzes the image
- Prints prediction results
- Saves explanation heatmap to `results/inference/`
- Records in Digital Twin (if patient_id provided)

**Batch Prediction (Multiple Images):**
```bash
python predict.py \
    --model results/vit_best_model.pth \
    --image_dir path/to/images_folder/ \
    --output predictions.csv
```

**Expected Output Files:**
- `results/inference/[image_name]_explanation.png` ← Grad-CAM heatmap
- `predictions.csv` ← Results table (for batch mode)
- `patient_records/[patient_id]/predictions.json` ← Digital Twin record

---

### 4️⃣ Running the Web Application

**Start the App:**
```bash
python app/gradio_app.py
```

**What happens:**
1. Server starts on `http://localhost:7860`
2. Browser opens automatically (usually)
3. Web interface loads with 5 tabs

**If server doesn't start:**
- Check if port 7860 is available
- Try different port: Edit `gradio_app.py`, change `server_port=7860` to `server_port=8080`

**Access from other devices:**
The script also creates a public URL (like `https://xxxxx.gradio.live`) that you can share.

**Expected Behavior:**
- Web interface opens in browser
- You can upload images and get predictions
- Create patient profiles
- View patient history
- Generate reports

---

## 📋 STEP-BY-STEP WORKFLOW

### Complete Workflow from Start to Finish:

#### Step 1: Setup Environment
```bash
# Navigate to project
cd parkinsons_digital_twin

# Install packages
pip install -r requirements.txt

# Create directories
mkdir -p data/healthy/{spiral,wave,handwriting}
mkdir -p data/parkinsons/{spiral,wave,handwriting}
```

#### Step 2: Prepare Dataset
```bash
# Copy your images to appropriate folders
cp ~/my_images/healthy_spirals/* data/healthy/spiral/
cp ~/my_images/pd_spirals/* data/parkinsons/spiral/
# ... repeat for wave and handwriting
```

#### Step 3: Train Model
```bash
# Start training
python train.py --epochs 20 --batch_size 16 --train_baseline

# Wait for training to complete (may take 1-3 hours depending on data size)
```

#### Step 4: Verify Training Results
```bash
# Check if model was created
ls -lh results/vit_best_model.pth

# View training plots
# Open: results/vit_training_history.png
# Open: results/vit_evaluation/confusion_matrix.png
```

#### Step 5: Test Single Prediction
```bash
# Test on a sample image
python predict.py \
    --model results/vit_best_model.pth \
    --image data/healthy/spiral/test_image.png

# Check the output and explanation heatmap
# Open: results/inference/test_image_explanation.png
```

#### Step 6: Launch Web App
```bash
# Start the application
python app/gradio_app.py

# Browser should open to http://localhost:7860
```

#### Step 7: Use Web Interface
1. Load model in "Image Analysis" tab
2. Create patient profile in "Patient Management" tab
3. Upload and analyze images
4. View patient history in "Digital Twin Monitoring" tab
5. Generate comprehensive reports

---

## 🔧 CONFIGURATION OPTIONS

### Model Configuration

Edit `train.py` to change:
```python
# Line ~100
parser.add_argument('--batch_size', type=int, default=16)  # Change batch size
parser.add_argument('--epochs', type=int, default=20)      # Change epochs
parser.add_argument('--learning_rate', type=float, default=1e-4)  # Change LR
```

### Web App Configuration

Edit `app/gradio_app.py` to change:
```python
# Line ~550
demo.launch(
    server_name="0.0.0.0",    # Change to "127.0.0.1" for localhost only
    server_port=7860,          # Change port number
    share=True,                # Set to False to disable public URL
)
```

### Data Augmentation

Edit `utils/data_preprocessing.py` to modify augmentation:
```python
# Line ~55
transforms.RandomRotation(degrees=15),        # Change rotation
transforms.ColorJitter(brightness=0.2),       # Change brightness
```

---

## 📊 EXPECTED FILE SIZES

After setup and training, approximate file sizes:

```
parkinsons_digital_twin/
├── models/vit_model.py              ~15 KB
├── utils/                           ~50 KB total
├── app/gradio_app.py                ~25 KB
├── train.py                         ~12 KB
├── predict.py                       ~8 KB
├── requirements.txt                 ~1 KB
├── README.md                        ~30 KB
│
├── data/                            Variable (depends on your images)
│   └── ...                          100 MB - 10 GB typical
│
└── results/
    ├── vit_best_model.pth           ~330 MB (ViT model)
    ├── cnn_best_model.pth           ~60 MB (CNN baseline)
    └── evaluation/                   ~5 MB (plots and metrics)
```

---

## 🐛 TROUBLESHOOTING

### Problem: "No module named 'transformers'"
**Solution:**
```bash
pip install transformers
```

### Problem: "CUDA out of memory"
**Solution:**
```bash
# Use smaller batch size
python train.py --batch_size 8
```

### Problem: "No images found"
**Solution:**
```bash
# Check data directory structure
ls -R data/

# Ensure images have correct extensions
# Supported: .png, .jpg, .jpeg
```

### Problem: "Model file not found"
**Solution:**
```bash
# Check if model exists
ls results/vit_best_model.pth

# If not, train first
python train.py --epochs 10
```

### Problem: "Gradio app won't start"
**Solution:**
```bash
# Check if gradio is installed
pip install gradio

# Try different port
# Edit gradio_app.py, change server_port=7860 to server_port=8080
```

---

## 📞 QUICK REFERENCE

### Essential Commands

```bash
# Setup
pip install -r requirements.txt

# Train model
python train.py --epochs 20

# Predict single image
python predict.py --model results/vit_best_model.pth --image test.png

# Launch web app
python app/gradio_app.py
```

### Important Paths

- **Trained Model**: `results/vit_best_model.pth`
- **Training Data**: `data/`
- **Patient Records**: `patient_records/`
- **Visualizations**: `results/inference/`
- **Web App**: `app/gradio_app.py`

---

## ✅ CHECKLIST

Before running the system:
- [ ] All Python files are in correct directories
- [ ] requirements.txt installed (`pip install -r requirements.txt`)
- [ ] Data organized in `data/healthy/` and `data/parkinsons/`
- [ ] Directories created (`results/`, `patient_records/`)
- [ ] Python 3.8+ installed
- [ ] (Optional) CUDA/GPU drivers installed

To train model:
- [ ] Dataset prepared in `data/` directory
- [ ] Run `python train.py`
- [ ] Wait for training completion
- [ ] Verify `results/vit_best_model.pth` exists

To use web app:
- [ ] Model trained (or pre-trained model available)
- [ ] Run `python app/gradio_app.py`
- [ ] Browser opens to localhost:7860
- [ ] Load model in web interface

---

**Need more help?** Check the main README.md file or review the code comments!

**Last Updated:** February 2026
