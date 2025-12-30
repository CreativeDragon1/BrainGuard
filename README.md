# BrainGuard — MRI-Based Alzheimer's Detection

Flask + PyTorch web application for automated dementia severity classification from brain MRI scans. Built for the Hack4Health AI for Alzheimer's challenge with transfer learning, comprehensive validation, and Grad-CAM explainability.

##  Overview

BrainGuard classifies brain MRI images into **four dementia severity levels**: Non-Demented, Very Mild, Mild, and Moderate Dementia. The system combines high-accuracy deep learning models with clinical interpretability through visual saliency maps, making predictions transparent and trustworthy for research applications.

##  Features

- **Flask Web UI** — Drag-and-drop MRI upload with real-time predictions and Grad-CAM overlays
- **Dual Model Architecture** — ResNet50 (transfer learning, high accuracy) and Custom CNN (fast, efficient)
- **Flexible Input Handling** — PNG/JPG images and 3D NIfTI files (`.nii`, `.nii.gz`) with automatic preprocessing
- **Multi-Class Probability Breakdown** — Detailed confidence scores across all four dementia classes
- **Grad-CAM Visualization** — Highlights influential brain regions driving model predictions
- **Complete Training Pipeline** — Full validation loop with checkpointing and comprehensive metrics (accuracy, precision, recall, F1, confusion matrices)
- **Comprehensive Documentation** — Technical report (2-3 pages), model card, and Colab-ready notebooks

##  Project Structure

```
BrainGuard/
├── app/
│   └── app.py                    # Flask server & prediction endpoints
├── models/
│   ├── cnn_model.py              # ResNet50 & Custom CNN architectures
│   ├── preprocessing.py          # Image normalization & augmentation
│   ├── grad_cam.py               # Saliency map generation
│   ├── best_resnet.pth           # Pre-trained ResNet50 weights
│   └── __init__.py
├── templates/
│   └── index.html                # Web UI layout
├── static/
│   ├── css/style.css             # Styling
│   └── js/script.js              # Client-side logic
├── notebooks/
│   └── alz_mri_classification.ipynb  # Training & EDA notebook
├── Assets/Datasets/              # Reference datasets
├── train.py                       # Training script
├── train_colab.ipynb             # Google Colab training notebook
├── verify_setup.py               # Environment verification
├── BrainGuard_Report.pdf         # Technical report (problem, methods, evaluation)
├── MODEL_CARD.md                 # Model details & limitations
├── requirements.txt              # Dependencies
└── README.md                      # This file
```

##  Quick Start

### 1. Installation

```bash
# Clone/download repository
cd BrainGuard

# Create Python 3.10+ environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
python app/app.py
```

Then open your browser to **http://localhost:5000**

- Upload an MRI image (PNG, JPG, or NIfTI format)
- Select model: **ResNet50** (better accuracy) or **Custom CNN** (faster inference)
- View predictions, confidence scores, class probabilities, and Grad-CAM overlay

### 3. Train Your Own Model

```bash
python train.py --model resnet --epochs 30 --batch-size 32 --lr 1e-3
```

Or use the **Google Colab notebook** for cloud training:
- Open `train_colab.ipynb` in Colab
- Includes dataset loading, augmentation, and visualization

## 📊 Model Details

### ResNet50 Transfer Learning
- Pre-trained on ImageNet weights
- First conv layer adapted for single-channel (grayscale) MRI input
- Excellent accuracy on medical imaging tasks

### Custom CNN
- 4-block architecture: 32→64→128→256 filters
- Batch normalization, ReLU activations, dropout (rate=0.5)
- Lightweight alternative for resource-constrained environments

### Data Preprocessing
- Grayscale conversion
- Resize to 224×224 pixels
- Normalization (ImageNet stats: mean=0.485, std=0.229)
- Augmentation: rotation (±15°), horizontal flips, color jitter

### Training Configuration
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Batch size: 32 | Epochs: up to 50

## 🔍 Explainability

**Grad-CAM (Gradient-weighted Class Activation Mapping)** generates saliency maps showing which brain regions most influenced each prediction. This enables:
- Verification that models focus on clinically relevant areas
- Detection of anatomical biomarkers (e.g., hippocampal atrophy)
- Trust and interpretability for research applications

## 📋 Key Files

| File | Purpose |
|------|---------|
| `BrainGuard_Report.pdf` | Technical report: problem framing, methods, evaluation (2-3 pages) |
| `MODEL_CARD.md` | Model specifications, data, training procedure, ethical considerations |
| `train.py` | Full training pipeline with validation & checkpointing |
| `app/app.py` | Flask API for inference & visualization |
| `models/grad_cam.py` | Saliency map generation & visualization |

## ✅ Submission Checklist

- ✅ Runnable web application (`app/app.py`)
- ✅ Training scripts (`train.py`, `train_colab.ipynb`)
- ✅ PDF technical report (2-3 pages) — `BrainGuard_Report.pdf`
- ✅ Model card with details & limitations — `MODEL_CARD.md`
- ✅ Complete documentation

## 🔧 Troubleshooting

**Issue**: Import errors for `nibabel` or `torch`  
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: CUDA/GPU not detected  
**Solution**: The app automatically falls back to CPU. For GPU support, ensure CUDA Toolkit 11.8+ is installed.

**Issue**: Port 5000 already in use  
**Solution**: Modify `app.py` line 1 to use a different port, e.g., `app.run(port=5001)`

## 📚 Documentation

- See `BrainGuard_Report.pdf` for technical details on methodology and evaluation
- See `MODEL_CARD.md` for data sources, preprocessing, metrics, and limitations
- Check `QUICK_START.md` and `SETUP_AND_TEST.md` for additional setup guidance

## 🎓 References

- **Dataset**: Kaggle Alzheimer MRI Disease Classification Dataset
- **Framework**: PyTorch
- **Visualization**: scikit-learn, NumPy, Pillow, Matplotlib
