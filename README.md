# BrainGuard — MRI-Based Alzheimer's Detection

Flask + PyTorch web application for automated dementia severity classification from brain MRI scans. Built for the Hack4Health AI for Alzheimer's challenge featuring ResNet50 transfer learning and Grad-CAM explainability.

## 🎯 Overview

BrainGuard classifies brain MRI images into **four dementia severity levels**: Non-Demented, Very Mild, Mild, and Moderate Dementia. The system uses ResNet50 transfer learning with Grad-CAM visual explanations, making AI predictions transparent and clinically interpretable.

## ✨ Features

- **Flask Web Interface** — Drag-and-drop MRI upload with real-time predictions
- **ResNet50 Transfer Learning** — High-accuracy ImageNet-pretrained model adapted for medical imaging
- **Flexible Input Formats** — PNG/JPG images and 3D NIfTI files (`.nii`, `.nii.gz`) with automatic slice extraction
- **Probability Breakdown** — Detailed confidence scores across all four dementia classes
- **Grad-CAM Visualization** — Highlights influential brain regions driving each prediction
- **Complete Pipeline** — Training scripts, validation metrics, and comprehensive documentation

## 📁 Project Structure

```
BrainGuard/
├── app/
│   └── app.py                    # Flask server & prediction API
├── models/
│   ├── cnn_model.py              # ResNet50 architecture
│   ├── preprocessing.py          # Image normalization & augmentation
│   ├── grad_cam.py               # Saliency map generation
│   └── best_resnet.pth           # Pre-trained model weights
├── templates/
│   └── index.html                # Web UI
├── static/
│   ├── css/style.css             # Styling
│   ├── js/script.js              # Client logic
│   └── uploads/                  # Uploaded MRI storage
├── notebooks/
│   └── alz_mri_classification.ipynb  # Training notebook
├── train.py                       # Training script
├── train_colab.ipynb             # Google Colab notebook
├── requirements.txt              # Python dependencies
├── MODEL_CARD.md                 # Model documentation
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to project directory
cd BrainGuard

# Create virtual environment (recommended)
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note for Windows users:** You may need to install Visual C++ Redistributable:
```powershell
winget install --id=Microsoft.VCRedist.2015+.x64 -e
```

### 2. Run the Application

```bash
python app/app.py
```

Then open your browser to **http://localhost:8000**

### 3. Usage

1. Upload an MRI scan (PNG, JPG, or NIfTI format)
2. Click **Analyze** to run the prediction
3. View:
   - Predicted dementia severity class
   - Confidence score
   - Probability distribution across all classes
   - Grad-CAM attention heatmap showing influential brain regions

## 🧠 Model Details

### ResNet50 Transfer Learning
- **Architecture**: ImageNet-pretrained ResNet50 with adapted first convolutional layer for single-channel (grayscale) MRI input
- **Classes**: 4 (Non-Demented, Very Mild, Mild, Moderate)
- **Input**: 224×224 grayscale images
- **Performance**: High accuracy with strong feature representations

### Data Preprocessing
1. **Grayscale conversion** for single-channel input
2. **Resize** to 224×224 pixels
3. **Normalization** using ImageNet statistics (mean=0.485, std=0.229)
4. **NIfTI handling**: Middle axial slice extraction from 3D volumes
5. **Augmentation** (training): Rotation (±15°), horizontal flips, color jitter

### Training Configuration
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Batch Size**: 32
- **Epochs**: Up to 50 with early stopping

## 🔍 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) generates visual explanations by:
- Computing gradients of the predicted class with respect to the final convolutional layer
- Creating heatmaps that highlight influential brain regions
- Overlaying attention maps on original MRI images
- Enabling verification that the model focuses on clinically relevant areas (e.g., hippocampal regions)

## 📊 Training Your Own Model

```bash
# Basic training
python train.py --model resnet --epochs 30 --batch-size 32 --lr 1e-3

# Or use Google Colab
# Open train_colab.ipynb in Colab for cloud-based training
```

Model checkpoints save to `models/best_resnet.pth`

## 📋 Key Files

| File | Purpose |
|------|---------|
| `app/app.py` | Flask API for inference & visualization |
| `models/cnn_model.py` | ResNet50 architecture definition |
| `models/grad_cam.py` | Saliency map generation |
| `train.py` | Full training pipeline with validation |
| `MODEL_CARD.md` | Model specifications & ethical considerations |
| `DEVPOST.md` | Project description for hackathon submission |

## 🔧 Troubleshooting

**Import errors for PyTorch/dependencies:**
```bash
pip install -r requirements.txt
```

**CUDA/GPU not detected:**  
App automatically falls back to CPU. For GPU support, ensure CUDA Toolkit is installed.

**Port 8000 already in use:**  
Modify the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port number
```

**OpenCV not available warning:**  
Install opencv-python for full Grad-CAM visualization:
```bash
pip install opencv-python
```

## 📚 Documentation

- **Technical Report**: See `BrainGuard_Report.html` (2-3 pages covering methods & evaluation)
- **Model Card**: See `MODEL_CARD.md` for data sources, preprocessing, metrics, and limitations
- **Devpost**: See `DEVPOST.md` for hackathon submission text

## 🎓 Data Sources

- **Primary**: Kaggle Alzheimer MRI Disease Classification Dataset
- **Optional**: ALZ_Variant dataset in `Assets/Datasets/`
- **Framework**: PyTorch with torchvision, scikit-learn, NumPy, Pillow

## ⚠️ Disclaimer

This system is a **research prototype** for educational purposes only. It is:
- NOT approved for clinical use
- NOT a substitute for professional medical diagnosis
- Intended for the Hack4Health AI challenge and learning purposes

## 🏆 Acknowledgments

Built for Hack4Health AI for Alzheimer's Challenge (Intermediate Level)

---

**Need help?** Check `MODEL_CARD.md` for detailed specifications or open an issue in the repository.
