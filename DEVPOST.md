# BrainGuard — MRI-Based Alzheimer’s Detection

## About the Project
BrainGuard is a Flask + PyTorch web app that classifies brain MRI scans into four dementia severity levels (Non-Demented, Very Mild, Mild, Moderate) and provides Grad-CAM visual explanations to highlight the regions influencing the prediction. Built for the Hack4Health AI for Alzheimer’s challenge, the system emphasizes interpretability, reliability, and ease of use.

## What It Does
- Upload MRI images (PNG/JPG or NIfTI); automatically handles middle-slice extractio for 3D volumes.
- ResNet50 transfer-learning model delivers high accuracy across four dementia classes.
- Grad-CAM overlays show the model’s attention for transparent, explainable AI.
- Web UI for drag-and-drop uploads, probability breakdowns, and confidence display.

## Tech Stack
- Backend: Flask, Python, PyTorch (ResNet50), Torchvision
- Explainability: Grad-CAM
- Data Handling: Pillow, NumPy, (optional) NiBabel for NIfTI
- Frontend: HTML/CSS/JS (vanilla), Font Awesome

## How We Built It
1. **Data Prep**: Grayscale conversion, resize to 224×224, normalization (ImageNet stats). For NIfTI, extract the middle axial slice.
2. **Model**: ResNet50 with single-channel first conv and a 4-class classifier head.
3. **Training**: CrossEntropyLoss + Adam (lr=1e-3), ReduceLROnPlateau scheduler, batch size 32, early-stopping mindset.
4. **Explainability**: Grad-CAM on the final conv block to generate saliency overlays.
5. **App**: Flask API for inference; web UI for uploads, predictions, probabilities, and attention maps.

## Results
- Strong validation performance with clear saliency maps aligning to clinically relevant regions (e.g., hippocampal areas).
- Fast inference on CPU; optional GPU support if available.

## How to Run (Local)
```bash
python -m venv venv
# On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app/app.py
# Open http://localhost:8000
```

## Challenges
- Handling diverse MRI formats (2D and 3D) with consistent preprocessing.
- Ensuring interpretability via Grad-CAM while keeping inference fast.

## What’s Next
- Full 3D model support (volumetric CNNs / transformers).
- Domain adaptation for multi-scanner generalization.
- Calibration and threshold tuning for deployment contexts.
- Additional explainability methods (e.g., Integrated Gradients).

## Team
BrainGuard — Hack4Health AI for Alzheimer’s Challenge
