# AlzDetect AI — Intermediate Hackathon Webapp

Flask + PyTorch web application for MRI-based Alzheimer detection. Built for the Hack4Health AI for Alzheimer’s (Intermediate level) with transfer learning, proper validation, and Grad-CAM explainability.

## Features
- Flask web UI with drag-and-drop MRI upload, live predictions, and Grad-CAM overlays.
- PyTorch backends: ResNet50 transfer learning and custom CNN (selectable).
- Supports PNG/JPG and NIfTI (`.nii`, `.nii.gz`) inputs; auto-slices NIfTI to middle slice.
- Probability breakdown across 4 classes: Mild, Moderate, Non-Demented, Very Mild Dementia.
- Grad-CAM visualization to highlight influential brain regions (interpretability requirement).
- Training script with validation loop, checkpoints, and metrics (acc/precision/recall/F1/confusion matrix).

## Project Structure
- `app/app.py` — Flask server and prediction endpoint.
- `templates/index.html` — Frontend layout.
- `static/css/style.css`, `static/js/script.js` — UI styling and client logic.
- `models/` — PyTorch architectures, preprocessing, Grad-CAM utils, model weights placeholder.
- `train.py` — Training loop template with model selection and checkpointing.
- `notebooks/` — EDA/training notebook template (Colab/Notebook requirement).
- `data/` — Place prepared training/validation data here if running locally.
- `Assets/Datasets/` — Source datasets (MRI parquet and ALZ variant) provided in the repo.

## Setup
1) Create a Python 3.10+ environment and install deps:
```bash
pip install -r requirements.txt
```
2) (Optional) If using NIfTI: `pip install nibabel` is already listed.

## Running the Webapp
```bash
python app/app.py
# then open http://localhost:5000
```
- Upload MRI (`.png/.jpg` or `.nii/.nii.gz`).
- Choose model: `ResNet50` (better accuracy) or `Custom CNN` (faster).
- View prediction, confidence, class probabilities, and Grad-CAM overlay.

## Training (Template)
1) Place/convert MRI images into a folder structure or load parquet; update dataset loader accordingly.
2) Use `train.py` as a starting point:
```bash
python train.py --model resnet --epochs 30 --batch-size 32 --lr 1e-3
```
3) Checkpoints save to `models/best_<model>.pth`. Copy the best weight to `models/best_resnet_model.pth` or `models/best_cnn_model.pth` for inference.
4) Add metrics/plots to the notebook under `notebooks/` and export a PDF for the 2–3 page report requirement.

## Model Card
Fill out `MODEL_CARD.md` with data sources, preprocessing, metrics, bias/limitations, and interpretability notes (Grad-CAM usage).

## Submission Checklist (Hackathon)
- Runnable notebook/Colab or scripts (provided: `train.py`, `notebooks/` template).
- PDF report (2–3 pages) covering motivation, data, methods, and evaluation.
- Model card (bias, interpretability, limitations).
- Optional: short demo video ≤ 2 minutes.

## Next Steps
- Hook your actual dataset loader into `train.py` and the notebook.
- Train and export best weights to `models/` for the Flask app.
- Add validation metrics and confusion matrix visuals to the report and model card.
- Deploy via Render/Heroku/Fly.io or run locally for the demo.
# BrainGuard
