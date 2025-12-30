# Model Card — AlzDetect AI (Intermediate)

## Model Details
- Model Names: `ResNet50` (transfer learning), `Custom CNN`
- Version: 0.1 (hackathon draft)
- Framework: PyTorch
- Task: MRI image classification into 4 classes — Mild Dementia, Moderate Dementia, Non-Demented, Very Mild Dementia

## Intended Use
- Intended users: Researchers/students participating in Hack4Health Alzheimer’s challenge.
- Intended use: Educational prototype for MRI-based Alzheimer detection with explainability (Grad-CAM). Not for clinical use.
- Out-of-scope use: Any diagnostic or clinical decision-making without medical oversight.

## Data
- Sources: Kaggle Alzheimer MRI Disease Classification Dataset; optional ALZ_Variant dataset in `Assets/Datasets/`.
- Preprocessing: Grayscale conversion, resize to 224x224, normalization (mean 0.485, std 0.229). NIfTI files sliced at middle axial slice.
- Splits: Define train/val/test splits with stratification if possible. Document counts per class.
- Class balance: Report class distribution; consider weighting or augmentation if imbalanced.

## Training Procedure
- Architectures: ResNet50 (pretrained ImageNet, conv1 adapted to 1-channel), Custom CNN baseline.
- Hyperparameters: Refer to `train.py` (epochs, batch size, LR, weight decay, scheduler). Document final choices.
- Augmentation: Rotation, horizontal flip, color jitter (light). Document exact pipeline used.
- Loss/Optimizer: CrossEntropyLoss; Adam optimizer; ReduceLROnPlateau scheduler.

## Evaluation
- Metrics: Accuracy, precision, recall, F1 (weighted), confusion matrix. Include per-class metrics.
- Validation: Hold-out or k-fold; ensure patient-level separation if applicable.
- Calibration: (Optional) temperature scaling; report if used.

## Explainability
- Method: Grad-CAM over final conv block (`layer4` for ResNet50). Provide sample overlays.
- Caveats: Saliency may highlight artifacts; verify alignment with clinical regions of interest.

## Ethical Considerations & Bias
- Demographics: Document age/sex/site distribution if available.
- Bias risks: Scanner/site bias, demographic underrepresentation. Consider domain adaptation or balanced sampling.
- Misuse risks: Not a clinical tool; must not be used for patient care decisions.

## Limitations
- Prototype only; limited data and validation.
- Single-slice NIfTI handling may miss 3D context.
- Performance may degrade on out-of-distribution scanners/sites.

## Deployment Notes
- Inference device: CPU/GPU; tested shapes 1x1x224x224.
- Expected input formats: PNG/JPG or NIfTI (`.nii`, `.nii.gz`).
- Model weights: Place best checkpoints at `models/best_resnet_model.pth` or `models/best_cnn_model.pth`.

## Contact
- Team: Hack4Health participant
- For issues: Document steps, data sample (non-PHI), stack trace, and environment details.
