# Google Colab Setup Guide — Alzheimer's MRI Detection

This guide shows how to run your complete project (training + inference + visualization) in Google Colab for the Hack4Health hackathon submission.

---

## Option 1: Upload Full Project to Google Drive

### Step 1: Prepare Your Files
1. Compress your project folder:
   ```bash
   cd /Users/jchheda/Desktop/Hackthon\ Project/
   zip -r Alzeimers.zip Alzeimers/ -x "*.venv*" -x "*__pycache__*" -x "*.DS_Store"
   ```

2. Upload `Alzeimers.zip` to your Google Drive (e.g., in a folder called `Hackathon`)

### Step 2: Create New Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Name it: `Alzheimers_MRI_Detection_Training.ipynb`

### Step 3: Mount Drive and Extract Project
```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Extract project
!unzip -q /content/drive/MyDrive/Hackathon/Alzeimers.zip -d /content/
%cd /content/Alzeimers
!ls -la
```

### Step 4: Install Dependencies
```python
# Cell 3: Install requirements
!pip install -q torch torchvision pandas pyarrow scikit-learn tqdm pillow opencv-python nibabel flask
```

### Step 5: Check GPU Availability
```python
# Cell 4: Verify GPU
import torch
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Option 2: Clone from GitHub (Recommended)

### Step 1: Push to GitHub
1. Create a new GitHub repo (e.g., `alzheimers-mri-detection`)
2. Push your project:
   ```bash
   cd /Users/jchheda/Desktop/Hackthon\ Project/Alzeimers
   git init
   echo ".venv/" > .gitignore
   echo "__pycache__/" >> .gitignore
   echo "*.pyc" >> .gitignore
   echo ".DS_Store" >> .gitignore
   echo "static/uploads/*" >> .gitignore
   git add .
   git commit -m "Initial commit: Alzheimer MRI detection project"
   git remote add origin https://github.com/YOUR_USERNAME/alzheimers-mri-detection.git
   git push -u origin main
   ```

### Step 2: Colab Setup
```python
# Cell 1: Clone repo
!git clone https://github.com/YOUR_USERNAME/alzheimers-mri-detection.git
%cd alzheimers-mri-detection

# Cell 2: Install dependencies
!pip install -q torch torchvision pandas pyarrow scikit-learn tqdm pillow opencv-python nibabel

# Cell 3: Check GPU
import torch
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
```

---

## Training in Colab

### Full Training Notebook
```python
# Cell 1: Import and setup
import os
import torch
import pandas as pd
from pathlib import Path

# Enable GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {DEVICE}")

# Cell 2: Load dataset
# Upload train.parquet and test.parquet to Colab or mount Drive
from google.colab import files

print("Upload train.parquet:")
uploaded = files.upload()  # Upload your train.parquet
print("Upload test.parquet:")
uploaded = files.upload()  # Upload your test.parquet

# Move to dataset folder
!mkdir -p Assets/Datasets/MRI\ Dataset
!mv train.parquet Assets/Datasets/MRI\ Dataset/
!mv test.parquet Assets/Datasets/MRI\ Dataset/

# Cell 3: Train model (with GPU acceleration)
!python train.py --model resnet --epochs 30 --batch-size 64 --lr 1e-3

# Cell 4: Download trained model
from google.colab import files
files.download('models/best_resnet.pth')
```

---

## Inference & Visualization in Colab

### Test Single Image
```python
# Cell 1: Load model
import torch
from models.cnn_model import ResNetModel
from models.grad_cam import GradCAM
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

model = ResNetModel(pretrained=True, num_classes=4)
checkpoint = torch.load('models/best_resnet.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Cell 2: Upload and test image
from google.colab import files
uploaded = files.upload()  # Upload an MRI image
img_path = list(uploaded.keys())[0]

img = Image.open(img_path).convert('L')
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485], std=[0.229])
])
img_tensor = transform(img).unsqueeze(0)

# Cell 3: Predict
with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)

classes = ['Mild Dementia', 'Moderate Dementia', 'Non-Demented', 'Very Mild Dementia']
print(f"Prediction: {classes[pred.item()]}")
print(f"Confidence: {conf.item()*100:.2f}%")
print("\nAll probabilities:")
for i, cls in enumerate(classes):
    print(f"  {cls}: {probs[0,i].item()*100:.2f}%")

# Cell 4: Grad-CAM visualization
grad_cam = GradCAM(model, target_layer='layer4')
cam = grad_cam.generate(img_tensor, pred.item())

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original MRI')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cam)
plt.title('Grad-CAM Heatmap')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img, cmap='gray')
plt.imshow(cam, alpha=0.5)
plt.title(f'Overlay: {classes[pred.item()]}')
plt.axis('off')

plt.tight_layout()
plt.show()
```

---

## Running Flask App in Colab (with ngrok)

```python
# Cell 1: Install ngrok
!pip install -q pyngrok flask

# Cell 2: Setup ngrok tunnel
from pyngrok import ngrok
import os

# Set port
os.environ['PORT'] = '5000'

# Start ngrok tunnel
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")

# Cell 3: Run Flask app in background
import subprocess
import time

# Run Flask in background
proc = subprocess.Popen(['python', 'app/app.py'], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE)
time.sleep(5)  # Wait for server to start

print(f"Flask app running at: {public_url}")
print("Click the link above to access your web interface!")

# Keep cell running (stop with interrupt button)
proc.wait()
```

---

## Complete Colab Template (Copy-Paste Ready)

### Create `Alzheimers_MRI_Colab.ipynb` with these cells:

```python
# === CELL 1: Setup ===
!git clone https://github.com/YOUR_USERNAME/alzheimers-mri-detection.git
%cd alzheimers-mri-detection
!pip install -q torch torchvision pandas pyarrow scikit-learn tqdm pillow opencv-python nibabel

# === CELL 2: Check GPU ===
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training device: {DEVICE}")

# === CELL 3: Upload Dataset ===
from google.colab import files
print("Upload train.parquet:")
files.upload()
print("Upload test.parquet:")
files.upload()
!mkdir -p Assets/Datasets/MRI\ Dataset
!mv train.parquet test.parquet Assets/Datasets/MRI\ Dataset/

# === CELL 4: Train Model (30 epochs, ~45 min on GPU) ===
!python train.py --model resnet --epochs 30 --batch-size 64 --lr 1e-3

# === CELL 5: Test Inference ===
import torch
from models.cnn_model import ResNetModel
from PIL import Image
import torchvision.transforms as T

model = ResNetModel(pretrained=True, num_classes=4)
checkpoint = torch.load('models/best_resnet.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Upload test image
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
img = Image.open(img_path).convert('L')

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485], std=[0.229])
])
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)

classes = ['Mild Dementia', 'Moderate Dementia', 'Non-Demented', 'Very Mild Dementia']
print(f"Prediction: {classes[pred.item()]} ({conf.item()*100:.1f}% confidence)")

# === CELL 6: Download Model ===
files.download('models/best_resnet.pth')
```

---

## Tips for Hackathon Submission

1. **Runtime**: Use GPU runtime (Runtime → Change runtime type → GPU)
2. **Session**: Colab sessions timeout after 12 hours; save checkpoints frequently
3. **Sharing**: Share the Colab notebook link (File → Share) in your submission
4. **Output**: Save training logs, plots, and model to Drive:
   ```python
   !cp models/best_resnet.pth /content/drive/MyDrive/Hackathon/
   ```

5. **Report**: Take screenshots of training progress, Grad-CAM visualizations, and metrics for your PDF report

---

## Troubleshooting

**Issue**: Out of memory
- **Fix**: Reduce `--batch-size` to 32 or 16

**Issue**: Runtime disconnected
- **Fix**: Run cells one at a time; use `!nvidia-smi` to monitor GPU

**Issue**: Can't find parquet files
- **Fix**: Check paths with `!ls Assets/Datasets/MRI\ Dataset/`

**Issue**: Slow on CPU
- **Fix**: Enable GPU (Runtime → Change runtime type → T4 GPU)
