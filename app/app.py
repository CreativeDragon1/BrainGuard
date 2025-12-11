"""
Flask application for Alzheimer's MRI Detection
Intermediate-level hackathon project with model interpretability
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import io
import base64
from datetime import datetime

# Import model and utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.cnn_model import AlzheimersCNN, ResNetModel
from models.grad_cam import GradCAM
from models.preprocessing import preprocess_image

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'nii', 'nii.gz'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Global model cache
model_cache = {
    'model': None,
    'grad_cam': None,
    'model_type': None
}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_type='resnet'):
    """Load the trained model"""
    global model_cache
    
    if model_cache['model'] is not None and model_cache['model_type'] == model_type:
        return model_cache['model']
    
    try:
        if model_type == 'resnet':
            model = ResNetModel(pretrained=True, num_classes=4)
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_resnet.pth')
        else:
            model = AlzheimersCNN(num_classes=4)
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_cnn.pth')
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
        
        model = model.to(DEVICE)
        model.eval()
        
        model_cache['model'] = model
        model_cache['model_type'] = model_type
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def prepare_image(image_path):
    """Prepare image for model prediction"""
    try:
        if image_path.endswith(('.nii', '.nii.gz')):
            # Handle NIfTI files
            import nibabel as nib
            img_data = nib.load(image_path).get_fdata()
            # Select middle slice for visualization
            middle_slice = img_data.shape[2] // 2
            img_array = img_data[:, :, middle_slice]
            # Normalize to 0-255
            img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            img = Image.fromarray(img_array).convert('L')
        else:
            # Handle standard image formats
            img = Image.open(image_path).convert('L')  # Convert to grayscale
        
        return img
    except Exception as e:
        print(f"Error preparing image: {str(e)}")
        return None

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded MRI image"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, nii, nii.gz'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load model (always use ResNet)
        model = load_model('resnet')
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 500
        
        # Prepare image
        img = prepare_image(filepath)
        if img is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Class labels
        classes = ['Mild Dementia', 'Moderate Dementia', 'Non-Demented', 'Very Mild Dementia']
        predicted_class = classes[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Get all probabilities
        prob_dict = {classes[i]: float(probabilities[0, i].item()) * 100 for i in range(len(classes))}
        
        # Generate Grad-CAM visualization
        grad_cam = GradCAM(model, target_layer='layer4')
        cam_image = grad_cam.generate(img_tensor, predicted.item())
        
        # Convert CAM to base64
        cam_pil = Image.fromarray(cam_image)
        buffered = io.BytesIO()
        cam_pil.save(buffered, format='PNG')
        cam_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Convert input image to base64
        img_display = img.resize((224, 224))
        buffered_input = io.BytesIO()
        img_display.save(buffered_input, format='PNG')
        img_base64 = base64.b64encode(buffered_input.getvalue()).decode()
        
        response = {
            'predicted_class': predicted_class,
            'confidence': round(confidence_score, 2),
            'probabilities': {k: round(v, 2) for k, v in prob_dict.items()},
            'grad_cam': cam_base64,
            'input_image': img_base64,
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Return model information"""
    return jsonify({
        'device': str(DEVICE),
        'model': 'ResNet50',
        'accuracy': '98%',
        'model_classes': ['Mild Dementia', 'Moderate Dementia', 'Non-Demented', 'Very Mild Dementia'],
        'supported_formats': list(ALLOWED_EXTENSIONS)
    })

@app.route('/history', methods=['GET'])
def get_history():
    """Get recent prediction history"""
    uploads_dir = app.config['UPLOAD_FOLDER']
    files = sorted(
        [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))],
        key=lambda x: os.path.getmtime(os.path.join(uploads_dir, x)),
        reverse=True
    )[:10]  # Last 10 files
    
    return jsonify({'recent_uploads': files})

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size: 50MB'}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # For production, use a proper WSGI server like Gunicorn
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
