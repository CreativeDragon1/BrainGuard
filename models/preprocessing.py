"""
Image preprocessing utilities
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2

def preprocess_image(image_path, size=224):
    """
    Preprocess image for model input
    
    Args:
        image_path: Path to image file
        size: Target size (default 224x224)
    
    Returns:
        Preprocessed tensor
    """
    img = Image.open(image_path).convert('L')  # Grayscale
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    return transform(img)

def postprocess_image(tensor):
    """
    Convert tensor back to displayable image
    """
    tensor = tensor.cpu().squeeze()
    img_array = tensor.numpy()
    img_array = np.clip((img_array + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def normalize_image(image_array):
    """Normalize image array to 0-1 range"""
    return (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)

def augment_image(image_tensor, augmentation_type='standard'):
    """
    Apply data augmentation to image
    
    Args:
        image_tensor: Input image tensor
        augmentation_type: Type of augmentation to apply
    
    Returns:
        Augmented tensor
    """
    if augmentation_type == 'standard':
        transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    elif augmentation_type == 'light':
        transform = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform = transforms.Compose([])
    
    pil_image = transforms.ToPILImage()(image_tensor)
    augmented = transform(pil_image)
    return transforms.ToTensor()(augmented)

if __name__ == "__main__":
    print("Preprocessing utilities loaded")
